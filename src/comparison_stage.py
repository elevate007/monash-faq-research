"""Run after the training notebook has reloaded its exported PEFT adapter.

Required globals: model, tokenizer, records, SYSTEM_PROMPT, ROOT, SEED,
benchmark_cases, calibration_cases. The standalone notebook supplies them.
"""
import contextlib
import hashlib
import importlib.metadata
import json
import platform
import time
import zipfile
from pathlib import Path
import torch
from retrieval import BM25, calibrate, context_messages, is_refusal, source_urls, REFUSAL

COMPARISON_DIR = ROOT.parent / 'monash_comparison'
COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
index = BM25(records)
calibration_result = calibrate(index, calibration_cases)
threshold = calibration_result['threshold']

def generate_answer(messages, adapter_enabled):
    ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                         return_tensors='pt').to(model.device)
    if ids.shape[1] > 1024:
        raise ValueError('Prompt exceeds preregistered limit; no silent truncation')
    manager = contextlib.nullcontext() if adapter_enabled else model.disable_adapter()
    torch.cuda.synchronize()
    started = time.perf_counter()
    with manager, torch.inference_mode():
        output = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                                do_sample=False, max_new_tokens=160, use_cache=True,
                                pad_token_id=tokenizer.pad_token_id,
                                eos_token_id=tokenizer.eos_token_id)
    torch.cuda.synchronize()
    answer = tokenizer.decode(output[0, ids.shape[1]:], skip_special_tokens=True)
    return answer, time.perf_counter()-started, int(output.shape[1]-ids.shape[1])

model.eval()
torch.cuda.reset_peak_memory_stats()
predictions = []
for case in benchmark_cases:
    hits = index.search(case['question'], 3)
    accepted = hits[0]['score'] > 0 and hits[0]['coverage'] >= threshold
    closed = [{'role':'system','content':SYSTEM_PROMPT}, {'role':'user','content':case['question']}]
    for method in ['base', 'qlora', 'rag_base', 'rag_qlora']:
        rag = method.startswith('rag_')
        answer, seconds, count = generate_answer(context_messages(case['question'], hits) if rag else closed,
                                                  method in ['qlora','rag_qlora'])
        urls = sorted({h['record']['source_url'] for h in hits}) if rag else []
        row = {**case, 'method':method, 'answer_body':answer, 'retrieved_ids':[h['id'] for h in hits] if rag else [],
               'candidate_source_urls': urls, 'top1_coverage':hits[0]['coverage'] if rag else None,
               'gate_accepted':accepted if rag else None, 'refusal_heuristic':is_refusal(answer),
               'generated_urls':source_urls(answer), 'seconds':seconds, 'output_tokens':count,
               'manual_correct':None, 'manual_refusal':None, 'manual_notes':None}
        predictions.append(row)
        # This is a post-processing ablation of exactly the same RAG output.
        if method=='rag_base':
            predictions.append({**row, 'method':'gated_rag_base',
                                'answer_body':answer if accepted else REFUSAL,
                                'candidate_source_urls':urls if accepted else [],
                                'refusal_heuristic':is_refusal(answer) if accepted else True,
                                'generated_urls':source_urls(answer) if accepted else [],
                                'seconds':None, 'output_tokens':None})
        print(f"BENCHMARK {case['id']} {method}: {answer}", flush=True)
    predictions.append({**case, 'method':'extractive_bm25',
                        'answer_body':hits[0]['record']['answer'] if accepted else REFUSAL,
                        'retrieved_ids':[h['id'] for h in hits],
                        'candidate_source_urls':[hits[0]['record']['source_url']] if accepted else [],
                        'top1_coverage':hits[0]['coverage'], 'gate_accepted':accepted,
                        'refusal_heuristic':not accepted, 'generated_urls':[], 'seconds':None,
                        'output_tokens':None, 'manual_correct':None, 'manual_refusal':None,'manual_notes':None})
    # Save incrementally so partial results survive a failed session.
    (COMPARISON_DIR/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in predictions), encoding='utf-8')

automatic = {}
for method in sorted({r['method'] for r in predictions}):
    rows = [r for r in predictions if r['method']==method]
    positive = [r for r in rows if r['kind']=='answerable']
    negative = [r for r in rows if r['kind']=='unsupported']
    automatic[method] = {'answerable_size':len(positive),'unsupported_size':len(negative),
                         'answerable_refusal_heuristic_count':sum(r['refusal_heuristic'] for r in positive),
                         'unsupported_refusal_heuristic_count':sum(r['refusal_heuristic'] for r in negative),
                         'generated_url_count':sum(len(r['generated_urls']) for r in rows),
                         'reference_in_candidate_sources_count':sum(r['source_url'] in r['candidate_source_urls'] for r in positive)}
manifest = {'gpu':torch.cuda.get_device_name(0),'seed':SEED,'python':platform.python_version(),
            'model_id':MODEL_ID,'model_revision':getattr(model.config,'_commit_hash',None),
            'adapter_source':str(ADAPTER_DIR),'max_new_tokens':160,'do_sample':False,
            'peak_allocated_gpu_bytes':torch.cuda.max_memory_allocated(),
            'versions':{n:importlib.metadata.version(n) for n in ('torch','transformers','peft','bitsandbytes')},
            'benchmark_sha256':hashlib.sha256(json.dumps(benchmark_cases, sort_keys=True).encode()).hexdigest(),
            'calibration_sha256':hashlib.sha256(json.dumps(calibration_cases, sort_keys=True).encode()).hexdigest(),
            'knowledge_base_sha256':hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest(),
            'note':'All 58 facts are available to retrieval. Ten answerable prompts paraphrase the ten fine-tuning holdout facts. Ten unsupported prompts ask for unavailable personal or future details. No independent blinded human assessment. Candidate URLs are retrieval metadata, not proven supporting citations. Gated output reuses rag_base output.'}
for name, obj in [('automatic_metrics.json',automatic),('comparison_manifest.json',manifest),('calibration_result.json',calibration_result)]:
    (COMPARISON_DIR/name).write_text(json.dumps(obj,indent=2),encoding='utf-8')
archive_path = ROOT.parent/'monash_comparison_results.zip'
with zipfile.ZipFile(archive_path,'w',zipfile.ZIP_DEFLATED) as archive:
    for path in COMPARISON_DIR.iterdir():
        archive.write(path,path.name)
display(FileLink(str(archive_path.relative_to(Path.cwd()))))
print('MONASH_COMPARISON_COMPLETE', flush=True)
