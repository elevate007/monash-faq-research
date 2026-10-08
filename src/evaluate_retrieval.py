"""CPU reproducible retrieval evaluation, with no language model dependency."""
import argparse
import hashlib
import json
from pathlib import Path
from retrieval import BM25, calibrate, read_jsonl, retrieve, REFUSAL

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root
    data = root / 'data'
    records = read_jsonl(data / 'monash_qa.jsonl')
    index = BM25(records)
    calibration = calibrate(index, read_jsonl(data / 'calibration.jsonl'))
    cases = read_jsonl(data / 'benchmark.jsonl')
    outputs = []
    answerable = [c for c in cases if c['kind'] == 'answerable']
    recalls, ranks = [], []
    for case in cases:
        hits, accepted = retrieve(index, case['question'], calibration['threshold'])
        ids = [h['id'] for h in hits]
        if case['kind'] == 'answerable':
            recalls.append(int(case['target_id'] in ids))
            ranks.append(1/(ids.index(case['target_id'])+1) if case['target_id'] in ids else 0)
        outputs.append({**case, 'retrieved_ids': ids, 'coverage': hits[0]['coverage'], 'accepted': accepted,
                        'prediction': hits[0]['record']['answer'] if accepted else REFUSAL,
                        'source_url_attached': hits[0]['record']['source_url'] if accepted else None})
    unsupported = [r for r in outputs if r['kind'] == 'unsupported']
    report = {'method': 'BM25 k1=1.5 b=0.75, word tokenization, question+answer index',
              'knowledge_base_size': len(records), 'answerable_size': len(answerable),
              'unsupported_size': len(unsupported), 'threshold': calibration['threshold'],
              'recall_at_3': sum(recalls)/len(recalls), 'mrr_at_3': sum(ranks)/len(ranks),
              'unsupported_gate_refusal_count': sum(not r['accepted'] for r in unsupported),
              'answerable_top1_correct_and_accepted_count': sum(r['accepted'] and r['retrieved_ids'][0] == r['target_id'] for r in outputs if r['kind']=='answerable'),
              'note': 'Diagnostic on a tiny author-written convenience set; retrieval has access to all 58 FAQ facts, including fine-tuning holdouts. Retrieval refusal is not model refusal or answer correctness.',
              'data_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in data.glob('*.jsonl')}}
    dest = root / 'results/comparison'
    dest.mkdir(parents=True, exist_ok=True)
    for name, obj in [('retrieval_metrics.json', report), ('calibration_result.json', calibration), ('retrieval_predictions.json', outputs)]:
        (dest/name).write_text(json.dumps(obj, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
