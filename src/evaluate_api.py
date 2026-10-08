"""Evaluate the live API on the frozen convenience set; no hidden gold prompts."""
import argparse
import hashlib
import json
import os
import platform
import time
import urllib.request
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--url',default='http://127.0.0.1:8000')
    parser.add_argument('--output',type=Path,default=ROOT/'results/serving')
    args=parser.parse_args()
    base=args.url.rstrip('/')
    key=os.getenv('FAQ_API_KEY','')
    def request(path,payload=None):
        headers={'Content-Type':'application/json'}
        if key:headers['X-API-Key']=key
        req=urllib.request.Request(base+path,headers=headers,data=json.dumps(payload).encode() if payload else None)
        with urllib.request.urlopen(req,timeout=180) as response:return json.load(response)
    ready=request('/readyz')
    cases=[json.loads(x) for x in (ROOT/'data/benchmark.jsonl').read_text().splitlines()]
    rows=[]
    for case in cases:
        started=time.perf_counter()
        response=request('/v1/answer',{'question':case['question']})
        row={**case,'response':response,'http_seconds':time.perf_counter()-started}
        row['exact_target_selection']=bool(not response['abstained'] and any(c['record_id']==case['target_id'] for c in response['citations'])) if case['kind']=='answerable' else None
        row['unsupported_refusal']=response['abstained'] if case['kind']=='unsupported' else None
        rows.append(row)
        print(case['id'],response['reason'],flush=True)
    positive=[r for r in rows if r['kind']=='answerable'];negative=[r for r in rows if r['kind']=='unsupported']
    report={'model':ready['model'],'python':platform.python_version(),'mode':rows[0]['response']['mode'],
            'benchmark_sha256':hashlib.sha256((ROOT/'data/benchmark.jsonl').read_bytes()).hexdigest(),
            'answerable_size':len(positive),'unsupported_size':len(negative),
            'exact_target_selection_count':sum(r['exact_target_selection'] for r in positive),
            'unsupported_refusal_count':sum(r['unsupported_refusal'] for r in negative),
            'answerable_refusal_count':sum(r['response']['abstained'] for r in positive),
            'monitoring':request('/v1/monitoring'),
            'note':'Tiny convenience-set live integration check, not independent factual accuracy. Grounded mode returns exact stored summaries; wrong record selection and stale facts remain possible. CPU FP32 inference differs from the original T4 NF4 experiment.'}
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'api_predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8')
    (args.output/'api_evaluation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    req=urllib.request.Request(base+'/metrics',headers={'X-API-Key':key} if key else {})
    with urllib.request.urlopen(req,timeout=30) as response:
        (args.output/'live_metrics.prom').write_bytes(response.read())
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
