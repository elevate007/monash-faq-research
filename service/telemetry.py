"""Bounded-label metrics and operational JSON logs; no question/answer logging."""
import json
import logging
import threading
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

LOG=logging.getLogger('monash_faq.operations')
if not LOG.handlers:
    handler=logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(message)s'))
    LOG.addHandler(handler)
LOG.setLevel(logging.INFO)
LOG.propagate=False

class Telemetry:
    def __init__(self):
        self.registry=CollectorRegistry()
        self.requests=Counter('faq_http_requests','HTTP responses by bounded route and status',['route','method','status'],registry=self.registry)
        duration_buckets=(.01,.05,.1,.25,.5,1,2,5,10,20,30,60,120,180)
        self.latency=Histogram('faq_http_request_duration_seconds','HTTP request latency',['route'],buckets=duration_buckets,registry=self.registry)
        self.answers=Counter('faq_answers','Answer decisions',['backend','mode','outcome','reason'],registry=self.registry)
        self.generation=Histogram('faq_generation_duration_seconds','Synchronous model generation latency',buckets=duration_buckets,registry=self.registry)
        self.coverage=Histogram('faq_retrieval_coverage','Lexical query coverage, not probability',buckets=(0,.2,.4,.5,.6,.8,1),registry=self.registry)
        self.tokens=Counter('faq_output_tokens','Generated model tokens',registry=self.registry)
        self.inflight=Gauge('faq_inference_inflight','Active model generations',registry=self.registry)
        self.ready=Gauge('faq_model_ready','Loaded serving backend readiness',registry=self.registry)
        self.gpu=Gauge('faq_gpu_allocated_bytes','Torch allocated GPU memory, zero on CPU',registry=self.registry)
        self.guardrails=Counter('faq_guardrail_blocks','Policy or output-validation blocks',['reason'],registry=self.registry)
        self._lock=threading.Lock()
        self._summary={'requests':0,'errors':0,'answers':0,'abstentions':0,'guardrail_blocks':0,
                       'generation_calls':0,'generation_errors':0,'total_generation_seconds':0.0}

    def add(self,**changes):
        with self._lock:
            for key,value in changes.items():self._summary[key]+=value

    def snapshot(self):
        with self._lock:return dict(self._summary)

    def decision(self,request_id,result,seconds):
        self.answers.labels(result['model']['backend'],result['mode'],
                           'abstained' if result['abstained'] else 'answered',result['reason']).inc()
        self.add(answers=1,abstentions=int(result['abstained']))
        if result['reason'] not in ('grounded_selection','generated_answer','extractive_answer','model_refusal'):
            self.guardrails.labels(result['reason']).inc()
            self.add(guardrail_blocks=1)
        LOG.info(json.dumps({'event':'answer_decision','request_id':request_id,'backend':result['model']['backend'],
            'mode':result['mode'],'reason':result['reason'],'abstained':result['abstained'],
            'latency_ms':round(seconds*1000,2),'retrieval_coverage':result['retrieval']['coverage']}))
