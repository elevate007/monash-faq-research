import json
import threading
import io
import logging
from dataclasses import replace
from fastapi.testclient import TestClient
from service.app import create_app
from service.config import Settings
from service.engine import grounded_messages
from service.telemetry import LOG
from prometheus_client.parser import text_string_to_metric_families

QUESTION='Do research scholarships pay visa application charges?'

class StubBackend:
    def __init__(self,text='{"evidence_id":"research_offers_02"}',fail=False):self.text=text;self.fail=fail;self.calls=0
    def generate(self,messages,max_new_tokens):
        self.calls+=1
        assert 'reference_answer' not in str(messages)
        if self.fail:raise RuntimeError('private backend error detail')
        return self.text,12
    def info(self):return {'backend':'qwen','model_id':'test-double','model_revision':None,'adapter_loaded':True,'device':'cpu','quantization':'test'}
    def gpu_bytes(self):return 0
    def close(self):pass

def client(stub=None,**settings):
    return TestClient(create_app(replace(Settings(),**settings),backend_factory=lambda s:stub or StubBackend()))

def test_grounded_quote_and_metadata_citation():
    with client() as c:
        r=c.post('/v1/answer',json={'question':QUESTION})
        assert r.status_code==200
        body=r.json();assert not body['abstained']
        assert body['answer']=='No. Those application charges are excluded.'
        assert body['support_check']=='exact_snapshot_text'
        assert body['citations'][0]['url'].startswith('https://www.monash.edu/')
        assert not body['citations'][0]['candidate_only']
        assert r.headers['x-request-id']==body['request_id']

def test_personal_future_and_no_overlap_skip_model():
    stub=StubBackend()
    with client(stub) as c:
        for question,reason in [('What is my personal application status today?','personal_information'),
                                ('What will my visa cost in 2028?','future_information'),
                                ('zyxwvu qqqqq','no_evidence')]:
            r=c.post('/v1/answer',json={'question':question}).json()
            assert r['abstained'] and r['reason']==reason and not r['citations']
        assert stub.calls==0

def test_invalid_id_and_invalid_json_fail_closed():
    for raw,reason in [('{"evidence_id":"made_up"}','invalid_evidence_id'),
                       ('Contact an invented advisor','invalid_model_output'),
                       ('{"evidence_id":true}','invalid_model_output')]:
        with client(StubBackend(raw)) as c:
            r=c.post('/v1/answer',json={'question':QUESTION}).json()
            assert r['abstained'] and r['reason']==reason and not r['citations']

def test_generated_prices_and_urls_blocked():
    for raw,reason in [('Visa application costs $999.','unsupported_specific'),
                       ('Visit https://invented.example','generated_url')]:
        with client(StubBackend(raw),response_mode='generative') as c:
            r=c.post('/v1/answer',json={'question':QUESTION}).json()
            assert r['abstained'] and r['reason']==reason

def test_generation_failure_does_not_expose_error_details():
    with client(StubBackend(fail=True)) as c:
        r=c.post('/v1/answer',json={'question':QUESTION})
        assert r.status_code==503 and 'private' not in r.text
        metrics=c.get('/metrics').text
        assert 'faq_guardrail_blocks_total{reason="generation_error"} 1.0' in metrics
        assert c.get('/v1/monitoring').json()['counters']['generation_errors']==1

def test_validation_auth_health_and_private_metrics():
    with client(api_key='test-only-key') as c:
        assert c.get('/healthz').status_code==200
        assert c.get('/readyz').status_code==200
        assert c.post('/v1/answer',json={'question':QUESTION}).status_code==401
        assert c.get('/metrics').status_code==401
        headers={'X-API-Key':'test-only-key'}
        assert c.post('/v1/answer',headers=headers,json={'question':'   '}).status_code==422
        assert c.post('/v1/answer',headers=headers,json={'question':QUESTION,'extra':'field'}).status_code==422
        assert c.post('/v1/answer',headers=headers,json={'question':QUESTION,'audience':'invented'}).status_code==422
        assert c.post('/v1/answer',headers=headers,json={'question':QUESTION}).status_code==200
        m=c.get('/metrics',headers=headers).text
        assert 'faq_answers_total' in m and 'faq_retrieval_coverage_bucket' in m
        assert QUESTION not in m and 'test-only-key' not in m

def test_busy_model_returns_retryable_status():
    with client() as c:
        engine=c.app.state.engine
        engine._slot.acquire()
        try:
            r=c.post('/v1/answer',json={'question':QUESTION})
            assert r.status_code==429 and r.headers['retry-after']=='2'
        finally:engine._slot.release()

def test_extractive_mode_is_explicit():
    with TestClient(create_app(Settings(backend='extractive'))) as c:
        r=c.post('/v1/answer',json={'question':QUESTION}).json()
        assert r['model']['backend']=='extractive' and not r['model']['adapter_loaded']
        assert r['mode']=='extractive'

def test_operational_logs_omit_question_and_key():
    stream=io.StringIO();handler=logging.StreamHandler(stream);LOG.addHandler(handler)
    try:
        private_question='What is my personal application status today?'
        with client(api_key='private-test-key') as c:
            r=c.post('/v1/answer',json={'question':private_question},headers={'X-API-Key':'private-test-key'})
            assert r.status_code==200
            events=stream.getvalue()
            assert 'answer_decision' in events and r.json()['request_id'] in events
            assert private_question not in events and 'private-test-key' not in events
    finally:LOG.removeHandler(handler)

def test_latency_metrics_cover_twenty_second_alert():
    with client() as c:
        c.app.state.telemetry.latency.labels('/v1/answer').observe(45)
        samples=[s for family in text_string_to_metric_families(c.get('/metrics').text) for s in family.samples
                 if s.name=='faq_http_request_duration_seconds_bucket' and s.labels.get('route')=='/v1/answer']
        values={s.labels['le']:s.value for s in samples}
        assert values['20.0']==0 and values['60.0']==1
