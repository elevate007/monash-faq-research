"""FastAPI entry point. Run one worker for a single model instance."""
import secrets
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.concurrency import run_in_threadpool
from service.backend import QwenBackend, ExtractiveBackend
from service.config import Settings
from service.engine import FAQEngine, ModelBusy, GenerationFailure
from service.telemetry import Telemetry
from service.schemas import AnswerResponse

class AnswerRequest(BaseModel):
    model_config=ConfigDict(extra='forbid',str_strip_whitespace=True)
    question:str=Field(min_length=3,max_length=1000)
    audience:str|None=Field(default=None,max_length=160)

def create_app(settings=None,backend_factory=None):
    settings=settings or Settings.from_env()
    telemetry=Telemetry()
    @asynccontextmanager
    async def lifespan(app):
        factory=backend_factory or (QwenBackend if settings.backend=='qwen' else lambda s:ExtractiveBackend())
        backend=await run_in_threadpool(factory,settings)
        try:
            app.state.engine=FAQEngine(settings,backend,telemetry)
            telemetry.gpu.set(backend.gpu_bytes())
            telemetry.ready.set(1)
            yield
        finally:
            telemetry.ready.set(0)
            await run_in_threadpool(backend.close)

    app=FastAPI(title='Monash FAQ Research API',version='0.2.0',lifespan=lifespan,
                description='Source-grounded FAQ serving with explicit abstention and operational monitoring. Snapshot support is not proof of current policy or correct relevance.')
    app.state.telemetry=telemetry

    def authorize(key):
        if settings.api_key and not secrets.compare_digest((key or '').encode(),settings.api_key.encode()):
            raise HTTPException(401,'Invalid API key')

    @app.middleware('http')
    async def operational_metrics(request,call_next):
        started=time.perf_counter();request.state.request_id=str(uuid.uuid4())
        status=500
        try:
            response=await call_next(request);status=response.status_code
            response.headers['X-Request-ID']=request.state.request_id
            return response
        finally:
            route=getattr(request.scope.get('route'),'path','unmatched')
            method=request.method if request.method in ('GET','POST','OPTIONS') else 'other'
            telemetry.requests.labels(route,method,str(status)).inc()
            telemetry.latency.labels(route).observe(time.perf_counter()-started)
            telemetry.add(requests=1,errors=int(status>=500))

    @app.get('/healthz')
    def health():return {'status':'alive'}

    @app.get('/readyz')
    def ready():
        if not hasattr(app.state,'engine'):raise HTTPException(503,'Backend is not ready')
        return {'status':'ready','model':app.state.engine.backend.info(),'snapshot_date':'2026-10-08'}

    @app.post('/v1/answer',response_model=AnswerResponse)
    async def answer(body:AnswerRequest,request:Request,x_api_key:str|None=Header(default=None)):
        authorize(x_api_key)
        try:
            return await run_in_threadpool(app.state.engine.answer,body.question,request.state.request_id,body.audience)
        except ModelBusy:
            raise HTTPException(429,'Model is busy; retry later',headers={'Retry-After':'2'})
        except GenerationFailure:
            telemetry.guardrails.labels('generation_error').inc();telemetry.add(guardrail_blocks=1)
            raise HTTPException(503,'Model generation failed; no unverified answer was returned')
        except ValueError:
            raise HTTPException(422,'Audience is not present in the snapshot')

    @app.get('/metrics',include_in_schema=False)
    def metrics(x_api_key:str|None=Header(default=None)):
        authorize(x_api_key)
        return Response(generate_latest(telemetry.registry),headers={'Content-Type':CONTENT_TYPE_LATEST})

    @app.get('/v1/monitoring')
    def monitoring(x_api_key:str|None=Header(default=None)):
        authorize(x_api_key)
        return {'model':app.state.engine.backend.info(),'mode':settings.response_mode if settings.backend=='qwen' else 'extractive',
                'counters':telemetry.snapshot(),
                'interpretation':'Operational signals, not factual accuracy or a definitive hallucination detector.',
                'snapshot_date':'2026-10-08','threshold':app.state.engine.threshold}

    @app.get('/v1/audiences')
    def audiences():return {'audiences':sorted({r['audience'] for r in app.state.engine.records})}

    @app.get('/',include_in_schema=False)
    def dashboard():return FileResponse(Path(__file__).with_name('dashboard.html'))
    return app

app=create_app()
