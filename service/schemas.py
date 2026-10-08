"""Typed public API responses; internal paths and raw drafts are excluded."""
from typing import Literal
from pydantic import BaseModel, Field

class ModelMetadata(BaseModel):
    backend:Literal['qwen','extractive']
    model_id:str|None
    model_revision:str|None
    adapter_loaded:bool
    device:str
    quantization:str|None

class Citation(BaseModel):
    record_id:str
    url:str
    audience:str
    candidate_only:bool

class RetrievalSignals(BaseModel):
    coverage:float=Field(ge=0,le=1)
    threshold:float=Field(ge=0,le=1.01)
    record_ids:list[str]
    scores:list[float]

class AnswerResponse(BaseModel):
    request_id:str
    answer:str
    abstained:bool
    reason:str
    citations:list[Citation]
    mode:Literal['grounded','generative','extractive']
    model:ModelMetadata
    snapshot_date:str
    support_check:Literal['none','exact_snapshot_text','surface_checks_only']
    retrieval:RetrievalSignals
    latency_ms:float=Field(ge=0)
