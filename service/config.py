"""Explicit serving configuration; no silent model fallback."""
import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = 'Qwen/Qwen2.5-1.5B-Instruct'
MODEL_REVISION = '989aa7980e4cf806f80c7fef2b1adb7bc71aa306'

@dataclass(frozen=True)
class Settings:
    backend: str = 'qwen'
    device: str = 'auto'
    adapter_dir: Path = ROOT / 'models/monash_qa_adapter'
    knowledge_base: Path = ROOT / 'data/monash_qa.jsonl'
    calibration_file: Path = ROOT / 'results/comparison/calibration_result.json'
    response_mode: str = 'grounded'
    api_key: str = ''
    max_new_tokens: int = 96
    cpu_threads: int = 8

    def __post_init__(self):
        if self.backend not in ('qwen','extractive'):
            raise ValueError('FAQ_BACKEND must be qwen or extractive')
        if self.device not in ('auto','cpu','cuda'):
            raise ValueError('FAQ_DEVICE must be auto, cpu, or cuda')
        if self.response_mode not in ('grounded','generative'):
            raise ValueError('FAQ_RESPONSE_MODE must be grounded or generative')
        if not 16 <= self.max_new_tokens <= 256:
            raise ValueError('FAQ_MAX_NEW_TOKENS must be between 16 and 256')
        if not 1 <= self.cpu_threads <= 32:
            raise ValueError('FAQ_CPU_THREADS must be between 1 and 32')

    @classmethod
    def from_env(cls):
        return cls(backend=os.getenv('FAQ_BACKEND','qwen'), device=os.getenv('FAQ_DEVICE','auto'),
                   adapter_dir=Path(os.getenv('FAQ_ADAPTER_DIR',str(ROOT/'models/monash_qa_adapter'))),
                   response_mode=os.getenv('FAQ_RESPONSE_MODE','grounded'), api_key=os.getenv('FAQ_API_KEY',''),
                   max_new_tokens=int(os.getenv('FAQ_MAX_NEW_TOKENS','96')),
                   cpu_threads=int(os.getenv('FAQ_CPU_THREADS','8')))
