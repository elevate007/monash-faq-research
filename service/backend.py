"""Reload the actual adapter on CUDA/NF4 or CPU/FP32."""
from service.config import MODEL_ID, MODEL_REVISION

class QwenBackend:
    def __init__(self, settings):
        import json
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        if not (settings.adapter_dir/'adapter_model.safetensors').is_file():
            raise RuntimeError('Adapter weights missing. Run python src/download_adapter.py first.')
        config=json.loads((settings.adapter_dir/'adapter_config.json').read_text())
        if config.get('base_model_name_or_path') != MODEL_ID:
            raise RuntimeError('The adapter does not match the fixed Qwen base model')
        if settings.device=='cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA was requested but is unavailable')
        self.device='cuda' if settings.device!='cpu' and torch.cuda.is_available() else 'cpu'
        self.torch=torch
        torch.set_num_threads(settings.cpu_threads)
        self.tokenizer=AutoTokenizer.from_pretrained(MODEL_ID,revision=MODEL_REVISION,trust_remote_code=False)
        if self.tokenizer.pad_token_id is None:self.tokenizer.pad_token=self.tokenizer.eos_token
        kwargs={'revision':MODEL_REVISION,'trust_remote_code':False,'attn_implementation':'eager'}
        if self.device=='cuda':
            dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            kwargs.update(device_map={'':0},torch_dtype=dtype,
                          quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',
                              bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=dtype))
        else:
            kwargs.update(torch_dtype=torch.float32)
        base=AutoModelForCausalLM.from_pretrained(MODEL_ID,**kwargs)
        self.model=PeftModel.from_pretrained(base,str(settings.adapter_dir),is_trainable=False)
        if self.device=='cpu':self.model.to('cpu')
        self.model.eval()
        self.quantization='nf4' if self.device=='cuda' else 'fp32'

    def generate(self,messages,max_new_tokens):
        ids=self.tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,
                                               return_tensors='pt').to(self.model.device)
        if ids.shape[1]>1536:raise ValueError('Prompt exceeds the serving token limit')
        with self.torch.inference_mode():
            output=self.model.generate(input_ids=ids,attention_mask=self.torch.ones_like(ids),
                do_sample=False,max_new_tokens=max_new_tokens,use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id,eos_token_id=self.tokenizer.eos_token_id)
        return self.tokenizer.decode(output[0,ids.shape[1]:],skip_special_tokens=True),int(output.shape[1]-ids.shape[1])

    def info(self):
        return {'backend':'qwen','model_id':MODEL_ID,'model_revision':MODEL_REVISION,
                'adapter_loaded':True,'device':self.device,'quantization':self.quantization}

    def gpu_bytes(self):
        return self.torch.cuda.memory_allocated(0) if self.device=='cuda' else 0

    def close(self):
        del self.model
        if self.device=='cuda':self.torch.cuda.empty_cache()

class ExtractiveBackend:
    """Explicit lightweight mode; it does not load or generate with an LLM."""
    def info(self):
        return {'backend':'extractive','model_id':None,'model_revision':None,
                'adapter_loaded':False,'device':'cpu','quantization':None}
    def gpu_bytes(self):return 0
    def close(self):pass
