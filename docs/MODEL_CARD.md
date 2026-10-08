# Model card

Base: [Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct). Adapter: 18,464,768 trainable parameters; NF4 QLoRA, assistant-only loss, validation-selected epoch. Runtime and resolved model revision are recorded in run manifests. Package versions are pinned, but GPU/PyTorch differences can still affect exact numeric reproducibility.

The initial actual Kaggle T4 run completed in 431.9 seconds, with 146.8 seconds of training and 33 optimizer steps over three epochs. Test loss moved from 3.0037 to 1.5889. Strict agent-assisted review supported 2/10 base answers and 1/10 adapter answers. The adapter invented source URLs on all ten held-out answers. Lower loss did not establish factual reliability.

The follow-up notebook retrains from the same fixed data and settings, exports/reloads a new adapter, and compares evidence-access conditions. Original pilot files are preserved separately. Do not assume the follow-up adapter has identical numeric weights to the original.

Intended use: inspect and reproduce a small negative-result fine-tuning experiment. Not validated for production student advising, personal outcomes, current fees/deadlines, or languages other than English. No personal records are used. Candidate URLs in the RAG system come from frozen metadata and do not prove that every answer sentence is supported.

Large model weights are not committed to Git. The executable training notebooks recreate adapters, and the Kaggle version contains run outputs subject to its visibility settings. Base weights retain the upstream model license; source data retains its own rights. See DATA_CARD.md and LICENSE.
