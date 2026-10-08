# Application and interview notes

## Suggested CV entry

**Monash FAQ: Parameter-Efficient Fine-Tuning and Retrieval Evaluation** | Python, PyTorch, Hugging Face, PEFT, Kaggle, BM25

- Developed a reproducible Qwen2.5-1.5B QLoRA pipeline with assistant-only loss, source-attributed FAQ records, fixed fact-level splits, validation-selected checkpoints, and adapter export/reload.
- Compared closed-book base/adapter models with retrieval-assisted answering and abstention; published raw outputs, agent-assisted review labels, integrity checks, and an error analysis separating loss from factual support.
- Investigated a negative fine-tuning result and the tradeoff between evidence retrieval and refusal, with an English research report and a local extractive demonstration.
- Extended the project with an AI-assisted FastAPI serving layer, source-grounded evidence selection, input/output guards, Prometheus metrics, Grafana provisioning, and API integration tests.

Project: https://github.com/elevate007/monash-faq-research

Use these statements only after you have personally run the notebooks, inspected the artifacts, and can explain the code and decisions. The project was built with substantial AI assistance; do not claim independent manual authorship, independent human review, an official Monash collaboration, a publication, or production deployment. Tailor your wording to your actual contribution and the position's requirements.

## Topics to be ready to explain

1. Why assistant-only masking matters, and how the chat-template boundary and padding are checked.
2. What NF4 and LoRA change, how many parameters are trainable, and why a single T4 suffices for this small model.
3. Why test loss cannot establish answer correctness, and how the original wrong answers illustrate that.
4. Why RAG sees held-out facts by design, why this is not training leakage, and why it changes the information-access comparison.
5. How BM25 scores documents, what Recall@3 and MRR@3 measure, and why source URL membership is not entailment.
6. How calibration differs from benchmark evaluation, and why a lexical gate can reject valid paraphrases.
7. Why one seed, twenty convenience cases, shared source pages, changed prompts, and agent-assisted labels limit conclusions.
8. What a stronger follow-up would require: fresh independent annotations, equal-information controls, multiple seeds, stronger retrieval baselines, and current sources.

## Contribution checklist

Before sending an RA application, reproduce a run yourself, review at least the ten answerable comparisons against the sources, identify two additional failure cases, and write your own short interpretation. These steps establish your understanding; this repository alone cannot guarantee eligibility or selection for a specific RA role.
