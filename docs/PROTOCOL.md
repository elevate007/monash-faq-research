# Experimental protocol

Protocol fixed before the comparison GPU run on 2026-10-08. The earlier ten-case pilot had already been inspected; this follow-up is exploratory, not a blinded or independent confirmatory benchmark.

## Research questions

1. Does a reduction in assistant-token cross-entropy translate into supported FAQ answers?
2. How does supplying retrieved evidence change answer support and unsupported-question handling?
3. Does a simple validation-calibrated retrieval gate trade answer coverage for abstention?

## Data and information access

The snapshot has 58 brief English paraphrases from ten official Monash public pages. The fixed training, validation, and test splits contain 43, 5, and 10 distinct fact groups. Each topic has one test fact. The source page is shared across splits within a topic: this is fact-level holdout, not unseen-source or unseen-topic generalization. Related training facts may help infer held-out facts.

The comparison has ten newly phrased questions about the original ten test facts, plus ten author-written questions asking for personal or future details absent from the snapshot. The questions were authored after the first pilot review, before this comparison run. They were not added to fine-tuning. This small convenience sample cannot establish population accuracy.

Retrieval indexes **all 58 facts**, including the test and validation facts. This represents a source-access system. Closed-book models see 43 training facts during fine-tuning and receive no retrieved answer at inference. These systems have different information access, so a retrieval improvement is not proof of superior model learning. The knowledge-base entry can be close to the reference answer by construction. The benchmark reference fields and target IDs are used for evaluation only, never to select retrieval hits or build prompts.

## Configurations

| Name | Model and inference evidence |
| --- | --- |
| base | Qwen2.5-1.5B-Instruct, same 4-bit base weights, adapter disabled |
| qlora | Exported and reloaded adapter, no retrieved context |
| rag_base | Base model plus three BM25 FAQ excerpts |
| rag_qlora | Adapter plus the same three excerpts and same RAG prompt |
| gated_rag_base | Reuse rag_base response only if calibrated top-hit coverage passes; otherwise fixed refusal |
| extractive_bm25 | Top retrieved answer if the same gate passes; otherwise fixed refusal |

The RAG prompt differs from the closed-book prompt: it asks for an answer supported only by the supplied excerpts and asks the model not to generate URLs. Prompt, evidence, and citation handling change together; the main comparison is a system comparison rather than a pure retrieval-only causal ablation. The base/adapter comparison within each prompt condition keeps the prompt fixed.

## Fixed settings

Training: NF4 with double quantization, LoRA rank 16/alpha 32/dropout 0.05/all linear layers, learning rate 1e-4, three epochs, batch one, accumulation four, maximum sequence 512, seed 42. Assistant-only labels mask the system/user prefix and padding. Best epoch is selected by validation loss. Test loss does not select checkpoints. FP16 compute on T4; CUDA FP16 gradient AMP disabled. One T4 is used even with Kaggle T4 x2 selected.

Inference: greedy decoding, 160 maximum new tokens, no sampling, maximum prompt 1024 with an explicit error rather than silent truncation. Base and adapter share quantization. Four generative configurations run sequentially; latency is descriptive, with no warm-up or repeated timing claim. Gated outputs reuse the generated answer and do not count as additional independent generation.

Retrieval: standard positive-IDF BM25, k1=1.5, b=0.75, lowercase alphanumeric tokens, published stopword list, question+answer documents, top three, deterministic record-ID tie break. Coverage is the fraction of distinct non-stopword query tokens found in the top document. Five validation-fact paraphrases plus five separate unsupported prompts select a threshold from 0.00 through 1.00 at 0.05 steps plus 1.01. Maximize correct decisions (accept a correct top hit for an answerable case; refuse an unsupported case), breaking ties toward higher abstention. No benchmark outcomes choose this threshold. Lexical coverage is a crude proxy and not calibrated probability.

## Metrics and review

Retrieval: Recall@3 and MRR@3 use exact target fact IDs. Gate decisions are reported separately from generated answer correctness. Candidate source URLs are attached directly from retrieved metadata, never synthesized. Candidate source membership is **not** entailment or verified citation support.

Model outputs: publish all raw answers, generated URLs, refusal surface heuristic, timing, and token counts. Review each answerable answer against the saved reference using the rubric in REVIEW_RUBRIC.md. Review unsupported answers for explicit refusal without invented personal details, dates, or prices. Automatic refusal matching is only a convenience heuristic. Manual labels are agent-assisted judgments, without independent human adjudication or inter-rater reliability. Report counts with denominators; do not describe them as production accuracy. Refusing an answerable question receives zero for answer correctness.

## Analysis commitments

Report every configuration and failed case. Preserve the original pilot export and distinguish the new retraining run. Do not tune on the benchmark, remove hard cases, or report lower loss as factual accuracy. Describe source snapshot drift, convenience sampling, shared source pages, tiny calibration data, prompt confounds, quantization, and missing independent review. Future work should collect an independently authored larger evaluation set and compare equal-information conditions, multiple seeds, dense retrieval, reranking, and refreshed sources.
