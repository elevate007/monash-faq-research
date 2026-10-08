# When lower loss is not enough: a small FAQ reliability study

## Abstract

We investigate domain adaptation and evidence access using Qwen2.5-1.5B-Instruct and 58 source-attributed Monash FAQ summaries. A three-epoch NF4 QLoRA pilot lowers held-out assistant-token loss from 3.0037 to 1.5889, but strict agent-assisted review supports only one of ten adapter answers, versus two base answers. A follow-up study evaluates ten paraphrased held-out facts and ten unsupported personal/future questions with four generative configurations and two post-processing baselines. Both retrieval-assisted generative systems produce eight fully supported answer bodies, while their unsupported-question behavior differs sharply. The validation-calibrated gate improves refusal counts but rejects five answerable questions. These exploratory results illustrate distinctions between optimization loss, evidence retrieval, answer support, and abstention. They are not population-accuracy estimates.

## Motivation and methods

University FAQs combine factual procedures, audience-specific rules, and information that can change. A fluent wrong answer may be more harmful than an explicit limitation. We therefore examine complete answer support separately from token loss and source formatting.

The fixed dataset split is 43 training facts, five validation facts, and ten test facts. QLoRA uses rank 16, alpha 32, dropout 0.05, all linear layers, assistant-only loss, learning rate 1e-4, three epochs, and seed 42 on one Tesla T4. The best checkpoint is selected by validation loss, then exported and reloaded. The original pilot and follow-up retraining are distinct saved Kaggle versions; the original review is preserved.

The follow-up compares base and adapter under closed-book prompts, then each under the same evidence-restricted RAG prompt. Retrieval is BM25 over all 58 FAQ summaries, returning three entries. A separate ten-case calibration set selects a lexical-coverage gate of 0.50; higher thresholds win tied calibration scores. A gated RAG variant reuses the base+RAG output when accepted, otherwise returns a fixed refusal. The extractive baseline returns the top saved answer when accepted. See [PROTOCOL.md](PROTOCOL.md) for the exact controls and limitations.

Retrieval intentionally sees held-out facts. Thus this is a comparison of evidence-access systems, rather than an equal-information comparison of learning algorithms. The gold target ID and reference answer never select retrieval hits or enter model prompts; relevant knowledge-base content can resemble the reference by construction.

## Results

![Exploratory system comparison](comparison_results.png)

Strict labels require complete essential reference support and no substantive unsupported addition. Partial answers receive zero. For unsupported cases, an explicit unavailable-information refusal is required; helpful general direction alone is insufficient. Labels are agent-assisted and unblinded, with no independent human adjudication.

| Configuration | Fully supported answer bodies / 10 | Appropriate unsupported-question refusals / 10 | Answerable questions refused / 10 | Generated URLs across 20 outputs |
| --- | ---: | ---: | ---: | ---: |
| Base | 0 | 4 | 0 | 8 |
| QLoRA | 0 | 0 | 0 | 21 |
| Base + BM25 RAG | 8 | 6 | 0 | 0 |
| QLoRA + BM25 RAG | 8 | 1 | 0 | 2 |
| Gated base + RAG | 3 | 10 | 5 | 0 |
| Extractive BM25 + gate | 4 | 9 | 5 | 0 |

The original ten exact-question pilot and the follow-up paraphrase study are different evaluations. Their 2/10 and 1/10 pilot counts must not be substituted for the new closed-book 0/10 counts. Several follow-up answers contain correct partial facts but fail the strict full-body rubric: for example, the base visa answer excludes application fees while presenting that as a general tendency and adding unsupported tuition-coverage claims. The review notes explain this conservative judgment, and readers can inspect and relabel the raw outputs.

BM25 achieves Recall@3 of **10/10** and MRR@3 of **0.9333** on the ten answerable cases. The correct fact is ranked third for the VTAC change-fee question and first for the other nine. The gate accepts five answerable questions, four with a correct top fact, and one unsupported future-tuition question. Gated base+RAG still refuses that accepted unsupported question through the model response, giving ten appropriate refusals; the extractive version returns a general fee summary and achieves nine.

The generation run produced 80 actual model responses and 40 derived gated/extractive outputs. Timing and token counts are descriptive only; the gated variants are not independent reruns. Raw prediction, calibration, runtime-version, GPU-memory, and content-hash artifacts are preserved in [results/comparison](../results/comparison).

## Error analysis

| Failure | Observed output | Interpretation |
| --- | --- | --- |
| Course-change route | QLoRA directs the user to an academic advisor; RAG supplies `mu.documents@monash.edu` | Evidence can recover a specific procedural fact |
| Scholarship program scope | QLoRA+RAG says Scholars is available across Years 10-12; entry requires endorsed Year 10 students | Context does not guarantee the model uses the relevant entry criterion |
| Accommodation alternative | Base+RAG suggests a different campus; reference says same-campus alternative | Retrieval can find the fact while generation changes a critical qualifier |
| Timetable trigger | QLoRA+RAG says after course allocation; reference says after enrolment | Partial matching is insufficient for complete procedural support |
| Fee enquiry contamination | Base+RAG adds the course-change email to an international fee response | Mixing top-three excerpts can introduce a wrong contact route |
| Future visa price | QLoRA invents $57; QLoRA+RAG invents $594/$794 and a renewal period | Domain fluency and supplied context do not establish unavailable future facts |
| Personal library count | Base+RAG gives a supported My Loans direction but never states it cannot access the count | The strict explicit-refusal metric distinguishes direction from abstention |
| False gate rejection | Correct international course-change fact ranks first but coverage is 0.444, below 0.50 | Lexical coverage penalizes valid paraphrases and verbose questions |

The adapter generates source URLs even when the RAG prompt forbids them. Candidate source URLs are attached from saved metadata separately, but this only avoids synthesizing metadata links; it does not verify that every generated claim is entailed by those sources. Some wrong answers share the correct source page because several facts originate from that page.

## Interpretation

The pilot demonstrates successful optimization and artifact export, but does not demonstrate reliable FAQ learning. The follow-up evidence-access systems improve strict answer support on this set. The adapter offers no advantage in supported-answer count over the base RAG model and performs worse under the explicit-refusal rubric. We cannot identify a causal mechanism from this single small run. In particular, this does not establish that QLoRA generally reduces refusal ability.

The gate's ten unsupported refusals coexist with five false refusals of answerable questions. Reporting only the refusal result would hide the coverage cost. Likewise, Recall@3 of 1.0 does not mean generated answers are all correct: the model can select a wrong excerpt, ignore an eligibility qualifier, or combine unrelated facts.

## Limitations and follow-up

The benchmark is a twenty-question author-written convenience sample, authored after inspection of the earlier pilot. Source pages and topics are shared across training and test. The calibration set is tiny. There is one training seed, greedy inference, four-bit quantization, no independent annotation, no inter-rater reliability, no matched prompt/evidence causal ablation, and no fresh-source audit. Model outputs sometimes contain token duplication or truncation. Future/personal unsupported questions do not represent every kind of out-of-domain request. Policies may change after the 2026-10-08 snapshot.

A stronger study should freeze an independently authored, larger evaluation set; obtain multiple human reviewers; test refreshed pages and unfamiliar topics; compare equal-information prompt conditions; repeat training seeds; evaluate dense retrieval and reranking; and measure citation entailment separately from URL validity. Scaling training examples should follow source review and clear evaluation goals rather than simply adding synthetic paraphrases.

## Reproducibility and provenance

The public notebooks embed the frozen data and retrieval module. Local CPU checks verify checksums, disjoint fact groups, holdout/calibration target membership, zero-overlap refusal, ranking of a known fact, prompt boundary separation, and deterministic calibration. Preparation also tested actual tokenizer label masks and a tiny random Qwen2 LoRA train/save/reload cycle; those CPU checks are distinct from the actual T4 model training.

Run manifests record resolved model revision, package versions, seed, decoding settings, GPU, and benchmark/calibration/knowledge-base hashes. The actual comparison archive passed ZIP integrity and hash matching, and contained all 120 unique configuration-case records. Large weights are excluded from Git; notebooks reproduce adapters. The project was developed with substantial AI assistance, including code generation, browser execution, curation, and review. Results are research artifacts that require user understanding and independent checking before a substantive research claim.
