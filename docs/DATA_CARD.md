# Dataset card

## Scope

58 short English FAQ summaries, ten official Monash pages, observed 2026-10-08. Topics: international applications, VTAC preferences, online study, Monash Scholars, graduate research offers, accommodation, Victorian library borrowing, graduate coursework, scholarships, and international postgraduate study. Intended use: an educational research pilot about domain adaptation and evidence-grounded answering.

## Acquisition and transformation

Accessible official page content was inspected using web search/open tools and summarized into independently phrased short answers. This is a curated dataset, not a completed bulk crawler. Each record retains URL, audience, topic, date, transformation, fact group, and split. Original pages are not redistributed in full. The optional collector checks robots rules and skips access denials; ordinary HTTP access returned a robots denial during preparation, and no bypass was used. Automatically collected candidates require review before inclusion.

## Splits and follow-up prompts

43 train / 5 validation / 10 test records. Fact groups are disjoint, while source pages and topics are shared. Benchmark: ten paraphrases of test facts and ten unanswerable personal/future questions. Calibration: five validation-fact paraphrases and five separate unsupported questions. The comparison RAG knowledge base intentionally includes all 58 facts. See PROTOCOL.md for why this differs from closed-book fine-tuning.

## Quality and limitations

Small manually curated convenience sample; no independent source audit, double annotation, diverse paraphrase study, or longitudinal refresh. Answers may lose qualifications through summarization. Policies and prices can change. Some FAQ facts share source pages and semantic overlap. Neither the data nor model should be used to decide a real application outcome. The project is independent and is not endorsed by Monash University.

## Rights and attribution

Official source content remains subject to the respective source site's rights and terms. The repository does not claim to relicense university content. Source-derived summaries and records are provided with provenance for research inspection; users must assess terms for their intended reuse. The code license is separate from the data. No Monash logos, private student records, credentials, or original full pages are included.
