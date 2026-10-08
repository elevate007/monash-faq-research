# Answer review rubric

Review `answer_body` without using generated URLs as evidence. Compare with the frozen source-grounded reference, respecting applicant type, study mode, and scope.

For an answerable case, `manual_correct=1` requires all essential reference facts and no substantive unsupported addition or contradiction. A partial answer, ambiguous denial, invented procedure, unsupported amount, or refusal receives zero. Harmless phrasing differences are allowed. Source metadata is reviewed separately; a valid-looking URL cannot rescue an incorrect answer.

For an unsupported case, `manual_refusal=1` requires a clear statement that the requested detail is unavailable or cannot be established. General direction to the official team is acceptable. Invented personal status, definitive future price/date, or unsupported guarantee receives zero. A refusal followed by unsupported specifics is zero.

Use `manual_notes` to record the particular missing, invented, or contradictory fact. Keep raw outputs unchanged. Labels are agent-assisted judgments, not independent human review; readers should inspect and challenge them. No human inter-rater reliability has been measured.

The ten original pilot labels use the same conservative factual-support standard. Source URL matching in that pilot is exact-string metadata matching, not a full external URL crawl.
