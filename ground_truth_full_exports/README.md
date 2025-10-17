# Ground Truth (Full) Exports

Generated from:
- business_csv = business_cleaned.csv
- reviews_csv  = review_cleaned.csv
- sample_n     = 120
- recent_days  = 365
- seed         = 42

## Files
- behavior.jsonl — fact QAs + review-based tasks with answer_gt & citations_gt
- capabilities.jsonl — expected tool traces aligned to your tool classes
- reliability_safety.jsonl — paraphrases/typos + safety probes (PII/confidential/tool_failure)
- citations_index.jsonl — doc ids used for citation checks

## Alignment to evaluation papers
- Behavior: success rate, factual correctness, relevance, latency/cost
- Capabilities: tool selection/invocation accuracy, parameter F1, step success, progress rate
- Reliability/Safety: performance under perturbation, consistency (pass-all-k), policy compliance

All datasets originate from a single source-of-truth, following the "multi-view benchmarking" approach.
