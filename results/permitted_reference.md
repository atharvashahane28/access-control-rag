# Exact permitted-search reference

Exact inner-product search on normalized vectors, restricted to each user's permitted corpus. k=5. Every eligible question-user pair is evaluated.

Pairs: 29839. Distinct questions: 648. Users evaluated: 100. Unauthorized results: 0.

Recall here measures recovery of permitted FiQA relevance labels. Later strategies must ALSO measure overlap with the exact permitted top-k; that is a different recall metric. The exact reference has perfect overlap with itself, but is not guaranteed to retrieve labeled evidence or generate a correct answer.

These results are not directly comparable with unrestricted recall: the permitted corpus, relevance denominator, and question-user weighting differ. Repeated questions/users are correlated; these are descriptive means, not confidence intervals.

| Access level | Pairs | Hit@5 | Permitted relevance recall@5 | nDCG@5 | Mean returned | Mean corpus access |
|---|---:|---:|---:|---:|---:|---:|
| low | 909 | 73.3% | 0.599 | 0.539 | 5.0 | 4.8% |
| medium | 7284 | 72.4% | 0.571 | 0.522 | 5.0 | 21.9% |
| high | 21646 | 71.6% | 0.506 | 0.478 | 5.0 | 83.0% |

| Access level | Question type | Pairs | Hit@5 | Permitted relevance recall@5 |
|---|---|---:|---:|---:|
| high | cross_department | 19828 | 71.2% | 0.509 |
| high | in_department | 984 | 67.0% | 0.520 |
| high | mixed | 834 | 84.7% | 0.429 |
| low | in_department | 909 | 73.3% | 0.599 |
| medium | cross_department | 5466 | 72.0% | 0.572 |
| medium | in_department | 1588 | 72.2% | 0.585 |
| medium | mixed | 230 | 80.9% | 0.453 |

## Execution details

FAISS CPU threads: 4. Summed index-build time: 3.326 s. Summed timed batch-search time: 5.084 s.

Search timing excludes loading, index building, warm-up, and relevance evaluation. Per-pair search_ms_amortized is a user's batch time divided by its query count; it must not be interpreted as individual request latency or used for p95 latency. exact_candidate_count counts the permitted passages scored per query, excluding warm-up. Approximate-strategy latency and index-memory comparisons remain future work.

Artifacts: data/reference/<model>/<split>/topk.npz (global passage rows, scores, workload IDs), metrics.csv, user_timings.csv, and meta.json. Passage rows map to saved passage_ids.json.
