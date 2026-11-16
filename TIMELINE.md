# Project timeline

Updated November 16, 2026. Completed milestones have verified results; future milestones are ordered by dependency, without estimated dates.

## Timeline at a glance

| Order | Status | Milestone | Outcome or completion criterion |
|---|---|---|---|
| 1 | Complete before this update | FiQA preparation and unrestricted embedding baseline | Existing download, cleaning, embedding, and validation scripts; baseline committed previously |
| 2 | Complete: November 16, 2026 | Reproduce embeddings on the local RTX 2070 | 57,600 passage and 648 question embeddings; prior baseline reproduced |
| 3 | Complete: November 16, 2026 | Synthetic organization and fixed workload | 20 departments, 60 roles, 100 users, 29,839 eligible pairs covering all 648 questions |
| 4 | Complete: November 16, 2026 | Exact permitted-search reference | Top 5 saved for every eligible pair; zero unauthorized results; validation passed |
| 5 | Next | Shared HNSW post-filtering | Candidate/search-effort sweep evaluated against the fixed permitted reference |
| 6 | Planned | Per-role HNSW indexes | Multi-role merge/deduplication, retrieval metrics, and actual index-memory measurements |
| 7 | Planned; investigate API support early | In-search filtering | Verified traversal semantics and a correctly labeled implementation |
| 8 | Planned | Controlled retrieval comparison | Development/test separation, matched timing, and access-level/department breakdowns |
| 9 | Planned | Fixed RAG pipeline and answer evaluation | Generator and separate judge selected; about 300 questions evaluated; about 50 judgments checked manually |
| 10 | Planned | Analysis and final report | Answer-quality/retrieval-quality relationships and memory/latency at comparable answer quality |
| 11 | Optional stretch | HoneyBee-style partitions and answer-aware tuning | Pursue after the required comparisons are complete |

## Completed work

1. Generated and validated 57,600 passage embeddings and 648 question embeddings on the RTX 2070 in the previous milestone. The unrestricted retrieval baseline was reproduced.
2. Added `src.organization`: seeded k-means clustering, overlapping roles, single- and multi-role users, permission unions, and a fixed eligible question-user workload.
3. Added `src.reference`: exact top-5 search over each user's permitted passages, saved passage rows/scores, and per-pair relevance metrics.
4. Added configuration for clustering, role sizes, user counts, and CPU thread limits; documented commands and experimental semantics in the README.
5. Added provenance checks and focused automated tests. Embeddings were reused, not regenerated.

## Workload generated

| Item | Result |
|---|---:|
| Synthetic departments | 20 |
| Overlapping roles | 60 |
| Users | 100 |
| Single-role users | 60 |
| Multi-role users | 40 |
| Eligible question-user pairs | 29,839 |
| Distinct test questions covered | 648 |
| K-means iterations | 81 |
| Clustering time | 30.0 seconds |

The experimental defaults are one low-access user and two medium/high-access users per home department. Base roles allow 1, 4, or 16 departments. The second medium/high user gets an additional outside singleton role, yielding 5 or 17 departments. These choices are configurable assumptions for the first workload, not calibrated claims about real organizations. Department sizes vary, so corpus access is measured from passage counts.

Every pair includes at least one permitted relevant passage. All eligible pairs are retained rather than sampling one user per question. In-department, cross-department, and mixed labels use only permitted relevant passages. Mixed cases are retained separately. The same saved pairs and permissions must be used by all strategies.

## Reference results

| Access level | Users | Pairs | Mean corpus access per user | Relevant passage in top 5 | Permitted-label Recall@5 |
|---|---:|---:|---:|---:|---:|
| Low | 20 | 909 | 5.0% | 73.3% | 0.599 |
| Medium | 40 | 7,284 | 21.6% | 72.4% | 0.571 |
| High | 40 | 21,646 | 82.9% | 71.6% | 0.506 |

Corpus access in this table averages users equally; retrieval metrics average eligible pairs within each level. All searches returned five passages. Unauthorized results: **zero**.

These numbers describe the exact permitted reference. They do not measure strategy-induced quality loss or answer quality. In particular, differing permitted relevance denominators and pair weights mean that the recall differences above must not be interpreted as an effect of access level on answer quality. All levels cover all 648 questions, but question frequencies differ between levels.

Later strategies need two separately named recall metrics: recovery of the exact permitted top-5 neighbors, and recovery of permitted FiQA relevance labels. Current search timings are warmed-up, amortized batch measurements, not individual request latency; full latency and memory comparisons remain to be implemented.

## Validation performed

- Seven unit tests passed: deterministic role generation, permission unions, eligible question assignment, mixed-evidence labels, missing-question rejection, stale/tampered artifacts, exact search against an independent brute-force oracle, result padding, and relevance-metric semantics.
- Audited all 29,839 real workload pairs for permissions, eligibility counts, and category correctness.
- Independently compared exact scores with NumPy brute force on 21 real pairs spanning every observed access-level/question-type combination; all passed within float32 tolerance.
- Confirmed no unauthorized results and all 648 test questions covered.
- Artifact readers verify input/configuration and output SHA-256 hashes to reject stale or altered saved data.

## Artifacts and rerunning

- Organization report: `results/organization_summary.md`.
- Exact-reference report: `results/permitted_reference.md`.
- Organization data: `data/organization/all-mpnet-base-v2/test/`.
- Reference data: `data/reference/all-mpnet-base-v2/test/`.
- Run logs and audit: `data/run-logs/`.

From the repository root in PowerShell:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m src.organization
.\.venv\Scripts\python.exe -m src.reference
```

The last two commands refuse to replace completed artifacts. Use `--force` only when rebuilding intentionally; rebuild the reference after regenerating the organization. These steps run on CPU and do not download or regenerate embeddings. The repository includes source, tests, configuration, this timeline, and small result reports. Dataset files, vectors, organization/reference data, caches, logs, and the local environment remain excluded from Git.

## Next steps

1. Implement shared-index HNSW post-filtering with configurable candidate counts and search effort. Compare against the saved exact permitted results on the identical workload. Record shortages and authorization violations explicitly.
2. Implement per-role HNSW indexes and merge/deduplicate results for multi-role users. Measure actual index memory, including document duplication across overlapping roles.
3. Verify the installed FAISS API's filtering/traversal behavior before implementing and naming the in-search strategy. Document whether forbidden nodes can be traversed or returned. Do not substitute post-filtering and call it in-search filtering.
4. Establish matched timing protocols, record per-query retrieval results, and compare quality/cost by access level and question type. Do not tune retrieval settings on final test outcomes; define a development workload first.
5. Choose the small open generator and separate judge, freeze generation/judging prompts, select about 300 evaluation questions with a documented user-pair sampling rule, and manually audit about 50 judgments. Mixed cases and repeated questions must be handled explicitly when sampling and estimating uncertainty.
6. Produce answer-quality versus retrieval-quality and cost comparisons, then write the final report. HoneyBee-style partitions and answer-aware tuning remain optional stretch work.
