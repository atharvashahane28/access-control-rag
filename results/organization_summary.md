# Synthetic organization

Deterministic k-means over the saved normalized passage embeddings. Departments are synthetic content clusters, not real company departments.

Seed: 42. Departments: 20. Roles: 60. Users: 100. Clustering: 30.0 seconds.

Roles overlap. Low roles cover 1 department; medium roles cover 4; high roles cover 16. Additional users per home/level add outside singleton roles, one extra department per user variant. Permissions are the union of roles.

All eligible question-user pairs are retained; each has at least one permitted relevant passage. Counts are not balanced across groups; compare group means separately. Repeated questions/users are correlated observations, not independent samples.

Question type uses permitted relevant evidence: in_department = home only; cross_department = other accessible departments only; mixed = both. Mixed pairs remain separate. Low-access users cannot have cross-department questions.

| Access level | Users | Pairs | Distinct questions | Mean corpus access |
|---|---:|---:|---:|---:|
| low | 20 | 909 | 648 | 5.0% |
| medium | 40 | 7284 | 648 | 21.6% |
| high | 40 | 21646 | 648 | 82.9% |

| Access level | Question type | Pairs |
|---|---|---:|
| high | cross_department | 19828 |
| high | in_department | 984 |
| high | mixed | 834 |
| low | in_department | 909 |
| medium | cross_department | 5466 |
| medium | in_department | 1588 |
| medium | mixed | 230 |

| Department | Passages |
|---|---:|
| 0 | 2709 |
| 1 | 2499 |
| 2 | 4286 |
| 3 | 3634 |
| 4 | 5204 |
| 5 | 2191 |
| 6 | 3116 |
| 7 | 2483 |
| 8 | 3020 |
| 9 | 3943 |
| 10 | 2852 |
| 11 | 1669 |
| 12 | 2268 |
| 13 | 2982 |
| 14 | 1910 |
| 15 | 3435 |
| 16 | 1533 |
| 17 | 2804 |
| 18 | 3109 |
| 19 | 1953 |

Artifacts: data/organization/<model>/<split>/. Run `python -m src.reference` next.
