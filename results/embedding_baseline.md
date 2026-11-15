# Embedding baseline

Exact search over all passages, with no permissions.

| Item | Value |
|---|---|
| Model | sentence-transformers/all-mpnet-base-v2 |
| Vector size | 768 |
| Most tokens the model reads per text | 384 |
| Passages cut off (too long) | 9.3% |
| Passages | 57600 |
| Questions | 648 |
| Questions with a relevant passage in the top 5 | 72.2% |
| Questions with a relevant passage in the top 10 | 78.2% |
| Recall@5 (share of relevant passages found) | 0.492 |
| Recall@10 (share of relevant passages found) | 0.581 |
| nDCG@10 | 0.500 |
