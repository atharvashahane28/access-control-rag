# Local GPU embedding run

Completed November 16, 2026 on Windows using the existing project scripts and configuration.

## Environment

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 2070, 8 GB |
| NVIDIA driver | 610.88 |
| Python | 3.13.6 |
| Project environment | `.venv/` |
| PyTorch | 2.14.1+cu126 (CUDA 12.6) |
| Sentence Transformers | 6.3.0 |
| Transformers | 5.23.0 |
| FAISS | faiss-cpu 1.16.0 |
| Model | sentence-transformers/all-mpnet-base-v2 |
| Model revision | e8c3b32edf5434bc2275fc9bab85f82640a19130 |
| Batch size | 32 |
| Seed | 42 |
| Embedding device | cuda:0 |

The GPU computation check passed. The project environment check passed, including 10/10 successful HNSW self-matches. `pip check` reported no broken requirements. FAISS validation uses the CPU; embedding generation used the GPU.

## Data and download verification

- Downloaded FiQA and verified the configured archive MD5: `17918ed23cd04fb15047f73e6c3bd9d9`.
- Data validation reproduced 57,600 cleaned passages, 648 test questions, and 1,705 relevance labels.
- The PyTorch wheel SHA-256 matched its official package index: `92d0ae66f10ae9dfaaf85d684d6794a3d6ca4ed938dcfaa25e94c54cff3764eb`.
- Model weights SHA-256 matched Hugging Face metadata: `78c0197b6159d92658e319bc1d72e4c73a9a03dd03815e70e555c5ef05615658`.
- Windows curl was used for the dataset, model weights, and PyTorch wheel after Python downloader certificate/stall issues. Certificate verification remained enabled.
- Model files are cached in `data/hf-cache/`. Generation used offline mode against that cache.

## Generation results

The 1,000-passage trial took 7.7 seconds (130.4 passages/second). Full passage encoding took 7.4 minutes; this excludes download, setup, model loading, token counting, and validation time.

| Output | Shape |
|---|---|
| Passage vectors | 57,600 x 768 |
| Question vectors | 648 x 768 |

Vectors are normalized float32 arrays. The validation script confirmed ID alignment, vector counts, finite values, and unit norms for both arrays. The model reads at most 384 tokens per text; 9.3% of passages were truncated, as in the prior recorded run.

Artifacts are in `data/embeddings/all-mpnet-base-v2/`:

- `passage_vectors.npy`
- `passage_ids.json`
- `question_vectors.npy`
- `question_ids.json`
- `meta.json`

## Retrieval validation

Exact search over the full corpus without permissions reproduced every metric in the existing `results/embedding_baseline.md` at its reported precision. This establishes matching aggregate retrieval metrics, not bitwise identity with the earlier vectors.

| Metric | Result |
|---|---|
| Relevant passage in top 5 (hit rate) | 72.2% |
| Relevant passage in top 10 (hit rate) | 78.2% |
| Recall@5 against FiQA relevance labels | 0.492 |
| Recall@10 against FiQA relevance labels | 0.581 |
| nDCG@10 | 0.500 |

## Reusing the environment

Run from the repository root in PowerShell:

```powershell
# Validate saved vectors; this does not need the model cache or GPU.
.\.venv\Scripts\python.exe -m src.check_embeddings

# Configure the existing local model cache before another embedding trial.
$env:HF_HOME = "$PWD\data\hf-cache"
$env:HF_HUB_OFFLINE = '1'
.\.venv\Scripts\python.exe -m src.embed --test
```

The full generation command is `python -m src.embed` using the project environment. Existing completed embeddings are protected unless `--force` is supplied; that option replaces them.

Detailed execution logs, environment metadata, a package freeze, and a copy of the prior baseline are saved in `data/run-logs/`. The environment, cache, logs, and embeddings are ignored by Git. No project source files, configuration, or requirements files were changed.
