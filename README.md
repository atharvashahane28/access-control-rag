# Permission-Aware Retrieval Augmented Generation (RAG)

## Does Access Control Hurt RAG Answers? Measuring Strategy-Induced Quality Loss in Permission-Aware Vector Search

Researchers: Atharva Shahane, Patricio Pizana

Faculty: Nick Koudas

Institution: University of Toronto, CSC2508 (Advanced Data Systems)

## Setup

```bash
python3 -m venv env
source env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python -m src.hello
```

All settings are in `configs/config.yaml`. Run every script from the repo root.

## Pipeline

```bash
python -m src.data
python -m src.check_data
python -m src.embed --test
python -m src.embed
python -m src.check_embeddings
python -m src.organization
python -m src.reference
python -m unittest discover -s tests -v
```

On this Windows machine, use `.\.venv\Scripts\python.exe` instead of `python`.
For embedding generation with the downloaded local model cache, set these
PowerShell variables first:

```powershell
$env:HF_HOME = "$PWD\data\hf-cache"
$env:HF_HUB_OFFLINE = '1'
```

Organization and reference generation reuse saved embeddings and run on CPU.
Completed artifacts are protected from replacement unless `--force` is supplied.
Their metadata records configuration and input/artifact SHA-256 hashes; changed
inputs require rebuilding downstream artifacts. Large files remain under ignored `data/`.

Step 3 clusters passages into synthetic departments, creates overlapping roles
and single-/multi-role users, and keeps every question-user pair with at least one
permitted relevant passage. Questions are labeled home-only, cross-only, or mixed
using only permitted relevant evidence. The initial role sizes and user counts are
experimental defaults, not claims about real organizations. Results are in
`results/organization_summary.md`.

Step 4 saves exact permitted top-k passage rows and scores, aligned to the fixed
workload. `src.reference.load_reference(cfg)` loads the validated artifacts for
later strategy comparisons. `results/permitted_reference.md` reports retrieval of
permitted FiQA relevance labels. This differs from ANN recall against the exact
permitted top-k, which later strategies must also measure. Current search timings
are amortized batch timings, not per-request latency measurements.

See [TIMELINE.md](TIMELINE.md) for the completed milestones, validation summary,
and ordered next steps. Detailed results are in
[the GPU run report](results/gpu_embedding_run.md),
[the organization report](results/organization_summary.md), and
[the exact permitted-reference report](results/permitted_reference.md).
