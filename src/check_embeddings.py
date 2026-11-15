"""Step 2 check: run with  python -m src.check_embeddings  from the repo root.

It checks that the saved vectors are sound, then runs a plain exact search
with no permissions: every question is compared with every passage. The
result is the baseline that shows whether the embedding model is good
enough to build on. It is written to:

    results/embedding_baseline.md
"""
import math

import faiss
import numpy as np

from src.config import load_config, set_seed
from src.data import load_corpus, load_qrels, load_queries
from src.embed import load_embeddings


def check_vectors(name, vectors, ids, expected_ids):
    """Stop with a clear message if the saved vectors do not match the data."""
    if ids != expected_ids:
        raise SystemExit(
            f"{name}: saved IDs do not match the dataset. "
            "Re-run:  python -m src.embed --force"
        )
    if vectors.shape[0] != len(ids):
        raise SystemExit(f"{name}: {vectors.shape[0]} vectors but {len(ids)} IDs.")
    if not np.isfinite(vectors).all():
        raise SystemExit(f"{name}: some vectors contain NaN or infinity.")
    lengths = np.linalg.norm(vectors, axis=1)
    if np.abs(lengths - 1).max() > 1e-3:
        raise SystemExit(f"{name}: vectors are not normalized to length 1.")
    print(f"{name}: {vectors.shape[0]} vectors of size {vectors.shape[1]}, all checks passed")


def main():
    cfg = load_config()
    set_seed(cfg["seed"])

    emb = load_embeddings(cfg)
    corpus = load_corpus(cfg)
    queries = load_queries(cfg)
    qrels = load_qrels(cfg)

    passage_ids = emb["passage_ids"]
    question_ids = emb["question_ids"]
    passage_vectors = emb["passage_vectors"]
    question_vectors = emb["question_vectors"]

    check_vectors("Passages ", passage_vectors, passage_ids, corpus["passage_id"].tolist())
    check_vectors("Questions", question_vectors, question_ids, queries["question_id"].tolist())

    # Exact search: compare every question with every passage.
    ks = sorted(cfg["embedding_check"]["ks"])
    max_k = ks[-1]
    index = faiss.IndexFlatIP(passage_vectors.shape[1])
    index.add(passage_vectors)
    _, rows = index.search(question_vectors, max_k)

    relevant = qrels.groupby("question_id")["passage_id"].apply(set).to_dict()

    hit = {k: [] for k in ks}       # is at least one relevant passage in the top k?
    recall = {k: [] for k in ks}    # what share of the relevant passages is in the top k?
    ndcg = []                       # rank-aware score at max_k, the number papers report
    for q, question_id in enumerate(question_ids):
        wanted = relevant[question_id]
        found = [passage_ids[i] in wanted for i in rows[q]]
        for k in ks:
            n_found = sum(found[:k])
            hit[k].append(n_found > 0)
            recall[k].append(n_found / len(wanted))
        gain = sum(1 / math.log2(rank + 2) for rank, ok in enumerate(found) if ok)
        best = sum(1 / math.log2(rank + 2) for rank in range(min(len(wanted), max_k)))
        ndcg.append(gain / best)

    meta = emb["meta"]
    cut = meta.get("share_passages_cut_off")
    stats = [
        ("Model", meta["model"]),
        ("Vector size", meta["vector_size"]),
        ("Most tokens the model reads per text", meta["max_tokens"]),
        ("Passages cut off (too long)", "not measured" if cut is None else f"{cut:.1%}"),
        ("Passages", len(passage_ids)),
        ("Questions", len(question_ids)),
    ]
    for k in ks:
        stats.append((f"Questions with a relevant passage in the top {k}", f"{np.mean(hit[k]):.1%}"))
    for k in ks:
        stats.append((f"Recall@{k} (share of relevant passages found)", f"{np.mean(recall[k]):.3f}"))
    stats.append((f"nDCG@{max_k}", f"{np.mean(ndcg):.3f}"))

    print()
    for name, value in stats:
        print(f"{name:50s} {value}")

    lines = [
        "# Embedding baseline",
        "",
        "Exact search over all passages, with no permissions.",
        "",
        "| Item | Value |",
        "|---|---|",
    ]
    lines += [f"| {name} | {value} |" for name, value in stats]

    results_dir = cfg["paths"]["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)
    out = results_dir / "embedding_baseline.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()