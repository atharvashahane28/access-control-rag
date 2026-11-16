"""Step 4: exact top-k search within each user's permitted corpus.

Run python -m src.reference [--force]. This is the permitted search reference,
not an approximate strategy and not an answer-quality measurement.
"""
import argparse
import json
import math
import time

import faiss
import numpy as np
import pandas as pd

from src.config import load_config, set_seed
from src.data import load_qrels
from src.embed import load_embeddings
from src.organization import (LEVELS, file_hash, load_organization, organization_dir,
                              validate_organization, write_json)


def reference_dir(cfg):
    return (cfg["paths"]["data_dir"] / "reference"
            / cfg["embedding"]["model"].split("/")[-1] / cfg["dataset"]["split"])


def exact_permitted_search(passage_vectors, question_vectors, allowed_rows, k):
    """Return global passage rows, scores, index-build time, and batch-search time.

    Missing results are padded with row -1 and score -inf. The timed search is
    batched and warmed up; its time divided by query count is amortized latency,
    not an individually measured request-latency distribution.
    """
    allowed_rows = np.asarray(allowed_rows, dtype=np.int64)
    if k < 1 or len(allowed_rows) == 0 or len(question_vectors) == 0:
        raise ValueError("Search requires positive k, permitted passages, and questions")
    if len(np.unique(allowed_rows)) != len(allowed_rows) or np.any(allowed_rows < 0) or np.any(allowed_rows >= len(passage_vectors)):
        raise ValueError("Invalid or duplicate permitted passage rows")
    start = time.perf_counter()
    index = faiss.IndexFlatIP(passage_vectors.shape[1])
    index.add(np.ascontiguousarray(passage_vectors[allowed_rows], dtype=np.float32))
    build_seconds = time.perf_counter() - start
    queries = np.ascontiguousarray(question_vectors, dtype=np.float32)
    index.search(queries[:1], k)  # warm-up excluded from timed search
    start = time.perf_counter()
    scores, local_rows = index.search(queries, k)
    seconds = time.perf_counter() - start
    valid = local_rows >= 0
    rows = np.full(local_rows.shape, -1, dtype=np.int64)
    rows[valid] = allowed_rows[local_rows[valid]]
    scores[~valid] = -np.inf
    return rows, scores, build_seconds, seconds


def relevance_metrics(found_ids, relevant_ids, k):
    """Binary FiQA relevance metrics; denominator includes only permitted labels."""
    hits = [pid in relevant_ids for pid in found_ids]
    dcg = sum(1 / math.log2(rank + 2) for rank, hit in enumerate(hits) if hit)
    ideal = sum(1 / math.log2(rank + 2) for rank in range(min(k, len(relevant_ids))))
    return {"relevant_hit": int(any(hits)), "relevance_recall": sum(hits) / len(relevant_ids),
            "ndcg": dcg / ideal}


def reference_signature(cfg):
    return {"schema_version": 1, "organization_meta_hash": file_hash(organization_dir(cfg) / "meta.json"),
            "k": cfg["retrieval"]["k"], "reference": cfg["reference"]}


def load_reference(cfg):
    """For later strategies: fixed workload plus aligned exact permitted top-k."""
    organization = load_organization(cfg)
    folder = reference_dir(cfg)
    if not (folder / "meta.json").exists():
        raise ValueError("No completed reference. Run python -m src.reference")
    meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    if meta["inputs"] != reference_signature(cfg):
        raise ValueError("Reference inputs changed; rebuild with python -m src.reference --force")
    for name, expected in meta["artifact_hashes"].items():
        if not (folder / name).exists() or file_hash(folder / name) != expected:
            raise ValueError(f"Reference artifact changed or missing: {name}")
    with np.load(folder / "topk.npz", allow_pickle=False) as arrays:
        topk = {name: arrays[name] for name in arrays.files}
    workload = organization["workload"]
    if not np.array_equal(topk["workload_ids"], workload.workload_id.to_numpy()):
        raise ValueError("Reference and workload are not aligned")
    return {"organization": organization, "topk": topk,
            "metrics": pd.read_csv(folder / "metrics.csv", dtype={"question_id": str}), "meta": meta}


def report(cfg, metrics, user_timings):
    k = cfg["retrieval"]["k"]
    lines = ["# Exact permitted-search reference", "",
             f"Exact inner-product search on normalized vectors, restricted to each user's "
             f"permitted corpus. k={k}. Every eligible question-user pair is evaluated.", "",
             f"Pairs: {len(metrics)}. Distinct questions: {metrics.question_id.nunique()}. "
             f"Users evaluated: {metrics.user_id.nunique()}. Unauthorized results: 0.", "",
             "Recall here measures recovery of permitted FiQA relevance labels. "
             "Later strategies must ALSO measure overlap with the exact permitted top-k; "
             "that is a different recall metric. The exact reference has perfect overlap with itself, "
             "but is not guaranteed to retrieve labeled evidence or generate a correct answer.", "",
             "These results are not directly comparable with unrestricted recall: the permitted "
             "corpus, relevance denominator, and question-user weighting differ. "
             "Repeated questions/users are correlated; these are descriptive means, not confidence intervals.", "",
             f"| Access level | Pairs | Hit@{k} | Permitted relevance recall@{k} | nDCG@{k} | Mean returned | Mean corpus access |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for level in LEVELS:
        group = metrics[metrics.access_level == level]
        lines.append(f"| {level} | {len(group)} | {group.relevant_hit.mean():.1%} | "
                     f"{group.relevance_recall.mean():.3f} | {group.ndcg.mean():.3f} | "
                     f"{group.n_returned.mean():.1f} | {group.access_fraction.mean():.1%} |")
    lines += ["", f"| Access level | Question type | Pairs | Hit@{k} | Permitted relevance recall@{k} |",
              "|---|---|---:|---:|---:|"]
    for (level, category), group in metrics.groupby(["access_level", "question_type"], sort=True):
        lines.append(f"| {level} | {category} | {len(group)} | {group.relevant_hit.mean():.1%} | {group.relevance_recall.mean():.3f} |")
    lines += ["", "## Execution details", "",
              f"FAISS CPU threads: {cfg['reference']['n_threads']}. "
              f"Summed index-build time: {user_timings.build_seconds.sum():.3f} s. "
              f"Summed timed batch-search time: {user_timings.search_seconds.sum():.3f} s.", "",
              "Search timing excludes loading, index building, warm-up, and relevance evaluation. "
              "Per-pair search_ms_amortized is a user's batch time divided by its query count; "
              "it must not be interpreted as individual request latency or used for p95 latency. "
              "exact_candidate_count counts the permitted passages scored per query, excluding warm-up. "
              "Approximate-strategy latency and index-memory comparisons remain future work.", "",
              "Artifacts: data/reference/<model>/<split>/topk.npz (global passage rows, scores, workload IDs), "
              "metrics.csv, user_timings.csv, and meta.json. Passage rows map to saved passage_ids.json.", ""]
    return "\n".join(lines)


def run(cfg, force=False):
    folder = reference_dir(cfg)
    if (folder / "meta.json").exists() and not force:
        raise ValueError("Reference already exists; use --force to replace it")
    org = load_organization(cfg)
    emb = load_embeddings(cfg)
    departments = org["departments"]["department_id"].to_numpy()
    if org["departments"].passage_id.tolist() != emb["passage_ids"]:
        raise ValueError("Department and embedding passage orders differ")
    workload = org["workload"]
    if not np.array_equal(workload.workload_id.to_numpy(), np.arange(len(workload))):
        raise ValueError("Workload IDs are not contiguous")
    validate_organization(departments, org["roles"], org["users"], workload,
                          len(emb["passage_ids"]), cfg["organization"]["n_departments"])
    qrels = load_qrels(cfg)
    relevant = qrels.groupby("question_id").passage_id.apply(set).to_dict()
    query_rows = {qid: i for i, qid in enumerate(emb["question_ids"])}
    passage_ids = np.array(emb["passage_ids"])
    k = cfg["retrieval"]["k"]
    rows = np.full((len(workload), k), -1, dtype=np.int64)
    scores = np.full((len(workload), k), -np.inf, dtype=np.float32)
    records, timings = [], []
    previous_threads = faiss.omp_get_max_threads()
    faiss.omp_set_num_threads(cfg["reference"]["n_threads"])
    try:
        for user in org["users"]:
            group = workload[workload.user_id == user["user_id"]]
            if group.empty:
                continue
            allowed_rows = np.flatnonzero(np.isin(departments, user["department_ids"]))
            allowed_ids = set(passage_ids[allowed_rows])
            queries = emb["question_vectors"][[query_rows[qid] for qid in group.question_id]]
            found, similarities, build_seconds, seconds = exact_permitted_search(
                emb["passage_vectors"], queries, allowed_rows, k)
            positions = group.workload_id.to_numpy()
            rows[positions], scores[positions] = found, similarities
            timings.append({"user_id": user["user_id"], "n_queries": len(group),
                            "n_allowed_passages": len(allowed_rows),
                            "build_seconds": build_seconds, "search_seconds": seconds})
            for pair, result in zip(group.to_dict("records"), found):
                valid = result[result >= 0]
                found_ids = passage_ids[valid].tolist()
                if not set(found_ids).issubset(allowed_ids) or len(set(found_ids)) != len(found_ids):
                    raise ValueError("Forbidden or duplicate reference results")
                permitted_relevant = relevant[pair["question_id"]] & allowed_ids
                if len(permitted_relevant) != pair["n_relevant_permitted"]:
                    raise ValueError("Workload relevance counts differ from permissions")
                records.append({**pair, "n_returned": len(valid),
                                **relevance_metrics(found_ids, permitted_relevant, k),
                                "exact_candidate_count": len(allowed_rows),
                                "search_ms_amortized": seconds * 1000 / len(group)})
            print(f"{user['user_id']}: {len(group)} queries, {len(allowed_rows)} permitted passages", flush=True)
    finally:
        faiss.omp_set_num_threads(previous_threads)
    metrics = pd.DataFrame(records).sort_values("workload_id").reset_index(drop=True)
    user_timings = pd.DataFrame(timings)
    if len(metrics) != len(workload) or not np.array_equal(metrics.workload_id, workload.workload_id):
        raise ValueError("Some workload pairs were not evaluated")
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "meta.json").unlink(missing_ok=True)
    np.savez_compressed(folder / "topk.npz", passage_rows=rows, scores=scores,
                        workload_ids=workload.workload_id.to_numpy())
    metrics.to_csv(folder / "metrics.csv", index=False)
    user_timings.to_csv(folder / "user_timings.csv", index=False)
    names = ("topk.npz", "metrics.csv", "user_timings.csv")
    meta = {"inputs": reference_signature(cfg), "n_pairs": len(workload),
            "unauthorized_results": 0, "faiss_version": faiss.__version__,
            "artifact_hashes": {name: file_hash(folder / name) for name in names}}
    out = cfg["paths"]["results_dir"] / "permitted_reference.md"
    out.write_text(report(cfg, metrics, user_timings), encoding="utf-8")
    write_json(folder / "meta.json", meta)
    print(f"Validated and saved {len(workload)} exact permitted top-{k} results. Wrote {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    cfg = load_config()
    set_seed(cfg["seed"])
    run(cfg, args.force)


if __name__ == "__main__":
    main()
