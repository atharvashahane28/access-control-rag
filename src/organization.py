"""Step 3: build a reproducible synthetic organization and eligible workload.

Run python -m src.organization [--force]. Large artifacts live in data/organization/;
results/organization_summary.md documents the workload and modeling assumptions.
"""
import argparse
import hashlib
import json
import time

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits

from src.check_embeddings import check_vectors
from src.config import load_config, set_seed
from src.data import dataset_dir, load_corpus, load_qrels, load_queries
from src.embed import embeddings_dir, load_embeddings

SCHEMA_VERSION = 1
LEVELS = ("low", "medium", "high")


def organization_dir(cfg):
    return (cfg["paths"]["data_dir"] / "organization"
            / cfg["embedding"]["model"].split("/")[-1] / cfg["dataset"]["split"])


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def input_signature(cfg):
    """Invalidate downstream artifacts when vectors, labels, or settings change."""
    folder = embeddings_dir(cfg)
    names = ("passage_vectors.npy", "question_vectors.npy", "passage_ids.json",
             "question_ids.json", "meta.json")
    return {
        "schema_version": SCHEMA_VERSION,
        "seed": cfg["seed"],
        "organization": cfg["organization"],
        "dataset": {key: cfg["dataset"][key] for key in ("name", "split")},
        "embedding": {key: cfg["embedding"][key] for key in ("model", "query_prefix")},
        "embedding_hashes": {name: file_hash(folder / name) for name in names},
        "qrels_hash": file_hash(dataset_dir(cfg) / "qrels" / (cfg["dataset"]["split"] + ".tsv")),
    }


def validated_inputs(cfg):
    emb = load_embeddings(cfg)
    corpus, queries, qrels = load_corpus(cfg), load_queries(cfg), load_qrels(cfg)
    for kind, table, column in (("passage", corpus, "passage_id"),
                                ("question", queries, "question_id")):
        vectors, ids = emb[kind + "_vectors"], emb[kind + "_ids"]
        if vectors.ndim != 2 or vectors.dtype != np.float32 or len(set(ids)) != len(ids):
            raise ValueError(f"Invalid {kind} vectors or duplicate IDs")
        check_vectors(kind, vectors, ids, table[column].tolist())
    meta = emb["meta"]
    if (meta["model"] != cfg["embedding"]["model"]
            or meta["query_prefix"] != (cfg["embedding"]["query_prefix"] or "")
            or emb["passage_vectors"].shape[1] != emb["question_vectors"].shape[1]):
        raise ValueError("Embeddings do not match the configured model/prefix/dimensions")
    return emb, qrels


def make_roles_users(settings, seed):
    """One role per home/level; extra users add distinct outside singleton roles."""
    n = settings["n_departments"]
    counts = settings["departments_per_role"]
    users_per_home = settings["users_per_home"]
    if counts["low"] != 1 or not (1 < counts["medium"] < counts["high"] < n):
        raise ValueError("Role sizes must satisfy low=1 < medium < high < departments")
    if any(not isinstance(users_per_home[level], int) or users_per_home[level] < 1
           or users_per_home[level] > n - counts[level] + 1 for level in LEVELS):
        raise ValueError("User counts must be positive and fit the outside departments")
    rng = np.random.default_rng(seed)
    roles, users = [], []
    for home in range(n):
        others = np.array([d for d in range(n) if d != home])
        for level in LEVELS:
            granted = sorted([home, *rng.choice(others, counts[level] - 1, replace=False).tolist()])
            roles.append({"role_id": f"{level}_{home:02d}", "access_level": level,
                          "home_department": home, "department_ids": granted})
    role_map = {role["role_id"]: role for role in roles}
    for home in range(n):
        for level in LEVELS:
            role_id = f"{level}_{home:02d}"
            base = role_map[role_id]["department_ids"]
            outside = rng.permutation([d for d in range(n) if d not in base]).tolist()
            for variant in range(users_per_home[level]):
                extra = outside[:variant]
                role_ids = [role_id, *[f"low_{d:02d}" for d in extra]]
                users.append({"user_id": f"{level}_{home:02d}_{variant:02d}",
                              "access_level": level, "home_department": home,
                              "role_ids": role_ids, "department_ids": sorted([*base, *extra])})
    return roles, users


def build_workload(passage_ids, question_ids, departments, qrels, users):
    """Keep all eligible pairs; categorize only evidence the user may access."""
    passage_row = {pid: i for i, pid in enumerate(passage_ids)}
    relevant = qrels.groupby("question_id")["passage_id"].apply(list).to_dict()
    if set(relevant) != set(question_ids):
        raise ValueError("Question IDs and relevance labels do not match")
    relevant_rows = {qid: np.array([passage_row[pid] for pid in relevant[qid]], dtype=int)
                     for qid in question_ids}
    records = []
    for user in users:
        allowed = np.isin(departments, user["department_ids"])
        n_allowed = int(allowed.sum())
        for qid in question_ids:
            rows = relevant_rows[qid]
            permitted = rows[allowed[rows]]
            if len(permitted) == 0:
                continue
            home = bool(np.any(departments[permitted] == user["home_department"]))
            cross = bool(np.any(departments[permitted] != user["home_department"]))
            category = "mixed" if home and cross else "in_department" if home else "cross_department"
            records.append({"workload_id": len(records), "question_id": qid,
                            "user_id": user["user_id"], "access_level": user["access_level"],
                            "home_department": user["home_department"], "question_type": category,
                            "n_allowed_passages": n_allowed,
                            "access_fraction": n_allowed / len(passage_ids),
                            "n_relevant_total": len(rows), "n_relevant_permitted": len(permitted)})
    workload = pd.DataFrame(records)
    if workload.empty or set(workload["question_id"]) != set(question_ids):
        raise ValueError("Workload does not cover every test question")
    return workload


def validate_organization(departments, roles, users, workload, n_passages, n_departments):
    if len(departments) != n_passages or set(departments.tolist()) != set(range(n_departments)):
        raise ValueError("Invalid or empty departments")
    role_map = {r["role_id"]: r for r in roles}
    if len(role_map) != len(roles) or len({u["user_id"] for u in users}) != len(users):
        raise ValueError("Duplicate role or user IDs")
    for user in users:
        union = sorted({d for rid in user["role_ids"] for d in role_map[rid]["department_ids"]})
        if union != user["department_ids"] or user["home_department"] not in union:
            raise ValueError("User permissions do not match the union of their roles")
    if (workload.duplicated(["question_id", "user_id"]).any()
            or (workload["n_relevant_permitted"] < 1).any()
            or ((workload["access_level"] == "low")
                & (workload["question_type"] != "in_department")).any()):
        raise ValueError("Invalid eligibility, duplicate pairs, or low-access categories")


def summary(cfg, departments, roles, users, workload, elapsed):
    counts = cfg["organization"]["departments_per_role"]
    lines = ["# Synthetic organization", "",
             "Deterministic k-means over the saved normalized passage embeddings. "
             "Departments are synthetic content clusters, not real company departments.", "",
             f"Seed: {cfg['seed']}. Departments: {cfg['organization']['n_departments']}. "
             f"Roles: {len(roles)}. Users: {len(users)}. Clustering: {elapsed:.1f} seconds.", "",
             f"Roles overlap. Low roles cover {counts['low']} department; medium roles cover "
             f"{counts['medium']}; high roles cover {counts['high']}. Additional users per home/level "
             "add outside singleton roles, one extra department per user variant. "
             "Permissions are the union of roles.", "",
             "All eligible question-user pairs are retained; each has at least one permitted relevant "
             "passage. Counts are not balanced across groups; compare group means separately. "
             "Repeated questions/users are correlated observations, not independent samples.", "",
             "Question type uses permitted relevant evidence: in_department = home only; "
             "cross_department = other accessible departments only; mixed = both. "
             "Mixed pairs remain separate. Low-access users cannot have cross-department questions.", "",
             "| Access level | Users | Pairs | Distinct questions | Mean corpus access |", "|---|---:|---:|---:|---:|"]
    for level in LEVELS:
        group = workload[workload.access_level == level]
        level_users = [u for u in users if u["access_level"] == level]
        fractions = [np.isin(departments, u["department_ids"]).mean() for u in level_users]
        lines.append(f"| {level} | {len(level_users)} | {len(group)} | "
                     f"{group.question_id.nunique()} | {np.mean(fractions):.1%} |")
    lines += ["", "| Access level | Question type | Pairs |", "|---|---|---:|"]
    for (level, category), group in workload.groupby(["access_level", "question_type"], sort=True):
        lines.append(f"| {level} | {category} | {len(group)} |")
    lines += ["", "| Department | Passages |", "|---|---:|"]
    for department, size in enumerate(np.bincount(departments)):
        lines.append(f"| {department} | {size} |")
    lines += ["", "Artifacts: data/organization/<model>/<split>/. Run `python -m src.reference` next.", ""]
    return "\n".join(lines)


def load_organization(cfg):
    folder = organization_dir(cfg)
    if not (folder / "meta.json").exists():
        raise ValueError("No completed organization. Run python -m src.organization")
    meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    if meta["inputs"] != input_signature(cfg):
        raise ValueError("Organization inputs changed. Rebuild with python -m src.organization --force")
    for name, expected in meta["artifact_hashes"].items():
        if not (folder / name).exists() or file_hash(folder / name) != expected:
            raise ValueError(f"Organization artifact changed or missing: {name}. Rebuild it.")
    return {
        "meta": meta,
        "departments": pd.read_csv(folder / "passage_departments.csv", dtype={"passage_id": str}),
        "roles": json.loads((folder / "roles.json").read_text(encoding="utf-8")),
        "users": json.loads((folder / "users.json").read_text(encoding="utf-8")),
        "workload": pd.read_csv(folder / "workload.csv", dtype={"question_id": str, "user_id": str}),
    }


def run(cfg, force=False):
    folder = organization_dir(cfg)
    if (folder / "meta.json").exists() and not force:
        raise ValueError("Organization already exists; use --force to replace it")
    emb, qrels = validated_inputs(cfg)
    signature = input_signature(cfg)
    settings = cfg["organization"]
    roles, users = make_roles_users(settings, cfg["seed"])
    print(f"Clustering {len(emb['passage_ids'])} passages into {settings['n_departments']} departments...", flush=True)
    start = time.perf_counter()
    with threadpool_limits(limits=settings["n_threads"]):
        model = KMeans(n_clusters=settings["n_departments"], random_state=cfg["seed"],
                       n_init=settings["kmeans_n_init"], max_iter=settings["kmeans_max_iter"],
                       tol=settings["kmeans_tol"], algorithm="lloyd").fit(emb["passage_vectors"])
    elapsed = time.perf_counter() - start
    departments = model.labels_
    workload = build_workload(emb["passage_ids"], emb["question_ids"], departments, qrels, users)
    validate_organization(departments, roles, users, workload, len(emb["passage_ids"]), settings["n_departments"])
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "meta.json").unlink(missing_ok=True)
    pd.DataFrame({"passage_id": emb["passage_ids"], "department_id": departments}).to_csv(folder / "passage_departments.csv", index=False)
    np.save(folder / "department_centers.npy", model.cluster_centers_)
    write_json(folder / "roles.json", roles)
    write_json(folder / "users.json", users)
    workload.to_csv(folder / "workload.csv", index=False)
    names = ("passage_departments.csv", "department_centers.npy", "roles.json", "users.json", "workload.csv")
    meta = {"inputs": signature, "artifact_hashes": {name: file_hash(folder / name) for name in names},
            "n_pairs": len(workload), "n_questions": len(emb["question_ids"]),
            "clustering_seconds": elapsed, "kmeans_iterations": int(model.n_iter_),
            "kmeans_inertia": float(model.inertia_), "sklearn_version": __import__("sklearn").__version__}
    cfg["paths"]["results_dir"].mkdir(parents=True, exist_ok=True)
    report = cfg["paths"]["results_dir"] / "organization_summary.md"
    report.write_text(summary(cfg, departments, roles, users, workload, elapsed), encoding="utf-8")
    write_json(folder / "meta.json", meta)
    print(f"Saved {len(roles)} roles, {len(users)} users, {len(workload)} eligible pairs ({elapsed:.1f}s clustering).")
    print(f"Wrote {report}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    cfg = load_config()
    set_seed(cfg["seed"])
    run(cfg, args.force)


if __name__ == "__main__":
    main()
