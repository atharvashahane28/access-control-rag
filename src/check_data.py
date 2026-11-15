"""Step 1 check: run with  python -m src.check_data  from the repo root.

Prints basic facts about the dataset and writes two small files:

    results/data_stats.md     the numbers
    results/data_samples.md   question and passage pairs to read by eye
"""
from src.config import load_config, set_seed
from src.data import load_corpus, load_qrels, load_queries


def main():
    cfg = load_config()
    set_seed(cfg["seed"])

    corpus = load_corpus(cfg)
    queries = load_queries(cfg)
    qrels = load_qrels(cfg)

    words = corpus["text"].str.split().str.len()
    per_question = qrels.groupby("question_id").size()
    passage_ids = set(corpus["passage_id"])
    raw_corpus = load_corpus(cfg, clean=False)
    raw_qrels = load_qrels(cfg, clean=False)

    stats = [
        ("Dataset", cfg["dataset"]["name"]),
        ("Split", cfg["dataset"]["split"]),
        ("Passages", len(corpus)),
        ("Duplicate passage IDs", int(corpus["passage_id"].duplicated().sum())),
        ("Empty passages removed", len(raw_corpus) - len(corpus)),
        ("Questions", len(queries)),
        ("Relevance labels", len(qrels)),
        ("Labels removed (pointed to an empty passage)", len(raw_qrels) - len(qrels)),
        ("Questions removed (no label left)",
         raw_qrels["question_id"].nunique() - len(queries)),
        ("Questions with no label", len(queries) - per_question.size),
        ("Labels pointing to a missing passage",
         int((~qrels["passage_id"].isin(passage_ids)).sum())),
        ("Relevant passages per question, mean", round(per_question.mean(), 2)),
        ("Relevant passages per question, median", int(per_question.median())),
        ("Relevant passages per question, min", int(per_question.min())),
        ("Relevant passages per question, max", int(per_question.max())),
        ("Questions with exactly 1 relevant passage", int((per_question == 1).sum())),
        ("Passage length in words, mean", round(words.mean(), 1)),
        ("Passage length in words, median", int(words.median())),
        ("Passage length in words, 95th percentile", int(words.quantile(0.95))),
        ("Passage length in words, max", int(words.max())),
        ("Question length in words, mean",
         round(queries["text"].str.split().str.len().mean(), 1)),
    ]

    lines = ["# Data statistics", "", "| Item | Value |", "|---|---|"]
    lines += [f"| {name} | {value} |" for name, value in stats]
    for name, value in stats:
        print(f"{name:45s} {value}")

    results_dir = cfg["paths"]["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "data_stats.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Question and passage pairs to read by eye.
    n = min(cfg["data_check"]["n_samples"], len(queries))
    max_chars = cfg["data_check"]["max_chars"]
    passage_text = corpus.set_index("passage_id")["text"]
    sample = queries.sample(n=n, random_state=cfg["seed"])

    out = ["# Sample question and passage pairs", ""]
    for i, q in enumerate(sample.itertuples(), start=1):
        relevant = qrels.loc[qrels["question_id"] == q.question_id, "passage_id"]
        pid = relevant.iloc[0]
        text = passage_text[pid]
        if len(text) > max_chars:
            text = text[:max_chars] + " [...]"
        out += [
            f"## {i}. Question {q.question_id} ({len(relevant)} relevant passages)",
            "",
            f"**Question:** {q.text}",
            "",
            f"**Passage {pid}:** {text}",
            "",
        ]
    (results_dir / "data_samples.md").write_text("\n".join(out), encoding="utf-8")

    print(f"\nWrote {results_dir / 'data_stats.md'}")
    print(f"Wrote {results_dir / 'data_samples.md'}")


if __name__ == "__main__":
    main()