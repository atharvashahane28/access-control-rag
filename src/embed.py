"""Step 2: turn every passage and every test question into a vector.

Run from the repo root:

    python -m src.embed --test     quick timing test on a sample, saves nothing
    python -m src.embed            full run, saves the vectors
    python -m src.embed --force    full run, replacing vectors saved earlier

The vectors go to data/embeddings/<model name>/ (not in git):

    passage_vectors.npy   one row per passage
    passage_ids.json      passage IDs, in the same order as the rows
    question_vectors.npy  one row per question
    question_ids.json     question IDs, in the same order as the rows
    meta.json             model name, vector size and other facts about the run

Every vector has length 1, so the inner product of two vectors is their
cosine similarity.

In later steps:

    from src.embed import load_embeddings
    emb = load_embeddings(cfg)
    emb["passage_ids"], emb["passage_vectors"]
    emb["question_ids"], emb["question_vectors"]
"""
import argparse
import json
import time

import numpy as np

from src.config import load_config, set_seed
from src.data import load_corpus, load_queries


def embeddings_dir(cfg):
    """Folder for this model's vectors, for example data/embeddings/all-mpnet-base-v2."""
    name = cfg["embedding"]["model"].split("/")[-1]
    return cfg["paths"]["data_dir"] / "embeddings" / name


def load_embeddings(cfg):
    """Load the saved vectors and IDs for the model named in the config."""
    folder = embeddings_dir(cfg)
    if not (folder / "meta.json").exists():
        raise SystemExit(
            f"No finished embeddings in {folder}. Run:  python -m src.embed"
        )

    def read_json(name):
        return json.loads((folder / name).read_text(encoding="utf-8"))

    return {
        "passage_ids": read_json("passage_ids.json"),
        "passage_vectors": np.load(folder / "passage_vectors.npy"),
        "question_ids": read_json("question_ids.json"),
        "question_vectors": np.load(folder / "question_vectors.npy"),
        "meta": read_json("meta.json"),
    }


def _load_model(cfg):
    # Imported here so that load_embeddings() works without loading torch.
    from sentence_transformers import SentenceTransformer

    name = cfg["embedding"]["model"]
    print(f"Loading model {name} (downloaded on first use)...")
    model = SentenceTransformer(name)
    print(f"Running on: {model.device}")
    print(f"Reads at most {model.max_seq_length} tokens per text")
    return model


def _encode(model, texts, cfg):
    """Embed a list of texts. Returns float32 vectors of length 1."""
    vectors = model.encode(
        texts,
        batch_size=cfg["embedding"]["batch_size"],
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return np.ascontiguousarray(vectors, dtype="float32")


def _share_cut_off(model, texts):
    """Fraction of texts longer than the model can read (their ending is lost).

    Returns None if the tokens cannot be counted, so that this extra
    measurement can never stop the run.
    """
    try:
        limit = model.max_seq_length
        too_long = 0
        for start in range(0, len(texts), 1000):
            batch = texts[start:start + 1000]
            token_ids = model.tokenizer(batch, truncation=False, verbose=False)["input_ids"]
            too_long += sum(len(ids) > limit for ids in token_ids)
        return too_long / len(texts)
    except Exception as error:  # noqa: BLE001
        print(f"Could not count tokens ({error}). Continuing without it.")
        return None


def _minutes(seconds):
    return f"{seconds / 60:.1f} min"


def run_test(cfg):
    """Embed a random sample of passages and estimate the time of a full run."""
    corpus = load_corpus(cfg)
    n = min(cfg["embedding"]["test_n"], len(corpus))
    sample = corpus.sample(n=n, random_state=cfg["seed"])["text"].tolist()

    model = _load_model(cfg)
    cut = _share_cut_off(model, sample)

    start = time.perf_counter()
    vectors = _encode(model, sample, cfg)
    seconds = time.perf_counter() - start

    print()
    print(f"Embedded {n} passages in {seconds:.1f} s ({n / seconds:.1f} passages per second)")
    print(f"Vector size: {vectors.shape[1]}")
    if cut is not None:
        print(f"Passages in the sample that get cut off: {cut:.1%}")
    print(f"Estimated time for all {len(corpus)} passages: {_minutes(seconds * len(corpus) / n)}")
    print("Nothing was saved. For the full run:  python -m src.embed")


def run_full(cfg, force):
    """Embed all passages and questions and save them."""
    folder = embeddings_dir(cfg)
    if (folder / "meta.json").exists() and not force:
        raise SystemExit(
            f"Embeddings already exist in {folder}. "
            "To replace them, run:  python -m src.embed --force"
        )
    folder.mkdir(parents=True, exist_ok=True)
    # meta.json marks a finished run, so remove an old one before starting.
    (folder / "meta.json").unlink(missing_ok=True)

    corpus = load_corpus(cfg)
    queries = load_queries(cfg)
    passages = corpus["text"].tolist()
    prefix = cfg["embedding"]["query_prefix"] or ""
    questions = [prefix + text for text in queries["text"]]

    model = _load_model(cfg)

    print("Counting how many passages are too long for the model...")
    cut = _share_cut_off(model, passages)
    if cut is not None:
        print(f"Passages that get cut off: {cut:.1%}")

    print(f"Embedding {len(passages)} passages...")
    start = time.perf_counter()
    passage_vectors = _encode(model, passages, cfg)
    passage_seconds = time.perf_counter() - start
    np.save(folder / "passage_vectors.npy", passage_vectors)
    (folder / "passage_ids.json").write_text(
        json.dumps(corpus["passage_id"].tolist()), encoding="utf-8"
    )
    print(f"Passages done in {_minutes(passage_seconds)}")

    print(f"Embedding {len(questions)} questions...")
    question_vectors = _encode(model, questions, cfg)
    np.save(folder / "question_vectors.npy", question_vectors)
    (folder / "question_ids.json").write_text(
        json.dumps(queries["question_id"].tolist()), encoding="utf-8"
    )

    meta = {
        "model": cfg["embedding"]["model"],
        "vector_size": int(passage_vectors.shape[1]),
        "max_tokens": int(model.max_seq_length),
        "normalized": True,
        "query_prefix": prefix,
        "n_passages": len(passages),
        "n_questions": len(questions),
        "share_passages_cut_off": cut,
        "passage_minutes": round(passage_seconds / 60, 1),
        "device": str(model.device),
    }
    (folder / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print()
    print(f"Saved passage vectors  {passage_vectors.shape}")
    print(f"Saved question vectors {question_vectors.shape}")
    print(f"Folder: {folder}")
    print("Next:  python -m src.check_embeddings")


def main():
    parser = argparse.ArgumentParser(description="Step 2: create embeddings.")
    parser.add_argument("--test", action="store_true",
                        help="time a small sample and save nothing")
    parser.add_argument("--force", action="store_true",
                        help="replace embeddings saved earlier")
    args = parser.parse_args()

    cfg = load_config()
    set_seed(cfg["seed"])

    if args.test:
        run_test(cfg)
    else:
        run_full(cfg, args.force)


if __name__ == "__main__":
    main()