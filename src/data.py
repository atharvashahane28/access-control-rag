"""Download FiQA (BEIR version) and load it into simple tables.

Download once, from the repo root:

    python -m src.data

Then, in any script:

    from src.data import load_corpus, load_queries, load_qrels
"""
import hashlib
import json
import urllib.request
import zipfile

import pandas as pd

from src.config import load_config


def dataset_dir(cfg):
    """Folder that holds the unzipped dataset, for example data/fiqa."""
    return cfg["paths"]["data_dir"] / cfg["dataset"]["name"]


def _md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(cfg):
    """Download and unzip the dataset. Does nothing if it is already there."""
    out_dir = dataset_dir(cfg)
    if (out_dir / "corpus.jsonl").exists():
        print(f"Dataset already present in {out_dir}")
        return out_dir

    data_dir = cfg["paths"]["data_dir"]
    data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = data_dir / (cfg["dataset"]["name"] + ".zip")

    print(f"Downloading {cfg['dataset']['url']}")
    urllib.request.urlretrieve(cfg["dataset"]["url"], zip_path)

    found = _md5(zip_path)
    if found != cfg["dataset"]["md5"]:
        raise SystemExit(
            f"Checksum mismatch for {zip_path}: expected "
            f"{cfg['dataset']['md5']}, got {found}. Delete the file and retry."
        )
    print("Checksum OK. Unzipping...")

    with zipfile.ZipFile(zip_path) as z:
        z.extractall(data_dir)
    print(f"Done. Files are in {out_dir}")
    return out_dir


def _read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_corpus(cfg, clean=True):
    """All passages. Columns: passage_id, text. IDs are strings.

    With clean=True (the default), passages with no text are removed.
    """
    rows = _read_jsonl(dataset_dir(cfg) / "corpus.jsonl")
    df = pd.DataFrame(
        {
            "passage_id": [str(r["_id"]) for r in rows],
            "text": [r["text"] for r in rows],
        }
    )
    if clean:
        df = df[df["text"].str.strip() != ""].reset_index(drop=True)
    return df


def load_qrels(cfg, clean=True):
    """Relevance labels for the split in the config.

    Columns: question_id, passage_id. One row per relevant passage.
    With clean=True (the default), labels that point to a removed (empty)
    passage are removed too.
    """
    path = dataset_dir(cfg) / "qrels" / (cfg["dataset"]["split"] + ".tsv")
    df = pd.read_csv(path, sep="\t", dtype=str)
    df = df[df["score"].astype(int) > 0]
    df = df.rename(columns={"query-id": "question_id", "corpus-id": "passage_id"})
    if clean:
        kept = set(load_corpus(cfg)["passage_id"])
        df = df[df["passage_id"].isin(kept)]
    return df[["question_id", "passage_id"]].reset_index(drop=True)


def load_queries(cfg):
    """Questions of the split in the config. Columns: question_id, text.

    queries.jsonl holds the questions of every split, so we keep only the
    ones that have relevance labels in our split.
    """
    rows = _read_jsonl(dataset_dir(cfg) / "queries.jsonl")
    df = pd.DataFrame(
        {
            "question_id": [str(r["_id"]) for r in rows],
            "text": [r["text"] for r in rows],
        }
    )
    wanted = set(load_qrels(cfg)["question_id"])
    return df[df["question_id"].isin(wanted)].reset_index(drop=True)


if __name__ == "__main__":
    download(load_config())