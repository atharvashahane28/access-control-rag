"""Step 0 check: run with  python -m src.hello  from the repo root.

It confirms that the libraries import, the config loads, and FAISS can
build and search a small HNSW index.
"""
import sys

import faiss
import matplotlib
import numpy as np
import pandas as pd
import sentence_transformers
import sklearn
import yaml

from src.config import load_config, set_seed


def main():
    cfg = load_config()
    set_seed(cfg["seed"])

    print("Python               ", sys.version.split()[0])
    print("faiss                ", faiss.__version__)
    print("numpy                ", np.__version__)
    print("pandas               ", pd.__version__)
    print("scikit-learn         ", sklearn.__version__)
    print("sentence-transformers", sentence_transformers.__version__)
    print("matplotlib           ", matplotlib.__version__)
    print("pyyaml               ", yaml.__version__)
    print()
    print("Config:", cfg)
    print()

    # Tiny FAISS test: 1000 random unit vectors, each query is one of them, so the nearest neighbour of query i should be vector i.
    dim, n = 64, 1000
    vectors = np.random.rand(n, dim).astype("float32")
    faiss.normalize_L2(vectors)
    index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
    index.add(vectors)
    _, ids = index.search(vectors[:10], cfg["retrieval"]["k"])
    hits = int((ids[:, 0] == np.arange(10)).sum())
    print(f"FAISS HNSW test: {hits}/10 queries found themselves first")
    print("First row of results:", ids[0].tolist())

    if hits < 9:
        sys.exit("FAISS install error.")
    print("\nStep 0 check passed.")


if __name__ == "__main__":
    main()