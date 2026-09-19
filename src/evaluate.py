import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def identity_from_filename(name):
    """Extract identity from filenames such as 1420_c5s3_052165_01.jpg."""
    return Path(name).name.split("_")[0]


def main(args):
    database = pd.read_csv(args.database)
    embedding_cols = [c for c in database.columns if c.startswith("embedding_")]

    embeddings = database[embedding_cols].to_numpy(dtype=np.float32)
    names = database["image"].tolist()
    identities = np.array([identity_from_filename(n) for n in names])

    # Self-match is excluded by starting the ranked list from index 1.
    hits = {k: 0 for k in args.top_k}

    for i in range(len(database)):
        distances = np.linalg.norm(embeddings - embeddings[i], axis=1)
        order = np.argsort(distances)
        order = order[order != i]

        for k in args.top_k:
            top_ids = identities[order[:k]]
            if identities[i] in top_ids:
                hits[k] += 1

    total = len(database)

    print(f"Queries evaluated: {total}")
    for k in args.top_k:
        print(f"Recall@{k}: {hits[k] / total:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", required=True)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--top-k", nargs="+", type=int, default=[1, 5, 10])
    args = parser.parse_args()
    main(args)
