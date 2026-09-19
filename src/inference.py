import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from .embeddings import image_to_tensor, load_model


def main(args):
    device = torch.device(
        args.device if args.device else
        ("cuda" if torch.cuda.is_available() else "cpu")
    )

    model = load_model(args.checkpoint, device)

    database = pd.read_csv(args.database)
    embedding_cols = [c for c in database.columns if c.startswith("embedding_")]
    database_embeddings = database[embedding_cols].to_numpy(dtype=np.float32)

    query = image_to_tensor(Path(args.image))
    with torch.no_grad():
        query_embedding = (
            model(query.unsqueeze(0).to(device))
            .squeeze(0)
            .cpu()
            .numpy()
        )

    distances = np.linalg.norm(
        database_embeddings - query_embedding[None, :], axis=1
    )

    top_indices = np.argsort(distances)[:args.top_k]

    print("\nTop matches:")
    for rank, idx in enumerate(top_indices, start=1):
        print(
            f"{rank:02d}. {database.iloc[idx]['image']} "
            f"| distance={distances[idx]:.4f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--database", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    main(args)
