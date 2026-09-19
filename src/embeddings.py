import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from .model import ReIDEmbeddingModel


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = ReIDEmbeddingModel(
        embedding_size=checkpoint.get("embedding_size", 512),
        pretrained=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def image_to_tensor(path):
    image = Image.open(path).convert("RGB")
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


def generate_database(model, image_names, image_dir, device):
    rows = []

    with torch.no_grad():
        for name in tqdm(image_names):
            tensor = image_to_tensor(Path(image_dir) / name)
            embedding = model(tensor.unsqueeze(0).to(device))
            embedding = embedding.squeeze(0).cpu().numpy()

            rows.append([name, *embedding.tolist()])

    columns = ["image"] + [f"embedding_{i}" for i in range(len(rows[0]) - 1)]
    return pd.DataFrame(rows, columns=columns)


def main(args):
    device = torch.device(
        args.device if args.device else
        ("cuda" if torch.cuda.is_available() else "cpu")
    )

    model = load_model(args.checkpoint, device)

    triplets = pd.read_csv(args.triplet_csv)
    image_names = triplets["Anchor"].drop_duplicates().tolist()

    database = generate_database(
        model, image_names, args.image_dir, device
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    database.to_csv(output, index=False)

    print(f"Saved {len(database)} embeddings to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--triplet-csv", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--output", default="outputs/results/database.csv")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    main(args)
