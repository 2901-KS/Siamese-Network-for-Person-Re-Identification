import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import TripletDataset, load_triplets
from .model import ReIDEmbeddingModel


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, criterion, optimizer, device, training):
    model.train(training)
    total_loss = 0.0

    context = torch.enable_grad() if training else torch.no_grad()

    with context:
        for anchor, positive, negative in tqdm(loader, leave=False):
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)

            anchor_emb = model(anchor)
            positive_emb = model(positive)
            negative_emb = model(negative)

            loss = criterion(anchor_emb, positive_emb, negative_emb)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()

    return total_loss / max(len(loader), 1)


def main(args):
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    set_seed(args.seed if args.seed is not None else cfg["training"]["seed"])

    data_root = Path(args.data_root or cfg["data"]["root"])
    image_dir = data_root / cfg["data"]["image_dir"]
    csv_path = data_root / cfg["data"]["triplet_csv"]

    checkpoint_dir = Path(cfg["paths"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_df, valid_df = load_triplets(
        csv_path,
        val_size=cfg["training"]["val_size"],
        seed=cfg["training"]["seed"],
    )

    trainset = TripletDataset(train_df, image_dir)
    validset = TripletDataset(valid_df, image_dir)

    trainloader = DataLoader(
        trainset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["training"]["num_workers"],
        pin_memory=torch.cuda.is_available(),
    )

    validloader = DataLoader(
        validset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["training"]["num_workers"],
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device(
        args.device if args.device else
        ("cuda" if torch.cuda.is_available() else "cpu")
    )

    model = ReIDEmbeddingModel(
        embedding_size=cfg["model"]["embedding_size"],
        pretrained=cfg["model"]["pretrained"],
    ).to(device)

    criterion = nn.TripletMarginLoss(margin=cfg["training"]["margin"])
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
    )

    best_valid_loss = float("inf")

    print(f"Device: {device}")
    print(f"Training triplets: {len(trainset)}")
    print(f"Validation triplets: {len(validset)}")

    history = []

    for epoch in range(cfg["training"]["epochs"]):
        train_loss = run_epoch(
            model, trainloader, criterion, optimizer, device, training=True
        )
        valid_loss = run_epoch(
            model, validloader, criterion, optimizer, device, training=False
        )

        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "valid_loss": valid_loss,
            }
        )

        print(
            f"Epoch {epoch + 1:02d}/{cfg['training']['epochs']} | "
            f"train_loss={train_loss:.5f} | valid_loss={valid_loss:.5f}"
        )

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "embedding_size": cfg["model"]["embedding_size"],
                    "backbone": cfg["model"]["backbone"],
                    "epoch": epoch + 1,
                    "valid_loss": valid_loss,
                },
                checkpoint_dir / "best_model.pt",
            )
            print("Saved best checkpoint.")

    np.savetxt(
        Path(cfg["paths"]["output_dir"]) / "training_history.csv",
        np.array(
            [[h["epoch"], h["train_loss"], h["valid_loss"]] for h in history]
        ),
        delimiter=",",
        header="epoch,train_loss,valid_loss",
        comments="",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    main(args)
