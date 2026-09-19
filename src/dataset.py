from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


class TripletDataset(Dataset):
    """Dataset returning Anchor, Positive and Negative tensors."""

    def __init__(self, dataframe, image_dir):
        self.df = dataframe.reset_index(drop=True)
        self.image_dir = Path(image_dir)

    def __len__(self):
        return len(self.df)

    @staticmethod
    def _load_image(path):
        image = Image.open(path).convert("RGB")
        array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        return tensor

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        anchor = self._load_image(self.image_dir / row["Anchor"])
        positive = self._load_image(self.image_dir / row["Positive"])
        negative = self._load_image(self.image_dir / row["Negative"])

        return anchor, positive, negative


def load_triplets(csv_path, val_size=0.20, seed=42):
    df = pd.read_csv(csv_path)
    required = {"Anchor", "Positive", "Negative"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    train_df, valid_df = train_test_split(
        df, test_size=val_size, random_state=seed
    )
    return train_df.reset_index(drop=True), valid_df.reset_index(drop=True)
