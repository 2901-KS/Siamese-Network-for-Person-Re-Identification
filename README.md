# Siamese Network for Person Re-Identification

A PyTorch-based person re-identification project that learns discriminative visual embeddings from anchor-positive-negative image triplets and retrieves visually similar people using embedding distance.

## Project Overview

The system follows this pipeline:

**Person images → Triplet construction → EfficientNet-B0 embedding network → Triplet Loss training → Embedding database → Distance-based retrieval**

The original implementation was developed in Google Colab. This repository separates the reusable implementation from the notebook and keeps GPU-heavy training in Colab while allowing lightweight inference/evaluation to be run locally.

## Key Features

- Anchor / Positive / Negative triplet dataset
- PyTorch `Dataset` and `DataLoader`
- EfficientNet-B0 backbone through `timm`
- 512-dimensional learned image embeddings
- Triplet Margin Loss
- Train/validation split
- Best-checkpoint selection using validation loss
- Embedding database generation
- Euclidean-distance image retrieval
- Optional Recall@K evaluation
- Colab training workflow for GPU acceleration
- Local CPU-compatible inference

## Dataset Format

The current notebook expects the dataset in this structure:

```text
dataset/
├── train/
│   ├── image_1.jpg
│   ├── image_2.jpg
│   └── ...
└── train.csv
```

`train.csv` should contain:

```text
Anchor,Negative,Positive
anchor_image.jpg,negative_image.jpg,positive_image.jpg
...
```

The uploaded notebook used a 4,000-row triplet CSV and an image directory under `train/`.

> Dataset files are intentionally not included in this repository. Add them locally or mount them from Google Drive in Colab.

## Model

The embedding model uses an ImageNet-pretrained EfficientNet-B0 backbone and replaces its classifier with a 512-dimensional projection:

```text
Input Image
    ↓
EfficientNet-B0
    ↓
512-D embedding
```

For each triplet `(Anchor, Positive, Negative)`, the model learns:

```text
distance(Anchor, Positive) < distance(Anchor, Negative)
```

using PyTorch `TripletMarginLoss`.

## Training

### Recommended: Google Colab

GPU training is recommended because the backbone is a pretrained CNN and the notebook trains the model end-to-end.

1. Open `notebooks/01_train_colab.ipynb` in Google Colab.
2. Enable a GPU runtime.
3. Mount Google Drive or upload the dataset.
4. Set the dataset path in the notebook.
5. Run the training cells.
6. The best checkpoint is saved as `best_model.pt`.
7. Embeddings can then be generated for retrieval.

The repository itself does **not** require a local GPU.

### Local inference

Once a trained checkpoint exists, CPU inference is supported:

```bash
python -m src.inference \
    --checkpoint checkpoints/best_model.pt \
    --image path/to/query.jpg \
    --database outputs/results/database.csv \
    --data-dir path/to/dataset/train \
    --top-k 5
```

## Training Configuration

Default values follow the original Colab implementation:

```yaml
batch_size: 32
learning_rate: 0.001
epochs: 15
embedding_size: 512
margin: 1.0
backbone: efficientnet_b0
```

These are configuration defaults, not reported optimized hyperparameters.

## Evaluation

The repository includes optional retrieval evaluation.

If image filenames encode identity in the same style as the original dataset, for example:

```text
1420_c5s3_052165_01.jpg
```

the identity can be extracted as:

```text
1420
```

The evaluation script can report Recall@K for a set of query images.

Run:

```bash
python -m src.evaluate \
    --database outputs/results/database.csv \
    --data-dir path/to/dataset/train \
    --top-k 1 5 10
```

Do not report evaluation numbers in a resume or README until the experiment has actually been run and recorded.

## Repository Structure

```text
siamese-person-reid/
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
├── configs/
│   └── config.yaml
├── src/
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── embeddings.py
│   ├── inference.py
│   └── evaluate.py
├── notebooks/
│   ├── 01_train_colab.ipynb
│   └── original_colab_notebook.ipynb
├── scripts/
│   └── prepare_dirs.py
├── outputs/
│   ├── figures/
│   └── results/
└── checkpoints/
```

## Reproducibility

Set the random seed before experiments:

```bash
python -m src.train --seed 42
```

For GPU training, use Google Colab with a GPU runtime. For inference and lightweight evaluation, CPU execution is supported.

## Future Improvements

- Hard-negative mining
- Data augmentation
- Learning-rate scheduling
- Embedding normalization
- Larger-scale identity-level evaluation
- mAP and CMC evaluation
- Approximate nearest-neighbor retrieval
- Experiment tracking with MLflow
- Fine-tuning strategies for the EfficientNet backbone

## Status

**Research / portfolio project**

The core methodology is based on the original Google Colab implementation and has been reorganized into a reproducible project structure.
