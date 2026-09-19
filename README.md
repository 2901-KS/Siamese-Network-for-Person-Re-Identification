# Siamese Network for Person Re-Identification

A PyTorch-based person re-identification project that learns discriminative visual embeddings from anchor-positive-negative image triplets and retrieves visually similar people using embedding distance.

## Project Overview

The system follows the pipeline:

**Person Images → Triplet Construction → EfficientNet-B0 Embedding Network → Triplet Loss Training → Embedding Database → Distance-Based Retrieval**

The project was originally developed as a Google Colab implementation and has been reorganized into a reusable project structure. GPU-intensive model training is performed in Google Colab, while trained-model inference and retrieval evaluation can also be performed locally.

## Key Features

- Anchor / Positive / Negative triplet dataset
- PyTorch `Dataset` and `DataLoader`
- EfficientNet-B0 backbone through `timm`
- 512-dimensional learned image embeddings
- PyTorch `TripletMarginLoss`
- Train/validation split
- Best-checkpoint selection based on validation loss
- Embedding database generation
- Euclidean-distance image retrieval
- Recall@K evaluation
- Google Colab GPU training workflow
- CPU-compatible inference using a trained checkpoint

## Dataset

The project uses person images organized under a `train/` directory and a triplet CSV file containing anchor, positive, and negative image relationships.

The expected dataset structure is:

```text
Person-Re-Id-Dataset/
├── train.csv
└── train/
    ├── image_1.jpg
    ├── image_2.jpg
    ├── image_3.jpg
    └── ...
