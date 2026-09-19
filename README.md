
# Siamese Network for Person Re-Identification

A PyTorch-based person re-identification project that learns discriminative visual embeddings from anchor-positive-negative image triplets and retrieves visually similar people using embedding distance.

## Project Overview

The system follows the pipeline:

**Person Images → Triplet Construction → EfficientNet-B0 Embedding Network → Triplet Loss Training → Embedding Database → Distance-Based Retrieval**

The project was originally developed as a Google Colab implementation and has been reorganized into a reusable project structure. GPU-intensive training is performed in Google Colab, while trained-model inference and retrieval evaluation can also be performed locally.

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
````

The `train.csv` file contains triplets in the following format:

```text
Anchor,Positive,Negative
anchor_image.jpg,positive_image.jpg,negative_image.jpg
...
```

The experiment used a 4,000-row triplet dataset, split into:

```text
Training triplets:   3,200
Validation triplets:   800
```

Dataset files are intentionally not included in this repository. They can be downloaded separately or mounted through Google Drive when using Google Colab.

## Model Architecture

The embedding model uses an ImageNet-pretrained EfficientNet-B0 backbone with the classification head replaced by a 512-dimensional embedding projection.

```text
Input Image
     |
     v
EfficientNet-B0
     |
     v
512-D Embedding
```

For every triplet `(Anchor, Positive, Negative)`, the model is trained so that the embedding distance between the anchor and positive image becomes smaller than the distance between the anchor and negative image.

The training objective is implemented using PyTorch `TripletMarginLoss`.

```text
distance(Anchor, Positive)
        <
distance(Anchor, Negative)
```

## Training

### Google Colab

GPU training is recommended because the model uses a pretrained CNN backbone.

1. Open `notebooks/01_train_colab.ipynb` in Google Colab.
2. Select a GPU runtime.
3. Clone or open the repository.
4. Install the required dependencies.
5. Download or mount the dataset.
6. Set the `DATA_ROOT` path.
7. Run the training command.

Example:

```bash
python -m src.train \
    --data-root "$DATA_ROOT" \
    --device cuda \
    --seed 42
```

The best-performing checkpoint is saved as:

```text
checkpoints/best_model.pt
```

The repository does not require a local GPU for inference.

## Training Configuration

The reported experiment used:

```yaml
batch_size: 32
learning_rate: 0.001
epochs: 15
embedding_size: 512
margin: 1.0
backbone: efficientnet_b0
```

Training was performed for 15 epochs using an NVIDIA Tesla T4 GPU in Google Colab.

The best validation loss was:

```text
Best validation loss: 0.14389
Best epoch: 14
```

## Experimental Results

The trained model generated embeddings for 3,285 images and was evaluated using distance-based retrieval.

| Metric               | Result |
| -------------------- | -----: |
| Embeddings generated |  3,285 |
| Queries evaluated    |  3,285 |
| Recall@1             | 40.88% |
| Recall@5             | 67.70% |
| Recall@10            | 76.53% |

The resulting embedding database is saved as:

```text
outputs/results/database.csv
```

## Embedding Generation

After training, generate the embedding database using:

```bash
python -m src.embeddings \
    --checkpoint checkpoints/best_model.pt \
    --triplet-csv "$DATA_ROOT/train.csv" \
    --image-dir "$DATA_ROOT/train" \
    --output outputs/results/database.csv \
    --device cuda
```

The command generates image embeddings and stores the resulting embedding database for retrieval.

## Retrieval Evaluation

Run the retrieval evaluation using:

```bash
python -m src.evaluate \
    --database outputs/results/database.csv \
    --top-k 1 5 10
```

The evaluation reports Recall@1, Recall@5, and Recall@10.

If image filenames encode identity in the same format as the dataset, for example:

```text
1420_c5s3_052165_01.jpg
```

the identity can be extracted as:

```text
1420
```

This identity information is used by the retrieval evaluation pipeline.

## Local Inference

Once a trained checkpoint and embedding database are available, inference can be performed locally using:

```bash
python -m src.inference \
    --checkpoint checkpoints/best_model.pt \
    --image path/to/query.jpg \
    --database outputs/results/database.csv \
    --data-dir path/to/dataset/train \
    --top-k 5
```

The inference pipeline loads the trained embedding model, generates an embedding for the query image, compares it with the stored embedding database, and returns the most similar images based on embedding distance.

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

Dataset files, trained checkpoints, and generated outputs are excluded from version control where appropriate.

## Reproducibility

The training pipeline supports reproducible experiment setup through a random seed.

Example:

```bash
python -m src.train \
    --data-root "$DATA_ROOT" \
    --device cuda \
    --seed 42
```

The reported experiment used:

```text
Random seed: 42
Training triplets: 3,200
Validation triplets: 800
Epochs: 15
Batch size: 32
Learning rate: 0.001
Embedding dimension: 512
Triplet margin: 1.0
Backbone: EfficientNet-B0
Hardware: NVIDIA Tesla T4
```

## Future Improvements

* Hard-negative mining
* Data augmentation
* Learning-rate scheduling
* Embedding normalization
* Larger-scale identity-level evaluation
* Mean Average Precision (mAP)
* Cumulative Match Characteristic (CMC) curves
* Approximate nearest-neighbor retrieval
* Experiment tracking with MLflow
* EfficientNet backbone fine-tuning strategies
* More comprehensive person re-identification benchmarks

## Status

**Research / Portfolio Project**

The core methodology is based on the original Google Colab implementation and has been reorganized into a reusable project structure with separate modules for dataset handling, model training, embedding generation, inference, and retrieval evaluation.

The current implementation has been trained and evaluated on the configured person re-identification triplet dataset, with Recall@1, Recall@5, and Recall@10 reported above.

```
```
