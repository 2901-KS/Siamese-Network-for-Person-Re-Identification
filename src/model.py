import torch.nn as nn
import timm


class ReIDEmbeddingModel(nn.Module):
    """EfficientNet-B0 backbone with a configurable embedding head."""

    def __init__(self, embedding_size=512, pretrained=True):
        super().__init__()

        self.backbone = timm.create_model(
            "efficientnet_b0",
            pretrained=pretrained,
            num_classes=embedding_size,
        )

    def forward(self, images):
        return self.backbone(images)
