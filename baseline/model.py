import torch
from torch import nn

import torchvision
resnet18 = torchvision.models.resnet18(weights='ResNet18_Weights.DEFAULT')

class SimpleView(nn.Module):
    def __init__(
        self,
        num_views: int,
        num_classes: int,
        backbone: nn.Module = resnet18,
    ):
        super().__init__()
        z_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()

        self.backbone = backbone
        self.fc = nn.Linear(
            in_features=z_dim * num_views,
            out_features=num_classes
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, v, c, h, w = x.shape
        x = x.reshape(b * v, c, h, w)
        z = self.backbone(x)
        z = z.reshape(b, v, -1)
        z = z.reshape(b, -1)

        return self.fc(z)
