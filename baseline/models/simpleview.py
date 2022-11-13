import torch
from torch import nn
from typing import Union
from copy import deepcopy
import torchvision
from .resnet18_4 import resnet18_4

models = {
    'resnet18': torchvision.models.resnet18(weights='ResNet18_Weights.DEFAULT'),
    'resnet18_4': resnet18_4(),
}

class SimpleView(nn.Module):
    def __init__(
        self,
        num_views: int,
        num_classes: int,
        architecture: str = 'resnet18_4',
    ):
        super().__init__()
        backbone = models[architecture]
        self.backbone = deepcopy(backbone)
        
        z_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
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
