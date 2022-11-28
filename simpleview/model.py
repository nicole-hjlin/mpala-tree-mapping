import torch
from torch import nn
from copy import deepcopy
from .resnet18_4 import resnet18_4

models = {
    'resnet18_4': resnet18_4(expand_projection=True),
    'resnet18_4_spicy': resnet18_4(expand_projection=False),
}

class SimpleView(nn.Module):
    def __init__(
        self,
        num_views: int,
        num_classes: int,
        expand_projections: bool,
    ):
        super().__init__()
        architecutre = 'resnet18_4_spicy' if expand_projections else 'resnet18_4'
        backbone = models[architecutre]
        self.backbone = deepcopy(backbone)

        z_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        self.fc = nn.Linear(
            in_features=z_dim * num_views if expand_projections else z_dim,
            out_features=num_classes
        )

        self.expand_projections = expand_projections

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.expand_projections:
            b, v, c, h, w = x.shape
            x = x.reshape(b * v, c, h, w)
            z = self.backbone(x)
            z = z.reshape(b, -1)
        else:
            z = self.backbone(x)

        return self.fc(z)
