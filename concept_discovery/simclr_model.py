"""
SimCLR Model - Self-Supervised Contrastive Learning
Core encoder and projection head for concept discovery from unlabeled images.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ProjectionHead(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256, output_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class SimCLREncoder(nn.Module):
    def __init__(self, backbone="resnet18", projection_dim=128, pretrained=False):
        super().__init__()
        if backbone == "resnet18":
            base = models.resnet18(pretrained=pretrained)
            self.feature_dim = 512
        elif backbone == "resnet50":
            base = models.resnet50(pretrained=pretrained)
            self.feature_dim = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        self.encoder = nn.Sequential(*list(base.children())[:-1])
        self.projector = ProjectionHead(self.feature_dim, self.feature_dim // 2, projection_dim)

    def forward(self, x):
        h = self.encoder(x)
        h = torch.flatten(h, 1)
        z = self.projector(h)
        z = F.normalize(z, dim=1)
        return h, z


class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5, device="cpu"):
        super().__init__()
        self.temperature = temperature
        self.device = device
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, z_i, z_j):
        batch_size = z_i.shape[0]
        N = 2 * batch_size
        z = torch.cat([z_i, z_j], dim=0)
        sim = torch.mm(z, z.t()) / self.temperature
        mask = torch.eye(N, dtype=torch.bool, device=self.device)
        sim.masked_fill_(mask, float('-inf'))
        labels = torch.cat([
            torch.arange(batch_size, 2 * batch_size),
            torch.arange(batch_size)
        ]).to(self.device)
        return self.criterion(sim, labels)
