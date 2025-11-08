import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba
from torch.nn import Linear


class MambaFeature(nn.Module):
    def __init__(self, n_bands, patch_size, encoder_embed_dim=144, emb_dropout=0., out_dim=256):
        super(MambaFeature, self).__init__()

        self.input_bands = n_bands
        self.out_dim = out_dim
        self.patch_size = patch_size

        self.conv1_1 = nn.Sequential(
            nn.Conv2d(n_bands, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),  # No effect on order
        )
        self.conv1_2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),  # No effect on order
        )
        self.conv2 = nn.Conv2d(64, 64, 1)
        self.conv3 = nn.Conv2d(64, 32, 1)
        self.encoder_embedding = nn.Linear((patch_size * 1) ** 2, self.patch_size ** 2)
        self.encoder_pos_embed = nn.Parameter(torch.randn(1, self.patch_size ** 2 + 1 + 2, encoder_embed_dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.mamba = Mamba(encoder_embed_dim)
        self.relu = nn.ReLU(4608)
        self.bn = nn.BatchNorm2d(32)

    def forward(self, x):
        x = self.conv1_1(x)
        x = self.conv1_2(x)

        x = x.flatten(2)
        x = self.encoder_embedding(x)
        # x = torch.einsum('nld->ndl', x)
        # x += self.encoder_pos_embed[:, :1]
        # x = self.dropout(x)
        # x = torch.einsum('nld->ndl', x)

        b, c, h = x.shape
        x = self.conv2(x.reshape(b, c, 12, 12)).reshape(b, c, h)
        x = self.mamba(x) + x
        x = self.relu(self.bn(self.conv3(x.reshape(b, c, 12, 12)))).reshape(b, 4608, 1, 1)

        return x