from math import ceil
import os
import torch
from torch import nn
import torch.nn.functional as F
from torchsummary import summary

from net.mtcnn_utils import *


class PNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 10, 3),
            nn.PReLU(10),

            nn.MaxPool2d(2, (2, 2)),

            nn.Conv2d(10, 16, 3),
            nn.PReLU(16),

            nn.Conv2d(16, 32, 3),
            nn.PReLU(32)
        )
        self.face_cls = nn.Conv2d(32, 2, 1)
        self.bbox_reg = nn.Conv2d(32, 4, 1)

    def forward(self, x):
        x = self.features(x)

        face_cls = self.face_cls(x)
        bbox_reg = self.bbox_reg(x)

        return face_cls, bbox_reg


class RNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 28, 3),
            nn.PReLU(28),

            nn.MaxPool2d(3, 2, ceil_mode=True),

            nn.Conv2d(28, 48, 3),
            nn.PReLU(48),

            nn.MaxPool2d(3, 2, ceil_mode=True),

            nn.Conv2d(48, 64, 2),
            nn.PReLU(64),
        )
        self.flat = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(576, 128)
        )
        self.face_cls = nn.Linear(128, 2)
        self.bbox_reg = nn.Linear(128, 4)

    def forward(self, x):
        x = self.features(x)
        x = self.flat(x)

        face_cls = self.face_cls(x)
        bbox_reg = self.bbox_reg(x)

        return face_cls, bbox_reg


class ONet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3),
            nn.PReLU(32),

            nn.MaxPool2d(3, 2, ceil_mode=True),

            nn.Conv2d(32, 64, 3),
            nn.PReLU(64),

            nn.MaxPool2d(3, 2, ceil_mode=True),

            nn.Conv2d(64, 64, 3),
            nn.PReLU(64),

            nn.MaxPool2d(2, 2, ceil_mode=True),

            nn.Conv2d(64, 128, 2),
            nn.PReLU(128)
        )

        self.flat = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(1152, 256)
        )

        self.face_cls = nn.Linear(256, 2)
        self.bbox_reg = nn.Linear(256, 4)

    def forward(self, x):
        x = self.features(x)
        x = self.flat(x)

        face_cls = self.face_cls(x)
        bbox_reg = self.bbox_reg(x)

        return face_cls, bbox_reg
