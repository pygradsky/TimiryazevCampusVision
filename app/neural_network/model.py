import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=12,
            kernel_size=3,
        )

        self.conv2 = nn.Conv2d(
            in_channels=12,
            out_channels=32,
            kernel_size=3
        )

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(in_features=32 * 7 * 7, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

