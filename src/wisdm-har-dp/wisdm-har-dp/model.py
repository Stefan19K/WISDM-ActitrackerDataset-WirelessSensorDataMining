"""Model architecture for Human Activity Recognition."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ActivityCNN(nn.Module):
    """1D Convolutional Neural Network for Activity Recognition."""

    def __init__(self, num_features: int, num_classes: int, hidden_size: int = 64, dropout: int = 0.5):
        super(ActivityCNN, self).__init__()

        # Reshape input for 1D convolution: (batch, 1, features)
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.gn1 = nn.GroupNorm(8, 32)  # GroupNorm is DP-compatible (unlike BatchNorm)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(8, 64)

        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.gn3 = nn.GroupNorm(8, 128)

        # Fully connected layers
        self.fc1 = nn.Linear(128 * num_features, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Reshape: (batch, features) -> (batch, 1, features)
        x = x.unsqueeze(1)

        # Convolutional layers
        x = F.relu(self.gn1(self.conv1(x)))
        x = F.relu(self.gn2(self.conv2(x)))
        x = F.relu(self.gn3(self.conv3(x)))

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x
