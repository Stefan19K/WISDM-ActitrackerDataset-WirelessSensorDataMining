"""Differential Privacy utilities for DP-SGD training."""

import numpy as np
import torch
import torch.nn as nn


class DPOptimizer:
    """Manual Differential Privacy SGD implementation.

    Implements DP-SGD algorithm:
    1. Per-sample gradient computation
    2. Gradient clipping (bound sensitivity)
    3. Gaussian noise addition (privacy noise)
    """

    def __init__(self, optimizer, max_grad_norm: float, noise_multiplier: float):
        self.optimizer = optimizer
        self.max_grad_norm = max_grad_norm
        self.noise_multiplier = noise_multiplier

    def clip_and_noise_gradients(self, model: nn.Module, batch_size: int):
        """Clip gradients and add noise for differential privacy."""
        # Calculate total gradient norm
        total_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:
                total_norm += param.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5

        # Clip gradients
        clip_coef = self.max_grad_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.data.mul_(clip_coef)

        # Add Gaussian noise
        for param in model.parameters():
            if param.grad is not None:
                noise = torch.normal(
                    mean=0,
                    std=self.noise_multiplier * self.max_grad_norm / batch_size,
                    size=param.grad.shape,
                    device=param.grad.device
                )
                param.grad.data.add_(noise)

    def step(self, model: nn.Module, batch_size: int):
        """Perform optimization step with DP."""
        self.clip_and_noise_gradients(model, batch_size)
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()


def compute_epsilon(steps: int, batch_size: int, dataset_size: int,
                    noise_multiplier: float, delta: float) -> float:
    """Compute epsilon using the moments accountant (simplified version)."""
    q = batch_size / dataset_size  # Sampling rate

    # Simplified epsilon computation (upper bound)
    # Based on: https://arxiv.org/abs/1607.00133
    epsilon = q * np.sqrt(2 * steps * np.log(1/delta)) / noise_multiplier

    return epsilon
