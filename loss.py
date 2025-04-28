"""
loss.py

Loss functions for spectral-controlled SSM training:
  - pred_loss: forecasting mean squared error over specified horizons.

Dependencies:
    torch
    numpy
    spectral_loss
"""

import torch
from torch import Tensor
import numpy as np
from typing import Any, Dict, List


class Loss:
    """
    Encapsulates forecasting loss for state-space models.

    Attributes:
        horizons (List[int]): Forecast horizons (e.g., [1, 10, 100]).
        h_max (int): Maximum horizon.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize the loss module from configuration.

        Args:
            config (dict): Configuration dictionary containing 'training' section
                           with keys 'horizons' and 'h_max'.
        """
        train_cfg = config.get("training", {})
        horizons = train_cfg.get("horizons", [1, 10, 100])
        if not isinstance(horizons, (list, tuple)):
            raise ValueError(f"'horizons' must be a list or tuple, got {type(horizons)}")
        self.horizons: List[int] = [int(h) for h in horizons]
        if not self.horizons:
            raise ValueError("At least one horizon must be specified in config['training']['horizons']")

        self.h_max: int = int(train_cfg.get("h_max", max(self.horizons)))
        # Validate horizons
        for h in self.horizons:
            if h > self.h_max:
                raise ValueError(f"Horizon {h} exceeds h_max {self.h_max}")

    def pred_loss(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        """
        Compute forecasting loss (mean squared error) over configured horizons.

        Args:
            y_pred (Tensor): Predicted outputs of shape (batch, T_eff, H, p).
            y_true (Tensor): Ground-truth outputs, same shape as y_pred.

        Returns:
            Tensor: Scalar tensor of the forecasting MSE.
        """
        if not isinstance(y_pred, Tensor) or not isinstance(y_true, Tensor):
            raise TypeError("y_pred and y_true must be torch.Tensor")
        if y_pred.shape != y_true.shape:
            raise ValueError(f"Shape mismatch: y_pred {y_pred.shape}, y_true {y_true.shape}")
        if y_pred.ndim != 4:
            raise ValueError(f"Expected y_pred.ndim=4 (batch, T_eff, H, p), got {y_pred.ndim}")

        # Squared error
        diff = y_pred - y_true
        sq_err = diff.pow(2)  # (batch, T_eff, H, p)

        # Sum over output dimension p
        sq_err = sq_err.sum(dim=-1)  # (batch, T_eff, H)

        # Sum over horizons H
        sq_err = sq_err.sum(dim=-1)  # (batch, T_eff)

        # Time-averaged per sample
        T_eff = sq_err.size(1)
        if T_eff <= 0:
            raise ValueError("Effective time dimension T_eff must be > 0")
        loss_per_sample = sq_err.sum(dim=-1) / float(T_eff)  # (batch,)

        # Mean over batch
        return loss_per_sample.mean()
