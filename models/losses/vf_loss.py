import sys
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from config_parser.vf_nerf_config import VFLossConfig, VFLossWeights

sys.path.append('.')  # isort:skip


class VFLoss(nn.Module):

    def __init__(self, config: VFLossConfig, weights: VFLossWeights) -> None:
        """
        Initialize the IBR loss.
        :param config: The IBR loss config.
        :param weights: The IBR loss weights.
        """

        super().__init__()
        self.config = config
        self.weights = weights

        # Create the loss functions.
        self.rgb_loss = nn.L1Loss()
        self.depth_loss = lambda x, y: F.l1_loss(
            x, y, reduction="none").clamp(max=self.config.depth_loss_clamp).mean()
        self.unit_norm_loss = lambda x: torch.mean((torch.norm(x, dim=1) - 1)**2)
        self.supervision_loss = nn.MSELoss()
        self.norm_smaller_than_one_loss = lambda x: torch.mean(torch.pow(F.relu(torch.norm(x, dim=1) - 1), 2))

    def forward(self, pred: Dict[str, torch.Tensor],
                gt: Dict[str, torch.Tensor], epoch: int) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute the IBR loss.
        :param pred: The predicted values.
        :param gt: The ground truth values.
        :param epoch: The epoch.
        :return: The IBR loss.
        """
        # Compute the rgb loss.
        rgb_loss = self.rgb_loss(pred["rgb"], gt["rgb"])

        # Compute the depth loss.
        if gt["depth"].nelement() > 0:
            depth_loss = self.depth_loss(pred["depth"], gt["depth"])
        else:
            depth_loss = torch.tensor(0.0).to(pred["rgb"].device)

        # Compute the unit norm loss.
        unit_norm_loss = self.unit_norm_loss(pred["normals"])

        # Compute the supervision loss.
        if pred["supervised_normals"].nelement() > 0:
            supervision_loss = self.supervision_loss(pred["supervised_normals"], gt["supervised_normals"])
        else:
            supervision_loss = torch.tensor(0.0).to(pred["rgb"].device)

        # Compute the norm smaller than one loss.
        if epoch >= self.config.norm_smaller_than_one_start:
            norm_smaller_than_one_loss = self.norm_smaller_than_one_loss(pred["normals"])
        else:
            norm_smaller_than_one_loss = torch.tensor(0.0).to(pred["rgb"].device)

        # Directional derivatives loss.
        directional_derivatives_loss = torch.tensor(0.0).to(pred["rgb"].device)
        if pred["directional_derivatives"] is not None and epoch >= self.config.directional_derivatives_start:
            directional_derivatives_loss = torch.mean(pred["directional_derivatives"])

        # Compute the total loss.
        loss = self.weights.rgb * rgb_loss + \
            self.weights.depth * depth_loss + \
            self.weights.unit_norm * unit_norm_loss + \
            self.weights.supervision * supervision_loss + \
            self.weights.norm_smaller_than_one * norm_smaller_than_one_loss + \
            self.weights.directional_derivatives * directional_derivatives_loss

        return loss, {
            "rgb_loss": rgb_loss.item(),
            "depth_loss": depth_loss.item(),
            "unit_norm_loss": unit_norm_loss.item(),
            "supervision_loss": supervision_loss.item(),
            "norm_smaller_than_one_loss": norm_smaller_than_one_loss.item(),
            "directional_derivatives_loss": directional_derivatives_loss.item()
        }
