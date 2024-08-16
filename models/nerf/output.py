from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch


@dataclass
class NerfOutput:
    points_coarse: torch.Tensor
    coarse_normals: torch.Tensor
    coarse_rgb_values: torch.Tensor
    coarse_depth_map: torch.Tensor
    mask: Optional[torch.Tensor] = None
    z_vals: Optional[torch.Tensor] = None
    points_fine: Optional[torch.Tensor] = None
    fine_normals: Optional[torch.Tensor] = None
    fine_rgb_values: Optional[torch.Tensor] = None
    fine_depth_map: Optional[torch.Tensor] = None
    fine_mask: Optional[torch.Tensor] = None
    directional_derivtives: Optional[torch.Tensor] = None
    ray_dirs: Optional[torch.Tensor] = None
    coarse_colors: Optional[torch.Tensor] = None

    def fine_active(self) -> bool:
        """
        Return true if the fine network is active.
        :returns: True if the fine network is active.
        """
        return self.fine_normals is not None

    def get_normals(self,
                    N_rays: int,
                    N_coarse: int,
                    N_fine: int) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Returns the points and neighbors normals of the coarse and fine network.
        :params N_rays: The number of rays.
        :params N_coarse: The number of coarse neighbors.
        :params N_fine: The number of fine neighbors.
        :returns: The normals.
        """
        coarse_normals = self.coarse_normals.reshape(N_rays, N_coarse, 3)

        if self.fine_active():
            fine_normals = self.fine_normals.reshape(N_rays, N_fine, 3)
            return coarse_normals[:, :-1, :].reshape(-1, 3), \
                coarse_normals[:, 1:, :].reshape(-1, 3), \
                fine_normals[:, :-1, :].reshape(-1, 3), \
                fine_normals[:, 1:, :].reshape(-1, 3)
        else:
            return coarse_normals[:, :-1, :].reshape(-1, 3), \
                coarse_normals[:, 1:, :].reshape(-1, 3), None, None
        
    def to_dict(self) -> Dict[str, torch.Tensor]:
        """
        Return a dictionary of tensors.
        :returns: A dictionary of tensors.
        """
        return {
            "points_coarse": self.points_coarse,
            "coarse_normals": self.coarse_normals,
            "coarse_rgb_values": self.coarse_rgb_values,
            "coarse_depth_map": self.coarse_depth_map,
            "mask": self.mask,
            "points_fine": self.points_fine,
            "fine_normals": self.fine_normals,
            "fine_rgb_values": self.fine_rgb_values,
            "fine_depth_map": self.fine_depth_map,
            "fine_mask": self.fine_mask
        }
