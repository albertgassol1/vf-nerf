from dataclasses import dataclass
from typing import Dict

import torch


@dataclass
class DatasetOutput:
    """
    Dataset output.
    """
    rgb: torch.Tensor
    uv: torch.Tensor
    intrinsics: torch.Tensor
    pose: torch.Tensor
    depth: torch.Tensor = torch.empty(0)
    far: torch.Tensor = torch.empty(0)

    def to_dict(self) -> Dict[str, torch.Tensor]:
        """
        Convert the dataset output to a dictionary.
        :return: The dictionary.
        """
        return {"rgb": self.rgb,
                "uv": self.uv,
                "intrinsics": self.intrinsics,
                "pose": self.pose,
                "depth": self.depth,
                "far": self.far}
