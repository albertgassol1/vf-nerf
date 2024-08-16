import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

sys.path.append('.')  # isort:skip

import torch


@dataclass
class DensityConfig:
    beta_bounds: List[float] = field(default_factory=lambda: [1e-4, 1e9])
    mean_bounds: List[float] = field(default_factory=lambda: [0.6, 1.0])
    scale_min: float = 0.1
    params_init: Dict[str, float] = field(default_factory=lambda: {'beta': 0.5,
                                                                   'mean': 0.7,
                                                                   'scale': 100.0})
    cutoff: float = -0.5

    def todict(self) -> Dict[str, Any]:
        return {'beta_bounds': self.beta_bounds,
                'mean_bounds': self.mean_bounds,
                'scale_min': self.scale_min,
                'params_init': self.params_init}


@dataclass
class VFNetConfig:
    """
    VFNet configuration.
    """
    input_dims: int
    output_dims: int
    dimensions: List[int]
    feature_vector_dims: int = 0
    embedder_multires: int = 0
    weight_norm: bool = True
    batch_norm: bool = True
    skip_connection_in: Optional[List[int]] = None
    bias_init: float = 0.0
    dropout: bool = True
    dropout_probability: float = 0.0
    xavier_init: bool = True
    init: str = "center"


@dataclass
class RenderingNetConfig:
    """
    RenderingNet configuration.
    """
    output_dims: int
    dimensions: List[int]
    feature_vector_dims: int = 0
    weight_norm: bool = False
    batch_norm: bool = True
    mode: str = "idr",
    embedder_multires: int = 0
    detach_normals: bool = False


@dataclass
class RaySamplerConfig:
    """
    Ray sampler configuration.
    """
    n_samples: int = 64
    n_importance: int = 64
    rays_per_batch: int = 1024
    perturb: bool = True
    near: float = 0.0
    far: float = 1.0
    fine_range: float = 0.5
    increase_every: int = 100
    max_samples: int = 100

    def fine_sampling(self):
        return self.n_importance > 0


@dataclass
class CudaConfig:
    """
    Cuda configuration.
    """
    device: torch.device = torch.device('cuda')
    num_gpus: int = 1


@dataclass
class SchedulerConfig:
    lr: float = 1e-3
    lr_decay_factor: float = 0.5
    lr_decay_steps: int = 50000
    clip_norm: float = 0.5
    weight_decay: float = 0.0


@dataclass
class VFNerfConfig:
    vf_net_config: VFNetConfig
    rendering_net_config: RenderingNetConfig
    ray_sampler_config: RaySamplerConfig
    cuda_config: CudaConfig
    scheduler_config: SchedulerConfig
    density_config: DensityConfig

    cos_sim_weights: torch.Tensor
    cos_sim_weights_anneal: str
    anneal_start: int
    anneal_end: int

    rendering: str
    normalize_rendering: bool
    dir_to_normal_th: float = -2.0
    numerical_jacobian: bool = False
    border_supervision: bool = True
    center_supervision: bool = True

    def __post_init__(self):
        if self.cos_sim_weights_anneal not in ["none", "hard", "soft"]:
            raise ValueError(f"Invalid cos_sim_weights_anneal: {self.cos_sim_weights_anneal}")
        if self.rendering not in ["nerf", "volsdf"]:
            raise ValueError(f"Invalid rendering: {self.rendering}")
        self.cos_sim_weights = torch.tensor(self.cos_sim_weights).float().to(self.cuda_config.device)

    def cos_sim_weights_dict(self) -> Dict[str, float]:
        """
        Return the cos sim weights as a dictionary.
        :return: The cos sim weights as a dictionary.
        """
        return {f"w_{i}": self.cos_sim_weights[i].item() for i in range(len(self.cos_sim_weights))}


@dataclass
class VFLossWeights:
    rgb: float
    depth: float
    unit_norm: float
    supervision: float
    norm_smaller_than_one: float
    directional_derivatives: float


@dataclass
class VFLossConfig:
    norm_smaller_than_one_start: int
    depth_loss_clamp: float
    directional_derivatives_start: int = 100


@dataclass
class VFSupervisedLossWeights:
    surface: float
    non_surface: float
    supervision: float
    rgb: float
    depth: float
    unit_norm: float
    similarity: float
    colors: float = 0.0
    directional_derivatives: float = 0.0


@dataclass
class DatasetConfig:
    dataset_name: str
    data_dir: str
    shuffle_views: bool
    pixels_per_batch: int
    scene: int
    data_root_dir: str
    all_pixels: bool = False
    factor: int = 20
    white_bkgd: bool = False
    split: str = "train"
    precrop_epochs: int = -10
    precrop_frac: float = 0.5
    far_per_ray: bool = False
    random_img_sampling: bool = False
    border_radius: float = 0.3
    crop_edge: int = 10


@dataclass
class VFRunnerConfig:
    dataset_config: DatasetConfig
    vf_nerf_config: VFNerfConfig
    vf_loss_weights: VFLossWeights
    vf_loss_config: VFLossConfig
    num_epochs: int
    save_frequency: int
    wandb_frequency: int
    timestamp: str = ""
    checkpoint: str = ""

    supervised_loss_weights: Optional[VFSupervisedLossWeights] = None

    exps_folder: str = "exps_vf_nerf"
    config_path: str = "confs/vf_nerf.conf"

    wandb_project: str = "vf_nerf"
    wandb_frequency: int = 100

    start_epoch: int = 0
    expname: str = ""

    offline: bool = False
