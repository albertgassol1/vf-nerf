import sys
from dataclasses import dataclass

sys.path.append('.')  # isort:skip

from config_parser.vf_nerf_config import VFRunnerConfig


@dataclass
class TrainConfig:
    initial_training_epochs: int
    supervised_vf_epochs: int
    joint_epochs: int
    supervise_every: int
    supervision_epochs: int
    refinement_init_lr: float = 1e-4

    reset_scheduler: bool = False


@dataclass
class JointOptimizationConfig:
    vf_config: VFRunnerConfig
    train_config: TrainConfig

    save_frequency: int
    num_bases: int
    decimation: float
    self_supervise: bool
