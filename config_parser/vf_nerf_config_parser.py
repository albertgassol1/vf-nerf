import sys

sys.path.append('.')  # isort:skip

import configargparse
import GPUtil
from pyhocon import ConfigFactory, ConfigTree
from config_parser.vf_nerf_config import *


def argparser():
    parser = configargparse.ArgumentParser(description='VFNerfRunner')
    parser.add_argument('--scene', type=str, default='65', help='Scene name.')
    parser.add_argument('--config_path', type=str, default="./confs/vf_nerf.conf", help='config file path')
    parser.add_argument('--gpu', type=str, default='auto', help='GPU to use.')
    parser.add_argument('--expname', type=str, default='dtu', help='Experiment name.')
    parser.add_argument('--timestamp', type=str, default='', help='Timestamp.')
    parser.add_argument('--checkpoint', type=str, default='', help='Checkpoint path.')
    parser.add_argument('--data_root_dir', type=str, default='data', help='Data root directory.')
    parser.add_argument('--offline', action='store_true', help='Whether to run offline.')

    return parser


def eval_argparser():
    parser = configargparse.ArgumentParser(description='Evaluate')
    parser.add_argument('--scene', type=str, default='65', help='Scene name.')
    parser.add_argument('--config_path', type=str, default="./confs/vf_nerf.conf", help='config file path')
    parser.add_argument('--gpu', type=str, default='auto', help='GPU to use.')
    parser.add_argument('--resolution', type=int, default=256, help='Resolution.')
    parser.add_argument('--method', type=str, default='render-images', help='Method.')
    parser.add_argument('--expname', type=str, default='dtu', help='Experiment name.')
    parser.add_argument('--data_root_dir', type=str, default='data', help='Data root directory.')
    parser.add_argument('--timestamp', type=str, default='', help='Timestamp.')
    parser.add_argument('--checkpoint', type=str, default='', help='Checkpoint path.')
    parser.add_argument('--eval_folder', type=str, default='evals_vf_nerf', help='Evaluation folder.')
    parser.add_argument('--chunk_size', type=int, default=1024, help='Chunk size.')
    parser.add_argument('--distance_thresh', type=float, default=0.05, help='Distance threshold for 3d metrics.')
    parser.add_argument('--num_quadrants', type=int, default=8, help='Number of quadrants for marching cubes.')

    return parser


def parse_config(scene: str,
                 config_path: str = "confs/vf_nerf.conf",
                 gpu: str = "auto",
                 expname: str = "dtu",
                 timestamp: str = "",
                 checkpoint: str = "",
                 data_root_dir: str = "data",
                 offline: bool = False) -> VFRunnerConfig:
    """
    Parse config file.
    :param scene: Scene name.
    :param config_path: Path to the config file.
    :param gpu: GPU to use.
    :param expname: Experiment name.
    :param timestamp: Timestamp.
    :param checkpoint: Checkpoint path.
    :param data_root_dir: Data root directory.
    :param offline: Whether to run offline.
    :return: The config.
    """

    # Get all config.
    all_conf: ConfigTree = ConfigFactory.parse_file(config_path)

    # Density config.
    density_config = DensityConfig(**all_conf.get_config("density"))
    # VFNet config.
    vf_net_config = VFNetConfig(**all_conf.get_config("vector_field_network"))
    # Renderer config.
    render_net_config = RenderingNetConfig(**all_conf.get_config("rendering"))
    # Ray sampler config.
    ray_sampler_config = RaySamplerConfig(**all_conf.get_config("ray_sampler"))
    # Scheduler config.
    scheduler_config = SchedulerConfig(**all_conf.get_config("scheduler"))
    # Cuda config.
    if gpu == "auto":
        device_ids = GPUtil.getAvailable(order='memory', limit=1, maxLoad=0.5, maxMemory=0.5, includeNan=False,
                                         excludeID=[], excludeUUID=[])
        if len(device_ids) > 0 and torch.cuda.is_available():
            gpu = f"cuda:{device_ids[0]}"
        else:
            gpu = "cpu"
    elif gpu != "cpu":
        raise ValueError("Only support auto gpu mode or cpu gpu mode.")
    cuda_config = CudaConfig(device=torch.device(gpu), num_gpus=torch.cuda.device_count())

    # IBRNerf config.
    vf_nerf_config = VFNerfConfig(vf_net_config, render_net_config, ray_sampler_config, cuda_config,
                                  scheduler_config, density_config, **all_conf.get_config("vf_nerf"))

    # Dataset config.
    dataset_config = DatasetConfig(**all_conf.get_config("dataset"), scene=scene, data_root_dir=data_root_dir)

    # Loss config.
    loss_config = VFLossConfig(**all_conf.get_config("loss").get_config("config"))
    loss_weights = VFLossWeights(**all_conf.get_config("loss").get_config("weights"))

    # IBRRunner config.
    vf_runner_config = VFRunnerConfig(dataset_config, vf_nerf_config, loss_weights, loss_config,
                                      **all_conf.get_config("train"), timestamp=timestamp,
                                      checkpoint=checkpoint, expname=f"{expname}_{scene}",
                                      offline=offline, config_path=config_path,
                                      supervised_loss_weights=VFSupervisedLossWeights(**all_conf.get_config("supervised_loss_weights")))

    return vf_runner_config
