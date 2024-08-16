import sys

sys.path.append('.') # isort:skip

import configargparse
from pyhocon import ConfigFactory, ConfigTree

import config_parser.vf_nerf_config as vf_config
import config_parser.vf_nerf_config_parser as vf_config_parser
from config_parser.joint_opt_config import *


def argparser():
    parser = configargparse.ArgumentParser(description='SupervisedVFRunner')
    parser.add_argument('--scene', type=str, default='65', help='Scene name.')
    parser.add_argument('--vf_config_path', type=str, default="./confs/vf_nerf.conf", help='config file path')
    parser.add_argument('--joint_config_path', type=str, default="./confs/joint_optimization.conf", help='config file path')
    parser.add_argument('--gpu', type=str, default='auto', help='GPU to use.')
    parser.add_argument('--expname', type=str, default='dtu', help='Experiment name.')
    parser.add_argument('--timestamp', type=str, default='', help='Timestamp.')
    parser.add_argument('--checkpoint', type=str, default='', help='Checkpoint path.')
    parser.add_argument('--data_root_dir', type=str, default='data', help='Data root directory.')
    parser.add_argument('--offline', action='store_true', help='Whether to run offline.')

    return parser

def parse_config(scene: str,
                 vf_config_path: str = "confs/vf_nerf.conf",
                 joint_config_path: str = "confs/joint_optimization.conf",
                 gpu: str = "auto",
                 expname: str = "dtu",
                 timestamp: str = "",
                 checkpoint: str = "",
                 data_root_dir: str = "data",
                 offline: bool = False) -> JointOptimizationConfig:
    """
    Parse config file.
    :param scene: Scene name.
    :param vol_config_path: Path to the config file.
    :param vf_config_path: Path to the config file.
    :param gpu: GPU to use.
    :param expname: Experiment name.
    :param timestamp: Timestamp.
    :param checkpoint: Checkpoint path.
    :param data_root_dir: Data root directory.
    :param offline: Whether to run offline.
    :return: JointOptimizationConfig
    """

    # Get vf config
    config_vf: vf_config.VFRunnerConfig = vf_config_parser.parse_config(scene, vf_config_path, gpu,
                                                                        expname, timestamp, checkpoint,
                                                                        data_root_dir, offline)
    config_vf.supervised_loss_weights = vf_config.VFSupervisedLossWeights(**ConfigFactory.parse_file(vf_config_path)["supervised_loss_weights"])


    # Get joint config
    joint_config: ConfigTree = ConfigFactory.parse_file(joint_config_path)

    train_config = TrainConfig(**joint_config.get_config("train"))
    config = JointOptimizationConfig(config_vf, train_config, **joint_config.get_config("joint_optimization"))

    config.vf_config.num_epochs = config.train_config.supervised_vf_epochs

    return config
