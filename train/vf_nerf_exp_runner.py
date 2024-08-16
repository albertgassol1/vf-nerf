import sys

# Euler append # isort:skip
sys.path.append('.')  # isort:skip

from config_parser.vf_nerf_config_parser import argparser, parse_config
from train.vector_field_nerf_train import VectorFieldNerfRunner


if __name__ == '__main__':
    parser = argparser()
    args = parser.parse_args()
    config = parse_config(args.scene, args.config_path, args.gpu,
                          args.expname, args.timestamp, args.checkpoint, 
                          args.data_root_dir, args.offline)

    # Create the runner.
    runner = VectorFieldNerfRunner(config)

    # Run the training.
    runner.train()
