import os
import sys

sys.path.append('.')  # isort:skip

import evaluation.methods as methods
import utils.utils as utils
from config_parser.vf_nerf_config import VFRunnerConfig
from config_parser.vf_nerf_config_parser import eval_argparser, parse_config
from datasets.normal_datasets import dataset_dict
from models.nerf.vector_field_nerf import VectorFieldNerf


def evaluate(config: VFRunnerConfig, method: str,
             resolution: int, eval_root_folder: str, chunk_size: int,
             distance_thresh: float, num_quadrants: int) -> None:
    """
    Evaluate the model.
    :param config: The config.
    :param method: The method.
    :param resolution: The resolution.
    :param eval_root_folder: The evaluation root folder.
    :param chunk_size: The chunk size.
    :param distance_thresh: The distance threshold.
    :param num_quadrants: The number of quadrants.
    """
    # Get the model path.
    path_to_model = os.path.join(config.exps_folder, config.expname, config.timestamp,
                                 "checkpoints", "vf_nerf", f"{config.checkpoint}.pth")
    config.vf_nerf_config.ray_sampler_config.perturb = False
    # Create the Nerf model.
    config.vf_nerf_config.dir_to_normal_th = -0.2
    model = VectorFieldNerf(config.vf_nerf_config)

    # Load the model.
    epoch = model.load(path_to_model)
    if model.config.ray_sampler_config.fine_sampling():
        model.fine_sampler.N_samples = \
            min(model.fine_sampler.N_samples + int( 5 * (epoch // model.config.ray_sampler_config.increase_every)), 
                model.fine_sampler.max_samples)
        print(f"Fine sampler N_samples: {model.fine_sampler.N_samples}")

    # Create the evaluation folder.
    utils.mkdir_ifnotexists(eval_root_folder)
    eval_folder = os.path.join(eval_root_folder, config.expname)
    utils.mkdir_ifnotexists(eval_folder)
    eval_folder = os.path.join(eval_folder, f"{config.timestamp}_{config.checkpoint}")
    utils.mkdir_ifnotexists(eval_folder)

    model.eval()

    print("Evaluating the model.")
    vf_net = model.fine_vector_field_network if model.config.ray_sampler_config.fine_sampling() else model.vector_field_network
    if method in ["marching-cubes-mesh", "all"]:
        dataset = dataset_dict[config.dataset_config.dataset_name](config.dataset_config)
        utils.mkdir_ifnotexists(os.path.join(eval_folder, "mesh"))
        methods.marching_cubes_mesh(vf_net, resolution,
                                    os.path.join(eval_folder, "mesh"),
                                    config.checkpoint, scale=dataset.scale,
                                    max_batch=100000,
                                    centroid=dataset.get_centroid("cpu"),
                                    device=config.vf_nerf_config.cuda_config.device,
                                    smooth_after=False,
                                    smooth_all=False)
        utils.mkdir_ifnotexists(os.path.join(eval_folder, "mesh-smoothed"))
        methods.marching_cubes_mesh(vf_net, resolution,
                                    os.path.join(eval_folder, "mesh-smoothed"),
                                    config.checkpoint, scale=dataset.scale,
                                    max_batch=100000,
                                    centroid=dataset.get_centroid("cpu"),
                                    device=config.vf_nerf_config.cuda_config.device,
                                    smooth_after=False,
                                    smooth_all=True)
        utils.mkdir_ifnotexists(os.path.join(eval_folder, "mesh-smoothed-after"))
        methods.marching_cubes_mesh(vf_net, resolution,
                                    os.path.join(eval_folder, "mesh-smoothed-after"),
                                    config.checkpoint, scale=dataset.scale,
                                    max_batch=100000,
                                    centroid=dataset.get_centroid("cpu"),
                                    device=config.vf_nerf_config.cuda_config.device,
                                    smooth_after=True,
                                    smooth_all=False)
    if method in ["quadrant-marching-cubes-mesh", "all"]:
        dataset = dataset_dict[config.dataset_config.dataset_name](config.dataset_config)
        utils.mkdir_ifnotexists(os.path.join(eval_folder, "merged-mesh"))
        methods.quadrant_marching_cubes(vf_net, resolution,
                                        os.path.join(eval_folder, "merged-mesh"),
                                        config.checkpoint, scale=dataset.scale,
                                        max_batch=100000,
                                        centroid=dataset.get_centroid("cpu"),
                                        num_quadrants=num_quadrants,
                                        device=config.vf_nerf_config.cuda_config.device,
                                        smooth_after=False,
                                        smooth_all=False)
        utils.mkdir_ifnotexists(os.path.join(eval_folder, "merged-mesh-smoothed"))
        methods.quadrant_marching_cubes(vf_net, resolution,
                                        os.path.join(eval_folder, "merged-mesh-smoothed"),
                                        config.checkpoint, scale=dataset.scale,
                                        max_batch=100000,
                                        centroid=dataset.get_centroid("cpu"),
                                        num_quadrants=num_quadrants,
                                        device=config.vf_nerf_config.cuda_config.device,
                                        smooth_after=False,
                                        smooth_all=True)
        utils.mkdir_ifnotexists(os.path.join(eval_folder, "merged-mesh-smoothed-after"))
        methods.quadrant_marching_cubes(vf_net, resolution,
                                        os.path.join(eval_folder, "merged-mesh-smoothed-after"),
                                        config.checkpoint, scale=dataset.scale,
                                        max_batch=100000,
                                        centroid=dataset.get_centroid("cpu"),
                                        num_quadrants=num_quadrants,
                                        device=config.vf_nerf_config.cuda_config.device,
                                        smooth_after=True,
                                        smooth_all=False)

    if method in ["plot-2d-slices", "all"]:
        dataset = dataset_dict[config.dataset_config.dataset_name](config.dataset_config)
        methods.plot_2d_slices(vf_net, path=eval_folder, scale=dataset.scale / 1.1 * 1.02,
                               centroid=dataset.get_centroid("cpu"),
                               device=config.vf_nerf_config.cuda_config.device)
        methods.plot_2d_slices(vf_net, path=eval_folder, scale=dataset.scale / 1.1 * 1.02,
                               centroid=dataset.get_centroid("cpu"),
                               device=config.vf_nerf_config.cuda_config.device, smooth=True)
    if method in ["plot-overall-scene", "all"]:
        dataset = dataset_dict[config.dataset_config.dataset_name](config.dataset_config)
        methods.plot_overall_scene(vf_net, path=eval_folder, scale=dataset.scale / 1.1,
                                   centroid=dataset.get_centroid("cpu"),
                                   device=config.vf_nerf_config.cuda_config.device)
        methods.plot_overall_scene(vf_net, path=eval_folder, scale=dataset.scale / 1.1,
                                   centroid=dataset.get_centroid("cpu"),
                                   device=config.vf_nerf_config.cuda_config.device, smooth=True)
    if method in ["plot-3d-slices", "all"]:
        methods.plot_3d_slices(vf_net, eval_folder, device=config.vf_nerf_config.cuda_config.device)
        methods.plot_3d_slices(vf_net, eval_folder, device=config.vf_nerf_config.cuda_config.device, smooth=True)
    if method in ["render-images", "all"]:
        methods.render_images(model, eval_folder, config.dataset_config, epoch, chunk_size,
                              device=config.vf_nerf_config.cuda_config.device)
    if method in ["metrics", "all"]:
        methods.metrics(model, eval_folder, config.dataset_config, epoch, chunk_size,
                        device=config.vf_nerf_config.cuda_config.device)
    if method in ["tsdf-mesh", "all"]:
        methods.tsdf_mesh(eval_folder, config.dataset_config)
    if method in ["3d-metrics", "all"]:
        dataset = dataset_dict[config.dataset_config.dataset_name](config.dataset_config)

        if config.timestamp in ["monosdf", "neuralangelo", "neuris", "manhattan_sdf", "mono_sdf"]:
            methods.metrics_3d_no_vf(eval_folder, config.checkpoint, config.dataset_config, distance_thresh=distance_thresh)
        else:
            methods.metrics_3d(eval_folder, config.dataset_config, distance_thresh=distance_thresh)
            
if __name__ == '__main__':
    parser = eval_argparser()
    args = parser.parse_args()
    config = parse_config(scene=args.scene, config_path=args.config_path, gpu=args.gpu,
                          expname=args.expname, timestamp=args.timestamp, checkpoint=args.checkpoint,
                          data_root_dir=args.data_root_dir)
    evaluate(config, args.method, args.resolution, args.eval_folder, args.chunk_size,
             args.distance_thresh, args.num_quadrants)
