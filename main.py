import argparse
import json
import os
import shutil
import time
import open3d as o3d
from camera_constraints import (
    CAMERA_CONSTRAINT_SHAPES,
    resolve_camera_constraint_args,
)
from config_utils import parse_args_with_json_config
from dataset.occupancy_map import build_swept_object_visualization_cloud
from dataset.dataset import *
from dataset.utils import configure_camera_models_from_args
from visualization.visual_cam_points import *
from torch.utils.tensorboard import SummaryWriter
from bip_optimizer import BIPCameraOpt
from optimization import CameraLayerOpt
from field.field_attribute import voxel_model
from mocap_config import resolve_mocap_templates
from run_paths import apply_run_naming_to_path
os.environ['PYOPENGL_PLATFORM'] = 'osmesa'


def resolve_output_dir(base_dir, relative_path):
    normalized_relative = os.path.normpath(relative_path)
    if normalized_relative in ("", ".", os.sep):
        raise ValueError("--path must not point to the output root")

    output_dir = os.path.abspath(os.path.join(base_dir, normalized_relative))
    base_dir_abs = os.path.abspath(base_dir)
    if os.path.commonpath([base_dir_abs, output_dir]) != base_dir_abs:
        raise ValueError(f"--path must stay inside {base_dir}/")
    return output_dir


def format_runtime(seconds):
    seconds = float(seconds)
    if seconds < 60.0:
        return f"{seconds:.2f}s"
    minutes, rem_seconds = divmod(seconds, 60.0)
    if minutes < 60.0:
        return f"{int(minutes)}m {rem_seconds:.2f}s"
    hours, rem_minutes = divmod(minutes, 60.0)
    return f"{int(hours)}h {int(rem_minutes)}m {rem_seconds:.2f}s"


def append_optimization_key_log(args, message):
    key_log_path = getattr(args, "optimization_key_log_path", None)
    if not key_log_path:
        return
    with open(key_log_path, "a", encoding="utf-8") as handle:
        handle.write(str(message) + "\n")


if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path',type=str, default='random/moto/')
    parser.add_argument('--mocap_sequence',type=str,default=None)
    parser.add_argument('--mocap_data_dir',type=str,default='mocap_data')
    parser.add_argument('--lr1',type=float,default=1e-3)
    parser.add_argument('--lr2',type=float,default=1e-3)
    parser.add_argument('--image_width',type=int,default=640)
    parser.add_argument('--image_height',type=int,default=480)
    parser.add_argument('--fx',type=float,default=320.0)
    parser.add_argument('--fy',type=float,default=320.0)
    parser.add_argument('--cx',type=float,default=319.5)
    parser.add_argument('--cy',type=float,default=239.5)
    parser.add_argument('--camera_models', type=json.loads, default=None)
    parser.add_argument('--model_physical_height',type=float,default=None)
    parser.add_argument('--cameranum',type=int,default=10)
    parser.add_argument('--epoches',type=int,default=20)
    parser.add_argument('--iterations',type=int,default=20)
    parser.add_argument('--non_gradient_reset_enable',dest='non_gradient_reset_enable',action='store_true')
    parser.add_argument('--no_non_gradient_reset_enable',dest='non_gradient_reset_enable',action='store_false')
    parser.set_defaults(non_gradient_reset_enable=True)
    parser.add_argument('--non_gradient_reset_interval',type=int,default=5)
    parser.add_argument('--reset_search_enable',dest='reset_search_enable',action='store_true')
    parser.add_argument('--no_reset_search_enable',dest='reset_search_enable',action='store_false')
    parser.set_defaults(reset_search_enable=False)
    parser.add_argument('--reset_search_trials',type=int,default=0)
    parser.add_argument('--reset_search_top_k',type=int,default=1)
    parser.add_argument('--reset_search_score',type=str,choices=['joint_gap','angle','combined'],default='angle')
    parser.add_argument('--reset_search_angle_weight',type=float,default=0.25)
    parser.add_argument('--reset_search_save_all',dest='reset_search_save_all',action='store_true')
    parser.add_argument('--no_reset_search_save_all',dest='reset_search_save_all',action='store_false')
    parser.set_defaults(reset_search_save_all=True)
    parser.add_argument('--epoch_checkpoint_interval',type=int,default=0)
    parser.add_argument('--epoch_checkpoint_epochs',type=int,nargs='*',default=None)
    parser.add_argument('--epoch_checkpoint_mode',type=str,choices=['epoch_best','epoch_current','global_best'],default='epoch_best')
    parser.add_argument('--epoch_checkpoint_save_reset',dest='epoch_checkpoint_save_reset',action='store_true')
    parser.add_argument('--no_epoch_checkpoint_save_reset',dest='epoch_checkpoint_save_reset',action='store_false')
    parser.set_defaults(epoch_checkpoint_save_reset=True)
    parser.add_argument('--config',type=str,default=None)
    parser.add_argument('--decay',type=float,default=1e-4)
    parser.add_argument('--solver',type=str,choices=['neof','bip'],default='neof')
    parser.add_argument('--optimizer',type=str,default="Adam")
    parser.add_argument('--kcoverage',type=int,default=3)
    parser.add_argument('--require_all_cameras_coverage',dest='require_all_cameras_coverage',action='store_true')
    parser.add_argument('--no_require_all_cameras_coverage',dest='require_all_cameras_coverage',action='store_false')
    parser.set_defaults(require_all_cameras_coverage=False)
    parser.add_argument('--isscene',type=int,default=0)
    parser.add_argument('--scene',type=int,dest='isscene')
    parser.add_argument('--wvis',type=float,default=0.4)
    parser.add_argument('--wcc',type=float,default=0.3)
    parser.add_argument('--wco',type=float,default=0.3)
    parser.add_argument('--wres',type=float,default=0.0)
    parser.add_argument('--modelname',type=str,default='scene/room_0.ply')
    parser.add_argument('--voxelnum',type=int,default=30000)
    parser.add_argument('--voxelsize',type=float,default=0.02)
    parser.add_argument('--target_visibility_mode',type=str,choices=['surface_occlusion','fov_only'],default='surface_occlusion')
    parser.add_argument('--eval_visibility_mode',type=str,choices=['surface_occlusion','fov_only'],default='surface_occlusion')
    parser.add_argument('--field_hidden_dim',type=int,default=32)
    parser.add_argument('--field_num_heads',type=int,default=1)
    parser.add_argument('--field_context_voxel_num',type=int,default=64)
    parser.add_argument('--field_normal_voxel_num',type=int,default=8)
    parser.add_argument('--field_lr',type=float,default=1e-3)
    parser.add_argument('--field_lr_decay',type=float,default=0.99)
    parser.add_argument('--field_fit_steps',type=int,default=1000)
    parser.add_argument('--field_fit_log_interval',type=int,default=100)
    parser.add_argument('--field_early_stop',dest='field_early_stop',action='store_true')
    parser.add_argument('--no_field_early_stop',dest='field_early_stop',action='store_false')
    parser.set_defaults(field_early_stop=True)
    parser.add_argument('--field_early_stop_patience',type=int,default=150)
    parser.add_argument('--field_early_stop_min_delta',type=float,default=1e-6)
    parser.add_argument('--field_early_stop_min_steps',type=int,default=300)
    parser.add_argument('--field_query_jitter_std',type=float,default=0.005)
    parser.add_argument('--field_exclude_closest_context',dest='field_exclude_closest_context',action='store_true')
    parser.add_argument('--no_field_exclude_closest_context',dest='field_exclude_closest_context',action='store_false')
    parser.set_defaults(field_exclude_closest_context=True)
    parser.add_argument('--neof_log_mode',type=str,choices=['compact','detailed'],default='detailed')
    parser.add_argument('--neof_pose_log_interval',type=int,default=1)
    parser.add_argument('--optimization_key_log_name',type=str,default='optimization_key.log')
    parser.add_argument('--occupancy_map_enable',dest='occupancy_map_enable',action='store_true')
    parser.add_argument('--no_occupancy_map_enable',dest='occupancy_map_enable',action='store_false')
    parser.set_defaults(occupancy_map_enable=False)
    parser.add_argument('--occupancy_map_mode',type=str,choices=['route_kde','voxel_weights','free_space_box'],default='route_kde')
    parser.add_argument('--occupancy_map_file',type=str,default=None)
    parser.add_argument('--occupancy_map_sigma_xyz',type=float,nargs=3,default=[0.08,0.08,0.12])
    parser.add_argument('--occupancy_map_floor',type=float,default=0.05)
    parser.add_argument('--occupancy_map_normalize',type=str,choices=['sum1','mean1','max1'],default='mean1')
    parser.add_argument('--occupancy_map_route_step',type=float,default=0.03)
    parser.add_argument('--occupancy_box_min',type=float,nargs=3,default=None)
    parser.add_argument('--occupancy_box_max',type=float,nargs=3,default=None)
    parser.add_argument('--occupancy_box_grid_step',type=float,default=0.04)
    parser.add_argument('--occupancy_box_distribution',type=str,choices=['uniform','gaussian','gaussian_mixture'],default='uniform')
    parser.add_argument('--occupancy_gaussian_center',type=float,nargs=3,default=None)
    parser.add_argument('--occupancy_gaussian_sigma_xyz',type=float,nargs=3,default=None)
    parser.add_argument('--show_occupancy_map',dest='show_occupancy_map',action='store_true')
    parser.add_argument('--no_show_occupancy_map',dest='show_occupancy_map',action='store_false')
    parser.set_defaults(show_occupancy_map=True)
    parser.add_argument('--show_static_geometry_in_occupancy',dest='show_static_geometry_in_occupancy',action='store_true')
    parser.add_argument('--no_show_static_geometry_in_occupancy',dest='show_static_geometry_in_occupancy',action='store_false')
    parser.set_defaults(show_static_geometry_in_occupancy=False)
    parser.add_argument('--occupancy_vis_mode',type=str,choices=['surface','volume'],default='volume')
    parser.add_argument('--show_occupancy_route',dest='show_occupancy_route',action='store_true')
    parser.add_argument('--no_show_occupancy_route',dest='show_occupancy_route',action='store_false')
    parser.set_defaults(show_occupancy_route=True)
    parser.add_argument('--show_occupancy_route_direction',dest='show_occupancy_route_direction',action='store_true')
    parser.add_argument('--no_show_occupancy_route_direction',dest='show_occupancy_route_direction',action='store_false')
    parser.set_defaults(show_occupancy_route_direction=True)
    parser.add_argument('--occupancy_vis_clip_percentile',type=float,default=95.0)
    parser.add_argument('--occupancy_vis_point_size',type=float,default=5.0)
    parser.add_argument('--occupancy_vis_route_radius',type=float,default=0.004)
    parser.add_argument('--occupancy_vis_route_arrow_radius',type=float,default=0.005)
    parser.add_argument('--occupancy_vis_route_arrow_length',type=float,default=0.05)
    parser.add_argument('--occupancy_vis_volume_step',type=float,default=0.04)
    parser.add_argument('--save_comparison_visualization',dest='save_comparison_visualization',action='store_true')
    parser.add_argument('--no_save_comparison_visualization',dest='save_comparison_visualization',action='store_false')
    parser.set_defaults(save_comparison_visualization=True)
    parser.add_argument('--save_coverage_visualization',dest='save_coverage_visualization',action='store_true')
    parser.add_argument('--no_save_coverage_visualization',dest='save_coverage_visualization',action='store_false')
    parser.set_defaults(save_coverage_visualization=True)
    parser.add_argument('--coverage_vis_point_size',type=float,default=6.0)
    parser.add_argument('--save_camera_responsibility_visualization',dest='save_camera_responsibility_visualization',action='store_true')
    parser.add_argument('--no_save_camera_responsibility_visualization',dest='save_camera_responsibility_visualization',action='store_false')
    parser.set_defaults(save_camera_responsibility_visualization=True)
    parser.add_argument('--save_per_camera_responsibility_visualization',dest='save_per_camera_responsibility_visualization',action='store_true')
    parser.add_argument('--no_save_per_camera_responsibility_visualization',dest='save_per_camera_responsibility_visualization',action='store_false')
    parser.set_defaults(save_per_camera_responsibility_visualization=True)
    parser.add_argument('--save_visual_summary',dest='save_visual_summary',action='store_true')
    parser.add_argument('--no_save_visual_summary',dest='save_visual_summary',action='store_false')
    parser.set_defaults(save_visual_summary=True)
    parser.add_argument('--visual_summary_columns',type=int,default=3)
    parser.add_argument('--occupancy_hotspot_fraction',type=float,default=0.2)
    parser.add_argument('--pose_lr_decay',type=float,default=0.95)
    parser.add_argument('--visibility_depth_temperature',type=float,default=0.05)
    parser.add_argument('--visibility_fov_temperature',type=float,default=0.05)
    parser.add_argument('--visibility_normal_temperature',type=float,default=0.1)
    parser.add_argument('--sampling_quality_temperature',type=float,default=0.5)
    parser.add_argument('--sampling_quality_topk',type=int,default=2)
    parser.add_argument('--sampling_quality_min_projected_voxel_px',type=float,default=1.5)
    parser.add_argument('--camera_constraint_enable',dest='camera_constraint_enable',action='store_true')
    parser.add_argument('--no_camera_constraint_enable',dest='camera_constraint_enable',action='store_false')
    parser.set_defaults(camera_constraint_enable=False)
    parser.add_argument(
        '--camera_constraint_shape',
        type=str,
        choices=CAMERA_CONSTRAINT_SHAPES,
        default='box',
    )
    parser.add_argument('--camera_constraint_box_min',type=float,nargs=3,default=None)
    parser.add_argument('--camera_constraint_box_max',type=float,nargs=3,default=None)
    parser.add_argument('--camera_constraint_cylinder_center_xy',type=float,nargs=2,default=None)
    parser.add_argument('--camera_constraint_cylinder_radius_xy',type=float,nargs=2,default=None)
    parser.add_argument('--camera_constraint_cylinder_z_range',type=float,nargs=2,default=None)
    parser.add_argument('--camera_constraint_dome_base_center',type=float,nargs=3,default=None)
    parser.add_argument('--camera_constraint_dome_radius_xyz',type=float,nargs=3,default=None)
    parser.add_argument('--camera_constraint_plane_center',type=float,nargs=3,default=None)
    parser.add_argument('--camera_constraint_plane_span_u',type=float,nargs=3,default=None)
    parser.add_argument('--camera_constraint_plane_span_v',type=float,nargs=3,default=None)
    parser.add_argument('--camera_init_strategy',type=str,choices=['random','grid'],default='random')
    parser.add_argument('--reset_random_samples_per_voxel',type=int,default=4)
    parser.add_argument('--bip_candidate_position_step',type=float,default=0.5)
    parser.add_argument('--bip_target_count',type=int,default=5)
    parser.add_argument('--bip_max_candidate_positions',type=int,default=500)
    parser.add_argument('--bip_max_base_candidates',type=int,default=800)
    parser.add_argument('--bip_pair_candidate_limit',type=int,default=20)
    parser.add_argument('--bip_preprocess_enable',dest='bip_preprocess_enable',action='store_true')
    parser.add_argument('--no_bip_preprocess_enable',dest='bip_preprocess_enable',action='store_false')
    parser.set_defaults(bip_preprocess_enable=True)
    parser.add_argument('--bip_visibility_filter_stat',type=str,choices=['max','mean','min'],default='max')
    parser.add_argument('--bip_min_visible_points',type=int,default=1)
    parser.add_argument('--bip_min_visible_fraction',type=float,default=0.0)
    parser.add_argument('--bip_time_limit',type=float,default=60.0)
    parser.add_argument('--bip_coverage_mode',type=str,choices=['kcoverage','pair_angle'],default='pair_angle')
    parser.add_argument('--bip_solver_backend',type=str,choices=['scipy_highs','ortools_cpsat'],default='scipy_highs')
    parser.add_argument('--bip_cpsat_weight_scale',type=float,default=1000.0)
    parser.add_argument('--bip_cpsat_num_workers',type=int,default=0)
    parser.add_argument('--bip_cpsat_random_seed',type=int,default=0)
    parser.add_argument('--bip_min_triangulation_angle_deg',type=float,default=15.0)
    parser.add_argument('--bip_max_triangulation_angle_deg',type=float,default=165.0)
    parser.add_argument('--bip_max_pair_variables',type=int,default=200000)
    parser.add_argument('--bip_allow_duplicate_positions',dest='bip_allow_duplicate_positions',action='store_true')
    parser.add_argument('--no_bip_allow_duplicate_positions',dest='bip_allow_duplicate_positions',action='store_false')
    parser.set_defaults(bip_allow_duplicate_positions=False)
    parser.add_argument('--timestamp_output',dest='timestamp_output',action='store_true')
    parser.add_argument('--no_timestamp_output',dest='timestamp_output',action='store_false')
    parser.set_defaults(timestamp_output=True)
    parser.add_argument('--run_timestamp',type=str,default=None)
    parser.add_argument('--clear_previous_results',dest='clear_previous_results',action='store_true')
    parser.add_argument('--no_clear_previous_results',dest='clear_previous_results',action='store_false')
    parser.set_defaults(clear_previous_results=True)
    parser.add_argument('--show_world_axes',dest='show_world_axes',action='store_true')
    parser.add_argument('--no_show_world_axes',dest='show_world_axes',action='store_false')
    parser.set_defaults(show_world_axes=True)
    parser.add_argument('--show_world_axis_tick_labels',dest='show_world_axis_tick_labels',action='store_true')
    parser.add_argument('--no_show_world_axis_tick_labels',dest='show_world_axis_tick_labels',action='store_false')
    parser.set_defaults(show_world_axis_tick_labels=False)
    parser.add_argument('--world_axis_length',type=float,default=None)
    parser.add_argument('--world_axis_tick_step',type=float,default=0.5)
    parser.add_argument('--world_axis_tick_size',type=float,default=None)
    parser.add_argument('--world_axis_radius',type=float,default=None)
    parser.add_argument('--world_axis_label_size',type=float,default=None)
    parser.add_argument('--camera_vis_scale',type=float,default=0.2)
    parser.add_argument('--show_camera_optical_axis',dest='show_camera_optical_axis',action='store_true')
    parser.add_argument('--no_show_camera_optical_axis',dest='show_camera_optical_axis',action='store_false')
    parser.set_defaults(show_camera_optical_axis=True)
    parser.add_argument('--show_camera_fov',dest='show_camera_fov',action='store_true')
    parser.add_argument('--no_show_camera_fov',dest='show_camera_fov',action='store_false')
    parser.set_defaults(show_camera_fov=True)
    parser.add_argument('--camera_optical_axis_length',type=float,default=None)
    parser.add_argument('--camera_line_radius',type=float,default=None)
    parser.add_argument('--vismode',type=str,choices=['save','interactive','none'],default='save')
    args = parse_args_with_json_config(parser)
    args = resolve_mocap_templates(args)
    args = resolve_camera_constraint_args(args)
    original_path = args.path
    args.path = apply_run_naming_to_path(
        args.path,
        args.solver,
        args.camera_constraint_shape,
        run_timestamp=args.run_timestamp,
        append_timestamp=args.timestamp_output,
    )
    if args.vismode == 'none':
        print("vismode=none is treated as headless save mode; visualization files will still be written.")
        args.vismode = 'save'
    args = configure_camera_models_from_args(args)
    if args.require_all_cameras_coverage:
        args.kcoverage = int(args.cameranum)
        if args.solver == 'bip' and args.bip_coverage_mode != 'kcoverage':
            print(
                "require_all_cameras_coverage=true: switching BIP coverage mode "
                f"from {args.bip_coverage_mode} to kcoverage."
            )
            args.bip_coverage_mode = 'kcoverage'
    if not args.camera_constraint_enable:
        raise ValueError("camera_init_strategy requires camera_constraint_enable=true")
    if args.decay != 1e-4 and args.pose_lr_decay == 0.95:
        args.pose_lr_decay = args.decay
    if args.field_hidden_dim % args.field_num_heads != 0:
        raise ValueError("field_hidden_dim must be divisible by field_num_heads")
    if args.field_fit_steps < 1:
        raise ValueError("field_fit_steps must be at least 1")
    if args.field_fit_log_interval < 1:
        raise ValueError("field_fit_log_interval must be at least 1")
    if args.field_early_stop_patience < 1:
        raise ValueError("field_early_stop_patience must be at least 1")
    if args.field_early_stop_min_steps < 1:
        raise ValueError("field_early_stop_min_steps must be at least 1")
    if args.pose_lr_decay <= 0:
        raise ValueError("pose_lr_decay must be positive")
    if args.neof_pose_log_interval < 0:
        raise ValueError("neof_pose_log_interval must be non-negative")
    if args.non_gradient_reset_interval < 1:
        raise ValueError("non_gradient_reset_interval must be at least 1")
    if args.reset_search_trials < 0:
        raise ValueError("reset_search_trials must be non-negative")
    if args.reset_search_top_k < 1:
        raise ValueError("reset_search_top_k must be at least 1")
    if args.reset_search_angle_weight < 0:
        raise ValueError("reset_search_angle_weight must be non-negative")
    if args.epoch_checkpoint_interval < 0:
        raise ValueError("epoch_checkpoint_interval must be non-negative")
    if args.epoch_checkpoint_epochs is not None and any(epoch < 1 for epoch in args.epoch_checkpoint_epochs):
        raise ValueError("epoch_checkpoint_epochs must contain positive epoch numbers")
    if args.occupancy_map_enable and args.occupancy_map_mode != 'free_space_box' and not args.occupancy_map_file:
        raise ValueError("occupancy_map_file must be provided when occupancy_map_enable=true unless occupancy_map_mode=free_space_box")
    if args.occupancy_map_enable and args.occupancy_map_mode == 'free_space_box':
        if args.occupancy_box_min is None or args.occupancy_box_max is None:
            raise ValueError("occupancy_box_min and occupancy_box_max must be provided for occupancy_map_mode=free_space_box")
        if len(args.occupancy_box_min) != 3 or len(args.occupancy_box_max) != 3:
            raise ValueError("occupancy_box_min and occupancy_box_max must each contain exactly 3 values")
        if any(b <= a for a, b in zip(args.occupancy_box_min, args.occupancy_box_max)):
            raise ValueError("occupancy_box_max must be strictly greater than occupancy_box_min on every axis")
    if any(value <= 0 for value in args.occupancy_map_sigma_xyz):
        raise ValueError("occupancy_map_sigma_xyz must be positive on every axis")
    if args.occupancy_map_floor < 0:
        raise ValueError("occupancy_map_floor must be non-negative")
    if args.occupancy_map_route_step <= 0:
        raise ValueError("occupancy_map_route_step must be positive")
    if args.occupancy_vis_clip_percentile <= 0 or args.occupancy_vis_clip_percentile > 100:
        raise ValueError("occupancy_vis_clip_percentile must be in (0, 100]")
    if args.occupancy_vis_point_size <= 0:
        raise ValueError("occupancy_vis_point_size must be positive")
    if args.occupancy_vis_route_radius <= 0:
        raise ValueError("occupancy_vis_route_radius must be positive")
    if args.occupancy_vis_route_arrow_radius <= 0:
        raise ValueError("occupancy_vis_route_arrow_radius must be positive")
    if args.occupancy_vis_route_arrow_length <= 0:
        raise ValueError("occupancy_vis_route_arrow_length must be positive")
    if args.occupancy_vis_volume_step <= 0:
        raise ValueError("occupancy_vis_volume_step must be positive")
    if args.occupancy_box_grid_step <= 0:
        raise ValueError("occupancy_box_grid_step must be positive")
    if args.occupancy_gaussian_sigma_xyz is not None and any(value <= 0 for value in args.occupancy_gaussian_sigma_xyz):
        raise ValueError("occupancy_gaussian_sigma_xyz must be positive on every axis")
    if args.model_physical_height is not None and args.model_physical_height <= 0:
        raise ValueError("model_physical_height must be positive when provided")
    if args.coverage_vis_point_size <= 0:
        raise ValueError("coverage_vis_point_size must be positive")
    if args.visual_summary_columns < 1:
        raise ValueError("visual_summary_columns must be at least 1")
    if args.occupancy_hotspot_fraction <= 0 or args.occupancy_hotspot_fraction > 1:
        raise ValueError("occupancy_hotspot_fraction must be in (0, 1]")
    if args.visibility_depth_temperature <= 0 or args.visibility_fov_temperature <= 0 or args.visibility_normal_temperature <= 0:
        raise ValueError("soft visibility temperatures must be positive")
    if args.sampling_quality_temperature <= 0:
        raise ValueError("sampling_quality_temperature must be positive")
    if args.sampling_quality_topk < 1:
        raise ValueError("sampling_quality_topk must be at least 1")
    if args.sampling_quality_min_projected_voxel_px <= 0:
        raise ValueError("sampling_quality_min_projected_voxel_px must be positive")
    if args.reset_random_samples_per_voxel < 1:
        raise ValueError("reset_random_samples_per_voxel must be at least 1")
    if args.bip_candidate_position_step <= 0:
        raise ValueError("bip_candidate_position_step must be positive")
    if args.bip_target_count < 1:
        raise ValueError("bip_target_count must be at least 1")
    if args.bip_max_candidate_positions < 0:
        raise ValueError("bip_max_candidate_positions must be non-negative")
    if args.bip_max_base_candidates < 0:
        raise ValueError("bip_max_base_candidates must be non-negative")
    if args.bip_pair_candidate_limit < 0:
        raise ValueError("bip_pair_candidate_limit must be non-negative")
    if args.bip_min_visible_points < 0:
        raise ValueError("bip_min_visible_points must be non-negative")
    if args.bip_min_visible_fraction < 0 or args.bip_min_visible_fraction > 1:
        raise ValueError("bip_min_visible_fraction must be in [0, 1]")
    if args.bip_time_limit <= 0:
        raise ValueError("bip_time_limit must be positive")
    if args.bip_cpsat_weight_scale <= 0:
        raise ValueError("bip_cpsat_weight_scale must be positive")
    if args.bip_cpsat_num_workers < 0:
        raise ValueError("bip_cpsat_num_workers must be non-negative")
    if args.bip_min_triangulation_angle_deg < 0 or args.bip_max_triangulation_angle_deg > 180:
        raise ValueError("BIP triangulation angle limits must stay within [0, 180] degrees")
    if args.bip_min_triangulation_angle_deg > args.bip_max_triangulation_angle_deg:
        raise ValueError("bip_min_triangulation_angle_deg must be <= bip_max_triangulation_angle_deg")
    if args.bip_max_pair_variables < 0:
        raise ValueError("bip_max_pair_variables must be non-negative")
    if args.camera_vis_scale <= 0:
        raise ValueError("camera_vis_scale must be positive")
    if args.camera_optical_axis_length is not None and args.camera_optical_axis_length <= 0:
        raise ValueError("camera_optical_axis_length must be positive when provided")
    if args.camera_line_radius is not None and args.camera_line_radius <= 0:
        raise ValueError("camera_line_radius must be positive when provided")
    if args.world_axis_length is not None and args.world_axis_length <= 0:
        raise ValueError("world_axis_length must be positive when provided")
    if args.world_axis_tick_step <= 0:
        raise ValueError("world_axis_tick_step must be positive")
    if args.world_axis_tick_size is not None and args.world_axis_tick_size <= 0:
        raise ValueError("world_axis_tick_size must be positive when provided")
    if args.world_axis_radius is not None and args.world_axis_radius <= 0:
        raise ValueError("world_axis_radius must be positive when provided")
    if args.world_axis_label_size is not None and args.world_axis_label_size <= 0:
        raise ValueError("world_axis_label_size must be positive when provided")
    #######################filePath###########################################
    tensorboarddir="tensorboardModel/"
    modelpath="modelpath/"+args.path
    pcdpath=resolve_output_dir("resultModel", args.path)
    posepath=os.path.join(pcdpath, "pose") + os.sep
    vispath=os.path.join(pcdpath,"visualization")
    if args.clear_previous_results and os.path.exists(pcdpath):
        shutil.rmtree(pcdpath)
        print(f"Cleared previous results: {pcdpath}")
    if os.path.exists(posepath)==False:
        os.makedirs(posepath)
    if os.path.exists(vispath)==False:
        os.makedirs(vispath)
    tensorboardpath=tensorboarddir+args.path
    if os.path.exists(tensorboardpath)==False:
        os.makedirs(tensorboardpath) 
    writer=SummaryWriter(tensorboardpath)
    run_info = {
        "input_path": original_path,
        "resolved_path": args.path,
        "solver": args.solver,
        "camera_constraint_shape": args.camera_constraint_shape,
        "timestamp_output": bool(args.timestamp_output),
        "run_timestamp": args.path.rstrip(os.sep).split("_")[-1] if args.timestamp_output else args.run_timestamp,
        "require_all_cameras_coverage": bool(args.require_all_cameras_coverage),
        "effective_kcoverage": int(args.kcoverage),
        "non_gradient_reset_enable": bool(args.non_gradient_reset_enable),
        "non_gradient_reset_interval": int(args.non_gradient_reset_interval),
        "reset_search_enable": bool(args.reset_search_enable),
        "reset_search_trials": int(args.reset_search_trials),
        "reset_search_top_k": int(args.reset_search_top_k),
        "reset_search_score": args.reset_search_score,
        "reset_search_angle_weight": float(args.reset_search_angle_weight),
        "reset_search_save_all": bool(args.reset_search_save_all),
        "epoch_checkpoint_interval": int(args.epoch_checkpoint_interval),
        "epoch_checkpoint_epochs": args.epoch_checkpoint_epochs,
        "epoch_checkpoint_mode": args.epoch_checkpoint_mode,
        "epoch_checkpoint_save_reset": bool(args.epoch_checkpoint_save_reset),
    }
    with open(os.path.join(pcdpath, "run_info.json"), "w", encoding="utf-8") as handle:
        json.dump(run_info, handle, indent=2)
    if args.optimization_key_log_name:
        args.optimization_key_log_path = os.path.join(pcdpath, args.optimization_key_log_name)
        with open(args.optimization_key_log_path, "w", encoding="utf-8") as handle:
            handle.write("Optimization key log\n")
            handle.write(f"result_dir: {pcdpath}\n")
            handle.write(f"solver: {args.solver}\n")
            handle.write(f"camera_constraint_shape: {args.camera_constraint_shape}\n")
            handle.write(f"target_visibility_mode: {args.target_visibility_mode}\n")
            handle.write(f"require_all_cameras_coverage: {bool(args.require_all_cameras_coverage)}\n")
            handle.write(f"effective_kcoverage: {int(args.kcoverage)}\n")
            handle.write(f"non_gradient_reset_enable: {bool(args.non_gradient_reset_enable)}\n")
            handle.write(f"non_gradient_reset_interval: {int(args.non_gradient_reset_interval)}\n")
            handle.write(f"reset_search_enable: {bool(args.reset_search_enable)}\n")
            handle.write(f"reset_search_trials: {int(args.reset_search_trials)}\n")
            handle.write(f"reset_search_top_k: {int(args.reset_search_top_k)}\n")
            handle.write(f"reset_search_score: {args.reset_search_score}\n")
            handle.write(f"epoch_checkpoint_interval: {int(args.epoch_checkpoint_interval)}\n")
            handle.write(f"epoch_checkpoint_epochs: {args.epoch_checkpoint_epochs}\n")
            handle.write(f"epoch_checkpoint_mode: {args.epoch_checkpoint_mode}\n")
            handle.write(f"epoch_checkpoint_save_reset: {bool(args.epoch_checkpoint_save_reset)}\n")
            handle.write("\n")
    else:
        args.optimization_key_log_path = None
    print(f"Resolved result directory: {pcdpath}")
    print(f"Visualization outputs will be written under: {vispath}")
    if args.optimization_key_log_path:
        print(f"Key optimization log: {args.optimization_key_log_path}")
    ###########################################################################
    pcd,voxelnormals,occupancy_weights,occupancy_info,minbound,camerapose,scale,geometry_metadata=Initdatafromrandom(args)
    if pcd is None:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(voxelnormals[:, :3])
        if voxelnormals.shape[1] >= 6:
            pcd.normals = o3d.utility.Vector3dVector(voxelnormals[:, 3:6])
    print(f"Camera rig | {len(args.camera_models)} cameras configured")
    for idx, camera_model in enumerate(args.camera_models):
        print(
            f"  Cam {idx:02d} | {camera_model['name']} | "
            f"image={int(round(camera_model['image_width']))}x{int(round(camera_model['image_height']))} | "
            f"fx={camera_model['fx']:.1f}, fy={camera_model['fy']:.1f} | "
            f"min_voxel_px={camera_model['min_projected_voxel_px']:.2f}"
        )
    print(
        "World-scale geometry | "
        f"target height={geometry_metadata['model_physical_height']:.3f} m | "
        f"source-to-world scale={geometry_metadata['world_scale']:.4f}"
    )
    geometry_payload = {
        "voxelnormals": voxelnormals,
        "scale": np.array(scale[0]),
        "center": np.array(scale[1]),
        "model_physical_height": np.array(geometry_metadata["model_physical_height"]),
        "world_scale": np.array(geometry_metadata["world_scale"]),
        "world_coordinate_system": np.array(int(geometry_metadata.get("world_coordinate_system", True))),
        "camera_models_json": np.array(json.dumps(args.camera_models)),
    }
    if args.occupancy_map_enable:
        geometry_payload["occupancy_weights"] = occupancy_weights
        geometry_payload["occupancy_mode"] = np.array(args.occupancy_map_mode)
        if "distribution_mode" in occupancy_info:
            geometry_payload["occupancy_distribution_mode"] = np.array(occupancy_info["distribution_mode"])
        if "route_points" in occupancy_info:
            geometry_payload["occupancy_route_points"] = occupancy_info["route_points"]
        if "route_weights" in occupancy_info:
            geometry_payload["occupancy_route_weights"] = occupancy_info["route_weights"]
        print(
            "Occupancy map enabled | "
            f"mode={occupancy_info['mode']} | "
            f"source={occupancy_info['path']} | "
            f"normalized range=[{occupancy_info['min']:.4f}, {occupancy_info['max']:.4f}]"
        )
    occupancy_vis_points = voxelnormals[:, :3]
    occupancy_vis_weights = occupancy_weights
    if (
        args.occupancy_map_enable
        and args.show_occupancy_map
        and args.occupancy_vis_mode == 'volume'
        and occupancy_info.get("mode") == "route_kde"
        and occupancy_info.get("route_points") is not None
        and occupancy_info.get("route_weights") is not None
    ):
        occupancy_vis_points, occupancy_vis_weights = build_swept_object_visualization_cloud(
            voxelnormals,
            occupancy_info["route_points"],
            occupancy_info["route_weights"],
            args.occupancy_vis_volume_step,
        )
    np.savez(os.path.join(pcdpath, "geometry_data.npz"), **geometry_payload)
    if args.solver == 'neof':
        camlayopt = CameraLayerOpt(args,voxelnormals,occupancy_weights,minbound,scale,posepath,writer,geometry_metadata=geometry_metadata)
    elif args.solver == 'bip':
        camlayopt = BIPCameraOpt(args,voxelnormals,occupancy_weights,minbound,scale,posepath,writer,geometry_metadata=geometry_metadata)
    else:
        raise ValueError(f"Unsupported solver: {args.solver}")
    position=camerapose[:,:3]
    rotation=compute_rotation_matrix_from_ortho6d(camerapose[:,3:])
    initial_position_np = position.detach().cpu().numpy()
    initial_rotation_np = rotation.detach().cpu().numpy()
    posename=posepath+str(0)+".npy"
    saveTrainingResult(posename,position,rotation,scale)
    camera_constraint_min = args.camera_constraint_min if args.camera_constraint_enable else None
    camera_constraint_max = args.camera_constraint_max if args.camera_constraint_enable else None
    show_base_geometry = not (args.occupancy_map_enable and not args.show_static_geometry_in_occupancy)
    camera_vis_kwargs = {
        "camera_models": args.camera_models,
        "camera_vis_scale": args.camera_vis_scale,
        "show_camera_optical_axis": args.show_camera_optical_axis,
        "show_camera_fov": args.show_camera_fov,
        "camera_optical_axis_length": args.camera_optical_axis_length,
        "camera_line_radius": args.camera_line_radius,
    }
    ###visualization mesh/ply with camera placement
    if args.vismode == 'interactive':
        visMesh(
            pcd,
            position.cpu().numpy(),
            rotation.cpu().numpy(),
            flag=1,
            image_path=os.path.join(vispath,"initial.png"),
            camera_constraint_shape=args.camera_constraint_shape,
            camera_constraint_min=camera_constraint_min,
            camera_constraint_max=camera_constraint_max,
            camera_constraint_data=args.camera_constraint_data,
            show_world_axes=args.show_world_axes,
            show_world_axis_tick_labels=args.show_world_axis_tick_labels,
            world_axis_length=args.world_axis_length,
            world_axis_tick_step=args.world_axis_tick_step,
            world_axis_tick_size=args.world_axis_tick_size,
            world_axis_radius=args.world_axis_radius,
            world_axis_label_size=args.world_axis_label_size,
            show_occupancy_map=args.occupancy_map_enable and args.show_occupancy_map,
            occupancy_points=occupancy_vis_points if args.occupancy_map_enable else None,
            occupancy_weights=occupancy_vis_weights if args.occupancy_map_enable else None,
            occupancy_route_points=occupancy_info.get("route_points") if args.occupancy_map_enable else None,
            show_occupancy_route=args.occupancy_map_enable and args.show_occupancy_route,
            occupancy_vis_clip_percentile=args.occupancy_vis_clip_percentile,
            occupancy_vis_point_size=args.occupancy_vis_point_size,
            occupancy_vis_route_radius=args.occupancy_vis_route_radius,
            show_occupancy_route_direction=args.occupancy_map_enable and args.show_occupancy_route_direction,
            occupancy_vis_route_arrow_radius=args.occupancy_vis_route_arrow_radius,
            occupancy_vis_route_arrow_length=args.occupancy_vis_route_arrow_length,
            scene_export_prefix=os.path.join(vispath, "initial"),
            show_base_geometry=show_base_geometry,
            **camera_vis_kwargs,
        )
    elif args.vismode == 'save':
        visMesh(
            pcd,
            position.cpu().numpy(),
            rotation.cpu().numpy(),
            image_path=os.path.join(vispath,"initial.png"),
            camera_constraint_shape=args.camera_constraint_shape,
            camera_constraint_min=camera_constraint_min,
            camera_constraint_max=camera_constraint_max,
            camera_constraint_data=args.camera_constraint_data,
            show_world_axes=args.show_world_axes,
            show_world_axis_tick_labels=args.show_world_axis_tick_labels,
            world_axis_length=args.world_axis_length,
            world_axis_tick_step=args.world_axis_tick_step,
            world_axis_tick_size=args.world_axis_tick_size,
            world_axis_radius=args.world_axis_radius,
            world_axis_label_size=args.world_axis_label_size,
            show_occupancy_map=args.occupancy_map_enable and args.show_occupancy_map,
            occupancy_points=occupancy_vis_points if args.occupancy_map_enable else None,
            occupancy_weights=occupancy_vis_weights if args.occupancy_map_enable else None,
            occupancy_route_points=occupancy_info.get("route_points") if args.occupancy_map_enable else None,
            show_occupancy_route=args.occupancy_map_enable and args.show_occupancy_route,
            occupancy_vis_clip_percentile=args.occupancy_vis_clip_percentile,
            occupancy_vis_point_size=args.occupancy_vis_point_size,
            occupancy_vis_route_radius=args.occupancy_vis_route_radius,
            show_occupancy_route_direction=args.occupancy_map_enable and args.show_occupancy_route_direction,
            occupancy_vis_route_arrow_radius=args.occupancy_vis_route_arrow_radius,
            occupancy_vis_route_arrow_length=args.occupancy_vis_route_arrow_length,
            scene_export_prefix=os.path.join(vispath, "initial"),
            show_base_geometry=show_base_geometry,
            **camera_vis_kwargs,
        )
    #############################################################################
    start_line = f"Optimization timing started | solver={args.solver}"
    print(start_line)
    append_optimization_key_log(args, start_line)
    optimization_start_time = time.perf_counter()
    position,rotation = camlayopt.opt(camerapose)
    optimization_elapsed_s = time.perf_counter() - optimization_start_time
    finish_line = (
        "Optimization timing finished | "
        f"solver={args.solver} | elapsed={format_runtime(optimization_elapsed_s)} "
        f"({optimization_elapsed_s:.3f}s)"
    )
    print(finish_line)
    append_optimization_key_log(args, finish_line)
    optimized_position_np = np.asarray(position, dtype=float)
    optimized_rotation_np = np.asarray(rotation, dtype=float)
    if args.vismode == 'interactive':
        visMesh(
            pcd,
            position,
            rotation,
            flag=1,
            image_path=os.path.join(vispath,"optimized.png"),
            camera_constraint_shape=args.camera_constraint_shape,
            camera_constraint_min=camera_constraint_min,
            camera_constraint_max=camera_constraint_max,
            camera_constraint_data=args.camera_constraint_data,
            show_world_axes=args.show_world_axes,
            show_world_axis_tick_labels=args.show_world_axis_tick_labels,
            world_axis_length=args.world_axis_length,
            world_axis_tick_step=args.world_axis_tick_step,
            world_axis_tick_size=args.world_axis_tick_size,
            world_axis_radius=args.world_axis_radius,
            world_axis_label_size=args.world_axis_label_size,
            show_occupancy_map=args.occupancy_map_enable and args.show_occupancy_map,
            occupancy_points=occupancy_vis_points if args.occupancy_map_enable else None,
            occupancy_weights=occupancy_vis_weights if args.occupancy_map_enable else None,
            occupancy_route_points=occupancy_info.get("route_points") if args.occupancy_map_enable else None,
            show_occupancy_route=args.occupancy_map_enable and args.show_occupancy_route,
            occupancy_vis_clip_percentile=args.occupancy_vis_clip_percentile,
            occupancy_vis_point_size=args.occupancy_vis_point_size,
            occupancy_vis_route_radius=args.occupancy_vis_route_radius,
            show_occupancy_route_direction=args.occupancy_map_enable and args.show_occupancy_route_direction,
            occupancy_vis_route_arrow_radius=args.occupancy_vis_route_arrow_radius,
            occupancy_vis_route_arrow_length=args.occupancy_vis_route_arrow_length,
            scene_export_prefix=os.path.join(vispath, "optimized"),
            show_base_geometry=show_base_geometry,
            **camera_vis_kwargs,
        )
    elif args.vismode == 'save':
        visMesh(
            pcd,
            position,
            rotation,
            image_path=os.path.join(vispath,"optimized.png"),
            camera_constraint_shape=args.camera_constraint_shape,
            camera_constraint_min=camera_constraint_min,
            camera_constraint_max=camera_constraint_max,
            camera_constraint_data=args.camera_constraint_data,
            show_world_axes=args.show_world_axes,
            show_world_axis_tick_labels=args.show_world_axis_tick_labels,
            world_axis_length=args.world_axis_length,
            world_axis_tick_step=args.world_axis_tick_step,
            world_axis_tick_size=args.world_axis_tick_size,
            world_axis_radius=args.world_axis_radius,
            world_axis_label_size=args.world_axis_label_size,
            show_occupancy_map=args.occupancy_map_enable and args.show_occupancy_map,
            occupancy_points=occupancy_vis_points if args.occupancy_map_enable else None,
            occupancy_weights=occupancy_vis_weights if args.occupancy_map_enable else None,
            occupancy_route_points=occupancy_info.get("route_points") if args.occupancy_map_enable else None,
            show_occupancy_route=args.occupancy_map_enable and args.show_occupancy_route,
            occupancy_vis_clip_percentile=args.occupancy_vis_clip_percentile,
            occupancy_vis_point_size=args.occupancy_vis_point_size,
            occupancy_vis_route_radius=args.occupancy_vis_route_radius,
            show_occupancy_route_direction=args.occupancy_map_enable and args.show_occupancy_route_direction,
            occupancy_vis_route_arrow_radius=args.occupancy_vis_route_arrow_radius,
            occupancy_vis_route_arrow_length=args.occupancy_vis_route_arrow_length,
            scene_export_prefix=os.path.join(vispath, "optimized"),
            show_base_geometry=show_base_geometry,
            **camera_vis_kwargs,
        )
    route_points = occupancy_info.get("route_points") if args.occupancy_map_enable else None
    comparison_route_geometries = []
    if args.occupancy_map_enable and args.show_occupancy_route and route_points is not None:
        comparison_route_geometries = getOccupancyRouteVis(
            route_points,
            radius=args.occupancy_vis_route_radius,
            show_direction=args.show_occupancy_route_direction,
            arrow_radius=args.occupancy_vis_route_arrow_radius,
            arrow_length=args.occupancy_vis_route_arrow_length,
        )
    comparison_constraint_geometries = []
    if camera_constraint_min is not None and camera_constraint_max is not None:
        comparison_constraint_geometries = getCameraConstraintVis(
            args.camera_constraint_shape,
            camera_constraint_min,
            camera_constraint_max,
            constraint_data=args.camera_constraint_data,
        )
    comparison_axis_geometries = []
    if args.show_world_axes and pcd is not None:
        comparison_axis_geometries = getWorldAxesVis(
            pcd,
            np.concatenate((initial_position_np, optimized_position_np), axis=0),
            constraint_min=camera_constraint_min,
            constraint_max=camera_constraint_max,
            axis_length=args.world_axis_length,
            tick_step=args.world_axis_tick_step,
            tick_size=args.world_axis_tick_size,
            axis_radius=args.world_axis_radius,
            show_tick_labels=args.show_world_axis_tick_labels,
            label_size=args.world_axis_label_size,
        )
    occupancy_cloud = None
    if args.occupancy_map_enable and args.show_occupancy_map:
        occupancy_cloud = getOccupancyPointCloud(
            occupancy_vis_points,
            occupancy_vis_weights,
            clip_percentile=args.occupancy_vis_clip_percentile,
        )
        saveOccupancyColorLegend(
            os.path.join(vispath, "occupancy_legend.png"),
            clip_percentile=args.occupancy_vis_clip_percentile,
        )
    summary_specs = [
        ("Initial layout", os.path.join(vispath, "initial.png")),
        ("Optimized layout", os.path.join(vispath, "optimized.png")),
    ]
    if args.occupancy_map_enable and args.show_occupancy_map:
        summary_specs.append(("Occupancy legend", os.path.join(vispath, "occupancy_legend.png")))
    comparison_camera_ray_length = resolve_camera_ray_length(
        pcd,
        np.concatenate((initial_position_np, optimized_position_np), axis=0),
        constraint_min=camera_constraint_min,
        constraint_max=camera_constraint_max,
        ray_length=args.camera_optical_axis_length,
    )
    camera_overlay_kwargs = {
        "camera_scale": args.camera_vis_scale,
        "show_optical_axis": args.show_camera_optical_axis,
        "show_fov": args.show_camera_fov,
        "optical_axis_length": comparison_camera_ray_length,
        "line_radius": args.camera_line_radius,
    }

    if args.save_comparison_visualization and pcd is not None:
        comparison_camera_geometries = []
        comparison_camera_geometries.extend(
            build_camera_geometries(
                initial_position_np,
                initial_rotation_np,
                color=np.array([[0.2, 0.55, 1.0]]),
                camera_models=args.camera_models,
                **camera_overlay_kwargs,
            )
        )
        comparison_camera_geometries.extend(
            build_camera_geometries(
                optimized_position_np,
                optimized_rotation_np,
                color=np.array([[1.0, 0.35, 0.1]]),
                camera_models=args.camera_models,
                **camera_overlay_kwargs,
            )
        )
        comparison_camera_geometries.extend(
            getCameraTransitionVis(initial_position_np, optimized_position_np)
        )
        visGeometryScene(
            pcd,
            overlay_geometries=
                comparison_camera_geometries
                + comparison_constraint_geometries
                + comparison_axis_geometries
                + comparison_route_geometries,
            image_path=os.path.join(vispath, "comparison.png"),
            scene_export_prefix=os.path.join(vispath, "comparison"),
            point_size=args.occupancy_vis_point_size,
            occupancy_cloud=occupancy_cloud,
            show_geometry=show_base_geometry,
        )
        summary_specs.append(("Layout comparison", os.path.join(vispath, "comparison.png")))

    initial_visibility = None
    optimized_visibility = None
    if args.save_coverage_visualization or args.save_camera_responsibility_visualization:
        if args.save_coverage_visualization:
            _, initial_visibility = voxel_model(
                args,
                voxelnormals,
                initial_rotation_np,
                initial_position_np,
            )
        _, optimized_visibility = voxel_model(
            args,
            voxelnormals,
            optimized_rotation_np,
            optimized_position_np,
        )

    if args.save_coverage_visualization:
        initial_coverage = np.sum(initial_visibility, axis=1)
        optimized_coverage = np.sum(optimized_visibility, axis=1)
        coverage_delta = optimized_coverage - initial_coverage

        initial_coverage_cloud = getCoveragePointCloud(voxelnormals[:, :3], initial_coverage, args.kcoverage)
        optimized_coverage_cloud = getCoveragePointCloud(voxelnormals[:, :3], optimized_coverage, args.kcoverage)
        coverage_delta_cloud = getCoverageDeltaPointCloud(voxelnormals[:, :3], coverage_delta)

        initial_coverage_overlays = (
            build_camera_geometries(
                initial_position_np,
                initial_rotation_np,
                color=np.array([[0.2, 0.55, 1.0]]),
                camera_models=args.camera_models,
                **camera_overlay_kwargs,
            )
            + comparison_constraint_geometries
            + comparison_axis_geometries
            + comparison_route_geometries
        )
        optimized_coverage_overlays = (
            build_camera_geometries(
                optimized_position_np,
                optimized_rotation_np,
                color=np.array([[1.0, 0.35, 0.1]]),
                camera_models=args.camera_models,
                **camera_overlay_kwargs,
            )
            + comparison_constraint_geometries
            + comparison_axis_geometries
            + comparison_route_geometries
        )
        delta_coverage_overlays = (
            build_camera_geometries(
                initial_position_np,
                initial_rotation_np,
                color=np.array([[0.2, 0.55, 1.0]]),
                camera_models=args.camera_models,
                **camera_overlay_kwargs,
            )
            + build_camera_geometries(
                optimized_position_np,
                optimized_rotation_np,
                color=np.array([[1.0, 0.35, 0.1]]),
                camera_models=args.camera_models,
                **camera_overlay_kwargs,
            )
            + getCameraTransitionVis(initial_position_np, optimized_position_np)
            + comparison_constraint_geometries
            + comparison_axis_geometries
            + comparison_route_geometries
        )

        visGeometryScene(
            initial_coverage_cloud,
            overlay_geometries=initial_coverage_overlays,
            image_path=os.path.join(vispath, "initial_coverage.png"),
            scene_export_prefix=os.path.join(vispath, "initial_coverage"),
            point_size=args.coverage_vis_point_size,
        )
        visGeometryScene(
            optimized_coverage_cloud,
            overlay_geometries=optimized_coverage_overlays,
            image_path=os.path.join(vispath, "optimized_coverage.png"),
            scene_export_prefix=os.path.join(vispath, "optimized_coverage"),
            point_size=args.coverage_vis_point_size,
        )
        visGeometryScene(
            coverage_delta_cloud,
            overlay_geometries=delta_coverage_overlays,
            image_path=os.path.join(vispath, "coverage_delta.png"),
            scene_export_prefix=os.path.join(vispath, "coverage_delta"),
            point_size=args.coverage_vis_point_size,
        )
        summary_specs.extend(
            [
                ("Initial coverage", os.path.join(vispath, "initial_coverage.png")),
                ("Optimized coverage", os.path.join(vispath, "optimized_coverage.png")),
                ("Coverage delta", os.path.join(vispath, "coverage_delta.png")),
            ]
        )

    per_camera_summary_specs = []
    if args.save_camera_responsibility_visualization and optimized_visibility is not None:
        hotspot_mask = buildHotspotMask(occupancy_weights, args.occupancy_hotspot_fraction) if args.occupancy_map_enable else np.any(optimized_visibility > 0, axis=1)
        camera_palette = buildCameraColorPalette(len(optimized_position_np))
        responsibility_labels, _ = computeCameraResponsibility(
            voxelnormals,
            optimized_position_np,
            optimized_visibility,
            hotspot_mask=hotspot_mask,
        )
        responsibility_cloud = getResponsibilityPointCloud(
            voxelnormals[:, :3],
            responsibility_labels,
            camera_palette,
            show_background=False,
        )
        responsibility_overlays = (
            build_camera_geometries(
                optimized_position_np,
                optimized_rotation_np,
                color=camera_palette,
                camera_models=args.camera_models,
                **camera_overlay_kwargs,
            )
            + comparison_constraint_geometries
            + comparison_axis_geometries
            + comparison_route_geometries
        )
        visGeometryScene(
            responsibility_cloud,
            overlay_geometries=responsibility_overlays,
            image_path=os.path.join(vispath, "camera_responsibility.png"),
            scene_export_prefix=os.path.join(vispath, "camera_responsibility"),
            point_size=args.coverage_vis_point_size,
        )
        summary_specs.append(("Hotspot responsibility", os.path.join(vispath, "camera_responsibility.png")))

        if args.save_per_camera_responsibility_visualization:
            for camera_idx in range(len(optimized_position_np)):
                camera_mask = responsibility_labels == camera_idx
                if not np.any(camera_mask):
                    continue
                camera_points = voxelnormals[camera_mask, :3]
                if args.occupancy_map_enable:
                    camera_cloud = getOccupancyPointCloud(
                        camera_points,
                        occupancy_weights[camera_mask],
                        clip_percentile=args.occupancy_vis_clip_percentile,
                    )
                else:
                    camera_cloud = getResponsibilityPointCloud(
                        camera_points,
                        np.zeros(len(camera_points), dtype=int),
                        camera_palette[camera_idx : camera_idx + 1],
                        show_background=False,
                    )
                image_path = os.path.join(vispath, f"camera_{camera_idx:02d}_responsibility.png")
                scene_prefix = os.path.join(vispath, f"camera_{camera_idx:02d}_responsibility")
                camera_overlays = (
                    build_camera_geometries(
                        optimized_position_np[camera_idx : camera_idx + 1],
                        optimized_rotation_np[camera_idx : camera_idx + 1],
                        color=camera_palette[camera_idx : camera_idx + 1],
                        camera_models=args.camera_models[camera_idx : camera_idx + 1],
                        **camera_overlay_kwargs,
                    )
                    + comparison_constraint_geometries
                    + comparison_axis_geometries
                    + comparison_route_geometries
                )
                visGeometryScene(
                    camera_cloud,
                    overlay_geometries=camera_overlays,
                    image_path=image_path,
                    scene_export_prefix=scene_prefix,
                    point_size=args.coverage_vis_point_size,
                )
                per_camera_summary_specs.append((f"Camera {camera_idx}", image_path))

    if args.save_visual_summary:
        summary_title = f"{os.path.basename(os.path.normpath(pcdpath))} visual summary"
        createImageSummaryMontage(
            summary_specs,
            os.path.join(vispath, "visual_summary.png"),
            title=summary_title,
            columns=args.visual_summary_columns,
        )
        if len(per_camera_summary_specs) > 0:
            createImageSummaryMontage(
                per_camera_summary_specs,
                os.path.join(vispath, "camera_responsibility_summary.png"),
                title="Per-camera hotspot responsibility",
                columns=args.visual_summary_columns,
            )
