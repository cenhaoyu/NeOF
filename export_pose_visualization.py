import argparse
import json
import os
from types import SimpleNamespace

os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

import numpy as np
import open3d as o3d

from camera_constraints import CAMERA_CONSTRAINT_SHAPES, resolve_camera_constraint_args
from config_utils import _strip_json_comments
from dataset.utils import configure_camera_models_from_args
from mocap_config import resolve_mocap_templates
from run_paths import apply_run_naming_to_path, find_latest_timestamped_run
from visualization.visual_cam_points import visMesh


DEFAULTS = {
    "path": "random/mocap/{mocap_sequence}/",
    "mocap_sequence": None,
    "mocap_data_dir": "mocap_data",
    "solver": "neof",
    "camera_constraint_enable": True,
    "camera_constraint_shape": "box",
    "image_width": 640,
    "image_height": 480,
    "fx": 320.0,
    "fy": 320.0,
    "cx": 319.5,
    "cy": 239.5,
    "camera_models": None,
    "cameranum": 1,
    "show_world_axes": True,
    "show_world_axis_tick_labels": False,
    "world_axis_length": None,
    "world_axis_tick_step": 0.5,
    "world_axis_tick_size": None,
    "world_axis_radius": None,
    "world_axis_label_size": None,
    "camera_vis_scale": 0.2,
    "show_camera_optical_axis": True,
    "show_camera_fov": True,
    "camera_optical_axis_length": None,
    "camera_line_radius": None,
    "occupancy_map_enable": False,
    "show_occupancy_map": False,
    "show_static_geometry_in_occupancy": True,
    "occupancy_vis_clip_percentile": 95.0,
    "occupancy_vis_point_size": 5.0,
    "occupancy_vis_route_radius": 0.004,
    "occupancy_vis_route_arrow_radius": 0.005,
    "occupancy_vis_route_arrow_length": 0.05,
    "show_occupancy_route": False,
    "show_occupancy_route_direction": False,
}


def load_config_namespace(config_path):
    values = dict(DEFAULTS)
    if config_path:
        with open(config_path, "r", encoding="utf-8") as handle:
            values.update(json.loads(_strip_json_comments(handle.read())))
    return SimpleNamespace(**values)


def resolve_result_dir(args):
    if args.result_dir:
        return args.result_dir

    base_path = apply_run_naming_to_path(
        args.path,
        args.solver,
        args.camera_constraint_shape,
        run_timestamp=None,
        append_timestamp=False,
    )
    if args.run_timestamp == "latest":
        latest_path = find_latest_timestamped_run("resultModel", base_path)
        if latest_path is None:
            raise FileNotFoundError(f"No timestamped run found for resultModel/{base_path}")
        return os.path.join("resultModel", latest_path)

    named_path = apply_run_naming_to_path(
        args.path,
        args.solver,
        args.camera_constraint_shape,
        run_timestamp=args.run_timestamp,
        append_timestamp=False,
    )
    return os.path.join("resultModel", named_path)


def pose_path_from_stage(result_dir, pose_stage):
    pose_dir = os.path.join(result_dir, "pose")
    if pose_stage == "initial":
        return os.path.join(pose_dir, "0.npy")
    if pose_stage == "final":
        return os.path.join(pose_dir, "after.npy")
    if pose_stage == "reset_search_selected":
        manifest_path = os.path.join(pose_dir, "epoch_checkpoints.json")
        if os.path.exists(manifest_path):
            with open(manifest_path, "r", encoding="utf-8") as handle:
                manifest = json.load(handle)
            for record in manifest.get("records", []):
                if record.get("stage") == "reset_search_selected":
                    return os.path.join(pose_dir, record["pose_file"])
        candidates = sorted(
            filename
            for filename in os.listdir(pose_dir)
            if filename.startswith("reset_search_rank_001_") and filename.endswith(".npy")
        )
        if candidates:
            return os.path.join(pose_dir, candidates[0])
    raise FileNotFoundError(f"Could not resolve pose stage {pose_stage!r} in {pose_dir}")


def load_pose(path):
    pose = np.load(path, allow_pickle=True)
    if pose.ndim != 2 or pose.shape[1] != 12:
        raise ValueError(f"Expected pose array with shape (N, 12), got {pose.shape}")
    return pose[:, :3].astype(float), pose[:, 3:].reshape(-1, 3, 3).astype(float)


def load_geometry(result_dir):
    geometry_path = os.path.join(result_dir, "geometry_data.npz")
    if not os.path.exists(geometry_path):
        raise FileNotFoundError(f"Geometry data not found: {geometry_path}")
    with np.load(geometry_path, allow_pickle=True) as data:
        voxelnormals = data["voxelnormals"].astype(float)
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(voxelnormals[:, :3])
    if voxelnormals.shape[1] >= 6:
        cloud.normals = o3d.utility.Vector3dVector(voxelnormals[:, 3:6])
    return cloud


def main():
    parser = argparse.ArgumentParser(description="Export a saved camera pose as the same Open3D scene zip used by main.py.")
    parser.add_argument("--config", type=str, default="configs/main_mocap.json")
    parser.add_argument("--result_dir", type=str, default=None)
    parser.add_argument("--path", type=str, default=None)
    parser.add_argument("--mocap_sequence", type=str, default=None)
    parser.add_argument("--mocap_data_dir", type=str, default=None)
    parser.add_argument("--solver", type=str, choices=["neof", "bip"], default=None)
    parser.add_argument("--camera_constraint_shape", type=str, choices=CAMERA_CONSTRAINT_SHAPES, default=None)
    parser.add_argument("--run_timestamp", type=str, default="latest")
    parser.add_argument("--pose_stage", type=str, choices=["initial", "reset_search_selected", "final"], default="reset_search_selected")
    parser.add_argument("--pose_file", type=str, default=None)
    parser.add_argument("--output_name", type=str, default=None)
    parsed = parser.parse_args()

    args = load_config_namespace(parsed.config)
    for name in ("path", "mocap_sequence", "mocap_data_dir", "solver", "camera_constraint_shape"):
        value = getattr(parsed, name)
        if value is not None:
            setattr(args, name, value)
    args = resolve_mocap_templates(args)
    args = resolve_camera_constraint_args(args)
    args = configure_camera_models_from_args(args)

    resolve_args = SimpleNamespace(**vars(args))
    resolve_args.result_dir = parsed.result_dir
    resolve_args.run_timestamp = parsed.run_timestamp
    result_dir = resolve_result_dir(resolve_args)
    pose_path = parsed.pose_file
    if pose_path is None:
        pose_path = pose_path_from_stage(result_dir, parsed.pose_stage)
    elif not os.path.isabs(pose_path):
        pose_path = os.path.join(result_dir, "pose", pose_path)

    position, rotation = load_pose(pose_path)
    geometry = load_geometry(result_dir)
    output_name = parsed.output_name or os.path.splitext(os.path.basename(pose_path))[0]
    vis_dir = os.path.join(result_dir, "visualization")
    os.makedirs(vis_dir, exist_ok=True)
    export_prefix = os.path.join(vis_dir, output_name)

    show_base_geometry = not (args.occupancy_map_enable and not args.show_static_geometry_in_occupancy)
    visMesh(
        geometry,
        position,
        rotation,
        image_path=export_prefix + ".png",
        camera_constraint_shape=args.camera_constraint_shape,
        camera_constraint_min=args.camera_constraint_min,
        camera_constraint_max=args.camera_constraint_max,
        camera_constraint_data=args.camera_constraint_data,
        show_world_axes=args.show_world_axes,
        show_world_axis_tick_labels=args.show_world_axis_tick_labels,
        world_axis_length=args.world_axis_length,
        world_axis_tick_step=args.world_axis_tick_step,
        world_axis_tick_size=args.world_axis_tick_size,
        world_axis_radius=args.world_axis_radius,
        world_axis_label_size=args.world_axis_label_size,
        scene_export_prefix=export_prefix,
        show_base_geometry=show_base_geometry,
        camera_models=args.camera_models,
        camera_vis_scale=args.camera_vis_scale,
        show_camera_optical_axis=args.show_camera_optical_axis,
        show_camera_fov=args.show_camera_fov,
        camera_optical_axis_length=args.camera_optical_axis_length,
        camera_line_radius=args.camera_line_radius,
    )
    print(f"Pose visualized: {pose_path}")
    print(f"Scene zip: {export_prefix}_scene.zip")
    print(f"Scene json: {export_prefix}_scene.json")
    print(f"Preview png: {export_prefix}.png")


if __name__ == "__main__":
    main()
