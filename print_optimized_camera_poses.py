import argparse
import os
import re

import numpy as np
from scipy.spatial.transform import Rotation as SciPyRotation

from camera_constraints import CAMERA_CONSTRAINT_SHAPES
from config_utils import parse_args_with_json_config
from mocap_config import resolve_mocap_templates
from run_paths import apply_run_naming_to_path, find_latest_timestamped_run


def find_latest_timestamped_run_with_pose(base_dir, relative_path_without_timestamp, pose_filename):
    parent_dir, leaf_dir = os.path.split(os.path.normpath(relative_path_without_timestamp))
    search_dir = os.path.join(base_dir, parent_dir) if parent_dir else base_dir
    if not os.path.isdir(search_dir):
        return None

    pattern = re.compile(rf"^{re.escape(leaf_dir)}_(\d{{14}})$")
    matches = []
    for candidate in os.listdir(search_dir):
        match = pattern.match(candidate)
        if not match:
            continue
        candidate_relative = os.path.join(parent_dir, candidate) if parent_dir else candidate
        pose_path = os.path.join(base_dir, candidate_relative, "pose", pose_filename)
        if os.path.exists(pose_path):
            matches.append((match.group(1), candidate_relative))
    if not matches:
        return None
    _, latest_relative = max(matches)
    return latest_relative


def resolve_result_dir(args, pose_filename):
    if args.result_dir:
        return args.result_dir

    if args.run_timestamp == "latest":
        base_path = apply_run_naming_to_path(
            args.path,
            args.solver,
            args.camera_constraint_shape,
            run_timestamp=None,
            append_timestamp=False,
        )
        latest_path = find_latest_timestamped_run_with_pose("resultModel", base_path, pose_filename)
        if latest_path is None:
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


def world_remap_matrix(mode):
    if mode == "identity":
        return np.eye(3, dtype=float)
    if mode == "yup":
        angle = -0.5 * np.pi
        c = np.cos(angle)
        s = np.sin(angle)
        return np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, c, -s],
                [0.0, s, c],
            ],
            dtype=float,
        )
    raise ValueError(f"Unsupported world remap: {mode}")


def convert_pose(row, rotation_convention, world_remap):
    camera_position = np.asarray(row[:3], dtype=float)
    rotation_world_to_camera = np.asarray(row[3:], dtype=float).reshape(3, 3)

    if rotation_convention == "neof":
        output_position = camera_position
        output_rotation = rotation_world_to_camera
    elif rotation_convention == "blender":
        remap = world_remap_matrix(world_remap)
        opencv_to_blender_camera = np.diag([1.0, -1.0, -1.0])
        output_position = remap @ camera_position
        output_rotation = remap @ rotation_world_to_camera.T @ opencv_to_blender_camera
    else:
        raise ValueError(f"Unsupported rotation convention: {rotation_convention}")

    euler_xyz_deg = SciPyRotation.from_matrix(output_rotation).as_euler("xyz", degrees=True)
    return output_position, euler_xyz_deg


def format_values(values, precision):
    return " ".join(f"{float(value):.{precision}f}" for value in values)


def selected_pose_files(args, result_dir):
    if args.pose_file:
        return [("custom", args.pose_file)]
    if args.pose_name == "both":
        return [
            ("initial", os.path.join(result_dir, "pose", "0.npy")),
            ("optimized", os.path.join(result_dir, "pose", "after.npy")),
        ]
    pose_filename = "after.npy" if args.pose_name == "optimized" else "0.npy"
    return [(args.pose_name, os.path.join(result_dir, "pose", pose_filename))]


def print_pose_file(label, pose_path, args):
    if not os.path.exists(pose_path):
        raise FileNotFoundError(f"Pose file not found: {pose_path}")

    pose = np.load(pose_path, allow_pickle=True)
    if pose.ndim != 2 or pose.shape[1] != 12:
        raise ValueError(f"Expected pose array with shape (N, 12), got {pose.shape}")

    if args.camera_index is None:
        camera_indices = list(range(pose.shape[0]))
    else:
        camera_indices = args.camera_index

    for camera_index in camera_indices:
        if camera_index < 0 or camera_index >= pose.shape[0]:
            raise IndexError(f"camera_index {camera_index} out of range [0, {pose.shape[0] - 1}]")
        position, rotation = convert_pose(
            pose[camera_index],
            rotation_convention=args.rotation_convention,
            world_remap=args.world_remap,
        )
        line = (
            f"--camera-position {format_values(position, args.precision)} "
            f"--camera-rotation {format_values(rotation, args.precision)}"
        )
        if not args.no_labels:
            line = f"{label}_camera_{camera_index:02d}: {line}"
        print(line)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Print initial and optimized camera poses as '--camera-position x y z "
            "--camera-rotation rx ry rz' command fragments."
        )
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--result_dir", type=str, default=None)
    parser.add_argument("--path", type=str, default="random/mocap/{mocap_sequence}/")
    parser.add_argument("--mocap_sequence", type=str, default=None)
    parser.add_argument("--mocap_data_dir", type=str, default="mocap_data")
    parser.add_argument("--solver", type=str, choices=["neof", "bip"], default="neof")
    parser.add_argument(
        "--camera_constraint_shape",
        type=str,
        choices=CAMERA_CONSTRAINT_SHAPES,
        default="box",
    )
    parser.add_argument("--run_timestamp", type=str, default="latest")
    parser.add_argument("--pose_name", type=str, choices=["initial", "optimized", "both"], default="both")
    parser.add_argument("--pose_file", type=str, default=None)
    parser.add_argument("--camera_index", type=int, action="append", default=None)
    parser.add_argument(
        "--rotation_convention",
        type=str,
        choices=["neof", "blender"],
        default="neof",
        help=(
            "neof prints Euler xyz from the saved world-to-camera matrix. "
            "blender converts OpenCV camera axes to Blender camera object orientation."
        ),
    )
    parser.add_argument(
        "--world_remap",
        type=str,
        choices=["identity", "yup"],
        default="identity",
        help="Only used for --rotation_convention blender.",
    )
    parser.add_argument("--precision", type=int, default=6)
    parser.add_argument("--no_labels", action="store_true")
    args = parse_args_with_json_config(parser, allow_unknown_config_keys=True)
    args = resolve_mocap_templates(args)

    pose_filename = "after.npy" if args.pose_name in ("optimized", "both") else "0.npy"
    result_dir = resolve_result_dir(args, pose_filename)
    for label, pose_path in selected_pose_files(args, result_dir):
        print_pose_file(label, pose_path, args)


if __name__ == "__main__":
    main()
