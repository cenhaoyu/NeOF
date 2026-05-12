import argparse
import json
import os

import numpy as np


JOINT_CENTER_SUFFIXES = ("AJC", "EJC", "HJC", "KJC", "SJC", "WJC")
DERIVED_POINT_NAMES = {"HC", "OT"}


def load_mocap_sequence(path, sequence_name=None):
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or len(payload) == 0:
        raise ValueError("mocap JSON must contain a non-empty object")
    if sequence_name is None:
        if len(payload) != 1:
            raise ValueError("--sequence is required when the JSON contains multiple sequences")
        sequence_name = next(iter(payload.keys()))
    if sequence_name not in payload:
        raise ValueError(f"sequence not found: {sequence_name}")
    frames = payload[sequence_name]
    if not isinstance(frames, list) or len(frames) == 0:
        raise ValueError(f"sequence {sequence_name} must contain at least one frame")
    return sequence_name, frames


def is_spatial_key(name):
    return "Angles" not in name and not name.endswith("_None")


def is_physical_marker_key(name):
    if not is_spatial_key(name):
        return False
    if name.startswith("mid") or name.endswith("_origin"):
        return False
    if name.endswith(JOINT_CENTER_SUFFIXES):
        return False
    if name in DERIVED_POINT_NAMES:
        return False
    return True


def select_marker_keys(frames, marker_set):
    key_sets = [set(frame.get("landmarks", {}).keys()) for frame in frames]
    common_keys = sorted(set.intersection(*key_sets))
    if marker_set == "physical":
        marker_keys = [key for key in common_keys if is_physical_marker_key(key)]
    elif marker_set == "spatial":
        marker_keys = [key for key in common_keys if is_spatial_key(key)]
    else:
        raise ValueError(f"Unsupported marker set: {marker_set}")
    if not marker_keys:
        raise ValueError("no marker keys selected")
    return marker_keys, common_keys


def extract_points(frames, marker_keys):
    frame_numbers = []
    points = np.zeros((len(frames), len(marker_keys), 3), dtype=float)
    for frame_index, frame in enumerate(frames):
        frame_numbers.append(int(frame.get("frame", frame_index)))
        landmarks = frame.get("landmarks")
        if not isinstance(landmarks, dict):
            raise ValueError(f"frame {frame_index} has no landmarks object")
        for marker_index, marker_name in enumerate(marker_keys):
            value = np.asarray(landmarks[marker_name], dtype=float)
            if value.shape != (3,):
                raise ValueError(f"landmark {marker_name} in frame {frame_index} must have exactly 3 coordinates")
            points[frame_index, marker_index] = value
    if not np.isfinite(points).all():
        raise ValueError("mocap marker coordinates contain NaN or inf")
    return np.asarray(frame_numbers, dtype=int), points


def farthest_pose_sample(points, max_frames, threshold_m):
    if max_frames <= 0 or len(points) <= max_frames:
        return np.arange(len(points), dtype=int)
    centered = points - np.mean(points, axis=1, keepdims=True)
    pose_vectors = centered.reshape(len(points), -1)
    center_pose = np.mean(pose_vectors, axis=0)
    first = int(np.argmin(np.linalg.norm(pose_vectors - center_pose[None, :], axis=1)))
    selected = [first]
    min_distance = np.linalg.norm(pose_vectors - pose_vectors[first][None, :], axis=1) / np.sqrt(points.shape[1])
    while len(selected) < max_frames:
        if threshold_m is not None and float(np.max(min_distance)) <= threshold_m:
            break
        next_index = int(np.argmax(min_distance))
        selected.append(next_index)
        distance = np.linalg.norm(pose_vectors - pose_vectors[next_index][None, :], axis=1) / np.sqrt(points.shape[1])
        min_distance = np.minimum(min_distance, distance)
    return np.asarray(sorted(selected), dtype=int)


def select_frames(points, mode, max_frames, threshold_m, stride):
    if mode == "all":
        return np.arange(len(points), dtype=int)
    if mode == "stride":
        stride = max(int(stride), 1)
        return np.arange(0, len(points), stride, dtype=int)
    if mode == "farthest":
        return farthest_pose_sample(points, max_frames=max_frames, threshold_m=threshold_m)
    raise ValueError(f"Unsupported frame sampling mode: {mode}")


def voxel_reduce(points, voxel_size):
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    if voxel_size is None or voxel_size <= 0:
        return points, np.ones(len(points), dtype=int)
    lower = np.min(points, axis=0)
    voxel_index = np.floor((points - lower[None, :]) / float(voxel_size)).astype(np.int64)
    unique_index, inverse = np.unique(voxel_index, axis=0, return_inverse=True)
    reduced = np.zeros((len(unique_index), 3), dtype=float)
    counts = np.zeros(len(unique_index), dtype=int)
    np.add.at(reduced, inverse, points)
    np.add.at(counts, inverse, 1)
    reduced /= counts[:, None]
    return reduced, counts


def write_pointnormal_csv(path, points):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    normals = np.zeros_like(points, dtype=float)
    pointnormals = np.hstack([points, normals])
    np.savetxt(path, pointnormals, delimiter=",", fmt="%.8f")


def build_summary(sequence_name, frames, common_keys, marker_keys, frame_numbers, selected_indices, reduced_points, counts, args):
    selected_frame_numbers = frame_numbers[selected_indices]
    return {
        "source_json": args.input_json,
        "sequence": sequence_name,
        "input_frame_count": int(len(frames)),
        "common_landmark_key_count": int(len(common_keys)),
        "marker_set": args.marker_set,
        "selected_marker_count": int(len(marker_keys)),
        "selected_markers": marker_keys,
        "frame_sampling": args.frame_sampling,
        "max_frames": int(args.max_frames),
        "pose_threshold_m": None if args.pose_threshold_m is None else float(args.pose_threshold_m),
        "stride": int(args.stride),
        "selected_frame_count": int(len(selected_indices)),
        "selected_frames": selected_frame_numbers.astype(int).tolist(),
        "raw_selected_point_count": int(len(selected_indices) * len(marker_keys)),
        "voxel_size_m": float(args.voxel_size),
        "output_point_count": int(len(reduced_points)),
        "voxel_count_min": int(np.min(counts)) if len(counts) else 0,
        "voxel_count_max": int(np.max(counts)) if len(counts) else 0,
        "bounds_min": np.min(reduced_points, axis=0).astype(float).tolist(),
        "bounds_max": np.max(reduced_points, axis=0).astype(float).tolist(),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess mocap marker JSON into NeOF/BIP point-normal CSV targets.")
    parser.add_argument("input_json", type=str)
    parser.add_argument("--sequence", type=str, default=None)
    parser.add_argument("--output_csv", type=str, default=None)
    parser.add_argument("--summary_json", type=str, default=None)
    parser.add_argument("--marker_set", type=str, choices=["physical", "spatial"], default="physical")
    parser.add_argument("--frame_sampling", type=str, choices=["farthest", "stride", "all"], default="farthest")
    parser.add_argument("--max_frames", type=int, default=50)
    parser.add_argument("--pose_threshold_m", type=float, default=0.02)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--voxel_size", type=float, default=0.02)
    return parser.parse_args()


def main():
    args = parse_args()
    sequence_name, frames = load_mocap_sequence(args.input_json, args.sequence)
    marker_keys, common_keys = select_marker_keys(frames, args.marker_set)
    frame_numbers, points = extract_points(frames, marker_keys)
    selected_indices = select_frames(points, args.frame_sampling, args.max_frames, args.pose_threshold_m, args.stride)
    selected_points = points[selected_indices].reshape(-1, 3)
    reduced_points, counts = voxel_reduce(selected_points, args.voxel_size)

    output_csv = args.output_csv
    if output_csv is None:
        stem = os.path.splitext(args.input_json)[0]
        output_csv = f"{stem}_targets.csv"
    write_pointnormal_csv(output_csv, reduced_points)

    summary = build_summary(
        sequence_name,
        frames,
        common_keys,
        marker_keys,
        frame_numbers,
        selected_indices,
        reduced_points,
        counts,
        args,
    )
    summary["output_csv"] = output_csv
    summary_path = args.summary_json
    if summary_path is None:
        summary_path = os.path.splitext(output_csv)[0] + "_summary.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(
        "Mocap preprocessing complete | "
        f"sequence={sequence_name} | frames={len(frames)} -> {len(selected_indices)} | "
        f"markers={len(marker_keys)} | points={len(selected_points)} -> {len(reduced_points)}"
    )
    print(f"  CSV: {output_csv}")
    print(f"  Summary: {summary_path}")


if __name__ == "__main__":
    main()
