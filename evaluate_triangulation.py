import argparse
import json
import os

import numpy as np

from camera_constraints import CAMERA_CONSTRAINT_SHAPES, apply_camera_constraint_shape_to_path
from config_utils import parse_args_with_json_config
from dataset.occupancy_map import build_occupancy_support, occupancy_uses_free_space_support
from dataset.pcd import getPointNormalfromPly
from dataset.utils import (
    configure_camera_models_from_args,
    get_camera_intrinsic,
    get_camera_intrinsics_array,
    get_camera_model,
)
from field.field_attribute import get_visiblep_opt, is_point_in_fov

try:
    from scipy.optimize import least_squares
except ImportError:  # pragma: no cover - depends on local env
    least_squares = None


DEFAULT_3D_THRESHOLDS = [1e-3, 5e-3, 1e-2]
DEFAULT_REPROJ_THRESHOLDS = [0.5, 1.0, 2.0]


def resolve_model_path(modelname):
    return modelname if modelname.startswith("data/") else os.path.join("data", modelname)


def apply_solver_to_path(relative_path, solver):
    if solver in (None, "", "neof"):
        return relative_path

    normalized_relative = os.path.normpath(relative_path)
    parent_dir, leaf_dir = os.path.split(normalized_relative)
    if leaf_dir in ("", ".", os.sep):
        raise ValueError("path must end with a valid directory name")

    suffix = f"_{solver}"
    for shape_name in CAMERA_CONSTRAINT_SHAPES:
        shape_suffix = f"_{shape_name}"
        if leaf_dir.endswith(shape_suffix):
            leaf_base = leaf_dir[: -len(shape_suffix)]
            if not leaf_base.endswith(suffix):
                leaf_base = f"{leaf_base}{suffix}"
            leaf_dir = f"{leaf_base}{shape_suffix}"
            break
    else:
        if not leaf_dir.endswith(suffix):
            leaf_dir = f"{leaf_dir}{suffix}"

    solver_path = os.path.join(parent_dir, leaf_dir) if parent_dir else leaf_dir
    if relative_path.endswith(os.sep):
        return solver_path + os.sep
    return solver_path


def resolve_pose_path(default_path, override_path):
    return override_path if override_path is not None else default_path


def load_pose(path):
    camerapose = np.load(path)
    position = camerapose[:, :3]
    rotation = camerapose[:, 3:].reshape(-1, 3, 3)
    return position, rotation


def load_saved_geometry(path, target):
    if not os.path.exists(path):
        return None, None, None, None, None, None, None, {}
    geometry = np.load(path)
    voxelnormals = geometry["voxelnormals"]
    scale = float(geometry["scale"])
    center = geometry["center"]
    occupancy_weights = geometry["occupancy_weights"] if "occupancy_weights" in geometry else None
    occupancy_route_points = geometry["occupancy_route_points"] if "occupancy_route_points" in geometry else None
    occupancy_mode = None
    occupancy_distribution_mode = None
    if "occupancy_mode" in geometry:
        occupancy_mode_value = geometry["occupancy_mode"]
        occupancy_mode = occupancy_mode_value.item() if np.ndim(occupancy_mode_value) == 0 else str(occupancy_mode_value[0])
    if "occupancy_distribution_mode" in geometry:
        distribution_value = geometry["occupancy_distribution_mode"]
        occupancy_distribution_mode = (
            distribution_value.item() if np.ndim(distribution_value) == 0 else str(distribution_value[0])
        )
    metadata = {}
    for key in (
        "model_physical_height",
        "world_scale",
        "world_coordinate_system",
        "model_normalized_height",
        "physical_scale",
    ):
        if key not in geometry:
            continue
        value = geometry[key]
        metadata[key] = value.item() if np.ndim(value) == 0 else value.tolist()
    if "camera_models_json" in geometry:
        camera_models_json = geometry["camera_models_json"]
        camera_models_json = camera_models_json.item() if np.ndim(camera_models_json) == 0 else str(camera_models_json[0])
        metadata["camera_models"] = json.loads(camera_models_json)
    del target
    return (
        voxelnormals.copy(),
        scale,
        center,
        occupancy_weights,
        occupancy_route_points,
        occupancy_mode,
        occupancy_distribution_mode,
        metadata,
    )


def load_target_geometry(args):
    saved_geometry, scale, center, occupancy_weights, occupancy_route_points, occupancy_mode, occupancy_distribution_mode, metadata = load_saved_geometry(
        args.geometry_file,
        args.target,
    )
    if saved_geometry is not None:
        return (
            saved_geometry,
            scale,
            center,
            occupancy_weights,
            occupancy_route_points,
            occupancy_mode,
            occupancy_distribution_mode,
            metadata,
        )
    model_path = resolve_model_path(args.modelname)
    _, voxelnormals, _, scale, geometry_info = getPointNormalfromPly(
        model_path,
        args.voxelnum,
        args.voxelsize,
        args.kcoverage,
        target_height=getattr(args, "model_physical_height", None),
    )
    voxelnormals = np.asarray(voxelnormals, dtype=float)
    occupancy_weights = None
    occupancy_route_points = None
    occupancy_mode = None
    occupancy_distribution_mode = None
    if getattr(args, "occupancy_map_enable", False):
        voxelnormals, occupancy_weights, occupancy_info = build_occupancy_support(
            args,
            voxelnormals,
            scale=None,
        )
        occupancy_route_points = occupancy_info.get("route_points")
        occupancy_mode = occupancy_info.get("mode")
        occupancy_distribution_mode = occupancy_info.get("distribution_mode")
    metadata = {
        "model_physical_height": float(geometry_info["model_world_height"]),
        "world_scale": float(geometry_info["world_scale_from_source"]),
        "world_coordinate_system": True,
    }
    if getattr(args, "camera_models", None) is not None:
        metadata["camera_models"] = args.camera_models
    return (
        voxelnormals.copy(),
        float(scale[0]),
        scale[1],
        occupancy_weights,
        occupancy_route_points,
        occupancy_mode,
        occupancy_distribution_mode,
        metadata,
    )


def normalize_positions(position, scale, center):
    return (position - center[None, :]) / scale


def to_evaluation_positions(position, scale, center, geometry_metadata):
    if bool(geometry_metadata.get("world_coordinate_system", False)):
        return position
    return normalize_positions(position, scale, center)


def build_projection_matrices(position, rotation, camera_intrinsics):
    t = -np.matmul(rotation, position[..., None]).squeeze(-1)
    extrinsic = np.concatenate((rotation, t[..., None]), axis=-1)
    projection = np.matmul(camera_intrinsics, extrinsic)
    return projection


def apply_pixel_measurement_model(pixels, args, rng):
    measured = pixels.copy()
    if args.pixel_noise_std > 0:
        measured += rng.normal(0.0, args.pixel_noise_std, size=measured.shape)
    if args.quantize_pixels:
        measured = np.round(measured)
    return measured


def collect_observations(voxelnormals, position, rotation, args, rng):
    observations = [[] for _ in range(len(voxelnormals))]
    for camera_idx in range(len(position)):
        camera_model = get_camera_model(camera_idx)
        camera_intrinsic = get_camera_intrinsic(camera_idx)
        if occupancy_uses_free_space_support(args):
            point_idx = is_point_in_fov(
                voxelnormals[:, :3],
                rotation[camera_idx],
                position[camera_idx],
                camera_model=camera_model,
            )
            if len(point_idx) == 0:
                continue
            point_cam = np.dot(rotation[camera_idx], (voxelnormals[point_idx, :3] - position[camera_idx]).T).T
        else:
            point_cam, point_idx = get_visiblep_opt(
                voxelnormals,
                position[camera_idx],
                rotation[camera_idx],
                args.radius,
                camera_model=camera_model,
            )
        if len(point_idx) == 0:
            continue
        pixels = np.dot(camera_intrinsic, point_cam.T).T
        pixels = pixels[:, :2] / pixels[:, 2:3]
        pixels = apply_pixel_measurement_model(pixels, args, rng)
        for idx, pixel in zip(point_idx.astype(int), pixels):
            observations[idx].append((camera_idx, pixel))
    return observations


def triangulate_point_linear(projection, point_observations):
    A = []
    for camera_idx, pixel in point_observations:
        u, v = pixel
        P = projection[camera_idx]
        A.append(u * P[2] - P[0])
        A.append(v * P[2] - P[1])
    A = np.asarray(A)
    _, _, vh = np.linalg.svd(A, full_matrices=False)
    homog = vh[-1]
    if np.abs(homog[-1]) < 1e-8:
        return None
    point = homog[:3] / homog[-1]
    return point if np.all(np.isfinite(point)) else None


def project_point(projection_matrix, point):
    homog = projection_matrix @ np.append(point, 1.0)
    if np.abs(homog[-1]) < 1e-8:
        return None
    pixel = homog[:2] / homog[-1]
    return pixel if np.all(np.isfinite(pixel)) else None


def reprojection_residuals(point, projection, point_observations):
    residuals = []
    for camera_idx, pixel in point_observations:
        reproj_pixel = project_point(projection[camera_idx], point)
        if reproj_pixel is None:
            return np.full(2 * len(point_observations), 1e6, dtype=float)
        residuals.extend(reproj_pixel - pixel)
    return np.asarray(residuals, dtype=float)


def refine_triangulated_point(point_init, projection, point_observations, args):
    if not args.refine:
        return point_init
    result = least_squares(
        lambda point: reprojection_residuals(point, projection, point_observations),
        x0=point_init,
        method="lm",
        max_nfev=args.refine_max_nfev,
    )
    if not np.all(np.isfinite(result.x)):
        return None
    if not np.all(np.isfinite(result.fun)):
        return None
    return result.x


def point_depths(position, rotation, point_observations, point):
    depths = []
    for camera_idx, _ in point_observations:
        point_cam = rotation[camera_idx] @ (point - position[camera_idx])
        depths.append(float(point_cam[2]))
    return np.asarray(depths, dtype=float)


def passes_cheirality(position, rotation, point_observations, point, min_depth):
    depths = point_depths(position, rotation, point_observations, point)
    return bool(np.all(depths > min_depth)), depths


def best_triangulation_angle_deg(position, point_observations, point):
    rays = []
    for camera_idx, _ in point_observations:
        ray = position[camera_idx] - point
        norm = np.linalg.norm(ray)
        if norm <= 1e-8:
            continue
        rays.append(ray / norm)
    if len(rays) < 2:
        return np.nan
    max_angle = 0.0
    for i in range(len(rays) - 1):
        for j in range(i + 1, len(rays)):
            cos_angle = np.clip(np.dot(rays[i], rays[j]), -1.0, 1.0)
            angle = np.degrees(np.arccos(np.clip(abs(cos_angle), 0.0, 1.0)))
            max_angle = max(max_angle, float(angle))
    return max_angle


def summarize_errors(errors, prefix):
    if not errors:
        return {
            f"{prefix}_rmse": np.nan,
            f"{prefix}_mean": np.nan,
            f"{prefix}_median": np.nan,
            f"{prefix}_p90": np.nan,
            f"{prefix}_max": np.nan,
        }
    arr = np.asarray(errors, dtype=float)
    return {
        f"{prefix}_rmse": float(np.sqrt(np.mean(arr ** 2))),
        f"{prefix}_mean": float(np.mean(arr)),
        f"{prefix}_median": float(np.median(arr)),
        f"{prefix}_p90": float(np.percentile(arr, 90)),
        f"{prefix}_max": float(np.max(arr)),
    }


def summarize_distribution(values, prefix):
    if not values:
        return {
            f"{prefix}_mean": np.nan,
            f"{prefix}_median": np.nan,
            f"{prefix}_p10": np.nan,
            f"{prefix}_p90": np.nan,
            f"{prefix}_min": np.nan,
            f"{prefix}_max": np.nan,
        }
    arr = np.asarray(values, dtype=float)
    return {
        f"{prefix}_mean": float(np.mean(arr)),
        f"{prefix}_median": float(np.median(arr)),
        f"{prefix}_p10": float(np.percentile(arr, 10)),
        f"{prefix}_p90": float(np.percentile(arr, 90)),
        f"{prefix}_min": float(np.min(arr)),
        f"{prefix}_max": float(np.max(arr)),
    }


def threshold_key(prefix, threshold):
    return f"{prefix}_le_{format(float(threshold), 'g')}"


def uniform_weights(length):
    return np.ones(length, dtype=float)


def safe_weighted_mean(values, weights):
    if len(values) == 0:
        return np.nan
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    denominator = float(np.sum(weights))
    if denominator <= 1e-12:
        return np.nan
    return float(np.sum(values * weights) / denominator)


def safe_weighted_rmse(values, weights):
    if len(values) == 0:
        return np.nan
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    denominator = float(np.sum(weights))
    if denominator <= 1e-12:
        return np.nan
    return float(np.sqrt(np.sum(weights * values ** 2) / denominator))


def weighted_percentile(values, weights, percentile):
    if len(values) == 0:
        return np.nan
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    denominator = float(np.sum(weights))
    if denominator <= 1e-12:
        return np.nan
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]
    cumulative = np.cumsum(sorted_weights) / denominator
    return float(sorted_values[np.searchsorted(cumulative, percentile / 100.0, side="left")])


def summarize_weighted_errors(errors, weights, prefix):
    if not errors:
        return {
            f"{prefix}_rmse": np.nan,
            f"{prefix}_mean": np.nan,
            f"{prefix}_median": np.nan,
            f"{prefix}_p90": np.nan,
            f"{prefix}_max": np.nan,
        }
    arr = np.asarray(errors, dtype=float)
    weight_arr = np.asarray(weights, dtype=float)
    return {
        f"{prefix}_rmse": safe_weighted_rmse(arr, weight_arr),
        f"{prefix}_mean": safe_weighted_mean(arr, weight_arr),
        f"{prefix}_median": weighted_percentile(arr, weight_arr, 50.0),
        f"{prefix}_p90": weighted_percentile(arr, weight_arr, 90.0),
        f"{prefix}_max": float(np.max(arr)),
    }


def weighted_rate(mask, weights):
    mask = np.asarray(mask, dtype=bool)
    weights = np.asarray(weights, dtype=float)
    denominator = float(np.sum(weights))
    if denominator <= 1e-12:
        return np.nan
    return float(np.sum(weights[mask]) / denominator)


def build_hotspot_mask(weights, hotspot_fraction):
    weights = np.asarray(weights, dtype=float)
    hotspot_fraction = float(hotspot_fraction)
    if hotspot_fraction <= 0 or hotspot_fraction > 1:
        raise ValueError("occupancy_hotspot_fraction must be in (0, 1]")
    hotspot_count = max(int(np.ceil(len(weights) * hotspot_fraction)), 1)
    ranking = np.argsort(weights)[::-1]
    mask = np.zeros(len(weights), dtype=bool)
    mask[ranking[:hotspot_count]] = True
    return mask


def evaluate_trial(voxelnormals, occupancy_weights, hotspot_mask, position, rotation, args, rng):
    projection = build_projection_matrices(position, rotation, get_camera_intrinsics_array())
    observations = collect_observations(voxelnormals, position, rotation, args, rng)
    occupancy_weights = np.asarray(occupancy_weights, dtype=float)

    point_errors = []
    point_reprojection_errors = []
    all_reprojection_errors = []
    best_angles = []
    observation_counts = []
    reconstructed_weights = []
    reconstructed_indices = []
    observed_mask = np.zeros(len(voxelnormals), dtype=bool)
    solved_mask = np.zeros(len(voxelnormals), dtype=bool)
    geometry_valid_mask = np.zeros(len(voxelnormals), dtype=bool)
    reconstructed_mask = np.zeros(len(voxelnormals), dtype=bool)

    observable_points = 0
    linear_solved_points = 0
    geometry_valid_points = 0
    reconstructed_points = 0
    dlt_fail_points = 0
    filtered_by_cheirality_points = 0
    filtered_by_angle_points = 0
    refinement_fail_points = 0

    for point_idx, point_observations in enumerate(observations):
        if len(point_observations) < args.min_views:
            continue
        observable_points += 1
        observed_mask[point_idx] = True

        point_linear = triangulate_point_linear(projection, point_observations)
        if point_linear is None:
            dlt_fail_points += 1
            continue
        linear_solved_points += 1
        solved_mask[point_idx] = True

        point_refined = refine_triangulated_point(point_linear, projection, point_observations, args)
        if point_refined is None:
            refinement_fail_points += 1
            continue

        if args.require_cheirality:
            valid_cheirality, _ = passes_cheirality(
                position,
                rotation,
                point_observations,
                point_refined,
                args.min_depth,
            )
            if not valid_cheirality:
                filtered_by_cheirality_points += 1
                continue

        best_angle = best_triangulation_angle_deg(position, point_observations, point_refined)
        if (
            np.isfinite(args.min_triangulation_angle_deg)
            and args.min_triangulation_angle_deg > 0
            and (not np.isfinite(best_angle) or best_angle < args.min_triangulation_angle_deg)
        ):
            filtered_by_angle_points += 1
            continue

        geometry_valid_points += 1
        geometry_valid_mask[point_idx] = True

        point_error = np.linalg.norm(point_refined - voxelnormals[point_idx, :3]) * float(args.error_scale)
        per_view_reproj = []
        for camera_idx, pixel in point_observations:
            reproj_pixel = project_point(projection[camera_idx], point_refined)
            if reproj_pixel is None:
                continue
            reproj_error = np.linalg.norm(reproj_pixel - pixel)
            per_view_reproj.append(reproj_error)
            all_reprojection_errors.append(reproj_error)

        if not per_view_reproj:
            refinement_fail_points += 1
            continue

        reconstructed_points += 1
        reconstructed_mask[point_idx] = True
        reconstructed_indices.append(point_idx)
        point_errors.append(point_error)
        point_reprojection_errors.append(float(np.mean(per_view_reproj)))
        best_angles.append(best_angle)
        observation_counts.append(len(point_observations))
        reconstructed_weights.append(float(occupancy_weights[point_idx]))

    total_points = len(voxelnormals)
    metrics = {
        "total_points": float(total_points),
        "observable_points": float(observable_points),
        "observable_rate": float(observable_points / total_points) if total_points else np.nan,
        "linear_solved_points": float(linear_solved_points),
        "linear_solved_rate": float(linear_solved_points / total_points) if total_points else np.nan,
        "geometry_valid_points": float(geometry_valid_points),
        "geometry_valid_rate": float(geometry_valid_points / total_points) if total_points else np.nan,
        "reconstructed_points": float(reconstructed_points),
        "reconstruction_rate": float(reconstructed_points / total_points) if total_points else np.nan,
        "reconstruction_success_rate": (
            float(reconstructed_points / observable_points) if observable_points else np.nan
        ),
        "dlt_fail_points": float(dlt_fail_points),
        "filtered_by_cheirality_points": float(filtered_by_cheirality_points),
        "filtered_by_angle_points": float(filtered_by_angle_points),
        "refinement_fail_points": float(refinement_fail_points),
    }
    metrics.update(summarize_errors(point_errors, "3d_error"))
    metrics.update(summarize_errors(point_reprojection_errors, "point_reproj_error"))
    metrics.update(summarize_errors(all_reprojection_errors, "view_reproj_error"))
    metrics.update(summarize_distribution(best_angles, "best_angle_deg"))
    metrics.update(summarize_distribution(observation_counts, "observations_per_point"))

    metrics["weighted_observable_rate"] = weighted_rate(observed_mask, occupancy_weights)
    metrics["weighted_linear_solved_rate"] = weighted_rate(solved_mask, occupancy_weights)
    metrics["weighted_geometry_valid_rate"] = weighted_rate(geometry_valid_mask, occupancy_weights)
    metrics["weighted_reconstruction_rate"] = weighted_rate(reconstructed_mask, occupancy_weights)
    if np.any(observed_mask):
        metrics["weighted_reconstruction_success_rate"] = float(
            np.sum(occupancy_weights[reconstructed_mask]) / np.sum(occupancy_weights[observed_mask])
        )
    else:
        metrics["weighted_reconstruction_success_rate"] = np.nan
    metrics.update(summarize_weighted_errors(point_errors, reconstructed_weights, "weighted_3d_error"))
    metrics.update(
        summarize_weighted_errors(
            point_reprojection_errors,
            reconstructed_weights,
            "weighted_point_reproj_error",
        )
    )
    metrics["weighted_best_angle_deg_mean"] = safe_weighted_mean(best_angles, reconstructed_weights)

    if hotspot_mask is not None:
        hotspot_mask = np.asarray(hotspot_mask, dtype=bool)
        hotspot_total = int(np.sum(hotspot_mask))
        hotspot_reconstructed = reconstructed_mask & hotspot_mask
        hotspot_errors = [
            point_errors[idx]
            for idx, point_idx in enumerate(reconstructed_indices)
            if hotspot_mask[point_idx]
        ]
        hotspot_reproj_errors = [
            point_reprojection_errors[idx]
            for idx, point_idx in enumerate(reconstructed_indices)
            if hotspot_mask[point_idx]
        ]
        metrics["hotspot_fraction"] = float(args.occupancy_hotspot_fraction)
        metrics["hotspot_points"] = float(hotspot_total)
        metrics["hotspot_reconstructed_points"] = float(np.sum(hotspot_reconstructed))
        metrics["hotspot_reconstruction_rate"] = (
            float(np.sum(hotspot_reconstructed) / hotspot_total) if hotspot_total > 0 else np.nan
        )
        metrics.update(summarize_errors(hotspot_errors, "hotspot_3d_error"))
        metrics.update(summarize_errors(hotspot_reproj_errors, "hotspot_point_reproj_error"))
        metrics["hotspot_weighted_reconstruction_rate"] = weighted_rate(hotspot_reconstructed, occupancy_weights * hotspot_mask)

    for threshold in args.success_3d_thresholds:
        key = threshold_key("3d_success_rate", threshold)
        metrics[key] = (
            float(np.mean(np.asarray(point_errors, dtype=float) <= threshold)) if point_errors else np.nan
        )
    for threshold in args.success_reproj_thresholds:
        key = threshold_key("reproj_success_rate", threshold)
        metrics[key] = (
            float(np.mean(np.asarray(point_reprojection_errors, dtype=float) <= threshold))
            if point_reprojection_errors
            else np.nan
        )
    return metrics


def run_trials(voxelnormals, occupancy_weights, hotspot_mask, position, rotation, args):
    trials = []
    for trial_idx in range(args.trials):
        rng = np.random.default_rng(args.seed + trial_idx)
        metrics = evaluate_trial(
            voxelnormals,
            occupancy_weights,
            hotspot_mask,
            position,
            rotation,
            args,
            rng,
        )
        trials.append(metrics)
    return trials


def aggregate_trials(trials):
    if not trials:
        raise ValueError("trials must be at least 1")
    summary = {}
    for key in trials[0].keys():
        values = np.asarray([trial[key] for trial in trials], dtype=float)
        finite = np.isfinite(values)
        if not np.any(finite):
            summary[key] = {"mean": None, "std": None, "min": None, "max": None}
            continue
        finite_values = values[finite]
        summary[key] = {
            "mean": float(np.mean(finite_values)),
            "std": float(np.std(finite_values)),
            "min": float(np.min(finite_values)),
            "max": float(np.max(finite_values)),
        }
    return summary


def get_summary_mean(summary, key):
    value = summary.get(key, {}).get("mean")
    return np.nan if value is None else float(value)


def format_metric(value, unit=""):
    if value is None or not np.isfinite(value):
        return "n/a"
    if abs(value) >= 1e-3:
        text = f"{value:.6f}"
    else:
        text = f"{value:.6e}"
    return text + (f" {unit}" if unit else "")


def format_count(value):
    if value is None or not np.isfinite(value):
        return "n/a"
    rounded = round(value)
    if abs(value - rounded) < 1e-6:
        return str(int(rounded))
    return f"{value:.3f}"


def format_percent(value):
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{value:.2f}%"


def format_signed_percent(value):
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{value:+.2f}%"


def format_signed_metric(value, unit=""):
    if value is None or not np.isfinite(value):
        return "n/a"
    if abs(value) >= 1e-3:
        text = f"{value:+.6f}"
    else:
        text = f"{value:+.6e}"
    return text + (f" {unit}" if unit else "")


def format_signed_count(value):
    if value is None or not np.isfinite(value):
        return "n/a"
    rounded = round(value)
    if abs(value - rounded) < 1e-6:
        return f"{int(rounded):+d}"
    return f"{value:+.3f}"


def format_summary(summary, key, unit=""):
    mean = summary[key]["mean"]
    std = summary[key]["std"]
    if mean is None:
        return "n/a"
    if std is None or std == 0:
        return format_metric(mean, unit)
    return f"{format_metric(mean, unit)} +- {format_metric(std, unit)}"


def print_threshold_metrics(summary, prefix, thresholds, label, unit=""):
    for threshold in thresholds:
        key = threshold_key(prefix, threshold)
        mean = get_summary_mean(summary, key)
        print(f"  {label} <= {threshold:g}{unit}: {format_percent(mean * 100)}")


def print_section(title, summary, args):
    total_points = get_summary_mean(summary, "total_points")
    reconstructed_points = get_summary_mean(summary, "reconstructed_points")
    reconstruction_rate = get_summary_mean(summary, "reconstruction_rate")

    print(title)
    print(
        f"  Reconstructed points: {format_count(reconstructed_points)}/{int(round(total_points))} "
        f"({format_percent(reconstruction_rate * 100)})"
    )
    print(f"  Best effective triangulation angle mean: {format_summary(summary, 'best_angle_deg_mean', 'deg')}")
    print(f"  3D RMSE: {format_summary(summary, '3d_error_rmse', args.length_unit)}")
    print(f"  3D P90 error: {format_summary(summary, '3d_error_p90', args.length_unit)}")
    print(f"  Per-point reprojection RMSE: {format_summary(summary, 'point_reproj_error_rmse', 'px')}")
    if getattr(args, "occupancy_eval_active", False):
        print(f"  Weighted reconstruction rate: {format_percent(get_summary_mean(summary, 'weighted_reconstruction_rate') * 100)}")
        print(f"  Weighted 3D RMSE: {format_summary(summary, 'weighted_3d_error_rmse', args.length_unit)}")
        print(
            f"  Weighted per-point reprojection RMSE: "
            f"{format_summary(summary, 'weighted_point_reproj_error_rmse', 'px')}"
        )
        hotspot_label = int(round(args.occupancy_hotspot_fraction * 100))
        print(
            f"  Hotspot-{hotspot_label}% reconstruction rate: "
            f"{format_percent(get_summary_mean(summary, 'hotspot_reconstruction_rate') * 100)}"
        )
        print(
            f"  Hotspot-{hotspot_label}% 3D RMSE: "
            f"{format_summary(summary, 'hotspot_3d_error_rmse', args.length_unit)}"
        )


def compute_percent_improvement(initial_value, optimized_value, higher_is_better):
    if not np.isfinite(initial_value) or not np.isfinite(optimized_value):
        return np.nan
    denominator = abs(initial_value)
    if denominator <= 1e-12:
        if abs(optimized_value - initial_value) <= 1e-12:
            return 0.0
        return np.nan
    if higher_is_better:
        return 100.0 * (optimized_value - initial_value) / denominator
    return 100.0 * (initial_value - optimized_value) / denominator


def build_comparison_rows(initial_summary, optimized_summary):
    del initial_summary, optimized_summary
    return [
        {
            "label": "Reconstructed points",
            "key": "reconstructed_points",
            "kind": "count",
            "higher_is_better": True,
            "delta_unit": "",
        },
        {
            "label": "Reconstruction rate",
            "key": "reconstruction_rate",
            "kind": "rate",
            "higher_is_better": True,
            "delta_unit": "pp",
        },
        {
            "label": "Best effective triangulation angle mean",
            "key": "best_angle_deg_mean",
            "kind": "metric",
            "higher_is_better": True,
            "unit": "deg",
            "delta_unit": "deg",
        },
        {
            "label": "3D RMSE",
            "key": "3d_error_rmse",
            "kind": "metric",
            "higher_is_better": False,
            "unit": None,
        },
        {
            "label": "3D P90 error",
            "key": "3d_error_p90",
            "kind": "metric",
            "higher_is_better": False,
            "unit": None,
        },
        {
            "label": "Per-point reprojection RMSE",
            "key": "point_reproj_error_rmse",
            "kind": "metric",
            "higher_is_better": False,
            "unit": "px",
            "delta_unit": "px",
        },
    ]


def build_comparison_summary(initial_summary, optimized_summary, args):
    rows = []
    row_specs = build_comparison_rows(initial_summary, optimized_summary)
    if getattr(args, "occupancy_eval_active", False):
        hotspot_label = int(round(args.occupancy_hotspot_fraction * 100))
        row_specs.extend(
            [
                {
                    "label": "Weighted reconstruction rate",
                    "key": "weighted_reconstruction_rate",
                    "kind": "rate",
                    "higher_is_better": True,
                    "delta_unit": "pp",
                },
                {
                    "label": "Weighted 3D RMSE",
                    "key": "weighted_3d_error_rmse",
                    "kind": "metric",
                    "higher_is_better": False,
                    "unit": None,
                },
                {
                    "label": "Weighted per-point reprojection RMSE",
                    "key": "weighted_point_reproj_error_rmse",
                    "kind": "metric",
                    "higher_is_better": False,
                    "unit": "px",
                    "delta_unit": "px",
                },
                {
                    "label": f"Hotspot-{hotspot_label}% reconstruction rate",
                    "key": "hotspot_reconstruction_rate",
                    "kind": "rate",
                    "higher_is_better": True,
                    "delta_unit": "pp",
                },
                {
                    "label": f"Hotspot-{hotspot_label}% 3D RMSE",
                    "key": "hotspot_3d_error_rmse",
                    "kind": "metric",
                    "higher_is_better": False,
                    "unit": None,
                },
            ]
        )
    for spec in row_specs:
        key = spec["key"]
        initial_value = get_summary_mean(initial_summary, key)
        optimized_value = get_summary_mean(optimized_summary, key)
        delta = optimized_value - initial_value if np.isfinite(initial_value) and np.isfinite(optimized_value) else np.nan
        percent_improvement = compute_percent_improvement(
            initial_value,
            optimized_value,
            higher_is_better=spec["higher_is_better"],
        )
        row = {
            "label": spec["label"],
            "key": key,
            "initial": None if not np.isfinite(initial_value) else float(initial_value),
            "optimized": None if not np.isfinite(optimized_value) else float(optimized_value),
            "delta": None if not np.isfinite(delta) else float(delta),
            "percent_improvement": None if not np.isfinite(percent_improvement) else float(percent_improvement),
            "kind": spec["kind"],
            "higher_is_better": bool(spec["higher_is_better"]),
        }
        unit = spec.get("unit")
        if unit is None and "3d_error" in key:
            unit = args.length_unit
        if unit:
            row["unit"] = unit
        delta_unit = spec.get("delta_unit")
        if delta_unit is None:
            delta_unit = unit if unit is not None else ""
        if delta_unit:
            row["delta_unit"] = delta_unit
        rows.append(row)
    return rows


def print_comparison_section(initial_summary, optimized_summary, args):
    print("Comparison")
    for row in build_comparison_summary(initial_summary, optimized_summary, args):
        kind = row["kind"]
        initial_value = row["initial"]
        optimized_value = row["optimized"]
        delta = row["delta"]
        percent_improvement = row["percent_improvement"]
        unit = row.get("unit", "")
        delta_unit = row.get("delta_unit", "")

        if kind == "count":
            initial_text = format_count(initial_value)
            optimized_text = format_count(optimized_value)
            delta_text = format_signed_count(delta)
        elif kind == "rate":
            initial_text = format_percent(initial_value * 100 if initial_value is not None else None)
            optimized_text = format_percent(optimized_value * 100 if optimized_value is not None else None)
            delta_text = format_signed_metric(delta * 100 if delta is not None else None, delta_unit)
        else:
            initial_text = format_metric(initial_value, unit)
            optimized_text = format_metric(optimized_value, unit)
            delta_text = format_signed_metric(delta, delta_unit)

        print(
            f"  {row['label']}: {initial_text} -> {optimized_text} | "
            f"delta {delta_text} | improvement {format_signed_percent(percent_improvement)}"
        )


def save_json_report(path, args, initial_trials, optimized_trials, initial_summary, optimized_summary):
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    comparison = build_comparison_summary(initial_summary, optimized_summary, args)
    report = {
        "config": vars(args),
        "initial": {"trials": initial_trials, "summary": initial_summary},
        "optimized": {"trials": optimized_trials, "summary": optimized_summary},
        "comparison": comparison,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--path", type=str, default="random/brother/")
    parser.add_argument("--solver", type=str, choices=["neof", "bip"], default="neof")
    parser.add_argument(
        "--camera_constraint_shape",
        type=str,
        choices=CAMERA_CONSTRAINT_SHAPES,
        default=None,
    )
    parser.add_argument("--modelname", type=str, required=True)
    parser.add_argument("--camera_models", type=json.loads, default=None)
    parser.add_argument("--image_width", type=int, default=640)
    parser.add_argument("--image_height", type=int, default=480)
    parser.add_argument("--fx", type=float, default=320.0)
    parser.add_argument("--fy", type=float, default=320.0)
    parser.add_argument("--cx", type=float, default=319.5)
    parser.add_argument("--cy", type=float, default=239.5)
    parser.add_argument("--model_physical_height", type=float, default=None)
    parser.add_argument("--kcoverage", type=int, default=3)
    parser.add_argument("--voxelnum", type=int, default=30000)
    parser.add_argument("--voxelsize", type=float, default=0.02)
    parser.add_argument("--target", type=str, choices=["voxel"], default="voxel")
    parser.add_argument("--radius", type=float, default=200.0)
    parser.add_argument("--min_views", type=int, default=2)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pixel_noise_std", type=float, default=0.0)
    parser.add_argument("--quantize_pixels", dest="quantize_pixels", action="store_true")
    parser.add_argument("--no_quantize_pixels", dest="quantize_pixels", action="store_false")
    parser.set_defaults(quantize_pixels=True)
    parser.add_argument("--occupancy_map_enable", dest="occupancy_map_enable", action="store_true")
    parser.add_argument("--no_occupancy_map_enable", dest="occupancy_map_enable", action="store_false")
    parser.set_defaults(occupancy_map_enable=False)
    parser.add_argument("--occupancy_map_mode", type=str, choices=["route_kde", "voxel_weights", "free_space_box"], default="route_kde")
    parser.add_argument("--occupancy_map_file", type=str, default=None)
    parser.add_argument("--occupancy_map_sigma_xyz", type=float, nargs=3, default=[0.08, 0.08, 0.12])
    parser.add_argument("--occupancy_map_floor", type=float, default=0.05)
    parser.add_argument("--occupancy_map_normalize", type=str, choices=["sum1", "mean1", "max1"], default="mean1")
    parser.add_argument("--occupancy_map_route_step", type=float, default=0.03)
    parser.add_argument("--occupancy_box_min", type=float, nargs=3, default=None)
    parser.add_argument("--occupancy_box_max", type=float, nargs=3, default=None)
    parser.add_argument("--occupancy_box_grid_step", type=float, default=0.04)
    parser.add_argument("--occupancy_box_distribution", type=str, choices=["uniform", "gaussian", "gaussian_mixture"], default="uniform")
    parser.add_argument("--occupancy_gaussian_center", type=float, nargs=3, default=None)
    parser.add_argument("--occupancy_gaussian_sigma_xyz", type=float, nargs=3, default=None)
    parser.add_argument("--occupancy_hotspot_fraction", type=float, default=0.2)
    parser.add_argument("--refine", dest="refine", action="store_true")
    parser.add_argument("--no_refine", dest="refine", action="store_false")
    parser.set_defaults(refine=True)
    parser.add_argument("--refine_max_nfev", type=int, default=100)
    parser.add_argument("--require_cheirality", dest="require_cheirality", action="store_true")
    parser.add_argument("--no_require_cheirality", dest="require_cheirality", action="store_false")
    parser.set_defaults(require_cheirality=True)
    parser.add_argument("--min_depth", type=float, default=1e-6)
    parser.add_argument("--min_triangulation_angle_deg", type=float, default=1.0)
    parser.add_argument("--length_unit", type=str, default="m")
    parser.add_argument("--success_3d_thresholds", type=float, nargs="*", default=DEFAULT_3D_THRESHOLDS)
    parser.add_argument(
        "--success_reproj_thresholds",
        type=float,
        nargs="*",
        default=DEFAULT_REPROJ_THRESHOLDS,
    )
    parser.add_argument("--pose_dir", type=str, default=None)
    parser.add_argument("--geometry_file", type=str, default=None)
    parser.add_argument("--initial_pose", type=str, default=None)
    parser.add_argument("--optimized_pose", type=str, default=None)
    parser.add_argument("--output_json", type=str, default=None)
    args = parse_args_with_json_config(parser, allow_unknown_config_keys=True)
    args.path = apply_solver_to_path(args.path, args.solver)
    if args.camera_constraint_shape is not None:
        args.path = apply_camera_constraint_shape_to_path(args.path, args.camera_constraint_shape)
    args = configure_camera_models_from_args(args)

    if args.trials < 1:
        raise ValueError("--trials must be at least 1")
    if args.occupancy_map_enable and args.occupancy_map_mode == "free_space_box":
        if args.occupancy_box_min is None or args.occupancy_box_max is None:
            raise ValueError("--occupancy_box_min and --occupancy_box_max are required for occupancy_map_mode=free_space_box")
        if any(b <= a for a, b in zip(args.occupancy_box_min, args.occupancy_box_max)):
            raise ValueError("--occupancy_box_max must be strictly greater than --occupancy_box_min on every axis")
    if args.occupancy_gaussian_sigma_xyz is not None and any(value <= 0 for value in args.occupancy_gaussian_sigma_xyz):
        raise ValueError("--occupancy_gaussian_sigma_xyz must be positive on every axis")
    if args.occupancy_hotspot_fraction <= 0 or args.occupancy_hotspot_fraction > 1:
        raise ValueError("--occupancy_hotspot_fraction must be in (0, 1]")
    if args.model_physical_height is not None and args.model_physical_height <= 0:
        raise ValueError("--model_physical_height must be positive when provided")
    if args.refine and least_squares is None:
        raise ImportError("scipy is required for nonlinear refinement; install scipy or use --no_refine")

    result_dir = os.path.join("resultModel", args.path)
    pose_dir = args.pose_dir if args.pose_dir is not None else os.path.join(result_dir, "pose")
    args.geometry_file = (
        args.geometry_file if args.geometry_file is not None else os.path.join(result_dir, "geometry_data.npz")
    )
    initial_pose_path = resolve_pose_path(os.path.join(pose_dir, "0.npy"), args.initial_pose)
    optimized_pose_path = resolve_pose_path(os.path.join(pose_dir, "after.npy"), args.optimized_pose)

    if not os.path.exists(initial_pose_path):
        raise FileNotFoundError(f"Initial pose file not found: {initial_pose_path}")
    if not os.path.exists(optimized_pose_path):
        raise FileNotFoundError(f"Optimized pose file not found: {optimized_pose_path}")

    initial_position, initial_rotation = load_pose(initial_pose_path)
    optimized_position, optimized_rotation = load_pose(optimized_pose_path)

    (
        voxelnormals,
        scale,
        center,
        occupancy_weights,
        occupancy_route_points,
        occupancy_mode,
        occupancy_distribution_mode,
        geometry_metadata,
    ) = load_target_geometry(args)
    if occupancy_mode is not None:
        args.occupancy_map_mode = occupancy_mode
    if occupancy_distribution_mode is not None:
        args.occupancy_box_distribution = occupancy_distribution_mode
    if "camera_models" not in geometry_metadata:
        raise ValueError(
            f"{args.geometry_file} is missing camera_models_json. "
            "Please regenerate geometry_data.npz with the current heterogeneous-camera branch."
        )
    args.camera_models = geometry_metadata["camera_models"]
    args.cameranum = len(args.camera_models)
    if "model_physical_height" in geometry_metadata:
        args.model_physical_height = float(geometry_metadata["model_physical_height"])
    args = configure_camera_models_from_args(args)
    if len(initial_position) != args.cameranum or len(optimized_position) != args.cameranum:
        raise ValueError(
            "Pose file camera count does not match geometry_data.npz camera_models_json. "
            f"initial={len(initial_position)}, optimized={len(optimized_position)}, rig={args.cameranum}"
        )
    args.error_scale = (
        1.0
        if bool(geometry_metadata.get("world_coordinate_system", False))
        else float(geometry_metadata.get("physical_scale", 1.0))
    )
    if occupancy_weights is None:
        occupancy_weights = uniform_weights(len(voxelnormals))
    occupancy_weights = np.asarray(occupancy_weights, dtype=float)
    args.occupancy_eval_active = bool(
        len(occupancy_weights) > 0 and not np.allclose(occupancy_weights, occupancy_weights[0])
    )
    hotspot_mask = build_hotspot_mask(occupancy_weights, args.occupancy_hotspot_fraction) if args.occupancy_eval_active else None
    initial_position = to_evaluation_positions(initial_position, scale, center, geometry_metadata)
    optimized_position = to_evaluation_positions(optimized_position, scale, center, geometry_metadata)

    initial_trials = run_trials(
        voxelnormals,
        occupancy_weights,
        hotspot_mask,
        initial_position,
        initial_rotation,
        args,
    )
    optimized_trials = run_trials(
        voxelnormals,
        occupancy_weights,
        hotspot_mask,
        optimized_position,
        optimized_rotation,
        args,
    )
    initial_summary = aggregate_trials(initial_trials)
    optimized_summary = aggregate_trials(optimized_trials)

    print(f"Target geometry: {args.target}")
    print(f"Initial pose file: {initial_pose_path}")
    print(f"Optimized pose file: {optimized_pose_path}")
    print(f"3D error unit: {args.length_unit}")
    print(f"Camera rig loaded for evaluation: {len(args.camera_models)} cameras")
    for idx, camera_model in enumerate(args.camera_models):
        print(
            f"  Cam {idx:02d} | {camera_model['name']} | "
            f"image={int(round(camera_model['image_width']))}x{int(round(camera_model['image_height']))} | "
            f"fx={camera_model['fx']:.1f}, fy={camera_model['fy']:.1f}"
        )
    if "model_physical_height" in geometry_metadata:
        print(f"World-scale target height: {float(geometry_metadata['model_physical_height']):.3f} m")
    print(
        f"Measurement model: quantize_pixels={args.quantize_pixels}, "
        f"pixel_noise_std={args.pixel_noise_std:.3f} px, "
        f"min_views={args.min_views}, trials={args.trials}, seed={args.seed}"
    )
    print(
        f"Triangulation settings: refine={args.refine}, "
        f"require_cheirality={args.require_cheirality}, "
        f"min_depth={args.min_depth:.1e}, "
        f"min_triangulation_angle_deg={args.min_triangulation_angle_deg:.3f}"
    )
    if args.occupancy_eval_active:
        print(
            "Occupancy-aware evaluation: enabled | "
            f"weight range=[{np.min(occupancy_weights):.4f}, {np.max(occupancy_weights):.4f}] | "
            f"hotspot fraction={args.occupancy_hotspot_fraction:.2f}"
        )
    elif occupancy_route_points is not None:
        print("Occupancy route points were found, but the saved weights are effectively uniform.")
    print_section("Initial", initial_summary, args)
    print_section("Optimized", optimized_summary, args)
    print_comparison_section(initial_summary, optimized_summary, args)

    if args.output_json is not None:
        save_json_report(
            args.output_json,
            args,
            initial_trials,
            optimized_trials,
            initial_summary,
            optimized_summary,
        )
        print(f"JSON report saved to: {args.output_json}")


if __name__ == "__main__":
    main()
