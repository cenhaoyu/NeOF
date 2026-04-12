import json
import os

import numpy as np

from config_utils import _strip_json_comments


def occupancy_uses_free_space_support(args):
    return bool(
        getattr(args, "occupancy_map_enable", False)
        and getattr(args, "occupancy_map_mode", None) == "free_space_box"
    )


def resolve_occupancy_map_path(path, config_path=None):
    if path is None:
        raise ValueError("occupancy_map_file must be provided when occupancy_map_enable=true")

    candidates = [path]
    if config_path is not None:
        config_dir = os.path.dirname(os.path.abspath(config_path))
        candidates.append(os.path.join(config_dir, path))

    for candidate in candidates:
        if os.path.exists(candidate):
            return os.path.abspath(candidate)
    raise FileNotFoundError(f"Occupancy map file not found: {path}")


def _as_points(name, value):
    points = np.asarray(value, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"{name} must be an array of shape [N, 3]")
    if len(points) == 0:
        raise ValueError(f"{name} must contain at least one point")
    return points


def _as_vector(name, value, expected_length):
    vector = np.asarray(value, dtype=float)
    if vector.shape != (expected_length,):
        raise ValueError(f"{name} must contain exactly {expected_length} values")
    return vector


def _as_weights(name, value, expected_length):
    weights = np.asarray(value, dtype=float)
    if weights.shape != (expected_length,):
        raise ValueError(f"{name} must contain exactly {expected_length} values")
    if np.any(weights < 0):
        raise ValueError(f"{name} must be non-negative")
    return weights


def normalize_occupancy_weights(weights, mode):
    weights = np.asarray(weights, dtype=float)
    if np.any(weights < 0):
        raise ValueError("occupancy weights must be non-negative")

    if mode == "sum1":
        denominator = np.sum(weights)
    elif mode == "mean1":
        denominator = np.mean(weights)
    elif mode == "max1":
        denominator = np.max(weights)
    else:
        raise ValueError(f"Unsupported occupancy_map_normalize mode: {mode}")

    if denominator <= 1e-12:
        raise ValueError("occupancy weights must not all be zero")
    return weights / denominator


def resample_polyline(points, step, weights=None):
    points = _as_points("occupancy path_points", points)
    if weights is None:
        weights = np.ones(len(points), dtype=float)
    else:
        weights = _as_weights("occupancy weights", weights, len(points))

    if len(points) == 1 or step is None or step <= 0:
        return points, weights

    resampled_points = [points[0]]
    resampled_weights = [weights[0]]

    for idx in range(len(points) - 1):
        start = points[idx]
        end = points[idx + 1]
        start_weight = weights[idx]
        end_weight = weights[idx + 1]
        segment_length = float(np.linalg.norm(end - start))
        sample_count = max(int(np.ceil(segment_length / step)), 1)

        for sample_idx in range(1, sample_count + 1):
            alpha = sample_idx / sample_count
            resampled_points.append((1.0 - alpha) * start + alpha * end)
            resampled_weights.append((1.0 - alpha) * start_weight + alpha * end_weight)

    return np.asarray(resampled_points, dtype=float), np.asarray(resampled_weights, dtype=float)


def _normalize_route_points(points, coordinate_space, scale):
    if coordinate_space == "world":
        return points
    if coordinate_space != "normalized":
        raise ValueError("route coordinate_space must be either 'world' or 'normalized'")
    if scale is None:
        raise ValueError("normalized-coordinate occupancy routes require world-scale metadata")

    if isinstance(scale, dict):
        world_scale = float(scale["world_scale"])
    else:
        world_scale = float(scale[0])
    return points * world_scale


def load_occupancy_source(path):
    extension = os.path.splitext(path)[1].lower()
    if extension == ".json":
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.loads(_strip_json_comments(handle.read()))
        if not isinstance(payload, dict):
            raise ValueError(f"Occupancy JSON must contain an object: {path}")
        return payload
    if extension == ".npy":
        return {"weights": np.load(path), "mode": "voxel_weights"}
    if extension == ".npz":
        npz = np.load(path)
        if "weights" in npz:
            return {"weights": npz["weights"], "mode": "voxel_weights"}
        if "occupancy_weights" in npz:
            return {"weights": npz["occupancy_weights"], "mode": "voxel_weights"}
        raise ValueError(f"NPZ occupancy file must contain 'weights' or 'occupancy_weights': {path}")
    raise ValueError(f"Unsupported occupancy file format: {extension}")


def evaluate_route_kde_weights(sample_points, route_points, route_weights, sigma_xyz, floor):
    sample_points = np.asarray(sample_points, dtype=float)
    route_points = _as_points("occupancy path_points", route_points)
    route_weights = _as_weights("occupancy weights", route_weights, len(route_points))
    sigma_xyz = np.asarray(sigma_xyz, dtype=float)
    if sigma_xyz.shape != (3,):
        raise ValueError("occupancy_map_sigma_xyz must contain exactly 3 values")
    if np.any(sigma_xyz <= 0):
        raise ValueError("occupancy_map_sigma_xyz must be positive on every axis")
    if floor < 0:
        raise ValueError("occupancy_map_floor must be non-negative")

    heat = np.zeros(len(sample_points), dtype=float)
    chunk_size = 2048
    sigma_sq = sigma_xyz[None, None, :] ** 2
    for start in range(0, len(sample_points), chunk_size):
        stop = min(start + chunk_size, len(sample_points))
        chunk = sample_points[start:stop]
        diff = chunk[:, None, :] - route_points[None, :, :]
        squared_mahalanobis = np.sum((diff ** 2) / sigma_sq, axis=-1)
        heat[start:stop] = np.exp(-0.5 * squared_mahalanobis) @ route_weights
    return heat + float(floor)


def evaluate_gaussian_weights(sample_points, center, sigma_xyz, floor=0.0, amplitude=1.0):
    sample_points = np.asarray(sample_points, dtype=float)
    center = _as_vector("occupancy gaussian center", center, 3)
    sigma_xyz = _as_vector("occupancy gaussian sigma_xyz", sigma_xyz, 3)
    if np.any(sigma_xyz <= 0):
        raise ValueError("occupancy gaussian sigma_xyz must be positive on every axis")
    if amplitude < 0:
        raise ValueError("occupancy gaussian amplitude must be non-negative")
    if floor < 0:
        raise ValueError("occupancy_map_floor must be non-negative")

    diff = sample_points - center[None, :]
    squared_mahalanobis = np.sum((diff ** 2) / (sigma_xyz[None, :] ** 2), axis=-1)
    return amplitude * np.exp(-0.5 * squared_mahalanobis) + float(floor)


def evaluate_gaussian_mixture_weights(sample_points, components, floor=0.0):
    sample_points = np.asarray(sample_points, dtype=float)
    if floor < 0:
        raise ValueError("occupancy_map_floor must be non-negative")
    if not isinstance(components, (list, tuple)) or len(components) == 0:
        raise ValueError("gaussian_mixture requires a non-empty 'components' list")

    weights = np.zeros(len(sample_points), dtype=float)
    normalized_components = []
    for idx, component in enumerate(components):
        if not isinstance(component, dict):
            raise ValueError(f"gaussian_mixture component {idx} must be an object")
        center = _as_vector(f"gaussian_mixture component {idx} center", component.get("center"), 3)
        sigma_xyz = _as_vector(f"gaussian_mixture component {idx} sigma_xyz", component.get("sigma_xyz"), 3)
        if np.any(sigma_xyz <= 0):
            raise ValueError(f"gaussian_mixture component {idx} sigma_xyz must be positive on every axis")
        amplitude = float(component.get("weight", 1.0))
        if amplitude < 0:
            raise ValueError(f"gaussian_mixture component {idx} weight must be non-negative")
        diff = sample_points - center[None, :]
        squared_mahalanobis = np.sum((diff ** 2) / (sigma_xyz[None, :] ** 2), axis=-1)
        weights += amplitude * np.exp(-0.5 * squared_mahalanobis)
        normalized_components.append(
            {
                "center": center.astype(np.float32),
                "sigma_xyz": sigma_xyz.astype(np.float32),
                "weight": amplitude,
            }
        )
    return weights + float(floor), normalized_components


def build_route_kde_weights(voxelnormals, route_points, route_weights, sigma_xyz, floor):
    voxel_points = np.asarray(voxelnormals[:, :3], dtype=float)
    return evaluate_route_kde_weights(voxel_points, route_points, route_weights, sigma_xyz, floor)


def build_route_kde_visualization_cloud(
    route_points,
    route_weights,
    sigma_xyz,
    floor,
    grid_step,
    padding_sigma=3.0,
    keep_threshold_ratio=0.08,
):
    route_points = _as_points("occupancy path_points", route_points)
    route_weights = _as_weights("occupancy weights", route_weights, len(route_points))
    sigma_xyz = np.asarray(sigma_xyz, dtype=float)
    if sigma_xyz.shape != (3,):
        raise ValueError("occupancy_map_sigma_xyz must contain exactly 3 values")
    if np.any(sigma_xyz <= 0):
        raise ValueError("occupancy_map_sigma_xyz must be positive on every axis")
    if grid_step <= 0:
        raise ValueError("occupancy_vis_volume_step must be positive")
    if padding_sigma <= 0:
        raise ValueError("occupancy_vis_volume_padding_sigma must be positive")
    if keep_threshold_ratio < 0 or keep_threshold_ratio > 1:
        raise ValueError("occupancy_vis_volume_keep_threshold must be in [0, 1]")

    padding = padding_sigma * sigma_xyz
    lower = np.min(route_points, axis=0) - padding
    upper = np.max(route_points, axis=0) + padding

    axes = [np.arange(lower[idx], upper[idx] + 0.5 * grid_step, grid_step, dtype=float) for idx in range(3)]
    grid_x, grid_y, grid_z = np.meshgrid(axes[0], axes[1], axes[2], indexing="ij")
    sample_points = np.stack((grid_x, grid_y, grid_z), axis=-1).reshape(-1, 3)
    weights = evaluate_route_kde_weights(sample_points, route_points, route_weights, sigma_xyz, floor)

    max_weight = float(np.max(weights)) if len(weights) > 0 else 0.0
    if max_weight <= 1e-12:
        keep_mask = np.zeros(len(weights), dtype=bool)
    else:
        keep_mask = weights >= keep_threshold_ratio * max_weight

    return sample_points[keep_mask].astype(np.float32), weights[keep_mask].astype(np.float32)


def build_swept_object_visualization_cloud(
    voxelnormals,
    route_points,
    route_weights,
    grid_step,
    reference_center=None,
):
    base_points = np.asarray(voxelnormals[:, :3], dtype=float)
    if len(base_points) == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    route_points = _as_points("occupancy path_points", route_points)
    route_weights = _as_weights("occupancy weights", route_weights, len(route_points))
    if grid_step <= 0:
        raise ValueError("occupancy_vis_volume_step must be positive")

    if reference_center is None:
        reference_center = np.mean(base_points, axis=0)
    reference_center = np.asarray(reference_center, dtype=float).reshape(3)

    translated_points = []
    translated_weights = []
    for route_point, route_weight in zip(route_points, route_weights):
        offset = route_point - reference_center
        translated_points.append(base_points + offset[None, :])
        translated_weights.append(np.full(len(base_points), float(route_weight), dtype=float))

    translated_points = np.concatenate(translated_points, axis=0)
    translated_weights = np.concatenate(translated_weights, axis=0)

    lower = np.min(translated_points, axis=0)
    voxel_index = np.floor((translated_points - lower[None, :]) / float(grid_step)).astype(np.int64)
    unique_index, inverse = np.unique(voxel_index, axis=0, return_inverse=True)
    accumulated_weights = np.zeros(len(unique_index), dtype=float)
    np.add.at(accumulated_weights, inverse, translated_weights)
    voxel_centers = lower[None, :] + (unique_index.astype(float) + 0.5) * float(grid_step)
    return voxel_centers.astype(np.float32), accumulated_weights.astype(np.float32)


def build_free_space_box_support(box_min, box_max, grid_step):
    box_min = np.asarray(box_min, dtype=float)
    box_max = np.asarray(box_max, dtype=float)
    if box_min.shape != (3,) or box_max.shape != (3,):
        raise ValueError("occupancy_box_min and occupancy_box_max must each contain exactly 3 values")
    if np.any(box_max <= box_min):
        raise ValueError("occupancy_box_max must be strictly greater than occupancy_box_min on every axis")
    if grid_step <= 0:
        raise ValueError("occupancy_box_grid_step must be positive")

    axes = [
        np.arange(box_min[idx], box_max[idx] + 0.5 * grid_step, grid_step, dtype=float)
        for idx in range(3)
    ]
    grid_x, grid_y, grid_z = np.meshgrid(axes[0], axes[1], axes[2], indexing="ij")
    support_points = np.stack((grid_x, grid_y, grid_z), axis=-1).reshape(-1, 3)
    support_normals = np.zeros_like(support_points, dtype=float)
    return np.concatenate((support_points, support_normals), axis=1).astype(np.float32)


def _build_route_kde_occupancy(args, voxelnormals, source, scale):
    route_points = _as_points("occupancy path_points", source.get("path_points"))
    route_weights = source.get("weights", np.ones(len(route_points), dtype=float))
    route_weights = _as_weights("occupancy weights", route_weights, len(route_points))
    route_step = source.get("resample_step", args.occupancy_map_route_step)
    coordinate_space = source.get("coordinate_space", "world")

    route_points = _normalize_route_points(route_points, coordinate_space, scale)
    route_points, route_weights = resample_polyline(route_points, route_step, route_weights)
    occupancy_weights = build_route_kde_weights(
        voxelnormals,
        route_points,
        route_weights,
        args.occupancy_map_sigma_xyz,
        args.occupancy_map_floor,
    )
    info = {
        "mode": "route_kde",
        "route_point_count": int(len(route_points)),
        "route_coordinate_space": coordinate_space,
        "route_step": float(route_step) if route_step is not None else None,
        "route_points": route_points.astype(np.float32),
        "route_weights": route_weights.astype(np.float32),
    }
    return occupancy_weights, info


def _build_direct_voxel_weights(args, voxelnormals, source):
    del args
    weights = np.asarray(source.get("weights"), dtype=float)
    if weights.shape != (len(voxelnormals),):
        raise ValueError(
            f"voxel weight file must contain exactly {len(voxelnormals)} values, got {weights.shape}"
        )
    if np.any(weights < 0):
        raise ValueError("voxel weights must be non-negative")
    info = {"mode": "voxel_weights"}
    return weights, info


def _build_free_space_box_support(args):
    support_voxelnormals = build_free_space_box_support(
        args.occupancy_box_min,
        args.occupancy_box_max,
        args.occupancy_box_grid_step,
    )
    sample_points = support_voxelnormals[:, :3]
    distribution_mode = getattr(args, "occupancy_box_distribution", "uniform")
    distribution_path = "procedural:free_space_box"
    source = None
    if getattr(args, "occupancy_map_file", None):
        occupancy_path = resolve_occupancy_map_path(args.occupancy_map_file, getattr(args, "config", None))
        candidate_source = load_occupancy_source(occupancy_path)
        candidate_mode = candidate_source.get("mode")
        if candidate_mode in {"gaussian", "gaussian_mixture"}:
            source = candidate_source
            distribution_path = occupancy_path
            distribution_mode = candidate_mode

    if distribution_mode == "uniform":
        weights = np.ones(len(support_voxelnormals), dtype=np.float32)
    elif distribution_mode == "gaussian":
        center = getattr(args, "occupancy_gaussian_center", None)
        sigma_xyz = getattr(args, "occupancy_gaussian_sigma_xyz", None)
        amplitude = 1.0
        if source is not None:
            center = source.get("center", center)
            sigma_xyz = source.get("sigma_xyz", sigma_xyz)
            amplitude = float(source.get("weight", amplitude))
        if center is None or sigma_xyz is None:
            raise ValueError(
                "gaussian free-space occupancy requires occupancy_gaussian_center and occupancy_gaussian_sigma_xyz "
                "or an occupancy_map_file with mode='gaussian'"
            )
        weights = evaluate_gaussian_weights(
            sample_points,
            center=center,
            sigma_xyz=sigma_xyz,
            floor=args.occupancy_map_floor,
            amplitude=amplitude,
        ).astype(np.float32)
    elif distribution_mode == "gaussian_mixture":
        if source is None:
            raise ValueError(
                "gaussian_mixture free-space occupancy requires occupancy_map_file pointing to a JSON mixture specification"
            )
        weights, normalized_components = evaluate_gaussian_mixture_weights(
            sample_points,
            source.get("components"),
            floor=args.occupancy_map_floor,
        )
        weights = weights.astype(np.float32)
    else:
        raise ValueError(f"Unsupported free-space occupancy distribution: {distribution_mode}")

    info = {
        "mode": "free_space_box",
        "distribution_mode": distribution_mode,
        "path": distribution_path,
        "support_mode": "free_space",
        "point_count": int(len(support_voxelnormals)),
        "box_min": np.asarray(args.occupancy_box_min, dtype=np.float32),
        "box_max": np.asarray(args.occupancy_box_max, dtype=np.float32),
        "grid_step": float(args.occupancy_box_grid_step),
    }
    if distribution_mode == "gaussian":
        info["gaussian_center"] = _as_vector(
            "occupancy gaussian center",
            source.get("center", getattr(args, "occupancy_gaussian_center", None)) if source is not None else getattr(args, "occupancy_gaussian_center", None),
            3,
        ).astype(np.float32)
        info["gaussian_sigma_xyz"] = _as_vector(
            "occupancy gaussian sigma_xyz",
            source.get("sigma_xyz", getattr(args, "occupancy_gaussian_sigma_xyz", None)) if source is not None else getattr(args, "occupancy_gaussian_sigma_xyz", None),
            3,
        ).astype(np.float32)
    if distribution_mode == "gaussian_mixture":
        info["gaussian_mixture_components"] = normalized_components
    return support_voxelnormals, weights, info


def build_occupancy_support(args, voxelnormals, scale=None):
    if not getattr(args, "occupancy_map_enable", False):
        weights = np.ones(len(voxelnormals), dtype=float)
        return voxelnormals.astype(np.float32), weights.astype(np.float32), {
            "enabled": False,
            "mode": "uniform",
            "support_mode": "surface",
            "path": "uniform",
        }

    mode = getattr(args, "occupancy_map_mode", "route_kde")
    support_voxelnormals = np.asarray(voxelnormals, dtype=np.float32)

    if mode == "free_space_box":
        support_voxelnormals, weights, info = _build_free_space_box_support(args)
    else:
        occupancy_path = resolve_occupancy_map_path(args.occupancy_map_file, getattr(args, "config", None))
        source = load_occupancy_source(occupancy_path)
        if "mode" in source:
            mode = source["mode"]

        if mode == "route_kde":
            weights, info = _build_route_kde_occupancy(args, support_voxelnormals, source, scale)
        elif mode == "voxel_weights":
            weights, info = _build_direct_voxel_weights(args, support_voxelnormals, source)
        else:
            raise ValueError(f"Unsupported occupancy_map_mode: {mode}")
        info["path"] = occupancy_path
        info["support_mode"] = "surface"

    normalized_weights = normalize_occupancy_weights(weights, args.occupancy_map_normalize)
    info.update(
        {
            "enabled": True,
            "normalize": args.occupancy_map_normalize,
            "min": float(np.min(normalized_weights)),
            "max": float(np.max(normalized_weights)),
            "mean": float(np.mean(normalized_weights)),
        }
    )
    return support_voxelnormals.astype(np.float32), normalized_weights.astype(np.float32), info


def build_occupancy_map(args, voxelnormals, scale=None):
    _, weights, info = build_occupancy_support(args, voxelnormals, scale=scale)
    return weights, info
