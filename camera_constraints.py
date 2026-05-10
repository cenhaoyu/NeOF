import math
import os

import numpy as np
import torch


VOLUME_CONSTRAINT_SHAPES = ("box", "cylinder", "dome")
SURFACE_CONSTRAINT_SHAPES = ("box_surface", "box_walls", "plane", "cylinder_surface", "dome_surface")
CAMERA_CONSTRAINT_SHAPES = VOLUME_CONSTRAINT_SHAPES + SURFACE_CONSTRAINT_SHAPES
def apply_camera_constraint_shape_to_path(relative_path, camera_constraint_shape):
    normalized_relative = os.path.normpath(relative_path)
    parent_dir, leaf_dir = os.path.split(normalized_relative)
    if leaf_dir in ("", ".", os.sep):
        raise ValueError("path must end with a valid directory name")

    for shape_name in CAMERA_CONSTRAINT_SHAPES:
        suffix = f"_{shape_name}"
        if leaf_dir.endswith(suffix):
            leaf_dir = leaf_dir[: -len(suffix)]
            break

    shaped_leaf_dir = f"{leaf_dir}_{camera_constraint_shape}"
    shaped_path = os.path.join(parent_dir, shaped_leaf_dir) if parent_dir else shaped_leaf_dir
    if relative_path.endswith(os.sep):
        return shaped_path + os.sep
    return shaped_path


def camera_constraint_is_surface(shape):
    return shape in SURFACE_CONSTRAINT_SHAPES


def _as_vector(name, value, length):
    array = np.asarray(value, dtype=float)
    if array.shape != (length,):
        raise ValueError(f"{name} must contain exactly {length} values")
    return array


def _plane_corners(center, span_u, span_v):
    corners = []
    for sign_u in (-1.0, 1.0):
        for sign_v in (-1.0, 1.0):
            corners.append(center + sign_u * span_u + sign_v * span_v)
    return np.asarray(corners, dtype=float)


def _resolve_box_constraint(args, shape="box"):
    box_min = _as_vector("camera_constraint_box_min", getattr(args, "camera_constraint_box_min", None), 3)
    box_max = _as_vector("camera_constraint_box_max", getattr(args, "camera_constraint_box_max", None), 3)
    if np.any(box_max <= box_min):
        raise ValueError(
            "camera_constraint_box_max must be strictly greater than camera_constraint_box_min on every axis"
        )
    args.camera_constraint_box_min = box_min
    args.camera_constraint_box_max = box_max
    return {
        "shape": shape,
        "box_min": box_min,
        "box_max": box_max,
    }


def _resolve_cylinder_like_constraint(args, shape):
    center_xy = _as_vector(
        "camera_constraint_cylinder_center_xy",
        getattr(args, "camera_constraint_cylinder_center_xy", None),
        2,
    )
    radius_xy = _as_vector(
        "camera_constraint_cylinder_radius_xy",
        getattr(args, "camera_constraint_cylinder_radius_xy", None),
        2,
    )
    z_range = _as_vector(
        "camera_constraint_cylinder_z_range",
        getattr(args, "camera_constraint_cylinder_z_range", None),
        2,
    )
    if np.any(radius_xy <= 0):
        raise ValueError("camera_constraint_cylinder_radius_xy must be positive on both axes")
    if z_range[1] <= z_range[0]:
        raise ValueError("camera_constraint_cylinder_z_range must satisfy z_max > z_min")

    box_min = np.array([center_xy[0] - radius_xy[0], center_xy[1] - radius_xy[1], z_range[0]], dtype=float)
    box_max = np.array([center_xy[0] + radius_xy[0], center_xy[1] + radius_xy[1], z_range[1]], dtype=float)
    args.camera_constraint_cylinder_center_xy = center_xy
    args.camera_constraint_cylinder_radius_xy = radius_xy
    args.camera_constraint_cylinder_z_range = z_range
    return {
        "shape": shape,
        "box_min": box_min,
        "box_max": box_max,
        "cylinder_center_xy": center_xy,
        "cylinder_radius_xy": radius_xy,
        "cylinder_z_range": z_range,
    }


def _resolve_dome_like_constraint(args, shape):
    base_center = _as_vector(
        "camera_constraint_dome_base_center",
        getattr(args, "camera_constraint_dome_base_center", None),
        3,
    )
    radius_xyz = _as_vector(
        "camera_constraint_dome_radius_xyz",
        getattr(args, "camera_constraint_dome_radius_xyz", None),
        3,
    )
    if np.any(radius_xyz <= 0):
        raise ValueError("camera_constraint_dome_radius_xyz must be positive on every axis")

    box_min = np.array(
        [base_center[0] - radius_xyz[0], base_center[1] - radius_xyz[1], base_center[2]],
        dtype=float,
    )
    box_max = np.array(
        [base_center[0] + radius_xyz[0], base_center[1] + radius_xyz[1], base_center[2] + radius_xyz[2]],
        dtype=float,
    )
    args.camera_constraint_dome_base_center = base_center
    args.camera_constraint_dome_radius_xyz = radius_xyz
    return {
        "shape": shape,
        "box_min": box_min,
        "box_max": box_max,
        "dome_base_center": base_center,
        "dome_radius_xyz": radius_xyz,
    }


def _resolve_plane_constraint(args):
    plane_center = _as_vector(
        "camera_constraint_plane_center",
        getattr(args, "camera_constraint_plane_center", None),
        3,
    )
    plane_span_u = _as_vector(
        "camera_constraint_plane_span_u",
        getattr(args, "camera_constraint_plane_span_u", None),
        3,
    )
    plane_span_v = _as_vector(
        "camera_constraint_plane_span_v",
        getattr(args, "camera_constraint_plane_span_v", None),
        3,
    )
    if np.linalg.norm(plane_span_u) <= 1e-8 or np.linalg.norm(plane_span_v) <= 1e-8:
        raise ValueError("camera_constraint_plane_span_u and camera_constraint_plane_span_v must be non-zero")
    if np.linalg.norm(np.cross(plane_span_u, plane_span_v)) <= 1e-8:
        raise ValueError("camera_constraint_plane_span_u and camera_constraint_plane_span_v must not be collinear")

    corners = _plane_corners(plane_center, plane_span_u, plane_span_v)
    box_min = np.min(corners, axis=0)
    box_max = np.max(corners, axis=0)
    args.camera_constraint_plane_center = plane_center
    args.camera_constraint_plane_span_u = plane_span_u
    args.camera_constraint_plane_span_v = plane_span_v
    return {
        "shape": "plane",
        "box_min": box_min,
        "box_max": box_max,
        "plane_center": plane_center,
        "plane_span_u": plane_span_u,
        "plane_span_v": plane_span_v,
    }


def resolve_camera_constraint_args(args):
    camera_constraint_enable = bool(getattr(args, "camera_constraint_enable", False))
    camera_constraint_shape = getattr(args, "camera_constraint_shape", "box")

    if camera_constraint_shape not in CAMERA_CONSTRAINT_SHAPES:
        raise ValueError(
            f"camera_constraint_shape must be one of {', '.join(CAMERA_CONSTRAINT_SHAPES)}"
        )

    if camera_constraint_enable:
        if camera_constraint_shape in ("box", "box_surface", "box_walls"):
            constraint_data = _resolve_box_constraint(args, shape=camera_constraint_shape)
        elif camera_constraint_shape in ("cylinder", "cylinder_surface"):
            constraint_data = _resolve_cylinder_like_constraint(args, camera_constraint_shape)
        elif camera_constraint_shape in ("dome", "dome_surface"):
            constraint_data = _resolve_dome_like_constraint(args, camera_constraint_shape)
        elif camera_constraint_shape == "plane":
            constraint_data = _resolve_plane_constraint(args)
        else:
            raise ValueError(f"Unsupported camera constraint shape: {camera_constraint_shape}")
        camera_constraint_min = constraint_data["box_min"]
        camera_constraint_max = constraint_data["box_max"]
    else:
        constraint_data = None
        camera_constraint_min = None
        camera_constraint_max = None

    args.camera_constraint_enable = camera_constraint_enable
    args.camera_constraint_shape = camera_constraint_shape
    args.camera_constraint_min = camera_constraint_min
    args.camera_constraint_max = camera_constraint_max
    args.camera_constraint_data = constraint_data
    return args


def camera_volume_center(box_min, box_max):
    return 0.5 * (np.asarray(box_min, dtype=float) + np.asarray(box_max, dtype=float))


def camera_volume_xy_radii(box_min, box_max):
    return 0.5 * (np.asarray(box_max, dtype=float)[:2] - np.asarray(box_min, dtype=float)[:2])


def camera_volume_height(box_min, box_max):
    return float(np.asarray(box_max, dtype=float)[2] - np.asarray(box_min, dtype=float)[2])


def _plane_numpy_data(constraint_data):
    if constraint_data is None:
        raise ValueError("plane constraints require resolved camera_constraint_data")
    center = np.asarray(constraint_data["plane_center"], dtype=float)
    span_u = np.asarray(constraint_data["plane_span_u"], dtype=float)
    span_v = np.asarray(constraint_data["plane_span_v"], dtype=float)
    return center, span_u, span_v


def _plane_torch_data(constraint_data):
    if constraint_data is None:
        raise ValueError("plane constraints require resolved camera_constraint_data")
    return (
        constraint_data["plane_center"],
        constraint_data["plane_span_u"],
        constraint_data["plane_span_v"],
    )


def point_in_camera_constraint(point, shape, box_min, box_max, constraint_data=None, eps=1e-6):
    if box_min is None or box_max is None:
        return True

    point = np.asarray(point, dtype=float)
    box_min = np.asarray(box_min, dtype=float)
    box_max = np.asarray(box_max, dtype=float)

    if shape == "box":
        return bool(np.all(point >= box_min - eps) and np.all(point <= box_max + eps))

    if shape == "box_surface":
        inside = np.all(point >= box_min - eps) and np.all(point <= box_max + eps)
        if not inside:
            return False
        half_extent = 0.5 * (box_max - box_min)
        center = 0.5 * (box_min + box_max)
        normalized = np.abs((point - center) / np.maximum(half_extent, eps))
        return bool(np.max(normalized) >= 1.0 - eps)

    if shape == "box_walls":
        inside = np.all(point >= box_min - eps) and np.all(point <= box_max + eps)
        if not inside:
            return False
        half_extent_xy = np.maximum(0.5 * (box_max[:2] - box_min[:2]), eps)
        center_xy = 0.5 * (box_min[:2] + box_max[:2])
        normalized_xy = np.abs((point[:2] - center_xy) / half_extent_xy)
        return bool(np.max(normalized_xy) >= 1.0 - eps)

    if shape == "plane":
        center, span_u, span_v = _plane_numpy_data(constraint_data)
        basis = np.stack((span_u, span_v), axis=1)
        coeffs, _, _, _ = np.linalg.lstsq(basis, point - center, rcond=None)
        reconstructed = center + basis @ coeffs
        residual = np.linalg.norm(point - reconstructed)
        scale = max(np.linalg.norm(span_u), np.linalg.norm(span_v), 1.0)
        return bool(
            residual <= eps * scale
            and np.all(coeffs >= -1.0 - eps)
            and np.all(coeffs <= 1.0 + eps)
        )

    if point[2] < box_min[2] - eps or point[2] > box_max[2] + eps:
        return False

    center = camera_volume_center(box_min, box_max)
    rx, ry = camera_volume_xy_radii(box_min, box_max)
    rx = max(float(rx), eps)
    ry = max(float(ry), eps)

    if shape == "cylinder":
        norm_xy = ((point[0] - center[0]) / rx) ** 2 + ((point[1] - center[1]) / ry) ** 2
        return bool(norm_xy <= 1.0 + eps)

    if shape == "cylinder_surface":
        norm_xy = ((point[0] - center[0]) / rx) ** 2 + ((point[1] - center[1]) / ry) ** 2
        return bool(abs(norm_xy - 1.0) <= eps)

    if shape in ("dome", "dome_surface"):
        rz = max(camera_volume_height(box_min, box_max), eps)
        nx = (point[0] - center[0]) / rx
        ny = (point[1] - center[1]) / ry
        nz = (point[2] - box_min[2]) / rz
        ellipsoid_value = nx * nx + ny * ny + nz * nz
        if point[2] < box_min[2] - eps:
            return False
        if shape == "dome":
            return bool(ellipsoid_value <= 1.0 + eps)
        return bool(abs(ellipsoid_value - 1.0) <= eps)

    raise ValueError(f"Unsupported camera constraint shape: {shape}")


def point_in_camera_volume(point, shape, box_min, box_max, constraint_data=None, eps=1e-6):
    return point_in_camera_constraint(
        point,
        shape,
        box_min,
        box_max,
        constraint_data=constraint_data,
        eps=eps,
    )


def point_in_box(point, box_min, box_max, eps=1e-8):
    return point_in_camera_constraint(point, "box", box_min, box_max, eps=eps)


def _intersect_intervals(interval_a, interval_b, eps=1e-8):
    if interval_a is None or interval_b is None:
        return None
    lower = max(interval_a[0], interval_b[0])
    upper = min(interval_a[1], interval_b[1])
    if lower > upper + eps:
        return None
    return lower, upper


def _positive_interval(interval):
    if interval is None:
        return None
    lower, upper = interval
    if upper < 0:
        return None
    return max(lower, 0.0), upper


def _quadratic_positive_interval(a, b, c, eps=1e-8):
    if abs(a) <= eps:
        if abs(b) <= eps:
            return (-np.inf, np.inf) if c <= eps else None
        root = -c / b
        return (-np.inf, root) if b > 0 else (root, np.inf)

    discriminant = b * b - 4.0 * a * c
    if discriminant < -eps:
        return None
    discriminant = max(discriminant, 0.0)
    sqrt_discriminant = math.sqrt(discriminant)
    root_1 = (-b - sqrt_discriminant) / (2.0 * a)
    root_2 = (-b + sqrt_discriminant) / (2.0 * a)
    return min(root_1, root_2), max(root_1, root_2)


def _axis_interval(origin_value, direction_value, minimum, maximum, eps=1e-8):
    if abs(direction_value) <= eps:
        if origin_value < minimum - eps or origin_value > maximum + eps:
            return None
        return -np.inf, np.inf

    t1 = (minimum - origin_value) / direction_value
    t2 = (maximum - origin_value) / direction_value
    return min(t1, t2), max(t1, t2)


def ray_box_positive_interval(origin, direction, box_min, box_max, eps=1e-8):
    if box_min is None or box_max is None:
        return 0.0, np.inf

    origin = np.asarray(origin, dtype=float)
    direction = np.asarray(direction, dtype=float)
    norm = np.linalg.norm(direction)
    if norm <= eps:
        return None
    direction = direction / norm

    t_min = -np.inf
    t_max = np.inf
    for axis in range(3):
        axis_interval = _axis_interval(origin[axis], direction[axis], box_min[axis], box_max[axis], eps=eps)
        if axis_interval is None:
            return None
        t_min = max(t_min, axis_interval[0])
        t_max = min(t_max, axis_interval[1])
        if t_min > t_max:
            return None

    return _positive_interval((t_min, t_max))


def ray_camera_volume_positive_interval(origin, direction, shape, box_min, box_max, eps=1e-8):
    if shape not in VOLUME_CONSTRAINT_SHAPES:
        raise ValueError(f"Ray intervals are only defined for volume constraints, got: {shape}")

    if box_min is None or box_max is None:
        return 0.0, np.inf

    origin = np.asarray(origin, dtype=float)
    direction = np.asarray(direction, dtype=float)
    norm = np.linalg.norm(direction)
    if norm <= eps:
        return None
    direction = direction / norm

    if shape == "box":
        return ray_box_positive_interval(origin, direction, box_min, box_max, eps=eps)

    box_min = np.asarray(box_min, dtype=float)
    box_max = np.asarray(box_max, dtype=float)
    center = camera_volume_center(box_min, box_max)
    rx, ry = camera_volume_xy_radii(box_min, box_max)
    rx = max(float(rx), eps)
    ry = max(float(ry), eps)
    z_interval = _axis_interval(origin[2], direction[2], box_min[2], box_max[2], eps=eps)
    if z_interval is None:
        return None

    if shape == "cylinder":
        a = (direction[0] / rx) ** 2 + (direction[1] / ry) ** 2
        b = 2.0 * (
            ((origin[0] - center[0]) * direction[0]) / (rx * rx)
            + ((origin[1] - center[1]) * direction[1]) / (ry * ry)
        )
        c = ((origin[0] - center[0]) / rx) ** 2 + ((origin[1] - center[1]) / ry) ** 2 - 1.0
        radial_interval = _quadratic_positive_interval(a, b, c, eps=eps)
        return _positive_interval(_intersect_intervals(radial_interval, z_interval, eps=eps))

    if shape == "dome":
        rz = max(camera_volume_height(box_min, box_max), eps)
        dome_center = np.array([center[0], center[1], box_min[2]], dtype=float)
        radii = np.array([rx, ry, rz], dtype=float)
        normalized_origin = (origin - dome_center) / radii
        normalized_direction = direction / radii
        a = float(np.dot(normalized_direction, normalized_direction))
        b = 2.0 * float(np.dot(normalized_origin, normalized_direction))
        c = float(np.dot(normalized_origin, normalized_origin) - 1.0)
        ellipsoid_interval = _quadratic_positive_interval(a, b, c, eps=eps)
        return _positive_interval(_intersect_intervals(ellipsoid_interval, z_interval, eps=eps))

    raise ValueError(f"Unsupported camera volume shape: {shape}")
def sample_camera_points_in_constraint(num_points, shape, box_min, box_max, constraint_data=None, rng=None):
    if box_min is None or box_max is None:
        raise ValueError("resolved camera constraint bounds are required for constrained sampling")

    if rng is None:
        rng = np.random.default_rng()

    box_min = np.asarray(box_min, dtype=float)
    box_max = np.asarray(box_max, dtype=float)
    center = camera_volume_center(box_min, box_max)
    rx, ry = camera_volume_xy_radii(box_min, box_max)
    rz = camera_volume_height(box_min, box_max)

    if shape == "box":
        return rng.uniform(box_min, box_max, size=(num_points, 3))

    if shape == "box_surface":
        lengths = box_max - box_min
        face_areas = np.array(
            [
                lengths[1] * lengths[2],
                lengths[1] * lengths[2],
                lengths[0] * lengths[2],
                lengths[0] * lengths[2],
                lengths[0] * lengths[1],
                lengths[0] * lengths[1],
            ],
            dtype=float,
        )
        probabilities = face_areas / np.sum(face_areas)
        face_ids = rng.choice(6, size=num_points, p=probabilities)
        points = np.empty((num_points, 3), dtype=float)
        points[:, 0] = rng.uniform(box_min[0], box_max[0], size=num_points)
        points[:, 1] = rng.uniform(box_min[1], box_max[1], size=num_points)
        points[:, 2] = rng.uniform(box_min[2], box_max[2], size=num_points)
        points[face_ids == 0, 0] = box_min[0]
        points[face_ids == 1, 0] = box_max[0]
        points[face_ids == 2, 1] = box_min[1]
        points[face_ids == 3, 1] = box_max[1]
        points[face_ids == 4, 2] = box_min[2]
        points[face_ids == 5, 2] = box_max[2]
        return points

    if shape == "box_walls":
        lengths = box_max - box_min
        face_areas = np.array(
            [
                lengths[1] * lengths[2],
                lengths[1] * lengths[2],
                lengths[0] * lengths[2],
                lengths[0] * lengths[2],
            ],
            dtype=float,
        )
        probabilities = face_areas / np.sum(face_areas)
        face_ids = rng.choice(4, size=num_points, p=probabilities)
        points = np.empty((num_points, 3), dtype=float)
        points[:, 0] = rng.uniform(box_min[0], box_max[0], size=num_points)
        points[:, 1] = rng.uniform(box_min[1], box_max[1], size=num_points)
        points[:, 2] = rng.uniform(box_min[2], box_max[2], size=num_points)
        points[face_ids == 0, 0] = box_min[0]
        points[face_ids == 1, 0] = box_max[0]
        points[face_ids == 2, 1] = box_min[1]
        points[face_ids == 3, 1] = box_max[1]
        return points

    if shape == "cylinder":
        theta = rng.uniform(-math.pi, math.pi, size=num_points)
        radial = np.sqrt(rng.uniform(0.0, 1.0, size=num_points))
        z = rng.uniform(box_min[2], box_max[2], size=num_points)
        x = center[0] + rx * radial * np.cos(theta)
        y = center[1] + ry * radial * np.sin(theta)
        return np.stack((x, y, z), axis=1)

    if shape == "dome":
        directions = rng.normal(size=(num_points, 3))
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        norms = np.where(norms <= 1e-8, 1.0, norms)
        directions = directions / norms
        directions[:, 2] = np.abs(directions[:, 2])
        radial = np.cbrt(rng.uniform(0.0, 1.0, size=(num_points, 1)))
        unit_points = directions * radial
        x = center[0] + rx * unit_points[:, 0]
        y = center[1] + ry * unit_points[:, 1]
        z = box_min[2] + rz * unit_points[:, 2]
        return np.stack((x, y, z), axis=1)

    if shape == "plane":
        plane_center, plane_span_u, plane_span_v = _plane_numpy_data(constraint_data)
        alpha = rng.uniform(-1.0, 1.0, size=(num_points, 1))
        beta = rng.uniform(-1.0, 1.0, size=(num_points, 1))
        return plane_center[None, :] + alpha * plane_span_u[None, :] + beta * plane_span_v[None, :]

    if shape == "cylinder_surface":
        theta = rng.uniform(-math.pi, math.pi, size=num_points)
        z = rng.uniform(box_min[2], box_max[2], size=num_points)
        x = center[0] + rx * np.cos(theta)
        y = center[1] + ry * np.sin(theta)
        return np.stack((x, y, z), axis=1)

    if shape == "dome_surface":
        directions = rng.normal(size=(num_points, 3))
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        norms = np.where(norms <= 1e-8, 1.0, norms)
        directions = directions / norms
        directions[:, 2] = np.abs(directions[:, 2])
        x = center[0] + rx * directions[:, 0]
        y = center[1] + ry * directions[:, 1]
        z = box_min[2] + rz * directions[:, 2]
        return np.stack((x, y, z), axis=1)

    raise ValueError(f"Unsupported camera constraint shape: {shape}")


def sample_camera_points_in_volume(num_points, shape, box_min, box_max, constraint_data=None, rng=None):
    return sample_camera_points_in_constraint(
        num_points,
        shape,
        box_min,
        box_max,
        constraint_data=constraint_data,
        rng=rng,
    )


def _torch_logit(value):
    return torch.log(value) - torch.log1p(-value)


def _torch_atanh(value):
    return 0.5 * (torch.log1p(value) - torch.log1p(-value))


def torch_position_to_constraint_parameter(position, shape, box_min, box_max, constraint_data=None, eps=1e-6):
    if shape == "box":
        normalized = (position - box_min) / torch.clamp(box_max - box_min, min=eps)
        normalized = torch.clamp(normalized, eps, 1.0 - eps)
        return _torch_logit(normalized)

    if shape == "box_surface":
        center = 0.5 * (box_min + box_max)
        half_extent = torch.clamp(0.5 * (box_max - box_min), min=eps)
        normalized = (position - center) / half_extent
        projected = 0.5 * torch.clamp(normalized, -1.0 + eps, 1.0 - eps)
        return _torch_atanh(projected)

    if shape == "box_walls":
        center_xy = 0.5 * (box_min[:2] + box_max[:2])
        half_extent_xy = torch.clamp(0.5 * (box_max[:2] - box_min[:2]), min=eps)
        z_range = torch.clamp(box_max[2] - box_min[2], min=eps)
        normalized_xy = (position[:, :2] - center_xy.unsqueeze(0)) / half_extent_xy.unsqueeze(0)
        projected_xy = 0.5 * torch.clamp(normalized_xy, -1.0 + eps, 1.0 - eps)
        z_norm = (position[:, 2] - box_min[2]) / z_range
        z_norm = torch.clamp(z_norm, eps, 1.0 - eps)
        return torch.stack(
            (
                _torch_atanh(projected_xy[:, 0]),
                _torch_atanh(projected_xy[:, 1]),
                _torch_logit(z_norm),
            ),
            dim=1,
        )

    center = 0.5 * (box_min + box_max)
    rx = torch.clamp(0.5 * (box_max[0] - box_min[0]), min=eps)
    ry = torch.clamp(0.5 * (box_max[1] - box_min[1]), min=eps)
    z_range = torch.clamp(box_max[2] - box_min[2], min=eps)

    if shape == "cylinder":
        norm_x = (position[:, 0] - center[0]) / rx
        norm_y = (position[:, 1] - center[1]) / ry
        radial = torch.clamp(norm_x * norm_x + norm_y * norm_y, eps, 1.0 - eps)
        theta = torch.atan2(norm_y, norm_x)
        theta_norm = torch.clamp(theta / math.pi, -1.0 + eps, 1.0 - eps)
        z_norm = (position[:, 2] - box_min[2]) / z_range
        z_norm = torch.clamp(z_norm, eps, 1.0 - eps)
        return torch.stack((_torch_logit(radial), _torch_atanh(theta_norm), _torch_logit(z_norm)), dim=1)

    if shape == "dome":
        norm_x = (position[:, 0] - center[0]) / rx
        norm_y = (position[:, 1] - center[1]) / ry
        norm_z = (position[:, 2] - box_min[2]) / z_range
        rho = torch.sqrt(torch.clamp(norm_x * norm_x + norm_y * norm_y + norm_z * norm_z, min=eps, max=1.0 - eps))
        radial = torch.clamp(rho * rho, eps, 1.0 - eps)
        theta = torch.atan2(norm_y, norm_x)
        theta_norm = torch.clamp(theta / math.pi, -1.0 + eps, 1.0 - eps)
        z_dir = torch.clamp(norm_z / torch.clamp(rho, min=eps), eps, 1.0 - eps)
        return torch.stack((_torch_logit(radial), _torch_atanh(theta_norm), _torch_logit(z_dir)), dim=1)

    if shape == "plane":
        plane_center, plane_span_u, plane_span_v = _plane_torch_data(constraint_data)
        basis = torch.stack((plane_span_u, plane_span_v), dim=1)
        basis_pinv = torch.linalg.pinv(basis)
        coeffs = torch.matmul(position - plane_center.unsqueeze(0), basis_pinv.T)
        coeffs = torch.clamp(coeffs, -1.0 + eps, 1.0 - eps)
        zero = torch.zeros_like(coeffs[:, 0])
        return torch.stack((_torch_atanh(coeffs[:, 0]), _torch_atanh(coeffs[:, 1]), zero), dim=1)

    if shape == "cylinder_surface":
        norm_x = (position[:, 0] - center[0]) / rx
        norm_y = (position[:, 1] - center[1]) / ry
        theta = torch.atan2(norm_y, norm_x)
        theta_norm = torch.clamp(theta / math.pi, -1.0 + eps, 1.0 - eps)
        z_norm = (position[:, 2] - box_min[2]) / z_range
        z_norm = torch.clamp(z_norm, eps, 1.0 - eps)
        zero = torch.zeros_like(theta_norm)
        return torch.stack((_torch_atanh(theta_norm), _torch_logit(z_norm), zero), dim=1)

    if shape == "dome_surface":
        norm_x = (position[:, 0] - center[0]) / rx
        norm_y = (position[:, 1] - center[1]) / ry
        norm_z = (position[:, 2] - box_min[2]) / z_range
        theta = torch.atan2(norm_y, norm_x)
        theta_norm = torch.clamp(theta / math.pi, -1.0 + eps, 1.0 - eps)
        z_dir = torch.clamp(norm_z, eps, 1.0 - eps)
        zero = torch.zeros_like(theta_norm)
        return torch.stack((_torch_atanh(theta_norm), _torch_logit(z_dir), zero), dim=1)

    raise ValueError(f"Unsupported camera constraint shape: {shape}")


def torch_constraint_parameter_to_position(parameter, shape, box_min, box_max, constraint_data=None):
    if shape == "box":
        return box_min + torch.sigmoid(parameter) * (box_max - box_min)

    if shape == "box_surface":
        center = 0.5 * (box_min + box_max)
        half_extent = 0.5 * (box_max - box_min)
        projected = torch.tanh(parameter)
        scale = torch.amax(torch.abs(projected), dim=1, keepdim=True)
        normalized = projected / torch.clamp(scale, min=1e-6)
        fallback = torch.zeros_like(normalized)
        fallback[:, 0] = 1.0
        normalized = torch.where(scale > 1e-6, normalized, fallback)
        return center + half_extent * normalized

    if shape == "box_walls":
        center_xy = 0.5 * (box_min[:2] + box_max[:2])
        half_extent_xy = 0.5 * (box_max[:2] - box_min[:2])
        projected_xy = torch.tanh(parameter[:, :2])
        scale = torch.amax(torch.abs(projected_xy), dim=1, keepdim=True)
        normalized_xy = projected_xy / torch.clamp(scale, min=1e-6)
        fallback_xy = torch.zeros_like(normalized_xy)
        fallback_xy[:, 0] = 1.0
        normalized_xy = torch.where(scale > 1e-6, normalized_xy, fallback_xy)
        xy = center_xy.unsqueeze(0) + half_extent_xy.unsqueeze(0) * normalized_xy
        z = box_min[2] + torch.sigmoid(parameter[:, 2]) * (box_max[2] - box_min[2])
        return torch.stack((xy[:, 0], xy[:, 1], z), dim=1)

    center = 0.5 * (box_min + box_max)
    rx = 0.5 * (box_max[0] - box_min[0])
    ry = 0.5 * (box_max[1] - box_min[1])
    z_range = box_max[2] - box_min[2]

    if shape == "cylinder":
        radial = torch.sqrt(torch.sigmoid(parameter[:, 0]))
        theta = math.pi * torch.tanh(parameter[:, 1])
        z_norm = torch.sigmoid(parameter[:, 2])
        x = center[0] + rx * radial * torch.cos(theta)
        y = center[1] + ry * radial * torch.sin(theta)
        z = box_min[2] + z_norm * z_range
        return torch.stack((x, y, z), dim=1)

    if shape == "dome":
        rho = torch.sqrt(torch.sigmoid(parameter[:, 0]))
        theta = math.pi * torch.tanh(parameter[:, 1])
        z_dir = torch.sigmoid(parameter[:, 2])
        xy_dir = torch.sqrt(torch.clamp(1.0 - z_dir * z_dir, min=0.0))
        x = center[0] + rx * rho * xy_dir * torch.cos(theta)
        y = center[1] + ry * rho * xy_dir * torch.sin(theta)
        z = box_min[2] + z_range * rho * z_dir
        return torch.stack((x, y, z), dim=1)

    if shape == "plane":
        plane_center, plane_span_u, plane_span_v = _plane_torch_data(constraint_data)
        alpha = torch.tanh(parameter[:, 0]).unsqueeze(1)
        beta = torch.tanh(parameter[:, 1]).unsqueeze(1)
        return plane_center.unsqueeze(0) + alpha * plane_span_u.unsqueeze(0) + beta * plane_span_v.unsqueeze(0)

    if shape == "cylinder_surface":
        theta = math.pi * torch.tanh(parameter[:, 0])
        z_norm = torch.sigmoid(parameter[:, 1])
        x = center[0] + rx * torch.cos(theta)
        y = center[1] + ry * torch.sin(theta)
        z = box_min[2] + z_norm * z_range
        return torch.stack((x, y, z), dim=1)

    if shape == "dome_surface":
        theta = math.pi * torch.tanh(parameter[:, 0])
        z_dir = torch.sigmoid(parameter[:, 1])
        xy_dir = torch.sqrt(torch.clamp(1.0 - z_dir * z_dir, min=0.0))
        x = center[0] + rx * xy_dir * torch.cos(theta)
        y = center[1] + ry * xy_dir * torch.sin(theta)
        z = box_min[2] + z_range * z_dir
        return torch.stack((x, y, z), dim=1)

    raise ValueError(f"Unsupported camera constraint shape: {shape}")


def torch_position_to_box_parameter(position, box_min, box_max, eps=1e-6):
    return torch_position_to_constraint_parameter(position, "box", box_min, box_max, eps=eps)


def torch_box_parameter_to_position(parameter, box_min, box_max):
    return torch_constraint_parameter_to_position(parameter, "box", box_min, box_max)
