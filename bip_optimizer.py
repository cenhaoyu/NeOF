import json
import math
import os
import time
from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, milp

try:
    from ortools.sat.python import cp_model
except ImportError:  # Optional backend; scipy_highs remains available without OR-Tools.
    cp_model = None

from camera_constraints import point_in_camera_constraint
from dataset.init_camera import getLookAtRotation
from dataset.utils import compute_rotation_matrix_from_ortho6d, get_camera_model, npToTensor, saveTrainingResult
from field.field_attribute import (
    get_visible_points_free_space,
    get_visiblep_opt,
    uses_fov_only_target_visibility,
    uses_free_space_support,
    voxel_model,
)


def format_seconds(seconds):
    return f"{seconds:.2f}s"


def format_vector(vector):
    return "[" + ", ".join(f"{float(value):.4f}" for value in np.asarray(vector, dtype=float)) + "]"


@dataclass(frozen=True)
class PoseCandidate:
    position: np.ndarray
    rotation: np.ndarray
    target: np.ndarray


@dataclass(frozen=True)
class BIPSolution:
    selected_pose_indices: list
    selected_decision_indices: list
    objective_value: float
    weighted_coverage_rate: float
    success: bool
    message: str
    solve_time_s: float


def _grid_axis(min_value, max_value, step):
    if step <= 0:
        raise ValueError("BIP candidate grid step must be positive")
    values = list(np.arange(float(min_value), float(max_value) + 1e-9, float(step), dtype=float))
    if not values or values[-1] < float(max_value) - 1e-9:
        values.append(float(max_value))
    return np.asarray(values, dtype=float)


def _span_parameters(span, step):
    length = 2.0 * float(np.linalg.norm(span))
    count = max(int(math.ceil(length / float(step))) + 1, 2)
    return np.linspace(-1.0, 1.0, count, dtype=float)


def _unique_rows(points, decimals=6):
    if len(points) == 0:
        return np.zeros((0, 3), dtype=float)
    rounded = np.round(np.asarray(points, dtype=float), decimals=decimals)
    _, unique_indices = np.unique(rounded, axis=0, return_index=True)
    unique_indices = np.sort(unique_indices)
    return np.asarray(points, dtype=float)[unique_indices]


def _ellipse_perimeter(rx, ry):
    rx = abs(float(rx))
    ry = abs(float(ry))
    if rx <= 1e-8 and ry <= 1e-8:
        return 0.0
    if rx <= 1e-8:
        return 4.0 * ry
    if ry <= 1e-8:
        return 4.0 * rx
    h = ((rx - ry) ** 2) / ((rx + ry) ** 2)
    return math.pi * (rx + ry) * (1.0 + 3.0 * h / (10.0 + math.sqrt(max(4.0 - 3.0 * h, 1e-8))))


def _farthest_sample_indices(points, sample_count):
    points = np.asarray(points, dtype=float)
    if len(points) <= sample_count:
        return np.arange(len(points), dtype=int)
    center = np.mean(points, axis=0)
    first = int(np.argmin(np.linalg.norm(points - center[None, :], axis=1)))
    selected = [first]
    min_distance = np.linalg.norm(points - points[first][None, :], axis=1)
    while len(selected) < sample_count:
        next_index = int(np.argmax(min_distance))
        selected.append(next_index)
        min_distance = np.minimum(
            min_distance,
            np.linalg.norm(points - points[next_index][None, :], axis=1),
        )
    return np.asarray(selected, dtype=int)


def _constraint_grid_positions(args):
    shape = args.camera_constraint_shape
    box_min = np.asarray(args.camera_constraint_min, dtype=float)
    box_max = np.asarray(args.camera_constraint_max, dtype=float)
    step = float(args.bip_candidate_position_step)
    data = args.camera_constraint_data

    if shape == "plane":
        center = np.asarray(data["plane_center"], dtype=float)
        span_u = np.asarray(data["plane_span_u"], dtype=float)
        span_v = np.asarray(data["plane_span_v"], dtype=float)
        positions = [
            center + alpha * span_u + beta * span_v
            for alpha in _span_parameters(span_u, step)
            for beta in _span_parameters(span_v, step)
        ]
        return _unique_rows(positions)

    if shape in ("box_surface", "box_walls"):
        axes = [_grid_axis(box_min[axis], box_max[axis], step) for axis in range(3)]
        positions = []
        fixed_axes = range(3) if shape == "box_surface" else range(2)
        for fixed_axis in fixed_axes:
            other_axes = [axis for axis in range(3) if axis != fixed_axis]
            for fixed_value in (box_min[fixed_axis], box_max[fixed_axis]):
                for first in axes[other_axes[0]]:
                    for second in axes[other_axes[1]]:
                        point = np.zeros(3, dtype=float)
                        point[fixed_axis] = fixed_value
                        point[other_axes[0]] = first
                        point[other_axes[1]] = second
                        positions.append(point)
        return _unique_rows(positions)

    center = 0.5 * (box_min + box_max)
    rx = 0.5 * (box_max[0] - box_min[0])
    ry = 0.5 * (box_max[1] - box_min[1])
    rz = box_max[2] - box_min[2]

    if shape == "cylinder_surface":
        theta_count = max(int(math.ceil(_ellipse_perimeter(rx, ry) / step)), 4)
        theta_values = np.linspace(-math.pi, math.pi, theta_count, endpoint=False, dtype=float)
        z_values = _grid_axis(box_min[2], box_max[2], step)
        positions = [
            np.array([center[0] + rx * math.cos(theta), center[1] + ry * math.sin(theta), z], dtype=float)
            for theta in theta_values
            for z in z_values
        ]
        return _unique_rows(positions)

    if shape == "dome_surface":
        theta_count = max(int(math.ceil(2.0 * math.pi * max(rx, ry) / step)), 4)
        z_count = max(int(math.ceil(max(rz, step) / step)) + 1, 2)
        theta_values = np.linspace(-math.pi, math.pi, theta_count, endpoint=False, dtype=float)
        z_dir_values = np.linspace(0.0, 1.0, z_count, dtype=float)
        positions = []
        for theta in theta_values:
            for z_dir in z_dir_values:
                xy_dir = math.sqrt(max(1.0 - z_dir * z_dir, 0.0))
                positions.append(
                    np.array(
                        [
                            center[0] + rx * xy_dir * math.cos(theta),
                            center[1] + ry * xy_dir * math.sin(theta),
                            box_min[2] + rz * z_dir,
                        ],
                        dtype=float,
                    )
                )
        return _unique_rows(positions)

    axes = [_grid_axis(box_min[axis], box_max[axis], step) for axis in range(3)]
    grid = np.stack(np.meshgrid(axes[0], axes[1], axes[2], indexing="ij"), axis=-1).reshape(-1, 3)
    valid = [
        point
        for point in grid
        if point_in_camera_constraint(
            point,
            shape,
            box_min,
            box_max,
            constraint_data=data,
            eps=max(step * 1e-4, 1e-6),
        )
    ]
    return _unique_rows(valid)


def _select_target_points(voxelnormals, occupancy_weights, target_count):
    points = np.asarray(voxelnormals[:, :3], dtype=float)
    weights = np.asarray(occupancy_weights, dtype=float)
    if len(points) == 0:
        raise ValueError("BIP target selection requires at least one support point")
    if np.sum(weights) > 1e-12:
        center = np.average(points, axis=0, weights=weights)
    else:
        center = np.mean(points, axis=0)
    target_count = max(int(target_count), 1)
    if target_count == 1 or len(points) == 1:
        return center[None, :]

    if np.max(weights) - np.min(weights) <= 1e-8:
        pool = points
        pool_weights = np.ones(len(points), dtype=float)
    else:
        pool_count = min(len(points), max(target_count * 30, target_count))
        pool_indices = np.argsort(weights)[::-1][:pool_count]
        pool = points[pool_indices]
        pool_weights = weights[pool_indices]
        if np.max(pool_weights) > 1e-12:
            pool_weights = pool_weights / np.max(pool_weights)
        else:
            pool_weights = np.ones_like(pool_weights)

    selected = [center]
    min_distance = np.linalg.norm(pool - center[None, :], axis=1)
    while len(selected) < target_count:
        score = min_distance * (0.25 + 0.75 * pool_weights)
        next_index = int(np.argmax(score))
        next_point = pool[next_index]
        if np.linalg.norm(next_point - selected[-1]) <= 1e-8 and len(selected) > 1:
            break
        selected.append(next_point)
        min_distance = np.minimum(min_distance, np.linalg.norm(pool - next_point[None, :], axis=1))
    return _unique_rows(selected)


def _build_pose_candidates(args, voxelnormals, occupancy_weights):
    positions = _constraint_grid_positions(args)
    if len(positions) == 0:
        raise ValueError("BIP candidate grid did not produce any valid camera positions")

    max_positions = int(getattr(args, "bip_max_candidate_positions", 0))
    if max_positions > 0 and len(positions) > max_positions:
        selected = _farthest_sample_indices(positions, max_positions)
        positions = positions[selected]

    targets = _select_target_points(voxelnormals, occupancy_weights, args.bip_target_count)
    candidates = []
    seen = set()
    for position in positions:
        for target in targets:
            if np.linalg.norm(target - position) <= 1e-8:
                continue
            key = tuple(np.round(np.r_[position, target], 6))
            if key in seen:
                continue
            seen.add(key)
            candidates.append(
                PoseCandidate(
                    position=np.asarray(position, dtype=float),
                    rotation=np.asarray(getLookAtRotation(position, target), dtype=float),
                    target=np.asarray(target, dtype=float),
                )
            )

    max_candidates = int(getattr(args, "bip_max_base_candidates", 0))
    if max_candidates > 0 and len(candidates) > max_candidates:
        sample_points = np.asarray([np.r_[candidate.position, candidate.target] for candidate in candidates])
        selected = _farthest_sample_indices(sample_points, max_candidates)
        candidates = [candidates[index] for index in selected]
    return candidates


def _candidate_position_groups(candidates, decimals=6):
    groups = {}
    for pose_index, candidate in enumerate(candidates):
        key = tuple(np.round(np.asarray(candidate.position, dtype=float), decimals))
        groups.setdefault(key, []).append(pose_index)
    return list(groups.values())


def _pose_group_ids(position_groups, base_candidate_count):
    group_ids = np.full(base_candidate_count, -1, dtype=int)
    for group_id, pose_indices in enumerate(position_groups):
        for pose_index in pose_indices:
            group_ids[int(pose_index)] = group_id
    missing = np.where(group_ids < 0)[0]
    if len(missing) > 0:
        raise ValueError(f"Missing candidate position groups for pose indices: {missing.tolist()}")
    return group_ids


def _require_cpsat():
    if cp_model is None:
        raise ImportError(
            "OR-Tools CP-SAT backend requires the 'ortools' package. "
            "Install it in neof_cam with: python -m pip install ortools"
        )
    return cp_model


def _cpsat_status_name(status):
    model = _require_cpsat()
    names = {
        model.OPTIMAL: "OPTIMAL",
        model.FEASIBLE: "FEASIBLE",
        model.INFEASIBLE: "INFEASIBLE",
        model.MODEL_INVALID: "MODEL_INVALID",
        model.UNKNOWN: "UNKNOWN",
    }
    return names.get(status, f"STATUS_{status}")


def _integer_objective_weights(occupancy_weights, scale):
    weights = np.asarray(occupancy_weights, dtype=float)
    if np.any(weights < 0):
        raise ValueError("occupancy weights must be non-negative")
    scale = float(scale)
    if scale <= 0:
        raise ValueError("bip_cpsat_weight_scale must be positive")
    integer_weights = np.rint(weights * scale).astype(np.int64)
    positive = weights > 0
    integer_weights[positive & (integer_weights < 1)] = 1
    if np.any(integer_weights < 0):
        raise ValueError("scaled CP-SAT objective weights overflowed int64")
    if np.sum(integer_weights) <= 0:
        raise ValueError("CP-SAT objective weights are all zero after scaling")
    return integer_weights


def _configure_cpsat_solver(time_limit_s, args):
    model = _require_cpsat()
    solver = model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit_s)
    num_workers = int(getattr(args, "bip_cpsat_num_workers", 0))
    if num_workers > 0:
        solver.parameters.num_search_workers = num_workers
    solver.parameters.random_seed = int(getattr(args, "bip_cpsat_random_seed", 0))
    return solver


def _selected_decisions_from_cpsat(solver, x_vars, base_candidate_count, camera_count):
    selected_pose_indices = []
    selected_decision_indices = []
    for slot in range(camera_count):
        start = slot * base_candidate_count
        stop = (slot + 1) * base_candidate_count
        local_values = [solver.BooleanValue(x_vars[index]) for index in range(start, stop)]
        true_indices = [index for index, value in enumerate(local_values) if value]
        if len(true_indices) == 0:
            raise RuntimeError(f"CP-SAT solution did not select a pose for camera slot {slot}")
        local_index = int(true_indices[0])
        selected_pose_indices.append(local_index)
        selected_decision_indices.append(start + local_index)
    return selected_pose_indices, selected_decision_indices


def _cpsat_solution_message(status, solver):
    model = _require_cpsat()
    name = _cpsat_status_name(status)
    objective = solver.ObjectiveValue()
    bound = solver.BestObjectiveBound()
    message = f"OR-Tools CP-SAT status: {name}; objective={objective:.0f}; best_bound={bound:.0f}"
    if status == model.FEASIBLE:
        message = f"{message}; using best incumbent returned by OR-Tools CP-SAT"
    return message


def _solve_weighted_k_coverage_bip(
    visibility,
    base_candidate_count,
    camera_count,
    kcoverage,
    occupancy_weights,
    allow_duplicate_positions,
    time_limit_s,
    duplicate_pose_groups=None,
):
    visibility = visibility.tocsr().astype(float)
    point_count, decision_count = visibility.shape
    total_vars = decision_count + point_count
    if decision_count != base_candidate_count * camera_count:
        raise ValueError("decision visibility shape does not match base candidate count and camera count")

    objective = np.r_[np.zeros(decision_count, dtype=float), -np.asarray(occupancy_weights, dtype=float)]
    integrality = np.ones(total_vars, dtype=int)
    bounds = Bounds(np.zeros(total_vars, dtype=float), np.ones(total_vars, dtype=float))

    blocks = []
    lower_bounds = []
    upper_bounds = []

    slot_rows = []
    slot_cols = []
    slot_data = []
    for slot in range(camera_count):
        for pose_index in range(base_candidate_count):
            slot_rows.append(slot)
            slot_cols.append(slot * base_candidate_count + pose_index)
            slot_data.append(1.0)
    slot_matrix_x = sparse.csr_matrix(
        (slot_data, (slot_rows, slot_cols)),
        shape=(camera_count, decision_count),
    )
    blocks.append(sparse.hstack([slot_matrix_x, sparse.csr_matrix((camera_count, point_count))], format="csr"))
    lower_bounds.extend([1.0] * camera_count)
    upper_bounds.extend([1.0] * camera_count)

    if not allow_duplicate_positions:
        position_groups = duplicate_pose_groups or [[pose_index] for pose_index in range(base_candidate_count)]
        duplicate_rows = []
        duplicate_cols = []
        duplicate_data = []
        for group_index, pose_indices in enumerate(position_groups):
            for pose_index in pose_indices:
                for slot in range(camera_count):
                    duplicate_rows.append(group_index)
                    duplicate_cols.append(slot * base_candidate_count + pose_index)
                    duplicate_data.append(1.0)
        duplicate_matrix_x = sparse.csr_matrix(
            (duplicate_data, (duplicate_rows, duplicate_cols)),
            shape=(len(position_groups), decision_count),
        )
        blocks.append(
            sparse.hstack(
                [duplicate_matrix_x, sparse.csr_matrix((len(position_groups), point_count))],
                format="csr",
            )
        )
        lower_bounds.extend([-np.inf] * len(position_groups))
        upper_bounds.extend([1.0] * len(position_groups))

    kcoverage = int(max(1, min(int(kcoverage), camera_count)))
    coverage_matrix = sparse.hstack(
        [-visibility, float(kcoverage) * sparse.identity(point_count, format="csr")],
        format="csr",
    )
    blocks.append(coverage_matrix)
    lower_bounds.extend([-np.inf] * point_count)
    upper_bounds.extend([0.0] * point_count)

    constraint_matrix = sparse.vstack(blocks, format="csr")
    constraints = LinearConstraint(
        constraint_matrix,
        np.asarray(lower_bounds, dtype=float),
        np.asarray(upper_bounds, dtype=float),
    )

    start_time = time.time()
    result = milp(
        c=objective,
        integrality=integrality,
        bounds=bounds,
        constraints=constraints,
        options={"time_limit": float(time_limit_s), "mip_rel_gap": 0.0},
    )
    solve_time_s = time.time() - start_time
    if result.x is None:
        return BIPSolution([], [], 0.0, 0.0, False, result.message, solve_time_s)
    message = str(result.message)
    if not result.success:
        message = f"{message}; using best incumbent returned by HiGHS"

    selected_pose_indices = []
    selected_decision_indices = []
    x = result.x[:decision_count]
    for slot in range(camera_count):
        start = slot * base_candidate_count
        stop = (slot + 1) * base_candidate_count
        local_index = int(np.argmax(x[start:stop]))
        selected_pose_indices.append(local_index)
        selected_decision_indices.append(start + local_index)

    selected_visibility = visibility[:, selected_decision_indices]
    covered = np.asarray(selected_visibility.sum(axis=1)).ravel() >= kcoverage
    weights = np.asarray(occupancy_weights, dtype=float)
    weight_sum = float(np.sum(weights))
    weighted_coverage_rate = float(np.sum(weights[covered]) / weight_sum) if weight_sum > 1e-12 else float(np.mean(covered))
    objective_value = float(np.sum(weights[covered]))
    return BIPSolution(
        selected_pose_indices=selected_pose_indices,
        selected_decision_indices=selected_decision_indices,
        objective_value=objective_value,
        weighted_coverage_rate=weighted_coverage_rate,
        success=True,
        message=message,
        solve_time_s=solve_time_s,
    )


def _solve_weighted_k_coverage_cpsat(
    visibility,
    base_candidate_count,
    camera_count,
    kcoverage,
    occupancy_weights,
    allow_duplicate_positions,
    time_limit_s,
    args,
    duplicate_pose_groups=None,
):
    model_api = _require_cpsat()
    visibility = visibility.tocsr()
    point_count, decision_count = visibility.shape
    if decision_count != base_candidate_count * camera_count:
        raise ValueError("decision visibility shape does not match base candidate count and camera count")

    model = model_api.CpModel()
    x_vars = [model.NewBoolVar(f"x_{index}") for index in range(decision_count)]
    y_vars = [model.NewBoolVar(f"y_{point_index}") for point_index in range(point_count)]

    for slot in range(camera_count):
        start = slot * base_candidate_count
        stop = (slot + 1) * base_candidate_count
        model.Add(sum(x_vars[start:stop]) == 1)

    if not allow_duplicate_positions:
        position_groups = duplicate_pose_groups or [[pose_index] for pose_index in range(base_candidate_count)]
        for pose_indices in position_groups:
            group_vars = [
                x_vars[slot * base_candidate_count + pose_index]
                for pose_index in pose_indices
                for slot in range(camera_count)
            ]
            model.Add(sum(group_vars) <= 1)

    kcoverage = int(max(1, min(int(kcoverage), camera_count)))
    for point_index in range(point_count):
        start = visibility.indptr[point_index]
        stop = visibility.indptr[point_index + 1]
        decision_indices = visibility.indices[start:stop]
        if len(decision_indices) == 0:
            model.Add(y_vars[point_index] == 0)
        else:
            model.Add(sum(x_vars[int(index)] for index in decision_indices) >= kcoverage * y_vars[point_index])

    objective_weights = _integer_objective_weights(
        occupancy_weights,
        getattr(args, "bip_cpsat_weight_scale", 1000.0),
    )
    model.Maximize(sum(int(weight) * y_var for weight, y_var in zip(objective_weights, y_vars)))

    solver = _configure_cpsat_solver(time_limit_s, args)
    start_time = time.time()
    status = solver.Solve(model)
    solve_time_s = time.time() - start_time
    if status not in (model_api.OPTIMAL, model_api.FEASIBLE):
        return BIPSolution([], [], 0.0, 0.0, False, f"OR-Tools CP-SAT status: {_cpsat_status_name(status)}", solve_time_s)

    try:
        selected_pose_indices, selected_decision_indices = _selected_decisions_from_cpsat(
            solver,
            x_vars,
            base_candidate_count,
            camera_count,
        )
    except RuntimeError as exc:
        return BIPSolution([], [], 0.0, 0.0, False, str(exc), solve_time_s)

    selected_visibility = visibility[:, selected_decision_indices]
    covered = np.asarray(selected_visibility.sum(axis=1)).ravel() >= kcoverage
    weights = np.asarray(occupancy_weights, dtype=float)
    weight_sum = float(np.sum(weights))
    weighted_coverage_rate = float(np.sum(weights[covered]) / weight_sum) if weight_sum > 1e-12 else float(np.mean(covered))
    objective_value = float(np.sum(weights[covered]))
    return BIPSolution(
        selected_pose_indices=selected_pose_indices,
        selected_decision_indices=selected_decision_indices,
        objective_value=objective_value,
        weighted_coverage_rate=weighted_coverage_rate,
        success=True,
        message=_cpsat_solution_message(status, solver),
        solve_time_s=solve_time_s,
    )


def _solve_weighted_pair_coverage_bip(
    pair_visibility,
    pair_decisions,
    base_candidate_count,
    camera_count,
    occupancy_weights,
    allow_duplicate_positions,
    time_limit_s,
    duplicate_pose_groups=None,
):
    pair_visibility = pair_visibility.tocsr().astype(float)
    point_count, pair_count = pair_visibility.shape
    decision_count = base_candidate_count * camera_count
    total_vars = decision_count + pair_count + point_count
    if pair_count == 0:
        return BIPSolution([], [], 0.0, 0.0, False, "No valid BIP camera pairs are available.", 0.0)

    objective = np.r_[
        np.zeros(decision_count + pair_count, dtype=float),
        -np.asarray(occupancy_weights, dtype=float),
    ]
    integrality = np.ones(total_vars, dtype=int)
    bounds = Bounds(np.zeros(total_vars, dtype=float), np.ones(total_vars, dtype=float))

    blocks = []
    lower_bounds = []
    upper_bounds = []

    slot_rows = []
    slot_cols = []
    slot_data = []
    for slot in range(camera_count):
        for pose_index in range(base_candidate_count):
            slot_rows.append(slot)
            slot_cols.append(slot * base_candidate_count + pose_index)
            slot_data.append(1.0)
    slot_matrix_x = sparse.csr_matrix(
        (slot_data, (slot_rows, slot_cols)),
        shape=(camera_count, decision_count),
    )
    blocks.append(
        sparse.hstack(
            [slot_matrix_x, sparse.csr_matrix((camera_count, pair_count + point_count))],
            format="csr",
        )
    )
    lower_bounds.extend([1.0] * camera_count)
    upper_bounds.extend([1.0] * camera_count)

    if not allow_duplicate_positions:
        position_groups = duplicate_pose_groups or [[pose_index] for pose_index in range(base_candidate_count)]
        duplicate_rows = []
        duplicate_cols = []
        duplicate_data = []
        for group_index, pose_indices in enumerate(position_groups):
            for pose_index in pose_indices:
                for slot in range(camera_count):
                    duplicate_rows.append(group_index)
                    duplicate_cols.append(slot * base_candidate_count + pose_index)
                    duplicate_data.append(1.0)
        duplicate_matrix_x = sparse.csr_matrix(
            (duplicate_data, (duplicate_rows, duplicate_cols)),
            shape=(len(position_groups), decision_count),
        )
        blocks.append(
            sparse.hstack(
                [duplicate_matrix_x, sparse.csr_matrix((len(position_groups), pair_count + point_count))],
                format="csr",
            )
        )
        lower_bounds.extend([-np.inf] * len(position_groups))
        upper_bounds.extend([1.0] * len(position_groups))

    link_rows = []
    link_cols = []
    link_data = []
    for pair_index, (first_decision, second_decision) in enumerate(pair_decisions):
        z_col = decision_count + pair_index
        first_row = 2 * pair_index
        second_row = first_row + 1
        link_rows.extend([first_row, first_row, second_row, second_row])
        link_cols.extend([z_col, first_decision, z_col, second_decision])
        link_data.extend([1.0, -1.0, 1.0, -1.0])
    link_matrix = sparse.csr_matrix(
        (link_data, (link_rows, link_cols)),
        shape=(2 * pair_count, total_vars),
    )
    blocks.append(link_matrix)
    lower_bounds.extend([-np.inf] * (2 * pair_count))
    upper_bounds.extend([0.0] * (2 * pair_count))

    coverage_matrix = sparse.hstack(
        [
            sparse.csr_matrix((point_count, decision_count)),
            -pair_visibility,
            sparse.identity(point_count, format="csr"),
        ],
        format="csr",
    )
    blocks.append(coverage_matrix)
    lower_bounds.extend([-np.inf] * point_count)
    upper_bounds.extend([0.0] * point_count)

    constraint_matrix = sparse.vstack(blocks, format="csr")
    constraints = LinearConstraint(
        constraint_matrix,
        np.asarray(lower_bounds, dtype=float),
        np.asarray(upper_bounds, dtype=float),
    )

    start_time = time.time()
    result = milp(
        c=objective,
        integrality=integrality,
        bounds=bounds,
        constraints=constraints,
        options={"time_limit": float(time_limit_s), "mip_rel_gap": 0.0},
    )
    solve_time_s = time.time() - start_time
    if result.x is None:
        return BIPSolution([], [], 0.0, 0.0, False, result.message, solve_time_s)
    message = str(result.message)
    if not result.success:
        message = f"{message}; using best incumbent returned by HiGHS"

    selected_pose_indices = []
    selected_decision_indices = []
    x = result.x[:decision_count]
    for slot in range(camera_count):
        start = slot * base_candidate_count
        stop = (slot + 1) * base_candidate_count
        local_index = int(np.argmax(x[start:stop]))
        selected_pose_indices.append(local_index)
        selected_decision_indices.append(start + local_index)

    selected = set(selected_decision_indices)
    active_pair_indices = [
        pair_index
        for pair_index, (first_decision, second_decision) in enumerate(pair_decisions)
        if first_decision in selected and second_decision in selected
    ]
    if active_pair_indices:
        covered = np.asarray(pair_visibility[:, active_pair_indices].sum(axis=1)).ravel() > 0
    else:
        covered = np.zeros(point_count, dtype=bool)
    weights = np.asarray(occupancy_weights, dtype=float)
    weight_sum = float(np.sum(weights))
    weighted_coverage_rate = float(np.sum(weights[covered]) / weight_sum) if weight_sum > 1e-12 else float(np.mean(covered))
    objective_value = float(np.sum(weights[covered]))
    return BIPSolution(
        selected_pose_indices=selected_pose_indices,
        selected_decision_indices=selected_decision_indices,
        objective_value=objective_value,
        weighted_coverage_rate=weighted_coverage_rate,
        success=True,
        message=message,
        solve_time_s=solve_time_s,
    )


def _solve_weighted_pair_coverage_cpsat(
    pair_visibility,
    pair_decisions,
    base_candidate_count,
    camera_count,
    occupancy_weights,
    allow_duplicate_positions,
    time_limit_s,
    args,
    duplicate_pose_groups=None,
):
    model_api = _require_cpsat()
    pair_visibility = pair_visibility.tocsr()
    point_count, pair_count = pair_visibility.shape
    decision_count = base_candidate_count * camera_count
    if pair_count == 0:
        return BIPSolution([], [], 0.0, 0.0, False, "No valid BIP camera pairs are available.", 0.0)

    model = model_api.CpModel()
    x_vars = [model.NewBoolVar(f"x_{index}") for index in range(decision_count)]
    z_vars = [model.NewBoolVar(f"z_{pair_index}") for pair_index in range(pair_count)]
    y_vars = [model.NewBoolVar(f"y_{point_index}") for point_index in range(point_count)]

    for slot in range(camera_count):
        start = slot * base_candidate_count
        stop = (slot + 1) * base_candidate_count
        model.Add(sum(x_vars[start:stop]) == 1)

    if not allow_duplicate_positions:
        position_groups = duplicate_pose_groups or [[pose_index] for pose_index in range(base_candidate_count)]
        for pose_indices in position_groups:
            group_vars = [
                x_vars[slot * base_candidate_count + pose_index]
                for pose_index in pose_indices
                for slot in range(camera_count)
            ]
            model.Add(sum(group_vars) <= 1)

    for pair_index, (first_decision, second_decision) in enumerate(pair_decisions):
        z_var = z_vars[pair_index]
        model.Add(z_var <= x_vars[int(first_decision)])
        model.Add(z_var <= x_vars[int(second_decision)])

    for point_index in range(point_count):
        start = pair_visibility.indptr[point_index]
        stop = pair_visibility.indptr[point_index + 1]
        pair_indices = pair_visibility.indices[start:stop]
        if len(pair_indices) == 0:
            model.Add(y_vars[point_index] == 0)
        else:
            model.Add(sum(z_vars[int(index)] for index in pair_indices) >= y_vars[point_index])

    objective_weights = _integer_objective_weights(
        occupancy_weights,
        getattr(args, "bip_cpsat_weight_scale", 1000.0),
    )
    model.Maximize(sum(int(weight) * y_var for weight, y_var in zip(objective_weights, y_vars)))

    solver = _configure_cpsat_solver(time_limit_s, args)
    start_time = time.time()
    status = solver.Solve(model)
    solve_time_s = time.time() - start_time
    if status not in (model_api.OPTIMAL, model_api.FEASIBLE):
        return BIPSolution([], [], 0.0, 0.0, False, f"OR-Tools CP-SAT status: {_cpsat_status_name(status)}", solve_time_s)

    try:
        selected_pose_indices, selected_decision_indices = _selected_decisions_from_cpsat(
            solver,
            x_vars,
            base_candidate_count,
            camera_count,
        )
    except RuntimeError as exc:
        return BIPSolution([], [], 0.0, 0.0, False, str(exc), solve_time_s)

    selected = set(selected_decision_indices)
    active_pair_indices = [
        pair_index
        for pair_index, (first_decision, second_decision) in enumerate(pair_decisions)
        if first_decision in selected and second_decision in selected
    ]
    if active_pair_indices:
        covered = np.asarray(pair_visibility[:, active_pair_indices].sum(axis=1)).ravel() > 0
    else:
        covered = np.zeros(point_count, dtype=bool)
    weights = np.asarray(occupancy_weights, dtype=float)
    weight_sum = float(np.sum(weights))
    weighted_coverage_rate = float(np.sum(weights[covered]) / weight_sum) if weight_sum > 1e-12 else float(np.mean(covered))
    objective_value = float(np.sum(weights[covered]))
    return BIPSolution(
        selected_pose_indices=selected_pose_indices,
        selected_decision_indices=selected_decision_indices,
        objective_value=objective_value,
        weighted_coverage_rate=weighted_coverage_rate,
        success=True,
        message=_cpsat_solution_message(status, solver),
        solve_time_s=solve_time_s,
    )


def _greedy_weighted_pair_coverage(
    pair_visibility,
    pair_decisions,
    base_candidate_count,
    camera_count,
    occupancy_weights,
    allow_duplicate_positions,
    message_prefix,
    duplicate_pose_groups=None,
):
    pair_visibility = pair_visibility.tocsr().astype(float)
    weights = np.asarray(occupancy_weights, dtype=float)
    weight_sum = float(np.sum(weights))
    decision_count = base_candidate_count * camera_count
    position_groups = duplicate_pose_groups or [[pose_index] for pose_index in range(base_candidate_count)]
    pose_group_ids = _pose_group_ids(position_groups, base_candidate_count)
    pair_by_decision = [[] for _ in range(decision_count)]
    for pair_index, (first_decision, second_decision) in enumerate(pair_decisions):
        pair_by_decision[first_decision].append(pair_index)
        pair_by_decision[second_decision].append(pair_index)

    def active_pair_indices(selected_decisions):
        selected_set = {decision for decision in selected_decisions if decision is not None}
        if len(selected_set) < 2:
            return []
        return [
            pair_index
            for pair_index, (first_decision, second_decision) in enumerate(pair_decisions)
            if first_decision in selected_set and second_decision in selected_set
        ]

    def weighted_coverage(pair_indices):
        if not pair_indices:
            return 0.0, np.zeros(pair_visibility.shape[0], dtype=bool)
        covered = np.asarray(pair_visibility[:, pair_indices].sum(axis=1)).ravel() > 0
        return float(np.sum(weights[covered])), covered

    selected_decisions = [None] * camera_count
    used_position_groups = set()
    for slot in range(camera_count):
        best_decision = None
        best_score = -np.inf
        for pose_index in range(base_candidate_count):
            pose_group_id = int(pose_group_ids[pose_index])
            if not allow_duplicate_positions and pose_group_id in used_position_groups:
                continue
            decision = slot * base_candidate_count + pose_index
            trial = list(selected_decisions)
            trial[slot] = decision
            pair_indices = active_pair_indices(trial)
            if pair_indices:
                score, _ = weighted_coverage(pair_indices)
            else:
                score, _ = weighted_coverage(pair_by_decision[decision])
            if score > best_score:
                best_score = score
                best_decision = decision
        if best_decision is None:
            return BIPSolution([], [], 0.0, 0.0, False, "Greedy pair fallback failed.", 0.0)
        selected_decisions[slot] = best_decision
        used_position_groups.add(int(pose_group_ids[best_decision % base_candidate_count]))

    best_pair_indices = active_pair_indices(selected_decisions)
    best_score, covered = weighted_coverage(best_pair_indices)
    improved = True
    while improved:
        improved = False
        for slot in range(camera_count):
            current_decision = selected_decisions[slot]
            current_pose = current_decision % base_candidate_count
            for pose_index in range(base_candidate_count):
                if pose_index == current_pose:
                    continue
                if not allow_duplicate_positions:
                    other_used = {
                        int(pose_group_ids[decision % base_candidate_count])
                        for index, decision in enumerate(selected_decisions)
                        if index != slot and decision is not None
                    }
                    if int(pose_group_ids[pose_index]) in other_used:
                        continue
                trial = list(selected_decisions)
                trial[slot] = slot * base_candidate_count + pose_index
                trial_pair_indices = active_pair_indices(trial)
                score, trial_covered = weighted_coverage(trial_pair_indices)
                if score > best_score + 1e-8:
                    selected_decisions = trial
                    best_score = score
                    covered = trial_covered
                    improved = True
                    break
            if improved:
                break

    selected_decision_indices = [int(decision) for decision in selected_decisions]
    selected_pose_indices = [int(decision % base_candidate_count) for decision in selected_decision_indices]
    weighted_coverage_rate = best_score / weight_sum if weight_sum > 1e-12 else float(np.mean(covered))
    return BIPSolution(
        selected_pose_indices=selected_pose_indices,
        selected_decision_indices=selected_decision_indices,
        objective_value=float(best_score),
        weighted_coverage_rate=float(weighted_coverage_rate),
        success=True,
        message=f"{message_prefix}; using greedy pair fallback",
        solve_time_s=0.0,
    )


class BIPCameraOpt:
    def __init__(self, args, voxelnormals, occupancy_weights, minbound, scale, posepath, writer, geometry_metadata=None):
        del minbound, writer
        self.args = args
        self.voxelnormals = np.asarray(voxelnormals, dtype=np.float32)
        self.geometry_metadata = geometry_metadata or {}
        if occupancy_weights is None:
            occupancy_weights = np.ones(len(voxelnormals), dtype=np.float32)
        self.voxel_occupancy_np = np.asarray(occupancy_weights, dtype=np.float32)
        if self.voxel_occupancy_np.shape != (len(self.voxelnormals),):
            raise ValueError(
                f"occupancy_weights must contain exactly {len(self.voxelnormals)} values, "
                f"got {self.voxel_occupancy_np.shape}"
            )
        self.voxel_occupancy_sum_np = float(np.sum(self.voxel_occupancy_np))
        self.scale = scale
        self.posepath = posepath
        self.free_space_support = uses_free_space_support(args)
        self.fov_only_visibility = self.free_space_support or uses_fov_only_target_visibility(args)
        self.preprocessing_summary = {}

    def coverage_gap_from_visibility(self, voxel_visibility, weighted=True):
        coverage = np.sum(voxel_visibility, axis=-1)
        kcoverage = max(int(self.args.kcoverage), 1)
        squared_gap = np.square(np.maximum(kcoverage - coverage, np.zeros(len(coverage))))
        if weighted:
            gap = np.sum(self.voxel_occupancy_np * squared_gap) / (
                kcoverage * kcoverage * self.voxel_occupancy_sum_np
            )
        else:
            gap = np.sum(squared_gap) / (kcoverage * kcoverage * len(coverage))
        return float(gap), coverage

    def print_coverage_summary(self, title, position, rotation):
        _, visibility = voxel_model(self.args, self.voxelnormals, rotation, position)
        weighted_gap, coverage = self.coverage_gap_from_visibility(visibility, weighted=True)
        unweighted_gap = None
        if self.args.occupancy_map_enable:
            unweighted_gap, _ = self.coverage_gap_from_visibility(visibility, weighted=False)
        print(title)
        if self.fov_only_visibility:
            visibility_label = "free-space box" if self.free_space_support else "FoV-only target visibility"
            print(f"  Target support: {visibility_label}")
        print(f"  Weighted voxel K-coverage deficit (normalized): {weighted_gap:.4f}")
        if unweighted_gap is not None:
            print(f"  Unweighted voxel K-coverage deficit (normalized): {unweighted_gap:.4f}")
        print(f"  Voxels seen by >=1 camera: {int(np.sum(np.sign(coverage)))}/{len(coverage)}")
        per_camera = np.sum(visibility > 0, axis=0)
        print(
            "  Per-camera visible points: "
            + ", ".join(
                f"cam{idx:02d}={int(count)}/{len(coverage)}"
                for idx, count in enumerate(per_camera)
            )
        )
        if getattr(self.args, "require_all_cameras_coverage", False):
            print(
                "  Voxels seen by every camera: "
                f"{int(np.sum(coverage == self.args.cameranum))}/{len(coverage)}"
            )

    def visible_indices_for_decision(self, slot, candidate):
        camera_model = get_camera_model(slot)
        if self.fov_only_visibility:
            visible_indices, _ = get_visible_points_free_space(
                self.voxelnormals,
                candidate.position,
                candidate.rotation,
                camera_model=camera_model,
            )
        else:
            _, visible_indices = get_visiblep_opt(
                self.voxelnormals,
                candidate.position,
                candidate.rotation,
                200,
                camera_model=camera_model,
            )
        return np.asarray(visible_indices, dtype=int)

    def build_decision_visibility(self, candidates):
        rows = []
        cols = []
        visible_counts = np.zeros((self.args.cameranum, len(candidates)), dtype=int)
        for slot in range(self.args.cameranum):
            for pose_index, candidate in enumerate(candidates):
                decision_index = slot * len(candidates) + pose_index
                visible_indices = self.visible_indices_for_decision(slot, candidate)
                visible_counts[slot, pose_index] = len(visible_indices)
                if len(visible_indices) == 0:
                    continue
                rows.extend(int(index) for index in visible_indices)
                cols.extend([decision_index] * len(visible_indices))
        data = np.ones(len(rows), dtype=np.int8)
        visibility = sparse.csr_matrix(
            (data, (rows, cols)),
            shape=(len(self.voxelnormals), self.args.cameranum * len(candidates)),
            dtype=np.int8,
        )
        return visibility, visible_counts

    def filter_candidates_by_visibility(self, candidates, visibility, visible_counts):
        preprocess_enable = bool(getattr(self.args, "bip_preprocess_enable", True))
        min_visible = int(getattr(self.args, "bip_min_visible_points", 0))
        min_fraction = float(getattr(self.args, "bip_min_visible_fraction", 0.0))
        fraction_threshold = int(math.ceil(min_fraction * len(self.voxelnormals))) if min_fraction > 0 else 0
        threshold = max(min_visible, fraction_threshold)
        score_mode = getattr(self.args, "bip_visibility_filter_stat", "max")
        self.preprocessing_summary = {
            "enabled": bool(preprocess_enable),
            "input_candidate_count": int(len(candidates)),
            "min_visible_points": int(min_visible),
            "min_visible_fraction": float(min_fraction),
            "effective_min_visible_points": int(threshold),
            "visibility_filter_stat": score_mode,
        }
        if not preprocess_enable or threshold <= 0 or len(candidates) == 0:
            self.preprocessing_summary.update(
                {
                    "output_candidate_count": int(len(candidates)),
                    "removed_candidate_count": 0,
                    "skipped": bool(not preprocess_enable or threshold <= 0),
                }
            )
            return candidates, visibility, visible_counts

        if score_mode == "mean":
            candidate_scores = np.mean(visible_counts, axis=0)
        elif score_mode == "min":
            candidate_scores = np.min(visible_counts, axis=0)
        else:
            candidate_scores = np.max(visible_counts, axis=0)

        keep_mask = candidate_scores >= threshold
        removed_count = int(len(candidates) - np.sum(keep_mask))
        self.preprocessing_summary.update(
            {
                "score_min": float(np.min(candidate_scores)) if len(candidate_scores) else 0.0,
                "score_median": float(np.median(candidate_scores)) if len(candidate_scores) else 0.0,
                "score_max": float(np.max(candidate_scores)) if len(candidate_scores) else 0.0,
                "removed_candidate_count": removed_count,
            }
        )
        if int(np.sum(keep_mask)) < self.args.cameranum:
            print(
                "BIP preprocessing visibility filter skipped because it would leave fewer candidates "
                f"than cameras ({int(np.sum(keep_mask))} < {self.args.cameranum})."
            )
            self.preprocessing_summary.update(
                {
                    "output_candidate_count": int(len(candidates)),
                    "removed_candidate_count": 0,
                    "skipped": True,
                    "skip_reason": "fewer_candidates_than_cameras",
                }
            )
            return candidates, visibility, visible_counts

        if np.all(keep_mask):
            print(
                "BIP preprocessing visibility filter kept all "
                f"{len(candidates)} candidates | stat={score_mode}, threshold={threshold}, "
                f"score range=[{self.preprocessing_summary['score_min']:.1f}, "
                f"{self.preprocessing_summary['score_max']:.1f}]"
            )
            self.preprocessing_summary.update(
                {
                    "output_candidate_count": int(len(candidates)),
                    "removed_candidate_count": 0,
                    "skipped": False,
                }
            )
            return candidates, visibility, visible_counts

        filtered_candidates = [candidate for candidate, keep in zip(candidates, keep_mask) if keep]
        print(
            f"BIP preprocessing visibility filter kept {len(filtered_candidates)}/{len(candidates)} "
            f"base poses | stat={score_mode}, threshold={threshold}, "
            f"removed={removed_count}"
        )
        self.preprocessing_summary.update(
            {
                "output_candidate_count": int(len(filtered_candidates)),
                "removed_candidate_count": removed_count,
                "skipped": False,
            }
        )
        filtered_visibility, filtered_counts = self.build_decision_visibility(filtered_candidates)
        return filtered_candidates, filtered_visibility, filtered_counts

    def reduce_pair_angle_candidates(self, candidates, visible_counts):
        limit = int(getattr(self.args, "bip_pair_candidate_limit", 0))
        if self.args.bip_coverage_mode != "pair_angle" or limit <= 0 or len(candidates) <= limit:
            return candidates, None
        if limit < self.args.cameranum and not self.args.bip_allow_duplicate_positions:
            raise ValueError(
                "bip_pair_candidate_limit must be at least cameranum when duplicate positions are disabled"
            )

        candidate_scores = np.mean(visible_counts, axis=0)
        if self.args.bip_allow_duplicate_positions:
            keep_indices = np.argsort(candidate_scores)[-limit:]
        else:
            position_groups = _candidate_position_groups(candidates)
            representatives = [
                max(pose_indices, key=lambda pose_index: candidate_scores[int(pose_index)])
                for pose_indices in position_groups
            ]
            representatives = sorted(
                representatives,
                key=lambda pose_index: candidate_scores[int(pose_index)],
                reverse=True,
            )
            keep_order = list(representatives[:limit])
            if len(keep_order) < limit:
                selected = set(int(index) for index in keep_order)
                remaining = [
                    int(index)
                    for index in np.argsort(candidate_scores)[::-1]
                    if int(index) not in selected
                ]
                keep_order.extend(remaining[: limit - len(keep_order)])
            keep_indices = np.asarray(keep_order, dtype=int)
        keep_indices = np.sort(keep_indices)
        reduced_candidates = [candidates[index] for index in keep_indices]
        print(
            f"  Pair-angle candidate reduction kept {len(reduced_candidates)}/{len(candidates)} "
            "base poses before pair construction."
        )
        return reduced_candidates, keep_indices

    def build_pair_visibility(self, candidates, decision_visibility):
        base_candidate_count = len(candidates)
        position_groups = _candidate_position_groups(candidates)
        pose_group_ids = _pose_group_ids(position_groups, base_candidate_count)
        possible_pairs = (
            self.args.cameranum
            * (self.args.cameranum - 1)
            * base_candidate_count
            * base_candidate_count
            // 2
        )
        max_pair_variables = int(getattr(self.args, "bip_max_pair_variables", 0))
        if max_pair_variables > 0 and possible_pairs > max_pair_variables:
            raise ValueError(
                "BIP pair-angle mode would create too many candidate decision pairs "
                f"({possible_pairs} > {max_pair_variables}). Increase bip_candidate_position_step, "
                "reduce bip_target_count/bip_max_base_candidates, or increase bip_max_pair_variables."
            )

        min_angle = math.radians(float(self.args.bip_min_triangulation_angle_deg))
        max_angle = math.radians(float(self.args.bip_max_triangulation_angle_deg))
        points = self.voxelnormals[:, :3].astype(float)
        visibility_dense = decision_visibility.toarray().astype(bool)

        rows = []
        cols = []
        pair_decisions = []
        for first_slot in range(self.args.cameranum - 1):
            for second_slot in range(first_slot + 1, self.args.cameranum):
                for first_pose in range(base_candidate_count):
                    first_decision = first_slot * base_candidate_count + first_pose
                    first_visible = visibility_dense[:, first_decision]
                    first_position = candidates[first_pose].position
                    for second_pose in range(base_candidate_count):
                        if (
                            not self.args.bip_allow_duplicate_positions
                            and pose_group_ids[first_pose] == pose_group_ids[second_pose]
                        ):
                            continue
                        second_decision = second_slot * base_candidate_count + second_pose
                        common = np.flatnonzero(first_visible & visibility_dense[:, second_decision])
                        if len(common) == 0:
                            continue
                        second_position = candidates[second_pose].position
                        first_rays = first_position[None, :] - points[common]
                        second_rays = second_position[None, :] - points[common]
                        first_norm = np.linalg.norm(first_rays, axis=1)
                        second_norm = np.linalg.norm(second_rays, axis=1)
                        valid_norm = (first_norm > 1e-8) & (second_norm > 1e-8)
                        if not np.any(valid_norm):
                            continue
                        safe_common = common[valid_norm]
                        first_unit = first_rays[valid_norm] / first_norm[valid_norm, None]
                        second_unit = second_rays[valid_norm] / second_norm[valid_norm, None]
                        cosines = np.sum(first_unit * second_unit, axis=1)
                        angles = np.arccos(np.clip(cosines, -1.0, 1.0))
                        valid = safe_common[(angles >= min_angle) & (angles <= max_angle)]
                        if len(valid) == 0:
                            continue
                        pair_index = len(pair_decisions)
                        pair_decisions.append((first_decision, second_decision))
                        rows.extend(int(index) for index in valid)
                        cols.extend([pair_index] * len(valid))

        pair_visibility = sparse.csr_matrix(
            (np.ones(len(rows), dtype=np.int8), (rows, cols)),
            shape=(len(self.voxelnormals), len(pair_decisions)),
            dtype=np.int8,
        )
        return pair_visibility, pair_decisions

    def save_bip_artifacts(self, candidates, solution, selected_position, selected_rotation):
        candidate_positions = np.asarray([candidate.position for candidate in candidates], dtype=np.float32)
        candidate_rotations = np.asarray([candidate.rotation for candidate in candidates], dtype=np.float32)
        candidate_targets = np.asarray([candidate.target for candidate in candidates], dtype=np.float32)
        np.savez(
            os.path.join(self.posepath, "bip_candidates.npz"),
            candidate_positions=candidate_positions,
            candidate_rotations=candidate_rotations,
            candidate_targets=candidate_targets,
            selected_pose_indices=np.asarray(solution.selected_pose_indices, dtype=np.int64),
            selected_decision_indices=np.asarray(solution.selected_decision_indices, dtype=np.int64),
            selected_positions=np.asarray(selected_position, dtype=np.float32),
            selected_rotations=np.asarray(selected_rotation, dtype=np.float32),
        )
        payload = {
            "solver": "bip",
            "success": bool(solution.success),
            "message": solution.message,
            "objective_value": float(solution.objective_value),
            "weighted_coverage_rate": float(solution.weighted_coverage_rate),
            "solve_time_s": float(solution.solve_time_s),
            "base_candidate_count": int(len(candidates)),
            "decision_variable_count": int(len(candidates) * self.args.cameranum),
            "selected_pose_indices": [int(index) for index in solution.selected_pose_indices],
            "selected_decision_indices": [int(index) for index in solution.selected_decision_indices],
            "bip_candidate_position_step": float(self.args.bip_candidate_position_step),
            "bip_target_count": int(self.args.bip_target_count),
            "bip_pair_candidate_limit": int(self.args.bip_pair_candidate_limit),
            "bip_preprocess_enable": bool(getattr(self.args, "bip_preprocess_enable", True)),
            "bip_visibility_filter_stat": getattr(self.args, "bip_visibility_filter_stat", "max"),
            "bip_min_visible_points": int(self.args.bip_min_visible_points),
            "bip_min_visible_fraction": float(getattr(self.args, "bip_min_visible_fraction", 0.0)),
            "bip_time_limit": float(self.args.bip_time_limit),
            "bip_coverage_mode": self.args.bip_coverage_mode,
            "require_all_cameras_coverage": bool(getattr(self.args, "require_all_cameras_coverage", False)),
            "effective_kcoverage": int(self.args.kcoverage),
            "bip_solver_backend": getattr(self.args, "bip_solver_backend", "scipy_highs"),
            "bip_cpsat_weight_scale": float(getattr(self.args, "bip_cpsat_weight_scale", 1000.0)),
            "bip_cpsat_num_workers": int(getattr(self.args, "bip_cpsat_num_workers", 0)),
            "bip_cpsat_random_seed": int(getattr(self.args, "bip_cpsat_random_seed", 0)),
            "bip_min_triangulation_angle_deg": float(self.args.bip_min_triangulation_angle_deg),
            "bip_max_triangulation_angle_deg": float(self.args.bip_max_triangulation_angle_deg),
            "bip_allow_duplicate_positions": bool(self.args.bip_allow_duplicate_positions),
            "preprocessing": self.preprocessing_summary,
        }
        with open(os.path.join(self.posepath, "bip_solution.json"), "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    def opt(self, camerapose):
        initial_position = camerapose[:, :3].detach().cpu().numpy()
        initial_rotation_np = compute_rotation_matrix_from_ortho6d(camerapose[:, 3:]).detach().cpu().numpy()
        self.print_coverage_summary("Initial placement quality", initial_position, initial_rotation_np)

        print("BIP candidate generation")
        candidate_start = time.time()
        candidates = _build_pose_candidates(self.args, self.voxelnormals, self.voxel_occupancy_np)
        print(
            f"  Generated {len(candidates)} base candidate poses "
            f"in {format_seconds(time.time() - candidate_start)}"
        )
        if len(candidates) < self.args.cameranum and not self.args.bip_allow_duplicate_positions:
            raise ValueError(
                "BIP generated fewer candidate poses than cameras. Increase candidate density, "
                "increase bip_target_count, or set bip_allow_duplicate_positions=true."
            )

        visibility_start = time.time()
        decision_visibility, visible_counts = self.build_decision_visibility(candidates)
        candidates, decision_visibility, visible_counts = self.filter_candidates_by_visibility(
            candidates,
            decision_visibility,
            visible_counts,
        )
        duplicate_pose_groups = _candidate_position_groups(candidates)
        if (
            not self.args.bip_allow_duplicate_positions
            and len(duplicate_pose_groups) < self.args.cameranum
        ):
            raise ValueError(
                "BIP generated fewer unique candidate positions than cameras. Increase candidate density, "
                "increase bip_target_count, or set bip_allow_duplicate_positions=true."
            )
        print(
            "  Candidate position groups: "
            f"{len(duplicate_pose_groups)} unique positions for {len(candidates)} pose candidates"
        )
        reduced_candidates, _ = self.reduce_pair_angle_candidates(candidates, visible_counts)
        if reduced_candidates is not candidates:
            candidates = reduced_candidates
            decision_visibility, visible_counts = self.build_decision_visibility(candidates)
            duplicate_pose_groups = _candidate_position_groups(candidates)
            if (
                not self.args.bip_allow_duplicate_positions
                and len(duplicate_pose_groups) < self.args.cameranum
            ):
                raise ValueError(
                    "BIP pair-angle candidate reduction left fewer unique positions than cameras. "
                    "Increase bip_pair_candidate_limit or set bip_allow_duplicate_positions=true."
                )
            print(
                "  Candidate position groups after pair reduction: "
                f"{len(duplicate_pose_groups)} unique positions for {len(candidates)} pose candidates"
            )
        print(
            f"  Built sparse decision visibility matrix {decision_visibility.shape} "
            f"with {decision_visibility.nnz} nonzeros in {format_seconds(time.time() - visibility_start)}"
        )
        print(
            "  Candidate visibility per slot: "
            f"min={int(np.min(visible_counts))}, "
            f"median={float(np.median(visible_counts)):.1f}, "
            f"max={int(np.max(visible_counts))}"
        )

        solver_backend = getattr(self.args, "bip_solver_backend", "scipy_highs")
        print(f"BIP solve | mode={self.args.bip_coverage_mode} | backend={solver_backend}")
        if self.args.bip_coverage_mode == "kcoverage":
            if solver_backend == "scipy_highs":
                solution = _solve_weighted_k_coverage_bip(
                    decision_visibility,
                    base_candidate_count=len(candidates),
                    camera_count=self.args.cameranum,
                    kcoverage=self.args.kcoverage,
                    occupancy_weights=self.voxel_occupancy_np,
                    allow_duplicate_positions=self.args.bip_allow_duplicate_positions,
                    time_limit_s=self.args.bip_time_limit,
                    duplicate_pose_groups=duplicate_pose_groups,
                )
            elif solver_backend == "ortools_cpsat":
                solution = _solve_weighted_k_coverage_cpsat(
                    decision_visibility,
                    base_candidate_count=len(candidates),
                    camera_count=self.args.cameranum,
                    kcoverage=self.args.kcoverage,
                    occupancy_weights=self.voxel_occupancy_np,
                    allow_duplicate_positions=self.args.bip_allow_duplicate_positions,
                    time_limit_s=self.args.bip_time_limit,
                    args=self.args,
                    duplicate_pose_groups=duplicate_pose_groups,
                )
            else:
                raise ValueError(f"Unsupported BIP solver backend: {solver_backend}")
        elif self.args.bip_coverage_mode == "pair_angle":
            pair_start = time.time()
            pair_visibility, pair_decisions = self.build_pair_visibility(candidates, decision_visibility)
            print(
                f"  Built pair-angle visibility matrix {pair_visibility.shape} "
                f"with {pair_visibility.nnz} nonzeros in {format_seconds(time.time() - pair_start)}"
            )
            if solver_backend == "scipy_highs":
                solution = _solve_weighted_pair_coverage_bip(
                    pair_visibility,
                    pair_decisions,
                    base_candidate_count=len(candidates),
                    camera_count=self.args.cameranum,
                    occupancy_weights=self.voxel_occupancy_np,
                    allow_duplicate_positions=self.args.bip_allow_duplicate_positions,
                    time_limit_s=self.args.bip_time_limit,
                    duplicate_pose_groups=duplicate_pose_groups,
                )
            elif solver_backend == "ortools_cpsat":
                solution = _solve_weighted_pair_coverage_cpsat(
                    pair_visibility,
                    pair_decisions,
                    base_candidate_count=len(candidates),
                    camera_count=self.args.cameranum,
                    occupancy_weights=self.voxel_occupancy_np,
                    allow_duplicate_positions=self.args.bip_allow_duplicate_positions,
                    time_limit_s=self.args.bip_time_limit,
                    args=self.args,
                    duplicate_pose_groups=duplicate_pose_groups,
                )
            else:
                raise ValueError(f"Unsupported BIP solver backend: {solver_backend}")
            if not solution.success:
                print(f"  Pair-angle BIP failed ({solution.message}); falling back to greedy pair selection.")
                solution = _greedy_weighted_pair_coverage(
                    pair_visibility,
                    pair_decisions,
                    base_candidate_count=len(candidates),
                    camera_count=self.args.cameranum,
                    occupancy_weights=self.voxel_occupancy_np,
                    allow_duplicate_positions=self.args.bip_allow_duplicate_positions,
                    message_prefix=solution.message,
                    duplicate_pose_groups=duplicate_pose_groups,
                )
        else:
            raise ValueError(f"Unsupported BIP coverage mode: {self.args.bip_coverage_mode}")
        if not solution.success:
            raise RuntimeError(f"BIP solver failed: {solution.message}")
        print(
            f"  Solver status: {solution.message}\n"
            f"  Weighted covered support rate: {solution.weighted_coverage_rate * 100.0:.2f}%\n"
            f"  Solve time: {format_seconds(solution.solve_time_s)}"
        )

        selected_position = np.asarray(
            [candidates[index].position for index in solution.selected_pose_indices],
            dtype=np.float32,
        )
        selected_rotation = np.asarray(
            [candidates[index].rotation for index in solution.selected_pose_indices],
            dtype=np.float32,
        )
        for slot, pose_index in enumerate(solution.selected_pose_indices):
            candidate = candidates[pose_index]
            print(
                f"  Camera {slot:02d} <- candidate {pose_index:04d} | "
                f"position={format_vector(candidate.position)} | "
                f"target={format_vector(candidate.target)}"
            )

        self.print_coverage_summary("Best placement quality", selected_position, selected_rotation)
        saveTrainingResult(
            os.path.join(self.posepath, "after.npy"),
            npToTensor(selected_position),
            npToTensor(selected_rotation),
            self.scale,
        )
        self.save_bip_artifacts(candidates, solution, selected_position, selected_rotation)
        return selected_position, selected_rotation
