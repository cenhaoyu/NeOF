import os
import re
from datetime import datetime

from camera_constraints import CAMERA_CONSTRAINT_SHAPES


SOLVER_NAMES = ("neof", "bip")
TIMESTAMP_PATTERN = re.compile(r"^(?P<base>.+)_(?P<timestamp>\d{14})$")


def make_run_timestamp():
    return datetime.now().strftime("%Y%m%d%H%M%S")


def normalize_run_timestamp(timestamp):
    if timestamp in (None, ""):
        return None
    timestamp = str(timestamp)
    if not re.fullmatch(r"\d{14}", timestamp):
        raise ValueError("run_timestamp must use YYYYMMDDHHMMSS format, for example 20260512094530")
    return timestamp


def _split_path_leaf(relative_path):
    normalized_relative = os.path.normpath(relative_path)
    parent_dir, leaf_dir = os.path.split(normalized_relative)
    if leaf_dir in ("", ".", os.sep):
        raise ValueError("path must end with a valid directory name")
    return parent_dir, leaf_dir


def _strip_known_suffixes(leaf_dir):
    existing_timestamp = None
    match = TIMESTAMP_PATTERN.match(leaf_dir)
    if match:
        leaf_dir = match.group("base")
        existing_timestamp = match.group("timestamp")

    for shape_name in CAMERA_CONSTRAINT_SHAPES:
        shape_suffix = f"_{shape_name}"
        if leaf_dir.endswith(shape_suffix):
            leaf_dir = leaf_dir[: -len(shape_suffix)]
            break

    for solver_name in SOLVER_NAMES:
        solver_suffix = f"_{solver_name}"
        if leaf_dir.endswith(solver_suffix):
            leaf_dir = leaf_dir[: -len(solver_suffix)]
            break

    return leaf_dir, existing_timestamp


def apply_run_naming_to_path(
    relative_path,
    solver,
    camera_constraint_shape,
    run_timestamp=None,
    append_timestamp=False,
    preserve_existing_timestamp=True,
):
    parent_dir, leaf_dir = _split_path_leaf(relative_path)
    leaf_base, existing_timestamp = _strip_known_suffixes(leaf_dir)
    solver = str(solver)
    if solver not in SOLVER_NAMES:
        raise ValueError(f"solver must be one of {', '.join(SOLVER_NAMES)}")
    if camera_constraint_shape not in CAMERA_CONSTRAINT_SHAPES:
        raise ValueError(f"camera_constraint_shape must be one of {', '.join(CAMERA_CONSTRAINT_SHAPES)}")

    timestamp = normalize_run_timestamp(run_timestamp)
    if timestamp is None and preserve_existing_timestamp:
        timestamp = existing_timestamp
    if append_timestamp and timestamp is None:
        timestamp = make_run_timestamp()

    named_leaf = f"{leaf_base}_{solver}_{camera_constraint_shape}"
    if timestamp is not None:
        named_leaf = f"{named_leaf}_{timestamp}"

    named_path = os.path.join(parent_dir, named_leaf) if parent_dir else named_leaf
    if relative_path.endswith(os.sep):
        return named_path + os.sep
    return named_path


def find_latest_timestamped_run(base_dir, relative_path_without_timestamp):
    parent_dir, leaf_dir = _split_path_leaf(relative_path_without_timestamp)
    search_dir = os.path.join(base_dir, parent_dir) if parent_dir else base_dir
    if not os.path.isdir(search_dir):
        return None
    pattern = re.compile(rf"^{re.escape(leaf_dir)}_(\d{{14}})$")
    matches = []
    for candidate in os.listdir(search_dir):
        match = pattern.match(candidate)
        if match:
            matches.append((match.group(1), candidate))
    if not matches:
        return None
    _, latest_leaf = max(matches)
    return os.path.join(parent_dir, latest_leaf) if parent_dir else latest_leaf
