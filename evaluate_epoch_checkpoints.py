import argparse
import csv
import json
import os
import re
import subprocess
import sys

from camera_constraints import CAMERA_CONSTRAINT_SHAPES
from config_utils import parse_args_with_json_config
from mocap_config import resolve_mocap_templates
from run_paths import apply_run_naming_to_path, find_latest_timestamped_run


def find_latest_run_with_manifest(base_dir, relative_path_without_timestamp, manifest_name):
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
        relative = os.path.join(parent_dir, candidate) if parent_dir else candidate
        manifest_path = os.path.join(base_dir, relative, "pose", manifest_name)
        if os.path.exists(manifest_path):
            matches.append((match.group(1), relative))
    if not matches:
        return None
    _, latest_relative = max(matches)
    return latest_relative


def resolve_result_dir(args):
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
        latest_path = find_latest_run_with_manifest(
            "resultModel",
            base_path,
            os.path.basename(args.checkpoint_manifest),
        )
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


def resolve_checkpoint_manifest(result_dir, checkpoint_manifest):
    if os.path.isabs(checkpoint_manifest):
        return checkpoint_manifest
    if os.path.exists(checkpoint_manifest):
        return checkpoint_manifest
    return os.path.join(result_dir, "pose", checkpoint_manifest)


def summary_mean(report, section, key):
    value = report.get(section, {}).get("summary", {}).get(key, {}).get("mean")
    return "" if value is None else value


def compact_json(value):
    if value is None:
        return ""
    return json.dumps(value, separators=(",", ":"))


def run_checkpoint_eval(args, result_dir, record, output_dir):
    pose_dir = os.path.join(result_dir, "pose")
    pose_file = record["pose_file"]
    pose_path = pose_file if os.path.isabs(pose_file) else os.path.join(pose_dir, pose_file)
    epoch = int(record["epoch"])
    stage = record.get("stage", "post_gradient")
    label = record.get("label") or f"epoch_{epoch:03d}_{stage}"
    output_json = os.path.join(output_dir, f"{label}_evaluation.json")

    command = [
        sys.executable,
        "evaluate_triangulation.py",
        "--config",
        args.config,
        "--solver",
        args.solver,
        "--path",
        args.path,
        "--camera_constraint_shape",
        args.camera_constraint_shape,
        "--pose_dir",
        pose_dir,
        "--geometry_file",
        os.path.join(result_dir, "geometry_data.npz"),
        "--initial_pose",
        os.path.join(pose_dir, "0.npy"),
        "--optimized_pose",
        pose_path,
        "--pixel_noise_std",
        str(args.pixel_noise_std),
        "--trials",
        str(args.trials),
        "--seed",
        str(args.seed),
        "--output_json",
        output_json,
    ]
    if args.run_timestamp is not None:
        command.extend(["--run_timestamp", args.run_timestamp])
    if args.eval_visibility_mode is not None:
        command.extend(["--eval_visibility_mode", args.eval_visibility_mode])
    if args.min_views is not None:
        command.extend(["--min_views", str(args.min_views)])
    if args.no_refine:
        command.append("--no_refine")
    if args.no_quantize_pixels:
        command.append("--no_quantize_pixels")

    print(f"Evaluating epoch {epoch:03d} {stage}: {pose_path}", flush=True)
    subprocess.run(command, cwd=os.getcwd(), check=True)

    with open(output_json, "r", encoding="utf-8") as handle:
        report = json.load(handle)
    row = {
        "epoch": epoch,
        "stage": stage,
        "mode": record.get("mode", ""),
        "label": label,
        "pose_file": pose_file,
        "field_loss": record.get("field_loss", ""),
        "checkpoint_voxel_kcoverage_deficit": record.get("voxel_kcoverage_deficit", ""),
        "checkpoint_exact_joint_observation_gap": record.get("exact_joint_observation_gap", ""),
        "checkpoint_best_angle_deg_mean": record.get("best_triangulation_angle_deg_mean", ""),
        "points_seen_by_every_camera": record.get("points_seen_by_every_camera", ""),
        "total_points": record.get("total_points", ""),
        "camera_positions": compact_json(record.get("camera_positions")),
        "observable_rate": summary_mean(report, "optimized", "observable_rate"),
        "reconstruction_rate": summary_mean(report, "optimized", "reconstruction_rate"),
        "best_angle_deg_mean": summary_mean(report, "optimized", "best_angle_deg_mean"),
        "3d_error_rmse": summary_mean(report, "optimized", "3d_error_rmse"),
        "3d_error_p90": summary_mean(report, "optimized", "3d_error_p90"),
        "point_reproj_error_rmse": summary_mean(report, "optimized", "point_reproj_error_rmse"),
        "evaluation_json": output_json,
    }
    return row


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Batch-evaluate NeOF/BIP epoch checkpoint pose files.")
    parser.add_argument("--config", type=str, default="configs/main_mocap.json")
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
    parser.add_argument("--checkpoint_manifest", type=str, default="epoch_checkpoints.json")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--pixel_noise_std", type=float, default=0.0)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval_visibility_mode", type=str, choices=["surface_occlusion", "fov_only"], default=None)
    parser.add_argument("--min_views", type=int, default=None)
    parser.add_argument(
        "--checkpoint_stages",
        type=str,
        nargs="*",
        choices=["reset_search_selected", "reset_search_candidate", "post_reset", "post_gradient"],
        default=None,
    )
    parser.add_argument("--no_refine", action="store_true")
    parser.add_argument("--no_quantize_pixels", action="store_true")
    args = parse_args_with_json_config(parser, allow_unknown_config_keys=True)
    args = resolve_mocap_templates(args)

    result_dir = resolve_result_dir(args)
    manifest_path = resolve_checkpoint_manifest(result_dir, args.checkpoint_manifest)
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Checkpoint manifest not found: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    records = manifest.get("records", [])
    if args.checkpoint_stages:
        stages = set(args.checkpoint_stages)
        records = [record for record in records if record.get("stage", "post_gradient") in stages]
    if not records:
        raise ValueError(f"Checkpoint manifest contains no records: {manifest_path}")

    output_dir = args.output_dir or os.path.join(result_dir, "evaluation_epochs")
    os.makedirs(output_dir, exist_ok=True)

    rows = [run_checkpoint_eval(args, result_dir, record, output_dir) for record in records]
    csv_path = os.path.join(output_dir, "epoch_evaluation_summary.csv")
    json_path = os.path.join(output_dir, "epoch_evaluation_summary.json")
    write_csv(csv_path, rows)
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "result_dir": result_dir,
                "checkpoint_manifest": manifest_path,
                "rows": rows,
            },
            handle,
            indent=2,
        )

    print(f"Epoch evaluation CSV saved to: {csv_path}")
    print(f"Epoch evaluation JSON saved to: {json_path}")


if __name__ == "__main__":
    main()
