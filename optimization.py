import copy
import csv
import json
import os
import time

from camera_constraints import (
    point_in_camera_constraint,
    sample_camera_points_in_constraint,
)
from dataset.dataset import *
from field.field_attribute import voxel_model
from field.field_attribute import uses_fov_only_target_visibility
from field.optimization import GenerateP3d
from field.visibility_field import *


def format_seconds(seconds):
    return f"{seconds:.2f}s"


class CameraLayerOpt:
    def __init__(self,args,voxelnormals,occupancy_weights,minbound,scale,posepath,writer,geometry_metadata=None):
        self.args = args
        self.log_writer = writer
        self.voxelnormals = voxelnormals
        self.geometry_metadata = geometry_metadata or {}
        self.free_space_support = bool(
            getattr(args, "occupancy_map_enable", False)
            and getattr(args, "occupancy_map_mode", None) == "free_space_box"
        )
        self.fov_only_visibility = self.free_space_support or uses_fov_only_target_visibility(args)
        if occupancy_weights is None:
            occupancy_weights = np.ones(len(voxelnormals), dtype=np.float32)
        occupancy_weights = np.asarray(occupancy_weights, dtype=np.float32)
        if occupancy_weights.shape != (len(voxelnormals),):
            raise ValueError(
                f"occupancy_weights must contain exactly {len(voxelnormals)} values, got {occupancy_weights.shape}"
            )
        self.voxel_occupancy_np = occupancy_weights
        self.voxel_occupancy = npToTensor(occupancy_weights)
        self.voxel_occupancy_sum_np = float(np.sum(occupancy_weights))
        self.minbound = minbound
        self.scale = scale
        self.posepath = posepath
        self.model = GenerateP3d(
            args.cameranum,
            camera_constraint_shape=args.camera_constraint_shape,
            camera_constraint_min=args.camera_constraint_min,
            camera_constraint_max=args.camera_constraint_max,
            camera_constraint_data=args.camera_constraint_data,
            target_support_mode="free_space" if self.fov_only_visibility else "surface",
        ).to(device)
        self.fieldmodel = AddAttention(
            6,
            args.field_hidden_dim,
            args.field_num_heads,
            context_voxel_num=args.field_context_voxel_num,
            normal_voxel_num=args.field_normal_voxel_num,
        ).to(device)

        weights = np.array([args.wvis, args.wcc, args.wco, args.wres], dtype=np.float32)
        if self.fov_only_visibility:
            weights[2] = 0.0
        weights_sum = float(np.sum(weights))
        if weights_sum <= 1e-8:
            weights = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        else:
            weights = weights / weights_sum
        self.loss_weights_np = weights
        self.attribute_upperbound_np = np.array([args.kcoverage, np.pi / 2, 1.0, 1.0], dtype=np.float32)

    def compact_log_mode(self):
        return getattr(self.args, "neof_log_mode", "detailed") == "compact"

    def log_line(self, message="", console=True):
        if console:
            print(message)
        key_log_path = getattr(self.args, "optimization_key_log_path", None)
        if key_log_path:
            with open(key_log_path, "a", encoding="utf-8") as handle:
                handle.write(str(message) + "\n")

    def should_print_pose_step(self, iters, new_epoch_best, new_global_best):
        if not self.compact_log_mode():
            return True
        interval = int(getattr(self.args, "neof_pose_log_interval", 1))
        if new_epoch_best or new_global_best:
            return True
        if interval > 0 and (iters + 1) % interval == 0:
            return True
        return iters + 1 == self.args.iterations

    def score_is_better(self, voxel_gap, joint_score, best_voxel_gap, best_joint_score):
        if getattr(self.args, "require_all_cameras_coverage", False):
            if voxel_gap < best_voxel_gap - 1e-9:
                return True
            if abs(voxel_gap - best_voxel_gap) <= 1e-9 and joint_score < best_joint_score:
                return True
            return False
        return joint_score < best_joint_score

    def epoch_checkpoint_epochs(self):
        requested = set()
        interval = int(getattr(self.args, "epoch_checkpoint_interval", 0))
        if interval > 0:
            requested.update(range(interval, int(self.args.epoches) + 1, interval))
        explicit_epochs = getattr(self.args, "epoch_checkpoint_epochs", None)
        if explicit_epochs:
            requested.update(int(epoch) for epoch in explicit_epochs)
        return sorted(epoch for epoch in requested if 1 <= epoch <= int(self.args.epoches))

    def save_epoch_checkpoint(
        self,
        epoch_number,
        position,
        rotation,
        checkpoint_mode,
        stage="post_gradient",
        label=None,
        pose_filename=None,
        field_loss=None,
        loss_components=None,
        global_step=None,
    ):
        position_np = position.detach().cpu().numpy() if hasattr(position, "detach") else np.asarray(position)
        rotation_np = rotation.detach().cpu().numpy() if hasattr(rotation, "detach") else np.asarray(rotation)
        world_position_np = position_np * self.scale[0] + self.scale[1]
        stage = str(stage)
        checkpoint_label = label or f"epoch_{int(epoch_number):03d}_{stage}"
        if pose_filename is None:
            if stage == "post_gradient":
                pose_filename = f"epoch_{int(epoch_number):03d}.npy"
            else:
                pose_filename = f"{checkpoint_label}.npy"
        pose_path = os.path.join(self.posepath, pose_filename)
        saveTrainingResult(pose_path, npToTensor(position_np), npToTensor(rotation_np), self.scale)

        metrics = self.evaluate_pose_quality(position_np, rotation_np, weighted=True)

        record = {
            "epoch": int(epoch_number),
            "mode": str(checkpoint_mode),
            "stage": stage,
            "label": checkpoint_label,
            "pose_file": pose_filename,
            "pose_path": pose_path,
            "field_loss": None if field_loss is None or not np.isfinite(field_loss) else float(field_loss),
            "loss_components": None
            if loss_components is None
            else {
                "visibility": float(loss_components[0]),
                "camera_camera_angle": float(loss_components[1]),
                "camera_object_angle": float(loss_components[2]),
                "sampling_quality": float(loss_components[3]),
            },
            "global_step": None if global_step is None else int(global_step),
            "voxel_kcoverage_deficit": metrics["voxel_kcoverage_deficit"],
            "exact_joint_observation_gap": metrics["exact_joint_observation_gap"],
            "per_camera_visible_points": metrics["per_camera_visible_points"],
            "points_seen_by_at_least_kcoverage": metrics["points_seen_by_at_least_kcoverage"],
            "points_seen_by_every_camera": metrics["points_seen_by_every_camera"],
            "total_points": metrics["total_points"],
            "best_triangulation_angle_deg_mean": metrics["best_triangulation_angle_deg_mean"],
            "camera_positions": world_position_np.tolist(),
            "camera_rotation_matrices": rotation_np.tolist(),
        }
        self.epoch_checkpoint_records.append(record)
        self.log_line(
            f"  Saved epoch checkpoint {epoch_number:03d} {stage} | "
            f"pose={pose_filename} | voxel_gap={metrics['voxel_kcoverage_deficit']:.4f} | "
            f"joint_gap={metrics['exact_joint_observation_gap']:.4f} | "
            f"angle={metrics['best_triangulation_angle_deg_mean']:.2f}deg | "
            f"allcams={metrics['points_seen_by_every_camera']}/{len(self.voxelnormals)}"
        )
        return record

    def write_epoch_checkpoint_summary_csv(self, path):
        if not getattr(self, "epoch_checkpoint_records", None):
            return
        fields = [
            "epoch",
            "stage",
            "mode",
            "pose_file",
            "field_loss",
            "global_step",
            "voxel_kcoverage_deficit",
            "exact_joint_observation_gap",
            "best_triangulation_angle_deg_mean",
            "points_seen_by_at_least_kcoverage",
            "points_seen_by_every_camera",
            "total_points",
            "camera_positions",
        ]
        with open(path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for record in self.epoch_checkpoint_records:
                row = {field: record.get(field, "") for field in fields}
                row["camera_positions"] = json.dumps(record.get("camera_positions", []), separators=(",", ":"))
                writer.writerow(row)

    def write_epoch_checkpoint_manifest(self):
        if not getattr(self, "epoch_checkpoint_records", None):
            return
        payload = {
            "checkpoint_mode": getattr(self.args, "epoch_checkpoint_mode", "epoch_best"),
            "checkpoint_interval": int(getattr(self.args, "epoch_checkpoint_interval", 0)),
            "checkpoint_epochs": self.epoch_checkpoint_epochs(),
            "non_gradient_reset_enable": bool(getattr(self.args, "non_gradient_reset_enable", True)),
            "non_gradient_reset_interval": int(getattr(self.args, "non_gradient_reset_interval", 5)),
            "reset_search_enable": bool(getattr(self.args, "reset_search_enable", False)),
            "reset_search_trials": int(getattr(self.args, "reset_search_trials", 0)),
            "reset_search_top_k": int(getattr(self.args, "reset_search_top_k", 1)),
            "reset_search_score": getattr(self.args, "reset_search_score", "angle"),
            "epoch_checkpoint_save_reset": bool(getattr(self.args, "epoch_checkpoint_save_reset", True)),
            "effective_kcoverage": int(self.args.kcoverage),
            "cameranum": int(self.args.cameranum),
            "records": self.epoch_checkpoint_records,
        }
        manifest_path = os.path.join(self.posepath, "epoch_checkpoints.json")
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        self.log_line(f"Epoch checkpoint manifest saved: {manifest_path}")
        summary_path = os.path.join(self.posepath, "epoch_checkpoints_summary.csv")
        self.write_epoch_checkpoint_summary_csv(summary_path)
        self.log_line(f"Epoch checkpoint summary CSV saved: {summary_path}")
        self.log_line("Epoch checkpoint summary")
        for record in self.epoch_checkpoint_records:
            field_loss = record.get("field_loss")
            field_loss_text = "none" if field_loss is None else f"{field_loss:.4f}"
            self.log_line(
                f"  epoch={record['epoch']:03d} | stage={record.get('stage', ''):<13} | "
                f"loss={field_loss_text} | voxel_gap={record['voxel_kcoverage_deficit']:.4f} | "
                f"joint_gap={record['exact_joint_observation_gap']:.4f} | "
                f"angle={record.get('best_triangulation_angle_deg_mean', np.nan):.2f}deg | "
                f"allcams={record['points_seen_by_every_camera']}/{record['total_points']}"
            )

    def loss_weights(self):
        return npToTensor(self.loss_weights_np)

    def attribute_upperbound(self):
        return npToTensor(self.attribute_upperbound_np)

    def build_field_queries(self, voxelmodel):
        query_points = voxelmodel[:, :3]
        jitter_std = max(float(self.args.field_query_jitter_std), 0.0)
        if jitter_std > 0:
            query_points = query_points + torch.randn_like(query_points) * jitter_std
        return query_points

    def predict_field(self, voxelmodel, query_points=None, exclude_closest_context=False):
        if query_points is None:
            query_points = voxelmodel[:, :3]
        return self.fieldmodel(
            query_points.unsqueeze(1),
            voxelmodel[:, :3],
            voxelmodel[:, 3:6],
            voxelmodel[:, 6:],
            exclude_closest_context=exclude_closest_context,
        )

    def field_supervision_loss(self, prediction, target):
        normalized_error = torch.abs(prediction - target) / self.attribute_upperbound().unsqueeze(0)
        weighted_error = normalized_error * self.loss_weights().unsqueeze(0)
        denominator = torch.clamp(self.voxel_occupancy.sum() * weighted_error.shape[1], min=1e-8)
        return torch.sum(weighted_error * self.voxel_occupancy.unsqueeze(1)) / denominator

    def global_need_score(self, attributes, weighted=True):
        normalized = attributes / self.attribute_upperbound_np[None, :]
        per_voxel_need = np.sum(normalized * self.loss_weights_np[None, :], axis=1)
        if weighted:
            return float(np.sum(per_voxel_need * self.voxel_occupancy_np) / self.voxel_occupancy_sum_np)
        return float(np.mean(per_voxel_need))

    def candidate_priority(self, predicted_attributes):
        normalized = predicted_attributes / self.attribute_upperbound_np[None, :]
        priority = np.sum(normalized * self.loss_weights_np[None, :], axis=1)
        return priority * self.voxel_occupancy_np

    def coverage_gap_from_visibility(self, voxel_visibility, weighted=True):
        coverage = np.sum(voxel_visibility, axis=-1)
        squared_gap = np.square(np.maximum(self.args.kcoverage - coverage, np.zeros(len(coverage))))
        if weighted:
            gap = np.sum(self.voxel_occupancy_np * squared_gap) / (
                self.args.kcoverage * self.args.kcoverage * self.voxel_occupancy_sum_np
            )
        else:
            gap = np.sum(squared_gap) / (self.args.kcoverage * self.args.kcoverage * len(coverage))
        return float(gap), coverage

    def best_triangulation_angle_for_point(self, position, visible_camera_indices, point):
        if len(visible_camera_indices) < 2:
            return np.nan
        rays = []
        for camera_idx in visible_camera_indices:
            ray = position[camera_idx] - point
            norm = np.linalg.norm(ray)
            if norm > 1e-8:
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

    def visibility_triangulation_angle_mean(self, position, visibility):
        visibility = np.asarray(visibility)
        min_views = max(2, min(int(self.args.kcoverage), int(self.args.cameranum)))
        angles = []
        weights = []
        for point_idx, point_visibility in enumerate(visibility):
            visible_camera_indices = np.flatnonzero(point_visibility > 0)
            if len(visible_camera_indices) < min_views:
                continue
            angle = self.best_triangulation_angle_for_point(
                position,
                visible_camera_indices,
                self.voxelnormals[point_idx, :3],
            )
            if np.isfinite(angle):
                angles.append(angle)
                weights.append(float(self.voxel_occupancy_np[point_idx]))
        if len(angles) == 0:
            return np.nan
        return float(np.average(np.asarray(angles, dtype=float), weights=np.asarray(weights, dtype=float)))

    def evaluate_pose_quality(self, position, rotation, weighted=True):
        voxelmodel, visibility = voxel_model(
            self.args,
            self.voxelnormals,
            rotation,
            position,
        )
        voxel_gap, coverage = self.coverage_gap_from_visibility(visibility, weighted=weighted)
        joint_score = self.global_need_score(voxelmodel[:, 6:].cpu().numpy(), weighted=weighted)
        per_camera_visible = np.sum(visibility > 0, axis=0).astype(int).tolist()
        all_camera_visible = int(np.sum(np.sum(visibility > 0, axis=1) == self.args.cameranum))
        kcoverage_visible = int(np.sum(coverage >= self.args.kcoverage))
        return {
            "voxelmodel": voxelmodel,
            "visibility": visibility,
            "voxel_kcoverage_deficit": float(voxel_gap),
            "exact_joint_observation_gap": float(joint_score),
            "per_camera_visible_points": per_camera_visible,
            "points_seen_by_at_least_kcoverage": kcoverage_visible,
            "points_seen_by_every_camera": all_camera_visible,
            "total_points": int(len(self.voxelnormals)),
            "best_triangulation_angle_deg_mean": self.visibility_triangulation_angle_mean(position, visibility),
        }

    def reset_search_score_key(self, metrics):
        voxel_gap = float(metrics["voxel_kcoverage_deficit"])
        joint_score = float(metrics["exact_joint_observation_gap"])
        angle = float(metrics.get("best_triangulation_angle_deg_mean", np.nan))
        if not np.isfinite(angle):
            angle = -1.0
        score_mode = getattr(self.args, "reset_search_score", "angle")
        if score_mode == "joint_gap":
            return (voxel_gap, joint_score, -angle)
        if score_mode == "combined":
            angle_weight = float(getattr(self.args, "reset_search_angle_weight", 0.25))
            combined = joint_score - angle_weight * (angle / 90.0)
            return (voxel_gap, combined, joint_score, -angle)
        return (voxel_gap, -angle, joint_score)

    def visibility_summary_lines(self, visibility):
        if visibility is None:
            return []
        visibility = np.asarray(visibility)
        if visibility.ndim != 2 or visibility.shape[0] == 0:
            return []
        point_count, camera_count = visibility.shape
        per_camera = np.sum(visibility > 0, axis=0)
        coverage = np.sum(visibility > 0, axis=1)
        kcoverage = int(max(1, min(int(self.args.kcoverage), camera_count)))
        lines = [
            "  Per-camera visible points: "
            + ", ".join(
                f"cam{idx:02d}={int(count)}/{point_count}"
                for idx, count in enumerate(per_camera)
            ),
            f"  Points seen by >= {kcoverage} cameras: {int(np.sum(coverage >= kcoverage))}/{point_count}",
        ]
        if getattr(self.args, "require_all_cameras_coverage", False):
            lines.append(
                f"  Points seen by every camera: {int(np.sum(coverage == camera_count))}/{point_count}"
            )
        return lines

    def judgeDistance(self,position,points):
        distance=np.linalg.norm(position-points,axis=1)
        judge=(distance<=0.02).nonzero()[0]
        if len(judge)==0:
            return True
        else:
            return False

    def judgeBoundingBox(self,p,points):
        min=np.min(points,axis=0)
        max=np.max(points,axis=0)
        if len((p>min).nonzero()[0]) ==3 and len((p<max).nonzero()[0]) ==3 :
            return True
        else:
            return False

    def within_camera_constraint(self, position):
        if not self.args.camera_constraint_enable:
            return True
        return point_in_camera_constraint(
            position,
            self.args.camera_constraint_shape,
            self.args.camera_constraint_min,
            self.args.camera_constraint_max,
            constraint_data=self.args.camera_constraint_data,
        )

    def sample_random_candidate_poses(self, target_point):
        centers = sample_camera_points_in_constraint(
            self.args.reset_random_samples_per_voxel,
            self.args.camera_constraint_shape,
            self.args.camera_constraint_min,
            self.args.camera_constraint_max,
            constraint_data=self.args.camera_constraint_data,
        )
        candidate_poses = []
        for center in centers:
            candidate_poses.append((center, getLookAtRotation(center, target_point)))
        return candidate_poses

    def build_pose_optimizer(self):
        if self.args.optimizer == 'SGD':
            optimizer = torch.optim.SGD(
                [
                    {'params':[self.model.position_parameter],'lr':self.args.lr1},
                    {'params':[self.model.rotate6d],'lr':self.args.lr2},
                ]
            )
        elif self.args.optimizer == 'Adam':
            optimizer = torch.optim.Adam(
                [
                    {'params':[self.model.position_parameter],'lr':self.args.lr1},
                    {'params':[self.model.rotate6d],'lr':self.args.lr2},
                ]
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.args.optimizer}")
        return optimizer

    def set_pose_optimizer_lr(self, optimizer, epoch):
        lr_factor = pow(self.args.pose_lr_decay, epoch)
        optimizer.param_groups[0]['lr'] = self.args.lr1 * lr_factor
        optimizer.param_groups[1]['lr'] = self.args.lr2 * lr_factor

    def fit_field(self, voxelmodel, epoch):
        fieldoptimizer = torch.optim.Adam(
            self.fieldmodel.parameters(),
            lr=self.args.field_lr * pow(self.args.field_lr_decay, epoch),
        )
        field_fit_start = time.time()
        best_loss = np.inf
        best_step = 0
        stale_steps = 0
        last_step = 0

        self.fieldmodel.train()
        for iter in range(self.args.field_fit_steps):
            query_points = self.build_field_queries(voxelmodel)
            predicted_attributes = self.predict_field(
                voxelmodel,
                query_points=query_points,
                exclude_closest_context=self.args.field_exclude_closest_context,
            )
            loss = self.field_supervision_loss(predicted_attributes, voxelmodel[:, 6:])
            fieldoptimizer.zero_grad()
            loss.backward()
            fieldoptimizer.step()

            current_loss = float(loss.item())
            last_step = iter + 1
            if best_loss - current_loss > self.args.field_early_stop_min_delta:
                best_loss = current_loss
                best_step = last_step
                stale_steps = 0
            else:
                stale_steps += 1

            if last_step % self.args.field_fit_log_interval == 0 or last_step == self.args.field_fit_steps:
                if not self.compact_log_mode():
                    self.log_line(
                        f"    Field fit step {last_step:04d}/{self.args.field_fit_steps} | "
                        f"supervision loss: {current_loss:.6e}",
                        console=True,
                    )

            if (
                self.args.field_early_stop
                and last_step >= self.args.field_early_stop_min_steps
                and stale_steps >= self.args.field_early_stop_patience
            ):
                self.log_line(
                    f"    Field fit early stop at step {last_step:04d}/{self.args.field_fit_steps} | "
                    f"best loss: {best_loss:.6e} at step {best_step:04d}"
                )
                break

        field_fit_time = time.time() - field_fit_start
        self.log_line(
            f"  Field fit time: {format_seconds(field_fit_time)} | "
            f"best loss: {best_loss:.6e} at step {best_step:04d}"
        )
        self.fieldmodel.eval()

    def reset_camera(self, position, rotation, min_camera, sample, scene_mode, priority_source="field"):
        voxelmodel, _ = voxel_model(
            self.args,
            self.voxelnormals,
            rotation,
            position,
        )
        if priority_source == "exact":
            predicted = voxelmodel[:, 6:].cpu().numpy()
        else:
            with torch.no_grad():
                predicted = self.predict_field(voxelmodel).cpu().numpy()
        origin_score = self.global_need_score(voxelmodel[:, 6:].cpu().numpy())

        rr = rotation[min_camera]
        rp = position[min_camera]
        if self.within_camera_constraint(rp) == False:
            origin_score = 1e6
        if scene_mode and self.judgeDistance(rp,self.voxelnormals[:,:3]) == False:
            origin_score = 1e6

        candidate_priority = self.candidate_priority(predicted)
        candidate_indices = np.argsort(candidate_priority)[::-1][:sample]

        for candidate_idx in candidate_indices:
            surface_point = self.voxelnormals[candidate_idx, :3]
            candidate_poses = self.sample_random_candidate_poses(surface_point)
            if len(candidate_poses) == 0:
                continue

            for p, r in candidate_poses:
                if self.within_camera_constraint(p) == False:
                    continue
                if scene_mode:
                    if self.judgeBoundingBox(p,self.voxelnormals[:,:3]) == False or self.judgeDistance(p,self.voxelnormals[:,:3]) == False:
                        continue
                candidate_position = position.copy()
                candidate_rotation = rotation.copy()
                candidate_position[min_camera] = p
                candidate_rotation[min_camera] = r

                candidate_voxelmodel, _ = voxel_model(
                    self.args,
                    self.voxelnormals,
                    candidate_rotation,
                    candidate_position,
                )
                current_score = self.global_need_score(candidate_voxelmodel[:, 6:].cpu().numpy())
                if current_score < origin_score:
                    rr = r
                    rp = p
                    origin_score = current_score

        rotation[min_camera] = rr
        position[min_camera] = rp
        torch.cuda.empty_cache()
        return npToTensor(position), npToTensor(rotation)

    def resetNewforScene(self,position,rotation,min_camera,priority_source="field"):
        return self.reset_camera(position, rotation, min_camera, sample=100, scene_mode=True, priority_source=priority_source)

    def resetNewforModel(self,position,rotation,min_camera,priority_source="field"):
        return self.reset_camera(position, rotation, min_camera, sample=20, scene_mode=False, priority_source=priority_source)

    def run_single_reset_search_trial(self, start_position, start_rotation):
        position = np.asarray(start_position, dtype=float).copy()
        rotation = np.asarray(start_rotation, dtype=float).copy()
        for min_camera in range(self.args.cameranum):
            if self.args.isscene:
                position, rotation = self.resetNewforScene(
                    position,
                    rotation,
                    min_camera,
                    priority_source="exact",
                )
            else:
                position, rotation = self.resetNewforModel(
                    position,
                    rotation,
                    min_camera,
                    priority_source="exact",
                )
            position = position.detach().cpu().numpy()
            rotation = rotation.detach().cpu().numpy()
        return position, rotation

    def reset_search_candidate_record(self, trial, position, rotation, source):
        metrics = self.evaluate_pose_quality(position, rotation, weighted=True)
        score_key = self.reset_search_score_key(metrics)
        return {
            "trial": int(trial),
            "source": str(source),
            "score_key": list(score_key),
            "position": np.asarray(position, dtype=float),
            "rotation": np.asarray(rotation, dtype=float),
            "voxel_kcoverage_deficit": metrics["voxel_kcoverage_deficit"],
            "exact_joint_observation_gap": metrics["exact_joint_observation_gap"],
            "best_triangulation_angle_deg_mean": metrics["best_triangulation_angle_deg_mean"],
            "points_seen_by_at_least_kcoverage": metrics["points_seen_by_at_least_kcoverage"],
            "points_seen_by_every_camera": metrics["points_seen_by_every_camera"],
            "total_points": metrics["total_points"],
            "per_camera_visible_points": metrics["per_camera_visible_points"],
        }

    def write_reset_search_candidates_csv(self, candidates):
        if not candidates:
            return
        path = os.path.join(self.posepath, "reset_search_candidates.csv")
        fields = [
            "rank",
            "trial",
            "source",
            "pose_file",
            "score_key",
            "voxel_kcoverage_deficit",
            "exact_joint_observation_gap",
            "best_triangulation_angle_deg_mean",
            "points_seen_by_at_least_kcoverage",
            "points_seen_by_every_camera",
            "total_points",
            "per_camera_visible_points",
            "camera_positions",
        ]
        with open(path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for rank, candidate in enumerate(candidates, start=1):
                position = candidate["position"]
                world_position = position * self.scale[0] + self.scale[1]
                row = {
                    "rank": rank,
                    "trial": candidate["trial"],
                    "source": candidate["source"],
                    "pose_file": candidate.get("pose_file", ""),
                    "score_key": json.dumps(candidate["score_key"], separators=(",", ":")),
                    "voxel_kcoverage_deficit": candidate["voxel_kcoverage_deficit"],
                    "exact_joint_observation_gap": candidate["exact_joint_observation_gap"],
                    "best_triangulation_angle_deg_mean": candidate["best_triangulation_angle_deg_mean"],
                    "points_seen_by_at_least_kcoverage": candidate["points_seen_by_at_least_kcoverage"],
                    "points_seen_by_every_camera": candidate["points_seen_by_every_camera"],
                    "total_points": candidate["total_points"],
                    "per_camera_visible_points": json.dumps(candidate["per_camera_visible_points"], separators=(",", ":")),
                    "camera_positions": json.dumps(world_position.tolist(), separators=(",", ":")),
                }
                writer.writerow(row)
        self.log_line(f"Reset-search candidate CSV saved: {path}")

    def run_reset_search(self, start_position, start_rotation):
        if (
            not getattr(self.args, "reset_search_enable", False)
            or int(getattr(self.args, "reset_search_trials", 0)) <= 0
        ):
            return start_position, start_rotation, None

        start_position_np = start_position.detach().cpu().numpy() if hasattr(start_position, "detach") else np.asarray(start_position)
        start_rotation_np = start_rotation.detach().cpu().numpy() if hasattr(start_rotation, "detach") else np.asarray(start_rotation)
        trial_count = int(getattr(self.args, "reset_search_trials", 0))
        top_k = int(getattr(self.args, "reset_search_top_k", 1))
        self.log_line(
            f"Reset-search stage | trials={trial_count} | top_k={top_k} | "
            f"score={getattr(self.args, 'reset_search_score', 'angle')}"
        )

        candidates = [
            self.reset_search_candidate_record(0, start_position_np, start_rotation_np, "initial")
        ]
        for trial in range(1, trial_count + 1):
            trial_start = time.time()
            position, rotation = self.run_single_reset_search_trial(start_position_np, start_rotation_np)
            candidate = self.reset_search_candidate_record(trial, position, rotation, "non_gradient_reset")
            candidates.append(candidate)
            self.log_line(
                f"  reset trial {trial:03d}/{trial_count:03d} | "
                f"voxel_gap={candidate['voxel_kcoverage_deficit']:.4f} | "
                f"joint_gap={candidate['exact_joint_observation_gap']:.4f} | "
                f"angle={candidate['best_triangulation_angle_deg_mean']:.2f}deg | "
                f"allcams={candidate['points_seen_by_every_camera']}/{candidate['total_points']} | "
                f"time={format_seconds(time.time() - trial_start)}"
            )

        candidates = sorted(candidates, key=lambda candidate: tuple(candidate["score_key"]))
        if getattr(self.args, "reset_search_save_all", True):
            for rank, candidate in enumerate(candidates, start=1):
                pose_filename = f"reset_search_rank_{rank:03d}_trial_{candidate['trial']:03d}.npy"
                candidate["pose_file"] = pose_filename
                saveTrainingResult(
                    os.path.join(self.posepath, pose_filename),
                    npToTensor(candidate["position"]),
                    npToTensor(candidate["rotation"]),
                    self.scale,
                )
        self.write_reset_search_candidates_csv(candidates)

        selected_count = min(top_k, len(candidates))
        for rank, candidate in enumerate(candidates[:selected_count], start=1):
            stage = "reset_search_selected" if rank == 1 else "reset_search_candidate"
            label = f"reset_search_rank_{rank:03d}_trial_{candidate['trial']:03d}"
            pose_filename = candidate.get("pose_file") or f"{label}.npy"
            self.save_epoch_checkpoint(
                0,
                npToTensor(candidate["position"]),
                npToTensor(candidate["rotation"]),
                stage,
                stage=stage,
                label=label,
                pose_filename=pose_filename,
            )

        selected = candidates[0]
        self.log_line(
            f"Reset-search selected trial {selected['trial']:03d} | "
            f"voxel_gap={selected['voxel_kcoverage_deficit']:.4f} | "
            f"joint_gap={selected['exact_joint_observation_gap']:.4f} | "
            f"angle={selected['best_triangulation_angle_deg_mean']:.2f}deg | "
            f"allcams={selected['points_seen_by_every_camera']}/{selected['total_points']}"
        )
        selected_position = npToTensor(selected["position"])
        selected_rotation = npToTensor(selected["rotation"])
        return selected_position, selected_rotation, selected

    def metric(self,position,rotation,weighted=True):
        _,voxel_visibility=voxel_model(self.args,self.voxelnormals,rotation,position)
        rate_v,_ = self.coverage_gap_from_visibility(voxel_visibility, weighted=weighted)
        return rate_v

    def print_coverage_summary(
        self,
        title,
        voxel_gap,
        joint_gap=None,
        position=None,
        unweighted_voxel_gap=None,
        unweighted_joint_gap=None,
        visibility=None,
    ):
        del position
        self.log_line(title)
        if self.fov_only_visibility:
            visibility_label = "free-space box" if self.free_space_support else "FoV-only target visibility"
            self.log_line(f"  Target support: {visibility_label} (camera-object angle term disabled)")
        if self.args.occupancy_map_enable:
            self.log_line(f"  Weighted voxel K-coverage deficit (normalized): {voxel_gap:.4f}")
            if unweighted_voxel_gap is not None:
                self.log_line(f"  Unweighted voxel K-coverage deficit (normalized): {unweighted_voxel_gap:.4f}")
        else:
            self.log_line(f"  Voxel K-coverage deficit (normalized): {voxel_gap:.4f}")
        if joint_gap is not None:
            if self.args.occupancy_map_enable:
                self.log_line(f"  Weighted exact joint observation gap: {joint_gap:.4f}")
                if unweighted_joint_gap is not None:
                    self.log_line(f"  Unweighted exact joint observation gap: {unweighted_joint_gap:.4f}")
            else:
                self.log_line(f"  Exact joint observation gap: {joint_gap:.4f}")
        for line in self.visibility_summary_lines(visibility):
            self.log_line(line)

    def opt(self,camerapose):
        self.epoch_checkpoint_records = []
        checkpoint_epochs = set(self.epoch_checkpoint_epochs())
        checkpoint_mode = getattr(self.args, "epoch_checkpoint_mode", "epoch_best")
        bestposition = camerapose[:,:3].detach().clone()
        bestrotation = compute_rotation_matrix_from_ortho6d(camerapose[:,3:]).detach().clone()
        best_joint_score = np.inf
        best_voxel_gap = np.inf
        best_field_loss = None
        best_loss_components = None
        best_global_step = None
        bbp = bestposition.detach().clone()
        bbr = bestrotation.detach().clone()
        self.model.set_pose(camerapose.detach())
        pose_optimizer = self.build_pose_optimizer()
        self.fieldmodel.eval()

        with torch.no_grad():
            unweighted_rate_v = None
            initial_voxelmodel, initial_visibility = voxel_model(
                self.args,
                self.voxelnormals,
                bbr.cpu().numpy(),
                bbp.cpu().numpy(),
            )
            rate_v, _ = self.coverage_gap_from_visibility(initial_visibility, weighted=True)
            best_voxel_gap = rate_v
            best_joint_score = self.global_need_score(initial_voxelmodel[:, 6:].cpu().numpy(), weighted=True)
            if self.args.occupancy_map_enable:
                unweighted_rate_v, _ = self.coverage_gap_from_visibility(initial_visibility, weighted=False)
            self.print_coverage_summary(
                "Initial placement quality",
                rate_v,
                position=bbp.cpu().numpy(),
                unweighted_voxel_gap=unweighted_rate_v,
                visibility=initial_visibility,
            )

        search_position, search_rotation, search_selected = self.run_reset_search(bbp, bbr)
        if search_selected is not None:
            bestposition = search_position.detach().clone()
            bestrotation = search_rotation.detach().clone()
            bbp = bestposition.detach().clone()
            bbr = bestrotation.detach().clone()
            best_voxel_gap = float(search_selected["voxel_kcoverage_deficit"])
            best_joint_score = float(search_selected["exact_joint_observation_gap"])
            best_field_loss = None
            best_loss_components = None
            best_global_step = None
            self.print_coverage_summary(
                "Selected reset-search placement quality",
                best_voxel_gap,
                best_joint_score,
                position=bbp.cpu().numpy(),
                visibility=self.evaluate_pose_quality(
                    bbp.cpu().numpy(),
                    bbr.cpu().numpy(),
                    weighted=True,
                )["visibility"],
            )

        for epoch in range(self.args.epoches):
            epoch_best_joint_score = np.inf
            epoch_best_voxel_gap = np.inf
            epoch_best_field_loss = None
            epoch_best_loss_components = None
            epoch_best_global_step = None
            position = copy.deepcopy(bestposition)
            rotation = copy.deepcopy(bestrotation)
            last_position = position.detach().clone()
            last_rotation = rotation.detach().clone()
            last_field_loss = None
            last_loss_components = None
            last_global_step = None
            outer_epoch = epoch + 1
            self.log_line(f"Outer epoch {outer_epoch:02d}/{self.args.epoches:02d}")

            if (
                getattr(self.args, "non_gradient_reset_enable", True)
                and outer_epoch % int(getattr(self.args, "non_gradient_reset_interval", 5)) == 0
            ):
                self.log_line("  Stage 1/2 | Non-gradient camera reset")
                for min_camera in range(self.args.cameranum):
                    if self.args.isscene:
                        position,rotation = self.resetNewforScene(position.cpu().numpy(),rotation.cpu().numpy(),min_camera)
                    else:
                        position,rotation = self.resetNewforModel(position.cpu().numpy(),rotation.cpu().numpy(),min_camera)
                    saveTrainingResult(self.posepath+str(epoch)+"_"+str(min_camera+1)+".npy",position,rotation,self.scale)
                pose_optimizer.state.clear()

                with torch.no_grad():
                    reset_position_np = position.detach().cpu().numpy()
                    reset_rotation_np = rotation.detach().cpu().numpy()
                    reset_voxelmodel, reset_visibility = voxel_model(
                        self.args,
                        self.voxelnormals,
                        reset_rotation_np,
                        reset_position_np,
                    )
                    reset_rate_v, _ = self.coverage_gap_from_visibility(reset_visibility, weighted=True)
                    reset_joint_score = self.global_need_score(
                        reset_voxelmodel[:, 6:].cpu().numpy(),
                        weighted=True,
                    )
                    reset_unweighted_rate_v = None
                    reset_unweighted_joint_score = None
                    if self.args.occupancy_map_enable:
                        reset_unweighted_rate_v, _ = self.coverage_gap_from_visibility(
                            reset_visibility,
                            weighted=False,
                        )
                        reset_unweighted_joint_score = self.global_need_score(
                            reset_voxelmodel[:, 6:].cpu().numpy(),
                            weighted=False,
                        )

                reset_status = []
                if self.score_is_better(
                    reset_rate_v,
                    reset_joint_score,
                    epoch_best_voxel_gap,
                    epoch_best_joint_score,
                ):
                    epoch_best_voxel_gap = reset_rate_v
                    epoch_best_joint_score = reset_joint_score
                    epoch_best_field_loss = None
                    epoch_best_loss_components = None
                    epoch_best_global_step = None
                    bestposition = position.detach().clone()
                    bestrotation = rotation.detach().clone()
                    reset_status.append("epoch best")
                    if self.score_is_better(
                        reset_rate_v,
                        reset_joint_score,
                        best_voxel_gap,
                        best_joint_score,
                    ):
                        best_voxel_gap = reset_rate_v
                        best_joint_score = reset_joint_score
                        best_field_loss = None
                        best_loss_components = None
                        best_global_step = None
                        bbp = position.detach().clone()
                        bbr = rotation.detach().clone()
                        reset_status.append("global best")

                self.print_coverage_summary(
                    "Post-reset placement quality",
                    reset_rate_v,
                    reset_joint_score,
                    position=reset_position_np,
                    unweighted_voxel_gap=reset_unweighted_rate_v,
                    unweighted_joint_gap=reset_unweighted_joint_score,
                    visibility=reset_visibility,
                )
                if reset_status:
                    self.log_line(f"  Post-reset accepted as {', '.join(reset_status)}")
                if (
                    getattr(self.args, "epoch_checkpoint_save_reset", True)
                    and outer_epoch in checkpoint_epochs
                ):
                    self.save_epoch_checkpoint(
                        outer_epoch,
                        position,
                        rotation,
                        "post_reset",
                        stage="post_reset",
                    )

            camerapose = torch.cat((position,torch.cat((rotation[:,0,:],rotation[:,2,:]),1)),1)
            self.model.set_pose(camerapose.detach())
            self.set_pose_optimizer_lr(pose_optimizer, epoch)

            voxelmodel_start = time.time()
            voxelmodel,_ = voxel_model(
                self.args,
                self.voxelnormals,
                rotation.cpu().numpy(),
                position.cpu().numpy(),
            )
            voxelmodel_time = time.time() - voxelmodel_start
            self.log_line(
                f"  Stage 2/2 | Fit neural observation field on {len(voxelmodel)} voxels "
                f"(attribute build {format_seconds(voxelmodel_time)})"
            )
            self.fit_field(voxelmodel, epoch)

            for iters in range(self.args.iterations):
                start = time.time()
                pose_optimizer.zero_grad()
                attribute,position,rotation = self.model(
                    voxelmodel,
                    self.fieldmodel,
                    voxel_weights=self.voxel_occupancy,
                    depth_temperature=self.args.visibility_depth_temperature,
                    fov_temperature=self.args.visibility_fov_temperature,
                    normal_temperature=self.args.visibility_normal_temperature,
                    quality_temperature=self.args.sampling_quality_temperature,
                    voxel_size=self.args.voxelsize,
                )
                raw_loss_components = torch.mean(attribute / self.voxel_occupancy_sum_np,dim=0)
                loss_components = raw_loss_components / self.attribute_upperbound()
                L = torch.sum(self.loss_weights() * loss_components)
                L.backward()
                pose_optimizer.step()
                torch.cuda.empty_cache()
                global_step = epoch * self.args.iterations + iters
                loss_value = float(L.detach().cpu())
                loss_component_values = loss_components.detach().cpu().numpy()

                with torch.no_grad():
                    position, rotation = self.model.get_pose()
                    last_position = position.detach().clone()
                    last_rotation = rotation.detach().clone()
                    last_field_loss = loss_value
                    last_loss_components = loss_component_values.tolist()
                    last_global_step = global_step
                    current_voxelmodel, voxel_visibility = voxel_model(
                        self.args,
                        self.voxelnormals,
                        rotation.detach().cpu().numpy(),
                        position.detach().cpu().numpy(),
                    )
                    rate_v, v_coverage = self.coverage_gap_from_visibility(voxel_visibility, weighted=True)
                    unweighted_rate_v = None
                    if self.args.occupancy_map_enable:
                        unweighted_rate_v, _ = self.coverage_gap_from_visibility(voxel_visibility, weighted=False)
                    num_v=np.sum(np.sign(v_coverage))
                    all_camera_visible = int(np.sum(np.sum(voxel_visibility > 0, axis=1) == self.args.cameranum))
                    joint_score = self.global_need_score(current_voxelmodel[:, 6:].cpu().numpy(), weighted=True)
                    unweighted_joint_score = None
                    if self.args.occupancy_map_enable:
                        unweighted_joint_score = self.global_need_score(
                            current_voxelmodel[:, 6:].cpu().numpy(),
                            weighted=False,
                        )

                new_epoch_best = False
                new_global_best = False
                if self.score_is_better(rate_v, joint_score, epoch_best_voxel_gap, epoch_best_joint_score):
                    epoch_best_voxel_gap = rate_v
                    epoch_best_joint_score = joint_score
                    epoch_best_field_loss = loss_value
                    epoch_best_loss_components = loss_component_values.tolist()
                    epoch_best_global_step = global_step
                    new_epoch_best = True
                    bestposition = position.detach().clone()
                    bestrotation = rotation.detach().clone()
                    if self.score_is_better(rate_v, joint_score, best_voxel_gap, best_joint_score):
                        best_voxel_gap = rate_v
                        best_joint_score = joint_score
                        best_field_loss = loss_value
                        best_loss_components = loss_component_values.tolist()
                        best_global_step = global_step
                        new_global_best = True
                        bbp = position.detach().clone()
                        bbr = rotation.detach().clone()

                self.log_writer.add_scalar("loss/total",L,global_step)
                self.log_writer.add_scalar("loss/raw_vis",raw_loss_components[0],global_step)
                self.log_writer.add_scalar("loss/raw_cc",raw_loss_components[1],global_step)
                self.log_writer.add_scalar("loss/raw_co",raw_loss_components[2],global_step)
                self.log_writer.add_scalar("loss/raw_res",raw_loss_components[3],global_step)
                self.log_writer.add_scalar("loss/norm_vis",loss_components[0],global_step)
                self.log_writer.add_scalar("loss/norm_cc",loss_components[1],global_step)
                self.log_writer.add_scalar("loss/norm_co",loss_components[2],global_step)
                self.log_writer.add_scalar("loss/norm_res",loss_components[3],global_step)
                self.log_writer.add_scalar("metric/joint_need_score",joint_score,global_step)
                self.log_writer.add_scalar("metric/voxel_uncoverage_rate",rate_v,global_step)
                self.log_writer.add_scalar("metric/voxel_coverage_num",num_v,global_step)
                if self.args.occupancy_map_enable:
                    self.log_writer.add_scalar("metric/unweighted_joint_need_score",unweighted_joint_score,global_step)
                    self.log_writer.add_scalar("metric/unweighted_voxel_uncoverage_rate",unweighted_rate_v,global_step)
                status = []
                if new_epoch_best:
                    status.append("epoch best")
                if new_global_best:
                    status.append("global best")
                status_text = f" [{', '.join(status)}]" if status else ""
                end = time.time()
                voxel_gap_label = "weighted voxel deficit" if self.args.occupancy_map_enable else "voxel K-coverage deficit"
                joint_gap_label = "weighted joint gap" if self.args.occupancy_map_enable else "joint observation gap"
                co_label = "camera-object angle deficit (disabled)" if self.fov_only_visibility else "camera-object angle deficit"
                quality_label = "sampling-quality deficit"
                compact_pose_line = (
                    f"  Pose step {iters+1:02d}/{self.args.iterations:02d} | "
                    f"global={global_step:03d}{status_text} | "
                    f"loss={loss_value:.4f} | "
                    f"vis={loss_component_values[0]:.4f}, cc={loss_component_values[1]:.4f}, "
                    f"co={loss_component_values[2]:.4f}, res={loss_component_values[3]:.4f} | "
                    f"{voxel_gap_label}={rate_v:.4f} | "
                    f"{joint_gap_label}={joint_score:.4f} | "
                    f"seen>0={int(num_v)}/{len(v_coverage)} | "
                    f"allcams={all_camera_visible}/{len(v_coverage)} | "
                    f"step={format_seconds(end-start)}"
                )
                if self.compact_log_mode():
                    self.log_line(
                        compact_pose_line,
                        console=self.should_print_pose_step(iters, new_epoch_best, new_global_best),
                    )
                else:
                    self.log_line(compact_pose_line, console=False)
                    print(
                        f"  Pose step {iters+1:02d}/{self.args.iterations:02d} | "
                        f"global step {global_step:03d}{status_text}\n"
                        f"    Field-based joint loss: {L:.4f}\n"
                        f"    Normalized loss components:\n"
                        f"      visibility deficit          = {loss_components[0]:.4f}\n"
                        f"      camera-camera angle deficit = {loss_components[1]:.4f}\n"
                        f"      {co_label:<28} = {loss_components[2]:.4f}\n"
                        f"      {quality_label:<28} = {loss_components[3]:.4f}\n"
                        f"    Exact voxel-model evaluation:\n"
                        f"      {voxel_gap_label:<28} = {rate_v:.4f}\n"
                        f"      {joint_gap_label:<28} = {joint_score:.4f}\n"
                        f"      voxels seen by >=1 camera   = {int(num_v)}/{len(v_coverage)}\n"
                        f"      voxels seen by every camera = {all_camera_visible}/{len(v_coverage)}\n"
                        f"      step time                   = {format_seconds(end-start)}"
                    )
                if self.args.occupancy_map_enable:
                    self.log_line(
                        f"      unweighted voxel deficit    = {unweighted_rate_v:.4f}\n"
                        f"      unweighted joint gap        = {unweighted_joint_score:.4f}",
                        console=not self.compact_log_mode()
                    )

            epoch_number = epoch + 1
            if epoch_number in checkpoint_epochs:
                if checkpoint_mode == "global_best":
                    checkpoint_position = bbp
                    checkpoint_rotation = bbr
                    checkpoint_field_loss = best_field_loss
                    checkpoint_loss_components = best_loss_components
                    checkpoint_global_step = best_global_step
                elif checkpoint_mode == "epoch_current":
                    checkpoint_position = last_position
                    checkpoint_rotation = last_rotation
                    checkpoint_field_loss = last_field_loss
                    checkpoint_loss_components = last_loss_components
                    checkpoint_global_step = last_global_step
                else:
                    checkpoint_position = bestposition
                    checkpoint_rotation = bestrotation
                    checkpoint_field_loss = epoch_best_field_loss
                    checkpoint_loss_components = epoch_best_loss_components
                    checkpoint_global_step = epoch_best_global_step
                self.save_epoch_checkpoint(
                    epoch_number,
                    checkpoint_position,
                    checkpoint_rotation,
                    checkpoint_mode,
                    stage="post_gradient",
                    field_loss=checkpoint_field_loss,
                    loss_components=checkpoint_loss_components,
                    global_step=checkpoint_global_step,
                )

        self.write_epoch_checkpoint_manifest()

        with torch.no_grad():
            best_voxelmodel, best_visibility = voxel_model(
                self.args,
                self.voxelnormals,
                bbr.cpu().numpy(),
                bbp.cpu().numpy(),
            )
            rate_v, _ = self.coverage_gap_from_visibility(best_visibility, weighted=True)
            unweighted_rate_v = None
            unweighted_joint_score = None
            if self.args.occupancy_map_enable:
                unweighted_rate_v, _ = self.coverage_gap_from_visibility(best_visibility, weighted=False)
                unweighted_joint_score = self.global_need_score(best_voxelmodel[:, 6:].cpu().numpy(), weighted=False)
            self.print_coverage_summary(
                "Best placement quality",
                rate_v,
                best_joint_score,
                position=bbp.cpu().numpy(),
                unweighted_voxel_gap=unweighted_rate_v,
                unweighted_joint_gap=unweighted_joint_score,
                visibility=best_visibility,
            )
        saveTrainingResult(self.posepath+"after.npy",bbp,bbr,self.scale)
        return bbp.cpu().numpy(),bbr.cpu().numpy()
