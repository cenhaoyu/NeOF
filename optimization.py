import copy
import time

from camera_constraints import (
    constrained_camera_center_on_ray,
    point_in_camera_constraint,
    sample_camera_points_in_constraint,
)
from dataset.dataset import *
from field.field_attribute import voxel_model
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
            target_support_mode="free_space" if self.free_space_support else "surface",
        ).to(device)
        self.fieldmodel = AddAttention(
            6,
            args.field_hidden_dim,
            args.field_num_heads,
            context_voxel_num=args.field_context_voxel_num,
            normal_voxel_num=args.field_normal_voxel_num,
        ).to(device)

        weights = np.array([args.wvis, args.wcc, args.wco], dtype=np.float32)
        if self.free_space_support:
            weights[2] = 0.0
        weights_sum = float(np.sum(weights))
        if weights_sum <= 1e-8:
            weights = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        else:
            weights = weights / weights_sum
        self.loss_weights_np = weights
        self.attribute_upperbound_np = np.array([args.kcoverage, np.pi / 2, 1.0], dtype=np.float32)

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

    def candidate_center_from_surface(self, surface_point, outward_direction, preferred_distance=None):
        distance = self.args.preferred_distance if preferred_distance is None else preferred_distance
        return constrained_camera_center_on_ray(
            surface_point,
            outward_direction,
            distance,
            self.args.camera_constraint_min,
            self.args.camera_constraint_max,
            shape=self.args.camera_constraint_shape,
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
                print(
                    f"    Field fit step {last_step:04d}/{self.args.field_fit_steps} | "
                    f"supervision loss: {current_loss:.6e}"
                )

            if (
                self.args.field_early_stop
                and last_step >= self.args.field_early_stop_min_steps
                and stale_steps >= self.args.field_early_stop_patience
            ):
                print(
                    f"    Field fit early stop at step {last_step:04d}/{self.args.field_fit_steps} | "
                    f"best loss: {best_loss:.6e} at step {best_step:04d}"
                )
                break

        field_fit_time = time.time() - field_fit_start
        print(
            f"  Field fit time: {format_seconds(field_fit_time)} | "
            f"best loss: {best_loss:.6e} at step {best_step:04d}"
        )
        self.fieldmodel.eval()

    def reset_camera(self, position, rotation, min_camera, sample, scene_mode):
        voxelmodel, _ = voxel_model(
            self.args,
            self.voxelnormals,
            rotation,
            position,
        )
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
        center = np.mean(self.voxelnormals[:,:3],axis=0)

        for candidate_idx in candidate_indices:
            surface_point = self.voxelnormals[candidate_idx, :3]
            candidate_poses = []
            if self.args.camera_constraint_enable:
                candidate_poses = self.sample_random_candidate_poses(surface_point)
            else:
                outward_directions = []
                normal_direction = self.voxelnormals[candidate_idx, 3:]
                if np.linalg.norm(normal_direction) > 1e-8 and not self.free_space_support:
                    outward_directions.append(normal_direction)
                center_direction = surface_point - center
                center_norm = np.linalg.norm(center_direction)
                if scene_mode:
                    if np.linalg.norm(normal_direction) > 1e-8 and not self.free_space_support:
                        outward_directions.append(-normal_direction)
                    if center_norm > 1e-8:
                        outward_directions.append(center_direction / center_norm)
                elif center_norm > 1e-8 and len(outward_directions) == 0:
                    outward_directions.append(center_direction / center_norm)

                for outward_direction in outward_directions:
                    candidate_center = self.candidate_center_from_surface(surface_point, outward_direction)
                    if candidate_center is None:
                        continue
                    candidate_poses.append((candidate_center, getLookAtRotation(candidate_center, surface_point)))
                    if scene_mode:
                        half_distance_center = self.candidate_center_from_surface(
                            surface_point,
                            outward_direction,
                            preferred_distance=0.5 * self.args.preferred_distance,
                        )
                        if half_distance_center is not None:
                            candidate_poses.append((half_distance_center, getLookAtRotation(half_distance_center, surface_point)))

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

    def resetNewforScene(self,position,rotation,min_camera):
        return self.reset_camera(position, rotation, min_camera, sample=100, scene_mode=True)

    def resetNewforModel(self,position,rotation,min_camera):
        return self.reset_camera(position, rotation, min_camera, sample=20, scene_mode=False)

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
    ):
        del position
        print(title)
        if self.free_space_support:
            print("  Target support: free-space box (camera-object angle term disabled)")
        if self.args.occupancy_map_enable:
            print(f"  Weighted voxel K-coverage deficit (normalized): {voxel_gap:.4f}")
            if unweighted_voxel_gap is not None:
                print(f"  Unweighted voxel K-coverage deficit (normalized): {unweighted_voxel_gap:.4f}")
        else:
            print(f"  Voxel K-coverage deficit (normalized): {voxel_gap:.4f}")
        if joint_gap is not None:
            if self.args.occupancy_map_enable:
                print(f"  Weighted exact joint observation gap: {joint_gap:.4f}")
                if unweighted_joint_gap is not None:
                    print(f"  Unweighted exact joint observation gap: {unweighted_joint_gap:.4f}")
            else:
                print(f"  Exact joint observation gap: {joint_gap:.4f}")

    def opt(self,camerapose):
        bestposition = camerapose[:,:3].detach().clone()
        bestrotation = compute_rotation_matrix_from_ortho6d(camerapose[:,3:]).detach().clone()
        best_joint_score = np.inf
        bbp = bestposition.detach().clone()
        bbr = bestrotation.detach().clone()
        self.model.set_pose(camerapose.detach())
        pose_optimizer = self.build_pose_optimizer()
        self.fieldmodel.eval()

        with torch.no_grad():
            rate_v = self.metric(bbp.cpu().numpy(),bbr.cpu().numpy(), weighted=True)
            unweighted_rate_v = None
            initial_voxelmodel, _ = voxel_model(
                self.args,
                self.voxelnormals,
                bbr.cpu().numpy(),
                bbp.cpu().numpy(),
            )
            best_joint_score = self.global_need_score(initial_voxelmodel[:, 6:].cpu().numpy(), weighted=True)
            if self.args.occupancy_map_enable:
                unweighted_rate_v = self.metric(bbp.cpu().numpy(),bbr.cpu().numpy(), weighted=False)
            self.print_coverage_summary(
                "Initial placement quality",
                rate_v,
                position=bbp.cpu().numpy(),
                unweighted_voxel_gap=unweighted_rate_v,
            )

        for epoch in range(self.args.epoches):
            epoch_best_joint_score = np.inf
            position = copy.deepcopy(bestposition)
            rotation = copy.deepcopy(bestrotation)
            outer_epoch = epoch + 1
            print(f"Outer epoch {outer_epoch:02d}/{self.args.epoches:02d}")

            if epoch % 5 == 0:
                print("  Stage 1/2 | Non-gradient camera reset")
                for min_camera in range(self.args.cameranum):
                    if self.args.isscene:
                        position,rotation = self.resetNewforScene(position.cpu().numpy(),rotation.cpu().numpy(),min_camera)
                    else:
                        position,rotation = self.resetNewforModel(position.cpu().numpy(),rotation.cpu().numpy(),min_camera)
                    saveTrainingResult(self.posepath+str(epoch)+"_"+str(min_camera+1)+".npy",position,rotation,self.scale)
                pose_optimizer.state.clear()

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
            print(
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
                )
                raw_loss_components = torch.mean(attribute / self.voxel_occupancy_sum_np,dim=0)
                loss_components = raw_loss_components / self.attribute_upperbound()
                L = torch.sum(self.loss_weights() * loss_components)
                L.backward()
                pose_optimizer.step()
                torch.cuda.empty_cache()

                with torch.no_grad():
                    position, rotation = self.model.get_pose()
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
                    joint_score = self.global_need_score(current_voxelmodel[:, 6:].cpu().numpy(), weighted=True)
                    unweighted_joint_score = None
                    if self.args.occupancy_map_enable:
                        unweighted_joint_score = self.global_need_score(
                            current_voxelmodel[:, 6:].cpu().numpy(),
                            weighted=False,
                        )

                new_epoch_best = False
                new_global_best = False
                if joint_score < epoch_best_joint_score:
                    epoch_best_joint_score = joint_score
                    new_epoch_best = True
                    bestposition = position.detach().clone()
                    bestrotation = rotation.detach().clone()
                    if joint_score < best_joint_score:
                        best_joint_score = joint_score
                        new_global_best = True
                        bbp = position.detach().clone()
                        bbr = rotation.detach().clone()

                global_step = epoch * self.args.iterations + iters
                self.log_writer.add_scalar("loss/total",L,global_step)
                self.log_writer.add_scalar("loss/raw_vis",raw_loss_components[0],global_step)
                self.log_writer.add_scalar("loss/raw_cc",raw_loss_components[1],global_step)
                self.log_writer.add_scalar("loss/raw_co",raw_loss_components[2],global_step)
                self.log_writer.add_scalar("loss/norm_vis",loss_components[0],global_step)
                self.log_writer.add_scalar("loss/norm_cc",loss_components[1],global_step)
                self.log_writer.add_scalar("loss/norm_co",loss_components[2],global_step)
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
                co_label = "camera-object angle deficit (disabled)" if self.free_space_support else "camera-object angle deficit"
                print(
                    f"  Pose step {iters+1:02d}/{self.args.iterations:02d} | "
                    f"global step {global_step:03d}{status_text}\n"
                    f"    Field-based joint loss: {L:.4f}\n"
                    f"    Normalized loss components:\n"
                    f"      visibility deficit          = {loss_components[0]:.4f}\n"
                    f"      camera-camera angle deficit = {loss_components[1]:.4f}\n"
                    f"      {co_label:<28} = {loss_components[2]:.4f}\n"
                    f"    Exact voxel-model evaluation:\n"
                    f"      {voxel_gap_label:<28} = {rate_v:.4f}\n"
                    f"      {joint_gap_label:<28} = {joint_score:.4f}\n"
                    f"      voxels seen by >=1 camera   = {int(num_v)}/{len(v_coverage)}\n"
                    f"      step time                   = {format_seconds(end-start)}"
                )
                if self.args.occupancy_map_enable:
                    print(
                        f"      unweighted voxel deficit    = {unweighted_rate_v:.4f}\n"
                        f"      unweighted joint gap        = {unweighted_joint_score:.4f}"
                    )

        with torch.no_grad():
            rate_v = self.metric(bbp.cpu().numpy(),bbr.cpu().numpy(), weighted=True)
            unweighted_rate_v = None
            unweighted_joint_score = None
            if self.args.occupancy_map_enable:
                unweighted_rate_v = self.metric(bbp.cpu().numpy(),bbr.cpu().numpy(), weighted=False)
                best_voxelmodel, _ = voxel_model(
                    self.args,
                    self.voxelnormals,
                    bbr.cpu().numpy(),
                    bbp.cpu().numpy(),
                )
                unweighted_joint_score = self.global_need_score(best_voxelmodel[:, 6:].cpu().numpy(), weighted=False)
            self.print_coverage_summary(
                "Best placement quality",
                rate_v,
                best_joint_score,
                position=bbp.cpu().numpy(),
                unweighted_voxel_gap=unweighted_rate_v,
                unweighted_joint_gap=unweighted_joint_score,
            )
        saveTrainingResult(self.posepath+"after.npy",bbp,bbr,self.scale)
        return bbp.cpu().numpy(),bbr.cpu().numpy()
