from dataset.dataset import *
from camera_constraints import (
    torch_constraint_parameter_to_position,
    torch_position_to_constraint_parameter,
)
from field.field_attribute import get_visiblep_opt
from torch.nn import Parameter


def safe_temperature(value):
    return max(float(value), 1e-6)


class GenerateP3d(torch.nn.Module):
    def __init__(
        self,
        cameranum,
        camera_constraint_shape="box",
        camera_constraint_min=None,
        camera_constraint_max=None,
        camera_constraint_data=None,
        target_support_mode="surface",
    ):
        super().__init__()
        self.use_camera_constraint = camera_constraint_min is not None and camera_constraint_max is not None
        self.camera_constraint_shape = camera_constraint_shape
        self.camera_constraint_data = None
        self.target_support_mode = target_support_mode
        if self.use_camera_constraint:
            self.register_buffer("camera_constraint_min", torch.tensor(camera_constraint_min, dtype=torch.float))
            self.register_buffer("camera_constraint_max", torch.tensor(camera_constraint_max, dtype=torch.float))
            if camera_constraint_shape == "plane":
                if camera_constraint_data is None:
                    raise ValueError("plane constraints require camera_constraint_data")
                self.register_buffer(
                    "camera_constraint_plane_center",
                    torch.tensor(camera_constraint_data["plane_center"], dtype=torch.float),
                )
                self.register_buffer(
                    "camera_constraint_plane_span_u",
                    torch.tensor(camera_constraint_data["plane_span_u"], dtype=torch.float),
                )
                self.register_buffer(
                    "camera_constraint_plane_span_v",
                    torch.tensor(camera_constraint_data["plane_span_v"], dtype=torch.float),
                )
        else:
            self.register_buffer("camera_constraint_min", torch.zeros(3, dtype=torch.float))
            self.register_buffer("camera_constraint_max", torch.ones(3, dtype=torch.float))
        self.position_parameter = Parameter(torch.zeros(size=[cameranum, 3], dtype=torch.float))
        self.rotate6d = Parameter(torch.zeros(size=[cameranum, 6], dtype=torch.float))

    def get_constraint_data(self):
        if not self.use_camera_constraint:
            return None
        if self.camera_constraint_shape == "plane":
            return {
                "shape": "plane",
                "plane_center": self.camera_constraint_plane_center,
                "plane_span_u": self.camera_constraint_plane_span_u,
                "plane_span_v": self.camera_constraint_plane_span_v,
            }
        return None

    def set_pose(self, camerapose):
        with torch.no_grad():
            position = camerapose[:, :3].to(device)
            if self.use_camera_constraint:
                self.position_parameter.copy_(
                    torch_position_to_constraint_parameter(
                        position,
                        self.camera_constraint_shape,
                        self.camera_constraint_min,
                        self.camera_constraint_max,
                        constraint_data=self.get_constraint_data(),
                    )
                )
            else:
                self.position_parameter.copy_(position)
            self.rotate6d.copy_(camerapose[:, 3:].to(device))

    def get_pose(self):
        if self.use_camera_constraint:
            position = torch_constraint_parameter_to_position(
                self.position_parameter,
                self.camera_constraint_shape,
                self.camera_constraint_min,
                self.camera_constraint_max,
                constraint_data=self.get_constraint_data(),
            )
        else:
            position = self.position_parameter
        rotation = compute_rotation_matrix_from_ortho6d(self.rotate6d)
        return position, rotation

    def reconstruction3D(self, P, point):
        point = npToTensor(point)
        A = torch.zeros(3 * len(P), 4 + len(P)).to(device)
        for i in range(len(P)):
            A[3 * i : 3 * (i + 1), :4] = -P[i]
            A[3 * i : 3 * (i + 1), 4 + i] = point[i]
        U, S, V = torch.svd(A)
        output0 = (V[0][-1] / V[3][-1]).to(torch.float32)
        output1 = (V[1][-1] / V[3][-1]).to(torch.float32)
        output2 = (V[2][-1] / V[3][-1]).to(torch.float32)
        return [output0, output1, output2]

    def getReconstructionLoss(self, camerapose, voxelnormals):
        del camerapose
        position, rotation = self.get_pose()
        t = torch.bmm(-rotation, position.unsqueeze(-1))
        E = torch.cat((rotation, t), -1)
        I = npToTensor(intrinsic).repeat(len(rotation), 1, 1)
        P = torch.bmm(I, E)
        relation = np.zeros([0, 5])
        for i in range(len(position)):
            with torch.no_grad():
                x_select, index = get_visiblep_opt(
                    voxelnormals,
                    position[i].detach().cpu().numpy(),
                    rotation[i].detach().cpu().numpy(),
                    200,
                )
                point3d = np.dot(intrinsic, x_select.T).T
                point3d = point3d / point3d[:, -1][:, None]
                relation = np.append(
                    relation,
                    np.append(index[:, None], np.append(np.ones([len(index), 1]) * i, point3d, 1), 1),
                    0,
                )
        reconstruction_loss = 0
        for j in range(len(voxelnormals)):
            index = (relation[:, 0] == j).nonzero()[0]
            if len(index) >= 2:
                output = self.reconstruction3D(P[relation[index, 1]], relation[index, 2:])
                reconstruction_loss += torch.sqrt(
                    (output[0] - voxelnormals[j, 0]) ** 2
                    + (output[1] - voxelnormals[j, 1]) ** 2
                    + (output[2] - voxelnormals[j, 2]) ** 2
                )
        return reconstruction_loss

    def soft_visibility_weights(
        self,
        position,
        rotation,
        voxel_points,
        voxel_normals,
        depth_temperature,
        fov_temperature,
        normal_temperature,
    ):
        eps = 1e-6
        tan_aov_x = float(intrinsic[0, 2] / intrinsic[0, 0])
        tan_aov_y = float(intrinsic[1, 2] / intrinsic[1, 1])

        relative = voxel_points.unsqueeze(0) - position.unsqueeze(1)
        point_cam = torch.matmul(rotation.unsqueeze(1), relative.unsqueeze(-1)).squeeze(-1)
        z = point_cam[:, :, 2]
        safe_z = torch.where(torch.abs(z) < eps, torch.full_like(z, eps), z)
        x_ratio = point_cam[:, :, 0] / safe_z
        y_ratio = point_cam[:, :, 1] / safe_z

        depth_gate = torch.sigmoid(z / safe_temperature(depth_temperature))

        fov_gate_x = torch.sigmoid((tan_aov_x - torch.abs(x_ratio)) / safe_temperature(fov_temperature))
        fov_gate_y = torch.sigmoid((tan_aov_y - torch.abs(y_ratio)) / safe_temperature(fov_temperature))

        ray = position.unsqueeze(1) - voxel_points.unsqueeze(0)
        ray_dir = ray / torch.clamp(torch.linalg.norm(ray, dim=-1, keepdim=True), min=eps)
        if self.target_support_mode == "free_space":
            normal_gate = torch.ones_like(depth_gate)
        else:
            normal_alignment = torch.sum(ray_dir * voxel_normals.unsqueeze(0), dim=-1)
            normal_gate = torch.sigmoid(normal_alignment / safe_temperature(normal_temperature))

        return depth_gate * fov_gate_x * fov_gate_y * normal_gate

    def forward(
        self,
        voxelmodel,
        fieldmodel,
        voxel_weights,
        depth_temperature,
        fov_temperature,
        normal_temperature,
    ):
        position, rotation = self.get_pose()
        voxel_points = voxelmodel[:, :3]
        voxel_normals = voxelmodel[:, 3:6]
        voxel_attributes = voxelmodel[:, 6:]

        predicted_attributes = fieldmodel(
            voxel_points.unsqueeze(1),
            voxel_points,
            voxel_normals,
            voxel_attributes,
        )
        if voxel_weights is not None:
            predicted_attributes = predicted_attributes * voxel_weights.unsqueeze(-1)
        visibility_weights = self.soft_visibility_weights(
            position,
            rotation,
            voxel_points,
            voxel_normals,
            depth_temperature=depth_temperature,
            fov_temperature=fov_temperature,
            normal_temperature=normal_temperature,
        )
        camera_attributes = torch.einsum("cn,nk->ck", visibility_weights, predicted_attributes)
        return camera_attributes, position, rotation
