from scipy.spatial.transform import Rotation as R
import math
import torch
import numpy as np
from camera_constraints import (
    camera_constraint_is_surface,
    point_in_camera_constraint,
    sample_camera_points_in_constraint,
)
from dataset.utils import *


def format_vector(vec):
    return "[" + ", ".join(f"{float(value):.4f}" for value in np.asarray(vec)) + "]"


class FarthestSampler:
    def __init__(self):
        pass
    def _calc_distances(self, p0, points):
        return ((p0 - points) ** 2).sum(axis=1)
    def __call__(self, pts, k):
        index=np.zeros(k, dtype=int)
        farthest_pts = np.zeros((k, 6), dtype=np.float32)
        farthest_pts[0] = pts[0]
        distances = self._calc_distances(farthest_pts[0,:3], pts[:,:3])
        for i in range(1, k):
            farthest_pts[i] = pts[np.argmax(distances)]
            index[i]=np.argmax(distances)
            distances = np.minimum(
                distances, self._calc_distances(farthest_pts[i,:3], pts[:,:3]))
        return farthest_pts,index
class KNNaverageCamera:
    def __init__(self):
        pass
    def _calc_distances(self, p0, points):
        return ((p0 - points) ** 2).sum(axis=1)
    def __call__(self, selectposition, pointnormals,k):
        selectnormal=np.zeros_like(selectposition)
        for i in range(len(selectposition)):
            distance=self._calc_distances(selectposition[i],pointnormals[:,:3])
            index=np.argsort(distance)[:k]
            selectnormal[i]=np.mean(pointnormals[index,3:],axis=0)
        return selectnormal
class KnnPoints:
    def __init__(self):
        pass
    def _calc_distances(self, p0, points):
        return ((p0 - points) ** 2).sum(axis=1)
    def __call__(self,pointnormals,k):
        index_all=torch.zeros([len(pointnormals),k],dtype=int)
        for i in range(len(pointnormals)):
            normali=pointnormals[i,3:].unsqueeze(-1)
            mm=torch.mm(pointnormals[:,3:],normali).squeeze(-1)
            index1=(mm>0).nonzero().squeeze(-1)
            pointselect=torch.index_select(pointnormals,0,index1)
            distance=self._calc_distances(pointnormals[i,:3],pointselect[:,:3])
            _,index=torch.sort(distance,descending=False)
            index_all[i]=index1[index[:k]]
        return index_all
#####################################init with existed camera parameter###################
def read_CameraParams(num,camerapath):
    cameraExlist={}
    for i in range(num):
        # name=camerapath+"CCD"+str(i+1)+".cal"
        name=camerapath+"CCD"+str(i)+".txt"
        intrinsic = np.genfromtxt(name,dtype=float,delimiter=' ',skip_footer=3)[:,:3]
        extrinsic = np.genfromtxt(name,dtype=float,delimiter=' ',skip_header=4)[:,:4]
        cameraExlist[i]=extrinsic
    return intrinsic,cameraExlist
def cameraextrinc(num,camerapath):
    intrinsic,cameraExlist=read_CameraParams(num,camerapath)
    Rs={}
    Cs={}
    for key,value in cameraExlist.items():
        R=value[:,0:3]
        T=value[:,3:]
        C=np.dot(-R.transpose(),T).reshape(3)
        # print(C)
        Rs[key]=R
        Cs[key]=C
    return intrinsic,Rs,Cs

############################################init with random camera parameter##############
def getQuaternion(fromVector, toVector):
        fromVector = np.array(fromVector)
        fromVector_e = fromVector / np.linalg.norm(fromVector)
        toVector = np.array(toVector)
        toVector_e = toVector / np.linalg.norm(toVector)
        cross = np.cross(toVector_e, fromVector_e)
        cross_e = cross / np.linalg.norm(cross)
        dot = np.dot(fromVector_e, toVector_e)
        angle = math.acos(dot)
        # print("angle",angle,"dot:",dot,"tovector:",toVector)
        if angle == 0 :
            return 1
        elif angle == math.pi:
            return -1
        else:
            return [cross_e[0]*math.sin(angle/2), cross_e[1]*math.sin(angle/2), cross_e[2]*math.sin(angle/2), math.cos(angle/2)]
def QuatToRotateMatrix(quat):
    if quat == 1:
        I=np.identity(3)
        for i in range(len(I)):
            for j in range(len(I[0])):
                if I[i][j]==0 :
                    I[i][j]=1e-8
        return np.identity(3)
    elif quat == -1 :
        I=np.identity(3)
        for i in range(len(I)):
            for j in range(len(I[0])):
                if I[i][j]==0 :
                    I[i][j]=1e-8
        I[2,2]=-1
        return I
    else:
        r1=R.from_quat(quat)
        rotate=r1.as_matrix()
    return rotate


def build_grid_axis(min_value, max_value, count):
    step = (max_value - min_value) / count
    return min_value + step * (np.arange(count, dtype=float) + 0.5)


def format_grid_shape(grid_shape):
    if isinstance(grid_shape, str):
        return grid_shape
    values = np.asarray(grid_shape, dtype=int).reshape(-1)
    return "x".join(str(int(value)) for value in values.tolist())


def allocate_weighted_integer_counts(total_count, weights):
    weights = np.asarray(weights, dtype=float)
    if total_count <= 0:
        return np.zeros_like(weights, dtype=int)
    positive_mask = weights > 0
    if not np.any(positive_mask):
        return np.zeros_like(weights, dtype=int)

    counts = np.zeros_like(weights, dtype=int)
    positive_indices = np.where(positive_mask)[0]
    base_count = min(len(positive_indices), total_count)
    counts[positive_indices[:base_count]] = 1
    remaining = total_count - np.sum(counts)
    if remaining <= 0:
        return counts

    normalized = weights[positive_mask] / np.sum(weights[positive_mask])
    raw = normalized * remaining
    floor = np.floor(raw).astype(int)
    counts[positive_indices] += floor
    leftover = remaining - int(np.sum(floor))
    if leftover > 0:
        order = np.argsort(-(raw - floor))
        counts[positive_indices[order[:leftover]]] += 1
    return counts


def allocate_symmetric_face_pair_counts(total_count, pair_weights):
    pair_weights = np.asarray(pair_weights, dtype=float)
    if total_count <= 0:
        return np.zeros(6, dtype=int)

    pair_counts = np.zeros(3, dtype=int)
    pair_instances = allocate_weighted_integer_counts(total_count // 2, pair_weights)
    pair_counts += 2 * pair_instances
    remainder = total_count - int(np.sum(pair_counts))
    if remainder > 0:
        pair_counts[int(np.argmax(pair_weights))] += remainder

    face_counts = np.zeros(6, dtype=int)
    for pair_index in range(3):
        base = pair_counts[pair_index] // 2
        face_counts[2 * pair_index] = base
        face_counts[2 * pair_index + 1] = base
        if pair_counts[pair_index] % 2 == 1:
            face_counts[2 * pair_index + 1] += 1
    return face_counts


def ellipse_perimeter_approx(rx, ry):
    rx = float(abs(rx))
    ry = float(abs(ry))
    if rx <= 1e-8 and ry <= 1e-8:
        return 0.0
    if rx <= 1e-8:
        return 4.0 * ry
    if ry <= 1e-8:
        return 4.0 * rx
    h = ((rx - ry) ** 2) / ((rx + ry) ** 2)
    return math.pi * (rx + ry) * (1.0 + (3.0 * h) / (10.0 + math.sqrt(max(4.0 - 3.0 * h, 1e-8))))


def ellipse_arc_midpoints(rx, ry, count, table_size=4096):
    if count <= 1:
        return np.array([0.0], dtype=float)

    theta_samples = np.linspace(-math.pi, math.pi, int(table_size) + 1, dtype=float)
    speed = np.sqrt((rx * np.sin(theta_samples)) ** 2 + (ry * np.cos(theta_samples)) ** 2)
    delta = theta_samples[1] - theta_samples[0]
    cumulative = np.zeros_like(theta_samples)
    cumulative[1:] = np.cumsum(0.5 * (speed[:-1] + speed[1:]) * delta)
    targets = (np.arange(count, dtype=float) + 0.5) * (cumulative[-1] / float(count))
    return np.interp(targets, cumulative, theta_samples)


def choose_volume_grid_shape(cameranum, box_min, box_max):
    lengths = np.asarray(box_max, dtype=float) - np.asarray(box_min, dtype=float)
    exact_candidates = []
    relaxed_candidates = []
    for nx in range(1, cameranum + 1):
        for ny in range(1, cameranum + 1):
            for nz in range(1, cameranum + 1):
                total = nx * ny * nz
                if total < cameranum:
                    continue
                dims = np.array([nx, ny, nz], dtype=int)
                cell = lengths / dims
                isotropy = np.std(np.log(np.maximum(cell, 1e-8)))
                if total == cameranum:
                    exact_candidates.append((isotropy, dims))
                else:
                    overfill = (total - cameranum) / max(float(cameranum), 1.0)
                    relaxed_candidates.append((isotropy + 0.2 * overfill, dims))
    if len(exact_candidates) > 0:
        exact_candidates.sort(key=lambda item: item[0])
        return exact_candidates[0][1]
    relaxed_candidates.sort(key=lambda item: item[0])
    return relaxed_candidates[0][1]


def choose_surface_grid_shape(cameranum, lengths):
    lengths = np.asarray(lengths, dtype=float)
    exact_candidates = []
    relaxed_candidates = []
    for nu in range(1, cameranum + 1):
        for nv in range(1, cameranum + 1):
            total = nu * nv
            if total < cameranum:
                continue
            dims = np.array([nu, nv], dtype=int)
            cell = lengths / dims
            isotropy = np.std(np.log(np.maximum(cell, 1e-8)))
            if total == cameranum:
                exact_candidates.append((isotropy, dims))
            else:
                overfill = (total - cameranum) / max(float(cameranum), 1.0)
                relaxed_candidates.append((isotropy + 0.2 * overfill, dims))
    if len(exact_candidates) > 0:
        exact_candidates.sort(key=lambda item: item[0])
        return exact_candidates[0][1]
    relaxed_candidates.sort(key=lambda item: item[0])
    return relaxed_candidates[0][1]


def build_surface_grid_points(cameranum, camera_constraint_shape, camera_constraint_min, camera_constraint_max, camera_constraint_data):
    if camera_constraint_shape == "box_surface":
        lengths = np.asarray(camera_constraint_max, dtype=float) - np.asarray(camera_constraint_min, dtype=float)
        face_specs = [
            (0, camera_constraint_min[0], 1, 2, lengths[1], lengths[2]),
            (0, camera_constraint_max[0], 1, 2, lengths[1], lengths[2]),
            (1, camera_constraint_min[1], 0, 2, lengths[0], lengths[2]),
            (1, camera_constraint_max[1], 0, 2, lengths[0], lengths[2]),
            (2, camera_constraint_min[2], 0, 1, lengths[0], lengths[1]),
            (2, camera_constraint_max[2], 0, 1, lengths[0], lengths[1]),
        ]
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
        pair_weights = np.array(
            [
                lengths[1] * lengths[2],
                lengths[0] * lengths[2],
                lengths[0] * lengths[1],
            ],
            dtype=float,
        )
        face_counts = allocate_symmetric_face_pair_counts(cameranum, pair_weights)
        surface_points = []
        for face_index, (fixed_axis, fixed_value, axis_u, axis_v, length_u, length_v) in enumerate(face_specs):
            if face_counts[face_index] <= 0:
                continue
            face_grid_shape = choose_surface_grid_shape(
                int(face_counts[face_index]),
                lengths=np.array([length_u, length_v], dtype=float),
            )
            values_u = build_grid_axis(camera_constraint_min[axis_u], camera_constraint_max[axis_u], int(face_grid_shape[0]))
            values_v = build_grid_axis(camera_constraint_min[axis_v], camera_constraint_max[axis_v], int(face_grid_shape[1]))
            for u in values_u:
                for v in values_v:
                    point = np.zeros(3, dtype=float)
                    point[fixed_axis] = fixed_value
                    point[axis_u] = u
                    point[axis_v] = v
                    surface_points.append(point)
        return np.asarray(surface_points, dtype=float), f"symmetric face grid ({len(surface_points)} points)"

    if camera_constraint_shape == "plane":
        span_u = np.asarray(camera_constraint_data["plane_span_u"], dtype=float)
        span_v = np.asarray(camera_constraint_data["plane_span_v"], dtype=float)
        plane_center = np.asarray(camera_constraint_data["plane_center"], dtype=float)
        grid_shape = choose_surface_grid_shape(
            cameranum,
            lengths=np.array([2.0 * np.linalg.norm(span_u), 2.0 * np.linalg.norm(span_v)], dtype=float),
        )
        alpha_values = np.linspace(-1.0, 1.0, int(grid_shape[0]), endpoint=False) + 1.0 / int(grid_shape[0])
        beta_values = np.linspace(-1.0, 1.0, int(grid_shape[1]), endpoint=False) + 1.0 / int(grid_shape[1])
        surface_points = []
        for alpha in alpha_values:
            for beta in beta_values:
                surface_points.append(plane_center + alpha * span_u + beta * span_v)
        return np.asarray(surface_points, dtype=float), grid_shape

    if camera_constraint_shape == "cylinder_surface":
        rx, ry = 0.5 * (np.asarray(camera_constraint_max, dtype=float)[:2] - np.asarray(camera_constraint_min, dtype=float)[:2])
        z_range = np.asarray(camera_constraint_max, dtype=float)[2] - np.asarray(camera_constraint_min, dtype=float)[2]
        perimeter_scale = ellipse_perimeter_approx(rx, ry)
        grid_shape = choose_surface_grid_shape(cameranum, lengths=np.array([perimeter_scale, z_range], dtype=float))
        theta_values = ellipse_arc_midpoints(rx, ry, int(grid_shape[0]))
        z_values = build_grid_axis(camera_constraint_min[2], camera_constraint_max[2], int(grid_shape[1]))
        center = 0.5 * (np.asarray(camera_constraint_min, dtype=float) + np.asarray(camera_constraint_max, dtype=float))
        surface_points = []
        for theta in theta_values:
            for z in z_values:
                surface_points.append([center[0] + rx * math.cos(theta), center[1] + ry * math.sin(theta), z])
        return np.asarray(surface_points, dtype=float), grid_shape

    if camera_constraint_shape == "dome_surface":
        rx, ry = 0.5 * (np.asarray(camera_constraint_max, dtype=float)[:2] - np.asarray(camera_constraint_min, dtype=float)[:2])
        rz = np.asarray(camera_constraint_max, dtype=float)[2] - np.asarray(camera_constraint_min, dtype=float)[2]
        dome_center = np.array(
            [
                0.5 * (camera_constraint_min[0] + camera_constraint_max[0]),
                0.5 * (camera_constraint_min[1] + camera_constraint_max[1]),
                camera_constraint_min[2],
            ],
            dtype=float,
        )
        meridian_scale = 0.5 * ellipse_perimeter_approx(0.5 * (rx + ry), rz)
        azimuth_scale = ellipse_perimeter_approx(rx, ry)
        grid_shape = choose_surface_grid_shape(cameranum, lengths=np.array([azimuth_scale, meridian_scale], dtype=float))
        theta_values = np.linspace(-math.pi, math.pi, int(grid_shape[0]), endpoint=False) + math.pi / int(grid_shape[0])
        z_dir_values = (np.arange(int(grid_shape[1]), dtype=float) + 0.5) / float(grid_shape[1])
        surface_points = []
        for theta in theta_values:
            for z_dir in z_dir_values:
                xy_dir = math.sqrt(max(1.0 - z_dir * z_dir, 0.0))
                surface_points.append(
                    [
                        dome_center[0] + rx * xy_dir * math.cos(theta),
                        dome_center[1] + ry * xy_dir * math.sin(theta),
                        dome_center[2] + rz * z_dir,
                    ]
                )
        return np.asarray(surface_points, dtype=float), grid_shape

    raise ValueError(f"Unsupported surface camera constraint shape: {camera_constraint_shape}")


def farthest_point_sample(points, sample_num):
    if len(points) <= sample_num:
        return np.arange(len(points), dtype=int)

    points = np.asarray(points, dtype=float)
    center = np.mean(points, axis=0)
    first = int(np.argmin(np.linalg.norm(points - center, axis=1)))
    selected = [first]
    min_distance = np.linalg.norm(points - points[first], axis=1)
    while len(selected) < sample_num:
        next_index = int(np.argmax(min_distance))
        selected.append(next_index)
        min_distance = np.minimum(min_distance, np.linalg.norm(points - points[next_index], axis=1))
    return np.array(selected, dtype=int)


def getLookAtRotation(camera_center, target_point):
    zaxis = np.asarray(target_point, dtype=float) - np.asarray(camera_center, dtype=float)
    if np.linalg.norm(zaxis) <= 1e-8:
        zaxis = np.array([0.0, 0.0, 1.0], dtype=float)
    return QuatToRotateMatrix(getQuaternion(np.array([0,0,1]), zaxis))


def initVolumeGridCameras(
    cameranum,
    voxelnormals,
    camera_constraint_min,
    camera_constraint_max,
    camera_constraint_shape="box",
    camera_constraint_data=None,
):
    if camera_constraint_min is None or camera_constraint_max is None:
        raise ValueError("resolved camera constraint bounds are required for grid initialization")

    if camera_constraint_is_surface(camera_constraint_shape):
        volume_points, grid_shape = build_surface_grid_points(
            cameranum,
            camera_constraint_shape,
            camera_constraint_min,
            camera_constraint_max,
            camera_constraint_data,
        )
    else:
        grid_shape = choose_volume_grid_shape(cameranum, camera_constraint_min, camera_constraint_max)
        xs = build_grid_axis(camera_constraint_min[0], camera_constraint_max[0], int(grid_shape[0]))
        ys = build_grid_axis(camera_constraint_min[1], camera_constraint_max[1], int(grid_shape[1]))
        zs = build_grid_axis(camera_constraint_min[2], camera_constraint_max[2], int(grid_shape[2]))
        grid_points = np.stack(np.meshgrid(xs, ys, zs, indexing='ij'), axis=-1).reshape(-1, 3)
        valid_mask = np.array(
            [
                point_in_camera_constraint(
                    point,
                    camera_constraint_shape,
                    camera_constraint_min,
                    camera_constraint_max,
                    constraint_data=camera_constraint_data,
                )
                for point in grid_points
            ],
            dtype=bool,
        )
        volume_points = grid_points[valid_mask]
        if len(volume_points) < cameranum:
            extra_points = sample_camera_points_in_constraint(
                cameranum - len(volume_points),
                camera_constraint_shape,
                camera_constraint_min,
                camera_constraint_max,
                constraint_data=camera_constraint_data,
            )
            volume_points = np.concatenate((volume_points, extra_points), axis=0)

    selected_indices = farthest_point_sample(volume_points, cameranum)
    selected_points = volume_points[selected_indices]

    target_point = np.mean(voxelnormals[:, :3], axis=0)
    Cs = {}
    Rs = {}
    for i, camera_center in enumerate(selected_points):
        Cs[i] = camera_center
        Rs[i] = getLookAtRotation(camera_center, target_point)
        print(
            f"Init camera {i+1:02d}/{cameranum:02d} | "
            f"{camera_constraint_shape}-grid center: {format_vector(camera_center)} | "
            f"look-at target: {format_vector(target_point)} | "
            f"grid shape: {format_grid_shape(grid_shape)}"
        )
    return Rs, Cs


def initVolumeRandomCameras(
    cameranum,
    voxelnormals,
    camera_constraint_min,
    camera_constraint_max,
    camera_constraint_shape="box",
    camera_constraint_data=None,
):
    if camera_constraint_min is None or camera_constraint_max is None:
        raise ValueError("resolved camera constraint bounds are required for random initialization")

    target_point = np.mean(voxelnormals[:, :3], axis=0)
    sampled_centers = sample_camera_points_in_constraint(
        cameranum,
        camera_constraint_shape,
        camera_constraint_min,
        camera_constraint_max,
        constraint_data=camera_constraint_data,
    )
    Cs = {}
    Rs = {}
    for i, camera_center in enumerate(sampled_centers):
        Cs[i] = camera_center
        Rs[i] = getLookAtRotation(camera_center, target_point)
        print(
            f"Init camera {i+1:02d}/{cameranum:02d} | "
            f"{camera_constraint_shape}-random center: {format_vector(camera_center)} | "
            f"look-at target: {format_vector(target_point)}"
        )
    return Rs, Cs

##################################get standard camera parameter from intrinsic,Rs,Cs##############################
# generate camerapose matrix (cameranum,9)
def getCameraPose(Rs,Cs):
    camerapose=np.empty(shape=[0,9])
    for i in range(len(Cs)):
        cpose=np.append(Cs[i].reshape(1,3),np.append(Rs[i][0,:].reshape(1,3),Rs[i][2,:].reshape(1,3),axis=1),axis=1)
        camerapose=np.append(camerapose,cpose,axis=0)
    return camerapose
def intrinsicTotensor(cameranum,intrinsic):
    I=np.tile(intrinsic,(cameranum,1,1))
    return torch.tensor(I,dtype=torch.float,requires_grad=False).to(device)
