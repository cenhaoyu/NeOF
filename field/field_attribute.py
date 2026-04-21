from dataset.utils import *
import open3d as o3d


def get_fov_from_intrinsic_matrix(K):
    fx = K[0, 0]
    fy = K[1, 1]
    fov_x = 2 * np.arctan(K[0,2]/K[0,0])
    fov_y = 2 * np.arctan(K[1,2]/K[1,1])
    return fov_x, fov_y


def filter_camera_space_points(point_c, camera_model=None, return_projection=False):
    if camera_model is None:
        camera_model = get_camera_model(0)
    valid_indices, pixels, depths = project_camera_space_points(point_c, camera_model)
    if return_projection:
        return valid_indices, pixels, depths
    return valid_indices


def is_point_in_fov(point, rotation, position, camera_model=None):
    point_c=np.dot(rotation,(point-position).T).T
    return filter_camera_space_points(point_c, camera_model=camera_model)


def get_visible_points(pointnormals,position,rotation,radius=300,camera_model=None,return_point_cam=False):
    point_cloud=o3d.geometry.PointCloud()
    point_cloud.points=o3d.utility.Vector3dVector(pointnormals[:,:3])
    point_cloud.normals=o3d.utility.Vector3dVector(pointnormals[:,3:])
    _, pt_map = point_cloud.hidden_point_removal(position, radius)
    point=pointnormals[pt_map,:3]
    point_c=np.dot(rotation,(point-position).T).T
    judge=filter_camera_space_points(point_c, camera_model=camera_model)
    visible_indices = np.array(pt_map)[judge]
    visible_points = pointnormals[visible_indices,:3]
    if return_point_cam:
        return visible_indices, visible_points, point_c[judge]
    return visible_indices, visible_points


def get_visible_points_free_space(pointnormals, position, rotation, camera_model=None, return_point_cam=False):
    point_c = np.dot(rotation, (pointnormals[:, :3] - position).T).T
    judge = filter_camera_space_points(point_c, camera_model=camera_model)
    visible_points = pointnormals[judge, :3]
    if return_point_cam:
        return judge, visible_points, point_c[judge]
    return judge, visible_points


def get_visiblep_opt(pointnormals,position,rotation,radius,camera_model=None):
    points=np.dot(rotation,(pointnormals[:,:3]-position).T).T
    point_cloud=o3d.geometry.PointCloud()
    point_cloud.points=o3d.utility.Vector3dVector(points)
    point_cloud.normals=o3d.utility.Vector3dVector(pointnormals[:,3:])
    _, pt_map = point_cloud.hidden_point_removal(np.ones(shape=[3])*1e-5, radius)
    
    point=points[pt_map]
    
    judge=filter_camera_space_points(point, camera_model=camera_model)
    return point[judge],np.asarray(pt_map)[judge]


def sampling_quality_score_from_point_cam(point_cam, camera_model, voxel_size, temperature):
    if len(point_cam) == 0:
        return np.zeros(0, dtype=float)
    depth = np.clip(point_cam[:, 2], 1e-6, None)
    projected_voxel_px = min(float(camera_model["fx"]), float(camera_model["fy"])) * float(voxel_size) / depth
    threshold = float(camera_model["min_projected_voxel_px"])
    temperature = max(float(temperature), 1e-6)
    return 1.0 / (1.0 + np.exp(-(projected_voxel_px - threshold) / temperature))


def aggregate_sampling_quality_deficit(quality_scores, topk):
    quality_scores = np.asarray(quality_scores, dtype=float)
    if quality_scores.ndim != 2:
        raise ValueError("quality_scores must have shape (N_voxel, N_camera)")
    topk = int(max(1, min(topk, quality_scores.shape[1])))
    strongest = np.sort(quality_scores, axis=1)[:, -topk:]
    quality = np.mean(strongest, axis=1)
    return 1.0 - np.clip(quality, 0.0, 1.0)


def clipped_cosine(vec_a, vec_b, clip_min=0.0, clip_max=1.0):
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a <= 1e-8 or norm_b <= 1e-8:
        return clip_max
    cosine = np.dot(vec_a, vec_b) / (norm_a * norm_b)
    return np.clip(cosine, clip_min, clip_max)


def effective_camera_camera_angle(vec_a, vec_b):
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a <= 1e-8 or norm_b <= 1e-8:
        return np.pi / 2
    cosine = np.dot(vec_a, vec_b) / (norm_a * norm_b)
    cosine = np.clip(np.abs(cosine), 0.0, 1.0)
    return float(np.arccos(cosine))


def calculateCOCC(cameraposition,pointnormal):
    ray = cameraposition - pointnormal[:3]
    raymean = np.mean(ray, axis=0)
    co = 1 - clipped_cosine(raymean, pointnormal[3:], clip_min=0.0, clip_max=1.0)

    if len(cameraposition) < 2:
        return co, np.pi / 2

    angle_terms = []
    for i in range(len(cameraposition) - 1):
        for j in range(i + 1, len(cameraposition)):
            angle = effective_camera_camera_angle(ray[i], ray[j])
            angle_terms.append(np.pi / 2 - angle)

    if len(angle_terms) == 0:
        return co, np.pi / 2
    cc = float(np.mean(angle_terms))
    return co, np.clip(cc, 0.0, np.pi / 2)


def calculate_camera_camera_deficit(cameraposition, point):
    if len(cameraposition) < 2:
        return np.pi / 2

    rays = cameraposition - point[None, :]
    angle_terms = []
    for i in range(len(rays) - 1):
        for j in range(i + 1, len(rays)):
            angle = effective_camera_camera_angle(rays[i], rays[j])
            angle_terms.append(np.pi / 2 - angle)

    if len(angle_terms) == 0:
        return np.pi / 2
    return float(np.clip(np.mean(angle_terms), 0.0, np.pi / 2))


def uses_free_space_support(args):
    return bool(
        getattr(args, "occupancy_map_enable", False)
        and getattr(args, "occupancy_map_mode", None) == "free_space_box"
    )


def voxel_model(args,voxelnormals,rotation,position):
    #################################get voxels in how many cameras################### 

    free_space_mode = uses_free_space_support(args)
    voxel_visibility = np.zeros([len(voxelnormals),len(position)])
    voxel_quality = np.zeros([len(voxelnormals),len(position)], dtype=float)
    for i in range(len(position)):
        camera_model = get_camera_model(i)
        if free_space_mode:
            judge, _, point_cam = get_visible_points_free_space(
                voxelnormals,
                position[i],
                rotation[i],
                camera_model=camera_model,
                return_point_cam=True,
            )
        else:
            point_cam, judge = get_visiblep_opt(
                voxelnormals,
                position[i],
                rotation[i],
                200,
                camera_model=camera_model,
            )
        voxel_visibility[judge,i]=1
        voxel_quality[judge, i] = sampling_quality_score_from_point_cam(
            point_cam,
            camera_model,
            voxel_size=args.voxelsize,
            temperature=args.sampling_quality_temperature,
        )
    voxel_unvis=args.kcoverage-np.clip(np.sum(voxel_visibility,axis=1),0,args.kcoverage)
    quality_deficit = aggregate_sampling_quality_deficit(
        voxel_quality,
        topk=getattr(args, "sampling_quality_topk", 2),
    )

    angle_cc = np.ones([len(voxelnormals)], dtype=float) * (np.pi / 2)
    angle_co = np.ones([len(voxelnormals)], dtype=float)
    if free_space_mode:
        angle_co = np.zeros([len(voxelnormals)], dtype=float)
    for i in range(len(voxelnormals)):
        cameraindex = np.where(voxel_visibility[i] > 0)[0]
        if len(cameraindex) == 0:
            continue
        if free_space_mode:
            angle_cc[i] = calculate_camera_camera_deficit(position[cameraindex], voxelnormals[i, :3])
            continue
        if len(cameraindex) == 1:
            ray = position[cameraindex[0]] - voxelnormals[i, :3]
            angle_co[i] = 1 - clipped_cosine(ray, voxelnormals[i, 3:], clip_min=0.0, clip_max=1.0)
            continue
        angle_co[i], angle_cc[i] = calculateCOCC(position[cameraindex], voxelnormals[i])

    grid_return = np.concatenate(
        (
            voxelnormals,
            voxel_unvis[:, None],
            angle_cc[:, None],
            angle_co[:, None],
            quality_deficit[:, None],
        ),
        axis=1,
    )
    return npToTensor(grid_return),voxel_visibility
