from dataset.utils import *
import open3d as o3d


def get_fov_from_intrinsic_matrix(K):
    fx = K[0, 0]
    fy = K[1, 1]
    fov_x = 2 * np.arctan(K[0,2]/K[0,0])
    fov_y = 2 * np.arctan(K[1,2]/K[1,1])
    return fov_x, fov_y


def filter_camera_space_points(point_c):
    z = point_c[:, 2]
    judge_h = np.where(z > 1e-8)[0]
    if len(judge_h) == 0:
        return judge_h

    tan_aov_x = intrinsic[0,2]/intrinsic[0,0]
    tan_aov_y = intrinsic[1,2]/intrinsic[1,1]
    x = point_c[judge_h,0] / point_c[judge_h,2]
    y = point_c[judge_h,1] / point_c[judge_h,2]
    judge = (x  >= -tan_aov_x) * (x <= tan_aov_x) * (y >= -tan_aov_y) * (y <= tan_aov_y)

    judge_aov=np.where(judge)[0]
    return judge_h[judge_aov]


def is_point_in_fov(point, rotation, position):
    #get aov
    # aov_x,aov_y=get_fov_from_intrinsic_matrix(intrinsic)
    # 将点从世界坐标系转换到相机坐标系
    point_c=np.dot(rotation,(point-position).T).T
    return filter_camera_space_points(point_c)


def get_visible_points(pointnormals,position,rotation,radius=300):
    point_cloud=o3d.geometry.PointCloud()
    point_cloud.points=o3d.utility.Vector3dVector(pointnormals[:,:3])
    point_cloud.normals=o3d.utility.Vector3dVector(pointnormals[:,3:])
    # point_cloud.colors = o3d.utility.Vector3dVector(np.random.uniform(0, 1,size=(len(pointnormals), 3)))
    _, pt_map = point_cloud.hidden_point_removal(position, radius)
    #correspond
    point=pointnormals[pt_map,:3]
    judge=is_point_in_fov(point,rotation,position)
    return np.array(pt_map)[judge],pointnormals[np.array(pt_map)[judge],:3]


def get_visible_points_free_space(pointnormals, position, rotation):
    judge = is_point_in_fov(pointnormals[:, :3], rotation, position)
    return judge, pointnormals[judge, :3]


def get_visiblep_opt(pointnormals,position,rotation,radius):
    points=np.dot(rotation,(pointnormals[:,:3]-position).T).T
    point_cloud=o3d.geometry.PointCloud()
    point_cloud.points=o3d.utility.Vector3dVector(points)
    point_cloud.normals=o3d.utility.Vector3dVector(pointnormals[:,3:])
    # point_cloud.colors = o3d.utility.Vector3dVector(np.random.uniform(0, 1,size=(len(pointnormals), 3)))
    _, pt_map = point_cloud.hidden_point_removal(np.ones(shape=[3])*1e-5, radius)
    # pcd_down=pcd.voxel_down_sample(voxel_size=0.001)
    # print(np.asarray(pcd.points).shape,np.asarray(pcd_down.points).shape)
    #correspond
    
    point=points[pt_map]
    
    judge=filter_camera_space_points(point)
    return point[judge],np.asarray(pt_map)[judge]


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

    #voxel center vis
    free_space_mode = uses_free_space_support(args)
    voxel_visibility = np.zeros([len(voxelnormals),len(position)])
    for i in range(len(position)):
        if free_space_mode:
            judge,_=get_visible_points_free_space(voxelnormals,position[i],rotation[i])
        else:
            judge,_=get_visible_points(voxelnormals,position[i],rotation[i],200)
        voxel_visibility[judge,i]=1
    voxel_unvis=args.kcoverage-np.clip(np.sum(voxel_visibility,axis=1),0,args.kcoverage)

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
        ),
        axis=1,
    )
    return npToTensor(grid_return),voxel_visibility
