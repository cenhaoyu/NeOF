import torch
import numpy as np
from scipy.spatial.transform import Rotation as R 

w,h=640,480
intrinsic=np.array([[320.0,0,319.5],
                   [0,320.0,239.5],
                   [0,0,1]], dtype=float)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DEFAULT_CAMERA_MODEL = {
    "name": "camera_00",
    "image_width": float(w),
    "image_height": float(h),
    "fx": float(intrinsic[0, 0]),
    "fy": float(intrinsic[1, 1]),
    "cx": float(intrinsic[0, 2]),
    "cy": float(intrinsic[1, 2]),
    "min_projected_voxel_px": 1.5,
}
CAMERA_MODELS = [dict(DEFAULT_CAMERA_MODEL)]
CAMERA_INTRINSICS = np.expand_dims(intrinsic.copy(), axis=0)


def _focal_length_from_fov(image_extent, fov_deg):
    if fov_deg is None:
        return None
    fov_deg = float(fov_deg)
    if fov_deg <= 0.0 or fov_deg >= 179.0:
        raise ValueError("FoV must be in (0, 179) degrees")
    return 0.5 * float(image_extent) / np.tan(np.deg2rad(fov_deg) * 0.5)


def _build_intrinsic_matrix(fx, fy, cx, cy):
    return np.array(
        [
            [float(fx), 0.0, float(cx)],
            [0.0, float(fy), float(cy)],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def _normalize_camera_model(camera_model, index, defaults):
    if camera_model is None:
        camera_model = {}
    if not isinstance(camera_model, dict):
        raise ValueError("each camera model must be a JSON object / Python dict")

    image_width = float(camera_model.get("image_width", defaults["image_width"]))
    image_height = float(camera_model.get("image_height", defaults["image_height"]))
    if image_width <= 0 or image_height <= 0:
        raise ValueError("camera image_width and image_height must be positive")

    fx = camera_model.get("fx", defaults["fx"])
    fy = camera_model.get("fy", defaults["fy"])
    cx = camera_model.get("cx", defaults["cx"])
    cy = camera_model.get("cy", defaults["cy"])
    fov_x_deg = camera_model.get("fov_x_deg")
    fov_y_deg = camera_model.get("fov_y_deg")
    if fx is None:
        fx = _focal_length_from_fov(image_width, fov_x_deg)
    if fy is None:
        fy = _focal_length_from_fov(image_height, fov_y_deg)
    if fx is None or fy is None:
        raise ValueError("each camera model must define fx/fy or corresponding fov_x_deg/fov_y_deg")

    fx = float(fx)
    fy = float(fy)
    if fx <= 0 or fy <= 0:
        raise ValueError("camera fx and fy must be positive")

    if cx is None:
        cx = (image_width - 1.0) * 0.5
    if cy is None:
        cy = (image_height - 1.0) * 0.5
    cx = float(cx)
    cy = float(cy)

    min_projected_voxel_px = float(
        camera_model.get(
            "min_projected_voxel_px",
            defaults["min_projected_voxel_px"],
        )
    )
    if min_projected_voxel_px <= 0:
        raise ValueError("min_projected_voxel_px must be positive")

    normalized = {
        "name": str(camera_model.get("name", f"camera_{index:02d}")),
        "image_width": image_width,
        "image_height": image_height,
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "min_projected_voxel_px": min_projected_voxel_px,
        "intrinsic": _build_intrinsic_matrix(fx, fy, cx, cy),
    }
    if fov_x_deg is not None:
        normalized["fov_x_deg"] = float(fov_x_deg)
    if fov_y_deg is not None:
        normalized["fov_y_deg"] = float(fov_y_deg)
    return normalized


def configure_camera_models(
    cameranum,
    camera_models=None,
    image_width=640,
    image_height=480,
    fx=320.0,
    fy=320.0,
    cx=319.5,
    cy=239.5,
    min_projected_voxel_px=1.5,
):
    defaults = {
        "image_width": float(image_width),
        "image_height": float(image_height),
        "fx": None if fx is None else float(fx),
        "fy": None if fy is None else float(fy),
        "cx": None if cx is None else float(cx),
        "cy": None if cy is None else float(cy),
        "min_projected_voxel_px": float(min_projected_voxel_px),
    }

    cameranum = int(cameranum)
    if cameranum < 1:
        raise ValueError("cameranum must be at least 1")

    if camera_models is None:
        camera_models = [{} for _ in range(cameranum)]
    else:
        if not isinstance(camera_models, list) or len(camera_models) == 0:
            raise ValueError("camera_models must be a non-empty list when provided")
        if len(camera_models) == 1 and cameranum > 1:
            camera_models = [dict(camera_models[0]) for _ in range(cameranum)]
        elif len(camera_models) != cameranum:
            raise ValueError(
                f"camera_models must contain exactly {cameranum} entries, got {len(camera_models)}"
            )

    normalized_models = []
    intrinsic_array = []
    for index, camera_model in enumerate(camera_models):
        normalized = _normalize_camera_model(camera_model, index, defaults)
        normalized_models.append(normalized)
        intrinsic_array.append(normalized["intrinsic"])

    intrinsic_array = np.asarray(intrinsic_array, dtype=float)

    global w, h, CAMERA_MODELS, CAMERA_INTRINSICS
    w = int(round(normalized_models[0]["image_width"]))
    h = int(round(normalized_models[0]["image_height"]))
    intrinsic[...] = intrinsic_array[0]
    CAMERA_MODELS = normalized_models
    CAMERA_INTRINSICS = intrinsic_array
    return get_camera_models_metadata()


def get_camera_models_metadata():
    return [
        {
            key: value
            for key, value in camera_model.items()
            if key != "intrinsic"
        }
        for camera_model in CAMERA_MODELS
    ]


def configure_camera_models_from_args(args):
    cameranum = getattr(args, "cameranum", None)
    if getattr(args, "camera_models", None) is not None:
        if cameranum is None:
            cameranum = len(args.camera_models)
        if len(args.camera_models) != int(cameranum):
            raise ValueError(
                f"camera_models contains {len(args.camera_models)} cameras but cameranum={cameranum}"
            )
    if cameranum is None:
        cameranum = 1

    models = configure_camera_models(
        cameranum=cameranum,
        camera_models=getattr(args, "camera_models", None),
        image_width=getattr(args, "image_width", 640),
        image_height=getattr(args, "image_height", 480),
        fx=getattr(args, "fx", None),
        fy=getattr(args, "fy", None),
        cx=getattr(args, "cx", None),
        cy=getattr(args, "cy", None),
        min_projected_voxel_px=getattr(args, "sampling_quality_min_projected_voxel_px", 1.5),
    )
    args.camera_models = models
    args.cameranum = len(models)
    args.image_width = int(round(models[0]["image_width"]))
    args.image_height = int(round(models[0]["image_height"]))
    args.fx = float(models[0]["fx"])
    args.fy = float(models[0]["fy"])
    args.cx = float(models[0]["cx"])
    args.cy = float(models[0]["cy"])
    return args


def get_camera_model(camera_index):
    return CAMERA_MODELS[int(camera_index)]


def get_camera_intrinsic(camera_index):
    return CAMERA_INTRINSICS[int(camera_index)]


def get_camera_intrinsics_array():
    return CAMERA_INTRINSICS.copy()


def get_camera_quality_thresholds():
    return np.asarray(
        [camera_model["min_projected_voxel_px"] for camera_model in CAMERA_MODELS],
        dtype=float,
    )


def project_camera_space_points(point_c, camera_model):
    point_c = np.asarray(point_c, dtype=float)
    z = point_c[:, 2]
    positive_depth = np.where(z > 1e-8)[0]
    if len(positive_depth) == 0:
        return positive_depth, np.zeros((0, 2), dtype=float), np.zeros(0, dtype=float)

    point_visible_depth = point_c[positive_depth]
    K = camera_model["intrinsic"]
    pixels_h = np.dot(K, point_visible_depth.T).T
    u = pixels_h[:, 0] / pixels_h[:, 2]
    v = pixels_h[:, 1] / pixels_h[:, 2]
    pixels = np.stack((u, v), axis=1)
    width = float(camera_model["image_width"])
    height = float(camera_model["image_height"])
    in_bounds = np.where(
        (u >= 0.0)
        & (u <= width - 1.0)
        & (v >= 0.0)
        & (v <= height - 1.0)
    )[0]
    valid_indices = positive_depth[in_bounds]
    return valid_indices, pixels[in_bounds], point_visible_depth[in_bounds, 2]
###########calculate rotation matrix from parameter 6D
# batch*n
def normalize_vector( v, return_mag =False):
    batch=v.shape[0]
    v_mag = torch.sqrt(v.pow(2).sum(1))# batch
    v_mag = torch.max(v_mag, torch.autograd.Variable(torch.FloatTensor([1e-8]).to(device)))
    v_mag = v_mag.view(batch,1).expand(batch,v.shape[1])
    v = v/v_mag
    if(return_mag==True):
        return v, v_mag[:,0]
    else:
        return v
# u, v batch*n
def cross_product( u, v):
    batch = u.shape[0]
    i = u[:,1]*v[:,2] - u[:,2]*v[:,1]
    j = u[:,2]*v[:,0] - u[:,0]*v[:,2]
    k = u[:,0]*v[:,1] - u[:,1]*v[:,0]
        
    out = torch.cat((i.view(batch,1), j.view(batch,1), k.view(batch,1)),1)#batch*3
        
    return out
#poses batch*6
#poses
def compute_rotation_matrix_from_ortho6d(ortho6d):
    x_raw = ortho6d[:,0:3]#batch*3
    z_raw = ortho6d[:,3:6]#batch*3
        
    z = normalize_vector(z_raw) #batch*3
    y = cross_product(z,x_raw) #batch*3
    y = normalize_vector(y)#batch*3
    x = cross_product(y,z)#batch*3
        
    x = x.view(-1,1,3)
    y = y.view(-1,1,3)
    z = z.view(-1,1,3)
    matrix = torch.cat((x,y,z), 1) #batch*3*3
    return matrix.to(torch.float)
####rotationTransform################
def RotationToEuler(Rmnumpy):
    euler=np.zeros(shape=(len(Rmnumpy),3))
    for i in range(len(Rmnumpy)):
        r=R.from_matrix(Rmnumpy[i])
        euler[i]=r.as_euler('xyz',degrees=True)
    return euler
def EulerToRotaionMatrix(euler):
    rotate=np.zeros(shape=(len(euler),3,3))
    for i in range(len(euler)):
        r=R.from_euler('zyx',euler[i],degrees=True)
        rotate[i]=r.as_matrix()
    return rotate
#######################numpy to tensor in device#########################
def npToTensor(matrix,dtype=torch.float):
	return torch.tensor(matrix,dtype=dtype,requires_grad=False).to(device)
def saveTrainingResult(path,position,rotation,scale=1):
    rotate=rotation.reshape(-1,9)
    cameraposedata=torch.cat((position,rotate),1)
    cameraposedata=cameraposedata.detach().cpu().numpy()
    cameraposedata[:,:3]=cameraposedata[:,:3]*scale[0]+scale[1]
    np.save(path,cameraposedata)
