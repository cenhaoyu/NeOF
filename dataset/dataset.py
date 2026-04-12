import os
import sys
sys.path.append(os.path.dirname(sys.path[0]))
from .init_camera import *
from .occupancy_map import build_occupancy_support
from .pcd import *

def readCamerapose(path):
  if os.path.exists(path):
    camerapose=np.load(path)
    print(camerapose.shape)
    position=camerapose[:,:3]
    rotation=camerapose[:,3:].reshape(-1,3,3)
    return position,rotation

def Initdatafromrandom(args):
    pointspath=args.modelname
    if pointspath.startswith("data/") == False:
        pointspath="data/"+pointspath
    if args.voxelnum is not None:
        pcd,voxelnormals,minbound,size,geometry_info=getPointNormalfromPly(
            pointspath,
            args.voxelnum,
            args.voxelsize,
            args.kcoverage,
            target_height=getattr(args, "model_physical_height", None),
        )
    else :
        voxelnormals=getPointNormalfromFile(pointspath)
        pcd = None
        minbound = np.min(voxelnormals[:, :3], axis=0)
        size = [1.0, np.zeros(3)]
        geometry_info = {
            "source_height": float(np.max(voxelnormals[:, 2]) - np.min(voxelnormals[:, 2])),
            "world_scale_from_source": 1.0,
            "model_world_height": float(np.max(voxelnormals[:, 2]) - np.min(voxelnormals[:, 2])),
            "source_center": np.zeros(3, dtype=float),
        }
    raw_voxelnormals = np.asarray(voxelnormals, dtype=float)
    if len(raw_voxelnormals) == 0:
        raise ValueError("Loaded geometry does not contain any support voxels")
    model_world_height = float(np.max(raw_voxelnormals[:, 2]) - np.min(raw_voxelnormals[:, 2]))
    if model_world_height <= 1e-8:
        raise ValueError("Loaded geometry has near-zero height after world scaling")
    model_physical_height = float(geometry_info["model_world_height"])
    voxelnormals = raw_voxelnormals.astype(np.float32)
    minbound = np.asarray(minbound, dtype=float)

    geometry_metadata = {
        "model_physical_height": float(model_physical_height),
        "world_scale": float(geometry_info["world_scale_from_source"]),
        "world_coordinate_system": True,
    }
    voxelnormals, occupancy_weights, occupancy_info = build_occupancy_support(args, voxelnormals, scale=None)
    if args.camera_init_strategy == "grid":
        Rs,Cs=initVolumeGridCameras(
            args.cameranum,
            voxelnormals,
            args.camera_constraint_min,
            args.camera_constraint_max,
            args.camera_constraint_shape,
            args.camera_constraint_data,
        )
    elif args.camera_init_strategy == "random":
        Rs,Cs=initVolumeRandomCameras(
            args.cameranum,
            voxelnormals,
            args.camera_constraint_min,
            args.camera_constraint_max,
            args.camera_constraint_shape,
            args.camera_constraint_data,
        )
    else:
        raise ValueError(f"Unsupported camera_init_strategy: {args.camera_init_strategy}")
    camerapose=getCameraPose(Rs,Cs)
    return (
        pcd,
        voxelnormals,
        occupancy_weights,
        occupancy_info,
        minbound,
        npToTensor(camerapose,dtype=torch.float),
        [1.0, np.zeros(3, dtype=float)],
        geometry_metadata,
    )

def VisReadData(camerapath,pointspath,voxelnum=None):
    if voxelnum is not None:
        pcd,voxelnormals,_,size,_=getPointNormalfromPly(pointspath,voxelnum,0.02,1)
    else :
        voxelnormals=getPointNormalfromFile(pointspath)
        pcd = None
        size = [1.0, np.zeros(3)]
    Rs,Cs=readCamerapose(camerapath)
    return pcd,npToTensor(voxelnormals),Cs,Rs,size

def generateP(num):
    Cmin=np.array([-1,-1])
    Cmax=np.array([1,1])
    points=np.random.random((num,2))*(Cmax-Cmin)+Cmin
    ones=np.ones((num,1))*0.8
    points=np.append(points,ones,1)
    normals=np.zeros_like(points)
    normals[:,2]=1
    pointnormals=np.append(points,normals,1) 
    print(pointnormals)
    return pointnormals
