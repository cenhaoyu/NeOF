from dataset.utils import *
import itertools
import open3d as o3d
# generate standard pointnormals matrix(n,6) from file
def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    print(centroid,m)
    return pc,m,centroid
def getPointNormalfromFile(csvfile):
    pointnormals=np.loadtxt(csvfile,dtype=float,delimiter=',')
    return pointnormals


def _ensure_point_normals(point_cloud):
    if len(point_cloud.normals) == len(point_cloud.points):
        return
    point_cloud.estimate_normals()
    point_cloud.normalize_normals()


def getPointNormalfromPly(path,num,voxelsize,kcoverage,target_height=None):
    ending=path.split(".")[-1]
    if ending=="obj":
        mesh = o3d.io.read_triangle_mesh(path,True)
        mesh.compute_vertex_normals()
        pcd=o3d.geometry.TriangleMesh.sample_points_uniformly(mesh, number_of_points=num)
    elif ending=="ply":
        pcd = o3d.io.read_point_cloud(path)
        mesh = None
    else:
        raise ValueError(f"Unsupported geometry format: {ending}")

    _ensure_point_normals(pcd)

    raw_points = np.asarray(pcd.points, dtype=float)
    if len(raw_points) == 0:
        raise ValueError(f"Loaded geometry contains no points: {path}")
    raw_maxbound = np.max(raw_points, axis=0)
    raw_minbound = np.min(raw_points, axis=0)
    raw_center = 0.5 * (raw_maxbound + raw_minbound)
    raw_size = raw_maxbound - raw_minbound
    raw_height = float(raw_size[2])
    if raw_height <= 1e-8:
        raise ValueError("Loaded geometry has near-zero height; cannot derive world scaling")
    world_scale = 1.0 if target_height is None else float(target_height) / raw_height
    if world_scale <= 0:
        raise ValueError("target_height must be positive when provided")

    points = (raw_points - raw_center[None, :]) * world_scale
    pcd.points=o3d.utility.Vector3dVector(points)
    if ending=='obj':
        mesh.vertices=o3d.utility.Vector3dVector((np.asarray(mesh.vertices, dtype=float)-raw_center[None, :]) * world_scale)
    scale=[1.0,np.zeros(3, dtype=float)]
    geometry_info = {
        "source_height": raw_height,
        "world_scale_from_source": world_scale,
        "model_world_height": raw_height * world_scale,
        "source_center": raw_center,
    }
    #generatevoxelgrid
    voxelgrid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd,voxel_size=voxelsize)
    boundbox=np.asarray(voxelgrid.get_axis_aligned_bounding_box().get_box_points())
    minbound=np.min(boundbox,axis=0)
    maxbound=np.max(boundbox,axis=0)
    minvoxel=voxelgrid.get_voxel(minbound)
    maxvoxel=voxelgrid.get_voxel(maxbound)
    voxelnum=maxvoxel-minvoxel
    voxelnum+=1
    voxelsurface=np.zeros(shape=(voxelnum[0],voxelnum[1],voxelnum[2],1),dtype=int)
    point_index=np.floor((points-minbound)//voxelsize).astype(int)
    voxelsurface[point_index[:,0],point_index[:,1],point_index[:,2]]=kcoverage
    index=np.asarray(list(itertools.product(np.arange(voxelnum[0]),np.arange(voxelnum[1]),np.arange(voxelnum[2]))))
    judge_surface=np.where(voxelsurface[index[:,0],index[:,1],index[:,2]]!=0)[0]
    voxelindex=index[judge_surface]
    ###voxelposition
    voxelcenter_surface=(voxelindex+0.5)*voxelsize+minbound
    ###voxelnormal
    normals=np.asarray(pcd.normals)
    voxelnormals_surface = np.zeros_like(voxelindex,dtype=float)
    for i in range(len(voxelindex)):
        voxelnormals_surface[i]=np.mean(normals[np.where((point_index == voxelindex[i]).all(1))[0]],axis=0)

    voxelnormals = np.append(voxelcenter_surface,voxelnormals_surface,axis=1)
    voxelnormals.astype(float)
    return mesh if ending=="obj" else pcd,voxelnormals,minbound,scale,geometry_info
