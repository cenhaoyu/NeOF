import argparse
import json
import os
import tempfile
import zipfile

import open3d as o3d


def resolve_path(base_dir, relative_path):
    if relative_path is None:
        return None
    return os.path.join(base_dir, relative_path)


def find_manifest_in_directory(base_dir):
    candidates = sorted(
        filename for filename in os.listdir(base_dir) if filename.endswith("_scene.json")
    )
    if not candidates:
        raise FileNotFoundError(f"No *_scene.json found in extracted archive: {base_dir}")
    return os.path.join(base_dir, candidates[0])


def load_scene(scene_manifest_path):
    with open(scene_manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    base_dir = os.path.dirname(os.path.abspath(scene_manifest_path))
    geometry_path = resolve_path(base_dir, manifest.get("geometry_file"))
    overlay_path = resolve_path(base_dir, manifest.get("overlay_file"))
    occupancy_path = resolve_path(base_dir, manifest.get("occupancy_file"))
    view_path = resolve_path(base_dir, manifest.get("view_file"))

    geometry_type = manifest.get("geometry_type")
    if geometry_type == "none":
        geometry = None
    elif geometry_type == "point_cloud":
        geometry = o3d.io.read_point_cloud(geometry_path)
    elif geometry_type == "triangle_mesh":
        geometry = o3d.io.read_triangle_mesh(geometry_path, True)
        if not geometry.has_vertex_normals():
            geometry.compute_vertex_normals()
    else:
        raise ValueError(f"Unsupported geometry_type in manifest: {geometry_type}")

    geometries = []
    if geometry is not None:
        geometries.append(geometry)
    if overlay_path is not None and os.path.exists(overlay_path):
        overlay = o3d.io.read_triangle_mesh(overlay_path, True)
        if not overlay.has_vertex_normals():
            overlay.compute_vertex_normals()
        geometries.append(overlay)
    if occupancy_path is not None and os.path.exists(occupancy_path):
        occupancy = o3d.io.read_point_cloud(occupancy_path)
        geometries.append(occupancy)

    return geometries, view_path


def show_scene_from_path(scene_path, width=1280, height=960):
    if scene_path.endswith(".zip"):
        with tempfile.TemporaryDirectory(prefix="saved_scene_") as temp_dir:
            with zipfile.ZipFile(scene_path, "r") as zip_file:
                zip_file.extractall(temp_dir)
            manifest_path = find_manifest_in_directory(temp_dir)
            geometries, view_path = load_scene(manifest_path)
            show_scene(geometries, view_path=view_path, width=width, height=height)
        return

    geometries, view_path = load_scene(scene_path)
    show_scene(geometries, view_path=view_path, width=width, height=height)


def show_scene(geometries, view_path=None, width=1280, height=960):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="saved_scene", width=width, height=height)
    for geometry in geometries:
        vis.add_geometry(geometry)
    vis.reset_view_point(True)

    if view_path is not None and os.path.exists(view_path):
        view_parameters = o3d.io.read_pinhole_camera_parameters(view_path)
        view_control = vis.get_view_control()
        try:
            view_control.convert_from_pinhole_camera_parameters(view_parameters, allow_arbitrary=True)
        except TypeError:
            view_control.convert_from_pinhole_camera_parameters(view_parameters)

    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=str, required=True, help="Path to *_scene.json or *_scene.zip exported by visMesh(save mode).")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=960)
    args = parser.parse_args()

    show_scene_from_path(args.scene, width=args.width, height=args.height)
