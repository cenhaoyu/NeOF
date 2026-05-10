import json
import os
import zipfile
import numpy as np 
import open3d as o3d
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from camera_constraints import camera_volume_center, camera_volume_height, camera_volume_xy_radii
from dataset.utils import get_camera_models_metadata
from dataset.init_camera import *
import matplotlib.pyplot as plt


_STROKE_GLYPHS = {
    "0": [((0.0, 0.0), (1.0, 0.0)), ((1.0, 0.0), (1.0, 2.0)), ((1.0, 2.0), (0.0, 2.0)), ((0.0, 2.0), (0.0, 0.0))],
    "1": [((0.5, 0.0), (0.5, 2.0)), ((0.2, 1.7), (0.5, 2.0))],
    "2": [((0.0, 2.0), (1.0, 2.0)), ((1.0, 2.0), (1.0, 1.0)), ((1.0, 1.0), (0.0, 1.0)), ((0.0, 1.0), (0.0, 0.0)), ((0.0, 0.0), (1.0, 0.0))],
    "3": [((0.0, 2.0), (1.0, 2.0)), ((1.0, 2.0), (1.0, 0.0)), ((0.0, 1.0), (1.0, 1.0)), ((0.0, 0.0), (1.0, 0.0))],
    "4": [((0.0, 2.0), (0.0, 1.0)), ((0.0, 1.0), (1.0, 1.0)), ((1.0, 2.0), (1.0, 0.0))],
    "5": [((1.0, 2.0), (0.0, 2.0)), ((0.0, 2.0), (0.0, 1.0)), ((0.0, 1.0), (1.0, 1.0)), ((1.0, 1.0), (1.0, 0.0)), ((1.0, 0.0), (0.0, 0.0))],
    "6": [((1.0, 2.0), (0.0, 2.0)), ((0.0, 2.0), (0.0, 0.0)), ((0.0, 1.0), (1.0, 1.0)), ((1.0, 1.0), (1.0, 0.0)), ((1.0, 0.0), (0.0, 0.0))],
    "7": [((0.0, 2.0), (1.0, 2.0)), ((1.0, 2.0), (0.4, 0.0))],
    "8": [((0.0, 0.0), (1.0, 0.0)), ((1.0, 0.0), (1.0, 2.0)), ((1.0, 2.0), (0.0, 2.0)), ((0.0, 2.0), (0.0, 0.0)), ((0.0, 1.0), (1.0, 1.0))],
    "9": [((1.0, 0.0), (1.0, 2.0)), ((1.0, 2.0), (0.0, 2.0)), ((0.0, 2.0), (0.0, 1.0)), ((0.0, 1.0), (1.0, 1.0)), ((0.0, 0.0), (1.0, 0.0))],
    "-": [((0.1, 1.0), (0.9, 1.0))],
    ".": [((0.45, 0.15), (0.55, 0.15))],
    "X": [((0.0, 0.0), (1.0, 2.0)), ((0.0, 2.0), (1.0, 0.0))],
    "Y": [((0.0, 2.0), (0.5, 1.1)), ((1.0, 2.0), (0.5, 1.1)), ((0.5, 1.1), (0.5, 0.0))],
    "Z": [((0.0, 2.0), (1.0, 2.0)), ((1.0, 2.0), (0.0, 0.0)), ((0.0, 0.0), (1.0, 0.0))],
}

_STROKE_GLYPH_WIDTH = {
    ".": 0.35,
    "-": 0.9,
    " ": 0.5,
}
def align_vector_to_another(a=np.array([0, 0, 1]), b=np.array([1, 0, 0])):
    """
    Aligns vector a to vector b with axis angle rotation
    """
    if np.array_equal(a, b):
        return None, None
    axis_ = np.cross(a, b)
    axis_norm = np.linalg.norm(axis_)
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    if axis_norm <= 1e-12:
        if dot > 0.0:
            return None, None
        fallback = np.array([1.0, 0.0, 0.0], dtype=float)
        if abs(a[0]) > 0.9:
            fallback = np.array([0.0, 1.0, 0.0], dtype=float)
        axis_ = np.cross(a, fallback)
        axis_norm = np.linalg.norm(axis_)
        if axis_norm <= 1e-12:
            return None, None
    axis_ = axis_ / axis_norm
    angle = np.arccos(dot)

    return axis_, angle

def normalized(a, axis=-1, order=2):
    """Normalizes a numpy array of points"""
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis), l2

class LineMesh(object):
    def __init__(self, points, lines=None, colors=[0,1,0], radius=0.15):
        """Creates a line represented as sequence of cylinder triangular meshes

        Arguments:
            points {ndarray} -- Numpy array of ponts Nx3.

        Keyword Arguments:
            lines {list[list] or None} -- List of point index pairs denoting line segments. If None, implicit lines from ordered pairwise points. (default: {None})
            colors {list} -- list of colors, or single color of the line (default: {[0, 1, 0]})
            radius {float} -- radius of cylinder (default: {0.15})
        """
        self.points = np.array(points)
        self.lines = np.array(lines) if lines is not None else self.lines_from_ordered_points(self.points)
        # print(self.lines,self.lines.shape)
        self.colors = np.array(colors)
        self.radius = radius
        self.cylinder_segments = []

        self.create_line_mesh()

    @staticmethod
    def lines_from_ordered_points(points):
        lines = [[i, i + 1] for i in range(0, points.shape[0] - 1, 1)]
        return np.array(lines)

    def create_line_mesh(self):
        first_points = self.points[self.lines[:, 0], :]
        second_points = self.points[self.lines[:, 1], :]
        line_segments = second_points - first_points
        line_segments_unit, line_lengths = normalized(line_segments)

        z_axis = np.array([0, 0, 1])
        for i in range(line_segments_unit.shape[0]):
            line_segment = line_segments_unit[i, :]
            line_length = line_lengths[i]
            # get axis angle rotation to allign cylinder with line segment
            axis, angle = align_vector_to_another(z_axis, line_segment)
            # Get translation vector
            translation = first_points[i, :] + line_segment * line_length * 0.5
            # create cylinder and apply transformations
            cylinder_segment = o3d.geometry.TriangleMesh.create_cylinder(
                self.radius, line_length)
            cylinder_segment = cylinder_segment.translate(
                translation, relative=False)
            if axis is not None:
                axis_a = axis * angle
                cylinder_segment = cylinder_segment.rotate(
                    R=o3d.geometry.get_rotation_matrix_from_axis_angle(axis_a))
            cylinder_segment.paint_uniform_color(self.colors[0,:])
            self.cylinder_segments.append(cylinder_segment)

    def add_line(self, vis):
        """Adds this line to the visualizer"""
        for cylinder in self.cylinder_segments:
            vis.add_geometry(cylinder)

    def remove_line(self, vis):
        """Removes this line from the visualizer"""
        for cylinder in self.cylinder_segments:
            vis.remove_geometry(cylinder)


def _build_line_geometries(points, connection, color, radius):
    tracking_mesh = LineMesh(
        np.asarray(points, dtype=float),
        np.asarray(connection, dtype=np.int32),
        np.asarray(color, dtype=float).reshape(1, 3),
        radius=float(radius),
    )
    return tracking_mesh.cylinder_segments


def _camera_local_to_world(local_points, rotation, position):
    return (rotation.T @ local_points.T).T + position.reshape(1, 3)


def _resolve_camera_models_for_visualization(camera_models, camera_count):
    if camera_models is None:
        camera_models = get_camera_models_metadata()
        if len(camera_models) == camera_count:
            return camera_models
        if camera_count == 1 and len(camera_models) > 0:
            return [camera_models[0]]
        raise ValueError(
            f"camera model count mismatch for visualization: have {len(camera_models)} models but {camera_count} poses"
        )

    if not isinstance(camera_models, list) or len(camera_models) == 0:
        raise ValueError("camera_models must be a non-empty list when provided")
    if len(camera_models) == 1 and camera_count > 1:
        return [dict(camera_models[0]) for _ in range(camera_count)]
    if len(camera_models) != camera_count:
        raise ValueError(
            f"camera model count mismatch for visualization: have {len(camera_models)} models but {camera_count} poses"
        )
    return camera_models


def _camera_frustum_points_at_depth(camera_model, frustum_depth):
    frustum_depth = max(float(frustum_depth), 1e-6)
    width = float(camera_model["image_width"])
    height = float(camera_model["image_height"])
    fx = float(camera_model["fx"])
    fy = float(camera_model["fy"])
    cx = float(camera_model["cx"])
    cy = float(camera_model["cy"])

    left = -cx / fx * frustum_depth
    right = (width - 1.0 - cx) / fx * frustum_depth
    top = -cy / fy * frustum_depth
    bottom = (height - 1.0 - cy) / fy * frustum_depth

    return np.array(
        [
            [0.0, 0.0, 0.0],
            [left, top, frustum_depth],
            [right, top, frustum_depth],
            [right, bottom, frustum_depth],
            [left, bottom, frustum_depth],
        ],
        dtype=float,
    )


def _camera_frustum_local_points(camera_model, camera_scale):
    return _camera_frustum_points_at_depth(camera_model, float(camera_scale) * 0.2)


def resolve_camera_ray_length(mesh, camerapose, constraint_min=None, constraint_max=None, ray_length=None):
    if ray_length is not None:
        ray_length = float(ray_length)
        if not np.isfinite(ray_length) or ray_length <= 0:
            raise ValueError("camera ray length must be finite and positive when provided")
        return ray_length

    axis_length = resolve_world_axis_length(
        mesh,
        camerapose,
        constraint_min=constraint_min,
        constraint_max=constraint_max,
        axis_length=None,
    )
    return max(1.0, 2.2 * float(axis_length))


def build_camera_geometries(
    camerapose,
    camerarotation,
    color=None,
    camera_scale=0.2,
    camera_models=None,
    show_optical_axis=True,
    optical_axis_length=None,
    show_fov=False,
    line_radius=None,
    optical_axis_color=None,
):
    camerapose = np.asarray(camerapose, dtype=float)
    camerarotation = np.asarray(camerarotation, dtype=float)
    if color is None:
        color = np.array([[165 / 255, 42 / 255, 42 / 255]])
    else:
        color = np.asarray(color, dtype=float)
        if color.ndim == 1:
            color = color.reshape(1, 3)
        elif color.ndim != 2 or color.shape[1] != 3:
            raise ValueError("color must have shape (3,) or (N, 3)")
        if color.shape[0] not in (1, len(camerapose)):
            raise ValueError("camera color count must be 1 or match the number of cameras")
    if camerapose.ndim != 2 or camerapose.shape[1] != 3:
        raise ValueError("camerapose must have shape (N, 3)")
    if camerarotation.ndim != 3 or camerarotation.shape[1:] != (3, 3):
        raise ValueError("camerarotation must have shape (N, 3, 3)")
    if len(camerapose) != len(camerarotation):
        raise ValueError("camerapose and camerarotation must contain the same number of cameras")

    camera_models = _resolve_camera_models_for_visualization(camera_models, len(camerapose))
    point_connection = np.array([[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [3, 4], [4, 1]], dtype=np.int32)
    axis_connection = np.array([[0, 1]], dtype=np.int32)
    line_radius = max(float(camera_scale) * 0.005, 0.001) if line_radius is None else float(line_radius)
    if line_radius <= 0:
        raise ValueError("line_radius must be positive")
    optical_axis_color = (
        np.array([1.0, 0.86, 0.15], dtype=float)
        if optical_axis_color is None
        else np.asarray(optical_axis_color, dtype=float)
    )

    geometries = []
    for idx in range(len(camerapose)):
        current_color = color[0] if color.shape[0] == 1 else color[idx]
        camera_model = camera_models[idx]
        local_frustum = _camera_frustum_local_points(camera_model, camera_scale)
        world_frustum = _camera_local_to_world(local_frustum, camerarotation[idx], camerapose[idx])
        geometries.extend(
            _build_line_geometries(
                world_frustum,
                point_connection,
                current_color,
                radius=line_radius,
            )
        )

        if show_fov and optical_axis_length is not None and optical_axis_length > local_frustum[1, 2] * 1.05:
            local_fov = _camera_frustum_points_at_depth(camera_model, optical_axis_length)
            world_fov = _camera_local_to_world(local_fov, camerarotation[idx], camerapose[idx])
            geometries.extend(
                _build_line_geometries(
                    world_fov,
                    point_connection,
                    current_color,
                    radius=max(0.9 * line_radius, 1e-6),
                )
            )

        if show_optical_axis:
            frustum_depth = local_frustum[1, 2]
            axis_length = float(optical_axis_length) if optical_axis_length is not None else frustum_depth * 2.5
            local_axis = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, axis_length]], dtype=float)
            world_axis = _camera_local_to_world(local_axis, camerarotation[idx], camerapose[idx])
            geometries.extend(
                _build_line_geometries(
                    world_axis,
                    axis_connection,
                    optical_axis_color,
                    radius=line_radius,
                )
            )
    return geometries


def getCameraTransitionVis(start_positions, end_positions, color=None, radius=0.0015):
    start_positions = np.asarray(start_positions, dtype=float)
    end_positions = np.asarray(end_positions, dtype=float)
    if start_positions.shape != end_positions.shape:
        raise ValueError("start_positions and end_positions must have the same shape")
    if color is None:
        color = np.array([[1.0, 0.85, 0.1]])
    else:
        color = np.asarray(color, dtype=float).reshape(1, 3)

    geometries = []
    for start, end in zip(start_positions, end_positions):
        if np.linalg.norm(end - start) <= 1e-8:
            continue
        geometries.extend(_line_segment_mesh(start, end, color=color[0], radius=radius))
    return geometries


def getBoxVis(box_min, box_max, color=None, radius=0.003):
    box_min = np.asarray(box_min, dtype=float)
    box_max = np.asarray(box_max, dtype=float)
    if color is None:
        color = np.array([[1.0, 0.55, 0.0]])
    else:
        color = np.asarray(color, dtype=float).reshape(1, 3)

    corners = np.array([
        [box_min[0], box_min[1], box_min[2]],
        [box_max[0], box_min[1], box_min[2]],
        [box_max[0], box_max[1], box_min[2]],
        [box_min[0], box_max[1], box_min[2]],
        [box_min[0], box_min[1], box_max[2]],
        [box_max[0], box_min[1], box_max[2]],
        [box_max[0], box_max[1], box_max[2]],
        [box_min[0], box_max[1], box_max[2]],
    ])
    edges = np.array([
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7],
    ])
    box_mesh = LineMesh(corners, edges.astype(np.int32), color, radius=radius)
    return box_mesh.cylinder_segments


def getPlaneVis(plane_center, plane_span_u, plane_span_v, color=None, radius=0.003):
    plane_center = np.asarray(plane_center, dtype=float)
    plane_span_u = np.asarray(plane_span_u, dtype=float)
    plane_span_v = np.asarray(plane_span_v, dtype=float)
    if color is None:
        color = np.array([[1.0, 0.55, 0.0]])
    else:
        color = np.asarray(color, dtype=float).reshape(1, 3)

    corners = np.array(
        [
            plane_center - plane_span_u - plane_span_v,
            plane_center + plane_span_u - plane_span_v,
            plane_center + plane_span_u + plane_span_v,
            plane_center - plane_span_u + plane_span_v,
        ],
        dtype=float,
    )
    edges = np.array([[0, 1], [1, 2], [2, 3], [3, 0], [0, 2], [1, 3]], dtype=np.int32)
    plane_mesh = LineMesh(corners, edges, color, radius=radius)
    return plane_mesh.cylinder_segments


def _polyline_segments(points, color, radius, closed=False):
    points = np.asarray(points, dtype=float)
    lines = [[i, i + 1] for i in range(len(points) - 1)]
    if closed and len(points) > 2:
        lines.append([len(points) - 1, 0])
    line_mesh = LineMesh(points, np.asarray(lines, dtype=np.int32), color, radius=radius)
    return line_mesh.cylinder_segments


def getCylinderVis(box_min, box_max, color=None, radius=0.003, ring_points=48, meridians=8):
    box_min = np.asarray(box_min, dtype=float)
    box_max = np.asarray(box_max, dtype=float)
    if color is None:
        color = np.array([[1.0, 0.55, 0.0]])
    else:
        color = np.asarray(color, dtype=float).reshape(1, 3)

    center = camera_volume_center(box_min, box_max)
    rx, ry = camera_volume_xy_radii(box_min, box_max)
    theta = np.linspace(0.0, 2.0 * np.pi, ring_points, endpoint=False)
    bottom_ring = np.stack(
        (center[0] + rx * np.cos(theta), center[1] + ry * np.sin(theta), np.full(ring_points, box_min[2])),
        axis=1,
    )
    top_ring = bottom_ring.copy()
    top_ring[:, 2] = box_max[2]

    geometries = []
    geometries.extend(_polyline_segments(bottom_ring, color, radius, closed=True))
    geometries.extend(_polyline_segments(top_ring, color, radius, closed=True))

    vertical_ids = np.linspace(0, ring_points - 1, meridians, dtype=int)
    for idx in vertical_ids:
        vertical = np.stack((bottom_ring[idx], top_ring[idx]), axis=0)
        geometries.extend(_polyline_segments(vertical, color, radius, closed=False))
    return geometries


def getDomeVis(box_min, box_max, color=None, radius=0.003, ring_points=48, meridians=8, latitudes=5):
    box_min = np.asarray(box_min, dtype=float)
    box_max = np.asarray(box_max, dtype=float)
    if color is None:
        color = np.array([[1.0, 0.55, 0.0]])
    else:
        color = np.asarray(color, dtype=float).reshape(1, 3)

    center = camera_volume_center(box_min, box_max)
    rx, ry = camera_volume_xy_radii(box_min, box_max)
    rz = camera_volume_height(box_min, box_max)
    theta = np.linspace(0.0, 2.0 * np.pi, ring_points, endpoint=False)

    base_ring = np.stack(
        (center[0] + rx * np.cos(theta), center[1] + ry * np.sin(theta), np.full(ring_points, box_min[2])),
        axis=1,
    )
    geometries = []
    geometries.extend(_polyline_segments(base_ring, color, radius, closed=True))

    elevation_values = np.linspace(np.pi / 10.0, np.pi / 2.0 - np.pi / 16.0, latitudes)
    for elevation in elevation_values:
        ring = np.stack(
            (
                center[0] + rx * np.cos(theta) * np.cos(elevation),
                center[1] + ry * np.sin(theta) * np.cos(elevation),
                np.full(ring_points, box_min[2] + rz * np.sin(elevation)),
            ),
            axis=1,
        )
        geometries.extend(_polyline_segments(ring, color, radius, closed=True))

    meridian_theta = np.linspace(0.0, 2.0 * np.pi, meridians, endpoint=False)
    elevation_line = np.linspace(0.0, np.pi / 2.0, 32)
    for phi in meridian_theta:
        meridian = np.stack(
            (
                center[0] + rx * np.cos(elevation_line) * np.cos(phi),
                center[1] + ry * np.cos(elevation_line) * np.sin(phi),
                box_min[2] + rz * np.sin(elevation_line),
            ),
            axis=1,
        )
        geometries.extend(_polyline_segments(meridian, color, radius, closed=False))
    return geometries


def getCameraConstraintVis(camera_constraint_shape, constraint_min, constraint_max, constraint_data=None, color=None, radius=0.003):
    if camera_constraint_shape == "box":
        return getBoxVis(constraint_min, constraint_max, color=color, radius=radius)
    if camera_constraint_shape in ("box_surface", "box_walls"):
        return getBoxVis(constraint_min, constraint_max, color=color, radius=radius)
    if camera_constraint_shape == "cylinder":
        return getCylinderVis(constraint_min, constraint_max, color=color, radius=radius)
    if camera_constraint_shape == "dome":
        return getDomeVis(constraint_min, constraint_max, color=color, radius=radius)
    if camera_constraint_shape == "plane":
        if constraint_data is None:
            raise ValueError("plane visualization requires constraint_data")
        return getPlaneVis(
            constraint_data["plane_center"],
            constraint_data["plane_span_u"],
            constraint_data["plane_span_v"],
            color=color,
            radius=radius,
        )
    if camera_constraint_shape == "cylinder_surface":
        return getCylinderVis(constraint_min, constraint_max, color=color, radius=radius)
    if camera_constraint_shape == "dome_surface":
        return getDomeVis(constraint_min, constraint_max, color=color, radius=radius)
    raise ValueError(f"Unsupported camera constraint shape: {camera_constraint_shape}")


def _normalize_occupancy_weights_for_vis(weights, clip_percentile):
    weights = np.asarray(weights, dtype=float)
    if len(weights) == 0:
        return weights
    weights = np.maximum(weights, 0.0)
    lower = float(np.min(weights))
    upper = float(np.percentile(weights, clip_percentile))
    if upper <= lower + 1e-12:
        upper = float(np.max(weights))
    if upper <= lower + 1e-12:
        return np.zeros_like(weights)
    clipped = np.clip(weights, lower, upper)
    return (clipped - lower) / (upper - lower)


def getOccupancyPointCloud(points, weights, clip_percentile=95.0, colormap="plasma"):
    points = np.asarray(points, dtype=float)
    weights = np.asarray(weights, dtype=float)
    normalized = _normalize_occupancy_weights_for_vis(weights, clip_percentile)
    colors = plt.get_cmap(colormap)(normalized)[:, :3]
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)
    return point_cloud


def saveOccupancyColorLegend(output_path, clip_percentile=95.0, colormap="plasma", title="Occupancy Weight"):
    figure, axis = plt.subplots(figsize=(2.4, 4.0), dpi=200)
    sm = plt.cm.ScalarMappable(cmap=plt.get_cmap(colormap), norm=plt.Normalize(vmin=0.0, vmax=1.0))
    sm.set_array([])
    colorbar = figure.colorbar(sm, cax=axis)
    colorbar.set_ticks([0.0, 0.5, 1.0])
    colorbar.set_ticklabels(["Low", "Medium", "High"])
    colorbar.ax.tick_params(labelsize=9)
    axis.set_title(title, fontsize=10, pad=10)
    figure.text(
        0.5,
        0.04,
        f"Colors are normalized and clipped at the {clip_percentile:.0f}th percentile",
        ha="center",
        va="bottom",
        fontsize=8,
    )
    figure.tight_layout(rect=[0.0, 0.08, 1.0, 1.0])
    figure.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def saveWorldAxisLegend(
    output_path,
    axis_length,
    tick_step,
    label_stride=1,
    title="World Axes (meters)",
):
    canvas_width = 620
    canvas_height = 320
    canvas = Image.new("RGB", (canvas_width, canvas_height), color=(250, 250, 250))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    title_bbox = font.getbbox(title)
    title_width = title_bbox[2] - title_bbox[0]
    draw.text(((canvas_width - title_width) / 2, 20), title, font=font, fill=(30, 30, 30))

    axis_items = [
        ("X axis", (255, 80, 80)),
        ("Y axis", (60, 185, 90)),
        ("Z axis", (60, 110, 255)),
    ]
    current_y = 72
    for label, color in axis_items:
        draw.line((58, current_y + 7, 118, current_y + 7), fill=color, width=5)
        draw.text((138, current_y), label, font=font, fill=(35, 35, 35))
        current_y += 30

    lines = [
        "Tick labels show world coordinates in meters (m).",
        f"Tick spacing: {tick_step:g} m",
    ]
    if int(label_stride) > 0:
        effective_label_step = tick_step * int(label_stride)
        lines.append(f"Displayed label interval: {effective_label_step:g} m")
    else:
        lines.append("Tick labels: disabled")
    lines.append(f"Axis span: -{axis_length:g} m to +{axis_length:g} m")
    current_y = 170
    for line in lines:
        draw.text((40, current_y), line, font=font, fill=(45, 45, 45))
        current_y += 30

    canvas.save(output_path)
    return output_path


def getOccupancyRouteVis(
    route_points,
    color=None,
    radius=0.004,
    show_direction=False,
    arrow_radius=None,
    arrow_length=None,
):
    route_points = np.asarray(route_points, dtype=float)
    if route_points.ndim != 2 or route_points.shape[1] != 3 or len(route_points) < 2:
        return []
    if color is None:
        color = np.array([[0.0, 0.9, 0.9]])
    else:
        color = np.asarray(color, dtype=float).reshape(1, 3)
    geometries = _polyline_segments(route_points, color, radius, closed=False)
    if not show_direction:
        return geometries

    if arrow_radius is None or arrow_radius <= 0:
        arrow_radius = 1.25 * radius
    if arrow_length is None or arrow_length <= 0:
        arrow_length = 10.0 * radius

    arrow_color = color[0]
    for start, end in zip(route_points[:-1], route_points[1:]):
        segment = end - start
        segment_length = float(np.linalg.norm(segment))
        if segment_length <= 1e-8:
            continue
        direction = segment / segment_length
        current_arrow_length = min(arrow_length, 0.65 * segment_length)
        arrow_end = end
        arrow_start = arrow_end - direction * current_arrow_length
        geometries.extend(_arrow_mesh(arrow_start, arrow_end, arrow_color, arrow_radius))
    return geometries


def _stroke_char_width(character):
    return float(_STROKE_GLYPH_WIDTH.get(character, 1.0))


def _estimate_stroke_text_width(text, scale, spacing):
    if not text:
        return 0.0
    total = 0.0
    for idx, character in enumerate(text):
        total += _stroke_char_width(character) * scale
        if idx < len(text) - 1:
            total += spacing
    return total


def _stroke_text_meshes(text, center, right, up, color, char_height, radius, spacing_factor=0.28):
    text = str(text)
    if len(text) == 0:
        return []

    right = np.asarray(right, dtype=float)
    up = np.asarray(up, dtype=float)
    right_norm = float(np.linalg.norm(right))
    up_norm = float(np.linalg.norm(up))
    if right_norm <= 1e-12 or up_norm <= 1e-12:
        return []
    right = right / right_norm
    up = up / up_norm
    center = np.asarray(center, dtype=float)

    scale = float(char_height) / 2.0
    spacing = spacing_factor * scale
    total_width = _estimate_stroke_text_width(text, scale, spacing)
    lower_left = center - right * (0.5 * total_width) - up * (0.5 * char_height)

    geometries = []
    cursor = 0.0
    for character in text:
        glyph_segments = _STROKE_GLYPHS.get(character)
        char_width = _stroke_char_width(character) * scale
        if glyph_segments is not None:
            for (x0, y0), (x1, y1) in glyph_segments:
                start = lower_left + right * (cursor + x0 * scale) + up * (y0 * scale)
                end = lower_left + right * (cursor + x1 * scale) + up * (y1 * scale)
                geometries.extend(_line_segment_mesh(start, end, color=color, radius=radius))
        cursor += char_width + spacing
    return geometries


def _tick_label_decimals(tick_step, max_decimals=4):
    tick_step = float(abs(tick_step))
    decimals = 0
    while decimals < max_decimals:
        scaled = tick_step * (10 ** decimals)
        if abs(scaled - round(scaled)) <= 1e-8:
            return decimals
        decimals += 1
    return max_decimals


def _format_axis_tick_label(value, tick_step):
    decimals = _tick_label_decimals(tick_step)
    rounded = round(float(value), decimals)
    if abs(rounded) < 10 ** (-(decimals + 1)):
        rounded = 0.0
    if decimals == 0 or abs(rounded - round(rounded)) <= 1e-8:
        return str(int(round(rounded)))
    return f"{rounded:.{decimals}f}".rstrip("0").rstrip(".")


def resolve_world_axis_style(
    mesh,
    camerapose,
    constraint_min=None,
    constraint_max=None,
    axis_length=None,
    tick_step=0.5,
    tick_size=None,
    axis_radius=None,
    label_size=None,
):
    axis_length = resolve_world_axis_length(
        mesh,
        camerapose,
        constraint_min=constraint_min,
        constraint_max=constraint_max,
        axis_length=axis_length,
    )
    tick_step = max(float(tick_step), 1e-6)
    if tick_size is None or tick_size <= 0:
        tick_size = max(0.02, 0.03 * axis_length)
    if axis_radius is None or axis_radius <= 0:
        axis_radius = max(0.0015, 0.0025 * axis_length)
    if label_size is None or label_size <= 0:
        label_size = max(1.8 * tick_size, 0.35 * tick_step)

    tick_values = np.arange(-axis_length, axis_length + 0.5 * tick_step, tick_step)
    nonzero_tick_values = [float(value) for value in tick_values if abs(value) >= 1e-8]
    max_label_width = 0.0
    for value in nonzero_tick_values:
        label = _format_axis_tick_label(value, tick_step)
        width = _estimate_stroke_text_width(label, label_size / 2.0, 0.28 * (label_size / 2.0))
        max_label_width = max(max_label_width, width)
    label_stride = 1
    if max_label_width > 0.0:
        label_stride = max(1, int(np.ceil((max_label_width + 0.3 * label_size) / max(tick_step, 1e-6))))

    return {
        "axis_length": axis_length,
        "tick_step": tick_step,
        "tick_size": tick_size,
        "axis_radius": axis_radius,
        "label_size": label_size,
        "label_stride": label_stride,
        "label_offset": 2.2 * tick_size + 0.55 * label_size,
        "tick_values": tick_values,
    }


def buildCameraColorPalette(camera_num, cmap_name="tab10"):
    camera_num = int(camera_num)
    if camera_num <= 0:
        return np.zeros((0, 3), dtype=float)
    cmap = plt.get_cmap(cmap_name)
    if hasattr(cmap, "colors") and len(cmap.colors) >= camera_num:
        return np.asarray(cmap.colors[:camera_num], dtype=float)
    return cmap(np.linspace(0.0, 1.0, camera_num, endpoint=False))[:, :3]


def buildHotspotMask(weights, hotspot_fraction):
    weights = np.asarray(weights, dtype=float).reshape(-1)
    hotspot_fraction = float(hotspot_fraction)
    if len(weights) == 0:
        return np.zeros((0,), dtype=bool)
    if hotspot_fraction <= 0.0 or hotspot_fraction > 1.0:
        raise ValueError("hotspot_fraction must be in (0, 1]")
    hotspot_count = max(int(np.ceil(len(weights) * hotspot_fraction)), 1)
    ranking = np.argsort(weights)[::-1]
    mask = np.zeros(len(weights), dtype=bool)
    mask[ranking[:hotspot_count]] = True
    return mask


def computeCameraResponsibility(voxelnormals, camerapose, visibility, hotspot_mask=None):
    voxelnormals = np.asarray(voxelnormals, dtype=float)
    camerapose = np.asarray(camerapose, dtype=float)
    visibility = np.asarray(visibility, dtype=float)
    if voxelnormals.ndim != 2 or voxelnormals.shape[1] < 6:
        raise ValueError("voxelnormals must have shape (N, 6) or more")
    if visibility.shape != (len(voxelnormals), len(camerapose)):
        raise ValueError("visibility must have shape (N_voxel, N_camera)")

    points = voxelnormals[:, :3]
    normals = voxelnormals[:, 3:6]
    rays = camerapose[None, :, :] - points[:, None, :]
    distances = np.linalg.norm(rays, axis=2)
    safe_distances = np.maximum(distances, 1e-8)
    ray_dirs = rays / safe_distances[:, :, None]
    normal_norms = np.linalg.norm(normals, axis=1, keepdims=True)
    safe_normals = normals / np.maximum(normal_norms, 1e-8)
    frontality = np.clip(np.sum(ray_dirs * safe_normals[:, None, :], axis=2), 0.0, 1.0)
    inverse_distance = 1.0 / safe_distances
    inverse_distance = inverse_distance / np.maximum(np.max(inverse_distance, axis=1, keepdims=True), 1e-8)

    responsibility_score = visibility * (0.7 * frontality + 0.3 * inverse_distance)
    labels = np.argmax(responsibility_score, axis=1).astype(int)
    labels[np.max(responsibility_score, axis=1) <= 1e-8] = -1

    if hotspot_mask is not None:
        hotspot_mask = np.asarray(hotspot_mask, dtype=bool).reshape(-1)
        if len(hotspot_mask) != len(labels):
            raise ValueError("hotspot_mask must match the number of voxels")
        labels[~hotspot_mask] = -1
    return labels, responsibility_score


def getResponsibilityPointCloud(points, labels, camera_colors, background_color=(0.18, 0.18, 0.18), show_background=False):
    points = np.asarray(points, dtype=float)
    labels = np.asarray(labels, dtype=int).reshape(-1)
    camera_colors = np.asarray(camera_colors, dtype=float)
    if len(points) != len(labels):
        raise ValueError("points and labels must have the same length")
    if camera_colors.ndim != 2 or camera_colors.shape[1] != 3:
        raise ValueError("camera_colors must have shape (N, 3)")

    if show_background:
        mask = np.ones(len(labels), dtype=bool)
    else:
        mask = labels >= 0
    filtered_points = points[mask]
    filtered_labels = labels[mask]
    colors = np.tile(np.asarray(background_color, dtype=float).reshape(1, 3), (len(filtered_points), 1))
    valid = filtered_labels >= 0
    if np.any(valid):
        colors[valid] = camera_colors[filtered_labels[valid]]

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(filtered_points)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)
    return point_cloud


def getScalarPointCloud(points, values, colormap="plasma", vmin=None, vmax=None):
    points = np.asarray(points, dtype=float)
    values = np.asarray(values, dtype=float)
    if len(points) != len(values):
        raise ValueError("points and values must have the same length")
    if len(points) == 0:
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(np.zeros((0, 3), dtype=float))
        return point_cloud

    finite = np.isfinite(values)
    safe_values = values.copy()
    if not np.any(finite):
        safe_values[:] = 0.0
        finite = np.ones_like(values, dtype=bool)
    finite_values = safe_values[finite]
    if vmin is None:
        vmin = float(np.min(finite_values))
    if vmax is None:
        vmax = float(np.max(finite_values))
    if vmax <= vmin + 1e-12:
        normalized = np.zeros_like(safe_values, dtype=float)
    else:
        normalized = np.clip((safe_values - vmin) / (vmax - vmin), 0.0, 1.0)
    colors = plt.get_cmap(colormap)(normalized)[:, :3]
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)
    return point_cloud


def getCoveragePointCloud(points, coverage_counts, kcoverage, colormap="viridis"):
    max_coverage = max(float(kcoverage), 1.0)
    return getScalarPointCloud(points, np.asarray(coverage_counts, dtype=float), colormap=colormap, vmin=0.0, vmax=max_coverage)


def getCoverageDeltaPointCloud(points, delta_counts, colormap="RdYlGn"):
    delta_counts = np.asarray(delta_counts, dtype=float)
    limit = max(float(np.max(np.abs(delta_counts))), 1.0)
    return getScalarPointCloud(points, delta_counts, colormap=colormap, vmin=-limit, vmax=limit)


def _line_segment_mesh(start, end, color, radius):
    points = np.stack((np.asarray(start, dtype=float), np.asarray(end, dtype=float)), axis=0)
    colors = np.asarray(color, dtype=float).reshape(1, 3)
    line_mesh = LineMesh(points, np.asarray([[0, 1]], dtype=np.int32), colors, radius=radius)
    return line_mesh.cylinder_segments


def _arrow_mesh(start, end, color, radius):
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    direction = end - start
    length = float(np.linalg.norm(direction))
    if length <= 1e-8:
        return []

    direction = direction / length
    cone_height = min(max(2.6 * radius, 0.35 * length), 0.65 * length)
    cylinder_height = max(length - cone_height, 1e-6)
    cone_radius = max(1.8 * radius, 1e-6)
    arrow = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=max(radius, 1e-6),
        cone_radius=cone_radius,
        cylinder_height=cylinder_height,
        cone_height=cone_height,
    )
    axis, angle = align_vector_to_another(np.array([0.0, 0.0, 1.0]), direction)
    if axis is not None:
        axis_angle = axis * angle
        arrow.rotate(o3d.geometry.get_rotation_matrix_from_axis_angle(axis_angle), center=np.zeros(3))
    arrow.translate(start, relative=True)
    arrow.paint_uniform_color(np.asarray(color, dtype=float))
    if not arrow.has_vertex_normals():
        arrow.compute_vertex_normals()
    return [arrow]


def resolve_world_axis_length(mesh, camerapose, constraint_min=None, constraint_max=None, axis_length=None):
    if axis_length is not None and axis_length > 0:
        return float(axis_length)

    min_bounds = []
    max_bounds = []
    if hasattr(mesh, "get_min_bound") and hasattr(mesh, "get_max_bound"):
        min_bounds.append(np.asarray(mesh.get_min_bound(), dtype=float))
        max_bounds.append(np.asarray(mesh.get_max_bound(), dtype=float))

    camerapose = np.asarray(camerapose, dtype=float)
    if camerapose.size > 0:
        min_bounds.append(np.min(camerapose, axis=0))
        max_bounds.append(np.max(camerapose, axis=0))

    if constraint_min is not None and constraint_max is not None:
        min_bounds.append(np.asarray(constraint_min, dtype=float))
        max_bounds.append(np.asarray(constraint_max, dtype=float))

    if len(min_bounds) == 0:
        return 1.0

    lower = np.min(np.stack(min_bounds, axis=0), axis=0)
    upper = np.max(np.stack(max_bounds, axis=0), axis=0)
    max_abs = float(np.max(np.abs(np.concatenate((lower, upper), axis=0))))
    return max(1.0, 1.1 * max_abs)


def getWorldAxesVis(
    mesh,
    camerapose,
    constraint_min=None,
    constraint_max=None,
    axis_length=None,
    tick_step=0.5,
    tick_size=None,
    axis_radius=None,
    show_tick_labels=True,
    label_size=None,
):
    style = resolve_world_axis_style(
        mesh,
        camerapose,
        constraint_min=constraint_min,
        constraint_max=constraint_max,
        axis_length=axis_length,
        tick_step=tick_step,
        tick_size=tick_size,
        axis_radius=axis_radius,
        label_size=label_size,
    )
    axis_length = style["axis_length"]
    tick_step = style["tick_step"]
    tick_size = style["tick_size"]
    axis_radius = style["axis_radius"]
    label_size = style["label_size"]
    label_stride = style["label_stride"]
    label_offset = style["label_offset"]

    axes = [
        ("x", np.array([1.0, 0.2, 0.2]), np.array([1.0, 0.0, 0.0])),
        ("y", np.array([0.2, 0.8, 0.2]), np.array([0.0, 1.0, 0.0])),
        ("z", np.array([0.2, 0.4, 1.0]), np.array([0.0, 0.0, 1.0])),
    ]

    geometries = []
    for _, color, direction in axes:
        geometries.extend(
            _line_segment_mesh(
                -axis_length * direction,
                axis_length * direction,
                color=color,
                radius=axis_radius,
            )
        )

    tick_values = np.arange(-axis_length, axis_length + 0.5 * tick_step, tick_step)
    for axis_name, color, _ in axes:
        for value in tick_values:
            if abs(value) < 1e-8:
                continue
            if axis_name == "x":
                start = np.array([value, -tick_size, 0.0], dtype=float)
                end = np.array([value, tick_size, 0.0], dtype=float)
            elif axis_name == "y":
                start = np.array([-tick_size, value, 0.0], dtype=float)
                end = np.array([tick_size, value, 0.0], dtype=float)
            else:
                start = np.array([-tick_size, 0.0, value], dtype=float)
                end = np.array([tick_size, 0.0, value], dtype=float)
            geometries.extend(_line_segment_mesh(start, end, color=color, radius=0.7 * axis_radius))

    if show_tick_labels:
        text_radius = max(0.55 * axis_radius, 1e-6)
        axis_label_size = 1.15 * label_size
        for axis_name, color, _ in axes:
            for value in tick_values:
                if abs(value) < 1e-8:
                    continue
                step_index = int(round(abs(value) / tick_step))
                if step_index % label_stride != 0:
                    continue
                label = _format_axis_tick_label(value, tick_step)
                if axis_name == "x":
                    label_center = np.array([value, -label_offset, 0.0], dtype=float)
                    right = np.array([1.0, 0.0, 0.0], dtype=float)
                    up = np.array([0.0, 1.0, 0.0], dtype=float)
                elif axis_name == "y":
                    label_center = np.array([label_offset, value, 0.0], dtype=float)
                    right = np.array([1.0, 0.0, 0.0], dtype=float)
                    up = np.array([0.0, 1.0, 0.0], dtype=float)
                else:
                    label_center = np.array([label_offset, 0.0, value], dtype=float)
                    right = np.array([1.0, 0.0, 0.0], dtype=float)
                    up = np.array([0.0, 0.0, 1.0], dtype=float)
                geometries.extend(
                    _stroke_text_meshes(
                        label,
                        label_center,
                        right=right,
                        up=up,
                        color=color,
                        char_height=label_size,
                        radius=text_radius,
                    )
                )

        geometries.extend(
            _stroke_text_meshes(
                "X",
                np.array([axis_length + 0.95 * label_offset, 0.0, 0.0], dtype=float),
                right=np.array([1.0, 0.0, 0.0], dtype=float),
                up=np.array([0.0, 1.0, 0.0], dtype=float),
                color=axes[0][1],
                char_height=axis_label_size,
                radius=text_radius,
            )
        )
        geometries.extend(
            _stroke_text_meshes(
                "Y",
                np.array([0.75 * label_offset, axis_length + 0.95 * label_offset, 0.0], dtype=float),
                right=np.array([1.0, 0.0, 0.0], dtype=float),
                up=np.array([0.0, 1.0, 0.0], dtype=float),
                color=axes[1][1],
                char_height=axis_label_size,
                radius=text_radius,
            )
        )
        geometries.extend(
            _stroke_text_meshes(
                "Z",
                np.array([0.75 * label_offset, 0.0, axis_length + 0.95 * label_offset], dtype=float),
                right=np.array([1.0, 0.0, 0.0], dtype=float),
                up=np.array([0.0, 0.0, 1.0], dtype=float),
                color=axes[2][1],
                char_height=axis_label_size,
                radius=text_radius,
            )
        )

    origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=4.0 * axis_radius, origin=[0.0, 0.0, 0.0])
    geometries.append(origin_frame)
    return geometries


def merge_triangle_meshes(meshes):
    merged = o3d.geometry.TriangleMesh()
    for mesh in meshes:
        merged += mesh
    return merged


def remove_if_exists(path):
    if os.path.exists(path):
        os.remove(path)


def cleanup_scene_export_prefix(export_prefix):
    export_dir = os.path.dirname(export_prefix) or "."
    os.makedirs(export_dir, exist_ok=True)
    base_name = os.path.basename(export_prefix)

    for filename in os.listdir(export_dir):
        if (
            filename.startswith(base_name + "_geometry")
            or filename.startswith(base_name + "_overlay")
            or filename.startswith(base_name + "_occupancy")
            or filename == base_name + "_view.json"
            or filename == base_name + "_scene.json"
            or filename == base_name + "_scene.zip"
        ):
            remove_if_exists(os.path.join(export_dir, filename))


def export_geometry_bundle(mesh, export_prefix):
    export_dir = os.path.dirname(export_prefix) or "."
    base_name = os.path.basename(export_prefix)

    if isinstance(mesh, o3d.geometry.PointCloud):
        if len(mesh.points) == 0:
            return "none", None, [], False
        geometry_path = export_prefix + "_geometry.ply"
        o3d.io.write_point_cloud(geometry_path, mesh)
        return "point_cloud", geometry_path, [geometry_path], False

    if not isinstance(mesh, o3d.geometry.TriangleMesh):
        raise TypeError(f"Unsupported Open3D geometry type: {type(mesh)!r}")

    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()

    texture_preserved = bool(mesh.has_textures() and mesh.has_triangle_uvs())
    if texture_preserved:
        geometry_path = export_prefix + "_geometry.obj"
        o3d.io.write_triangle_mesh(
            geometry_path,
            mesh,
            write_vertex_normals=True,
            write_vertex_colors=True,
            write_triangle_uvs=True,
        )
        geometry_assets = [
            os.path.join(export_dir, filename)
            for filename in sorted(os.listdir(export_dir))
            if filename.startswith(base_name + "_geometry")
        ]
        return "triangle_mesh", geometry_path, geometry_assets, True

    geometry_path = export_prefix + "_geometry.ply"
    o3d.io.write_triangle_mesh(
        geometry_path,
        mesh,
        write_vertex_normals=True,
        write_vertex_colors=True,
    )
    return "triangle_mesh", geometry_path, [geometry_path], False


def create_scene_archive(archive_path, bundle_paths):
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as zip_file:
        for path in bundle_paths:
            zip_file.write(path, arcname=os.path.basename(path))


def build_hidden_geometry_placeholder():
    return o3d.geometry.PointCloud()


def save_visualization_scene(
    mesh,
    camera_geometries,
    constraint_geometries,
    export_prefix,
    view_parameters=None,
    occupancy_cloud=None,
):
    export_dir = os.path.dirname(export_prefix)
    if export_dir:
        os.makedirs(export_dir, exist_ok=True)
    cleanup_scene_export_prefix(export_prefix)

    geometry_type, geometry_path, geometry_assets, texture_preserved = export_geometry_bundle(
        mesh,
        export_prefix,
    )

    overlay_path = None
    overlay_meshes = list(camera_geometries) + list(constraint_geometries)
    overlay_assets = []
    if len(overlay_meshes) > 0:
        overlay_mesh = merge_triangle_meshes(overlay_meshes)
        if not overlay_mesh.has_vertex_normals():
            overlay_mesh.compute_vertex_normals()
        overlay_path = export_prefix + "_overlay.ply"
        o3d.io.write_triangle_mesh(
            overlay_path,
            overlay_mesh,
            write_vertex_normals=True,
            write_vertex_colors=True,
        )
        overlay_assets = [overlay_path]

    occupancy_path = None
    occupancy_assets = []
    if occupancy_cloud is not None and len(occupancy_cloud.points) > 0:
        occupancy_path = export_prefix + "_occupancy.ply"
        o3d.io.write_point_cloud(occupancy_path, occupancy_cloud)
        occupancy_assets = [occupancy_path]

    view_path = None
    if view_parameters is not None:
        view_path = export_prefix + "_view.json"
        o3d.io.write_pinhole_camera_parameters(view_path, view_parameters)

    manifest = {
        "geometry_type": geometry_type,
        "geometry_file": os.path.basename(geometry_path) if geometry_path is not None else None,
        "overlay_file": os.path.basename(overlay_path) if overlay_path is not None else None,
        "occupancy_file": os.path.basename(occupancy_path) if occupancy_path is not None else None,
        "view_file": os.path.basename(view_path) if view_path is not None else None,
        "texture_preserved": texture_preserved,
    }
    manifest_path = export_prefix + "_scene.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    archive_path = export_prefix + "_scene.zip"
    bundle_paths = list(geometry_assets) + overlay_assets + occupancy_assets + [manifest_path]
    if view_path is not None:
        bundle_paths.append(view_path)
    create_scene_archive(archive_path, bundle_paths)
    return manifest_path


def visGeometryScene(
    geometry,
    overlay_geometries=None,
    flag=0,
    image_path=None,
    scene_export_prefix=None,
    point_size=5.0,
    occupancy_cloud=None,
    show_geometry=True,
):
    overlay_geometries = [] if overlay_geometries is None else list(overlay_geometries)
    vis = o3d.visualization.Visualizer()
    visible = bool(flag)
    vis.create_window(window_name="scene", width=1280, height=960, left=50, visible=visible)
    render_option = vis.get_render_option()
    render_option.point_size = float(point_size)

    for overlay in overlay_geometries:
        vis.add_geometry(overlay)
    if occupancy_cloud is not None:
        vis.add_geometry(occupancy_cloud)
    if show_geometry:
        vis.add_geometry(geometry)
    vis.reset_view_point(True)
    vis.poll_events()
    vis.update_renderer()

    def export_current_view():
        if image_path is not None:
            vis.capture_screen_image(image_path, do_render=True)
        if scene_export_prefix is not None:
            view_parameters = vis.get_view_control().convert_to_pinhole_camera_parameters()
            save_visualization_scene(
                geometry if show_geometry else build_hidden_geometry_placeholder(),
                overlay_geometries,
                [],
                scene_export_prefix,
                view_parameters=view_parameters,
                occupancy_cloud=occupancy_cloud,
            )

    if image_path is not None or scene_export_prefix is not None:
        export_current_view()
    if flag == 1:
        vis.run()
        vis.poll_events()
        vis.update_renderer()
        export_current_view()
    vis.destroy_window()


def createImageSummaryMontage(image_specs, output_path, title=None, columns=3, background=(18, 18, 18), tile_padding=20, title_padding=18):
    valid_specs = [(label, path) for label, path in image_specs if path is not None and os.path.exists(path)]
    if len(valid_specs) == 0:
        return None

    columns = max(int(columns), 1)
    font = ImageFont.load_default()
    opened = []
    max_width = 0
    max_height = 0
    label_heights = []
    for label, path in valid_specs:
        image = Image.open(path).convert("RGB")
        opened.append((label, image))
        max_width = max(max_width, image.width)
        max_height = max(max_height, image.height)
        bbox = font.getbbox(label)
        label_heights.append((bbox[3] - bbox[1]) + 12)

    label_height = max(label_heights) if label_heights else 24
    rows = int(np.ceil(len(opened) / columns))
    title_height = 0
    if title:
        title_bbox = font.getbbox(title)
        title_height = (title_bbox[3] - title_bbox[1]) + 2 * title_padding

    tile_width = max_width
    tile_height = max_height + label_height
    canvas_width = columns * tile_width + (columns + 1) * tile_padding
    canvas_height = title_height + rows * tile_height + (rows + 1) * tile_padding
    canvas = Image.new("RGB", (canvas_width, canvas_height), color=background)
    draw = ImageDraw.Draw(canvas)

    current_y = tile_padding
    if title:
        title_bbox = font.getbbox(title)
        title_width = title_bbox[2] - title_bbox[0]
        draw.text(((canvas_width - title_width) / 2, current_y), title, font=font, fill=(245, 245, 245))
        current_y += title_height

    for idx, (label, image) in enumerate(opened):
        row = idx // columns
        col = idx % columns
        tile_x = tile_padding + col * (tile_width + tile_padding)
        tile_y = current_y + row * (tile_height + tile_padding)

        fitted = ImageOps.contain(image, (tile_width, max_height))
        image_x = tile_x + (tile_width - fitted.width) // 2
        image_y = tile_y
        canvas.paste(fitted, (image_x, image_y))

        text_bbox = font.getbbox(label)
        text_width = text_bbox[2] - text_bbox[0]
        text_x = tile_x + (tile_width - text_width) / 2
        text_y = tile_y + max_height + 6
        draw.text((text_x, text_y), label, font=font, fill=(235, 235, 235))

    canvas.save(output_path)
    return output_path


def visMesh(
    mesh,
    camerapose,
    camerarotation,
    flag=0,
    image_path=None,
    camera_constraint_shape="box",
    camera_constraint_min=None,
    camera_constraint_max=None,
    camera_constraint_data=None,
    show_world_axes=True,
    show_world_axis_tick_labels=True,
    world_axis_length=None,
    world_axis_tick_step=0.5,
    world_axis_tick_size=None,
    world_axis_radius=None,
    world_axis_label_size=None,
    show_occupancy_map=False,
    occupancy_points=None,
    occupancy_weights=None,
    occupancy_route_points=None,
    show_occupancy_route=True,
    show_occupancy_route_direction=True,
    occupancy_vis_clip_percentile=95.0,
    occupancy_vis_point_size=5.0,
    occupancy_vis_route_radius=0.004,
    occupancy_vis_route_arrow_radius=0.005,
    occupancy_vis_route_arrow_length=0.05,
    scene_export_prefix=None,
    show_base_geometry=True,
    camera_models=None,
    camera_vis_scale=0.2,
    show_camera_optical_axis=True,
    show_camera_fov=True,
    camera_optical_axis_length=None,
    camera_line_radius=None,
):
    vis = o3d.visualization.Visualizer()
    visible = bool(flag)
    vis.create_window(window_name="scene",width=1280,height=960,left=50,visible=visible)
    render_option = vis.get_render_option()
    render_option.point_size = float(occupancy_vis_point_size)

    ##############control view ############
    resolved_camera_ray_length = resolve_camera_ray_length(
        mesh,
        camerapose,
        constraint_min=camera_constraint_min,
        constraint_max=camera_constraint_max,
        ray_length=camera_optical_axis_length,
    )
    camera_geometries = build_camera_geometries(
        camerapose,
        camerarotation,
        camera_models=camera_models,
        camera_scale=camera_vis_scale,
        show_optical_axis=show_camera_optical_axis,
        optical_axis_length=resolved_camera_ray_length,
        show_fov=show_camera_fov,
        line_radius=camera_line_radius,
    )
    for geometry in camera_geometries:
        vis.add_geometry(geometry)
    constraint_geometries = []
    if camera_constraint_min is not None and camera_constraint_max is not None:
        constraint_geometries = getCameraConstraintVis(
            camera_constraint_shape,
            camera_constraint_min,
            camera_constraint_max,
            constraint_data=camera_constraint_data,
        )
        for geometry in constraint_geometries:
            vis.add_geometry(geometry)
    axis_geometries = []
    if show_world_axes:
        axis_geometries = getWorldAxesVis(
            mesh,
            camerapose,
            constraint_min=camera_constraint_min,
            constraint_max=camera_constraint_max,
            axis_length=world_axis_length,
            tick_step=world_axis_tick_step,
            tick_size=world_axis_tick_size,
            axis_radius=world_axis_radius,
            show_tick_labels=show_world_axis_tick_labels,
            label_size=world_axis_label_size,
        )
        for geometry in axis_geometries:
            vis.add_geometry(geometry)
    route_geometries = []
    if show_occupancy_route and occupancy_route_points is not None:
        route_geometries = getOccupancyRouteVis(
            occupancy_route_points,
            radius=occupancy_vis_route_radius,
            show_direction=show_occupancy_route_direction,
            arrow_radius=occupancy_vis_route_arrow_radius,
            arrow_length=occupancy_vis_route_arrow_length,
        )
        for geometry in route_geometries:
            vis.add_geometry(geometry)
    occupancy_cloud = None
    if show_occupancy_map and occupancy_points is not None and occupancy_weights is not None:
        occupancy_cloud = getOccupancyPointCloud(
            occupancy_points,
            occupancy_weights,
            clip_percentile=occupancy_vis_clip_percentile,
        )
        vis.add_geometry(occupancy_cloud)
    if show_base_geometry:
        vis.add_geometry(mesh)
    vis.reset_view_point(True)
    vis.poll_events()
    vis.update_renderer()
    def export_current_view():
        if image_path is not None:
            vis.capture_screen_image(image_path, do_render=True)
        if scene_export_prefix is not None:
            view_parameters = vis.get_view_control().convert_to_pinhole_camera_parameters()
            save_visualization_scene(
                mesh if show_base_geometry else build_hidden_geometry_placeholder(),
                camera_geometries,
                constraint_geometries + axis_geometries + route_geometries,
                scene_export_prefix,
                view_parameters=view_parameters,
                occupancy_cloud=occupancy_cloud,
            )

    if image_path is not None or scene_export_prefix is not None:
        export_current_view()
    if flag==1 :
        vis.run()
        vis.poll_events()
        vis.update_renderer()
        export_current_view()
    vis.destroy_window()

def visFieldColor(pointnormals,coverage_num,name=None,scale=None):
    point_cloud = o3d.geometry.PointCloud()# add param
    points_array=pointnormals[:,:3]
    normal_array=pointnormals[:,3:]
    if scale != None:
        points_array = points_array*scale[0]+scale[1]
    point_cloud.points=o3d.utility.Vector3dVector(points_array)
    point_cloud.normals=o3d.utility.Vector3dVector(normal_array)
    label=5
    print(np.max(coverage_num),np.min(coverage_num))
    colors=plt.get_cmap("coolwarm")(np.minimum(label,coverage_num)/(label if label > 0 else 1))
    point_cloud.colors=o3d.utility.Vector3dVector(colors[:,:3])
    o3d.visualization.draw_geometries([point_cloud])
    o3d.io.write_point_cloud(name,point_cloud)

def visualization(pointnormals,camerapose,camerarotation,extrinsic=None,coverage_num=None,imagep=None,flag=0):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="scene",width=1280,height=960,left=50)
    point_cloud = o3d.geometry.PointCloud()# add param
    points_array=pointnormals[:,:3]
    normal_array=pointnormals[:,3:]
    ##############control view ############
    crt=vis.get_view_control()
    pinholecameracurrent=crt.convert_to_pinhole_camera_parameters()
    if len(camerapose)==1 and extrinsic is not None:
        pinholecamera=o3d.camera.PinholeCameraParameters()
        pinholecamera.extrinsic=extrinsic
        pinholecamera.intrinsic=pinholecameracurrent.intrinsic
    ########################################
    point_cloud.points=o3d.utility.Vector3dVector(points_array)
    point_cloud.normals=o3d.utility.Vector3dVector(normal_array)
    # if coverage_num!= None:
    label=4
    colors=plt.get_cmap("plasma")(1-(np.minimum(label,coverage_num)/(label if label > 0 else 1)))
    point_cloud.colors=o3d.utility.Vector3dVector(colors[:,:3])
    # o3d.io.write_point_cloud(imagep,point_cloud)
    for geometry in build_camera_geometries(camerapose, camerarotation):
        vis.add_geometry(geometry)
    
    vis.add_geometry(point_cloud)
    if len(camerapose)==1 and extrinsic is not None:
        crt.convert_from_pinhole_camera_parameters(pinholecamera,allow_arbitrary=True)
    vis.poll_events()
    vis.update_renderer()
    if flag==1 :
        vis.run()
    if imagep !=None:
        vis.capture_screen_image(imagep+".png")
    vis.close()
def generateline(index,points,camerapose):
    camera_points=np.append(camerapose,points,axis=0)
    point_connection=np.zeros(shape=[0,2])
    for i in range(len(index[0])):
        point_connection=np.append(point_connection,np.array([[index[0][i]+len(camerapose),index[1][i]]]),axis=0)
    return camera_points,point_connection
def visaddline(pointnormals,camerapose,camerarotation,coverage_num,index,imagepath=None,flag=0):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="scene",width=1280,height=960,left=50)
    crt=vis.get_view_control()
    point_cloud = o3d.geometry.PointCloud()# add param
    points_array=pointnormals[:,:3]
    normal_array=pointnormals[:,3:]
    color_array=np.zeros_like(points_array)
    for i in range(len(coverage_num)):
        if coverage_num[i] == 0:
            color_array[i]=[255./255,0./255,0./255]
        elif coverage_num[i] == 1:
            color_array[i]=[255./255,255./255,0./255]
        elif coverage_num[i] == 2 :
            color_array[i]=[0./255,0./255,255./255]
        elif coverage_num[i] == 3 :
            color_array[i]=[0./255,255./255,0./255]
    point_cloud.points=o3d.utility.Vector3dVector(points_array)
    point_cloud.normals=o3d.utility.Vector3dVector(normal_array)
    point_cloud.colors=o3d.utility.Vector3dVector(color_array)
    camera_color1=np.array([[0,0,1]])
    camera_color2=np.array( [[0,0,0],
                                [0,0,0],
                                [0,0,0],
                                [0,0,0],
                                [0,0,0]])
    for geometry in build_camera_geometries(camerapose, camerarotation, color=camera_color1, camera_scale=0.1):
        vis.add_geometry(geometry)
    point,connection=generateline(index,pointnormals[:,:3],camerapose)
    connect_lineset=o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(point),lines=o3d.utility.Vector2iVector(connection))
    connect_lineset.colors = o3d.utility.Vector3dVector(camera_color2)
    vis.add_geometry(connect_lineset)
    vis.add_geometry(point_cloud)
    vis.poll_events()
    vis.update_renderer()
    if flag==1 :
        vis.run()
    if imagepath != None:
        vis.capture_screen_image(imagepath+".png")
    vis.close()
def visualPointselect(pointnormals):
    point_cloud = o3d.geometry.PointCloud()# add param
    points_array=np.append(pointnormals[:,:2],np.ones((len(pointnormals),1)),1)
    normal_array=np.zeros_like(points_array)
    normal_array[:,2]=1
    color_array=np.zeros_like(points_array)
    color_array[:,0]=255./255*pointnormals[:,2]
    max=np.argmax(pointnormals[:,2])
    color_array[max]=[0,1,0]
    point_cloud.points=o3d.utility.Vector3dVector(points_array)
    point_cloud.normals=o3d.utility.Vector3dVector(normal_array)
    point_cloud.colors=o3d.utility.Vector3dVector(color_array)
    o3d.visualization.draw_geometries([point_cloud])
