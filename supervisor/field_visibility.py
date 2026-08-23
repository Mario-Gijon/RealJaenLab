"""Visible FIELD footprint derived from a world->image homography.

The canonical pitch is always complete and stable. This module answers:

    Given the current camera homography, which part of the real 105 x 68 m
    playing surface is inside the video viewport?

H maps world pitch coordinates -> image pixels.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0
HALF_LENGTH = PITCH_LENGTH / 2.0
HALF_WIDTH = PITCH_WIDTH / 2.0
PITCH_AREA_M2 = PITCH_LENGTH * PITCH_WIDTH

PITCH_POLYGON_WORLD = np.asarray(
    [
        (-HALF_LENGTH, -HALF_WIDTH),
        (HALF_LENGTH, -HALF_WIDTH),
        (HALF_LENGTH, HALF_WIDTH),
        (-HALF_LENGTH, HALF_WIDTH),
    ],
    dtype=np.float64,
)

PITCH_LINES = {
    "top_touchline": ((-52.5, -34.0), (52.5, -34.0)),
    "bottom_touchline": ((-52.5, 34.0), (52.5, 34.0)),
    "left_goal_line": ((-52.5, -34.0), (-52.5, 34.0)),
    "right_goal_line": ((52.5, -34.0), (52.5, 34.0)),
    "halfway_line": ((0.0, -34.0), (0.0, 34.0)),
    "left_penalty_top": ((-52.5, -20.16), (-36.0, -20.16)),
    "left_penalty_front": ((-36.0, -20.16), (-36.0, 20.16)),
    "left_penalty_bottom": ((-36.0, 20.16), (-52.5, 20.16)),
    "right_penalty_top": ((52.5, -20.16), (36.0, -20.16)),
    "right_penalty_front": ((36.0, -20.16), (36.0, 20.16)),
    "right_penalty_bottom": ((36.0, 20.16), (52.5, 20.16)),
    "left_goal_top": ((-52.5, -9.16), (-47.0, -9.16)),
    "left_goal_front": ((-47.0, -9.16), (-47.0, 9.16)),
    "left_goal_bottom": ((-47.0, 9.16), (-52.5, 9.16)),
    "right_goal_top": ((52.5, -9.16), (47.0, -9.16)),
    "right_goal_front": ((47.0, -9.16), (47.0, 9.16)),
    "right_goal_bottom": ((47.0, 9.16), (52.5, 9.16)),
}

PITCH_LANDMARKS = {
    "top_left_corner": (-52.5, -34.0),
    "top_right_corner": (52.5, -34.0),
    "bottom_right_corner": (52.5, 34.0),
    "bottom_left_corner": (-52.5, 34.0),
    "halfway_top": (0.0, -34.0),
    "halfway_bottom": (0.0, 34.0),
    "centre_mark": (0.0, 0.0),
    "left_penalty_top": (-36.0, -20.16),
    "left_penalty_bottom": (-36.0, 20.16),
    "right_penalty_top": (36.0, -20.16),
    "right_penalty_bottom": (36.0, 20.16),
    "left_goal_top": (-47.0, -9.16),
    "left_goal_bottom": (-47.0, 9.16),
    "right_goal_top": (47.0, -9.16),
    "right_goal_bottom": (47.0, 9.16),
    "left_penalty_spot": (-41.5, 0.0),
    "right_penalty_spot": (41.5, 0.0),
    "centre_circle_halfway_top": (0.0, -9.15),
    "centre_circle_halfway_bottom": (0.0, 9.15),
}


def _as_h(matrix: Any) -> np.ndarray | None:
    try:
        h = np.asarray(matrix, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if h.shape != (3, 3) or not np.isfinite(h).all():
        return None
    scale = float(h[2, 2])
    if abs(scale) < 1e-12:
        return None
    h = h / scale
    try:
        determinant = float(np.linalg.det(h))
        condition = float(np.linalg.cond(h))
    except np.linalg.LinAlgError:
        return None
    if (
        not np.isfinite(determinant)
        or abs(determinant) < 1e-12
        or not np.isfinite(condition)
        or condition > 1e13
    ):
        return None
    return h


def _project_points(points_xy: np.ndarray, h: np.ndarray):
    points = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    homogeneous = np.column_stack((points, np.ones(len(points))))
    q = homogeneous @ h.T
    denominator = q[:, 2]
    valid = (
        np.isfinite(q).all(axis=1)
        & np.isfinite(denominator)
        & (np.abs(denominator) > 1e-10)
    )
    projected = np.full((len(points), 2), np.nan, dtype=np.float64)
    projected[valid] = q[valid, :2] / denominator[valid, None]
    valid &= np.isfinite(projected).all(axis=1)
    return projected, valid


def project_world_to_image(point_world, homography):
    h = _as_h(homography)
    if h is None:
        return None
    projected, valid = _project_points(np.asarray([point_world]), h)
    return projected[0].tolist() if bool(valid[0]) else None


def project_image_to_world(point_image, homography):
    h = _as_h(homography)
    if h is None:
        return None
    try:
        inverse = np.linalg.inv(h)
    except np.linalg.LinAlgError:
        return None
    projected, valid = _project_points(np.asarray([point_image]), inverse)
    return projected[0].tolist() if bool(valid[0]) else None


def _polygon_area(points: np.ndarray) -> float:
    polygon = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(polygon) < 3:
        return 0.0
    x = polygon[:, 0]
    y = polygon[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) * 0.5)


def _clip_polygon_half_plane(polygon, *, axis, bound, keep_greater):
    if len(polygon) == 0:
        return polygon.copy()

    def inside(point):
        value = float(point[axis])
        return value >= bound - 1e-9 if keep_greater else value <= bound + 1e-9

    output = []
    previous = polygon[-1]
    previous_inside = inside(previous)

    for current in polygon:
        current_inside = inside(current)
        if current_inside != previous_inside:
            delta = float(current[axis] - previous[axis])
            if abs(delta) > 1e-12:
                t = (bound - float(previous[axis])) / delta
                intersection = previous + t * (current - previous)
                intersection[axis] = bound
                output.append(intersection)
        if current_inside:
            output.append(current.copy())
        previous = current
        previous_inside = current_inside

    return np.asarray(output, dtype=np.float64) if output else np.empty((0, 2))


def _clip_polygon_to_viewport(polygon_image, width, height):
    polygon = np.asarray(polygon_image, dtype=np.float64).reshape(-1, 2)
    polygon = _clip_polygon_half_plane(polygon, axis=0, bound=0.0, keep_greater=True)
    polygon = _clip_polygon_half_plane(polygon, axis=0, bound=float(width), keep_greater=False)
    polygon = _clip_polygon_half_plane(polygon, axis=1, bound=0.0, keep_greater=True)
    polygon = _clip_polygon_half_plane(polygon, axis=1, bound=float(height), keep_greater=False)
    return polygon


def _clip_segment_to_viewport(start, end, width, height):
    x0, y0 = map(float, start)
    x1, y1 = map(float, end)
    dx, dy = x1 - x0, y1 - y0
    p = (-dx, dx, -dy, dy)
    q = (x0, float(width) - x0, y0, float(height) - y0)
    u0, u1 = 0.0, 1.0

    for pi, qi in zip(p, q):
        if abs(pi) < 1e-12:
            if qi < 0.0:
                return None
            continue
        ratio = qi / pi
        if pi < 0.0:
            u0 = max(u0, ratio)
        else:
            u1 = min(u1, ratio)
        if u0 > u1:
            return None

    a = np.asarray([x0 + u0 * dx, y0 + u0 * dy])
    b = np.asarray([x0 + u1 * dx, y0 + u1 * dy])
    return None if np.linalg.norm(b - a) < 1e-6 else (a, b)


def point_in_polygon(point_xy, polygon_xy, *, tolerance=1e-9):
    point = np.asarray(point_xy, dtype=np.float64).reshape(2)
    polygon = np.asarray(polygon_xy, dtype=np.float64).reshape(-1, 2)
    if len(polygon) < 3 or not np.isfinite(point).all():
        return False

    px, py = map(float, point)
    inside = False
    for index in range(len(polygon)):
        a = polygon[index]
        b = polygon[(index + 1) % len(polygon)]
        ab = b - a
        ap = point - a
        cross = abs(float(ab[0] * ap[1] - ab[1] * ap[0]))
        if cross <= tolerance * max(1.0, float(np.linalg.norm(ab))):
            dot = float(np.dot(ap, ab))
            if -tolerance <= dot <= float(np.dot(ab, ab)) + tolerance:
                return True
        ay, by = float(a[1]), float(b[1])
        if (ay > py) != (by > py):
            x_cross = float(a[0]) + (py - ay) * (float(b[0]) - float(a[0])) / (by - ay)
            if px < x_cross:
                inside = not inside
    return inside


def pitch_membership(point_world, *, borderline_margin_m=1.5):
    try:
        x, y = map(float, point_world)
    except (TypeError, ValueError):
        return {"status": "invalid", "distance_outside_m": None}

    if not math.isfinite(x) or not math.isfinite(y):
        return {"status": "invalid", "distance_outside_m": None}

    dx = max(abs(x) - HALF_LENGTH, 0.0)
    dy = max(abs(y) - HALF_WIDTH, 0.0)
    outside = math.hypot(dx, dy)

    if dx == 0.0 and dy == 0.0:
        to_boundary = min(HALF_LENGTH - abs(x), HALF_WIDTH - abs(y))
        return {
            "status": "borderline" if to_boundary <= borderline_margin_m else "inside",
            "distance_outside_m": 0.0,
            "distance_to_boundary_m": float(to_boundary),
        }

    return {
        "status": "borderline" if outside <= borderline_margin_m else "outside",
        "distance_outside_m": float(outside),
        "distance_to_boundary_m": 0.0,
    }


def compute_visible_pitch_footprint(homography, image_width, image_height):
    """Return the exact visible part of the finite pitch under H."""
    h = _as_h(homography)
    try:
        width = int(image_width)
        height = int(image_height)
    except (TypeError, ValueError):
        width, height = 0, 0

    if h is None:
        return {"status": "invalid_homography", "polygon_world": [], "polygon_image": []}
    if width <= 0 or height <= 0:
        return {"status": "invalid_image_size", "polygon_world": [], "polygon_image": []}

    projected_pitch, valid = _project_points(PITCH_POLYGON_WORLD, h)
    if not np.all(valid):
        return {"status": "pitch_projection_failed", "polygon_world": [], "polygon_image": []}

    clipped_image = _clip_polygon_to_viewport(projected_pitch, width, height)
    if len(clipped_image) < 3:
        return {
            "status": "not_visible",
            "polygon_world": [],
            "polygon_image": [],
            "visible_area_m2": 0.0,
            "visible_fraction": 0.0,
            "visible_lines": [],
            "visible_landmarks": [],
        }

    try:
        inverse = np.linalg.inv(h)
    except np.linalg.LinAlgError:
        return {"status": "invalid_homography", "polygon_world": [], "polygon_image": []}

    clipped_world, world_valid = _project_points(clipped_image, inverse)
    if not np.all(world_valid):
        return {
            "status": "inverse_projection_failed",
            "polygon_world": [],
            "polygon_image": clipped_image.tolist(),
        }

    clipped_world[:, 0] = np.clip(clipped_world[:, 0], -HALF_LENGTH, HALF_LENGTH)
    clipped_world[:, 1] = np.clip(clipped_world[:, 1], -HALF_WIDTH, HALF_WIDTH)

    visible_area = min(PITCH_AREA_M2, _polygon_area(clipped_world))
    visible_lines = []

    for semantic, (world_start, world_end) in PITCH_LINES.items():
        projected, line_valid = _project_points(np.asarray([world_start, world_end]), h)
        if not np.all(line_valid):
            continue
        clipped = _clip_segment_to_viewport(projected[0], projected[1], width, height)
        if clipped is None:
            continue
        image_start, image_end = clipped
        world_clipped, clipped_valid = _project_points(np.asarray([image_start, image_end]), inverse)
        if not np.all(clipped_valid):
            continue
        full_length_px = float(np.linalg.norm(projected[1] - projected[0]))
        visible_length_px = float(np.linalg.norm(image_end - image_start))
        visible_lines.append({
            "semantic": semantic,
            "image_start": image_start.tolist(),
            "image_end": image_end.tolist(),
            "world_start": world_clipped[0].tolist(),
            "world_end": world_clipped[1].tolist(),
            "visible_fraction_of_segment": (
                float(min(1.0, visible_length_px / full_length_px))
                if full_length_px > 1e-9 else 0.0
            ),
        })

    visible_landmarks = []
    names = list(PITCH_LANDMARKS)
    world = np.asarray([PITCH_LANDMARKS[name] for name in names])
    image, valid = _project_points(world, h)
    for index, name in enumerate(names):
        if not valid[index]:
            continue
        x, y = map(float, image[index])
        if 0.0 <= x <= width and 0.0 <= y <= height:
            visible_landmarks.append({
                "semantic": name,
                "world": world[index].tolist(),
                "image": [x, y],
            })

    minimum = np.min(clipped_world, axis=0)
    maximum = np.max(clipped_world, axis=0)
    centroid = np.mean(clipped_world, axis=0)

    return {
        "status": "ok",
        "polygon_world": clipped_world.tolist(),
        "polygon_image": clipped_image.tolist(),
        "visible_area_m2": float(visible_area),
        "visible_fraction": float(max(0.0, min(1.0, visible_area / PITCH_AREA_M2))),
        "world_centroid": centroid.tolist(),
        "world_bbox": {
            "min_x": float(minimum[0]),
            "min_y": float(minimum[1]),
            "max_x": float(maximum[0]),
            "max_y": float(maximum[1]),
        },
        "visible_lines": visible_lines,
        "visible_landmarks": visible_landmarks,
    }
