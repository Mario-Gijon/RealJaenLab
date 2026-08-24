"""Parametric, planar football-pitch geometry for RealJaen Supervisor.

The pitch is centred at ``(0, 0)``.  ``x`` runs along its length and ``y``
runs across its width.  This module deliberately contains geometry only: it
does not estimate dimensions or solve a camera calibration.
"""

from __future__ import annotations

import math
from typing import Any


DEFAULT_PITCH_LENGTH_M = 105.0
DEFAULT_PITCH_WIDTH_M = 68.0

GOAL_WIDTH_M = 7.32
GOAL_HALF_WIDTH_M = GOAL_WIDTH_M / 2.0
PENALTY_AREA_DEPTH_M = 16.5
GOAL_AREA_DEPTH_M = 5.5
PENALTY_SPOT_DISTANCE_M = 11.0
CENTRE_CIRCLE_RADIUS_M = 9.15


def _point(x: float, y: float) -> dict[str, float]:
    return {"x": float(x), "y": float(y)}


def _line(identifier: str, label: str, start: tuple[float, float], end: tuple[float, float]) -> dict[str, Any]:
    return {
        "id": identifier,
        "label": label,
        "world": {"start": _point(*start), "end": _point(*end)},
    }


def _landmark(identifier: str, label: str, x: float, y: float) -> dict[str, Any]:
    return {"id": identifier, "label": label, "world": _point(x, y)}


def build_pitch_model(length_m: float = DEFAULT_PITCH_LENGTH_M, width_m: float = DEFAULT_PITCH_WIDTH_M) -> dict[str, Any]:
    """Build all planar pitch geometry for the supplied total dimensions.

    Goal, penalty-area, goal-area, spot and circle dimensions are fixed in
    metres.  Only the complete pitch length and width vary.
    """
    length_m = float(length_m)
    width_m = float(width_m)
    minimum_length = 2.0 * PENALTY_AREA_DEPTH_M
    minimum_width = 2.0 * (GOAL_HALF_WIDTH_M + PENALTY_AREA_DEPTH_M)
    if not math.isfinite(length_m) or length_m <= minimum_length:
        raise ValueError(f"length_m must be finite and greater than {minimum_length}")
    if not math.isfinite(width_m) or width_m <= minimum_width:
        raise ValueError(f"width_m must be finite and greater than {minimum_width}")

    half_length = length_m / 2.0
    half_width = width_m / 2.0
    penalty_y = GOAL_HALF_WIDTH_M + PENALTY_AREA_DEPTH_M
    goal_area_y = GOAL_HALF_WIDTH_M + GOAL_AREA_DEPTH_M
    left_penalty_front = -half_length + PENALTY_AREA_DEPTH_M
    right_penalty_front = half_length - PENALTY_AREA_DEPTH_M
    left_goal_front = -half_length + GOAL_AREA_DEPTH_M
    right_goal_front = half_length - GOAL_AREA_DEPTH_M
    left_penalty_spot = -half_length + PENALTY_SPOT_DISTANCE_M
    right_penalty_spot = half_length - PENALTY_SPOT_DISTANCE_M
    penalty_arc_angle = math.acos(GOAL_AREA_DEPTH_M / CENTRE_CIRCLE_RADIUS_M)

    landmarks = [
        _landmark("top_left_corner", "Esquina superior izquierda", -half_length, -half_width),
        _landmark("top_right_corner", "Esquina superior derecha", half_length, -half_width),
        _landmark("bottom_right_corner", "Esquina inferior derecha", half_length, half_width),
        _landmark("bottom_left_corner", "Esquina inferior izquierda", -half_length, half_width),
        _landmark("halfway_top", "Mitad de campo / banda superior", 0.0, -half_width),
        _landmark("halfway_bottom", "Mitad de campo / banda inferior", 0.0, half_width),
        _landmark("centre_mark", "Punto central", 0.0, 0.0),
        _landmark("left_penalty_top", "Esquina superior del área izquierda", left_penalty_front, -penalty_y),
        _landmark("left_penalty_bottom", "Esquina inferior del área izquierda", left_penalty_front, penalty_y),
        _landmark("right_penalty_top", "Esquina superior del área derecha", right_penalty_front, -penalty_y),
        _landmark("right_penalty_bottom", "Esquina inferior del área derecha", right_penalty_front, penalty_y),
        _landmark("left_goal_top", "Esquina superior del área pequeña izquierda", left_goal_front, -goal_area_y),
        _landmark("left_goal_bottom", "Esquina inferior del área pequeña izquierda", left_goal_front, goal_area_y),
        _landmark("right_goal_top", "Esquina superior del área pequeña derecha", right_goal_front, -goal_area_y),
        _landmark("right_goal_bottom", "Esquina inferior del área pequeña derecha", right_goal_front, goal_area_y),
        _landmark("left_penalty_spot", "Punto de penalti izquierdo", left_penalty_spot, 0.0),
        _landmark("right_penalty_spot", "Punto de penalti derecho", right_penalty_spot, 0.0),
        _landmark("left_goal_post_top", "Base del poste superior de la portería izquierda (arriba en el diagrama)", -half_length, -GOAL_HALF_WIDTH_M),
        _landmark("left_goal_post_bottom", "Base del poste inferior de la portería izquierda (abajo en el diagrama)", -half_length, GOAL_HALF_WIDTH_M),
        _landmark("right_goal_post_top", "Base del poste superior de la portería derecha (arriba en el diagrama)", half_length, -GOAL_HALF_WIDTH_M),
        _landmark("right_goal_post_bottom", "Base del poste inferior de la portería derecha (abajo en el diagrama)", half_length, GOAL_HALF_WIDTH_M),
        _landmark("left_penalty_goal_line_top", "Intersección área de penalti / línea de gol izquierda, superior (diagrama)", -half_length, -penalty_y),
        _landmark("left_penalty_goal_line_bottom", "Intersección área de penalti / línea de gol izquierda, inferior (diagrama)", -half_length, penalty_y),
        _landmark("right_penalty_goal_line_top", "Intersección área de penalti / línea de gol derecha, superior (diagrama)", half_length, -penalty_y),
        _landmark("right_penalty_goal_line_bottom", "Intersección área de penalti / línea de gol derecha, inferior (diagrama)", half_length, penalty_y),
        _landmark("left_goal_area_goal_line_top", "Intersección área pequeña / línea de gol izquierda, superior (diagrama)", -half_length, -goal_area_y),
        _landmark("left_goal_area_goal_line_bottom", "Intersección área pequeña / línea de gol izquierda, inferior (diagrama)", -half_length, goal_area_y),
        _landmark("right_goal_area_goal_line_top", "Intersección área pequeña / línea de gol derecha, superior (diagrama)", half_length, -goal_area_y),
        _landmark("right_goal_area_goal_line_bottom", "Intersección área pequeña / línea de gol derecha, inferior (diagrama)", half_length, goal_area_y),
        _landmark("centre_circle_halfway_top", "Intersección círculo central / medio campo, superior (diagrama)", 0.0, -CENTRE_CIRCLE_RADIUS_M),
        _landmark("centre_circle_halfway_bottom", "Intersección círculo central / medio campo, inferior (diagrama)", 0.0, CENTRE_CIRCLE_RADIUS_M),
    ]

    lines = [
        _line("halfway_line", "Línea de medio campo", (0.0, -half_width), (0.0, half_width)),
        _line("top_touchline", "Banda superior", (-half_length, -half_width), (half_length, -half_width)),
        _line("bottom_touchline", "Banda inferior", (-half_length, half_width), (half_length, half_width)),
        _line("left_goal_line", "Línea de fondo izquierda", (-half_length, -half_width), (-half_length, half_width)),
        _line("right_goal_line", "Línea de fondo derecha", (half_length, -half_width), (half_length, half_width)),
        _line("left_penalty_top", "Área izquierda — línea superior", (-half_length, -penalty_y), (left_penalty_front, -penalty_y)),
        _line("left_penalty_front", "Área izquierda — línea frontal", (left_penalty_front, -penalty_y), (left_penalty_front, penalty_y)),
        _line("left_penalty_bottom", "Área izquierda — línea inferior", (left_penalty_front, penalty_y), (-half_length, penalty_y)),
        _line("right_penalty_top", "Área derecha — línea superior", (half_length, -penalty_y), (right_penalty_front, -penalty_y)),
        _line("right_penalty_front", "Área derecha — línea frontal", (right_penalty_front, -penalty_y), (right_penalty_front, penalty_y)),
        _line("right_penalty_bottom", "Área derecha — línea inferior", (right_penalty_front, penalty_y), (half_length, penalty_y)),
        _line("left_goal_top", "Área pequeña izquierda — línea superior", (-half_length, -goal_area_y), (left_goal_front, -goal_area_y)),
        _line("left_goal_front", "Área pequeña izquierda — línea frontal", (left_goal_front, -goal_area_y), (left_goal_front, goal_area_y)),
        _line("left_goal_bottom", "Área pequeña izquierda — línea inferior", (left_goal_front, goal_area_y), (-half_length, goal_area_y)),
        _line("right_goal_top", "Área pequeña derecha — línea superior", (half_length, -goal_area_y), (right_goal_front, -goal_area_y)),
        _line("right_goal_front", "Área pequeña derecha — línea frontal", (right_goal_front, -goal_area_y), (right_goal_front, goal_area_y)),
        _line("right_goal_bottom", "Área pequeña derecha — línea inferior", (right_goal_front, goal_area_y), (half_length, goal_area_y)),
    ]

    arcs = [
        {"id": "centre_circle", "label": "Círculo central", "world": {"center": _point(0.0, 0.0), "radius": CENTRE_CIRCLE_RADIUS_M}},
        {
            "id": "left_penalty_arc", "label": "Arco de penalti izquierdo (exterior del área)",
            "world": {"center": _point(left_penalty_spot, 0.0), "radius": CENTRE_CIRCLE_RADIUS_M,
                      "startAngle": -penalty_arc_angle, "endAngle": penalty_arc_angle},
        },
        {
            "id": "right_penalty_arc", "label": "Arco de penalti derecho (exterior del área)",
            "world": {"center": _point(right_penalty_spot, 0.0), "radius": CENTRE_CIRCLE_RADIUS_M,
                      "startAngle": math.pi - penalty_arc_angle, "endAngle": math.pi + penalty_arc_angle},
        },
    ]

    polygon = [
        _point(-half_length, -half_width), _point(half_length, -half_width),
        _point(half_length, half_width), _point(-half_length, half_width),
    ]
    segments = [
        [*polygon, _point(-half_length, -half_width)],
        [_point(0.0, -half_width), _point(0.0, half_width)],
        [_point(-half_length, -penalty_y), _point(left_penalty_front, -penalty_y), _point(left_penalty_front, penalty_y), _point(-half_length, penalty_y)],
        [_point(half_length, -penalty_y), _point(right_penalty_front, -penalty_y), _point(right_penalty_front, penalty_y), _point(half_length, penalty_y)],
        [_point(-half_length, -goal_area_y), _point(left_goal_front, -goal_area_y), _point(left_goal_front, goal_area_y), _point(-half_length, goal_area_y)],
        [_point(half_length, -goal_area_y), _point(right_goal_front, -goal_area_y), _point(right_goal_front, goal_area_y), _point(half_length, goal_area_y)],
    ]

    return {
        "length_m": length_m,
        "width_m": width_m,
        "half_length_m": half_length,
        "half_width_m": half_width,
        "pitch_area_m2": length_m * width_m,
        "pitch_polygon": polygon,
        "landmarks": landmarks,
        "calibration_lines": lines,
        "arcs": arcs,
        "segments": segments,
        "mark_points": [_point(0.0, 0.0), _point(left_penalty_spot, 0.0), _point(right_penalty_spot, 0.0)],
        "constants": {
            "goal_width_m": GOAL_WIDTH_M,
            "goal_half_width_m": GOAL_HALF_WIDTH_M,
            "penalty_area_depth_m": PENALTY_AREA_DEPTH_M,
            "goal_area_depth_m": GOAL_AREA_DEPTH_M,
            "penalty_spot_distance_m": PENALTY_SPOT_DISTANCE_M,
            "centre_circle_radius_m": CENTRE_CIRCLE_RADIUS_M,
            "penalty_arc_angle_rad": penalty_arc_angle,
        },
    }
