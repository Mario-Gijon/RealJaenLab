"""Canonical football-pitch geometry for RealJaenLab FIELD calibration.

V1 adds explicit semantic calibration landmarks so the manual solver can mix:
- line annotations;
- automatically-derived line intersections;
- direct point annotations.

Coordinate convention
---------------------
The pitch is represented in metres.

- x = 0.0 is the left goal line.
- x = pitch length is the right goal line.
- y = 0.0 is touchline A.
- y = pitch width is touchline B.

This module contains geometry only.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Iterable


GOAL_WIDTH = 7.32
GOAL_AREA_DEPTH = 5.50
PENALTY_AREA_DEPTH = 16.50
PENALTY_SPOT_DISTANCE = 11.00
CENTER_CIRCLE_RADIUS = 9.15
PENALTY_ARC_RADIUS = 9.15
CORNER_ARC_RADIUS = 1.00

DEFAULT_PITCH_LENGTH = 105.0
DEFAULT_PITCH_WIDTH = 68.0


@dataclass(frozen=True, slots=True)
class Point2D:
    x: float
    y: float

    def as_list(self) -> list[float]:
        return [float(self.x), float(self.y)]


@dataclass(frozen=True, slots=True)
class LineSegment:
    name: str
    start: Point2D
    end: Point2D
    family: str
    label: str

    def homogeneous_line(self) -> tuple[float, float, float]:
        x1, y1 = self.start.x, self.start.y
        x2, y2 = self.end.x, self.end.y
        a = y1 - y2
        b = x2 - x1
        c = x1 * y2 - x2 * y1
        norm = math.hypot(a, b)
        if norm == 0:
            raise ValueError(f"Degenerate pitch segment: {self.name}")
        return a / norm, b / norm, c / norm

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "label": self.label,
            "family": self.family,
            "start": self.start.as_list(),
            "end": self.end.as_list(),
        }


@dataclass(frozen=True, slots=True)
class Polyline:
    name: str
    label: str
    points: tuple[Point2D, ...]
    closed: bool = False

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "label": self.label,
            "closed": self.closed,
            "points": [point.as_list() for point in self.points],
        }


@dataclass(frozen=True, slots=True)
class PitchPoint:
    name: str
    label: str
    point: Point2D
    category: str = "reference"

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "label": self.label,
            "category": self.category,
            "point": self.point.as_list(),
        }


class PitchModel:
    """Mathematical football-pitch model used by FIELD calibration."""

    def __init__(
        self,
        length: float = DEFAULT_PITCH_LENGTH,
        width: float = DEFAULT_PITCH_WIDTH,
    ) -> None:
        self.length = float(length)
        self.width = float(width)
        self._validate_dimensions()

        self._lines = self._build_lines()
        self._line_map = {line.name: line for line in self._lines}
        self._points = self._build_points()
        self._point_map = {point.name: point for point in self._points}
        self._curves = self._build_curves()

    def _validate_dimensions(self) -> None:
        if not math.isfinite(self.length) or not math.isfinite(self.width):
            raise ValueError("Pitch dimensions must be finite numbers.")
        if self.length <= 2 * PENALTY_AREA_DEPTH:
            raise ValueError(
                f"Pitch length must be greater than {2 * PENALTY_AREA_DEPTH:.1f} m."
            )
        minimum_width = GOAL_WIDTH + 2 * PENALTY_AREA_DEPTH
        if self.width < minimum_width:
            raise ValueError(
                f"Pitch width must be at least {minimum_width:.2f} m "
                "to contain the standard penalty areas."
            )

    @property
    def center_x(self) -> float:
        return self.length / 2.0

    @property
    def center_y(self) -> float:
        return self.width / 2.0

    @property
    def straight_lines(self) -> tuple[LineSegment, ...]:
        return self._lines

    @property
    def curves(self) -> tuple[Polyline, ...]:
        return self._curves

    @property
    def points(self) -> tuple[PitchPoint, ...]:
        return self._points

    @property
    def semantic_line_names(self) -> tuple[str, ...]:
        return tuple(line.name for line in self._lines)

    @property
    def semantic_point_names(self) -> tuple[str, ...]:
        return tuple(point.name for point in self._points)

    def line(self, name: str) -> LineSegment:
        try:
            return self._line_map[name]
        except KeyError as exc:
            raise KeyError(f"Unknown semantic pitch line {name!r}.") from exc

    def point(self, name: str) -> PitchPoint:
        try:
            return self._point_map[name]
        except KeyError as exc:
            raise KeyError(f"Unknown semantic pitch point {name!r}.") from exc

    def lines_by_family(self, family: str) -> tuple[LineSegment, ...]:
        if family not in {"x_constant", "y_constant"}:
            raise ValueError("Line family must be 'x_constant' or 'y_constant'.")
        return tuple(line for line in self._lines if line.family == family)

    def geometry_dict(self) -> dict:
        return {
            "coordinate_system": {
                "units": "metres",
                "origin": "left_goal_line_touchline_a",
                "x_axis": "left_goal_line_to_right_goal_line",
                "y_axis": "touchline_a_to_touchline_b",
            },
            "dimensions": {"length": self.length, "width": self.width},
            "constants": {
                "goal_width": GOAL_WIDTH,
                "goal_area_depth": GOAL_AREA_DEPTH,
                "penalty_area_depth": PENALTY_AREA_DEPTH,
                "penalty_spot_distance": PENALTY_SPOT_DISTANCE,
                "center_circle_radius": CENTER_CIRCLE_RADIUS,
                "penalty_arc_radius": PENALTY_ARC_RADIUS,
                "corner_arc_radius": CORNER_ARC_RADIUS,
            },
            "lines": [line.as_dict() for line in self._lines],
            "curves": [curve.as_dict() for curve in self._curves],
            "points": [point.as_dict() for point in self._points],
        }

    def _build_lines(self) -> tuple[LineSegment, ...]:
        cy = self.center_y
        penalty_half_width = GOAL_WIDTH / 2.0 + PENALTY_AREA_DEPTH
        penalty_y_a = cy - penalty_half_width
        penalty_y_b = cy + penalty_half_width
        goal_area_half_width = GOAL_WIDTH / 2.0 + GOAL_AREA_DEPTH
        goal_area_y_a = cy - goal_area_half_width
        goal_area_y_b = cy + goal_area_half_width
        left_penalty_x = PENALTY_AREA_DEPTH
        right_penalty_x = self.length - PENALTY_AREA_DEPTH
        left_goal_area_x = GOAL_AREA_DEPTH
        right_goal_area_x = self.length - GOAL_AREA_DEPTH

        return (
            LineSegment("touchline_a", Point2D(0, 0), Point2D(self.length, 0), "y_constant", "Línea de banda A"),
            LineSegment("touchline_b", Point2D(0, self.width), Point2D(self.length, self.width), "y_constant", "Línea de banda B"),
            LineSegment("goal_line_left", Point2D(0, 0), Point2D(0, self.width), "x_constant", "Línea de fondo izquierda"),
            LineSegment("goal_line_right", Point2D(self.length, 0), Point2D(self.length, self.width), "x_constant", "Línea de fondo derecha"),
            LineSegment("halfway_line", Point2D(self.center_x, 0), Point2D(self.center_x, self.width), "x_constant", "Línea de medio campo"),
            LineSegment("left_penalty_front", Point2D(left_penalty_x, penalty_y_a), Point2D(left_penalty_x, penalty_y_b), "x_constant", "Frontal del área izquierda"),
            LineSegment("left_penalty_side_a", Point2D(0, penalty_y_a), Point2D(left_penalty_x, penalty_y_a), "y_constant", "Lateral A del área izquierda"),
            LineSegment("left_penalty_side_b", Point2D(0, penalty_y_b), Point2D(left_penalty_x, penalty_y_b), "y_constant", "Lateral B del área izquierda"),
            LineSegment("right_penalty_front", Point2D(right_penalty_x, penalty_y_a), Point2D(right_penalty_x, penalty_y_b), "x_constant", "Frontal del área derecha"),
            LineSegment("right_penalty_side_a", Point2D(right_penalty_x, penalty_y_a), Point2D(self.length, penalty_y_a), "y_constant", "Lateral A del área derecha"),
            LineSegment("right_penalty_side_b", Point2D(right_penalty_x, penalty_y_b), Point2D(self.length, penalty_y_b), "y_constant", "Lateral B del área derecha"),
            LineSegment("left_goal_area_front", Point2D(left_goal_area_x, goal_area_y_a), Point2D(left_goal_area_x, goal_area_y_b), "x_constant", "Frontal del área pequeña izquierda"),
            LineSegment("left_goal_area_side_a", Point2D(0, goal_area_y_a), Point2D(left_goal_area_x, goal_area_y_a), "y_constant", "Lateral A del área pequeña izquierda"),
            LineSegment("left_goal_area_side_b", Point2D(0, goal_area_y_b), Point2D(left_goal_area_x, goal_area_y_b), "y_constant", "Lateral B del área pequeña izquierda"),
            LineSegment("right_goal_area_front", Point2D(right_goal_area_x, goal_area_y_a), Point2D(right_goal_area_x, goal_area_y_b), "x_constant", "Frontal del área pequeña derecha"),
            LineSegment("right_goal_area_side_a", Point2D(right_goal_area_x, goal_area_y_a), Point2D(self.length, goal_area_y_a), "y_constant", "Lateral A del área pequeña derecha"),
            LineSegment("right_goal_area_side_b", Point2D(right_goal_area_x, goal_area_y_b), Point2D(self.length, goal_area_y_b), "y_constant", "Lateral B del área pequeña derecha"),
        )

    def _build_points(self) -> tuple[PitchPoint, ...]:
        cy = self.center_y
        penalty_half_width = GOAL_WIDTH / 2.0 + PENALTY_AREA_DEPTH
        py_a = cy - penalty_half_width
        py_b = cy + penalty_half_width
        goal_area_half_width = GOAL_WIDTH / 2.0 + GOAL_AREA_DEPTH
        gy_a = cy - goal_area_half_width
        gy_b = cy + goal_area_half_width
        lpx = PENALTY_AREA_DEPTH
        rpx = self.length - PENALTY_AREA_DEPTH
        lgx = GOAL_AREA_DEPTH
        rgx = self.length - GOAL_AREA_DEPTH

        points = [
            PitchPoint("corner_left_a", "Esquina izquierda A", Point2D(0, 0), "intersection"),
            PitchPoint("corner_left_b", "Esquina izquierda B", Point2D(0, self.width), "intersection"),
            PitchPoint("corner_right_a", "Esquina derecha A", Point2D(self.length, 0), "intersection"),
            PitchPoint("corner_right_b", "Esquina derecha B", Point2D(self.length, self.width), "intersection"),
            PitchPoint("halfway_touchline_a", "Medio campo con banda A", Point2D(self.center_x, 0), "intersection"),
            PitchPoint("halfway_touchline_b", "Medio campo con banda B", Point2D(self.center_x, self.width), "intersection"),
            PitchPoint("left_penalty_front_a", "Área izquierda: frontal con lateral A", Point2D(lpx, py_a), "intersection"),
            PitchPoint("left_penalty_front_b", "Área izquierda: frontal con lateral B", Point2D(lpx, py_b), "intersection"),
            PitchPoint("left_penalty_goal_a", "Área izquierda: fondo con lateral A", Point2D(0, py_a), "intersection"),
            PitchPoint("left_penalty_goal_b", "Área izquierda: fondo con lateral B", Point2D(0, py_b), "intersection"),
            PitchPoint("right_penalty_front_a", "Área derecha: frontal con lateral A", Point2D(rpx, py_a), "intersection"),
            PitchPoint("right_penalty_front_b", "Área derecha: frontal con lateral B", Point2D(rpx, py_b), "intersection"),
            PitchPoint("right_penalty_goal_a", "Área derecha: fondo con lateral A", Point2D(self.length, py_a), "intersection"),
            PitchPoint("right_penalty_goal_b", "Área derecha: fondo con lateral B", Point2D(self.length, py_b), "intersection"),
            PitchPoint("left_goal_area_front_a", "Área pequeña izquierda: frontal con lateral A", Point2D(lgx, gy_a), "intersection"),
            PitchPoint("left_goal_area_front_b", "Área pequeña izquierda: frontal con lateral B", Point2D(lgx, gy_b), "intersection"),
            PitchPoint("left_goal_area_goal_a", "Área pequeña izquierda: fondo con lateral A", Point2D(0, gy_a), "intersection"),
            PitchPoint("left_goal_area_goal_b", "Área pequeña izquierda: fondo con lateral B", Point2D(0, gy_b), "intersection"),
            PitchPoint("right_goal_area_front_a", "Área pequeña derecha: frontal con lateral A", Point2D(rgx, gy_a), "intersection"),
            PitchPoint("right_goal_area_front_b", "Área pequeña derecha: frontal con lateral B", Point2D(rgx, gy_b), "intersection"),
            PitchPoint("right_goal_area_goal_a", "Área pequeña derecha: fondo con lateral A", Point2D(self.length, gy_a), "intersection"),
            PitchPoint("right_goal_area_goal_b", "Área pequeña derecha: fondo con lateral B", Point2D(self.length, gy_b), "intersection"),
            PitchPoint("center_spot", "Punto central", Point2D(self.center_x, cy), "mark"),
            PitchPoint("left_penalty_spot", "Punto de penalti izquierdo", Point2D(PENALTY_SPOT_DISTANCE, cy), "mark"),
            PitchPoint("right_penalty_spot", "Punto de penalti derecho", Point2D(self.length - PENALTY_SPOT_DISTANCE, cy), "mark"),
        ]
        return tuple(points)

    def _build_curves(self) -> tuple[Polyline, ...]:
        return (
            Polyline("center_circle", "Círculo central", tuple(_sample_circle(Point2D(self.center_x, self.center_y), CENTER_CIRCLE_RADIUS, 128)), True),
            Polyline("left_penalty_arc", "Arco de penalti izquierdo", tuple(self._sample_left_penalty_arc(48))),
            Polyline("right_penalty_arc", "Arco de penalti derecho", tuple(self._sample_right_penalty_arc(48))),
            Polyline("corner_arc_left_a", "Arco de esquina izquierda A", tuple(_sample_arc(Point2D(0, 0), CORNER_ARC_RADIUS, 0, math.pi / 2, 24))),
            Polyline("corner_arc_left_b", "Arco de esquina izquierda B", tuple(_sample_arc(Point2D(0, self.width), CORNER_ARC_RADIUS, -math.pi / 2, 0, 24))),
            Polyline("corner_arc_right_a", "Arco de esquina derecha A", tuple(_sample_arc(Point2D(self.length, 0), CORNER_ARC_RADIUS, math.pi / 2, math.pi, 24))),
            Polyline("corner_arc_right_b", "Arco de esquina derecha B", tuple(_sample_arc(Point2D(self.length, self.width), CORNER_ARC_RADIUS, math.pi, 3 * math.pi / 2, 24))),
        )

    def _sample_left_penalty_arc(self, samples: int) -> Iterable[Point2D]:
        x_offset = PENALTY_AREA_DEPTH - PENALTY_SPOT_DISTANCE
        half_angle = math.acos(x_offset / PENALTY_ARC_RADIUS)
        return _sample_arc(Point2D(PENALTY_SPOT_DISTANCE, self.center_y), PENALTY_ARC_RADIUS, -half_angle, half_angle, samples)

    def _sample_right_penalty_arc(self, samples: int) -> Iterable[Point2D]:
        x_offset = PENALTY_AREA_DEPTH - PENALTY_SPOT_DISTANCE
        half_angle = math.acos(x_offset / PENALTY_ARC_RADIUS)
        return _sample_arc(Point2D(self.length - PENALTY_SPOT_DISTANCE, self.center_y), PENALTY_ARC_RADIUS, math.pi - half_angle, math.pi + half_angle, samples)


def _sample_circle(center: Point2D, radius: float, samples: int) -> Iterable[Point2D]:
    if samples < 8:
        raise ValueError("A circle requires at least 8 samples.")
    for i in range(samples):
        a = 2 * math.pi * i / samples
        yield Point2D(center.x + radius * math.cos(a), center.y + radius * math.sin(a))


def _sample_arc(center: Point2D, radius: float, start_angle: float, end_angle: float, samples: int) -> Iterable[Point2D]:
    if samples < 2:
        raise ValueError("An arc requires at least 2 samples.")
    step = (end_angle - start_angle) / (samples - 1)
    for i in range(samples):
        a = start_angle + i * step
        yield Point2D(center.x + radius * math.cos(a), center.y + radius * math.sin(a))


def default_pitch() -> PitchModel:
    return PitchModel()


def _self_check() -> None:
    pitch = default_pitch()
    assert len(pitch.straight_lines) == 17
    assert len(pitch.points) == 25
    assert pitch.point("halfway_touchline_a").point == Point2D(52.5, 0.0)
    assert pitch.point("right_penalty_spot").point == Point2D(94.0, 34.0)
    assert pitch.point("left_penalty_front_a").point == Point2D(16.5, 13.84)
    print("PitchModel V1 self-check: OK")
    print(f"Dimensions: {pitch.length:.1f} x {pitch.width:.1f} m")
    print(f"Semantic lines: {len(pitch.straight_lines)}")
    print(f"Semantic calibration points: {len(pitch.points)}")
    print(json.dumps(pitch.geometry_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _self_check()
