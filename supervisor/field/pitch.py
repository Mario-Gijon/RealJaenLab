"""Canonical football-pitch geometry for RealJaenLab FIELD calibration.

Coordinate convention
---------------------
The pitch is represented in metres.

- x = 0.0 is the left goal line.
- x = pitch length is the right goal line.
- y = 0.0 is touchline A.
- y = pitch width is touchline B.

This module contains geometry only. It does not depend on FastAPI, NumPy,
OpenCV, browser coordinates, or any previous RealJaenLab calibration code.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Iterable


# Standard football-marking dimensions, in metres.
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
    """Finite semantic pitch line.

    ``family`` identifies the two orthogonal direction families used later by
    the homography solver:

    - ``x_constant``: goal lines, halfway line, penalty fronts, goal-area fronts.
    - ``y_constant``: touchlines and the side lines of the boxes.
    """

    name: str
    start: Point2D
    end: Point2D
    family: str
    label: str

    def homogeneous_line(self) -> tuple[float, float, float]:
        """Return normalized ax + by + c = 0 coefficients."""
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

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "label": self.label,
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

    def line(self, name: str) -> LineSegment:
        try:
            return self._line_map[name]
        except KeyError as exc:
            available = ", ".join(self.semantic_line_names)
            raise KeyError(
                f"Unknown semantic pitch line {name!r}. Available: {available}"
            ) from exc

    def lines_by_family(self, family: str) -> tuple[LineSegment, ...]:
        if family not in {"x_constant", "y_constant"}:
            raise ValueError("Line family must be 'x_constant' or 'y_constant'.")
        return tuple(line for line in self._lines if line.family == family)

    def geometry_dict(self) -> dict:
        """Return JSON-friendly canonical geometry for APIs/frontends."""
        return {
            "coordinate_system": {
                "units": "metres",
                "origin": "left_goal_line_touchline_a",
                "x_axis": "left_goal_line_to_right_goal_line",
                "y_axis": "touchline_a_to_touchline_b",
            },
            "dimensions": {
                "length": self.length,
                "width": self.width,
            },
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
            LineSegment(
                "touchline_a",
                Point2D(0.0, 0.0),
                Point2D(self.length, 0.0),
                "y_constant",
                "Línea de banda A",
            ),
            LineSegment(
                "touchline_b",
                Point2D(0.0, self.width),
                Point2D(self.length, self.width),
                "y_constant",
                "Línea de banda B",
            ),
            LineSegment(
                "goal_line_left",
                Point2D(0.0, 0.0),
                Point2D(0.0, self.width),
                "x_constant",
                "Línea de fondo izquierda",
            ),
            LineSegment(
                "goal_line_right",
                Point2D(self.length, 0.0),
                Point2D(self.length, self.width),
                "x_constant",
                "Línea de fondo derecha",
            ),
            LineSegment(
                "halfway_line",
                Point2D(self.center_x, 0.0),
                Point2D(self.center_x, self.width),
                "x_constant",
                "Línea de medio campo",
            ),
            LineSegment(
                "left_penalty_front",
                Point2D(left_penalty_x, penalty_y_a),
                Point2D(left_penalty_x, penalty_y_b),
                "x_constant",
                "Frontal del área izquierda",
            ),
            LineSegment(
                "left_penalty_side_a",
                Point2D(0.0, penalty_y_a),
                Point2D(left_penalty_x, penalty_y_a),
                "y_constant",
                "Lateral A del área izquierda",
            ),
            LineSegment(
                "left_penalty_side_b",
                Point2D(0.0, penalty_y_b),
                Point2D(left_penalty_x, penalty_y_b),
                "y_constant",
                "Lateral B del área izquierda",
            ),
            LineSegment(
                "right_penalty_front",
                Point2D(right_penalty_x, penalty_y_a),
                Point2D(right_penalty_x, penalty_y_b),
                "x_constant",
                "Frontal del área derecha",
            ),
            LineSegment(
                "right_penalty_side_a",
                Point2D(right_penalty_x, penalty_y_a),
                Point2D(self.length, penalty_y_a),
                "y_constant",
                "Lateral A del área derecha",
            ),
            LineSegment(
                "right_penalty_side_b",
                Point2D(right_penalty_x, penalty_y_b),
                Point2D(self.length, penalty_y_b),
                "y_constant",
                "Lateral B del área derecha",
            ),
            LineSegment(
                "left_goal_area_front",
                Point2D(left_goal_area_x, goal_area_y_a),
                Point2D(left_goal_area_x, goal_area_y_b),
                "x_constant",
                "Frontal del área pequeña izquierda",
            ),
            LineSegment(
                "left_goal_area_side_a",
                Point2D(0.0, goal_area_y_a),
                Point2D(left_goal_area_x, goal_area_y_a),
                "y_constant",
                "Lateral A del área pequeña izquierda",
            ),
            LineSegment(
                "left_goal_area_side_b",
                Point2D(0.0, goal_area_y_b),
                Point2D(left_goal_area_x, goal_area_y_b),
                "y_constant",
                "Lateral B del área pequeña izquierda",
            ),
            LineSegment(
                "right_goal_area_front",
                Point2D(right_goal_area_x, goal_area_y_a),
                Point2D(right_goal_area_x, goal_area_y_b),
                "x_constant",
                "Frontal del área pequeña derecha",
            ),
            LineSegment(
                "right_goal_area_side_a",
                Point2D(right_goal_area_x, goal_area_y_a),
                Point2D(self.length, goal_area_y_a),
                "y_constant",
                "Lateral A del área pequeña derecha",
            ),
            LineSegment(
                "right_goal_area_side_b",
                Point2D(right_goal_area_x, goal_area_y_b),
                Point2D(self.length, goal_area_y_b),
                "y_constant",
                "Lateral B del área pequeña derecha",
            ),
        )

    def _build_points(self) -> tuple[PitchPoint, ...]:
        return (
            PitchPoint(
                "center_spot",
                "Punto central",
                Point2D(self.center_x, self.center_y),
            ),
            PitchPoint(
                "left_penalty_spot",
                "Punto de penalti izquierdo",
                Point2D(PENALTY_SPOT_DISTANCE, self.center_y),
            ),
            PitchPoint(
                "right_penalty_spot",
                "Punto de penalti derecho",
                Point2D(self.length - PENALTY_SPOT_DISTANCE, self.center_y),
            ),
        )

    def _build_curves(self) -> tuple[Polyline, ...]:
        return (
            Polyline(
                "center_circle",
                "Círculo central",
                tuple(
                    _sample_circle(
                        Point2D(self.center_x, self.center_y),
                        CENTER_CIRCLE_RADIUS,
                        samples=128,
                    )
                ),
                closed=True,
            ),
            Polyline(
                "left_penalty_arc",
                "Arco de penalti izquierdo",
                tuple(self._sample_left_penalty_arc(samples=48)),
            ),
            Polyline(
                "right_penalty_arc",
                "Arco de penalti derecho",
                tuple(self._sample_right_penalty_arc(samples=48)),
            ),
            Polyline(
                "corner_arc_left_a",
                "Arco de esquina izquierda A",
                tuple(
                    _sample_arc(
                        center=Point2D(0.0, 0.0),
                        radius=CORNER_ARC_RADIUS,
                        start_angle=0.0,
                        end_angle=math.pi / 2.0,
                        samples=24,
                    )
                ),
            ),
            Polyline(
                "corner_arc_left_b",
                "Arco de esquina izquierda B",
                tuple(
                    _sample_arc(
                        center=Point2D(0.0, self.width),
                        radius=CORNER_ARC_RADIUS,
                        start_angle=-math.pi / 2.0,
                        end_angle=0.0,
                        samples=24,
                    )
                ),
            ),
            Polyline(
                "corner_arc_right_a",
                "Arco de esquina derecha A",
                tuple(
                    _sample_arc(
                        center=Point2D(self.length, 0.0),
                        radius=CORNER_ARC_RADIUS,
                        start_angle=math.pi / 2.0,
                        end_angle=math.pi,
                        samples=24,
                    )
                ),
            ),
            Polyline(
                "corner_arc_right_b",
                "Arco de esquina derecha B",
                tuple(
                    _sample_arc(
                        center=Point2D(self.length, self.width),
                        radius=CORNER_ARC_RADIUS,
                        start_angle=math.pi,
                        end_angle=3.0 * math.pi / 2.0,
                        samples=24,
                    )
                ),
            ),
        )

    def _sample_left_penalty_arc(self, samples: int) -> Iterable[Point2D]:
        # Only the part of the 9.15 m circle lying outside the penalty area.
        x_offset_to_front = PENALTY_AREA_DEPTH - PENALTY_SPOT_DISTANCE
        half_angle = math.acos(x_offset_to_front / PENALTY_ARC_RADIUS)
        return _sample_arc(
            center=Point2D(PENALTY_SPOT_DISTANCE, self.center_y),
            radius=PENALTY_ARC_RADIUS,
            start_angle=-half_angle,
            end_angle=half_angle,
            samples=samples,
        )

    def _sample_right_penalty_arc(self, samples: int) -> Iterable[Point2D]:
        x_offset_to_front = PENALTY_AREA_DEPTH - PENALTY_SPOT_DISTANCE
        half_angle = math.acos(x_offset_to_front / PENALTY_ARC_RADIUS)
        return _sample_arc(
            center=Point2D(self.length - PENALTY_SPOT_DISTANCE, self.center_y),
            radius=PENALTY_ARC_RADIUS,
            start_angle=math.pi - half_angle,
            end_angle=math.pi + half_angle,
            samples=samples,
        )


def _sample_circle(
    center: Point2D,
    radius: float,
    samples: int,
) -> Iterable[Point2D]:
    if samples < 8:
        raise ValueError("A circle requires at least 8 samples.")
    for index in range(samples):
        angle = 2.0 * math.pi * index / samples
        yield Point2D(
            center.x + radius * math.cos(angle),
            center.y + radius * math.sin(angle),
        )


def _sample_arc(
    center: Point2D,
    radius: float,
    start_angle: float,
    end_angle: float,
    samples: int,
) -> Iterable[Point2D]:
    if samples < 2:
        raise ValueError("An arc requires at least 2 samples.")
    step = (end_angle - start_angle) / (samples - 1)
    for index in range(samples):
        angle = start_angle + step * index
        yield Point2D(
            center.x + radius * math.cos(angle),
            center.y + radius * math.sin(angle),
        )


def default_pitch() -> PitchModel:
    return PitchModel(DEFAULT_PITCH_LENGTH, DEFAULT_PITCH_WIDTH)


def _self_check() -> None:
    """Small dependency-free diagnostic, not an automated test suite."""
    pitch = default_pitch()

    assert len(pitch.straight_lines) == 17
    assert pitch.line("halfway_line").start.x == 52.5
    assert pitch.line("touchline_a").start == Point2D(0.0, 0.0)
    assert pitch.line("touchline_b").end == Point2D(105.0, 68.0)

    left_front = pitch.line("left_penalty_front")
    assert math.isclose(left_front.start.x, 16.5)
    assert math.isclose(left_front.start.y, 13.84)
    assert math.isclose(left_front.end.y, 54.16)

    right_front = pitch.line("right_penalty_front")
    assert math.isclose(right_front.start.x, 88.5)

    geometry = pitch.geometry_dict()
    assert geometry["dimensions"] == {"length": 105.0, "width": 68.0}

    print("PitchModel self-check: OK")
    print(f"Dimensions: {pitch.length:.1f} x {pitch.width:.1f} m")
    print(f"Semantic straight lines: {len(pitch.straight_lines)}")
    print(f"Curves: {len(pitch.curves)}")
    print(f"Reference points: {len(pitch.points)}")
    print()
    print(json.dumps(geometry, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _self_check()
