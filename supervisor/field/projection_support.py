"""Evidence-supported pitch projection for RealJaenLab FIELD.

This module is intentionally separate from homography solving.

The homography answers:
    "Where would every canonical pitch element project in this frame?"

Projection support answers a different question:
    "Which part of that projection is currently supported by the human evidence
     and by the image footprint?"

Why this exists
---------------
A valid homography can still extrapolate poorly far away from the annotated
region. Drawing the entire canonical pitch may therefore place mathematically
projected lines over stands/sky even when the local calibration on the visible
grass is good.

This module DOES NOT hide a bad homography and DOES NOT alter H.

It computes:

1. the part of the canonical pitch whose plane falls inside the current image;
2. the part of that visible pitch that is supported by the current human
   line/point annotations;
3. the corresponding support polygon in image coordinates.

The TEST UI can then clip the cyan overlay to this support polygon while keeping
the full homography available for diagnostics and future FIELD state work.

Important line semantics
------------------------
A human line annotation contains two arbitrary visible image points on a known
semantic pitch line. Those clicks are back-projected with H^-1, then projected
onto the finite canonical semantic line segment. This estimates which part of
the canonical line was actually observed.

No artificial line intersections are created.
No player/person information is used.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Sequence

import numpy as np

from supervisor.field.pitch import (
    DEFAULT_PITCH_LENGTH,
    DEFAULT_PITCH_WIDTH,
    PitchModel,
    Point2D,
)


EPS = 1e-10
DEFAULT_SUPPORT_MARGIN_M = 6.0
DEFAULT_SUPPORT_EXPANSION_SAMPLES = 12


class ProjectionSupportError(ValueError):
    """Invalid input or projective geometry for support estimation."""


@dataclass(frozen=True, slots=True)
class SupportResult:
    video_width: float
    video_height: float
    pitch_length: float
    pitch_width: float
    visible_pitch_polygon: tuple[Point2D, ...]
    support_pitch_polygon: tuple[Point2D, ...]
    visible_image_polygon: tuple[tuple[float, float], ...]
    support_image_polygon: tuple[tuple[float, float], ...]
    evidence_pitch_points: tuple[Point2D, ...]
    point_evidence_count: int
    line_evidence_count: int
    support_margin_m: float
    visible_pitch_area_m2: float
    support_pitch_area_m2: float
    projective_valid: bool
    warnings: tuple[str, ...]

    @property
    def support_ratio_of_visible(self) -> float:
        if self.visible_pitch_area_m2 <= EPS:
            return 0.0
        return min(
            1.0,
            max(
                0.0,
                self.support_pitch_area_m2 / self.visible_pitch_area_m2,
            ),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "video_width": float(self.video_width),
            "video_height": float(self.video_height),
            "pitch_length": float(self.pitch_length),
            "pitch_width": float(self.pitch_width),
            "visible_pitch_polygon": [
                point.as_list()
                for point in self.visible_pitch_polygon
            ],
            "support_pitch_polygon": [
                point.as_list()
                for point in self.support_pitch_polygon
            ],
            "visible_image_polygon": [
                [float(x), float(y)]
                for x, y in self.visible_image_polygon
            ],
            "support_image_polygon": [
                [float(x), float(y)]
                for x, y in self.support_image_polygon
            ],
            "evidence_pitch_points": [
                point.as_list()
                for point in self.evidence_pitch_points
            ],
            "point_evidence_count": int(self.point_evidence_count),
            "line_evidence_count": int(self.line_evidence_count),
            "support_margin_m": float(self.support_margin_m),
            "visible_pitch_area_m2": float(self.visible_pitch_area_m2),
            "support_pitch_area_m2": float(self.support_pitch_area_m2),
            "support_ratio_of_visible": float(
                self.support_ratio_of_visible
            ),
            "projective_valid": bool(self.projective_valid),
            "warnings": list(self.warnings),
        }


def _as_homography(
    value: Sequence[Sequence[float]] | np.ndarray,
    label: str,
) -> np.ndarray:
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (3, 3):
        raise ProjectionSupportError(
            f"{label} must be a 3x3 matrix."
        )
    if not np.all(np.isfinite(matrix)):
        raise ProjectionSupportError(
            f"{label} contains non-finite values."
        )
    if abs(float(np.linalg.det(matrix))) < EPS:
        raise ProjectionSupportError(
            f"{label} is singular."
        )
    return matrix


def _project_xy(
    matrix: np.ndarray,
    x: float,
    y: float,
) -> tuple[float, float]:
    projected = matrix @ np.array(
        [float(x), float(y), 1.0],
        dtype=float,
    )
    w = float(projected[2])
    if not math.isfinite(w) or abs(w) < EPS:
        raise ProjectionSupportError(
            "A support point projects to infinity."
        )
    u = float(projected[0] / w)
    v = float(projected[1] / w)
    if not (math.isfinite(u) and math.isfinite(v)):
        raise ProjectionSupportError(
            "A support point projects to non-finite coordinates."
        )
    return u, v


def _point_from_raw(
    value: Sequence[float],
    label: str,
) -> tuple[float, float]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 2
    ):
        raise ProjectionSupportError(
            f"{label} must contain exactly two coordinates."
        )
    x = float(value[0])
    y = float(value[1])
    if not (math.isfinite(x) and math.isfinite(y)):
        raise ProjectionSupportError(
            f"{label} coordinates must be finite."
        )
    return x, y


def _signed_area(
    polygon: Sequence[Point2D],
) -> float:
    if len(polygon) < 3:
        return 0.0
    total = 0.0
    for index, current in enumerate(polygon):
        following = polygon[(index + 1) % len(polygon)]
        total += (
            current.x * following.y
            - following.x * current.y
        )
    return 0.5 * total


def _polygon_area(
    polygon: Sequence[Point2D],
) -> float:
    return abs(_signed_area(polygon))


def _ensure_ccw(
    polygon: Sequence[Point2D],
) -> list[Point2D]:
    result = list(polygon)
    if _signed_area(result) < 0.0:
        result.reverse()
    return result


def _cross(
    origin: Point2D,
    a: Point2D,
    b: Point2D,
) -> float:
    return (
        (a.x - origin.x) * (b.y - origin.y)
        - (a.y - origin.y) * (b.x - origin.x)
    )


def _convex_hull(
    points: Sequence[Point2D],
) -> list[Point2D]:
    unique = sorted(
        {
            (round(float(point.x), 9), round(float(point.y), 9))
            for point in points
        }
    )
    if len(unique) <= 1:
        return [
            Point2D(x, y)
            for x, y in unique
        ]

    pts = [
        Point2D(x, y)
        for x, y in unique
    ]

    lower: list[Point2D] = []
    for point in pts:
        while (
            len(lower) >= 2
            and _cross(lower[-2], lower[-1], point) <= EPS
        ):
            lower.pop()
        lower.append(point)

    upper: list[Point2D] = []
    for point in reversed(pts):
        while (
            len(upper) >= 2
            and _cross(upper[-2], upper[-1], point) <= EPS
        ):
            upper.pop()
        upper.append(point)

    hull = lower[:-1] + upper[:-1]
    return _ensure_ccw(hull)


def _line_intersection(
    p1: Point2D,
    p2: Point2D,
    q1: Point2D,
    q2: Point2D,
) -> Point2D:
    px = p2.x - p1.x
    py = p2.y - p1.y
    qx = q2.x - q1.x
    qy = q2.y - q1.y

    denominator = px * qy - py * qx
    if abs(denominator) < EPS:
        # Parallel clipping boundaries are rare here. Returning the segment
        # endpoint is numerically safer than creating an infinite point.
        return p2

    rx = q1.x - p1.x
    ry = q1.y - p1.y
    t = (rx * qy - ry * qx) / denominator
    return Point2D(
        p1.x + t * px,
        p1.y + t * py,
    )


def _inside_half_plane(
    point: Point2D,
    edge_a: Point2D,
    edge_b: Point2D,
) -> bool:
    # Clip polygons are always normalized CCW.
    return _cross(edge_a, edge_b, point) >= -1e-9


def _intersect_convex_polygons(
    subject_polygon: Sequence[Point2D],
    clip_polygon: Sequence[Point2D],
) -> list[Point2D]:
    """Sutherland-Hodgman clipping for convex polygons."""

    output = _ensure_ccw(subject_polygon)
    clip = _ensure_ccw(clip_polygon)

    if len(output) < 3 or len(clip) < 3:
        return []

    for index, clip_a in enumerate(clip):
        clip_b = clip[(index + 1) % len(clip)]
        input_polygon = output
        output = []

        if not input_polygon:
            break

        previous = input_polygon[-1]
        previous_inside = _inside_half_plane(
            previous,
            clip_a,
            clip_b,
        )

        for current in input_polygon:
            current_inside = _inside_half_plane(
                current,
                clip_a,
                clip_b,
            )

            if current_inside:
                if not previous_inside:
                    output.append(
                        _line_intersection(
                            previous,
                            current,
                            clip_a,
                            clip_b,
                        )
                    )
                output.append(current)
            elif previous_inside:
                output.append(
                    _line_intersection(
                        previous,
                        current,
                        clip_a,
                        clip_b,
                    )
                )

            previous = current
            previous_inside = current_inside

    return _ensure_ccw(output)


def _pitch_rectangle(
    pitch: PitchModel,
) -> list[Point2D]:
    return [
        Point2D(0.0, 0.0),
        Point2D(pitch.length, 0.0),
        Point2D(pitch.length, pitch.width),
        Point2D(0.0, pitch.width),
    ]


def _image_footprint_on_pitch(
    inverse_homography: np.ndarray,
    video_width: float,
    video_height: float,
) -> list[Point2D]:
    image_corners = (
        (0.0, 0.0),
        (video_width, 0.0),
        (video_width, video_height),
        (0.0, video_height),
    )
    result = []
    for u, v in image_corners:
        x, y = _project_xy(
            inverse_homography,
            u,
            v,
        )
        result.append(
            Point2D(x, y)
        )
    return _ensure_ccw(result)


def _project_to_canonical_segment(
    point: Point2D,
    start: Point2D,
    end: Point2D,
) -> Point2D:
    dx = end.x - start.x
    dy = end.y - start.y
    denominator = dx * dx + dy * dy
    if denominator < EPS:
        return start

    t = (
        (point.x - start.x) * dx
        + (point.y - start.y) * dy
    ) / denominator
    t = min(1.0, max(0.0, t))

    return Point2D(
        start.x + t * dx,
        start.y + t * dy,
    )


def _collect_evidence_pitch_points(
    pitch: PitchModel,
    inverse_homography: np.ndarray,
    line_annotations: Sequence[dict[str, Any]],
    point_annotations: Sequence[dict[str, Any]],
) -> tuple[list[Point2D], int, int]:
    evidence: list[Point2D] = []
    point_count = 0
    line_count = 0

    for annotation in point_annotations:
        name = annotation.get("point_name")
        if not isinstance(name, str) or not name:
            continue
        canonical = pitch.point(name).point
        evidence.append(canonical)
        point_count += 1

    for annotation in line_annotations:
        name = annotation.get("line_name")
        raw_points = annotation.get("points")
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(raw_points, Sequence)
            or isinstance(raw_points, (str, bytes))
            or len(raw_points) != 2
        ):
            continue

        canonical = pitch.line(name)
        added = False

        for index, raw_point in enumerate(raw_points):
            u, v = _point_from_raw(
                raw_point,
                f"{name} click {index + 1}",
            )
            x, y = _project_xy(
                inverse_homography,
                u,
                v,
            )
            observed_pitch_point = Point2D(x, y)
            on_semantic_segment = _project_to_canonical_segment(
                observed_pitch_point,
                canonical.start,
                canonical.end,
            )
            evidence.append(on_semantic_segment)
            added = True

        if added:
            line_count += 1

    return evidence, point_count, line_count


def _expand_support_points(
    points: Sequence[Point2D],
    margin_m: float,
    samples: int,
) -> list[Point2D]:
    if margin_m <= 0.0:
        return list(points)

    result: list[Point2D] = list(points)
    sample_count = max(8, int(samples))

    for point in points:
        for index in range(sample_count):
            angle = 2.0 * math.pi * index / sample_count
            result.append(
                Point2D(
                    point.x + margin_m * math.cos(angle),
                    point.y + margin_m * math.sin(angle),
                )
            )

    return result


def _project_polygon(
    polygon: Sequence[Point2D],
    homography: np.ndarray,
) -> list[tuple[float, float]]:
    return [
        _project_xy(
            homography,
            point.x,
            point.y,
        )
        for point in polygon
    ]


def _projective_pitch_is_valid(
    pitch: PitchModel,
    homography: np.ndarray,
) -> tuple[bool, float]:
    """Check that the projective horizon does not cross the finite pitch."""

    values: list[float] = []
    for x in np.linspace(0.0, pitch.length, 11):
        for y in np.linspace(0.0, pitch.width, 9):
            w = float(
                homography[2, 0] * x
                + homography[2, 1] * y
                + homography[2, 2]
            )
            values.append(w)

    array = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(array)):
        return False, 0.0

    same_sign = bool(
        np.all(array > 0.0)
        or np.all(array < 0.0)
    )
    maximum = float(np.max(np.abs(array)))
    minimum = float(np.min(np.abs(array)))
    relative_margin = (
        minimum / maximum
        if maximum > EPS
        else 0.0
    )

    return same_sign, relative_margin


def build_projection_support(
    *,
    homography_pitch_to_image: Sequence[Sequence[float]] | np.ndarray,
    homography_image_to_pitch: Sequence[Sequence[float]] | np.ndarray,
    line_annotations: Sequence[dict[str, Any]] = (),
    point_annotations: Sequence[dict[str, Any]] = (),
    video_width: float,
    video_height: float,
    pitch_length: float = DEFAULT_PITCH_LENGTH,
    pitch_width: float = DEFAULT_PITCH_WIDTH,
    support_margin_m: float = DEFAULT_SUPPORT_MARGIN_M,
) -> SupportResult:
    """Compute visible and evidence-supported FIELD polygons.

    The support polygon is conservative on purpose. It is the convex hull of
    the actually observed semantic evidence, expanded by a small metric margin,
    then intersected with:
      - the finite pitch rectangle;
      - the current image footprint on the pitch plane.

    The result is intended to CLIP diagnostic rendering, not to modify H.
    """

    width = float(video_width)
    height = float(video_height)

    if not (
        math.isfinite(width)
        and math.isfinite(height)
        and width > 0.0
        and height > 0.0
    ):
        raise ProjectionSupportError(
            "Video dimensions must be positive finite numbers."
        )

    margin = float(support_margin_m)
    if not math.isfinite(margin) or margin < 0.0:
        raise ProjectionSupportError(
            "Support margin must be a finite non-negative number."
        )

    pitch = PitchModel(
        pitch_length,
        pitch_width,
    )
    h = _as_homography(
        homography_pitch_to_image,
        "H_pitch_to_image",
    )
    h_inverse = _as_homography(
        homography_image_to_pitch,
        "H_image_to_pitch",
    )

    warnings: list[str] = []

    projective_valid, horizon_margin = (
        _projective_pitch_is_valid(
            pitch,
            h,
        )
    )
    if not projective_valid:
        warnings.append(
            "El horizonte proyectivo atraviesa el terreno de juego; "
            "la homografía no debe considerarse globalmente válida."
        )
    elif horizon_margin < 0.02:
        warnings.append(
            "La homografía coloca el horizonte proyectivo muy cerca del "
            "terreno; la extrapolación puede ser inestable."
        )

    image_footprint = _image_footprint_on_pitch(
        h_inverse,
        width,
        height,
    )
    pitch_rectangle = _pitch_rectangle(pitch)
    visible_pitch = _intersect_convex_polygons(
        image_footprint,
        pitch_rectangle,
    )

    evidence, point_count, line_count = (
        _collect_evidence_pitch_points(
            pitch,
            h_inverse,
            line_annotations,
            point_annotations,
        )
    )

    support_pitch: list[Point2D] = []

    if len(evidence) >= 3:
        expanded = _expand_support_points(
            evidence,
            margin,
            DEFAULT_SUPPORT_EXPANSION_SAMPLES,
        )
        hull = _convex_hull(expanded)

        if len(hull) >= 3:
            support_pitch = _intersect_convex_polygons(
                hull,
                pitch_rectangle,
            )

            if (
                len(support_pitch) >= 3
                and len(visible_pitch) >= 3
            ):
                support_pitch = (
                    _intersect_convex_polygons(
                        support_pitch,
                        visible_pitch,
                    )
                )
    else:
        warnings.append(
            "No hay suficientes puntos de soporte bidimensionales para "
            "construir una región respaldada por evidencia."
        )

    if len(visible_pitch) < 3:
        warnings.append(
            "No se pudo obtener una región visible estable del terreno."
        )

    if len(support_pitch) < 3:
        warnings.append(
            "No se pudo obtener una región de proyección respaldada por "
            "las anotaciones actuales."
        )

    visible_image = (
        _project_polygon(
            visible_pitch,
            h,
        )
        if len(visible_pitch) >= 3
        else []
    )
    support_image = (
        _project_polygon(
            support_pitch,
            h,
        )
        if len(support_pitch) >= 3
        else []
    )

    return SupportResult(
        video_width=width,
        video_height=height,
        pitch_length=pitch.length,
        pitch_width=pitch.width,
        visible_pitch_polygon=tuple(visible_pitch),
        support_pitch_polygon=tuple(support_pitch),
        visible_image_polygon=tuple(visible_image),
        support_image_polygon=tuple(support_image),
        evidence_pitch_points=tuple(evidence),
        point_evidence_count=point_count,
        line_evidence_count=line_count,
        support_margin_m=margin,
        visible_pitch_area_m2=_polygon_area(
            visible_pitch
        ),
        support_pitch_area_m2=_polygon_area(
            support_pitch
        ),
        projective_valid=projective_valid,
        warnings=tuple(warnings),
    )


def build_projection_support_payload(
    solved: dict[str, Any],
    request_payload: dict[str, Any],
    *,
    support_margin_m: float = DEFAULT_SUPPORT_MARGIN_M,
) -> dict[str, Any]:
    """Convenience adapter for the current FastAPI solve endpoint."""

    if not isinstance(solved, dict):
        raise ProjectionSupportError(
            "Solved calibration must be an object."
        )
    if not isinstance(request_payload, dict):
        raise ProjectionSupportError(
            "Calibration request must be an object."
        )

    return build_projection_support(
        homography_pitch_to_image=solved["H_pitch_to_image"],
        homography_image_to_pitch=solved["H_image_to_pitch"],
        line_annotations=request_payload.get(
            "annotations",
            (),
        ),
        point_annotations=request_payload.get(
            "point_annotations",
            (),
        ),
        video_width=float(
            request_payload["video_width"]
        ),
        video_height=float(
            request_payload["video_height"]
        ),
        pitch_length=float(
            solved.get(
                "pitch_length",
                DEFAULT_PITCH_LENGTH,
            )
        ),
        pitch_width=float(
            solved.get(
                "pitch_width",
                DEFAULT_PITCH_WIDTH,
            )
        ),
        support_margin_m=support_margin_m,
    ).as_dict()


# ---------------------------------------------------------------------------
# Dependency-free synthetic diagnostics
# ---------------------------------------------------------------------------

def _synthetic_homography() -> np.ndarray:
    return np.array(
        [
            [10.5, -1.2, 260.0],
            [1.2, 7.8, 170.0],
            [0.0025, -0.0012, 1.0],
        ],
        dtype=float,
    )


def _project_point(
    homography: np.ndarray,
    point: Point2D,
) -> list[float]:
    u, v = _project_xy(
        homography,
        point.x,
        point.y,
    )
    return [u, v]


def _point_inside_convex_polygon(
    point: Point2D,
    polygon: Sequence[Point2D],
) -> bool:
    if len(polygon) < 3:
        return False
    ccw = _ensure_ccw(polygon)
    return all(
        _inside_half_plane(
            point,
            ccw[index],
            ccw[(index + 1) % len(ccw)],
        )
        for index in range(len(ccw))
    )


def _self_check() -> None:
    pitch = PitchModel()
    h = _synthetic_homography()
    h_inverse = np.linalg.inv(h)

    # One partial visible touchline segment + four exact landmarks clustered
    # around the right half. The support region should stay local rather than
    # silently claiming the whole 105x68 pitch.
    touchline = pitch.line("touchline_b")
    line_a = Point2D(
        touchline.start.x
        + 0.42 * (touchline.end.x - touchline.start.x),
        touchline.start.y,
    )
    line_b = Point2D(
        touchline.start.x
        + 0.72 * (touchline.end.x - touchline.start.x),
        touchline.start.y,
    )

    line_annotations = [
        {
            "line_name": "touchline_b",
            "points": [
                _project_point(h, line_a),
                _project_point(h, line_b),
            ],
        }
    ]

    point_names = (
        "right_penalty_front_a",
        "right_penalty_front_b",
        "right_goal_area_front_a",
        "right_penalty_spot",
    )
    point_annotations = [
        {
            "point_name": name,
            "point": _project_point(
                h,
                pitch.point(name).point,
            ),
        }
        for name in point_names
    ]

    result = build_projection_support(
        homography_pitch_to_image=h,
        homography_image_to_pitch=h_inverse,
        line_annotations=line_annotations,
        point_annotations=point_annotations,
        video_width=1920.0,
        video_height=1080.0,
        support_margin_m=5.0,
    )

    assert result.projective_valid
    assert len(result.visible_pitch_polygon) >= 3
    assert len(result.support_pitch_polygon) >= 3
    assert len(result.support_image_polygon) >= 3
    assert result.point_evidence_count == 4
    assert result.line_evidence_count == 1
    assert result.support_pitch_area_m2 > 0.0
    assert (
        result.support_pitch_area_m2
        <= result.visible_pitch_area_m2 + 1e-6
    )

    # The support should contain the right penalty spot evidence...
    right_spot = pitch.point(
        "right_penalty_spot"
    ).point
    assert _point_inside_convex_polygon(
        right_spot,
        result.support_pitch_polygon,
    )

    # ...but should not automatically claim a far left corner that had no
    # supporting evidence.
    far_left = pitch.point(
        "corner_left_a"
    ).point
    assert not _point_inside_convex_polygon(
        far_left,
        result.support_pitch_polygon,
    )

    for u, v in result.support_image_polygon:
        assert math.isfinite(u)
        assert math.isfinite(v)
        assert -1e-5 <= u <= 1920.0 + 1e-5
        assert -1e-5 <= v <= 1080.0 + 1e-5

    print("Projection support self-check: OK")
    print(
        "Visible pitch area: "
        f"{result.visible_pitch_area_m2:.2f} m²"
    )
    print(
        "Evidence-supported area: "
        f"{result.support_pitch_area_m2:.2f} m²"
    )
    print(
        "Support / visible ratio: "
        f"{100.0 * result.support_ratio_of_visible:.1f}%"
    )
    print(
        "Evidence: "
        f"{result.point_evidence_count} points + "
        f"{result.line_evidence_count} partial line"
    )
    print(
        "Far unsupported corner excluded: OK"
    )


if __name__ == "__main__":
    _self_check()
