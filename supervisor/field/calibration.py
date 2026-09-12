"""Human-assisted homography solver for RealJaenLab FIELD calibration.

V1 supports a mixed calibration model:

1. Semantic pitch-line annotations (two image clicks per line).
2. Direct semantic point annotations (one image click per canonical landmark).
3. Automatic point correspondences derived from intersections of annotated
   x-constant and y-constant lines.
4. A normalized DLT homography solved from the combined point set.

The solver remains UI/framework independent.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Sequence

import numpy as np

from supervisor.field.pitch import (
    DEFAULT_PITCH_LENGTH,
    DEFAULT_PITCH_WIDTH,
    LineSegment,
    PitchModel,
    Point2D,
)


EPS = 1e-10
MIN_CLICK_DISTANCE_PX = 2.0
DUPLICATE_LINE_ANGLE_DEG = 0.10
DUPLICATE_LINE_OFFSET_PX = 1.0


class CalibrationError(ValueError):
    pass


class InsufficientGeometryError(CalibrationError):
    pass


class DegenerateCalibrationError(CalibrationError):
    pass


@dataclass(frozen=True, slots=True)
class ImagePoint:
    u: float
    v: float

    def as_list(self) -> list[float]:
        return [float(self.u), float(self.v)]


@dataclass(frozen=True, slots=True)
class LineAnnotation:
    line_name: str
    p1: ImagePoint
    p2: ImagePoint

    def as_dict(self) -> dict[str, Any]:
        return {
            "line_name": self.line_name,
            "points": [self.p1.as_list(), self.p2.as_list()],
        }


@dataclass(frozen=True, slots=True)
class PointAnnotation:
    point_name: str
    image_point: ImagePoint

    def as_dict(self) -> dict[str, Any]:
        return {
            "point_name": self.point_name,
            "point": self.image_point.as_list(),
        }


@dataclass(frozen=True, slots=True)
class CalibrationResult:
    pitch_length: float
    pitch_width: float
    homography_pitch_to_image: np.ndarray
    homography_image_to_pitch: np.ndarray
    line_annotations: tuple[LineAnnotation, ...]
    point_annotations: tuple[PointAnnotation, ...]
    derived_correspondences: tuple[dict[str, Any], ...]
    direct_correspondences: tuple[dict[str, Any], ...]
    metrics: dict[str, Any]
    projected_geometry: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "pitch_length": float(self.pitch_length),
            "pitch_width": float(self.pitch_width),
            "H_pitch_to_image": self.homography_pitch_to_image.tolist(),
            "H_image_to_pitch": self.homography_image_to_pitch.tolist(),
            "annotation_count": len(self.line_annotations) + len(self.point_annotations),
            "line_annotation_count": len(self.line_annotations),
            "point_annotation_count": len(self.point_annotations),
            "derived_correspondence_count": len(self.derived_correspondences),
            "direct_correspondence_count": len(self.direct_correspondences),
            "correspondence_count": len(self.derived_correspondences) + len(self.direct_correspondences),
            "annotations": [a.as_dict() for a in self.line_annotations],
            "point_annotations": [a.as_dict() for a in self.point_annotations],
            "derived_correspondences": list(self.derived_correspondences),
            "direct_correspondences": list(self.direct_correspondences),
            "metrics": self.metrics,
            "projected_geometry": self.projected_geometry,
        }


def _parse_image_point(raw: Sequence[float], label: str) -> ImagePoint:
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or len(raw) != 2:
        raise CalibrationError(f"{label} must contain exactly two coordinates.")
    try:
        u, v = float(raw[0]), float(raw[1])
    except (TypeError, ValueError) as exc:
        raise CalibrationError(f"{label} coordinates must be numeric.") from exc
    if not (math.isfinite(u) and math.isfinite(v)):
        raise CalibrationError(f"{label} coordinates must be finite.")
    return ImagePoint(u, v)


def _validate_image_bounds(point: ImagePoint, video_width, video_height, label: str) -> None:
    if video_width is not None and not (0.0 <= point.u <= float(video_width)):
        raise CalibrationError(f"{label} is outside video width.")
    if video_height is not None and not (0.0 <= point.v <= float(video_height)):
        raise CalibrationError(f"{label} is outside video height.")


def parse_line_annotations(
    raw_annotations: Sequence[LineAnnotation | dict[str, Any]],
    pitch: PitchModel,
    *,
    video_width=None,
    video_height=None,
) -> tuple[LineAnnotation, ...]:
    if not isinstance(raw_annotations, Sequence) or isinstance(raw_annotations, (str, bytes)):
        raise CalibrationError("Line annotations must be a sequence.")

    result = []
    seen = set()
    for raw in raw_annotations:
        if isinstance(raw, LineAnnotation):
            ann = raw
        elif isinstance(raw, dict):
            name = raw.get("line_name")
            pts = raw.get("points")
            if not isinstance(name, str) or not name:
                raise CalibrationError("Each line annotation requires line_name.")
            if not isinstance(pts, Sequence) or isinstance(pts, (str, bytes)) or len(pts) != 2:
                raise CalibrationError(f"Line annotation {name!r} must contain exactly two points.")
            ann = LineAnnotation(name, _parse_image_point(pts[0], name), _parse_image_point(pts[1], name))
        else:
            raise CalibrationError("Each line annotation must be an object.")

        pitch.line(ann.line_name)
        if ann.line_name in seen:
            raise CalibrationError(f"Semantic line {ann.line_name!r} is annotated more than once.")
        seen.add(ann.line_name)

        if math.hypot(ann.p2.u - ann.p1.u, ann.p2.v - ann.p1.v) < MIN_CLICK_DISTANCE_PX:
            raise CalibrationError(f"Line annotation {ann.line_name!r} uses coincident clicks.")

        _validate_image_bounds(ann.p1, video_width, video_height, ann.line_name)
        _validate_image_bounds(ann.p2, video_width, video_height, ann.line_name)
        result.append(ann)
    return tuple(result)


def parse_point_annotations(
    raw_annotations: Sequence[PointAnnotation | dict[str, Any]],
    pitch: PitchModel,
    *,
    video_width=None,
    video_height=None,
) -> tuple[PointAnnotation, ...]:
    if not isinstance(raw_annotations, Sequence) or isinstance(raw_annotations, (str, bytes)):
        raise CalibrationError("Point annotations must be a sequence.")

    result = []
    seen = set()
    for raw in raw_annotations:
        if isinstance(raw, PointAnnotation):
            ann = raw
        elif isinstance(raw, dict):
            name = raw.get("point_name")
            raw_point = raw.get("point")
            if not isinstance(name, str) or not name:
                raise CalibrationError("Each point annotation requires point_name.")
            ann = PointAnnotation(name, _parse_image_point(raw_point, name))
        else:
            raise CalibrationError("Each point annotation must be an object.")

        pitch.point(ann.point_name)
        if ann.point_name in seen:
            raise CalibrationError(f"Semantic point {ann.point_name!r} is annotated more than once.")
        seen.add(ann.point_name)
        _validate_image_bounds(ann.image_point, video_width, video_height, ann.point_name)
        result.append(ann)
    return tuple(result)


def _image_line_from_annotation(annotation: LineAnnotation) -> np.ndarray:
    p1 = np.array([annotation.p1.u, annotation.p1.v, 1.0], dtype=float)
    p2 = np.array([annotation.p2.u, annotation.p2.v, 1.0], dtype=float)
    line = np.cross(p1, p2)
    norm = math.hypot(float(line[0]), float(line[1]))
    if norm < EPS:
        raise DegenerateCalibrationError(f"Observed line {annotation.line_name!r} is degenerate.")
    return line / norm


def _constant_coordinate(line: LineSegment) -> float:
    if line.family == "x_constant":
        return float(line.start.x)
    if line.family == "y_constant":
        return float(line.start.y)
    raise CalibrationError(f"Unsupported line family {line.family!r}.")


def _intersect_image_lines(a: np.ndarray, b: np.ndarray, a_name: str, b_name: str) -> ImagePoint:
    p = np.cross(a, b)
    if abs(float(p[2])) < EPS:
        raise DegenerateCalibrationError(
            f"Observed lines {a_name!r} and {b_name!r} are parallel or nearly parallel."
        )
    return ImagePoint(float(p[0] / p[2]), float(p[1] / p[2]))


def _line_angle_deg(line: np.ndarray) -> float:
    # direction vector is (-b, a)
    return math.degrees(math.atan2(float(line[0]), float(-line[1])))


def _angular_distance_deg(a: float, b: float) -> float:
    d = abs((a - b) % 180.0)
    return min(d, 180.0 - d)


def _reject_duplicate_observed_lines(
    annotations: Sequence[LineAnnotation],
    observed_lines: dict[str, np.ndarray],
) -> None:
    for i, first in enumerate(annotations):
        l1 = observed_lines[first.line_name]
        for second in annotations[i + 1:]:
            l2 = observed_lines[second.line_name]
            if _angular_distance_deg(_line_angle_deg(l1), _line_angle_deg(l2)) > DUPLICATE_LINE_ANGLE_DEG:
                continue
            # Normalized ax+by+c. Account for sign.
            if np.dot(l1[:2], l2[:2]) < 0:
                l2 = -l2
            if abs(float(l1[2] - l2[2])) <= DUPLICATE_LINE_OFFSET_PX:
                raise DegenerateCalibrationError(
                    f"Observed lines {first.line_name!r} and {second.line_name!r} are effectively duplicates."
                )


def _normalize_points(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centroid = points.mean(axis=0)
    shifted = points - centroid
    mean_distance = np.mean(np.sqrt(np.sum(shifted * shifted, axis=1)))
    if mean_distance < EPS:
        raise DegenerateCalibrationError("Point configuration has no spatial extent.")
    scale = math.sqrt(2.0) / mean_distance
    transform = np.array(
        [[scale, 0, -scale * centroid[0]], [0, scale, -scale * centroid[1]], [0, 0, 1]],
        dtype=float,
    )
    homogeneous = np.column_stack([points, np.ones(len(points))])
    normalized_h = (transform @ homogeneous.T).T
    return normalized_h[:, :2], transform


def _normalize_homography_scale(h: np.ndarray) -> np.ndarray:
    if abs(float(h[2, 2])) > EPS:
        return h / h[2, 2]
    norm = np.linalg.norm(h)
    if norm < EPS:
        raise DegenerateCalibrationError("Homography scale is zero.")
    return h / norm


def _normalized_dlt(source: np.ndarray, destination: np.ndarray) -> np.ndarray:
    if source.shape != destination.shape or source.ndim != 2 or source.shape[1] != 2:
        raise CalibrationError("DLT expects matching Nx2 point arrays.")
    if len(source) < 4:
        raise InsufficientGeometryError("At least four point correspondences are required.")

    src_norm, t_src = _normalize_points(source)
    dst_norm, t_dst = _normalize_points(destination)

    rows = []
    for (x, y), (u, v) in zip(src_norm, dst_norm):
        rows.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        rows.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    a = np.asarray(rows, dtype=float)

    _, s, vt = np.linalg.svd(a)
    if len(s) < 8 or s[-2] < EPS:
        raise DegenerateCalibrationError("Calibration point configuration is rank deficient.")

    h_norm = vt[-1].reshape(3, 3)
    h = np.linalg.inv(t_dst) @ h_norm @ t_src
    h = _normalize_homography_scale(h)

    # Reject unstable solutions.
    if np.linalg.matrix_rank(h) < 3:
        raise DegenerateCalibrationError("Solved homography is singular.")
    return h


def _unique_correspondences(
    derived_pairs: list[tuple[list[float], list[float]]],
    direct_pairs: list[tuple[list[float], list[float]]],
) -> tuple[np.ndarray, np.ndarray]:
    """Build one image correspondence per canonical pitch point.

    Direct human point clicks intentionally override an automatically-derived
    line intersection at the same canonical location. This avoids feeding DLT
    contradictory destinations for one world point when both annotation types
    are present.
    """
    by_pitch: dict[tuple[float, float], tuple[list[float], list[float]]] = {}

    def key(point: list[float]) -> tuple[float, float]:
        return (round(float(point[0]), 9), round(float(point[1]), 9))

    for pitch_point, image_point in derived_pairs:
        by_pitch.setdefault(key(pitch_point), (pitch_point, image_point))

    for pitch_point, image_point in direct_pairs:
        by_pitch[key(pitch_point)] = (pitch_point, image_point)

    pairs = list(by_pitch.values())
    if not pairs:
        return np.empty((0, 2)), np.empty((0, 2))
    return (
        np.asarray([pitch_point for pitch_point, _ in pairs], dtype=float),
        np.asarray([image_point for _, image_point in pairs], dtype=float),
    )


def solve_calibration(
    annotations: Sequence[LineAnnotation | dict[str, Any]] = (),
    *,
    point_annotations: Sequence[PointAnnotation | dict[str, Any]] = (),
    pitch_length: float = DEFAULT_PITCH_LENGTH,
    pitch_width: float = DEFAULT_PITCH_WIDTH,
    video_width=None,
    video_height=None,
) -> CalibrationResult:
    pitch = PitchModel(pitch_length, pitch_width)
    lines = parse_line_annotations(annotations, pitch, video_width=video_width, video_height=video_height)
    points = parse_point_annotations(point_annotations, pitch, video_width=video_width, video_height=video_height)

    observed_lines = {a.line_name: _image_line_from_annotation(a) for a in lines}
    x_lines = [a for a in lines if pitch.line(a.line_name).family == "x_constant"]
    y_lines = [a for a in lines if pitch.line(a.line_name).family == "y_constant"]

    _reject_duplicate_observed_lines(x_lines, observed_lines)
    _reject_duplicate_observed_lines(y_lines, observed_lines)

    derived_pairs: list[tuple[list[float], list[float]]] = []
    direct_pairs: list[tuple[list[float], list[float]]] = []
    derived = []

    for x_ann in x_lines:
        x_value = _constant_coordinate(pitch.line(x_ann.line_name))
        for y_ann in y_lines:
            y_value = _constant_coordinate(pitch.line(y_ann.line_name))
            image_intersection = _intersect_image_lines(
                observed_lines[x_ann.line_name],
                observed_lines[y_ann.line_name],
                x_ann.line_name,
                y_ann.line_name,
            )
            pitch_point = [x_value, y_value]
            image_point = image_intersection.as_list()
            derived_pairs.append((pitch_point, image_point))
            derived.append(
                {
                    "source": "line_intersection",
                    "x_line": x_ann.line_name,
                    "y_line": y_ann.line_name,
                    "pitch_point": pitch_point,
                    "image_point": image_point,
                }
            )

    direct = []
    for ann in points:
        canonical = pitch.point(ann.point_name).point
        pitch_point = canonical.as_list()
        image_point = ann.image_point.as_list()
        direct_pairs.append((pitch_point, image_point))
        direct.append(
            {
                "source": "direct_point",
                "point_name": ann.point_name,
                "pitch_point": pitch_point,
                "image_point": image_point,
            }
        )

    source, destination = _unique_correspondences(derived_pairs, direct_pairs)

    if len(source) < 4:
        raise InsufficientGeometryError(
            "Insufficient geometry: provide at least four independent point correspondences. "
            "These may come from direct semantic points, line intersections, or a mixture of both."
        )

    # Ensure the pitch points are genuinely 2D, not collinear.
    centered = source - source.mean(axis=0)
    if np.linalg.matrix_rank(centered) < 2:
        raise InsufficientGeometryError(
            "Insufficient geometry: the available pitch correspondences are collinear."
        )

    h = _normalized_dlt(source, destination)
    if not np.all(np.isfinite(h)):
        raise DegenerateCalibrationError("Solved homography contains non-finite values.")

    det = float(np.linalg.det(h))
    if not math.isfinite(det) or abs(det) < EPS:
        raise DegenerateCalibrationError("Solved homography is singular.")

    try:
        inverse = _normalize_homography_scale(np.linalg.inv(h))
    except np.linalg.LinAlgError as exc:
        raise DegenerateCalibrationError("Solved homography cannot be inverted.") from exc

    metrics = _reprojection_metrics(pitch, lines, points, observed_lines, h)
    projected_geometry = project_pitch_geometry(pitch, h)

    return CalibrationResult(
        pitch_length=pitch.length,
        pitch_width=pitch.width,
        homography_pitch_to_image=h,
        homography_image_to_pitch=inverse,
        line_annotations=lines,
        point_annotations=points,
        derived_correspondences=tuple(derived),
        direct_correspondences=tuple(direct),
        metrics=metrics,
        projected_geometry=projected_geometry,
    )


def solve_calibration_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise CalibrationError("Calibration payload must be an object.")
    return solve_calibration(
        payload.get("annotations", []),
        point_annotations=payload.get("point_annotations", []),
        pitch_length=float(payload.get("pitch_length", DEFAULT_PITCH_LENGTH)),
        pitch_width=float(payload.get("pitch_width", DEFAULT_PITCH_WIDTH)),
        video_width=payload.get("video_width"),
        video_height=payload.get("video_height"),
    ).as_dict()


def project_points(
    points: Sequence[Point2D | Sequence[float]],
    homography_pitch_to_image: np.ndarray,
) -> list[ImagePoint]:
    matrix = np.asarray(homography_pitch_to_image, dtype=float)
    if matrix.shape != (3, 3):
        raise CalibrationError("Homography must be 3x3.")
    result = []
    for point in points:
        x, y = (point.x, point.y) if isinstance(point, Point2D) else (float(point[0]), float(point[1]))
        projected = matrix @ np.array([x, y, 1.0], dtype=float)
        if abs(float(projected[2])) < EPS:
            raise DegenerateCalibrationError("A point projects to infinity.")
        result.append(ImagePoint(float(projected[0] / projected[2]), float(projected[1] / projected[2])))
    return result


def image_to_pitch_points(
    points: Sequence[ImagePoint | Sequence[float]],
    homography_image_to_pitch: np.ndarray,
) -> list[Point2D]:
    matrix = np.asarray(homography_image_to_pitch, dtype=float)
    result = []
    for point in points:
        u, v = (point.u, point.v) if isinstance(point, ImagePoint) else (float(point[0]), float(point[1]))
        projected = matrix @ np.array([u, v, 1.0], dtype=float)
        if abs(float(projected[2])) < EPS:
            raise DegenerateCalibrationError("An image point maps to infinity.")
        result.append(Point2D(float(projected[0] / projected[2]), float(projected[1] / projected[2])))
    return result


def _point_to_line_distance(point: ImagePoint, line: np.ndarray) -> float:
    return abs(float(line[0] * point.u + line[1] * point.v + line[2]))


def _reprojection_metrics(
    pitch: PitchModel,
    line_annotations: Sequence[LineAnnotation],
    point_annotations: Sequence[PointAnnotation],
    observed_lines: dict[str, np.ndarray],
    h: np.ndarray,
) -> dict[str, Any]:
    line_errors = []
    per_line = {}
    for ann in line_annotations:
        canonical = pitch.line(ann.line_name)
        samples = [
            Point2D(
                canonical.start.x + (canonical.end.x - canonical.start.x) * t,
                canonical.start.y + (canonical.end.y - canonical.start.y) * t,
            )
            for t in np.linspace(0.0, 1.0, 21)
        ]
        projected = project_points(samples, h)
        errors = [_point_to_line_distance(p, observed_lines[ann.line_name]) for p in projected]
        line_errors.extend(errors)
        per_line[ann.line_name] = {
            "median_px": float(np.median(errors)),
            "p95_px": float(np.percentile(errors, 95)),
            "max_px": float(np.max(errors)),
        }

    point_errors = []
    per_point = {}
    for ann in point_annotations:
        canonical = pitch.point(ann.point_name).point
        predicted = project_points([canonical], h)[0]
        error = math.hypot(predicted.u - ann.image_point.u, predicted.v - ann.image_point.v)
        point_errors.append(error)
        per_point[ann.point_name] = float(error)

    combined = line_errors + point_errors
    if combined:
        median = float(np.median(combined))
        p95 = float(np.percentile(combined, 95))
        maximum = float(np.max(combined))
    else:
        median = p95 = maximum = float("nan")

    return {
        "median_px": median,
        "p95_px": p95,
        "max_px": maximum,
        "line_sample_count": len(line_errors),
        "point_sample_count": len(point_errors),
        "per_line": per_line,
        "per_point": per_point,
    }


def project_pitch_geometry(pitch: PitchModel, h: np.ndarray) -> dict[str, Any]:
    lines = []
    for line in pitch.straight_lines:
        p = project_points([line.start, line.end], h)
        lines.append({
            "name": line.name,
            "label": line.label,
            "family": line.family,
            "points": [p[0].as_list(), p[1].as_list()],
        })

    curves = []
    for curve in pitch.curves:
        p = project_points(curve.points, h)
        curves.append({
            "name": curve.name,
            "label": curve.label,
            "closed": curve.closed,
            "points": [q.as_list() for q in p],
        })

    points = []
    for point in pitch.points:
        p = project_points([point.point], h)[0]
        points.append({
            "name": point.name,
            "label": point.label,
            "category": point.category,
            "point": p.as_list(),
        })

    return {"lines": lines, "curves": curves, "points": points}


def _synthetic_homography() -> np.ndarray:
    return np.array(
        [
            [12.0, -2.3, 260.0],
            [1.8, 7.8, 180.0],
            [0.0045, -0.0018, 1.0],
        ],
        dtype=float,
    )


def _annotation_from_projected_line(pitch: PitchModel, name: str, h: np.ndarray) -> LineAnnotation:
    line = pitch.line(name)
    projected = project_points([line.start, line.end], h)
    return LineAnnotation(name, projected[0], projected[1])


def _point_annotation_from_projection(pitch: PitchModel, name: str, h: np.ndarray) -> PointAnnotation:
    point = pitch.point(name).point
    projected = project_points([point], h)[0]
    return PointAnnotation(name, projected)


def _self_check() -> None:
    pitch = PitchModel()
    known = _normalize_homography_scale(_synthetic_homography())

    # Mixed case: only 1 x line + 1 y line gives one derived intersection,
    # plus three explicit semantic points gives the remaining constraints.
    lines = [
        _annotation_from_projected_line(pitch, "halfway_line", known),
        _annotation_from_projected_line(pitch, "touchline_a", known),
    ]
    points = [
        _point_annotation_from_projection(pitch, "left_penalty_front_b", known),
        _point_annotation_from_projection(pitch, "right_penalty_front_b", known),
        _point_annotation_from_projection(pitch, "right_goal_area_front_a", known),
    ]

    solved = solve_calibration(lines, point_annotations=points)
    recovered = _normalize_homography_scale(solved.homography_pitch_to_image)
    max_error = float(np.max(np.abs(recovered - known)))
    assert max_error < 1e-8, max_error

    # Points-only case.
    point_only = [
        _point_annotation_from_projection(pitch, "corner_left_a", known),
        _point_annotation_from_projection(pitch, "corner_right_a", known),
        _point_annotation_from_projection(pitch, "corner_left_b", known),
        _point_annotation_from_projection(pitch, "right_penalty_front_b", known),
    ]
    solved_points = solve_calibration(point_annotations=point_only)
    point_only_error = float(
        np.max(
            np.abs(
                _normalize_homography_scale(solved_points.homography_pitch_to_image)
                - known
            )
        )
    )
    assert point_only_error < 1e-8, point_only_error

    # Insufficient geometry.
    rejected = False
    try:
        solve_calibration(point_annotations=point_only[:3])
    except InsufficientGeometryError:
        rejected = True
    assert rejected

    print("Calibration solver V1 self-check: OK")
    print(f"Mixed line+point recovery max matrix error: {max_error:.3e}")
    print(f"Point-only recovery max matrix error: {point_only_error:.3e}")
    print(f"Mixed correspondences: {solved.as_dict()['correspondence_count']}")
    print("Insufficient-geometry rejection: OK")


if __name__ == "__main__":
    _self_check()
