"""Human-assisted homography solver for RealJaenLab FIELD calibration.

The solver deliberately knows nothing about FastAPI or the web interface.
It consumes semantic pitch-line annotations expressed in native video pixels
and returns a pitch-to-image homography plus diagnostic/projected geometry.

V1 approach
-----------
1. Each semantic annotation defines an observed infinite image line from two clicks.
2. Canonical pitch lines belong to one of two orthogonal families:
   x_constant or y_constant.
3. Every annotated x/y pair yields:
   - a canonical pitch intersection (x, y), and
   - the corresponding intersection of the two observed image lines.
4. A normalized DLT estimates H_pitch_to_image from those correspondences.
5. The complete canonical pitch is projected for visual validation.

This is a baseline solver. It does not detect lines automatically and it does
not model lens/panorama distortion.
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
    """Base error for invalid or insufficient calibration geometry."""


class InsufficientGeometryError(CalibrationError):
    """The annotations do not constrain a projective homography."""


class DegenerateCalibrationError(CalibrationError):
    """The supplied geometry produces an unstable/invalid homography."""


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
class CalibrationResult:
    pitch_length: float
    pitch_width: float
    homography_pitch_to_image: np.ndarray
    homography_image_to_pitch: np.ndarray
    annotations: tuple[LineAnnotation, ...]
    derived_correspondences: tuple[dict[str, Any], ...]
    metrics: dict[str, Any]
    projected_geometry: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "pitch_length": float(self.pitch_length),
            "pitch_width": float(self.pitch_width),
            "H_pitch_to_image": self.homography_pitch_to_image.tolist(),
            "H_image_to_pitch": self.homography_image_to_pitch.tolist(),
            "annotation_count": len(self.annotations),
            "derived_correspondence_count": len(self.derived_correspondences),
            "annotations": [annotation.as_dict() for annotation in self.annotations],
            "derived_correspondences": list(self.derived_correspondences),
            "metrics": self.metrics,
            "projected_geometry": self.projected_geometry,
        }


def parse_annotations(
    raw_annotations: Sequence[LineAnnotation | dict[str, Any]],
    pitch: PitchModel,
    *,
    video_width: int | float | None = None,
    video_height: int | float | None = None,
) -> tuple[LineAnnotation, ...]:
    """Validate API/UI-friendly annotation data.

    Expected dict form:

    {
        "line_name": "halfway_line",
        "points": [[u1, v1], [u2, v2]]
    }
    """
    if not isinstance(raw_annotations, Sequence) or isinstance(
        raw_annotations, (str, bytes)
    ):
        raise CalibrationError("Annotations must be a sequence.")

    result: list[LineAnnotation] = []
    seen_names: set[str] = set()

    for raw in raw_annotations:
        if isinstance(raw, LineAnnotation):
            annotation = raw
        elif isinstance(raw, dict):
            name = raw.get("line_name")
            points = raw.get("points")
            if not isinstance(name, str) or not name:
                raise CalibrationError("Each annotation requires a semantic line_name.")
            if (
                not isinstance(points, Sequence)
                or isinstance(points, (str, bytes))
                or len(points) != 2
            ):
                raise CalibrationError(
                    f"Annotation {name!r} must contain exactly two image points."
                )
            p1 = _parse_image_point(points[0], name)
            p2 = _parse_image_point(points[1], name)
            annotation = LineAnnotation(name, p1, p2)
        else:
            raise CalibrationError("Each annotation must be a mapping or LineAnnotation.")

        # Semantic existence check.
        pitch.line(annotation.line_name)

        if annotation.line_name in seen_names:
            raise CalibrationError(
                f"Semantic line {annotation.line_name!r} is annotated more than once. "
                "Replace the existing annotation instead of duplicating it."
            )
        seen_names.add(annotation.line_name)

        click_distance = math.hypot(
            annotation.p2.u - annotation.p1.u,
            annotation.p2.v - annotation.p1.v,
        )
        if click_distance < MIN_CLICK_DISTANCE_PX:
            raise CalibrationError(
                f"Annotation {annotation.line_name!r} uses coincident or nearly "
                "coincident clicks."
            )

        for point in (annotation.p1, annotation.p2):
            if not (math.isfinite(point.u) and math.isfinite(point.v)):
                raise CalibrationError("Annotation coordinates must be finite.")
            if video_width is not None and not (0.0 <= point.u <= float(video_width)):
                raise CalibrationError(
                    f"Annotation {annotation.line_name!r} has a point outside video width."
                )
            if video_height is not None and not (0.0 <= point.v <= float(video_height)):
                raise CalibrationError(
                    f"Annotation {annotation.line_name!r} has a point outside video height."
                )

        result.append(annotation)

    return tuple(result)


def solve_calibration(
    annotations: Sequence[LineAnnotation | dict[str, Any]],
    *,
    pitch_length: float = DEFAULT_PITCH_LENGTH,
    pitch_width: float = DEFAULT_PITCH_WIDTH,
    video_width: int | float | None = None,
    video_height: int | float | None = None,
) -> CalibrationResult:
    """Solve a pitch-to-image homography from semantic line annotations."""
    pitch = PitchModel(pitch_length, pitch_width)
    parsed = parse_annotations(
        annotations,
        pitch,
        video_width=video_width,
        video_height=video_height,
    )

    observed_lines = {
        annotation.line_name: _image_line_from_annotation(annotation)
        for annotation in parsed
    }

    x_annotations = [
        annotation
        for annotation in parsed
        if pitch.line(annotation.line_name).family == "x_constant"
    ]
    y_annotations = [
        annotation
        for annotation in parsed
        if pitch.line(annotation.line_name).family == "y_constant"
    ]

    if len(x_annotations) < 2 or len(y_annotations) < 2:
        raise InsufficientGeometryError(
            "Insufficient geometry: annotate at least two distinct x-constant "
            "pitch lines and two distinct y-constant pitch lines."
        )

    _reject_duplicate_observed_lines(x_annotations, observed_lines)
    _reject_duplicate_observed_lines(y_annotations, observed_lines)

    pitch_points: list[list[float]] = []
    image_points: list[list[float]] = []
    derived: list[dict[str, Any]] = []

    for x_annotation in x_annotations:
        x_pitch_line = pitch.line(x_annotation.line_name)
        x_value = _constant_coordinate(x_pitch_line)

        for y_annotation in y_annotations:
            y_pitch_line = pitch.line(y_annotation.line_name)
            y_value = _constant_coordinate(y_pitch_line)

            image_intersection = _intersect_image_lines(
                observed_lines[x_annotation.line_name],
                observed_lines[y_annotation.line_name],
                x_annotation.line_name,
                y_annotation.line_name,
            )

            pitch_points.append([x_value, y_value])
            image_points.append([image_intersection.u, image_intersection.v])
            derived.append(
                {
                    "x_line": x_annotation.line_name,
                    "y_line": y_annotation.line_name,
                    "pitch_point": [float(x_value), float(y_value)],
                    "image_point": image_intersection.as_list(),
                }
            )

    source = np.asarray(pitch_points, dtype=float)
    destination = np.asarray(image_points, dtype=float)

    if len(source) < 4:
        raise InsufficientGeometryError(
            "At least four independent pitch/image intersections are required."
        )

    homography = _normalized_dlt(source, destination)

    if not np.all(np.isfinite(homography)):
        raise DegenerateCalibrationError("Solved homography contains non-finite values.")

    determinant = float(np.linalg.det(homography))
    if not math.isfinite(determinant) or abs(determinant) < EPS:
        raise DegenerateCalibrationError("Solved homography is singular.")

    try:
        inverse = np.linalg.inv(homography)
    except np.linalg.LinAlgError as exc:
        raise DegenerateCalibrationError("Solved homography cannot be inverted.") from exc

    inverse = _normalize_homography_scale(inverse)

    metrics = _reprojection_metrics(pitch, parsed, observed_lines, homography)
    projected_geometry = project_pitch_geometry(pitch, homography)

    return CalibrationResult(
        pitch_length=pitch.length,
        pitch_width=pitch.width,
        homography_pitch_to_image=homography,
        homography_image_to_pitch=inverse,
        annotations=parsed,
        derived_correspondences=tuple(derived),
        metrics=metrics,
        projected_geometry=projected_geometry,
    )


def solve_calibration_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Convenience adapter for a future FastAPI endpoint.

    Context fields such as asset_id/timestamp remain the responsibility of the
    application layer; this function solves geometry only.
    """
    if not isinstance(payload, dict):
        raise CalibrationError("Calibration payload must be an object.")

    result = solve_calibration(
        payload.get("annotations", []),
        pitch_length=float(payload.get("pitch_length", DEFAULT_PITCH_LENGTH)),
        pitch_width=float(payload.get("pitch_width", DEFAULT_PITCH_WIDTH)),
        video_width=payload.get("video_width"),
        video_height=payload.get("video_height"),
    )
    return result.as_dict()


def project_points(
    points: Sequence[Point2D | Sequence[float]],
    homography_pitch_to_image: np.ndarray,
) -> list[ImagePoint]:
    """Project pitch points into the image."""
    matrix = _validate_homography(homography_pitch_to_image)
    result: list[ImagePoint] = []

    for point in points:
        if isinstance(point, Point2D):
            x, y = point.x, point.y
        else:
            if len(point) != 2:
                raise CalibrationError("Projected points must contain x and y.")
            x, y = float(point[0]), float(point[1])

        projected = matrix @ np.array([x, y, 1.0], dtype=float)
        if abs(projected[2]) < EPS:
            raise DegenerateCalibrationError(
                "A pitch point projects to infinity under the solved homography."
            )
        u = projected[0] / projected[2]
        v = projected[1] / projected[2]
        if not (math.isfinite(float(u)) and math.isfinite(float(v))):
            raise DegenerateCalibrationError("Projected point is non-finite.")
        result.append(ImagePoint(float(u), float(v)))

    return result


def image_to_pitch_points(
    points: Sequence[ImagePoint | Sequence[float]],
    homography_image_to_pitch: np.ndarray,
) -> list[Point2D]:
    """Map image pixels back into pitch coordinates."""
    matrix = _validate_homography(homography_image_to_pitch)
    result: list[Point2D] = []

    for point in points:
        if isinstance(point, ImagePoint):
            u, v = point.u, point.v
        else:
            if len(point) != 2:
                raise CalibrationError("Image points must contain u and v.")
            u, v = float(point[0]), float(point[1])

        projected = matrix @ np.array([u, v, 1.0], dtype=float)
        if abs(projected[2]) < EPS:
            raise DegenerateCalibrationError(
                "An image point maps to infinity under the solved homography."
            )
        x = projected[0] / projected[2]
        y = projected[1] / projected[2]
        result.append(Point2D(float(x), float(y)))

    return result


def project_pitch_geometry(
    pitch: PitchModel,
    homography_pitch_to_image: np.ndarray,
) -> dict[str, Any]:
    """Project the complete canonical pitch into image coordinates."""
    lines = []
    for line in pitch.straight_lines:
        projected = project_points((line.start, line.end), homography_pitch_to_image)
        lines.append(
            {
                "name": line.name,
                "label": line.label,
                "family": line.family,
                "points": [point.as_list() for point in projected],
            }
        )

    curves = []
    for curve in pitch.curves:
        projected = project_points(curve.points, homography_pitch_to_image)
        curves.append(
            {
                "name": curve.name,
                "label": curve.label,
                "closed": curve.closed,
                "points": [point.as_list() for point in projected],
            }
        )

    points = []
    for pitch_point in pitch.points:
        projected = project_points(
            (pitch_point.point,),
            homography_pitch_to_image,
        )[0]
        points.append(
            {
                "name": pitch_point.name,
                "label": pitch_point.label,
                "point": projected.as_list(),
            }
        )

    return {
        "lines": lines,
        "curves": curves,
        "points": points,
    }


def _parse_image_point(raw: Any, line_name: str) -> ImagePoint:
    if (
        not isinstance(raw, Sequence)
        or isinstance(raw, (str, bytes))
        or len(raw) != 2
    ):
        raise CalibrationError(
            f"Annotation {line_name!r} contains an invalid image point."
        )
    try:
        return ImagePoint(float(raw[0]), float(raw[1]))
    except (TypeError, ValueError) as exc:
        raise CalibrationError(
            f"Annotation {line_name!r} contains non-numeric coordinates."
        ) from exc


def _image_line_from_annotation(annotation: LineAnnotation) -> np.ndarray:
    p1 = np.array([annotation.p1.u, annotation.p1.v, 1.0], dtype=float)
    p2 = np.array([annotation.p2.u, annotation.p2.v, 1.0], dtype=float)
    line = np.cross(p1, p2)
    norm = math.hypot(float(line[0]), float(line[1]))
    if norm < EPS:
        raise CalibrationError(
            f"Annotation {annotation.line_name!r} does not define a valid image line."
        )
    line /= norm

    # Stable sign convention to make duplicate checks deterministic.
    if line[0] < -EPS or (abs(line[0]) <= EPS and line[1] < 0):
        line *= -1.0
    return line


def _constant_coordinate(line: LineSegment) -> float:
    if line.family == "x_constant":
        if not math.isclose(line.start.x, line.end.x, abs_tol=1e-9):
            raise CalibrationError(f"Pitch line {line.name!r} is not x-constant.")
        return float(line.start.x)
    if line.family == "y_constant":
        if not math.isclose(line.start.y, line.end.y, abs_tol=1e-9):
            raise CalibrationError(f"Pitch line {line.name!r} is not y-constant.")
        return float(line.start.y)
    raise CalibrationError(f"Unsupported pitch-line family: {line.family!r}")


def _intersect_image_lines(
    line_a: np.ndarray,
    line_b: np.ndarray,
    name_a: str,
    name_b: str,
) -> ImagePoint:
    point = np.cross(line_a, line_b)
    if abs(float(point[2])) < EPS:
        raise DegenerateCalibrationError(
            f"Observed lines {name_a!r} and {name_b!r} are parallel or intersect "
            "too far away to provide a stable correspondence."
        )
    point /= point[2]
    if not np.all(np.isfinite(point)):
        raise DegenerateCalibrationError(
            f"Intersection of {name_a!r} and {name_b!r} is non-finite."
        )
    return ImagePoint(float(point[0]), float(point[1]))


def _reject_duplicate_observed_lines(
    annotations: Sequence[LineAnnotation],
    observed_lines: dict[str, np.ndarray],
) -> None:
    angle_limit = math.radians(DUPLICATE_LINE_ANGLE_DEG)

    for index, first in enumerate(annotations):
        first_line = observed_lines[first.line_name]
        for second in annotations[index + 1 :]:
            second_line = observed_lines[second.line_name]

            # Normals are unit length. The stable sign convention generally
            # aligns them, but abs() keeps the angular test orientation-free.
            dot = float(np.clip(np.dot(first_line[:2], second_line[:2]), -1.0, 1.0))
            angle = math.acos(abs(dot))

            aligned_second = second_line.copy()
            if np.dot(first_line[:2], aligned_second[:2]) < 0:
                aligned_second *= -1.0
            offset_difference = abs(float(first_line[2] - aligned_second[2]))

            if (
                angle <= angle_limit
                and offset_difference <= DUPLICATE_LINE_OFFSET_PX
            ):
                raise DegenerateCalibrationError(
                    f"Observed lines {first.line_name!r} and {second.line_name!r} "
                    "are effectively duplicates."
                )


def _normalized_dlt(source: np.ndarray, destination: np.ndarray) -> np.ndarray:
    if source.shape != destination.shape or source.ndim != 2 or source.shape[1] != 2:
        raise CalibrationError("DLT correspondences must be Nx2 source/destination arrays.")
    if len(source) < 4:
        raise InsufficientGeometryError("DLT requires at least four correspondences.")

    source_normalized, t_source = _normalize_points(source)
    destination_normalized, t_destination = _normalize_points(destination)

    rows = []
    for (x, y), (u, v) in zip(source_normalized, destination_normalized):
        rows.append([-x, -y, -1.0, 0.0, 0.0, 0.0, u * x, u * y, u])
        rows.append([0.0, 0.0, 0.0, -x, -y, -1.0, v * x, v * y, v])

    design = np.asarray(rows, dtype=float)

    rank = int(np.linalg.matrix_rank(design))
    if rank < 8:
        raise DegenerateCalibrationError(
            f"Calibration geometry is rank-deficient (DLT rank {rank}, expected >= 8)."
        )

    _, _, vh = np.linalg.svd(design, full_matrices=True)
    normalized_h = vh[-1].reshape(3, 3)

    homography = np.linalg.inv(t_destination) @ normalized_h @ t_source
    return _normalize_homography_scale(homography)


def _normalize_points(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    distances = np.linalg.norm(centered, axis=1)
    mean_distance = float(np.mean(distances))

    if mean_distance < EPS:
        raise DegenerateCalibrationError("Correspondence points collapse to one location.")

    scale = math.sqrt(2.0) / mean_distance
    transform = np.array(
        [
            [scale, 0.0, -scale * centroid[0]],
            [0.0, scale, -scale * centroid[1]],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )

    homogeneous = np.column_stack([points, np.ones(len(points), dtype=float)])
    normalized_h = (transform @ homogeneous.T).T
    return normalized_h[:, :2] / normalized_h[:, 2:3], transform


def _normalize_homography_scale(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != (3, 3):
        raise CalibrationError("Homography must be a 3x3 matrix.")

    if abs(float(matrix[2, 2])) > EPS:
        normalized = matrix / matrix[2, 2]
    else:
        norm = float(np.linalg.norm(matrix))
        if norm < EPS:
            raise DegenerateCalibrationError("Homography has zero scale.")
        normalized = matrix / norm

    if not np.all(np.isfinite(normalized)):
        raise DegenerateCalibrationError("Homography normalization produced non-finite values.")
    return normalized


def _validate_homography(matrix: np.ndarray) -> np.ndarray:
    array = np.asarray(matrix, dtype=float)
    if array.shape != (3, 3):
        raise CalibrationError("Homography must be a 3x3 matrix.")
    if not np.all(np.isfinite(array)):
        raise CalibrationError("Homography contains non-finite values.")
    return array


def _reprojection_metrics(
    pitch: PitchModel,
    annotations: Sequence[LineAnnotation],
    observed_lines: dict[str, np.ndarray],
    homography: np.ndarray,
) -> dict[str, Any]:
    all_distances: list[float] = []
    per_line: dict[str, dict[str, float]] = {}

    for annotation in annotations:
        segment = pitch.line(annotation.line_name)
        samples = _sample_segment(segment, count=21)
        projected = project_points(samples, homography)
        line = observed_lines[annotation.line_name]

        distances = [
            abs(line[0] * point.u + line[1] * point.v + line[2])
            for point in projected
        ]
        all_distances.extend(float(distance) for distance in distances)

        per_line[annotation.line_name] = {
            "median_px": float(np.median(distances)),
            "p95_px": float(np.percentile(distances, 95)),
            "max_px": float(np.max(distances)),
        }

    if not all_distances:
        raise CalibrationError("No reprojection distances could be calculated.")

    return {
        "median_px": float(np.median(all_distances)),
        "p95_px": float(np.percentile(all_distances, 95)),
        "max_px": float(np.max(all_distances)),
        "per_line": per_line,
        "note": (
            "These metrics measure fit to the user-provided lines. With the minimum "
            "four-line configuration they can be near zero by construction; visual "
            "agreement of pitch markings not used for solving is the important EXP-001 "
            "validation."
        ),
    }


def _sample_segment(segment: LineSegment, count: int) -> tuple[Point2D, ...]:
    if count < 2:
        raise ValueError("A segment requires at least two samples.")

    return tuple(
        Point2D(
            segment.start.x + (segment.end.x - segment.start.x) * index / (count - 1),
            segment.start.y + (segment.end.y - segment.start.y) * index / (count - 1),
        )
        for index in range(count)
    )


def _project_with_matrix(
    points: Iterable[Sequence[float]],
    matrix: np.ndarray,
) -> list[list[float]]:
    result = []
    for x, y in points:
        value = matrix @ np.array([float(x), float(y), 1.0], dtype=float)
        value /= value[2]
        result.append([float(value[0]), float(value[1])])
    return result


def _synthetic_annotations(
    pitch: PitchModel,
    known_h: np.ndarray,
    names: Sequence[str],
) -> list[dict[str, Any]]:
    annotations: list[dict[str, Any]] = []
    for name in names:
        segment = pitch.line(name)
        endpoints = _project_with_matrix(
            (
                (segment.start.x, segment.start.y),
                (segment.end.x, segment.end.y),
            ),
            known_h,
        )
        annotations.append({"line_name": name, "points": endpoints})
    return annotations


def _self_check() -> None:
    """Dependency-light synthetic validation for manual development."""
    pitch = PitchModel()

    known_h = np.array(
        [
            [11.0, 1.4, 180.0],
            [0.6, 7.5, 130.0],
            [0.008, 0.0015, 1.0],
        ],
        dtype=float,
    )
    known_h = _normalize_homography_scale(known_h)

    names = (
        "goal_line_left",
        "halfway_line",
        "right_penalty_front",
        "touchline_a",
        "touchline_b",
        "left_penalty_side_a",
    )
    annotations = _synthetic_annotations(pitch, known_h, names)

    result = solve_calibration(
        annotations,
        video_width=1920,
        video_height=1080,
    )

    estimated = _normalize_homography_scale(result.homography_pitch_to_image)
    matrix_error = float(np.max(np.abs(estimated - known_h)))

    if matrix_error > 1e-7:
        raise AssertionError(
            f"Synthetic homography recovery error is too high: {matrix_error}"
        )

    probe_pitch = [[10.0, 10.0], [52.5, 34.0], [90.0, 60.0]]
    probe_image = project_points(probe_pitch, estimated)
    recovered_pitch = image_to_pitch_points(
        [point.as_list() for point in probe_image],
        result.homography_image_to_pitch,
    )
    roundtrip_error = max(
        math.hypot(recovered.x - original[0], recovered.y - original[1])
        for original, recovered in zip(probe_pitch, recovered_pitch)
    )

    if roundtrip_error > 1e-7:
        raise AssertionError(f"Pitch/image roundtrip error is too high: {roundtrip_error}")

    insufficient_ok = False
    try:
        solve_calibration(
            _synthetic_annotations(
                pitch,
                known_h,
                ("goal_line_left", "halfway_line", "touchline_a"),
            )
        )
    except InsufficientGeometryError:
        insufficient_ok = True

    if not insufficient_ok:
        raise AssertionError("Insufficient geometry was not rejected.")

    print("Calibration solver self-check: OK")
    print(f"Known-H recovery max matrix error: {matrix_error:.3e}")
    print(f"Pitch/image roundtrip max error: {roundtrip_error:.3e} m")
    print(f"Annotations used: {len(result.annotations)}")
    print(f"Derived intersections: {len(result.derived_correspondences)}")
    print(f"Median line-fit error: {result.metrics['median_px']:.3e} px")
    print(f"P95 line-fit error: {result.metrics['p95_px']:.3e} px")
    print("Insufficient-geometry rejection: OK")


if __name__ == "__main__":
    _self_check()
