"""Human-assisted homography solver for RealJaenLab FIELD calibration.

V2 makes semantic field lines first-class homography constraints.

Key idea
--------
A human line annotation means:

    "this visible image segment belongs to this known pitch line"

The two image clicks ONLY define the observed image line. They are not assumed
to correspond to the canonical endpoints of the pitch segment.

Therefore V2 does NOT create artificial point correspondences by intersecting
unrelated annotated lines.

The solver combines:

- direct semantic point correspondences;
- direct semantic line correspondences;
- a normalized mixed linear homography estimate;
- a small robust geometric refinement using point reprojection error and
  point-to-line distance.

This allows a partial visible touchline, penalty-area line, halfway line, etc.
to constrain the homography even when its real corners/intersections are not
visible in the frame.

The module remains independent from FastAPI and the browser UI.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence

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

# Robust-refinement defaults. These are deliberately conservative because
# manual annotations have finite precision.
DEFAULT_HUBER_DELTA_PX = 6.0
DEFAULT_MAX_REFINEMENT_ITERATIONS = 25


class CalibrationError(ValueError):
    """Base error for invalid calibration input."""


class InsufficientGeometryError(CalibrationError):
    """The supplied annotations do not constrain a unique homography."""


class DegenerateCalibrationError(CalibrationError):
    """The supplied geometry yields an unstable or invalid homography."""


@dataclass(frozen=True, slots=True)
class ImagePoint:
    u: float
    v: float

    def as_list(self) -> list[float]:
        return [float(self.u), float(self.v)]


@dataclass(frozen=True, slots=True)
class LineAnnotation:
    """One semantic pitch line observed in the image.

    p1 and p2 are arbitrary visible points on the painted line. They are NOT
    canonical endpoint correspondences.
    """

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
    metrics: dict[str, Any]
    projected_geometry: dict[str, Any]
    linear_rank: int
    refinement_iterations: int

    def as_dict(self) -> dict[str, Any]:
        line_count = len(self.line_annotations)
        point_count = len(self.point_annotations)

        return {
            "pitch_length": float(self.pitch_length),
            "pitch_width": float(self.pitch_width),
            "H_pitch_to_image": self.homography_pitch_to_image.tolist(),
            "H_image_to_pitch": self.homography_image_to_pitch.tolist(),
            "annotation_count": line_count + point_count,
            "line_annotation_count": line_count,
            "point_annotation_count": point_count,
            # V2 terminology.
            "line_constraint_count": line_count,
            "direct_correspondence_count": point_count,
            "constraint_count": line_count + point_count,
            "constraint_equation_count": 2 * (line_count + point_count),
            "linear_rank": int(self.linear_rank),
            "refinement_iterations": int(self.refinement_iterations),
            # Compatibility with the V1 TEST UI. V2 no longer creates line
            # intersections, so this is intentionally always zero.
            "derived_correspondence_count": 0,
            "correspondence_count": point_count,
            "derived_correspondences": [],
            "direct_correspondences": [
                {
                    "source": "direct_point",
                    "point_name": annotation.point_name,
                    "pitch_point": None,  # populated below for API compatibility
                    "image_point": annotation.image_point.as_list(),
                }
                for annotation in self.point_annotations
            ],
            "annotations": [
                annotation.as_dict() for annotation in self.line_annotations
            ],
            "point_annotations": [
                annotation.as_dict() for annotation in self.point_annotations
            ],
            "metrics": self.metrics,
            "projected_geometry": self.projected_geometry,
        }


def _parse_image_point(raw: Sequence[float], label: str) -> ImagePoint:
    if (
        not isinstance(raw, Sequence)
        or isinstance(raw, (str, bytes))
        or len(raw) != 2
    ):
        raise CalibrationError(f"{label} must contain exactly two coordinates.")

    try:
        u, v = float(raw[0]), float(raw[1])
    except (TypeError, ValueError) as exc:
        raise CalibrationError(f"{label} coordinates must be numeric.") from exc

    if not (math.isfinite(u) and math.isfinite(v)):
        raise CalibrationError(f"{label} coordinates must be finite.")

    return ImagePoint(u, v)


def _validate_image_bounds(
    point: ImagePoint,
    video_width: int | float | None,
    video_height: int | float | None,
    label: str,
) -> None:
    if video_width is not None and not (0.0 <= point.u <= float(video_width)):
        raise CalibrationError(f"{label} is outside video width.")

    if video_height is not None and not (0.0 <= point.v <= float(video_height)):
        raise CalibrationError(f"{label} is outside video height.")


def parse_line_annotations(
    raw_annotations: Sequence[LineAnnotation | dict[str, Any]],
    pitch: PitchModel,
    *,
    video_width: int | float | None = None,
    video_height: int | float | None = None,
) -> tuple[LineAnnotation, ...]:
    if not isinstance(raw_annotations, Sequence) or isinstance(
        raw_annotations, (str, bytes)
    ):
        raise CalibrationError("Line annotations must be a sequence.")

    result: list[LineAnnotation] = []
    seen: set[str] = set()

    for raw in raw_annotations:
        if isinstance(raw, LineAnnotation):
            annotation = raw
        elif isinstance(raw, dict):
            name = raw.get("line_name")
            points = raw.get("points")

            if not isinstance(name, str) or not name:
                raise CalibrationError(
                    "Each line annotation requires line_name."
                )

            if (
                not isinstance(points, Sequence)
                or isinstance(points, (str, bytes))
                or len(points) != 2
            ):
                raise CalibrationError(
                    f"Line annotation {name!r} must contain exactly two points."
                )

            annotation = LineAnnotation(
                name,
                _parse_image_point(points[0], name),
                _parse_image_point(points[1], name),
            )
        else:
            raise CalibrationError(
                "Each line annotation must be an object."
            )

        # Semantic existence check.
        pitch.line(annotation.line_name)

        if annotation.line_name in seen:
            raise CalibrationError(
                f"Semantic line {annotation.line_name!r} is annotated more than once."
            )
        seen.add(annotation.line_name)

        click_distance = math.hypot(
            annotation.p2.u - annotation.p1.u,
            annotation.p2.v - annotation.p1.v,
        )
        if click_distance < MIN_CLICK_DISTANCE_PX:
            raise CalibrationError(
                f"Line annotation {annotation.line_name!r} uses coincident clicks."
            )

        _validate_image_bounds(
            annotation.p1,
            video_width,
            video_height,
            annotation.line_name,
        )
        _validate_image_bounds(
            annotation.p2,
            video_width,
            video_height,
            annotation.line_name,
        )

        result.append(annotation)

    return tuple(result)


def parse_point_annotations(
    raw_annotations: Sequence[PointAnnotation | dict[str, Any]],
    pitch: PitchModel,
    *,
    video_width: int | float | None = None,
    video_height: int | float | None = None,
) -> tuple[PointAnnotation, ...]:
    if not isinstance(raw_annotations, Sequence) or isinstance(
        raw_annotations, (str, bytes)
    ):
        raise CalibrationError("Point annotations must be a sequence.")

    result: list[PointAnnotation] = []
    seen: set[str] = set()

    for raw in raw_annotations:
        if isinstance(raw, PointAnnotation):
            annotation = raw
        elif isinstance(raw, dict):
            name = raw.get("point_name")
            raw_point = raw.get("point")

            if not isinstance(name, str) or not name:
                raise CalibrationError(
                    "Each point annotation requires point_name."
                )

            annotation = PointAnnotation(
                name,
                _parse_image_point(raw_point, name),
            )
        else:
            raise CalibrationError(
                "Each point annotation must be an object."
            )

        pitch.point(annotation.point_name)

        if annotation.point_name in seen:
            raise CalibrationError(
                f"Semantic point {annotation.point_name!r} is annotated more than once."
            )
        seen.add(annotation.point_name)

        _validate_image_bounds(
            annotation.image_point,
            video_width,
            video_height,
            annotation.point_name,
        )
        result.append(annotation)

    return tuple(result)


def _image_line_from_annotation(annotation: LineAnnotation) -> np.ndarray:
    p1 = np.array([annotation.p1.u, annotation.p1.v, 1.0], dtype=float)
    p2 = np.array([annotation.p2.u, annotation.p2.v, 1.0], dtype=float)

    line = np.cross(p1, p2)
    norm = math.hypot(float(line[0]), float(line[1]))

    if norm < EPS:
        raise DegenerateCalibrationError(
            f"Observed line {annotation.line_name!r} is degenerate."
        )

    return line / norm


def _line_angle_deg(line: np.ndarray) -> float:
    # A direction vector of ax + by + c = 0 is (-b, a).
    return math.degrees(math.atan2(float(line[0]), float(-line[1])))


def _angular_distance_deg(a: float, b: float) -> float:
    distance = abs((a - b) % 180.0)
    return min(distance, 180.0 - distance)


def _reject_duplicate_observed_lines(
    annotations: Sequence[LineAnnotation],
    observed_lines: dict[str, np.ndarray],
    pitch: PitchModel,
) -> None:
    """Catch accidental duplicate image annotations within one pitch family."""

    for index, first in enumerate(annotations):
        first_pitch_line = pitch.line(first.line_name)
        l1 = observed_lines[first.line_name]

        for second in annotations[index + 1 :]:
            second_pitch_line = pitch.line(second.line_name)

            if first_pitch_line.family != second_pitch_line.family:
                continue

            l2 = observed_lines[second.line_name].copy()

            if (
                _angular_distance_deg(
                    _line_angle_deg(l1),
                    _line_angle_deg(l2),
                )
                > DUPLICATE_LINE_ANGLE_DEG
            ):
                continue

            if np.dot(l1[:2], l2[:2]) < 0:
                l2 = -l2

            if abs(float(l1[2] - l2[2])) <= DUPLICATE_LINE_OFFSET_PX:
                raise DegenerateCalibrationError(
                    f"Observed lines {first.line_name!r} and "
                    f"{second.line_name!r} are effectively duplicates."
                )


def _hartley_transform(points: np.ndarray, label: str) -> np.ndarray:
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
        raise DegenerateCalibrationError(
            f"{label} normalization requires at least two 2D samples."
        )

    centroid = points.mean(axis=0)
    shifted = points - centroid
    mean_distance = float(
        np.mean(np.sqrt(np.sum(shifted * shifted, axis=1)))
    )

    if mean_distance < EPS:
        raise DegenerateCalibrationError(
            f"{label} geometry has no spatial extent."
        )

    scale = math.sqrt(2.0) / mean_distance

    return np.array(
        [
            [scale, 0.0, -scale * centroid[0]],
            [0.0, scale, -scale * centroid[1]],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def _collect_normalization_samples(
    pitch: PitchModel,
    line_annotations: Sequence[LineAnnotation],
    point_annotations: Sequence[PointAnnotation],
) -> tuple[np.ndarray, np.ndarray]:
    """Collect representative source/image coordinates for conditioning.

    Human line-click endpoints are used only to define the image normalization
    scale. They are never treated as endpoint correspondences.
    """

    pitch_samples: list[list[float]] = []
    image_samples: list[list[float]] = []

    for annotation in point_annotations:
        canonical = pitch.point(annotation.point_name).point
        pitch_samples.append(canonical.as_list())
        image_samples.append(annotation.image_point.as_list())

    for annotation in line_annotations:
        canonical = pitch.line(annotation.line_name)

        # Two arbitrary known points on the canonical line are sufficient to
        # express its line constraint. They do NOT correspond to the two clicks.
        pitch_samples.append(canonical.start.as_list())
        pitch_samples.append(canonical.end.as_list())

        image_samples.append(annotation.p1.as_list())
        image_samples.append(annotation.p2.as_list())

    if len(pitch_samples) < 2 or len(image_samples) < 2:
        raise InsufficientGeometryError(
            "Insufficient geometry: more independent annotations are required."
        )

    return (
        np.asarray(pitch_samples, dtype=float),
        np.asarray(image_samples, dtype=float),
    )


def _normalized_image_line(
    image_line: np.ndarray,
    image_transform: np.ndarray,
) -> np.ndarray:
    """Transform an image line into normalized image coordinates.

    If x_n = T x, then l_n = T^{-T} l.
    """

    normalized = np.linalg.inv(image_transform).T @ image_line
    norm = math.hypot(float(normalized[0]), float(normalized[1]))

    if norm < EPS:
        raise DegenerateCalibrationError(
            "Normalized image line is degenerate."
        )

    return normalized / norm


def _point_constraint_rows(
    canonical: Point2D,
    observed: ImagePoint,
    pitch_transform: np.ndarray,
    image_transform: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    source = pitch_transform @ np.array(
        [canonical.x, canonical.y, 1.0],
        dtype=float,
    )
    destination = image_transform @ np.array(
        [observed.u, observed.v, 1.0],
        dtype=float,
    )

    x, y, w = source
    u = destination[0] / destination[2]
    v = destination[1] / destination[2]

    row_u = np.array(
        [-x, -y, -w, 0.0, 0.0, 0.0, u * x, u * y, u * w],
        dtype=float,
    )
    row_v = np.array(
        [0.0, 0.0, 0.0, -x, -y, -w, v * x, v * y, v * w],
        dtype=float,
    )

    return row_u, row_v


def _line_constraint_rows(
    canonical: LineSegment,
    observed_line: np.ndarray,
    pitch_transform: np.ndarray,
    image_transform: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return two direct constraints for one pitch-line correspondence.

    For H mapping pitch -> image and observed image line l:

        l^T H X = 0

    for every canonical point X on the known pitch line.

    Two distinct canonical points on that line give two independent linear
    constraints. The human click endpoints are NOT matched to these points.
    """

    line_normalized = _normalized_image_line(
        observed_line,
        image_transform,
    )

    rows: list[np.ndarray] = []

    for canonical_point in (canonical.start, canonical.end):
        source = pitch_transform @ np.array(
            [canonical_point.x, canonical_point.y, 1.0],
            dtype=float,
        )

        # vec(H) is row-major. kron(l, X) produces:
        # [l0*X0, l0*X1, l0*X2, l1*X0, ..., l2*X2]
        rows.append(np.kron(line_normalized, source))

    return rows[0], rows[1]


def _normalize_homography_scale(homography: np.ndarray) -> np.ndarray:
    if abs(float(homography[2, 2])) > EPS:
        return homography / homography[2, 2]

    norm = float(np.linalg.norm(homography))
    if norm < EPS:
        raise DegenerateCalibrationError("Homography scale is zero.")

    return homography / norm


def _solve_mixed_linear_homography(
    pitch: PitchModel,
    line_annotations: Sequence[LineAnnotation],
    point_annotations: Sequence[PointAnnotation],
    observed_lines: dict[str, np.ndarray],
) -> tuple[np.ndarray, int]:
    pitch_samples, image_samples = _collect_normalization_samples(
        pitch,
        line_annotations,
        point_annotations,
    )

    pitch_transform = _hartley_transform(
        pitch_samples,
        "Pitch",
    )
    image_transform = _hartley_transform(
        image_samples,
        "Image",
    )

    rows: list[np.ndarray] = []

    for annotation in point_annotations:
        canonical = pitch.point(annotation.point_name).point
        rows.extend(
            _point_constraint_rows(
                canonical,
                annotation.image_point,
                pitch_transform,
                image_transform,
            )
        )

    for annotation in line_annotations:
        canonical = pitch.line(annotation.line_name)
        rows.extend(
            _line_constraint_rows(
                canonical,
                observed_lines[annotation.line_name],
                pitch_transform,
                image_transform,
            )
        )

    matrix = np.asarray(rows, dtype=float)

    if matrix.ndim != 2 or matrix.shape[1] != 9 or len(matrix) < 8:
        raise InsufficientGeometryError(
            "Insufficient geometry: the current points and lines provide "
            "fewer than eight independent homography equations."
        )

    # A projective homography has 8 degrees of freedom (9 parameters up to
    # scale), so A must have rank >= 8.
    singular_values = np.linalg.svd(
        matrix,
        compute_uv=False,
    )
    tolerance = max(matrix.shape) * np.finfo(float).eps * singular_values[0]
    rank = int(np.sum(singular_values > tolerance))

    if rank < 8:
        raise InsufficientGeometryError(
            "Insufficient geometry: the current point/line configuration "
            f"has rank {rank}, but rank 8 is required. Add information in "
            "another part or direction of the pitch."
        )

    _, _, vt = np.linalg.svd(matrix)
    normalized_h = vt[-1].reshape(3, 3)

    homography = (
        np.linalg.inv(image_transform)
        @ normalized_h
        @ pitch_transform
    )
    homography = _normalize_homography_scale(homography)

    if not np.all(np.isfinite(homography)):
        raise DegenerateCalibrationError(
            "Solved homography contains non-finite values."
        )

    if np.linalg.matrix_rank(homography) < 3:
        raise DegenerateCalibrationError(
            "Solved homography is singular."
        )

    return homography, rank


def _project_one(
    x: float,
    y: float,
    homography: np.ndarray,
) -> ImagePoint:
    projected = homography @ np.array([x, y, 1.0], dtype=float)

    if abs(float(projected[2])) < EPS:
        raise DegenerateCalibrationError(
            "A pitch point projects to infinity."
        )

    return ImagePoint(
        float(projected[0] / projected[2]),
        float(projected[1] / projected[2]),
    )


def project_points(
    points: Sequence[Point2D | Sequence[float]],
    homography_pitch_to_image: np.ndarray,
) -> list[ImagePoint]:
    matrix = np.asarray(homography_pitch_to_image, dtype=float)

    if matrix.shape != (3, 3):
        raise CalibrationError("Homography must be 3x3.")

    result: list[ImagePoint] = []

    for point in points:
        if isinstance(point, Point2D):
            x, y = point.x, point.y
        else:
            x, y = float(point[0]), float(point[1])

        result.append(_project_one(x, y, matrix))

    return result


def image_to_pitch_points(
    points: Sequence[ImagePoint | Sequence[float]],
    homography_image_to_pitch: np.ndarray,
) -> list[Point2D]:
    matrix = np.asarray(homography_image_to_pitch, dtype=float)

    if matrix.shape != (3, 3):
        raise CalibrationError("Homography must be 3x3.")

    result: list[Point2D] = []

    for point in points:
        if isinstance(point, ImagePoint):
            u, v = point.u, point.v
        else:
            u, v = float(point[0]), float(point[1])

        projected = matrix @ np.array([u, v, 1.0], dtype=float)

        if abs(float(projected[2])) < EPS:
            raise DegenerateCalibrationError(
                "An image point maps to infinity."
            )

        result.append(
            Point2D(
                float(projected[0] / projected[2]),
                float(projected[1] / projected[2]),
            )
        )

    return result


def _homography_to_parameters(homography: np.ndarray) -> np.ndarray:
    h = _normalize_homography_scale(homography)

    if abs(float(h[2, 2])) < EPS:
        raise DegenerateCalibrationError(
            "Refinement requires a finite homography scale."
        )

    return np.array(
        [
            h[0, 0],
            h[0, 1],
            h[0, 2],
            h[1, 0],
            h[1, 1],
            h[1, 2],
            h[2, 0],
            h[2, 1],
        ],
        dtype=float,
    )


def _parameters_to_homography(parameters: np.ndarray) -> np.ndarray:
    if parameters.shape != (8,):
        raise CalibrationError(
            "Homography refinement expects eight parameters."
        )

    return np.array(
        [
            [parameters[0], parameters[1], parameters[2]],
            [parameters[3], parameters[4], parameters[5]],
            [parameters[6], parameters[7], 1.0],
        ],
        dtype=float,
    )


def _signed_point_to_line_distance(
    point: ImagePoint,
    line: np.ndarray,
) -> float:
    return float(
        line[0] * point.u
        + line[1] * point.v
        + line[2]
    )


def _geometric_residuals(
    parameters: np.ndarray,
    pitch: PitchModel,
    line_annotations: Sequence[LineAnnotation],
    point_annotations: Sequence[PointAnnotation],
    observed_lines: dict[str, np.ndarray],
) -> np.ndarray:
    """Residuals in native image pixels.

    Point annotation -> 2 residuals: du, dv.
    Line annotation  -> 2 residuals: perpendicular distance of two known
                        canonical points on that line to the observed image line.
    """

    homography = _parameters_to_homography(parameters)
    residuals: list[float] = []

    for annotation in point_annotations:
        canonical = pitch.point(annotation.point_name).point
        projected = _project_one(
            canonical.x,
            canonical.y,
            homography,
        )

        residuals.append(
            projected.u - annotation.image_point.u
        )
        residuals.append(
            projected.v - annotation.image_point.v
        )

    for annotation in line_annotations:
        canonical = pitch.line(annotation.line_name)
        observed = observed_lines[annotation.line_name]

        for canonical_point in (canonical.start, canonical.end):
            projected = _project_one(
                canonical_point.x,
                canonical_point.y,
                homography,
            )
            residuals.append(
                _signed_point_to_line_distance(
                    projected,
                    observed,
                )
            )

    return np.asarray(residuals, dtype=float)


def _huber_cost(
    residuals: np.ndarray,
    delta: float,
) -> float:
    absolute = np.abs(residuals)
    quadratic = absolute <= delta

    return float(
        np.sum(
            np.where(
                quadratic,
                0.5 * residuals * residuals,
                delta * (absolute - 0.5 * delta),
            )
        )
    )


def _huber_sqrt_weights(
    residuals: np.ndarray,
    delta: float,
) -> np.ndarray:
    absolute = np.abs(residuals)
    weights = np.ones_like(residuals)

    mask = absolute > delta
    weights[mask] = np.sqrt(
        delta / np.maximum(absolute[mask], EPS)
    )

    return weights


def _finite_difference_jacobian(
    parameters: np.ndarray,
    residual_function,
) -> np.ndarray:
    baseline = residual_function(parameters)

    jacobian = np.empty(
        (len(baseline), len(parameters)),
        dtype=float,
    )

    for index in range(len(parameters)):
        step = 1e-6 * max(1.0, abs(float(parameters[index])))

        plus = parameters.copy()
        minus = parameters.copy()
        plus[index] += step
        minus[index] -= step

        forward = residual_function(plus)
        backward = residual_function(minus)

        jacobian[:, index] = (forward - backward) / (2.0 * step)

    return jacobian


def _robust_refine_homography(
    initial_homography: np.ndarray,
    pitch: PitchModel,
    line_annotations: Sequence[LineAnnotation],
    point_annotations: Sequence[PointAnnotation],
    observed_lines: dict[str, np.ndarray],
    *,
    huber_delta_px: float = DEFAULT_HUBER_DELTA_PX,
    max_iterations: int = DEFAULT_MAX_REFINEMENT_ITERATIONS,
) -> tuple[np.ndarray, int]:
    """Small dependency-free Levenberg/IRLS refinement.

    This refinement is intentionally modest. The mixed normalized linear solve
    provides the initial H. Refinement then optimizes the geometrically meaningful
    native-pixel residuals without adding SciPy/OpenCV as dependencies.
    """

    parameters = _homography_to_parameters(
        initial_homography,
    )

    def residual_function(candidate: np.ndarray) -> np.ndarray:
        return _geometric_residuals(
            candidate,
            pitch,
            line_annotations,
            point_annotations,
            observed_lines,
        )

    residuals = residual_function(parameters)

    if not np.all(np.isfinite(residuals)):
        return initial_homography, 0

    current_cost = _huber_cost(
        residuals,
        huber_delta_px,
    )
    damping = 1e-3
    completed_iterations = 0

    for iteration in range(max_iterations):
        try:
            jacobian = _finite_difference_jacobian(
                parameters,
                residual_function,
            )
        except DegenerateCalibrationError:
            break

        if not np.all(np.isfinite(jacobian)):
            break

        weights = _huber_sqrt_weights(
            residuals,
            huber_delta_px,
        )
        weighted_jacobian = jacobian * weights[:, None]
        weighted_residuals = residuals * weights

        normal_matrix = (
            weighted_jacobian.T @ weighted_jacobian
        )
        gradient = (
            weighted_jacobian.T @ weighted_residuals
        )

        # Scale-aware LM damping.
        diagonal = np.diag(normal_matrix).copy()
        diagonal[diagonal < 1.0] = 1.0

        accepted = False

        for _ in range(8):
            system = normal_matrix + damping * np.diag(diagonal)

            try:
                step = np.linalg.solve(
                    system,
                    -gradient,
                )
            except np.linalg.LinAlgError:
                damping *= 10.0
                continue

            candidate = parameters + step

            try:
                candidate_residuals = residual_function(
                    candidate
                )
            except DegenerateCalibrationError:
                damping *= 10.0
                continue

            if not np.all(np.isfinite(candidate_residuals)):
                damping *= 10.0
                continue

            candidate_cost = _huber_cost(
                candidate_residuals,
                huber_delta_px,
            )

            if candidate_cost < current_cost:
                parameters = candidate
                residuals = candidate_residuals
                improvement = current_cost - candidate_cost
                current_cost = candidate_cost
                damping = max(damping / 3.0, 1e-9)
                accepted = True
                completed_iterations = iteration + 1

                if (
                    improvement < 1e-8
                    or float(np.linalg.norm(step)) < 1e-9
                ):
                    return (
                        _normalize_homography_scale(
                            _parameters_to_homography(parameters)
                        ),
                        completed_iterations,
                    )

                break

            damping *= 10.0

        if not accepted:
            break

    refined = _parameters_to_homography(parameters)

    if (
        not np.all(np.isfinite(refined))
        or abs(float(np.linalg.det(refined))) < EPS
    ):
        return initial_homography, 0

    return (
        _normalize_homography_scale(refined),
        completed_iterations,
    )


def _line_sample_points(
    line: LineSegment,
    sample_count: int = 21,
) -> list[Point2D]:
    samples: list[Point2D] = []

    for t in np.linspace(0.0, 1.0, sample_count):
        samples.append(
            Point2D(
                line.start.x
                + (line.end.x - line.start.x) * float(t),
                line.start.y
                + (line.end.y - line.start.y) * float(t),
            )
        )

    return samples


def _reprojection_metrics(
    pitch: PitchModel,
    line_annotations: Sequence[LineAnnotation],
    point_annotations: Sequence[PointAnnotation],
    observed_lines: dict[str, np.ndarray],
    homography: np.ndarray,
) -> dict[str, Any]:
    """Return balanced diagnostics.

    V1 mixed every sampled point of every line into the global metric, which
    could make one annotated line dominate the summary. V2 uses one representative
    error per annotation for the global summary:

    - point annotation -> its reprojection distance;
    - line annotation  -> median distance along that canonical line.

    Detailed sample-level metrics are still returned separately.
    """

    per_line: dict[str, Any] = {}
    line_sample_errors: list[float] = []
    annotation_errors: list[float] = []

    for annotation in line_annotations:
        canonical = pitch.line(annotation.line_name)
        observed = observed_lines[annotation.line_name]

        samples = _line_sample_points(
            canonical,
            sample_count=21,
        )
        projected = project_points(
            samples,
            homography,
        )

        errors = [
            abs(
                _signed_point_to_line_distance(
                    point,
                    observed,
                )
            )
            for point in projected
        ]

        median = float(np.median(errors))
        p95 = float(np.percentile(errors, 95))
        maximum = float(np.max(errors))

        per_line[annotation.line_name] = {
            "median_px": median,
            "p95_px": p95,
            "max_px": maximum,
        }

        line_sample_errors.extend(errors)
        annotation_errors.append(median)

    per_point: dict[str, float] = {}
    point_errors: list[float] = []

    for annotation in point_annotations:
        canonical = pitch.point(annotation.point_name).point
        predicted = _project_one(
            canonical.x,
            canonical.y,
            homography,
        )

        error = math.hypot(
            predicted.u - annotation.image_point.u,
            predicted.v - annotation.image_point.v,
        )

        point_errors.append(error)
        annotation_errors.append(error)
        per_point[annotation.point_name] = float(error)

    def summarize(values: Sequence[float]) -> dict[str, float]:
        if not values:
            return {
                "median_px": float("nan"),
                "p95_px": float("nan"),
                "max_px": float("nan"),
            }

        array = np.asarray(values, dtype=float)

        return {
            "median_px": float(np.median(array)),
            "p95_px": float(np.percentile(array, 95)),
            "max_px": float(np.max(array)),
        }

    balanced = summarize(annotation_errors)
    point_summary = summarize(point_errors)
    line_summary = summarize(
        [
            details["median_px"]
            for details in per_line.values()
        ]
    )
    line_sample_summary = summarize(
        line_sample_errors
    )

    return {
        # Existing top-level names retained for TEST compatibility.
        "median_px": balanced["median_px"],
        "p95_px": balanced["p95_px"],
        "max_px": balanced["max_px"],
        # New clearer diagnostics.
        "balanced_annotation_error": balanced,
        "point_error": point_summary,
        "line_error": line_summary,
        "line_sample_error": line_sample_summary,
        "point_sample_count": len(point_errors),
        "line_sample_count": len(line_sample_errors),
        "per_point": per_point,
        "per_line": per_line,
    }


def project_pitch_geometry(
    pitch: PitchModel,
    homography: np.ndarray,
) -> dict[str, Any]:
    lines: list[dict[str, Any]] = []

    for line in pitch.straight_lines:
        projected = project_points(
            [line.start, line.end],
            homography,
        )
        lines.append(
            {
                "name": line.name,
                "label": line.label,
                "family": line.family,
                "points": [
                    projected[0].as_list(),
                    projected[1].as_list(),
                ],
            }
        )

    curves: list[dict[str, Any]] = []

    for curve in pitch.curves:
        projected = project_points(
            curve.points,
            homography,
        )
        curves.append(
            {
                "name": curve.name,
                "label": curve.label,
                "closed": curve.closed,
                "points": [
                    point.as_list()
                    for point in projected
                ],
            }
        )

    points: list[dict[str, Any]] = []

    for pitch_point in pitch.points:
        projected = project_points(
            [pitch_point.point],
            homography,
        )[0]
        points.append(
            {
                "name": pitch_point.name,
                "label": pitch_point.label,
                "category": pitch_point.category,
                "point": projected.as_list(),
            }
        )

    return {
        "lines": lines,
        "curves": curves,
        "points": points,
    }


def solve_calibration(
    annotations: Sequence[LineAnnotation | dict[str, Any]] = (),
    *,
    point_annotations: Sequence[PointAnnotation | dict[str, Any]] = (),
    pitch_length: float = DEFAULT_PITCH_LENGTH,
    pitch_width: float = DEFAULT_PITCH_WIDTH,
    video_width: int | float | None = None,
    video_height: int | float | None = None,
    refine: bool = True,
) -> CalibrationResult:
    pitch = PitchModel(
        pitch_length,
        pitch_width,
    )

    lines = parse_line_annotations(
        annotations,
        pitch,
        video_width=video_width,
        video_height=video_height,
    )
    points = parse_point_annotations(
        point_annotations,
        pitch,
        video_width=video_width,
        video_height=video_height,
    )

    if not lines and not points:
        raise InsufficientGeometryError(
            "No calibration annotations were provided."
        )

    observed_lines = {
        annotation.line_name: _image_line_from_annotation(
            annotation
        )
        for annotation in lines
    }

    _reject_duplicate_observed_lines(
        lines,
        observed_lines,
        pitch,
    )

    linear_homography, linear_rank = (
        _solve_mixed_linear_homography(
            pitch,
            lines,
            points,
            observed_lines,
        )
    )

    homography = linear_homography
    refinement_iterations = 0

    if refine:
        homography, refinement_iterations = (
            _robust_refine_homography(
                linear_homography,
                pitch,
                lines,
                points,
                observed_lines,
            )
        )

    if not np.all(np.isfinite(homography)):
        raise DegenerateCalibrationError(
            "Solved homography contains non-finite values."
        )

    determinant = float(
        np.linalg.det(homography)
    )

    if (
        not math.isfinite(determinant)
        or abs(determinant) < EPS
    ):
        raise DegenerateCalibrationError(
            "Solved homography is singular."
        )

    try:
        inverse = _normalize_homography_scale(
            np.linalg.inv(homography)
        )
    except np.linalg.LinAlgError as exc:
        raise DegenerateCalibrationError(
            "Solved homography cannot be inverted."
        ) from exc

    metrics = _reprojection_metrics(
        pitch,
        lines,
        points,
        observed_lines,
        homography,
    )

    projected_geometry = project_pitch_geometry(
        pitch,
        homography,
    )

    result = CalibrationResult(
        pitch_length=pitch.length,
        pitch_width=pitch.width,
        homography_pitch_to_image=homography,
        homography_image_to_pitch=inverse,
        line_annotations=lines,
        point_annotations=points,
        metrics=metrics,
        projected_geometry=projected_geometry,
        linear_rank=linear_rank,
        refinement_iterations=refinement_iterations,
    )

    # Populate canonical direct point coordinates in the serialized structure
    # without changing the dataclass representation.
    return result


def solve_calibration_payload(
    payload: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise CalibrationError(
            "Calibration payload must be an object."
        )

    result = solve_calibration(
        payload.get("annotations", []),
        point_annotations=payload.get(
            "point_annotations",
            [],
        ),
        pitch_length=float(
            payload.get(
                "pitch_length",
                DEFAULT_PITCH_LENGTH,
            )
        ),
        pitch_width=float(
            payload.get(
                "pitch_width",
                DEFAULT_PITCH_WIDTH,
            )
        ),
        video_width=payload.get("video_width"),
        video_height=payload.get("video_height"),
    )

    serialized = result.as_dict()

    # Fill canonical point coordinates for API compatibility/diagnostics.
    pitch = PitchModel(
        result.pitch_length,
        result.pitch_width,
    )

    for correspondence in serialized["direct_correspondences"]:
        canonical = pitch.point(
            correspondence["point_name"]
        ).point
        correspondence["pitch_point"] = canonical.as_list()

    return serialized


# ---------------------------------------------------------------------------
# Dependency-free synthetic diagnostics
# ---------------------------------------------------------------------------

def _synthetic_homography() -> np.ndarray:
    return np.array(
        [
            [12.0, -2.3, 260.0],
            [1.8, 7.8, 180.0],
            [0.0045, -0.0018, 1.0],
        ],
        dtype=float,
    )


def _annotation_from_projected_line(
    pitch: PitchModel,
    name: str,
    homography: np.ndarray,
    *,
    start_fraction: float = 0.2,
    end_fraction: float = 0.75,
) -> LineAnnotation:
    """Simulate a human marking only part of a visible canonical line."""

    line = pitch.line(name)

    source_a = Point2D(
        line.start.x
        + (line.end.x - line.start.x) * start_fraction,
        line.start.y
        + (line.end.y - line.start.y) * start_fraction,
    )
    source_b = Point2D(
        line.start.x
        + (line.end.x - line.start.x) * end_fraction,
        line.start.y
        + (line.end.y - line.start.y) * end_fraction,
    )

    projected = project_points(
        [source_a, source_b],
        homography,
    )

    return LineAnnotation(
        name,
        projected[0],
        projected[1],
    )


def _point_annotation_from_projection(
    pitch: PitchModel,
    name: str,
    homography: np.ndarray,
) -> PointAnnotation:
    canonical = pitch.point(name).point
    projected = project_points(
        [canonical],
        homography,
    )[0]

    return PointAnnotation(
        name,
        projected,
    )


def _matrix_error(
    solved: np.ndarray,
    known: np.ndarray,
) -> float:
    a = _normalize_homography_scale(solved)
    b = _normalize_homography_scale(known)

    return float(
        np.max(np.abs(a - b))
    )


def _self_check() -> None:
    pitch = PitchModel()
    known = _normalize_homography_scale(
        _synthetic_homography()
    )

    # 1) Points-only: preserve V1 capability.
    point_only = [
        _point_annotation_from_projection(
            pitch,
            "corner_left_a",
            known,
        ),
        _point_annotation_from_projection(
            pitch,
            "corner_right_a",
            known,
        ),
        _point_annotation_from_projection(
            pitch,
            "corner_left_b",
            known,
        ),
        _point_annotation_from_projection(
            pitch,
            "right_penalty_front_b",
            known,
        ),
    ]

    solved_points = solve_calibration(
        point_annotations=point_only,
    )
    point_error = _matrix_error(
        solved_points.homography_pitch_to_image,
        known,
    )

    assert point_error < 1e-7, point_error

    # 2) Lines-only: four well-distributed semantic lines. Human clicks cover
    # only partial visible segments, not canonical endpoints.
    line_only = [
        _annotation_from_projected_line(
            pitch,
            "touchline_a",
            known,
            start_fraction=0.12,
            end_fraction=0.48,
        ),
        _annotation_from_projected_line(
            pitch,
            "touchline_b",
            known,
            start_fraction=0.35,
            end_fraction=0.82,
        ),
        _annotation_from_projected_line(
            pitch,
            "halfway_line",
            known,
            start_fraction=0.18,
            end_fraction=0.62,
        ),
        _annotation_from_projected_line(
            pitch,
            "right_penalty_front",
            known,
            start_fraction=0.22,
            end_fraction=0.77,
        ),
    ]

    solved_lines = solve_calibration(
        line_only,
    )
    line_error = _matrix_error(
        solved_lines.homography_pitch_to_image,
        known,
    )

    assert line_error < 1e-7, line_error

    # 3) Mixed case: lines contribute directly, with no fabricated line
    # intersections.
    mixed_lines = [
        _annotation_from_projected_line(
            pitch,
            "touchline_b",
            known,
            start_fraction=0.28,
            end_fraction=0.66,
        ),
        _annotation_from_projected_line(
            pitch,
            "right_penalty_front",
            known,
            start_fraction=0.15,
            end_fraction=0.72,
        ),
    ]
    mixed_points = [
        _point_annotation_from_projection(
            pitch,
            "right_penalty_front_a",
            known,
        ),
        _point_annotation_from_projection(
            pitch,
            "right_goal_area_front_b",
            known,
        ),
        _point_annotation_from_projection(
            pitch,
            "left_penalty_front_b",
            known,
        ),
    ]

    solved_mixed = solve_calibration(
        mixed_lines,
        point_annotations=mixed_points,
    )
    mixed_error = _matrix_error(
        solved_mixed.homography_pitch_to_image,
        known,
    )

    assert mixed_error < 1e-7, mixed_error

    # 4) Insufficient geometry: three parallel pitch lines cannot determine H.
    insufficient = [
        _annotation_from_projected_line(
            pitch,
            "goal_line_left",
            known,
        ),
        _annotation_from_projected_line(
            pitch,
            "halfway_line",
            known,
        ),
        _annotation_from_projected_line(
            pitch,
            "goal_line_right",
            known,
        ),
    ]

    rejected = False

    try:
        solve_calibration(
            insufficient,
        )
    except InsufficientGeometryError:
        rejected = True

    assert rejected

    print("Calibration solver V2 self-check: OK")
    print(
        "Point-only recovery max matrix error: "
        f"{point_error:.3e}"
    )
    print(
        "Line-only partial-segment recovery max matrix error: "
        f"{line_error:.3e}"
    )
    print(
        "Mixed line+point recovery max matrix error: "
        f"{mixed_error:.3e}"
    )
    print(
        "Line intersections fabricated by solver: 0"
    )
    print(
        "Insufficient-geometry rejection: OK"
    )
    print(
        "V2 line meaning: clicks define the observed image line only; "
        "they are NOT canonical endpoint correspondences."
    )


if __name__ == "__main__":
    _self_check()
