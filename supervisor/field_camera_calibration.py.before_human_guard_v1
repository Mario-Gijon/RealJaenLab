"""Geometric projective-camera calibration for the RealJaen Supervisor.

This module calibrates ONE manual FIELD anchor from semantic observations of the
known 105 x 68 m pitch.

The important difference from the previous mixed solver is that semantic lines
are treated as true projective line correspondences, not as a handful of
artificial point samples that may be ignored by a robust loss.

A manual segment labelled ``bottom_touchline`` means:

    the canonical world line y = +34 must project onto THIS image line.

The two image endpoints only define the observed infinite image line. They are
NOT assumed to be the physical endpoints of the 105 m touchline.

Why a projective camera instead of a rigid pinhole camera here?
---------------------------------------------------------------
Veo can apply virtual pan/tilt/zoom, cropping, stitching and stabilisation. A
planar world->image homography is therefore the safest representation of the
effective camera seen by the Supervisor. Pan/tilt/zoom are naturally expressed
by a time-varying homography. Later, multiple manual anchors can be joined into
a smooth temporal camera trajectory.

This V1 focuses on making each anchor geometrically strong:

* points -> point correspondences;
* lines  -> exact semantic world-line / image-line correspondences;
* arcs   -> circular world constraints from the user's traced image samples;
* manual line residuals are NOT robustly down-weighted as outliers;
* point/arc residuals use robust IRLS so a slightly bad click cannot dominate
  a carefully drawn semantic boundary;
* mixed DLT uses Hartley-style coordinate normalisation;
* diagnostics expose per-point and per-line errors, vanishing points and a
  local zoom proxy for future temporal camera modelling.

No FastAPI or UI code lives here.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
import math
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class CameraCalibrationConfig:
    """Pixel-noise assumptions and numerical limits.

    These values are not arbitrary relative feature "importance" weights.
    They are interpreted as approximate manual annotation uncertainty.
    A semantic line is intentionally tighter than a point click because the
    user normally draws a long visible painted line.
    """

    point_sigma_px: float = 3.5
    line_sigma_px: float = 1.5
    arc_sigma_px: float = 4.0

    point_huber_delta_sigma: float = 3.0
    arc_huber_delta_sigma: float = 3.0

    max_iterations: int = 45
    initial_damping: float = 1e-3
    max_condition: float = 1e10

    # A solved manual semantic line should normally be very close to its image
    # observation. This is diagnostic, not a hard rejection threshold.
    good_line_error_px: float = 5.0
    warning_line_error_px: float = 12.0


# ---------------------------------------------------------------------------
# Basic projective geometry
# ---------------------------------------------------------------------------


def _finite_h(matrix: Any, max_condition: float) -> np.ndarray | None:
    try:
        h = np.asarray(matrix, dtype=np.float64)
    except (TypeError, ValueError):
        return None

    if h.shape != (3, 3) or not np.isfinite(h).all():
        return None

    if abs(float(h[2, 2])) < 1e-12:
        return None

    h = h / float(h[2, 2])

    try:
        det = float(np.linalg.det(h))
        cond = float(np.linalg.cond(h))
    except np.linalg.LinAlgError:
        return None

    if (
        not np.isfinite(det)
        or not np.isfinite(cond)
        or abs(det) < 1e-12
        or cond > max_condition
    ):
        return None

    return h


def _project_points(
    world_xy: np.ndarray,
    h: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Project points and return a validity mask instead of failing globally."""

    points = np.asarray(world_xy, dtype=np.float64).reshape(-1, 2)
    homogeneous = np.column_stack(
        (points, np.ones(len(points), dtype=np.float64))
    )
    projected_h = homogeneous @ h.T
    denominator = projected_h[:, 2]

    valid = (
        np.isfinite(projected_h).all(axis=1)
        & np.isfinite(denominator)
        & (np.abs(denominator) > 1e-9)
    )

    projected = np.full((len(points), 2), np.nan, dtype=np.float64)
    projected[valid] = (
        projected_h[valid, :2] / denominator[valid, None]
    )
    valid &= np.isfinite(projected).all(axis=1)
    return projected, valid


def _point_to_homogeneous(point: dict[str, Any]) -> np.ndarray | None:
    try:
        return np.array(
            [float(point["x"]), float(point["y"]), 1.0],
            dtype=np.float64,
        )
    except (KeyError, TypeError, ValueError):
        return None


def _line_from_endpoints(
    start: dict[str, Any],
    end: dict[str, Any],
) -> np.ndarray | None:
    p1 = _point_to_homogeneous(start)
    p2 = _point_to_homogeneous(end)
    if p1 is None or p2 is None:
        return None

    line = np.cross(p1, p2)
    norm = math.hypot(float(line[0]), float(line[1]))
    if norm < 1e-9 or not np.isfinite(line).all():
        return None

    return line / norm


def _align_line_sign(
    line: np.ndarray,
    reference: np.ndarray,
) -> np.ndarray:
    return -line if float(np.dot(line, reference)) < 0.0 else line


def _world_line(constraint: dict[str, Any]) -> np.ndarray | None:
    world = constraint.get("world", {})
    return _line_from_endpoints(
        world.get("start", {}),
        world.get("end", {}),
    )


def _image_line(constraint: dict[str, Any]) -> np.ndarray | None:
    image = constraint.get("image", {})
    return _line_from_endpoints(
        image.get("start", {}),
        image.get("end", {}),
    )


def _transform_world_line_to_image(
    world_line: np.ndarray,
    h: np.ndarray,
) -> np.ndarray | None:
    """Map a world-plane line to its exact image line: l ~ H^-T L."""

    try:
        predicted = np.linalg.inv(h).T @ world_line
    except np.linalg.LinAlgError:
        return None

    norm = math.hypot(
        float(predicted[0]),
        float(predicted[1]),
    )
    if norm < 1e-9 or not np.isfinite(predicted).all():
        return None

    return predicted / norm


# ---------------------------------------------------------------------------
# Constraint parsing
# ---------------------------------------------------------------------------


def _parse_points(
    constraints: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    parsed: list[dict[str, Any]] = []

    for item in constraints:
        try:
            world = np.array(
                [
                    float(item["world"]["x"]),
                    float(item["world"]["y"]),
                ],
                dtype=np.float64,
            )
            image = np.array(
                [
                    float(item["image"]["x"]),
                    float(item["image"]["y"]),
                ],
                dtype=np.float64,
            )
        except (KeyError, TypeError, ValueError):
            continue

        if not np.isfinite(world).all() or not np.isfinite(image).all():
            continue

        parsed.append(
            {
                "semantic": item.get("semantic"),
                "world": world,
                "image": image,
                "source": item.get("source", "manual"),
            }
        )

    return parsed


def _parse_lines(
    constraints: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    parsed: list[dict[str, Any]] = []

    for item in constraints:
        world_line = _world_line(item)
        observed_line = _image_line(item)
        if world_line is None or observed_line is None:
            continue

        try:
            world_start = np.array(
                [
                    float(item["world"]["start"]["x"]),
                    float(item["world"]["start"]["y"]),
                ],
                dtype=np.float64,
            )
            world_end = np.array(
                [
                    float(item["world"]["end"]["x"]),
                    float(item["world"]["end"]["y"]),
                ],
                dtype=np.float64,
            )
            image_start = np.array(
                [
                    float(item["image"]["start"]["x"]),
                    float(item["image"]["start"]["y"]),
                ],
                dtype=np.float64,
            )
            image_end = np.array(
                [
                    float(item["image"]["end"]["x"]),
                    float(item["image"]["end"]["y"]),
                ],
                dtype=np.float64,
            )
        except (KeyError, TypeError, ValueError):
            continue

        if np.linalg.norm(image_end - image_start) < 8.0:
            # A tiny image segment does not establish line orientation reliably.
            continue

        direction = world_end - world_start
        if abs(direction[0]) >= abs(direction[1]):
            family = "longitudinal_x"
        else:
            family = "transversal_y"

        parsed.append(
            {
                "semantic": item.get("semantic"),
                "world_line": world_line,
                "observed_line": observed_line,
                "world_start": world_start,
                "world_end": world_end,
                "image_start": image_start,
                "image_end": image_end,
                "family": family,
                "source": item.get("source", "manual"),
            }
        )

    return parsed


def _sample_fitted_ellipse(
    geometry: dict[str, Any],
    count: int = 24,
) -> np.ndarray | None:
    try:
        center = geometry["center"]
        cx = float(center["x"])
        cy = float(center["y"])
        rx = abs(float(geometry["radiusX"]))
        ry = abs(float(geometry["radiusY"]))
        rotation = float(geometry.get("rotation", 0.0))

        arc = geometry.get("arc", {})
        start_angle = float(arc.get("startAngle", 0.0))
        end_angle = float(arc.get("endAngle", 2.0 * math.pi))
        counterclockwise = bool(
            arc.get("counterclockwise", False)
        )
    except (KeyError, TypeError, ValueError):
        return None

    if rx < 1.0 or ry < 1.0:
        return None

    if counterclockwise:
        while end_angle > start_angle:
            end_angle -= 2.0 * math.pi
    else:
        while end_angle < start_angle:
            end_angle += 2.0 * math.pi

    angles = np.linspace(start_angle, end_angle, count)
    ca = math.cos(rotation)
    sa = math.sin(rotation)

    x = rx * np.cos(angles)
    y = ry * np.sin(angles)

    return np.column_stack(
        (
            cx + ca * x - sa * y,
            cy + sa * x + ca * y,
        )
    )


def _parse_arcs(
    constraints: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    parsed: list[dict[str, Any]] = []

    for item in constraints:
        try:
            world_center = np.array(
                [
                    float(item["world"]["center"]["x"]),
                    float(item["world"]["center"]["y"]),
                ],
                dtype=np.float64,
            )
            world_radius = float(item["world"]["radius"])
        except (KeyError, TypeError, ValueError):
            continue

        if world_radius <= 0:
            continue

        image = item.get("image", {})
        samples: np.ndarray | None = None

        raw_samples = (
            image.get("samples")
            or item.get("fit", {}).get("samples")
        )
        if raw_samples:
            try:
                samples = np.asarray(
                    [
                        [float(p["x"]), float(p["y"])]
                        for p in raw_samples
                    ],
                    dtype=np.float64,
                )
            except (KeyError, TypeError, ValueError):
                samples = None

        if samples is None or len(samples) < 4:
            fitted = (
                image.get("fitted")
                or image.get("geometry")
                or image
            )
            if isinstance(fitted, dict):
                samples = _sample_fitted_ellipse(fitted)

        if (
            samples is None
            or len(samples) < 4
            or not np.isfinite(samples).all()
        ):
            continue

        # Do not let a very densely sampled mouse trace accidentally give the
        # arc hundreds of times more influence than one semantic line.
        if len(samples) > 32:
            indices = np.linspace(
                0,
                len(samples) - 1,
                32,
            ).round().astype(int)
            samples = samples[indices]

        parsed.append(
            {
                "semantic": item.get("semantic"),
                "world_center": world_center,
                "world_radius": world_radius,
                "image_samples": samples,
                "source": item.get("source", "manual"),
            }
        )

    return parsed


# ---------------------------------------------------------------------------
# Normalised mixed DLT
# ---------------------------------------------------------------------------


def _normalisation_transform(
    points: np.ndarray,
) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(points) == 0:
        return np.eye(3, dtype=np.float64)

    center = np.mean(points, axis=0)
    distance = np.linalg.norm(points - center, axis=1)
    mean_distance = float(np.mean(distance))

    scale = (
        math.sqrt(2.0) / mean_distance
        if mean_distance > 1e-9
        else 1.0
    )

    return np.array(
        [
            [scale, 0.0, -scale * center[0]],
            [0.0, scale, -scale * center[1]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _apply_transform_point(
    point: np.ndarray,
    transform: np.ndarray,
) -> np.ndarray:
    homogeneous = transform @ np.array(
        [float(point[0]), float(point[1]), 1.0],
        dtype=np.float64,
    )
    return homogeneous / homogeneous[2]


def _mixed_normalised_dlt(
    points: list[dict[str, Any]],
    lines: list[dict[str, Any]],
    max_condition: float,
) -> np.ndarray | None:
    """Solve points + semantic line incidence in one normalised linear system."""

    world_for_norm: list[np.ndarray] = []
    image_for_norm: list[np.ndarray] = []

    for item in points:
        world_for_norm.append(item["world"])
        image_for_norm.append(item["image"])

    for item in lines:
        world_for_norm.extend(
            (item["world_start"], item["world_end"])
        )
        image_for_norm.extend(
            (item["image_start"], item["image_end"])
        )

    if len(world_for_norm) < 4 or len(image_for_norm) < 4:
        return None

    tw = _normalisation_transform(
        np.asarray(world_for_norm, dtype=np.float64)
    )
    ti = _normalisation_transform(
        np.asarray(image_for_norm, dtype=np.float64)
    )

    rows: list[list[float]] = []

    for item in points:
        world_h = _apply_transform_point(
            item["world"],
            tw,
        )
        image_h = _apply_transform_point(
            item["image"],
            ti,
        )

        x, y, w = map(float, world_h)
        u, v = map(float, image_h[:2])

        rows.append(
            [-x, -y, -w, 0, 0, 0, u * x, u * y, u * w]
        )
        rows.append(
            [0, 0, 0, -x, -y, -w, v * x, v * y, v * w]
        )

    for item in lines:
        image_start_h = _apply_transform_point(
            item["image_start"],
            ti,
        )
        image_end_h = _apply_transform_point(
            item["image_end"],
            ti,
        )
        observed_line = np.cross(
            image_start_h,
            image_end_h,
        )

        norm = math.hypot(
            float(observed_line[0]),
            float(observed_line[1]),
        )
        if norm < 1e-9:
            continue
        observed_line /= norm

        # Two independent world points on the semantic world line impose:
        #       l_image^T H X_world = 0
        # without ever dividing by the projective denominator.
        for endpoint in (
            item["world_start"],
            item["world_end"],
        ):
            world_h = _apply_transform_point(
                endpoint,
                tw,
            )
            x, y, w = map(float, world_h)
            a, b, c = map(float, observed_line)

            rows.append(
                [
                    a * x,
                    a * y,
                    a * w,
                    b * x,
                    b * y,
                    b * w,
                    c * x,
                    c * y,
                    c * w,
                ]
            )

    if len(rows) < 8:
        return None

    matrix = np.asarray(rows, dtype=np.float64)
    if np.linalg.matrix_rank(matrix) < 8:
        return None

    try:
        _, _, vt = np.linalg.svd(matrix)
    except np.linalg.LinAlgError:
        return None

    h_normalised = vt[-1].reshape(3, 3)

    try:
        h = np.linalg.inv(ti) @ h_normalised @ tw
    except np.linalg.LinAlgError:
        return None

    if abs(float(h[2, 2])) < 1e-12:
        return None

    h /= float(h[2, 2])
    return _finite_h(h, max_condition)


def _point_baseline(
    points: list[dict[str, Any]],
    max_condition: float,
) -> np.ndarray | None:
    if len(points) < 4:
        return None

    world = np.asarray(
        [item["world"] for item in points],
        dtype=np.float64,
    )
    image = np.asarray(
        [item["image"] for item in points],
        dtype=np.float64,
    )

    centered_world = world - np.mean(world, axis=0)
    centered_image = image - np.mean(image, axis=0)

    if (
        np.linalg.matrix_rank(centered_world) < 2
        or np.linalg.matrix_rank(centered_image) < 2
    ):
        return None

    h, _ = cv2.findHomography(
        world,
        image,
        method=0,
    )
    return _finite_h(h, max_condition)


# ---------------------------------------------------------------------------
# Non-linear geometric refinement
# ---------------------------------------------------------------------------


def _h_to_params(h: np.ndarray) -> np.ndarray:
    h = h / h[2, 2]
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
        dtype=np.float64,
    )


def _params_to_h(params: np.ndarray) -> np.ndarray:
    return np.array(
        [
            [params[0], params[1], params[2]],
            [params[3], params[4], params[5]],
            [params[6], params[7], 1.0],
        ],
        dtype=np.float64,
    )


def _local_pixel_scale(
    world_center: np.ndarray,
    world_radius: float,
    h: np.ndarray,
) -> float | None:
    sample = np.asarray(
        [
            world_center,
            world_center
            + np.array([world_radius, 0.0]),
        ],
        dtype=np.float64,
    )
    projected, valid = _project_points(sample, h)
    if not np.all(valid):
        return None

    px_distance = float(
        np.linalg.norm(projected[1] - projected[0])
    )
    if not np.isfinite(px_distance) or world_radius <= 0:
        return None

    return px_distance / world_radius


def _residual_vector(
    h: np.ndarray,
    points: list[dict[str, Any]],
    lines: list[dict[str, Any]],
    arcs: list[dict[str, Any]],
    config: CameraCalibrationConfig,
) -> tuple[np.ndarray, list[str]] | None:
    residuals: list[float] = []
    kinds: list[str] = []

    if points:
        world = np.asarray(
            [item["world"] for item in points],
            dtype=np.float64,
        )
        observed = np.asarray(
            [item["image"] for item in points],
            dtype=np.float64,
        )
        projected, valid = _project_points(world, h)
        if not np.all(valid):
            return None

        point_residual = (
            projected - observed
        ) / config.point_sigma_px

        residuals.extend(point_residual.reshape(-1).tolist())
        kinds.extend(["point"] * point_residual.size)

    for item in lines:
        predicted = _transform_world_line_to_image(
            item["world_line"],
            h,
        )
        if predicted is None:
            return None

        predicted = _align_line_sign(
            predicted,
            item["observed_line"],
        )

        # The human-drawn endpoints are merely two samples on the observed
        # infinite image line. Requiring both to lie on the projected semantic
        # line enforces both its offset AND its orientation.
        for endpoint in (
            item["image_start"],
            item["image_end"],
        ):
            distance_px = float(
                predicted[0] * endpoint[0]
                + predicted[1] * endpoint[1]
                + predicted[2]
            )
            residuals.append(
                distance_px / config.line_sigma_px
            )
            kinds.append("line")

    if arcs:
        try:
            inverse_h = np.linalg.inv(h)
        except np.linalg.LinAlgError:
            return None

        for item in arcs:
            image_samples = item["image_samples"]
            world_back, valid = _project_points(
                image_samples,
                inverse_h,
            )
            if int(np.count_nonzero(valid)) < 4:
                return None

            world_back = world_back[valid]
            radial_error_m = (
                np.linalg.norm(
                    world_back
                    - item["world_center"][None, :],
                    axis=1,
                )
                - item["world_radius"]
            )

            px_per_metre = _local_pixel_scale(
                item["world_center"],
                item["world_radius"],
                h,
            )
            if px_per_metre is None:
                continue

            radial_error_px = (
                radial_error_m * px_per_metre
            )
            arc_residual = (
                radial_error_px / config.arc_sigma_px
            )

            residuals.extend(arc_residual.tolist())
            kinds.extend(["arc"] * len(arc_residual))

    if not residuals:
        return None

    vector = np.asarray(residuals, dtype=np.float64)
    if not np.isfinite(vector).all():
        return None

    return vector, kinds


def _robust_weights(
    residuals: np.ndarray,
    kinds: list[str],
    config: CameraCalibrationConfig,
) -> np.ndarray:
    """IRLS weights.

    Semantic manual lines remain full-weight. This is deliberate: if the user
    says a visible painted line is ``bottom_touchline``, the solver must not
    silently classify that line as an outlier just because clustered point
    clicks prefer a different extrapolation.
    """

    weights = np.ones(len(residuals), dtype=np.float64)

    for index, kind in enumerate(kinds):
        value = abs(float(residuals[index]))

        if kind == "line":
            weights[index] = 1.0
            continue

        delta = (
            config.point_huber_delta_sigma
            if kind == "point"
            else config.arc_huber_delta_sigma
        )

        if value > delta:
            weights[index] = delta / max(value, 1e-12)

    return weights


def _weighted_cost(
    residuals: np.ndarray,
    kinds: list[str],
    config: CameraCalibrationConfig,
) -> float:
    weights = _robust_weights(
        residuals,
        kinds,
        config,
    )
    return float(
        0.5 * np.sum(weights * residuals * residuals)
    )


def _optimise_h(
    initial_h: np.ndarray,
    points: list[dict[str, Any]],
    lines: list[dict[str, Any]],
    arcs: list[dict[str, Any]],
    config: CameraCalibrationConfig,
) -> np.ndarray | None:
    params = _h_to_params(initial_h)
    damping = config.initial_damping

    evaluated = _residual_vector(
        initial_h,
        points,
        lines,
        arcs,
        config,
    )
    if evaluated is None:
        return None

    residuals, kinds = evaluated
    best_cost = _weighted_cost(
        residuals,
        kinds,
        config,
    )

    for _ in range(config.max_iterations):
        h = _finite_h(
            _params_to_h(params),
            config.max_condition,
        )
        if h is None:
            return None

        evaluated = _residual_vector(
            h,
            points,
            lines,
            arcs,
            config,
        )
        if evaluated is None:
            return None

        residuals, kinds = evaluated
        weights = _robust_weights(
            residuals,
            kinds,
            config,
        )

        jacobian = np.zeros(
            (len(residuals), 8),
            dtype=np.float64,
        )

        for column in range(8):
            step = 1e-6 * max(
                1.0,
                abs(float(params[column])),
            )
            candidate_params = params.copy()
            candidate_params[column] += step

            candidate_h = _finite_h(
                _params_to_h(candidate_params),
                config.max_condition,
            )
            if candidate_h is None:
                jacobian[:, column] = 0.0
                continue

            candidate_eval = _residual_vector(
                candidate_h,
                points,
                lines,
                arcs,
                config,
            )
            if candidate_eval is None:
                jacobian[:, column] = 0.0
                continue

            candidate_residuals, _ = candidate_eval
            if len(candidate_residuals) != len(residuals):
                return None

            jacobian[:, column] = (
                candidate_residuals - residuals
            ) / step

        sqrt_w = np.sqrt(weights)
        weighted_j = jacobian * sqrt_w[:, None]
        weighted_r = residuals * sqrt_w

        lhs = (
            weighted_j.T @ weighted_j
            + damping * np.eye(8, dtype=np.float64)
        )
        rhs = -(weighted_j.T @ weighted_r)

        try:
            delta = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            damping *= 10.0
            continue

        if not np.isfinite(delta).all():
            return None

        candidate_params = params + delta
        candidate_h = _finite_h(
            _params_to_h(candidate_params),
            config.max_condition,
        )
        if candidate_h is None:
            damping *= 5.0
            continue

        candidate_eval = _residual_vector(
            candidate_h,
            points,
            lines,
            arcs,
            config,
        )
        if candidate_eval is None:
            damping *= 5.0
            continue

        candidate_residuals, candidate_kinds = (
            candidate_eval
        )
        candidate_cost = _weighted_cost(
            candidate_residuals,
            candidate_kinds,
            config,
        )

        if candidate_cost < best_cost:
            params = candidate_params
            best_cost = candidate_cost
            damping = max(1e-8, damping * 0.35)

            if float(np.linalg.norm(delta)) < 1e-7:
                break
        else:
            damping = min(1e8, damping * 4.0)

    return _finite_h(
        _params_to_h(params),
        config.max_condition,
    )


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def _point_diagnostics(
    h: np.ndarray,
    points: list[dict[str, Any]],
) -> tuple[float | None, list[dict[str, Any]]]:
    if not points:
        return None, []

    world = np.asarray(
        [item["world"] for item in points],
        dtype=np.float64,
    )
    observed = np.asarray(
        [item["image"] for item in points],
        dtype=np.float64,
    )

    projected, valid = _project_points(world, h)
    details: list[dict[str, Any]] = []
    values: list[float] = []

    for index, item in enumerate(points):
        if not valid[index]:
            error = None
        else:
            error = float(
                np.linalg.norm(
                    projected[index] - observed[index]
                )
            )
            values.append(error)

        details.append(
            {
                "semantic": item.get("semantic"),
                "error_px": error,
            }
        )

    mean = (
        float(np.mean(values))
        if values
        else None
    )
    return mean, details


def _line_diagnostics(
    h: np.ndarray,
    lines: list[dict[str, Any]],
) -> tuple[float | None, list[dict[str, Any]]]:
    values: list[float] = []
    details: list[dict[str, Any]] = []

    for item in lines:
        predicted = _transform_world_line_to_image(
            item["world_line"],
            h,
        )
        if predicted is None:
            details.append(
                {
                    "semantic": item.get("semantic"),
                    "error_px": None,
                    "family": item.get("family"),
                }
            )
            continue

        predicted = _align_line_sign(
            predicted,
            item["observed_line"],
        )

        endpoint_errors = []
        for endpoint in (
            item["image_start"],
            item["image_end"],
        ):
            endpoint_errors.append(
                abs(
                    float(
                        predicted[0] * endpoint[0]
                        + predicted[1] * endpoint[1]
                        + predicted[2]
                    )
                )
            )

        error = float(np.mean(endpoint_errors))
        values.append(error)

        if error <= 5.0:
            quality = "good"
        elif error <= 12.0:
            quality = "warning"
        else:
            quality = "bad"

        details.append(
            {
                "semantic": item.get("semantic"),
                "error_px": error,
                "endpoint_errors_px": endpoint_errors,
                "family": item.get("family"),
                "quality": quality,
            }
        )

    mean = (
        float(np.mean(values))
        if values
        else None
    )
    return mean, details


def _vanishing_point(
    h: np.ndarray,
    world_direction: tuple[float, float],
) -> dict[str, float] | None:
    direction_h = np.array(
        [
            float(world_direction[0]),
            float(world_direction[1]),
            0.0,
        ],
        dtype=np.float64,
    )
    image_h = h @ direction_h

    if (
        not np.isfinite(image_h).all()
        or abs(float(image_h[2])) < 1e-9
    ):
        return None

    return {
        "x": float(image_h[0] / image_h[2]),
        "y": float(image_h[1] / image_h[2]),
    }


def _zoom_proxy(
    h: np.ndarray,
) -> float | None:
    """Local image scale near field centre, useful for temporal zoom tracking."""

    epsilon = 0.5
    sample = np.array(
        [
            [0.0, 0.0],
            [epsilon, 0.0],
            [0.0, epsilon],
        ],
        dtype=np.float64,
    )
    projected, valid = _project_points(sample, h)
    if not np.all(valid):
        return None

    sx = float(
        np.linalg.norm(projected[1] - projected[0])
        / epsilon
    )
    sy = float(
        np.linalg.norm(projected[2] - projected[0])
        / epsilon
    )

    if sx <= 0 or sy <= 0:
        return None

    return float(math.sqrt(sx * sy))


def _constraint_spread(
    points: list[dict[str, Any]],
    lines: list[dict[str, Any]],
) -> dict[str, float | None]:
    world: list[np.ndarray] = []
    image: list[np.ndarray] = []

    for item in points:
        world.append(item["world"])
        image.append(item["image"])

    for item in lines:
        world.extend(
            (item["world_start"], item["world_end"])
        )
        image.extend(
            (item["image_start"], item["image_end"])
        )

    if not world:
        return {
            "world_span_x_m": None,
            "world_span_y_m": None,
            "image_span_x_px": None,
            "image_span_y_px": None,
        }

    world_arr = np.asarray(world, dtype=np.float64)
    image_arr = np.asarray(image, dtype=np.float64)

    return {
        "world_span_x_m": float(
            np.ptp(world_arr[:, 0])
        ),
        "world_span_y_m": float(
            np.ptp(world_arr[:, 1])
        ),
        "image_span_x_px": float(
            np.ptp(image_arr[:, 0])
        ),
        "image_span_y_px": float(
            np.ptp(image_arr[:, 1])
        ),
    }


def _candidate_score(
    h: np.ndarray,
    points: list[dict[str, Any]],
    lines: list[dict[str, Any]],
    arcs: list[dict[str, Any]],
    config: CameraCalibrationConfig,
) -> float:
    evaluated = _residual_vector(
        h,
        points,
        lines,
        arcs,
        config,
    )
    if evaluated is None:
        return float("inf")

    residuals, kinds = evaluated
    return _weighted_cost(
        residuals,
        kinds,
        config,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def solve_field_camera_calibration(
    point_constraints: list[dict[str, Any]],
    line_constraints: list[dict[str, Any]],
    arc_constraints: list[dict[str, Any]],
    image_width: int | None,
    image_height: int | None,
    *,
    config: CameraCalibrationConfig | None = None,
) -> dict[str, Any]:
    """Solve one manual FIELD camera anchor.

    The function intentionally accepts the same three constraint collections
    already produced by the current Supervisor frontend.
    """

    cfg = config or CameraCalibrationConfig()

    points = _parse_points(point_constraints)
    lines = _parse_lines(line_constraints)
    arcs = _parse_arcs(arc_constraints)

    starts: list[tuple[str, np.ndarray]] = []

    mixed = _mixed_normalised_dlt(
        points,
        lines,
        cfg.max_condition,
    )
    if mixed is not None:
        starts.append(("mixed_normalised_dlt", mixed))

    point_only = _point_baseline(
        points,
        cfg.max_condition,
    )
    if point_only is not None:
        if not starts or np.linalg.norm(
            point_only - starts[0][1]
        ) > 1e-6:
            starts.append(("point_baseline", point_only))

    if not starts:
        return {
            "status": "not_enough_constraints",
            "detail": (
                "Las referencias actuales no determinan una "
                "calibración proyectiva estable."
            ),
            "diagnostics": {
                "points": len(points),
                "lines": len(lines),
                "arcs": len(arcs),
                "constraint_spread": _constraint_spread(
                    points,
                    lines,
                ),
            },
        }

    candidates: list[dict[str, Any]] = []

    for name, initial_h in starts:
        refined = _optimise_h(
            initial_h,
            points,
            lines,
            arcs,
            cfg,
        )

        for suffix, candidate in (
            ("initial", initial_h),
            ("refined", refined),
        ):
            if candidate is None:
                continue

            score = _candidate_score(
                candidate,
                points,
                lines,
                arcs,
                cfg,
            )
            if not np.isfinite(score):
                continue

            candidates.append(
                {
                    "name": f"{name}:{suffix}",
                    "homography": candidate,
                    "score": float(score),
                }
            )

    if not candidates:
        return {
            "status": "optimization_failed",
            "detail": (
                "No se pudo obtener una calibración geométrica "
                "estable con las referencias actuales."
            ),
        }

    chosen = min(
        candidates,
        key=lambda item: item["score"],
    )
    h = chosen["homography"]

    point_error, point_details = _point_diagnostics(
        h,
        points,
    )
    line_error, line_details = _line_diagnostics(
        h,
        lines,
    )

    try:
        condition = float(np.linalg.cond(h))
    except np.linalg.LinAlgError:
        condition = float("inf")

    vanishing_x = _vanishing_point(h, (1.0, 0.0))
    vanishing_y = _vanishing_point(h, (0.0, 1.0))

    if (
        vanishing_x is not None
        and vanishing_y is not None
    ):
        vx = np.array(
            [vanishing_x["x"], vanishing_x["y"], 1.0],
            dtype=np.float64,
        )
        vy = np.array(
            [vanishing_y["x"], vanishing_y["y"], 1.0],
            dtype=np.float64,
        )
        horizon = np.cross(vx, vy)
        norm = math.hypot(
            float(horizon[0]),
            float(horizon[1]),
        )
        if norm > 1e-9:
            horizon = horizon / norm
            horizon_json = {
                "a": float(horizon[0]),
                "b": float(horizon[1]),
                "c": float(horizon[2]),
            }
        else:
            horizon_json = None
    else:
        horizon_json = None

    bad_lines = [
        item
        for item in line_details
        if (
            item.get("error_px") is not None
            and item["error_px"]
            > cfg.warning_line_error_px
        )
    ]

    status = "solved"
    detail = None

    if bad_lines:
        status = "solved_with_warnings"
        names = [
            str(item.get("semantic") or "línea")
            for item in bad_lines
        ]
        detail = (
            "La calibración se resolvió, pero algunas líneas "
            "manuales siguen en conflicto: "
            + ", ".join(names)
            + "."
        )

    return {
        "status": status,
        "detail": detail,
        "homography": h.tolist(),
        "reprojection_error": point_error,
        "camera": {
            # These are projective/virtual-camera diagnostics, intentionally
            # not claimed to be physical Veo lens intrinsics.
            "vanishing_point_longitudinal": vanishing_x,
            "vanishing_point_transversal": vanishing_y,
            "horizon_line": horizon_json,
            "zoom_proxy_px_per_m": _zoom_proxy(h),
            "principal_view_projection": (
                _project_points(
                    np.array([[0.0, 0.0]], dtype=np.float64),
                    h,
                )[0][0].tolist()
            ),
        },
        "diagnostics": {
            "mode": "geometric_projective_camera_v1",
            "selected_candidate": chosen["name"],
            "score": chosen["score"],
            "homography_condition": condition,
            "points": len(points),
            "lines": len(lines),
            "arcs": len(arcs),
            "point_error_px": point_error,
            "line_error_px": line_error,
            "point_errors": point_details,
            "line_errors": line_details,
            "constraint_spread": _constraint_spread(
                points,
                lines,
            ),
            "config": asdict(cfg),
        },
    }


# Compatibility alias so the backend can switch to this module with the
# smallest possible integration change if desired.
def solve_field_calibration(
    point_constraints: list[dict[str, Any]],
    line_constraints: list[dict[str, Any]],
    arc_constraints: list[dict[str, Any]],
    image_width: int | None,
    image_height: int | None,
) -> dict[str, Any]:
    return solve_field_camera_calibration(
        point_constraints,
        line_constraints,
        arc_constraints,
        image_width,
        image_height,
    )