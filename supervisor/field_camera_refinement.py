"""Partial manual refinement of an existing FIELD camera homography.

This module is used when the Supervisor already has a reasonable homography for
the current frame (normally propagated/refined from a previous manual anchor)
and the user wants to correct it with NEW visible evidence.

Typical example
---------------
The propagated field is already close, but the halfway line becomes visible and
is slightly tilted. The user marks only:

* the halfway line;
* a few centre-circle / halfway landmarks;

Those observations may be collinear and therefore cannot determine a new
homography from scratch. They *can* refine an existing homography.

The algorithm therefore solves a different problem from
``field_camera_calibration.py``:

    minimise(manual point/line/arc error)
    + keep unobserved degrees of freedom near the incoming homography

The second term is an image-space prior evaluated on canonical pitch support
points that are visible (or nearly visible) under the incoming homography. It
prevents under-constrained corrections from inventing a new projective camera,
while still allowing manual semantic lines to pull the camera into alignment.

Semantic lines are true projective line constraints. The endpoints drawn by the
user define the observed infinite image line; they are NOT treated as endpoints
of the physical pitch marking.

No FastAPI or UI code lives here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import numpy as np


@dataclass(frozen=True)
class FieldCameraRefinementConfig:
    point_sigma_px: float = 3.0
    line_sigma_px: float = 1.25
    arc_sigma_px: float = 4.0

    # Soft prior around the incoming propagated/refined camera. This stabilises
    # degrees of freedom that the new partial manual evidence cannot observe.
    prior_sigma_px: float = 28.0
    prior_huber_delta_sigma: float = 2.5
    point_huber_delta_sigma: float = 3.0
    arc_huber_delta_sigma: float = 3.0

    max_iterations: int = 50
    initial_damping: float = 1e-3
    max_condition: float = 1e10

    # Only canonical support points whose incoming projection is within this
    # expanded viewport are used as the camera prior.
    prior_view_margin_ratio: float = 0.35
    min_prior_points: int = 4
    max_prior_points: int = 16

    # Safety diagnostics / acceptance.
    max_prior_median_displacement_px: float = 90.0
    max_prior_p90_displacement_px: float = 180.0
    warning_line_error_px: float = 12.0


# Spread across the 105x68 pitch. Only points relevant to the current view are
# retained after projection through the incoming homography.
_CANONICAL_SUPPORT_POINTS = np.asarray(
    [
        (-52.5, -34.0), (-36.0, -34.0), (0.0, -34.0), (36.0, -34.0), (52.5, -34.0),
        (-52.5, -20.16), (-36.0, -20.16), (0.0, -20.16), (36.0, -20.16), (52.5, -20.16),
        (-52.5, 0.0), (-36.0, 0.0), (0.0, 0.0), (36.0, 0.0), (52.5, 0.0),
        (-52.5, 20.16), (-36.0, 20.16), (0.0, 20.16), (36.0, 20.16), (52.5, 20.16),
        (-52.5, 34.0), (-36.0, 34.0), (0.0, 34.0), (36.0, 34.0), (52.5, 34.0),
        (-41.5, -9.16), (-41.5, 9.16), (41.5, -9.16), (41.5, 9.16),
        (-47.0, -9.16), (-47.0, 9.16), (47.0, -9.16), (47.0, 9.16),
    ],
    dtype=np.float64,
)


def _finite_h(matrix: Any, max_condition: float) -> np.ndarray | None:
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
        or not np.isfinite(condition)
        or abs(determinant) < 1e-12
        or condition > max_condition
    ):
        return None
    return h


def _project_points(
    world_xy: np.ndarray,
    h: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(world_xy, dtype=np.float64).reshape(-1, 2)
    homogeneous = np.column_stack((points, np.ones(len(points), dtype=np.float64)))
    projected_h = homogeneous @ h.T
    denominator = projected_h[:, 2]

    valid = (
        np.isfinite(projected_h).all(axis=1)
        & np.isfinite(denominator)
        & (np.abs(denominator) > 1e-9)
    )

    projected = np.full((len(points), 2), np.nan, dtype=np.float64)
    projected[valid] = projected_h[valid, :2] / denominator[valid, None]
    valid &= np.isfinite(projected).all(axis=1)
    return projected, valid


def _point_h(point: dict[str, Any]) -> np.ndarray | None:
    try:
        return np.asarray(
            [float(point["x"]), float(point["y"]), 1.0],
            dtype=np.float64,
        )
    except (KeyError, TypeError, ValueError):
        return None


def _line_from_endpoints(
    start: dict[str, Any],
    end: dict[str, Any],
) -> np.ndarray | None:
    first = _point_h(start)
    second = _point_h(end)
    if first is None or second is None:
        return None

    line = np.cross(first, second)
    norm = math.hypot(float(line[0]), float(line[1]))
    if norm < 1e-9 or not np.isfinite(line).all():
        return None
    return line / norm


def _transform_world_line_to_image(
    world_line: np.ndarray,
    h: np.ndarray,
) -> np.ndarray | None:
    try:
        predicted = np.linalg.inv(h).T @ world_line
    except np.linalg.LinAlgError:
        return None

    norm = math.hypot(float(predicted[0]), float(predicted[1]))
    if norm < 1e-9 or not np.isfinite(predicted).all():
        return None
    return predicted / norm


def _parse_points(
    constraints: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    parsed: list[dict[str, Any]] = []
    for item in constraints:
        try:
            world = np.asarray(
                [float(item["world"]["x"]), float(item["world"]["y"])],
                dtype=np.float64,
            )
            image = np.asarray(
                [float(item["image"]["x"]), float(item["image"]["y"])],
                dtype=np.float64,
            )
        except (KeyError, TypeError, ValueError):
            continue

        if np.isfinite(world).all() and np.isfinite(image).all():
            parsed.append(
                {
                    "semantic": item.get("semantic"),
                    "world": world,
                    "image": image,
                }
            )
    return parsed


def _parse_lines(
    constraints: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    parsed: list[dict[str, Any]] = []
    for item in constraints:
        world = item.get("world", {})
        image = item.get("image", {})
        world_line = _line_from_endpoints(
            world.get("start", {}),
            world.get("end", {}),
        )
        image_line = _line_from_endpoints(
            image.get("start", {}),
            image.get("end", {}),
        )
        if world_line is None or image_line is None:
            continue

        try:
            image_start = np.asarray(
                [float(image["start"]["x"]), float(image["start"]["y"])],
                dtype=np.float64,
            )
            image_end = np.asarray(
                [float(image["end"]["x"]), float(image["end"]["y"])],
                dtype=np.float64,
            )
        except (KeyError, TypeError, ValueError):
            continue

        if np.linalg.norm(image_end - image_start) < 8.0:
            continue

        parsed.append(
            {
                "semantic": item.get("semantic"),
                "world_line": world_line,
                "image_line": image_line,
                "image_start": image_start,
                "image_end": image_end,
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
        counterclockwise = bool(arc.get("counterclockwise", False))
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
    cosine = math.cos(rotation)
    sine = math.sin(rotation)
    x = rx * np.cos(angles)
    y = ry * np.sin(angles)

    return np.column_stack(
        (
            cx + cosine * x - sine * y,
            cy + sine * x + cosine * y,
        )
    )


def _parse_arcs(
    constraints: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    parsed: list[dict[str, Any]] = []
    for item in constraints:
        try:
            center = np.asarray(
                [
                    float(item["world"]["center"]["x"]),
                    float(item["world"]["center"]["y"]),
                ],
                dtype=np.float64,
            )
            radius = float(item["world"]["radius"])
        except (KeyError, TypeError, ValueError):
            continue

        if radius <= 0.0:
            continue

        image = item.get("image", {})
        samples = None
        raw_samples = image.get("samples") or item.get("fit", {}).get("samples")
        if raw_samples:
            try:
                samples = np.asarray(
                    [[float(p["x"]), float(p["y"])] for p in raw_samples],
                    dtype=np.float64,
                )
            except (KeyError, TypeError, ValueError):
                samples = None

        if samples is None or len(samples) < 4:
            fitted = image.get("fitted") or image.get("geometry") or image
            if isinstance(fitted, dict):
                samples = _sample_fitted_ellipse(fitted)

        if samples is None or len(samples) < 4 or not np.isfinite(samples).all():
            continue

        if len(samples) > 28:
            indices = np.linspace(0, len(samples) - 1, 28).round().astype(int)
            samples = samples[indices]

        parsed.append(
            {
                "semantic": item.get("semantic"),
                "world_center": center,
                "world_radius": radius,
                "image_samples": samples,
            }
        )
    return parsed


def _select_prior_support(
    initial_h: np.ndarray,
    image_width: int,
    image_height: int,
    config: FieldCameraRefinementConfig,
) -> tuple[np.ndarray, np.ndarray]:
    projected, valid = _project_points(_CANONICAL_SUPPORT_POINTS, initial_h)

    margin_x = image_width * config.prior_view_margin_ratio
    margin_y = image_height * config.prior_view_margin_ratio
    nearby = (
        valid
        & (projected[:, 0] >= -margin_x)
        & (projected[:, 0] <= image_width + margin_x)
        & (projected[:, 1] >= -margin_y)
        & (projected[:, 1] <= image_height + margin_y)
    )

    world = _CANONICAL_SUPPORT_POINTS[nearby]
    image = projected[nearby]

    if len(world) > config.max_prior_points:
        # Deterministic spread through the already spatially distributed list.
        indices = np.linspace(
            0,
            len(world) - 1,
            config.max_prior_points,
        ).round().astype(int)
        world = world[indices]
        image = image[indices]

    return world, image


def _local_pixel_scale(
    world_center: np.ndarray,
    world_radius: float,
    h: np.ndarray,
) -> float | None:
    samples = np.asarray(
        [
            world_center,
            world_center + np.asarray([world_radius, 0.0], dtype=np.float64),
        ],
        dtype=np.float64,
    )
    projected, valid = _project_points(samples, h)
    if not np.all(valid):
        return None

    value = float(np.linalg.norm(projected[1] - projected[0]) / world_radius)
    return value if np.isfinite(value) and value > 0.0 else None


def _h_to_params(h: np.ndarray) -> np.ndarray:
    h = h / h[2, 2]
    return np.asarray(
        [
            h[0, 0], h[0, 1], h[0, 2],
            h[1, 0], h[1, 1], h[1, 2],
            h[2, 0], h[2, 1],
        ],
        dtype=np.float64,
    )


def _params_to_h(params: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            [params[0], params[1], params[2]],
            [params[3], params[4], params[5]],
            [params[6], params[7], 1.0],
        ],
        dtype=np.float64,
    )


def _residual_vector(
    h: np.ndarray,
    points: list[dict[str, Any]],
    lines: list[dict[str, Any]],
    arcs: list[dict[str, Any]],
    prior_world: np.ndarray,
    prior_image: np.ndarray,
    config: FieldCameraRefinementConfig,
) -> tuple[np.ndarray, list[str]] | None:
    residuals: list[float] = []
    kinds: list[str] = []

    if points:
        world = np.asarray([item["world"] for item in points], dtype=np.float64)
        observed = np.asarray([item["image"] for item in points], dtype=np.float64)
        projected, valid = _project_points(world, h)
        if not np.all(valid):
            return None

        values = (projected - observed) / config.point_sigma_px
        residuals.extend(values.reshape(-1).tolist())
        kinds.extend(["point"] * values.size)

    for item in lines:
        predicted = _transform_world_line_to_image(item["world_line"], h)
        if predicted is None:
            return None

        # Distance does not depend on homogeneous line sign.
        for endpoint in (item["image_start"], item["image_end"]):
            distance_px = float(
                predicted[0] * endpoint[0]
                + predicted[1] * endpoint[1]
                + predicted[2]
            )
            residuals.append(distance_px / config.line_sigma_px)
            kinds.append("line")

    if arcs:
        try:
            inverse_h = np.linalg.inv(h)
        except np.linalg.LinAlgError:
            return None

        for item in arcs:
            world_back, valid = _project_points(item["image_samples"], inverse_h)
            if int(np.count_nonzero(valid)) < 4:
                continue

            world_back = world_back[valid]
            radial_error_m = (
                np.linalg.norm(
                    world_back - item["world_center"][None, :],
                    axis=1,
                )
                - item["world_radius"]
            )
            px_per_m = _local_pixel_scale(
                item["world_center"],
                item["world_radius"],
                h,
            )
            if px_per_m is None:
                continue

            values = radial_error_m * px_per_m / config.arc_sigma_px
            residuals.extend(values.tolist())
            kinds.extend(["arc"] * len(values))

    if len(prior_world):
        projected, valid = _project_points(prior_world, h)
        if not np.all(valid):
            return None

        values = (projected - prior_image) / config.prior_sigma_px
        residuals.extend(values.reshape(-1).tolist())
        kinds.extend(["prior"] * values.size)

    if not residuals:
        return None

    vector = np.asarray(residuals, dtype=np.float64)
    if not np.isfinite(vector).all():
        return None
    return vector, kinds


def _weights(
    residuals: np.ndarray,
    kinds: list[str],
    config: FieldCameraRefinementConfig,
) -> np.ndarray:
    weights = np.ones(len(residuals), dtype=np.float64)

    for index, kind in enumerate(kinds):
        value = abs(float(residuals[index]))

        # A human semantic line should not be silently rejected as an outlier.
        if kind == "line":
            continue

        if kind == "point":
            delta = config.point_huber_delta_sigma
        elif kind == "arc":
            delta = config.arc_huber_delta_sigma
        else:
            delta = config.prior_huber_delta_sigma

        if value > delta:
            weights[index] = delta / max(value, 1e-12)

    return weights


def _cost(
    residuals: np.ndarray,
    kinds: list[str],
    config: FieldCameraRefinementConfig,
) -> float:
    weights = _weights(residuals, kinds, config)
    return float(0.5 * np.sum(weights * residuals * residuals))


def _optimise(
    initial_h: np.ndarray,
    points: list[dict[str, Any]],
    lines: list[dict[str, Any]],
    arcs: list[dict[str, Any]],
    prior_world: np.ndarray,
    prior_image: np.ndarray,
    config: FieldCameraRefinementConfig,
) -> tuple[np.ndarray, float] | None:
    params = _h_to_params(initial_h)
    damping = config.initial_damping

    evaluated = _residual_vector(
        initial_h,
        points,
        lines,
        arcs,
        prior_world,
        prior_image,
        config,
    )
    if evaluated is None:
        return None

    residuals, kinds = evaluated
    best_cost = _cost(residuals, kinds, config)
    best_params = params.copy()

    for _ in range(config.max_iterations):
        current_h = _finite_h(_params_to_h(params), config.max_condition)
        if current_h is None:
            return None

        evaluated = _residual_vector(
            current_h,
            points,
            lines,
            arcs,
            prior_world,
            prior_image,
            config,
        )
        if evaluated is None:
            return None

        residuals, kinds = evaluated
        weights = _weights(residuals, kinds, config)

        jacobian = np.zeros((len(residuals), 8), dtype=np.float64)
        for column in range(8):
            step = 1e-6 * max(1.0, abs(float(params[column])))
            candidate_params = params.copy()
            candidate_params[column] += step
            candidate_h = _finite_h(
                _params_to_h(candidate_params),
                config.max_condition,
            )
            if candidate_h is None:
                continue

            candidate_eval = _residual_vector(
                candidate_h,
                points,
                lines,
                arcs,
                prior_world,
                prior_image,
                config,
            )
            if candidate_eval is None:
                continue

            candidate_residuals, _ = candidate_eval
            if len(candidate_residuals) != len(residuals):
                return None

            jacobian[:, column] = (
                candidate_residuals - residuals
            ) / step

        sqrt_weights = np.sqrt(weights)
        weighted_j = jacobian * sqrt_weights[:, None]
        weighted_r = residuals * sqrt_weights

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
            prior_world,
            prior_image,
            config,
        )
        if candidate_eval is None:
            damping *= 5.0
            continue

        candidate_residuals, candidate_kinds = candidate_eval
        candidate_cost = _cost(
            candidate_residuals,
            candidate_kinds,
            config,
        )

        if candidate_cost < best_cost:
            params = candidate_params
            best_params = candidate_params.copy()
            best_cost = candidate_cost
            damping = max(1e-8, damping * 0.35)
            if float(np.linalg.norm(delta)) < 1e-7:
                break
        else:
            damping = min(1e8, damping * 4.0)

    final_h = _finite_h(
        _params_to_h(best_params),
        config.max_condition,
    )
    if final_h is None:
        return None
    return final_h, best_cost


def _point_diagnostics(
    h: np.ndarray,
    points: list[dict[str, Any]],
) -> tuple[float | None, list[dict[str, Any]]]:
    if not points:
        return None, []

    world = np.asarray([item["world"] for item in points], dtype=np.float64)
    observed = np.asarray([item["image"] for item in points], dtype=np.float64)
    projected, valid = _project_points(world, h)

    details = []
    values = []
    for index, item in enumerate(points):
        error = None
        if valid[index]:
            error = float(np.linalg.norm(projected[index] - observed[index]))
            values.append(error)
        details.append(
            {
                "semantic": item.get("semantic"),
                "error_px": error,
            }
        )

    return (float(np.mean(values)) if values else None), details


def _line_diagnostics(
    h: np.ndarray,
    lines: list[dict[str, Any]],
) -> tuple[float | None, list[dict[str, Any]]]:
    values = []
    details = []

    for item in lines:
        predicted = _transform_world_line_to_image(item["world_line"], h)
        if predicted is None:
            details.append(
                {
                    "semantic": item.get("semantic"),
                    "error_px": None,
                    "quality": "bad",
                }
            )
            continue

        endpoint_errors = [
            abs(
                float(
                    predicted[0] * endpoint[0]
                    + predicted[1] * endpoint[1]
                    + predicted[2]
                )
            )
            for endpoint in (item["image_start"], item["image_end"])
        ]
        error = float(np.mean(endpoint_errors))
        values.append(error)

        quality = (
            "good"
            if error <= 5.0
            else ("warning" if error <= 12.0 else "bad")
        )
        details.append(
            {
                "semantic": item.get("semantic"),
                "error_px": error,
                "endpoint_errors_px": endpoint_errors,
                "quality": quality,
            }
        )

    return (float(np.mean(values)) if values else None), details


def _camera_diagnostics(h: np.ndarray) -> dict[str, Any]:
    def vanishing(direction: tuple[float, float]) -> dict[str, float] | None:
        value = h @ np.asarray(
            [direction[0], direction[1], 0.0],
            dtype=np.float64,
        )
        if not np.isfinite(value).all() or abs(float(value[2])) < 1e-9:
            return None
        return {
            "x": float(value[0] / value[2]),
            "y": float(value[1] / value[2]),
        }

    longitudinal = vanishing((1.0, 0.0))
    transversal = vanishing((0.0, 1.0))

    horizon = None
    if longitudinal is not None and transversal is not None:
        first = np.asarray(
            [longitudinal["x"], longitudinal["y"], 1.0],
            dtype=np.float64,
        )
        second = np.asarray(
            [transversal["x"], transversal["y"], 1.0],
            dtype=np.float64,
        )
        line = np.cross(first, second)
        norm = math.hypot(float(line[0]), float(line[1]))
        if norm > 1e-9:
            line /= norm
            horizon = {
                "a": float(line[0]),
                "b": float(line[1]),
                "c": float(line[2]),
            }

    epsilon = 0.5
    samples = np.asarray(
        [[0.0, 0.0], [epsilon, 0.0], [0.0, epsilon]],
        dtype=np.float64,
    )
    projected, valid = _project_points(samples, h)
    zoom_proxy = None
    principal = None
    if valid[0]:
        principal = projected[0].tolist()
    if np.all(valid):
        sx = float(np.linalg.norm(projected[1] - projected[0]) / epsilon)
        sy = float(np.linalg.norm(projected[2] - projected[0]) / epsilon)
        if sx > 0.0 and sy > 0.0:
            zoom_proxy = float(math.sqrt(sx * sy))

    return {
        "vanishing_point_longitudinal": longitudinal,
        "vanishing_point_transversal": transversal,
        "horizon_line": horizon,
        "zoom_proxy_px_per_m": zoom_proxy,
        "principal_view_projection": principal,
    }


def refine_field_camera_calibration(
    initial_homography: list[list[float]] | np.ndarray,
    point_constraints: list[dict[str, Any]],
    line_constraints: list[dict[str, Any]],
    arc_constraints: list[dict[str, Any]],
    image_width: int,
    image_height: int,
    *,
    config: FieldCameraRefinementConfig | None = None,
) -> dict[str, Any]:
    """Refine an existing camera using partial manual semantic evidence.

    Unlike solving from scratch, collinear points or a single semantic line are
    allowed because the incoming homography supplies the missing degrees of
    freedom through a soft image-space prior.
    """

    cfg = config or FieldCameraRefinementConfig()
    initial_h = _finite_h(initial_homography, cfg.max_condition)

    if initial_h is None:
        return {
            "status": "invalid_initial_homography",
            "detail": "La proyección inicial del campo no es válida.",
        }

    try:
        width = int(image_width)
        height = int(image_height)
    except (TypeError, ValueError):
        width = 0
        height = 0

    if width <= 0 or height <= 0:
        return {
            "status": "invalid_image_size",
            "detail": "La resolución del vídeo no es válida.",
        }

    points = _parse_points(point_constraints)
    lines = _parse_lines(line_constraints)
    arcs = _parse_arcs(arc_constraints)

    if not points and not lines and not arcs:
        return {
            "status": "not_enough_constraints",
            "detail": "Añade al menos una referencia manual para corregir esta proyección.",
        }

    prior_world, prior_image = _select_prior_support(
        initial_h,
        width,
        height,
        cfg,
    )

    if len(prior_world) < cfg.min_prior_points:
        return {
            "status": "invalid_initial_homography",
            "detail": (
                "La proyección existente no deja suficientes referencias "
                "del campo cerca de la vista actual para refinarla con seguridad."
            ),
        }

    initial_eval = _residual_vector(
        initial_h,
        points,
        lines,
        arcs,
        prior_world,
        prior_image,
        cfg,
    )
    if initial_eval is None:
        return {
            "status": "optimization_failed",
            "detail": "No se pudieron evaluar las referencias sobre la proyección inicial.",
        }

    initial_residuals, initial_kinds = initial_eval
    initial_cost = _cost(initial_residuals, initial_kinds, cfg)

    optimised = _optimise(
        initial_h,
        points,
        lines,
        arcs,
        prior_world,
        prior_image,
        cfg,
    )
    if optimised is None:
        return {
            "status": "optimization_failed",
            "detail": "No se pudo refinar la cámara a partir de la proyección existente.",
        }

    refined_h, final_cost = optimised

    refined_prior, valid = _project_points(prior_world, refined_h)
    if not np.all(valid):
        return {
            "status": "optimization_failed",
            "detail": "La corrección volvió inestable la zona visible del campo.",
        }

    prior_displacements = np.linalg.norm(refined_prior - prior_image, axis=1)
    prior_median = float(np.median(prior_displacements))
    prior_p90 = float(np.percentile(prior_displacements, 90))

    if (
        prior_median > cfg.max_prior_median_displacement_px
        or prior_p90 > cfg.max_prior_p90_displacement_px
    ):
        return {
            "status": "correction_too_large",
            "detail": (
                "La corrección manual exige alejar demasiado la cámara de la "
                "proyección existente. Añade otra línea/punto visible o crea "
                "una calibración completa en este fotograma."
            ),
            "diagnostics": {
                "prior_median_displacement_px": prior_median,
                "prior_p90_displacement_px": prior_p90,
            },
        }

    point_error, point_details = _point_diagnostics(refined_h, points)
    line_error, line_details = _line_diagnostics(refined_h, lines)

    bad_lines = [
        item
        for item in line_details
        if item.get("error_px") is not None
        and item["error_px"] > cfg.warning_line_error_px
    ]

    status = "solved_with_warnings" if bad_lines else "solved"
    detail = None
    if bad_lines:
        detail = (
            "La corrección se resolvió, pero algunas líneas siguen en conflicto: "
            + ", ".join(str(item.get("semantic") or "línea") for item in bad_lines)
            + "."
        )

    try:
        condition = float(np.linalg.cond(refined_h))
    except np.linalg.LinAlgError:
        condition = float("inf")

    return {
        "status": status,
        "detail": detail,
        "homography": refined_h.tolist(),
        "reprojection_error": point_error,
        "camera": _camera_diagnostics(refined_h),
        "diagnostics": {
            "mode": "geometric_projective_camera_refinement_v1",
            "used_initial_homography": True,
            "points": len(points),
            "lines": len(lines),
            "arcs": len(arcs),
            "prior_support_points": int(len(prior_world)),
            "initial_cost": float(initial_cost),
            "final_cost": float(final_cost),
            "cost_improvement": float(initial_cost - final_cost),
            "point_error_px": point_error,
            "line_error_px": line_error,
            "point_errors": point_details,
            "line_errors": line_details,
            "prior_median_displacement_px": prior_median,
            "prior_p90_displacement_px": prior_p90,
            "homography_condition": condition,
            "config": asdict(cfg),
        },
    }