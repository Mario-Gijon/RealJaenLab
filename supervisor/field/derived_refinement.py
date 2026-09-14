"""Prior-guided refinement for propagated RealJaenLab FIELD geometry.

This module refines an already propagated DERIVED pitch projection using sparse
semantic evidence in the target frame.

It is deliberately NOT a homography-from-scratch solver.  The transported
projection is treated as a prior and only a small image-space correction is
allowed.  This lets a target frame with too little geometry for independent
calibration still improve a good HUMAN -> DERIVED prediction.

Supported correction families, in increasing complexity:

- translation
- similarity
- affine
- homography

All corrections are parameterised around identity in normalized image
coordinates and regularized toward identity.  Point annotations constrain
semantic projected points directly.  Line annotations keep the V3 meaning:
their two clicks are arbitrary visible points on the semantic painted line,
not canonical line endpoints.

No SciPy dependency is required.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Sequence

import numpy as np


EPS = 1e-10
MODEL_ORDER = ("translation", "similarity", "affine", "homography")

DEFAULT_HUBER_DELTA_PX = 5.0
DEFAULT_REGULARIZATION_WEIGHT = 2.0
DEFAULT_ADEQUACY_P95_DIAGONAL_FRACTION = 0.0030
DEFAULT_NEAR_BEST_DIAGONAL_FRACTION = 0.0008
DEFAULT_MAX_OPTIMIZATION_ROUNDS = 120
DEFAULT_MIN_STEP = 2.5e-6


class DerivedRefinementError(ValueError):
    """Invalid or insufficient input for DERIVED refinement."""


def _finite(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise DerivedRefinementError(f"{label} must be numeric.") from exc
    if not math.isfinite(result):
        raise DerivedRefinementError(f"{label} must be finite.")
    return result


def _point(value: Sequence[float], label: str) -> tuple[float, float]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 2
    ):
        raise DerivedRefinementError(f"{label} must contain two coordinates.")
    return _finite(value[0], f"{label}[0]"), _finite(value[1], f"{label}[1]")


def _validate_dimensions(width: Any, height: Any) -> tuple[float, float, float]:
    w = _finite(width, "video_width")
    h = _finite(height, "video_height")
    if w <= 0 or h <= 0:
        raise DerivedRefinementError("video_width and video_height must be positive.")
    return w, h, math.hypot(w, h)


def _normalization(width: float, height: float) -> tuple[np.ndarray, np.ndarray]:
    diagonal = math.hypot(width, height)
    cx, cy = width / 2.0, height / 2.0
    n = np.array(
        [
            [1.0 / diagonal, 0.0, -cx / diagonal],
            [0.0, 1.0 / diagonal, -cy / diagonal],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    inv = np.array(
        [
            [diagonal, 0.0, cx],
            [0.0, diagonal, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    return n, inv


def _matrix_from_params(
    model: str,
    params: np.ndarray,
    width: float,
    height: float,
) -> np.ndarray:
    if model == "translation":
        tx, ty = params
        q = np.array([[1.0, 0.0, tx], [0.0, 1.0, ty], [0.0, 0.0, 1.0]])
    elif model == "similarity":
        tx, ty, log_scale, theta = params
        scale = math.exp(float(log_scale))
        c, s = math.cos(float(theta)), math.sin(float(theta))
        q = np.array(
            [
                [scale * c, -scale * s, tx],
                [scale * s, scale * c, ty],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
    elif model == "affine":
        tx, ty, a11, a12, a21, a22 = params
        q = np.array(
            [
                [1.0 + a11, a12, tx],
                [a21, 1.0 + a22, ty],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
    elif model == "homography":
        tx, ty, a11, a12, a21, a22, p1, p2 = params
        q = np.array(
            [
                [1.0 + a11, a12, tx],
                [a21, 1.0 + a22, ty],
                [p1, p2, 1.0],
            ],
            dtype=float,
        )
    else:
        raise DerivedRefinementError(f"Unknown refinement model {model!r}.")

    n, inv = _normalization(width, height)
    matrix = inv @ q @ n
    if not np.all(np.isfinite(matrix)) or abs(float(np.linalg.det(matrix))) < EPS:
        raise DerivedRefinementError(f"{model} correction became degenerate.")
    if abs(float(matrix[2, 2])) > EPS:
        matrix = matrix / float(matrix[2, 2])
    return matrix


def _model_spec(model: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return bounds, coordinate-descent steps, and prior scales."""
    if model == "translation":
        bounds = np.array([0.15, 0.15], dtype=float)
        steps = np.array([0.012, 0.012], dtype=float)
        prior = np.array([0.035, 0.035], dtype=float)
    elif model == "similarity":
        bounds = np.array([0.15, 0.15, 0.12, 0.10], dtype=float)
        steps = np.array([0.012, 0.012, 0.010, 0.008], dtype=float)
        prior = np.array([0.035, 0.035, 0.040, 0.030], dtype=float)
    elif model == "affine":
        bounds = np.array([0.15, 0.15, 0.15, 0.15, 0.15, 0.15], dtype=float)
        steps = np.array([0.012, 0.012, 0.010, 0.010, 0.010, 0.010], dtype=float)
        prior = np.array([0.035, 0.035, 0.050, 0.050, 0.050, 0.050], dtype=float)
    elif model == "homography":
        bounds = np.array([0.15] * 8, dtype=float)
        steps = np.array([0.012, 0.012, 0.010, 0.010, 0.010, 0.010, 0.010, 0.010], dtype=float)
        prior = np.array([0.035, 0.035, 0.050, 0.050, 0.050, 0.050, 0.060, 0.060], dtype=float)
    else:
        raise DerivedRefinementError(f"Unknown refinement model {model!r}.")
    return bounds, steps, prior


def _minimum_constraint_count(model: str) -> int:
    return {
        "translation": 2,
        "similarity": 4,
        "affine": 6,
        "homography": 8,
    }[model]


def _constraint_count(
    point_annotations: Sequence[dict[str, Any]],
    line_annotations: Sequence[dict[str, Any]],
) -> int:
    # A semantic point carries x/y evidence. Each clicked line point contributes
    # one normal-distance constraint to its semantic projected line.
    return 2 * len(point_annotations) + 2 * len(line_annotations)


def _apply_matrix_point(matrix: np.ndarray, point: tuple[float, float]) -> tuple[float, float]:
    q = matrix @ np.array([point[0], point[1], 1.0], dtype=float)
    if abs(float(q[2])) < EPS:
        raise DerivedRefinementError("Correction maps projected geometry to infinity.")
    return float(q[0] / q[2]), float(q[1] / q[2])


def _apply_matrix_points(
    matrix: np.ndarray,
    points: Sequence[Sequence[float]],
) -> list[list[float]]:
    result: list[list[float]] = []
    for index, raw in enumerate(points):
        transformed = _apply_matrix_point(matrix, _point(raw, f"geometry point {index}"))
        result.append([transformed[0], transformed[1]])
    return result


def transform_projected_geometry(
    projected_geometry: dict[str, Any],
    matrix: Sequence[Sequence[float]] | np.ndarray,
) -> dict[str, Any]:
    """Apply one image-space correction to complete projected pitch geometry."""
    if not isinstance(projected_geometry, dict):
        raise DerivedRefinementError("transported_geometry must be an object.")
    m = np.asarray(matrix, dtype=float)
    if m.shape != (3, 3) or not np.all(np.isfinite(m)):
        raise DerivedRefinementError("Correction matrix must be a finite 3x3 matrix.")

    lines = []
    for item in projected_geometry.get("lines", []):
        if not isinstance(item, dict):
            continue
        lines.append(
            {
                **item,
                "points": _apply_matrix_points(m, item.get("points", [])),
            }
        )

    curves = []
    for item in projected_geometry.get("curves", []):
        if not isinstance(item, dict):
            continue
        curves.append(
            {
                **item,
                "points": _apply_matrix_points(m, item.get("points", [])),
            }
        )

    points = []
    for item in projected_geometry.get("points", []):
        if not isinstance(item, dict):
            continue
        raw = item.get("point")
        transformed = _apply_matrix_point(m, _point(raw, f"projected point {item.get('name', '?')}"))
        points.append({**item, "point": [transformed[0], transformed[1]]})

    return {**projected_geometry, "lines": lines, "curves": curves, "points": points}


def _geometry_maps(projected_geometry: dict[str, Any]) -> tuple[dict[str, tuple[float, float]], dict[str, list[tuple[float, float]]]]:
    point_map: dict[str, tuple[float, float]] = {}
    line_map: dict[str, list[tuple[float, float]]] = {}

    for item in projected_geometry.get("points", []):
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        if isinstance(name, str) and name:
            point_map[name] = _point(item.get("point"), f"projected point {name}")

    for item in projected_geometry.get("lines", []):
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        raw_points = item.get("points")
        if not isinstance(name, str) or not name:
            continue
        if not isinstance(raw_points, Sequence) or isinstance(raw_points, (str, bytes)) or len(raw_points) < 2:
            continue
        line_map[name] = [_point(p, f"projected line {name}") for p in raw_points]

    if not point_map and not line_map:
        raise DerivedRefinementError("transported_geometry contains no usable semantic geometry.")
    return point_map, line_map


def _parse_annotations(
    *,
    point_annotations: Any,
    line_annotations: Any,
    point_map: dict[str, tuple[float, float]],
    line_map: dict[str, list[tuple[float, float]]],
    width: float,
    height: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if point_annotations is None:
        point_annotations = []
    if line_annotations is None:
        line_annotations = []
    if (
        not isinstance(point_annotations, Sequence)
        or isinstance(point_annotations, (str, bytes))
        or not isinstance(line_annotations, Sequence)
        or isinstance(line_annotations, (str, bytes))
    ):
        raise DerivedRefinementError("Annotations must be sequences.")

    parsed_points: list[dict[str, Any]] = []
    seen_points: set[str] = set()
    for raw in point_annotations:
        if not isinstance(raw, dict):
            raise DerivedRefinementError("Each point annotation must be an object.")
        name = raw.get("point_name")
        if not isinstance(name, str) or not name:
            raise DerivedRefinementError("Each point annotation requires point_name.")
        if name not in point_map:
            raise DerivedRefinementError(f"Projected geometry has no semantic point {name!r}.")
        if name in seen_points:
            raise DerivedRefinementError(f"Duplicate point annotation {name!r}.")
        observed = _point(raw.get("point"), f"point annotation {name}")
        if not (0.0 <= observed[0] <= width and 0.0 <= observed[1] <= height):
            raise DerivedRefinementError(f"Point annotation {name!r} is outside the image.")
        parsed_points.append({"point_name": name, "point": observed})
        seen_points.add(name)

    parsed_lines: list[dict[str, Any]] = []
    seen_lines: set[str] = set()
    for raw in line_annotations:
        if not isinstance(raw, dict):
            raise DerivedRefinementError("Each line annotation must be an object.")
        name = raw.get("line_name")
        clicks = raw.get("points")
        if not isinstance(name, str) or not name:
            raise DerivedRefinementError("Each line annotation requires line_name.")
        if name not in line_map:
            raise DerivedRefinementError(f"Projected geometry has no semantic line {name!r}.")
        if name in seen_lines:
            raise DerivedRefinementError(f"Duplicate line annotation {name!r}.")
        if (
            not isinstance(clicks, Sequence)
            or isinstance(clicks, (str, bytes))
            or len(clicks) != 2
        ):
            raise DerivedRefinementError(f"Line annotation {name!r} requires two clicks.")
        parsed_clicks = [_point(clicks[0], f"{name} click 1"), _point(clicks[1], f"{name} click 2")]
        for click in parsed_clicks:
            if not (0.0 <= click[0] <= width and 0.0 <= click[1] <= height):
                raise DerivedRefinementError(f"Line annotation {name!r} is outside the image.")
        if math.hypot(parsed_clicks[1][0] - parsed_clicks[0][0], parsed_clicks[1][1] - parsed_clicks[0][1]) < 2.0:
            raise DerivedRefinementError(f"Line annotation {name!r} clicks are too close.")
        parsed_lines.append({"line_name": name, "points": parsed_clicks})
        seen_lines.add(name)

    if not parsed_points and not parsed_lines:
        raise DerivedRefinementError("At least one sparse target annotation is required.")

    return parsed_points, parsed_lines


def _point_segment_distance(
    p: tuple[float, float],
    a: tuple[float, float],
    b: tuple[float, float],
) -> float:
    px, py = p
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    denom = dx * dx + dy * dy
    if denom < EPS:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * dx + (py - ay) * dy) / denom
    t = min(1.0, max(0.0, t))
    qx, qy = ax + t * dx, ay + t * dy
    return math.hypot(px - qx, py - qy)


def _point_polyline_distance(
    point: tuple[float, float],
    polyline: Sequence[tuple[float, float]],
) -> float:
    if len(polyline) < 2:
        raise DerivedRefinementError("Projected semantic line has too few samples.")
    return min(
        _point_segment_distance(point, polyline[index], polyline[index + 1])
        for index in range(len(polyline) - 1)
    )


def _huber(value: float, delta: float) -> float:
    a = abs(float(value))
    if a <= delta:
        return 0.5 * a * a
    return delta * (a - 0.5 * delta)


def _summary(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {
            "median_px": float("nan"),
            "p95_px": float("nan"),
            "max_px": float("nan"),
            "rmse_px": float("nan"),
        }
    arr = np.asarray(values, dtype=float)
    return {
        "median_px": float(np.median(arr)),
        "p95_px": float(np.percentile(arr, 95)),
        "max_px": float(np.max(arr)),
        "rmse_px": float(np.sqrt(np.mean(arr * arr))),
    }


def _evidence_errors(
    *,
    matrix: np.ndarray,
    point_map: dict[str, tuple[float, float]],
    line_map: dict[str, list[tuple[float, float]]],
    point_annotations: Sequence[dict[str, Any]],
    line_annotations: Sequence[dict[str, Any]],
) -> tuple[list[float], list[float], list[dict[str, Any]]]:
    point_errors: list[float] = []
    line_errors: list[float] = []
    details: list[dict[str, Any]] = []

    for annotation in point_annotations:
        name = annotation["point_name"]
        observed = annotation["point"]
        predicted = _apply_matrix_point(matrix, point_map[name])
        error = math.hypot(predicted[0] - observed[0], predicted[1] - observed[1])
        point_errors.append(error)
        details.append(
            {
                "kind": "point",
                "name": name,
                "error_px": float(error),
                "observed": [float(observed[0]), float(observed[1])],
                "predicted": [float(predicted[0]), float(predicted[1])],
            }
        )

    for annotation in line_annotations:
        name = annotation["line_name"]
        corrected_polyline = [_apply_matrix_point(matrix, p) for p in line_map[name]]
        click_errors = [
            _point_polyline_distance(click, corrected_polyline)
            for click in annotation["points"]
        ]
        line_errors.extend(click_errors)
        details.append(
            {
                "kind": "line",
                "name": name,
                "click_errors_px": [float(value) for value in click_errors],
                "median_px": float(np.median(np.asarray(click_errors, dtype=float))),
            }
        )

    return point_errors, line_errors, details


def _objective(
    *,
    model: str,
    params: np.ndarray,
    width: float,
    height: float,
    point_map: dict[str, tuple[float, float]],
    line_map: dict[str, list[tuple[float, float]]],
    point_annotations: Sequence[dict[str, Any]],
    line_annotations: Sequence[dict[str, Any]],
    huber_delta_px: float,
    regularization_weight: float,
    prior_scales: np.ndarray,
) -> tuple[float, float, float, list[float], list[float]]:
    try:
        matrix = _matrix_from_params(model, params, width, height)
        point_errors, line_errors, _ = _evidence_errors(
            matrix=matrix,
            point_map=point_map,
            line_map=line_map,
            point_annotations=point_annotations,
            line_annotations=line_annotations,
        )
    except DerivedRefinementError:
        return float("inf"), float("inf"), float("inf"), [], []

    all_errors = point_errors + line_errors
    if not all_errors:
        return float("inf"), float("inf"), float("inf"), [], []

    data_loss = float(np.mean([_huber(value, huber_delta_px) for value in all_errors]))
    normalized = params / np.maximum(prior_scales, EPS)
    regularization = float(regularization_weight * np.mean(normalized * normalized))
    return data_loss + regularization, data_loss, regularization, point_errors, line_errors


def _initial_params(
    model: str,
    *,
    diagonal: float,
    point_map: dict[str, tuple[float, float]],
    point_annotations: Sequence[dict[str, Any]],
) -> np.ndarray:
    size = {"translation": 2, "similarity": 4, "affine": 6, "homography": 8}[model]
    params = np.zeros(size, dtype=float)
    if point_annotations:
        deltas = []
        for annotation in point_annotations:
            predicted = point_map[annotation["point_name"]]
            observed = annotation["point"]
            deltas.append(
                [
                    (observed[0] - predicted[0]) / diagonal,
                    (observed[1] - predicted[1]) / diagonal,
                ]
            )
        median = np.median(np.asarray(deltas, dtype=float), axis=0)
        params[0] = float(median[0])
        params[1] = float(median[1])
    return params


def _optimize_model(
    model: str,
    *,
    width: float,
    height: float,
    diagonal: float,
    point_map: dict[str, tuple[float, float]],
    line_map: dict[str, list[tuple[float, float]]],
    point_annotations: Sequence[dict[str, Any]],
    line_annotations: Sequence[dict[str, Any]],
    huber_delta_px: float,
    regularization_weight: float,
    max_rounds: int,
) -> dict[str, Any]:
    bounds, steps, prior_scales = _model_spec(model)
    params = _initial_params(
        model,
        diagonal=diagonal,
        point_map=point_map,
        point_annotations=point_annotations,
    )
    params = np.clip(params, -bounds, bounds)

    def evaluate(candidate: np.ndarray):
        return _objective(
            model=model,
            params=candidate,
            width=width,
            height=height,
            point_map=point_map,
            line_map=line_map,
            point_annotations=point_annotations,
            line_annotations=line_annotations,
            huber_delta_px=huber_delta_px,
            regularization_weight=regularization_weight,
            prior_scales=prior_scales,
        )

    best = evaluate(params)
    rounds = 0

    while rounds < max_rounds and float(np.max(steps)) >= DEFAULT_MIN_STEP:
        rounds += 1
        improved = False
        for index in range(len(params)):
            local_params = params
            local_best = best
            for direction in (-1.0, 1.0):
                candidate = params.copy()
                candidate[index] = float(
                    np.clip(candidate[index] + direction * steps[index], -bounds[index], bounds[index])
                )
                score = evaluate(candidate)
                if score[0] + 1e-12 < local_best[0]:
                    local_params = candidate
                    local_best = score
            if local_best[0] + 1e-12 < best[0]:
                params = local_params
                best = local_best
                improved = True
        if not improved:
            steps *= 0.5

    matrix = _matrix_from_params(model, params, width, height)
    point_errors, line_errors, details = _evidence_errors(
        matrix=matrix,
        point_map=point_map,
        line_map=line_map,
        point_annotations=point_annotations,
        line_annotations=line_annotations,
    )
    all_errors = point_errors + line_errors

    return {
        "parameters": [float(value) for value in params],
        "matrix": matrix.tolist(),
        "objective": float(best[0]),
        "data_loss": float(best[1]),
        "regularization": float(best[2]),
        "optimization_rounds": int(rounds),
        "residual_summary": _summary(all_errors),
        "point_summary": _summary(point_errors),
        "line_summary": _summary(line_errors),
        "annotation_details": details,
        "refined_geometry": transform_projected_geometry(
            {
                "lines": [
                    {"name": name, "points": [[p[0], p[1]] for p in points]}
                    for name, points in line_map.items()
                ],
                "curves": [],
                "points": [
                    {"name": name, "point": [p[0], p[1]]}
                    for name, p in point_map.items()
                ],
            },
            matrix,
        ),
    }


def _matrix_interpretation(
    matrix: Sequence[Sequence[float]],
    width: float,
    height: float,
) -> dict[str, Any]:
    m = np.asarray(matrix, dtype=float)
    cx, cy = width / 2.0, height / 2.0
    q0 = _apply_matrix_point(m, (cx, cy))
    shift = (q0[0] - cx, q0[1] - cy)
    diagonal = math.hypot(width, height)

    # Numerical Jacobian around image center.
    eps = 1.0
    qx = _apply_matrix_point(m, (cx + eps, cy))
    qy = _apply_matrix_point(m, (cx, cy + eps))
    jac = np.array(
        [
            [(qx[0] - q0[0]) / eps, (qy[0] - q0[0]) / eps],
            [(qx[1] - q0[1]) / eps, (qy[1] - q0[1]) / eps],
        ],
        dtype=float,
    )
    u, singular_values, vt = np.linalg.svd(jac)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    major = float(max(singular_values[0], EPS))
    minor = float(max(singular_values[1], EPS))
    scale = math.sqrt(major * minor)
    rotation_deg = math.degrees(math.atan2(rotation[1, 0], rotation[0, 0]))

    normalized_matrix = m / (m[2, 2] if abs(float(m[2, 2])) > EPS else 1.0)
    return {
        "center_shift_px": [float(shift[0]), float(shift[1])],
        "center_shift_normalized": [float(shift[0] / diagonal), float(shift[1] / diagonal)],
        "scale_geometric": float(scale),
        "log_scale": float(math.log(max(scale, EPS))),
        "scale_major": major,
        "scale_minor": minor,
        "anisotropy_ratio": float(major / minor),
        "rotation_deg": float(rotation_deg),
        "projective_terms_per_diagonal": [
            float(normalized_matrix[2, 0] * diagonal),
            float(normalized_matrix[2, 1] * diagonal),
        ],
    }


def _recommend_model(
    models: dict[str, dict[str, Any]],
    *,
    diagonal: float,
) -> str:
    usable = [
        name
        for name in MODEL_ORDER
        if name in models and "error" not in models[name]
    ]
    if not usable:
        raise DerivedRefinementError("No correction model could be fitted.")

    adequacy = diagonal * DEFAULT_ADEQUACY_P95_DIAGONAL_FRACTION
    near_best = diagonal * DEFAULT_NEAR_BEST_DIAGONAL_FRACTION

    for name in usable:
        p95 = models[name]["residual_summary"]["p95_px"]
        if math.isfinite(p95) and p95 <= adequacy:
            return name

    best_name = min(
        usable,
        key=lambda name: models[name]["residual_summary"]["p95_px"],
    )
    best_p95 = models[best_name]["residual_summary"]["p95_px"]
    for name in usable:
        p95 = models[name]["residual_summary"]["p95_px"]
        if p95 <= best_p95 + near_best:
            return name
    return best_name


def refine_derived_projection_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Refine one transported DERIVED field projection from sparse target evidence.

    Expected payload:

    {
      "transported_geometry": {...},
      "video_width": 1920,
      "video_height": 1080,
      "point_annotations": [
        {"point_name": "...", "point": [u, v]}
      ],
      "annotations": [
        {"line_name": "...", "points": [[u1, v1], [u2, v2]]}
      ]
    }
    """
    if not isinstance(payload, dict):
        raise DerivedRefinementError("Payload must be an object.")

    width, height, diagonal = _validate_dimensions(
        payload.get("video_width"),
        payload.get("video_height"),
    )
    geometry = payload.get("transported_geometry")
    if not isinstance(geometry, dict):
        raise DerivedRefinementError("transported_geometry is required.")

    point_map, line_map = _geometry_maps(geometry)
    point_annotations, line_annotations = _parse_annotations(
        point_annotations=payload.get("point_annotations", []),
        line_annotations=payload.get("annotations", []),
        point_map=point_map,
        line_map=line_map,
        width=width,
        height=height,
    )

    constraints = _constraint_count(point_annotations, line_annotations)
    huber_delta = _finite(
        payload.get("huber_delta_px", DEFAULT_HUBER_DELTA_PX),
        "huber_delta_px",
    )
    regularization_weight = _finite(
        payload.get("regularization_weight", DEFAULT_REGULARIZATION_WEIGHT),
        "regularization_weight",
    )
    if huber_delta <= 0 or regularization_weight < 0:
        raise DerivedRefinementError("Refinement tuning values are invalid.")

    identity = np.eye(3, dtype=float)
    baseline_point_errors, baseline_line_errors, baseline_details = _evidence_errors(
        matrix=identity,
        point_map=point_map,
        line_map=line_map,
        point_annotations=point_annotations,
        line_annotations=line_annotations,
    )
    baseline_all = baseline_point_errors + baseline_line_errors

    models: dict[str, dict[str, Any]] = {}
    for model in MODEL_ORDER:
        required = _minimum_constraint_count(model)
        if constraints < required:
            models[model] = {
                "error": (
                    f"Insufficient sparse constraints for {model}: "
                    f"{constraints} available, {required} required."
                )
            }
            continue
        fitted = _optimize_model(
            model,
            width=width,
            height=height,
            diagonal=diagonal,
            point_map=point_map,
            line_map=line_map,
            point_annotations=point_annotations,
            line_annotations=line_annotations,
            huber_delta_px=huber_delta,
            regularization_weight=regularization_weight,
            max_rounds=int(payload.get("max_optimization_rounds", DEFAULT_MAX_OPTIMIZATION_ROUNDS)),
        )
        fitted["transform"] = _matrix_interpretation(
            fitted["matrix"],
            width,
            height,
        )
        # Re-apply to the full original geometry so curves/metadata are preserved.
        fitted["refined_geometry"] = transform_projected_geometry(
            geometry,
            fitted["matrix"],
        )
        models[model] = fitted

    recommended = _recommend_model(models, diagonal=diagonal)
    recommended_summary = models[recommended]["residual_summary"]
    baseline_summary = _summary(baseline_all)

    baseline_p95 = baseline_summary["p95_px"]
    refined_p95 = recommended_summary["p95_px"]
    improvement = (
        100.0 * (baseline_p95 - refined_p95) / baseline_p95
        if math.isfinite(baseline_p95) and baseline_p95 > EPS
        else 0.0
    )

    warnings: list[str] = []
    if constraints < 8:
        warnings.append(
            "Sparse evidence does not constrain every projective degree of freedom; "
            "identity regularization is intentionally carrying part of the solution."
        )
    if refined_p95 > diagonal * 0.006:
        warnings.append(
            "Refined target evidence still has a high P95 residual; treat this DERIVED state as low confidence."
        )
    if models[recommended]["regularization"] > models[recommended]["data_loss"]:
        warnings.append(
            "The prior-to-identity penalty is stronger than the remaining annotation loss; "
            "the correction is deliberately conservative."
        )

    return {
        "mode": "derived_refinement",
        "constraint_count": int(constraints),
        "point_annotation_count": len(point_annotations),
        "line_annotation_count": len(line_annotations),
        "baseline": {
            "residual_summary": baseline_summary,
            "point_summary": _summary(baseline_point_errors),
            "line_summary": _summary(baseline_line_errors),
            "annotation_details": baseline_details,
        },
        "models": models,
        "recommended_model": recommended,
        "recommended_p95_improvement_percent": float(improvement),
        "video_width": float(width),
        "video_height": float(height),
        "warnings": warnings,
        "note": (
            "DERIVED refinement applies a small regularized image-space correction "
            "to an existing propagated projection. It is not an independent target homography solve."
        ),
    }


# ---------------------------------------------------------------------------
# Synthetic self-check
# ---------------------------------------------------------------------------

def _synthetic_geometry() -> dict[str, Any]:
    return {
        "lines": [
            {
                "name": "halfway_line",
                "points": [[960.0, 250.0], [960.0, 850.0]],
            },
            {
                "name": "touchline_a",
                "points": [[350.0, 760.0], [1570.0, 760.0]],
            },
        ],
        "curves": [],
        "points": [
            {"name": "center_spot", "point": [960.0, 550.0]},
            {"name": "halfway_touchline_a", "point": [960.0, 760.0]},
            {"name": "left_penalty_spot", "point": [520.0, 535.0]},
        ],
    }


def _self_check() -> None:
    width, height = 1920.0, 1080.0
    geometry = _synthetic_geometry()

    truth_params = np.array([0.0070, -0.0045, math.log(1.040), math.radians(1.20)])
    truth_matrix = _matrix_from_params("similarity", truth_params, width, height)
    truth_geometry = transform_projected_geometry(geometry, truth_matrix)

    truth_points = {item["name"]: item["point"] for item in truth_geometry["points"]}
    truth_lines = {item["name"]: item["points"] for item in truth_geometry["lines"]}

    # Two direct semantic points + two partial semantic line segments.
    line_a = truth_lines["halfway_line"]
    line_b = truth_lines["touchline_a"]
    payload = {
        "transported_geometry": geometry,
        "video_width": width,
        "video_height": height,
        "point_annotations": [
            {"point_name": "center_spot", "point": truth_points["center_spot"]},
            {
                "point_name": "halfway_touchline_a",
                "point": truth_points["halfway_touchline_a"],
            },
        ],
        "annotations": [
            {
                "line_name": "halfway_line",
                "points": [
                    [
                        0.25 * line_a[0][0] + 0.75 * line_a[-1][0],
                        0.25 * line_a[0][1] + 0.75 * line_a[-1][1],
                    ],
                    [
                        0.70 * line_a[0][0] + 0.30 * line_a[-1][0],
                        0.70 * line_a[0][1] + 0.30 * line_a[-1][1],
                    ],
                ],
            },
            {
                "line_name": "touchline_a",
                "points": [
                    [
                        0.30 * line_b[0][0] + 0.70 * line_b[-1][0],
                        0.30 * line_b[0][1] + 0.70 * line_b[-1][1],
                    ],
                    [
                        0.75 * line_b[0][0] + 0.25 * line_b[-1][0],
                        0.75 * line_b[0][1] + 0.25 * line_b[-1][1],
                    ],
                ],
            },
        ],
    }

    result = refine_derived_projection_payload(payload)
    if result["recommended_model"] not in {"similarity", "affine"}:
        raise AssertionError(
            f"Expected similarity/affine recommendation, got {result['recommended_model']}"
        )

    chosen = result["models"][result["recommended_model"]]
    if chosen["residual_summary"]["p95_px"] > 1.25:
        raise AssertionError("Synthetic sparse refinement residual is too high.")

    recovered = chosen["transform"]
    if abs(recovered["scale_geometric"] - 1.040) > 0.012:
        raise AssertionError("Synthetic refinement did not recover scale closely enough.")

    # Ensure underconstrained high-complexity models remain unavailable.
    sparse = refine_derived_projection_payload(
        {
            "transported_geometry": geometry,
            "video_width": width,
            "video_height": height,
            "point_annotations": [
                {"point_name": "center_spot", "point": truth_points["center_spot"]}
            ],
            "annotations": [],
        }
    )
    if "error" not in sparse["models"]["similarity"]:
        raise AssertionError("Similarity should be rejected with only one point.")

    print("DERIVED refinement self-check: OK")
    print(f"Synthetic truth scale:      {1.040:.4f}")
    print(f"Recommended model:          {result['recommended_model']}")
    print(f"Recovered local scale:      {recovered['scale_geometric']:.4f}")
    print(
        "Baseline -> refined P95:   "
        f"{result['baseline']['residual_summary']['p95_px']:.3f} px -> "
        f"{chosen['residual_summary']['p95_px']:.3f} px"
    )
    print("Sparse-complexity rejection: OK")


if __name__ == "__main__":
    _self_check()
