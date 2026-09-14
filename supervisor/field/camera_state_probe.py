"""Camera-state relationship probe for RealJaenLab FIELD.

This module is intentionally diagnostic.  It does NOT define the final
CameraState architecture yet.  Instead, it compares several image-to-image
motion models between already calibrated HUMAN frames so we can discover the
smallest numeric representation that explains Veo viewpoint changes.

The probe assumes every supplied HUMAN frame belongs to the same physical
pitch and therefore uses one fixed pitch length/width across all states.
Changing pitch dimensions per frame would mix FIELD geometry with CAMERA state,
which is exactly what this experiment is designed to avoid.

For each state it consumes one complete calibration solve response and selects
one already-projected camera model (normally ``anisotropic_2p``).  It then
matches identical canonical pitch samples across frames and fits, in increasing
complexity:

- translation
- similarity (uniform zoom + rotation + translation)
- affine
- homography

No OpenCV/SciPy dependency is required; NumPy is enough.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Sequence

import numpy as np


EPS = 1e-10
MODEL_ORDER = ("translation", "similarity", "affine", "homography")
DEFAULT_PROJECTION_MODEL = "anisotropic_2p"
DEFAULT_VISIBLE_MARGIN_FRACTION = 0.04
DEFAULT_ADEQUACY_P95_DIAGONAL_FRACTION = 0.002
DEFAULT_NEAR_BEST_DIAGONAL_FRACTION = 0.0006
MIN_PRIMARY_CORRESPONDENCES = 8


class CameraStateProbeError(ValueError):
    """Invalid or insufficient input for the camera-state probe."""


@dataclass(frozen=True, slots=True)
class FrameState:
    state_id: str
    frame_index: int | None
    timestamp_seconds: float | None
    width: float
    height: float
    pitch_length: float
    pitch_width: float
    projection_model: str
    projected_geometry: dict[str, Any]


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def _finite_number(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise CameraStateProbeError(f"{label} must be numeric.") from exc
    if not math.isfinite(number):
        raise CameraStateProbeError(f"{label} must be finite.")
    return number


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _point(value: Sequence[float], label: str) -> tuple[float, float]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 2
    ):
        raise CameraStateProbeError(f"{label} must contain two coordinates.")
    x = _finite_number(value[0], f"{label}[0]")
    y = _finite_number(value[1], f"{label}[1]")
    return x, y


def _summary(errors: Sequence[float], diagonal: float) -> dict[str, float]:
    arr = np.asarray(errors, dtype=float)
    if arr.size == 0:
        return {
            "median_px": float("nan"),
            "p95_px": float("nan"),
            "max_px": float("nan"),
            "rmse_px": float("nan"),
            "median_diagonal_fraction": float("nan"),
            "p95_diagonal_fraction": float("nan"),
        }
    median = float(np.median(arr))
    p95 = float(np.percentile(arr, 95))
    maximum = float(np.max(arr))
    rmse = float(np.sqrt(np.mean(arr * arr)))
    diagonal = max(float(diagonal), EPS)
    return {
        "median_px": median,
        "p95_px": p95,
        "max_px": maximum,
        "rmse_px": rmse,
        "median_diagonal_fraction": median / diagonal,
        "p95_diagonal_fraction": p95 / diagonal,
    }


def _apply_matrix(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack([points, np.ones(len(points), dtype=float)])
    projected = (matrix @ homogeneous.T).T
    w = projected[:, 2]
    if np.any(np.abs(w) < EPS):
        raise CameraStateProbeError("A fitted transform maps a sample to infinity.")
    return projected[:, :2] / w[:, None]


def _residuals(matrix: np.ndarray, source: np.ndarray, target: np.ndarray) -> np.ndarray:
    predicted = _apply_matrix(matrix, source)
    return np.linalg.norm(predicted - target, axis=1)


def _normalize_matrix(matrix: np.ndarray) -> np.ndarray:
    result = np.asarray(matrix, dtype=float)
    if result.shape != (3, 3) or not np.all(np.isfinite(result)):
        raise CameraStateProbeError("Transform must be a finite 3x3 matrix.")
    if abs(float(result[2, 2])) > EPS:
        result = result / float(result[2, 2])
    else:
        norm = float(np.linalg.norm(result))
        if norm < EPS:
            raise CameraStateProbeError("Degenerate transform matrix.")
        result = result / norm
    return result


# ---------------------------------------------------------------------------
# Motion-model fits
# ---------------------------------------------------------------------------


def _fit_translation(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    delta = np.mean(target - source, axis=0)
    return np.array(
        [[1.0, 0.0, delta[0]], [0.0, 1.0, delta[1]], [0.0, 0.0, 1.0]],
        dtype=float,
    )


def _fit_similarity(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    if len(source) < 2:
        raise CameraStateProbeError("Similarity fit requires at least two samples.")

    src_mean = source.mean(axis=0)
    dst_mean = target.mean(axis=0)
    src_centered = source - src_mean
    dst_centered = target - dst_mean

    variance = float(np.sum(src_centered * src_centered) / len(source))
    if variance < EPS:
        raise CameraStateProbeError("Similarity source samples have no extent.")

    covariance = (dst_centered.T @ src_centered) / len(source)
    u, singular_values, vt = np.linalg.svd(covariance)
    correction = np.eye(2)
    if np.linalg.det(u @ vt) < 0:
        correction[1, 1] = -1.0

    rotation = u @ correction @ vt
    scale = float(np.trace(np.diag(singular_values) @ correction) / variance)
    if not math.isfinite(scale) or abs(scale) < EPS:
        raise CameraStateProbeError("Similarity scale is degenerate.")

    translation = dst_mean - scale * (rotation @ src_mean)
    matrix = np.eye(3, dtype=float)
    matrix[:2, :2] = scale * rotation
    matrix[:2, 2] = translation
    return matrix


def _fit_affine(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    if len(source) < 3:
        raise CameraStateProbeError("Affine fit requires at least three samples.")
    design = np.column_stack([source, np.ones(len(source), dtype=float)])
    params, _, rank, _ = np.linalg.lstsq(design, target, rcond=None)
    if rank < 3:
        raise CameraStateProbeError("Affine source geometry is degenerate.")
    matrix = np.array(
        [
            [params[0, 0], params[1, 0], params[2, 0]],
            [params[0, 1], params[1, 1], params[2, 1]],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    return matrix


def _hartley_transform(points: np.ndarray) -> np.ndarray:
    centroid = points.mean(axis=0)
    shifted = points - centroid
    mean_distance = float(np.mean(np.linalg.norm(shifted, axis=1)))
    if mean_distance < EPS:
        raise CameraStateProbeError("Homography samples have no spatial extent.")
    scale = math.sqrt(2.0) / mean_distance
    return np.array(
        [
            [scale, 0.0, -scale * centroid[0]],
            [0.0, scale, -scale * centroid[1]],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def _fit_homography(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    if len(source) < 4:
        raise CameraStateProbeError("Homography fit requires at least four samples.")

    ts = _hartley_transform(source)
    tt = _hartley_transform(target)
    src_n = _apply_matrix(ts, source)
    dst_n = _apply_matrix(tt, target)

    rows: list[list[float]] = []
    for (x, y), (u, v) in zip(src_n, dst_n):
        rows.append([-x, -y, -1.0, 0.0, 0.0, 0.0, u * x, u * y, u])
        rows.append([0.0, 0.0, 0.0, -x, -y, -1.0, v * x, v * y, v])

    a = np.asarray(rows, dtype=float)
    _, _, vt = np.linalg.svd(a)
    h = vt[-1].reshape(3, 3)
    denormalized = np.linalg.inv(tt) @ h @ ts
    return _normalize_matrix(denormalized)


def _fit_model(name: str, source: np.ndarray, target: np.ndarray) -> np.ndarray:
    if name == "translation":
        return _fit_translation(source, target)
    if name == "similarity":
        return _fit_similarity(source, target)
    if name == "affine":
        return _fit_affine(source, target)
    if name == "homography":
        return _fit_homography(source, target)
    raise CameraStateProbeError(f"Unknown camera-state model {name!r}.")


# ---------------------------------------------------------------------------
# Transform interpretation
# ---------------------------------------------------------------------------


def _project_xy(matrix: np.ndarray, x: float, y: float) -> tuple[float, float]:
    q = matrix @ np.array([x, y, 1.0], dtype=float)
    if abs(float(q[2])) < EPS:
        raise CameraStateProbeError("Transform center maps to infinity.")
    return float(q[0] / q[2]), float(q[1] / q[2])


def _homography_jacobian(matrix: np.ndarray, x: float, y: float) -> np.ndarray:
    h = matrix
    den = float(h[2, 0] * x + h[2, 1] * y + h[2, 2])
    if abs(den) < EPS:
        raise CameraStateProbeError("Cannot evaluate local transform Jacobian.")
    num_u = float(h[0, 0] * x + h[0, 1] * y + h[0, 2])
    num_v = float(h[1, 0] * x + h[1, 1] * y + h[1, 2])
    den2 = den * den
    return np.array(
        [
            [
                (h[0, 0] * den - num_u * h[2, 0]) / den2,
                (h[0, 1] * den - num_u * h[2, 1]) / den2,
            ],
            [
                (h[1, 0] * den - num_v * h[2, 0]) / den2,
                (h[1, 1] * den - num_v * h[2, 1]) / den2,
            ],
        ],
        dtype=float,
    )


def _local_linear_summary(jacobian: np.ndarray) -> dict[str, float]:
    u, singular_values, vt = np.linalg.svd(jacobian)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt

    s1 = float(max(singular_values[0], EPS))
    s2 = float(max(singular_values[1], EPS))
    geometric_scale = math.sqrt(s1 * s2)
    rotation_deg = math.degrees(math.atan2(rotation[1, 0], rotation[0, 0]))

    return {
        "scale_geometric": geometric_scale,
        "scale_major": s1,
        "scale_minor": s2,
        "anisotropy_ratio": s1 / s2,
        "rotation_deg": float(rotation_deg),
        "log_scale": float(math.log(max(geometric_scale, EPS))),
    }


def _interpret_matrix(matrix: np.ndarray, width: float, height: float) -> dict[str, Any]:
    matrix = _normalize_matrix(matrix)
    cx, cy = 0.5 * width, 0.5 * height
    mapped_x, mapped_y = _project_xy(matrix, cx, cy)
    jacobian = _homography_jacobian(matrix, cx, cy)
    diagonal = math.hypot(width, height)
    result: dict[str, Any] = {
        "matrix": matrix.tolist(),
        "center_source_px": [cx, cy],
        "center_target_px": [mapped_x, mapped_y],
        "center_shift_px": [mapped_x - cx, mapped_y - cy],
        "center_shift_normalized": [
            (mapped_x - cx) / max(width, EPS),
            (mapped_y - cy) / max(height, EPS),
        ],
        "local_jacobian": jacobian.tolist(),
        "projective_terms": [float(matrix[2, 0]), float(matrix[2, 1])],
        "projective_terms_per_diagonal": [
            float(matrix[2, 0] * diagonal),
            float(matrix[2, 1] * diagonal),
        ],
    }
    result.update(_local_linear_summary(jacobian))
    return result


# ---------------------------------------------------------------------------
# Projected-geometry parsing
# ---------------------------------------------------------------------------


def _subsample_indices(length: int, maximum: int) -> list[int]:
    if length <= 0:
        return []
    if length <= maximum:
        return list(range(length))
    return sorted({int(round(v)) for v in np.linspace(0, length - 1, maximum)})


def _geometry_catalog(geometry: dict[str, Any]) -> dict[str, tuple[float, float]]:
    if not isinstance(geometry, dict):
        raise CameraStateProbeError("projected_geometry must be an object.")

    catalog: dict[str, tuple[float, float]] = {}

    for item in geometry.get("points", []) or []:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            continue
        try:
            catalog[f"point:{item['name']}"] = _point(item.get("point"), item["name"])
        except CameraStateProbeError:
            continue

    for family, maximum in (("lines", 7), ("curves", 11)):
        for item in geometry.get(family, []) or []:
            if not isinstance(item, dict) or not isinstance(item.get("name"), str):
                continue
            raw_points = item.get("points")
            if not isinstance(raw_points, Sequence) or isinstance(raw_points, (str, bytes)):
                continue
            indices = _subsample_indices(len(raw_points), maximum)
            for slot, index in enumerate(indices):
                try:
                    catalog[f"{family[:-1]}:{item['name']}:{slot}"] = _point(
                        raw_points[index], f"{family}:{item['name']}:{index}"
                    )
                except CameraStateProbeError:
                    continue

    if len(catalog) < 4:
        raise CameraStateProbeError(
            "Projected geometry does not contain enough finite canonical samples."
        )
    return catalog


def _select_projected_geometry(solve: dict[str, Any], projection_model: str) -> dict[str, Any]:
    if projection_model == "pure_homography":
        geometry = solve.get("projected_geometry")
        if not isinstance(geometry, dict):
            raise CameraStateProbeError("Pure-homography projected_geometry is missing.")
        return geometry

    probe = solve.get("camera_model_probe")
    models = probe.get("models") if isinstance(probe, dict) else None
    model = models.get(projection_model) if isinstance(models, dict) else None
    geometry = model.get("projected_geometry") if isinstance(model, dict) else None
    if not isinstance(geometry, dict):
        raise CameraStateProbeError(
            f"Projected geometry for camera model {projection_model!r} is missing."
        )
    return geometry


def _parse_state(raw: dict[str, Any], default_model: str, index: int) -> FrameState:
    if not isinstance(raw, dict):
        raise CameraStateProbeError("Each camera state must be an object.")
    solve = raw.get("solve")
    if not isinstance(solve, dict):
        raise CameraStateProbeError("Each camera state requires a solve response.")

    state_id = str(raw.get("state_id") or raw.get("id") or f"state_{index}")
    model = str(raw.get("projection_model") or default_model)

    width = _finite_number(
        raw.get("video_width", solve.get("video_width")), f"{state_id}.video_width"
    )
    height = _finite_number(
        raw.get("video_height", solve.get("video_height")), f"{state_id}.video_height"
    )
    if width <= 0.0 or height <= 0.0:
        raise CameraStateProbeError(f"{state_id} requires positive video dimensions.")

    pitch_length = _finite_number(solve.get("pitch_length"), f"{state_id}.pitch_length")
    pitch_width = _finite_number(solve.get("pitch_width"), f"{state_id}.pitch_width")

    return FrameState(
        state_id=state_id,
        frame_index=_optional_int(raw.get("frame_index", solve.get("frame_index"))),
        timestamp_seconds=_optional_float(
            raw.get("timestamp_seconds", solve.get("timestamp_seconds"))
        ),
        width=width,
        height=height,
        pitch_length=pitch_length,
        pitch_width=pitch_width,
        projection_model=model,
        projected_geometry=_select_projected_geometry(solve, model),
    )


def _validate_shared_field(states: Sequence[FrameState], allow_mixed: bool) -> list[str]:
    warnings: list[str] = []
    reference = states[0]
    for state in states[1:]:
        same_length = math.isclose(state.pitch_length, reference.pitch_length, abs_tol=1e-6)
        same_width = math.isclose(state.pitch_width, reference.pitch_width, abs_tol=1e-6)
        if same_length and same_width:
            continue
        message = (
            "Camera-state probe received mixed pitch dimensions: "
            f"{reference.state_id}={reference.pitch_length:.3f}x{reference.pitch_width:.3f}, "
            f"{state.state_id}={state.pitch_length:.3f}x{state.pitch_width:.3f}. "
            "Pitch dimensions must stay fixed while studying camera state."
        )
        if not allow_mixed:
            raise CameraStateProbeError(message)
        warnings.append(message)
    return warnings


def _validate_shared_resolution(states: Sequence[FrameState]) -> list[str]:
    warnings: list[str] = []
    reference = states[0]
    for state in states[1:]:
        if math.isclose(state.width, reference.width) and math.isclose(state.height, reference.height):
            continue
        warnings.append(
            "States use different image resolutions; normalized diagnostics remain valid "
            "but raw pixel residuals are less directly comparable."
        )
        break
    return warnings


# ---------------------------------------------------------------------------
# Pairwise relation analysis
# ---------------------------------------------------------------------------


def _inside(point: tuple[float, float], width: float, height: float, margin_fraction: float) -> bool:
    margin_x = margin_fraction * width
    margin_y = margin_fraction * height
    return (
        -margin_x <= point[0] <= width + margin_x
        and -margin_y <= point[1] <= height + margin_y
    )


def _common_correspondences(
    source_state: FrameState,
    target_state: FrameState,
    margin_fraction: float,
) -> tuple[np.ndarray, np.ndarray, list[str], str, int]:
    source_catalog = _geometry_catalog(source_state.projected_geometry)
    target_catalog = _geometry_catalog(target_state.projected_geometry)
    common = sorted(set(source_catalog).intersection(target_catalog))
    if len(common) < 4:
        raise CameraStateProbeError(
            f"States {source_state.state_id!r} and {target_state.state_id!r} "
            "do not share enough canonical projection samples."
        )

    overlap = [
        key
        for key in common
        if _inside(source_catalog[key], source_state.width, source_state.height, margin_fraction)
        and _inside(target_catalog[key], target_state.width, target_state.height, margin_fraction)
    ]

    if len(overlap) >= MIN_PRIMARY_CORRESPONDENCES:
        selected = overlap
        scope = "visible_overlap"
    else:
        selected = common
        scope = "all_field_fallback"

    source = np.asarray([source_catalog[key] for key in selected], dtype=float)
    target = np.asarray([target_catalog[key] for key in selected], dtype=float)
    return source, target, selected, scope, len(overlap)


def _fit_relation_models(
    source: np.ndarray,
    target: np.ndarray,
    target_width: float,
    target_height: float,
    adequacy_p95_px: float,
    near_best_margin_px: float,
) -> tuple[dict[str, Any], str]:
    diagonal = math.hypot(target_width, target_height)
    models: dict[str, Any] = {}

    for name in MODEL_ORDER:
        try:
            matrix = _fit_model(name, source, target)
            errors = _residuals(matrix, source, target)
            summary = _summary(errors, diagonal)
            models[name] = {
                "name": name,
                "degrees_of_freedom": {
                    "translation": 2,
                    "similarity": 4,
                    "affine": 6,
                    "homography": 8,
                }[name],
                "residual_summary": summary,
                "transform": _interpret_matrix(matrix, target_width, target_height),
            }
        except CameraStateProbeError as exc:
            models[name] = {"name": name, "error": str(exc)}

    valid = [name for name in MODEL_ORDER if "residual_summary" in models[name]]
    if not valid:
        raise CameraStateProbeError("No camera-state relation model could be fitted.")

    best = min(valid, key=lambda name: models[name]["residual_summary"]["p95_px"])
    best_p95 = float(models[best]["residual_summary"]["p95_px"])

    recommended = best
    for name in valid:
        metrics = models[name]["residual_summary"]
        p95 = float(metrics["p95_px"])
        maximum = float(metrics["max_px"])
        if (
            p95 <= adequacy_p95_px
            and p95 <= best_p95 + near_best_margin_px
            and maximum <= 3.0 * adequacy_p95_px
        ):
            recommended = name
            break

    for index, name in enumerate(valid):
        metrics = models[name]["residual_summary"]
        if index == 0:
            models[name]["p95_improvement_from_previous_percent"] = None
            continue
        previous = valid[index - 1]
        previous_p95 = float(models[previous]["residual_summary"]["p95_px"])
        current_p95 = float(metrics["p95_px"])
        if previous_p95 > EPS:
            improvement = 100.0 * (previous_p95 - current_p95) / previous_p95
        else:
            improvement = 0.0
        models[name]["p95_improvement_from_previous_percent"] = float(improvement)

    return models, recommended


def _relation(
    source_state: FrameState,
    target_state: FrameState,
    margin_fraction: float,
    adequacy_fraction: float,
    near_best_fraction: float,
) -> dict[str, Any]:
    source, target, keys, scope, visible_overlap_count = _common_correspondences(
        source_state, target_state, margin_fraction
    )
    diagonal = math.hypot(target_state.width, target_state.height)
    adequacy_p95_px = max(1.5, adequacy_fraction * diagonal)
    near_best_margin_px = max(0.75, near_best_fraction * diagonal)

    models, recommended = _fit_relation_models(
        source,
        target,
        target_state.width,
        target_state.height,
        adequacy_p95_px,
        near_best_margin_px,
    )

    warnings: list[str] = []
    if scope == "all_field_fallback":
        warnings.append(
            "Too few canonical samples are simultaneously visible in both frames; "
            "the relation was fitted using complete-field extrapolated geometry."
        )

    return {
        "source_state_id": source_state.state_id,
        "target_state_id": target_state.state_id,
        "source_frame_index": source_state.frame_index,
        "target_frame_index": target_state.frame_index,
        "source_timestamp_seconds": source_state.timestamp_seconds,
        "target_timestamp_seconds": target_state.timestamp_seconds,
        "projection_model": source_state.projection_model,
        "correspondence_scope": scope,
        "correspondence_count": len(keys),
        "visible_overlap_count": visible_overlap_count,
        "adequacy_p95_px": float(adequacy_p95_px),
        "near_best_margin_px": float(near_best_margin_px),
        "recommended_model": recommended,
        "models": models,
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# Public payload API
# ---------------------------------------------------------------------------


def probe_camera_states_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Compare multiple HUMAN calibrations and probe a minimal CameraState model.

    Expected payload shape::

        {
          "projection_model": "anisotropic_2p",
          "reference_state_id": "frame_1000",          # optional
          "states": [
            {
              "state_id": "frame_1000",
              "frame_index": 1000,
              "timestamp_seconds": 40.0,
              "solve": <complete /api/field/calibration/solve response>
            },
            ...
          ]
        }

    Every state should use the SAME pitch dimensions.  Mixed dimensions are
    rejected by default because they confound FIELD size with CAMERA state.
    """

    if not isinstance(payload, dict):
        raise CameraStateProbeError("Payload must be an object.")
    raw_states = payload.get("states")
    if (
        not isinstance(raw_states, Sequence)
        or isinstance(raw_states, (str, bytes))
        or len(raw_states) < 2
    ):
        raise CameraStateProbeError("At least two HUMAN camera states are required.")

    default_model = str(payload.get("projection_model") or DEFAULT_PROJECTION_MODEL)
    states = [_parse_state(raw, default_model, index) for index, raw in enumerate(raw_states)]

    state_ids = [state.state_id for state in states]
    if len(set(state_ids)) != len(state_ids):
        raise CameraStateProbeError("Camera state identifiers must be unique.")

    if len({state.projection_model for state in states}) != 1:
        raise CameraStateProbeError(
            "All states must use the same projection_model for a meaningful comparison."
        )

    allow_mixed = bool(payload.get("allow_mixed_pitch_dimensions", False))
    warnings = _validate_shared_field(states, allow_mixed)
    warnings.extend(_validate_shared_resolution(states))

    reference_id = str(payload.get("reference_state_id") or states[0].state_id)
    by_id = {state.state_id: state for state in states}
    if reference_id not in by_id:
        raise CameraStateProbeError(f"Unknown reference_state_id {reference_id!r}.")
    reference = by_id[reference_id]

    margin_fraction = float(
        payload.get("visible_margin_fraction", DEFAULT_VISIBLE_MARGIN_FRACTION)
    )
    adequacy_fraction = float(
        payload.get(
            "adequacy_p95_diagonal_fraction",
            DEFAULT_ADEQUACY_P95_DIAGONAL_FRACTION,
        )
    )
    near_best_fraction = float(
        payload.get("near_best_diagonal_fraction", DEFAULT_NEAR_BEST_DIAGONAL_FRACTION)
    )

    relations = []
    for state in states:
        if state.state_id == reference.state_id:
            continue
        relations.append(
            _relation(
                reference,
                state,
                margin_fraction,
                adequacy_fraction,
                near_best_fraction,
            )
        )

    pairwise_summary = []
    for i, source in enumerate(states):
        for target in states[i + 1 :]:
            relation = _relation(
                source,
                target,
                margin_fraction,
                adequacy_fraction,
                near_best_fraction,
            )
            chosen = relation["models"][relation["recommended_model"]]
            pairwise_summary.append(
                {
                    "source_state_id": source.state_id,
                    "target_state_id": target.state_id,
                    "recommended_model": relation["recommended_model"],
                    "p95_px": chosen["residual_summary"]["p95_px"],
                    "median_px": chosen["residual_summary"]["median_px"],
                    "correspondence_scope": relation["correspondence_scope"],
                    "visible_overlap_count": relation["visible_overlap_count"],
                }
            )

    recommendation_counts = {name: 0 for name in MODEL_ORDER}
    for relation in relations:
        recommendation_counts[relation["recommended_model"]] += 1

    dominant_model = max(
        MODEL_ORDER,
        key=lambda name: (recommendation_counts[name], -MODEL_ORDER.index(name)),
    )

    return {
        "purpose": (
            "Diagnostic comparison of numeric camera-state relation models. "
            "This does not define the production CameraState yet."
        ),
        "projection_model": reference.projection_model,
        "pitch_profile": {
            "length": reference.pitch_length,
            "width": reference.pitch_width,
        },
        "reference_state_id": reference.state_id,
        "state_count": len(states),
        "states": [
            {
                "state_id": state.state_id,
                "frame_index": state.frame_index,
                "timestamp_seconds": state.timestamp_seconds,
                "video_width": state.width,
                "video_height": state.height,
                "projection_model": state.projection_model,
            }
            for state in states
        ],
        "relations_to_reference": relations,
        "pairwise_summary": pairwise_summary,
        "recommendation_counts": recommendation_counts,
        "dominant_recommended_model": dominant_model,
        "warnings": warnings,
        "interpretation": {
            "translation": "View change is mostly image displacement.",
            "similarity": "A single uniform zoom + rotation + displacement explains the view change.",
            "affine": "Different axis scale/shear is materially useful beyond uniform zoom.",
            "homography": "Projective change is needed; a simple pan/zoom state is insufficient.",
        },
    }


# ---------------------------------------------------------------------------
# Synthetic self-check
# ---------------------------------------------------------------------------


def _synthetic_geometry(points: np.ndarray) -> dict[str, Any]:
    return {
        "points": [
            {"name": f"p{i}", "label": f"P{i}", "category": "synthetic", "point": p.tolist()}
            for i, p in enumerate(points)
        ],
        "lines": [],
        "curves": [],
    }


def _synthetic_solve(points: np.ndarray, length: float = 105.0, width: float = 68.0) -> dict[str, Any]:
    return {
        "pitch_length": length,
        "pitch_width": width,
        "video_width": 1920,
        "video_height": 1080,
        "projected_geometry": _synthetic_geometry(points),
    }


def _self_check() -> None:
    xx, yy = np.meshgrid(np.linspace(250.0, 1600.0, 5), np.linspace(180.0, 900.0, 4))
    base = np.column_stack([xx.ravel(), yy.ravel()])

    angle = math.radians(3.5)
    scale = 1.18
    rotation = np.array(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]],
        dtype=float,
    )
    similarity = np.eye(3, dtype=float)
    similarity[:2, :2] = scale * rotation
    similarity[:2, 2] = [84.0, -37.0]
    moved = _apply_matrix(similarity, base)

    payload = {
        "projection_model": "pure_homography",
        "reference_state_id": "A",
        "states": [
            {"state_id": "A", "frame_index": 100, "solve": _synthetic_solve(base)},
            {"state_id": "B", "frame_index": 200, "solve": _synthetic_solve(moved)},
        ],
    }
    result = probe_camera_states_payload(payload)
    relation = result["relations_to_reference"][0]
    recovered = relation["models"]["similarity"]["transform"]
    recovered_scale = float(recovered["scale_geometric"])
    recovered_rotation = float(recovered["rotation_deg"])
    p95 = float(relation["models"]["similarity"]["residual_summary"]["p95_px"])

    if abs(recovered_scale - scale) > 1e-6:
        raise AssertionError((recovered_scale, scale))
    if abs(recovered_rotation - math.degrees(angle)) > 1e-6:
        raise AssertionError((recovered_rotation, math.degrees(angle)))
    if p95 > 1e-6:
        raise AssertionError(p95)
    if relation["recommended_model"] != "similarity":
        raise AssertionError(relation["recommended_model"])

    mixed_rejected = False
    bad_payload = dict(payload)
    bad_payload["states"] = [
        payload["states"][0],
        {"state_id": "B", "solve": _synthetic_solve(moved, length=110.0)},
    ]
    try:
        probe_camera_states_payload(bad_payload)
    except CameraStateProbeError:
        mixed_rejected = True
    if not mixed_rejected:
        raise AssertionError("Mixed pitch dimensions were not rejected.")

    print("Camera state probe self-check: OK")
    print("Synthetic truth: similarity")
    print(f"Recovered scale:    {recovered_scale:.6f} (truth {scale:.6f})")
    print(
        f"Recovered rotation: {recovered_rotation:+.6f} deg "
        f"(truth {math.degrees(angle):+.6f})"
    )
    print(f"Similarity P95:     {p95:.6f} px")
    print(f"Recommended model:  {relation['recommended_model']}")
    print("Mixed-pitch rejection: OK")


if __name__ == "__main__":
    _self_check()
