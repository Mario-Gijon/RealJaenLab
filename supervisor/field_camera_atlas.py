"""Camera Atlas V1 for RealJaen FIELD.

This is an analysis module, not a production tracker.

Rules:
- Only strong manual HUMAN calibrations build the atlas.
- Propagated/refined frames never teach the atlas.
- Camera states are compared through the projection of the canonical 105x68 m
  pitch, not through raw homography coefficients.
- V1 measures low-dimensional structure and performs a leave-one-anchor-out
  interpolation benchmark on gold anchors.

No video decoding, optical flow, line detection, review mutation or automatic
homography update lives here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import numpy as np

from supervisor.field_visibility import compute_visible_pitch_footprint


_SIGNATURE_WORLD = np.asarray(
    [
        (-52.5, -34.0), (0.0, -34.0), (52.5, -34.0),
        (-52.5, -20.16), (-36.0, -20.16), (0.0, -20.16),
        (36.0, -20.16), (52.5, -20.16),
        (-52.5, 0.0), (-41.5, 0.0), (-36.0, 0.0),
        (0.0, 0.0),
        (36.0, 0.0), (41.5, 0.0), (52.5, 0.0),
        (-52.5, 20.16), (-36.0, 20.16), (0.0, 20.16),
        (36.0, 20.16), (52.5, 20.16),
        (-52.5, 34.0), (0.0, 34.0), (52.5, 34.0),
        (-47.0, -9.16), (-47.0, 9.16),
        (47.0, -9.16), (47.0, 9.16),
        (0.0, -9.15), (0.0, 9.15),
    ],
    dtype=np.float64,
)


@dataclass(frozen=True)
class CameraAtlasConfig:
    pca_target_variance: float = 0.95
    pca_secondary_variance: float = 0.99
    max_reported_components: int = 6
    projection_feature_clip: float = 5.0
    reject_explicit_manual_refine: bool = True


def _as_h(matrix: Any) -> np.ndarray | None:
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
        or abs(determinant) < 1e-12
        or not np.isfinite(condition)
        or condition > 1e13
    ):
        return None
    return h


def _project_points(points_xy: np.ndarray, h: np.ndarray):
    points = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    homogeneous = np.column_stack((points, np.ones(len(points))))
    q = homogeneous @ h.T
    denominator = q[:, 2]

    valid = (
        np.isfinite(q).all(axis=1)
        & np.isfinite(denominator)
        & (np.abs(denominator) > 1e-10)
    )

    projected = np.full((len(points), 2), np.nan, dtype=np.float64)
    projected[valid] = q[valid, :2] / denominator[valid, None]
    valid &= np.isfinite(projected).all(axis=1)
    return projected, valid


def _selected_strategy(calibration: dict[str, Any]) -> str | None:
    for key in ("diagnostics", "solverDiagnostics", "solver_diagnostics"):
        diagnostics = calibration.get(key)
        if not isinstance(diagnostics, dict):
            continue
        strategy = diagnostics.get("calibration_strategy")
        if isinstance(strategy, dict):
            selected = strategy.get("selected_mode")
            if selected in {"full", "refine"}:
                return str(selected)

    for key in ("selectedMode", "selected_mode", "solveMode", "solve_mode"):
        selected = calibration.get(key)
        if selected in {"full", "refine"}:
            return str(selected)

    return None


def classify_anchor(
    calibration: dict[str, Any],
    *,
    config: CameraAtlasConfig | None = None,
) -> dict[str, Any]:
    """Apply the gold-anchor policy to one saved calibration."""
    cfg = config or CameraAtlasConfig()

    if not isinstance(calibration, dict):
        return {"strong": False, "reason": "invalid_record", "basis": None}

    if str(calibration.get("source", "")).lower() != "manual":
        return {
            "strong": False,
            "reason": "not_manual",
            "basis": calibration.get("source"),
        }

    if _as_h(calibration.get("homography")) is None:
        return {
            "strong": False,
            "reason": "invalid_homography",
            "basis": "manual",
        }

    selected = _selected_strategy(calibration)
    if cfg.reject_explicit_manual_refine and selected == "refine":
        return {
            "strong": False,
            "reason": "manual_depends_on_prior_refine",
            "basis": "manual_refine",
        }

    return {
        "strong": True,
        "reason": "strong_manual_anchor",
        "basis": "manual_full" if selected == "full" else "manual_legacy_or_unspecified",
    }


def _camera_diagnostics(
    h: np.ndarray,
    image_width: int,
    image_height: int,
) -> dict[str, Any]:
    support = np.asarray(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
        dtype=np.float64,
    )
    projected, valid = _project_points(support, h)

    centre_norm = None
    zoom_proxy = None

    if valid[0]:
        centre_norm = [
            float(projected[0, 0] / max(float(image_width), 1.0)),
            float(projected[0, 1] / max(float(image_height), 1.0)),
        ]

    if np.all(valid):
        sx = float(np.linalg.norm(projected[1] - projected[0]))
        sy = float(np.linalg.norm(projected[2] - projected[0]))
        if sx > 0.0 and sy > 0.0:
            zoom_proxy = float(math.sqrt(sx * sy))

    return {
        "centre_projection_norm": centre_norm,
        "zoom_proxy_px_per_m": zoom_proxy,
    }


def _projection_signature(
    h: np.ndarray,
    image_width: int,
    image_height: int,
):
    projected, valid = _project_points(_SIGNATURE_WORLD, h)
    normalised = np.full_like(projected, np.nan)

    if np.any(valid):
        normalised[valid, 0] = projected[valid, 0] / max(float(image_width), 1.0)
        normalised[valid, 1] = projected[valid, 1] / max(float(image_height), 1.0)

    return normalised, valid


def _feature_vector(
    signature: np.ndarray,
    valid: np.ndarray,
    config: CameraAtlasConfig,
) -> np.ndarray | None:
    if not np.all(valid) or not np.isfinite(signature).all():
        return None
    clipped = np.clip(
        signature,
        -config.projection_feature_clip,
        config.projection_feature_clip,
    )
    return clipped.reshape(-1)


def camera_distance(
    first_homography: Any,
    second_homography: Any,
) -> dict[str, float] | None:
    """Distance in image pixels between two cameras on canonical support."""
    first_h = _as_h(first_homography)
    second_h = _as_h(second_homography)
    if first_h is None or second_h is None:
        return None

    first, first_valid = _project_points(_SIGNATURE_WORLD, first_h)
    second, second_valid = _project_points(_SIGNATURE_WORLD, second_h)

    common = first_valid & second_valid
    if int(np.count_nonzero(common)) < 4:
        return None

    distances = np.linalg.norm(first[common] - second[common], axis=1)

    return {
        "median_px": float(np.median(distances)),
        "p90_px": float(np.percentile(distances, 90)),
        "max_px": float(np.max(distances)),
    }


def _normalise_points_for_dlt(points: np.ndarray):
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(points) < 4 or not np.isfinite(points).all():
        return None

    centroid = np.mean(points, axis=0)
    shifted = points - centroid
    mean_distance = float(np.mean(np.linalg.norm(shifted, axis=1)))
    if mean_distance < 1e-12:
        return None

    scale = math.sqrt(2.0) / mean_distance
    transform = np.asarray(
        [
            [scale, 0.0, -scale * centroid[0]],
            [0.0, scale, -scale * centroid[1]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    homogeneous = np.column_stack((points, np.ones(len(points))))
    normalised_h = homogeneous @ transform.T
    return normalised_h[:, :2], transform


def _solve_homography_dlt(
    world_points: np.ndarray,
    image_points: np.ndarray,
) -> np.ndarray | None:
    world_result = _normalise_points_for_dlt(world_points)
    image_result = _normalise_points_for_dlt(image_points)
    if world_result is None or image_result is None:
        return None

    world_norm, world_transform = world_result
    image_norm, image_transform = image_result

    rows = []
    for (x, y), (u, v) in zip(world_norm, image_norm):
        rows.append([-x, -y, -1.0, 0.0, 0.0, 0.0, u*x, u*y, u])
        rows.append([0.0, 0.0, 0.0, -x, -y, -1.0, v*x, v*y, v])

    try:
        _, _, vt = np.linalg.svd(np.asarray(rows, dtype=np.float64))
        normalised_h = vt[-1].reshape(3, 3)
        h = np.linalg.inv(image_transform) @ normalised_h @ world_transform
    except np.linalg.LinAlgError:
        return None

    return _as_h(h)


def interpolate_camera(
    left_homography: Any,
    right_homography: Any,
    alpha: float,
) -> list[list[float]] | None:
    """Interpolate observable pitch projection, never raw H coefficients."""
    left_h = _as_h(left_homography)
    right_h = _as_h(right_homography)
    if left_h is None or right_h is None:
        return None

    alpha = float(max(0.0, min(1.0, alpha)))

    left_projection, left_valid = _project_points(_SIGNATURE_WORLD, left_h)
    right_projection, right_valid = _project_points(_SIGNATURE_WORLD, right_h)
    common = left_valid & right_valid

    if int(np.count_nonzero(common)) < 6:
        return None

    world = _SIGNATURE_WORLD[common]
    image = (
        (1.0 - alpha) * left_projection[common]
        + alpha * right_projection[common]
    )

    h = _solve_homography_dlt(world, image)
    return h.tolist() if h is not None else None


def _pca_analysis(
    feature_matrix: np.ndarray,
    config: CameraAtlasConfig,
) -> dict[str, Any]:
    matrix = np.asarray(feature_matrix, dtype=np.float64)

    if matrix.ndim != 2 or matrix.shape[0] < 2:
        return {
            "status": "not_enough_anchors",
            "explained_variance_ratio": [],
            "cumulative_variance": [],
            "dimension_95": None,
            "dimension_99": None,
            "state_coordinates": [],
        }

    # Image-normalised projected coordinates already share the same units.
    # Centre the cloud but do not standardise every feature independently:
    # per-coordinate standardisation would amplify almost-static landmarks and
    # can make a genuinely low-dimensional camera family look artificially
    # higher-dimensional.
    mean = np.mean(matrix, axis=0)
    centred = matrix - mean

    global_scale = float(np.sqrt(np.mean(centred ** 2)))
    if global_scale < 1e-10:
        global_scale = 1.0
    normalised = centred / global_scale

    try:
        _, singular_values, vt = np.linalg.svd(
            normalised,
            full_matrices=False,
        )
    except np.linalg.LinAlgError:
        return {
            "status": "svd_failed",
            "explained_variance_ratio": [],
            "cumulative_variance": [],
            "dimension_95": None,
            "dimension_99": None,
            "state_coordinates": [],
        }

    eigenvalues = (singular_values ** 2) / max(matrix.shape[0] - 1, 1)
    total = float(np.sum(eigenvalues))
    ratios = eigenvalues / total if total > 1e-12 else np.zeros_like(eigenvalues)
    cumulative = np.cumsum(ratios)

    def dimension_for(target: float):
        matches = np.flatnonzero(cumulative >= target)
        return int(matches[0] + 1) if len(matches) else None

    count = min(config.max_reported_components, len(singular_values))
    coordinates = normalised @ vt[:count].T

    return {
        "status": "ok",
        "explained_variance_ratio": ratios[:count].tolist(),
        "cumulative_variance": cumulative[:count].tolist(),
        "dimension_95": dimension_for(config.pca_target_variance),
        "dimension_99": dimension_for(config.pca_secondary_variance),
        "state_coordinates": coordinates.tolist(),
    }


def build_camera_atlas(
    calibrations: list[dict[str, Any]],
    image_width: int,
    image_height: int,
    *,
    video_name: str | None = None,
    config: CameraAtlasConfig | None = None,
) -> dict[str, Any]:
    """Build Camera Atlas V1 from strong manual anchors only."""
    cfg = config or CameraAtlasConfig()

    try:
        width = int(image_width)
        height = int(image_height)
    except (TypeError, ValueError):
        width = 0
        height = 0

    if width <= 0 or height <= 0:
        return {
            "status": "invalid_image_size",
            "anchors": [],
            "excluded": [],
            "config": asdict(cfg),
        }

    candidates_by_frame: dict[int, dict[str, Any]] = {}
    excluded = []

    for calibration in calibrations if isinstance(calibrations, list) else []:
        if not isinstance(calibration, dict):
            continue

        if video_name is not None:
            calibration_video = calibration.get("video")
            if (
                calibration_video is not None
                and calibration_video != video_name
            ):
                continue

        try:
            frame = int(calibration.get("frame"))
        except (TypeError, ValueError):
            excluded.append({"frame": None, "reason": "invalid_frame"})
            continue

        candidates_by_frame[frame] = calibration

    internal_anchors = []

    for frame, calibration in sorted(candidates_by_frame.items()):
        classification = classify_anchor(calibration, config=cfg)

        if not classification["strong"]:
            excluded.append({"frame": frame, **classification})
            continue

        h = _as_h(calibration.get("homography"))
        signature, valid = _projection_signature(h, width, height)
        feature = _feature_vector(signature, valid, cfg)

        if feature is None:
            excluded.append({
                "frame": frame,
                "strong": False,
                "reason": "signature_projection_failed",
            })
            continue

        internal_anchors.append({
            "frame": frame,
            "homography": h.tolist(),
            "gold_basis": classification["basis"],
            "camera": _camera_diagnostics(h, width, height),
            "visibility": compute_visible_pitch_footprint(
                h,
                width,
                height,
            ),
            "_feature": feature,
        })

    if not internal_anchors:
        return {
            "status": "no_strong_anchors",
            "anchor_count": 0,
            "anchors": [],
            "excluded": excluded,
            "analysis": {
                "pca": _pca_analysis(
                    np.empty((0, len(_SIGNATURE_WORLD) * 2)),
                    cfg,
                ),
                "leave_one_out": [],
                "leave_one_out_summary": None,
            },
            "config": asdict(cfg),
        }

    feature_matrix = np.vstack(
        [anchor["_feature"] for anchor in internal_anchors]
    )
    pca = _pca_analysis(feature_matrix, cfg)

    anchors = []
    coordinates = pca.get("state_coordinates", [])
    for index, anchor in enumerate(internal_anchors):
        anchors.append({
            key: value
            for key, value in anchor.items()
            if key != "_feature"
        })
        anchors[-1]["atlas_state"] = (
            coordinates[index] if index < len(coordinates) else []
        )

    consecutive = []
    for left, right in zip(internal_anchors, internal_anchors[1:]):
        distance = camera_distance(
            left["homography"],
            right["homography"],
        )
        if distance is None:
            continue
        frame_gap = int(right["frame"] - left["frame"])
        consecutive.append({
            "frame_a": int(left["frame"]),
            "frame_b": int(right["frame"]),
            "frame_gap": frame_gap,
            **distance,
            "median_px_per_frame": (
                float(distance["median_px"] / frame_gap)
                if frame_gap > 0
                else None
            ),
        })

    leave_one_out = []

    for index in range(1, len(internal_anchors) - 1):
        left = internal_anchors[index - 1]
        target = internal_anchors[index]
        right = internal_anchors[index + 1]

        left_frame = int(left["frame"])
        target_frame = int(target["frame"])
        right_frame = int(right["frame"])

        if not (left_frame < target_frame < right_frame):
            continue

        alpha = (
            (target_frame - left_frame)
            / (right_frame - left_frame)
        )

        predicted_h = interpolate_camera(
            left["homography"],
            right["homography"],
            alpha,
        )

        if predicted_h is None:
            leave_one_out.append({
                "frame": target_frame,
                "status": "prediction_failed",
                "left_frame": left_frame,
                "right_frame": right_frame,
            })
            continue

        error = camera_distance(
            predicted_h,
            target["homography"],
        )

        if error is None:
            leave_one_out.append({
                "frame": target_frame,
                "status": "comparison_failed",
                "left_frame": left_frame,
                "right_frame": right_frame,
            })
            continue

        predicted_visibility = compute_visible_pitch_footprint(
            predicted_h,
            width,
            height,
        )
        target_visibility = target["visibility"]

        predicted_fraction = predicted_visibility.get("visible_fraction")
        target_fraction = target_visibility.get("visible_fraction")

        visibility_fraction_error = None
        if (
            isinstance(predicted_fraction, (int, float))
            and isinstance(target_fraction, (int, float))
        ):
            visibility_fraction_error = abs(
                float(predicted_fraction)
                - float(target_fraction)
            )

        leave_one_out.append({
            "frame": target_frame,
            "status": "ok",
            "left_frame": left_frame,
            "right_frame": right_frame,
            "alpha": float(alpha),
            "predicted_homography": predicted_h,
            "error": error,
            "visible_fraction_absolute_error": visibility_fraction_error,
        })

    successful = [
        item
        for item in leave_one_out
        if item.get("status") == "ok"
    ]

    leave_one_out_summary = None
    if successful:
        median_errors = np.asarray(
            [item["error"]["median_px"] for item in successful]
        )
        p90_errors = np.asarray(
            [item["error"]["p90_px"] for item in successful]
        )

        leave_one_out_summary = {
            "tested_anchors": len(successful),
            "median_of_median_error_px": float(np.median(median_errors)),
            "p90_of_median_error_px": float(np.percentile(median_errors, 90)),
            "max_median_error_px": float(np.max(median_errors)),
            "median_p90_error_px": float(np.median(p90_errors)),
        }

    return {
        "status": "ok",
        "gold_policy": {
            "manual_only": True,
            "propagated_used_for_training": False,
            "refined_used_for_training": False,
            "explicit_manual_refine_is_gold": (
                not cfg.reject_explicit_manual_refine
            ),
        },
        "anchor_count": len(anchors),
        "anchors": anchors,
        "excluded": excluded,
        "analysis": {
            "pca": pca,
            "consecutive_camera_distance": consecutive,
            "leave_one_out": leave_one_out,
            "leave_one_out_summary": leave_one_out_summary,
        },
        "config": asdict(cfg),
    }


def summarize_camera_atlas(atlas: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(atlas, dict):
        return {"status": "invalid_atlas"}

    analysis = atlas.get("analysis", {})
    pca = analysis.get("pca", {})

    return {
        "status": atlas.get("status"),
        "strong_anchor_count": int(atlas.get("anchor_count", 0) or 0),
        "excluded_count": len(atlas.get("excluded", [])),
        "dimension_95": pca.get("dimension_95"),
        "dimension_99": pca.get("dimension_99"),
        "explained_variance_ratio": pca.get(
            "explained_variance_ratio",
            [],
        ),
        "leave_one_out": analysis.get("leave_one_out_summary"),
    }
