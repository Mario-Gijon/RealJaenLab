"""Global camera-motion estimation independent of pitch calibration.

Purpose
-------
Estimate the image-to-image motion of a fixed-position football camera without
using pitch lines, pitch geometry, a HUMAN homography, or a field mask.

For a camera whose optical centre is approximately fixed, pan/tilt/roll and
focal-length changes induce a projective image mapping that can be estimated
from static scene structure. Moving players are treated as RANSAC outliers.

This module deliberately does NOT decide where the football pitch is. It only
answers:

    "How did the camera view change from frame A to frame B?"

The result can later be combined with HUMAN anchors / Camera Atlas states.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np


EPS = 1e-9


@dataclass(frozen=True)
class GlobalCameraMotionConfig:
    """Robust defaults for Veo-like fixed-position football video."""

    analysis_width: int = 960
    sample_step_frames: int = 2

    grid_rows: int = 6
    grid_cols: int = 10
    max_features_per_cell: int = 45
    feature_quality_level: float = 0.008
    feature_min_distance_px: float = 6.0
    feature_block_size: int = 7
    border_margin_fraction: float = 0.025

    lk_window_px: int = 27
    lk_max_level: int = 4
    forward_backward_max_error_px: float = 1.8

    ransac_reprojection_threshold_px: float = 2.8
    ransac_max_iters: int = 4000
    ransac_confidence: float = 0.997

    minimum_tracks: int = 45
    minimum_inliers: int = 30
    minimum_inlier_ratio: float = 0.42

    minimum_inlier_span_x_fraction: float = 0.28
    minimum_inlier_span_y_fraction: float = 0.20
    minimum_inlier_hull_area_fraction: float = 0.035
    minimum_occupied_grid_cells: int = 7

    maximum_median_reprojection_px: float = 2.4
    maximum_median_symmetric_error_px: float = 3.0
    maximum_step_condition: float = 2e8
    maximum_projective_term: float = 0.35


def _normalise_h(value: Any, maximum_condition: float = 1e14) -> np.ndarray | None:
    try:
        h = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if h.shape != (3, 3) or not np.isfinite(h).all():
        return None
    if abs(float(h[2, 2])) < EPS:
        return None
    h = h / float(h[2, 2])
    try:
        condition = float(np.linalg.cond(h))
        determinant = float(np.linalg.det(h))
    except np.linalg.LinAlgError:
        return None
    if (
        not math.isfinite(condition)
        or condition > maximum_condition
        or not math.isfinite(determinant)
        or abs(determinant) < 1e-12
    ):
        return None
    return h


def _resize_frame(frame: np.ndarray, analysis_width: int) -> tuple[np.ndarray, float]:
    height, width = frame.shape[:2]
    if width <= analysis_width:
        return frame, 1.0
    scale = float(analysis_width) / float(width)
    resized = cv2.resize(
        frame,
        (int(round(width * scale)), int(round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )
    return resized, scale


def _gray(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.createCLAHE(
        clipLimit=1.5,
        tileGridSize=(8, 8),
    ).apply(gray)


def _grid_features(gray: np.ndarray, cfg: GlobalCameraMotionConfig) -> np.ndarray | None:
    """Detect features across the complete image instead of one dominant region."""

    height, width = gray.shape[:2]
    margin_x = int(round(width * cfg.border_margin_fraction))
    margin_y = int(round(height * cfg.border_margin_fraction))

    x0, x1 = margin_x, max(margin_x + 1, width - margin_x)
    y0, y1 = margin_y, max(margin_y + 1, height - margin_y)

    points: list[np.ndarray] = []
    for row in range(cfg.grid_rows):
        top = y0 + (y1 - y0) * row // cfg.grid_rows
        bottom = y0 + (y1 - y0) * (row + 1) // cfg.grid_rows
        for col in range(cfg.grid_cols):
            left = x0 + (x1 - x0) * col // cfg.grid_cols
            right = x0 + (x1 - x0) * (col + 1) // cfg.grid_cols

            if bottom - top < 8 or right - left < 8:
                continue

            roi = gray[top:bottom, left:right]
            local = cv2.goodFeaturesToTrack(
                roi,
                mask=None,
                maxCorners=cfg.max_features_per_cell,
                qualityLevel=cfg.feature_quality_level,
                minDistance=cfg.feature_min_distance_px,
                blockSize=cfg.feature_block_size,
                useHarrisDetector=False,
            )
            if local is None:
                continue
            local = local.reshape(-1, 2)
            local[:, 0] += float(left)
            local[:, 1] += float(top)
            points.append(local)

    if not points:
        return None

    merged = np.concatenate(points, axis=0).astype(np.float32)
    return merged.reshape(-1, 1, 2)


def _forward_backward_tracks(
    previous_gray: np.ndarray,
    current_gray: np.ndarray,
    points: np.ndarray,
    cfg: GlobalCameraMotionConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    next_points, status_forward, _ = cv2.calcOpticalFlowPyrLK(
        previous_gray,
        current_gray,
        points,
        None,
        winSize=(cfg.lk_window_px, cfg.lk_window_px),
        maxLevel=cfg.lk_max_level,
        criteria=(
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            30,
            0.01,
        ),
    )
    if next_points is None or status_forward is None:
        return None

    back_points, status_back, _ = cv2.calcOpticalFlowPyrLK(
        current_gray,
        previous_gray,
        next_points,
        None,
        winSize=(cfg.lk_window_px, cfg.lk_window_px),
        maxLevel=cfg.lk_max_level,
        criteria=(
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            30,
            0.01,
        ),
    )
    if back_points is None or status_back is None:
        return None

    source = points.reshape(-1, 2)
    target = next_points.reshape(-1, 2)
    backward = back_points.reshape(-1, 2)

    valid = (
        (status_forward.ravel() > 0)
        & (status_back.ravel() > 0)
        & np.isfinite(source).all(axis=1)
        & np.isfinite(target).all(axis=1)
        & np.isfinite(backward).all(axis=1)
    )

    fb_error = np.linalg.norm(backward - source, axis=1)
    valid &= fb_error <= cfg.forward_backward_max_error_px

    return source[valid], target[valid], fb_error[valid]


def _distribution_metrics(
    points: np.ndarray,
    image_width: int,
    image_height: int,
    cfg: GlobalCameraMotionConfig,
) -> tuple[bool, dict[str, float | int]]:
    if len(points) < 3:
        return False, {}

    span_x = float(np.ptp(points[:, 0])) / max(float(image_width), 1.0)
    span_y = float(np.ptp(points[:, 1])) / max(float(image_height), 1.0)

    hull = cv2.convexHull(points.astype(np.float32))
    hull_area = float(cv2.contourArea(hull))
    hull_fraction = hull_area / max(float(image_width * image_height), 1.0)

    rows = np.clip(
        (points[:, 1] / max(float(image_height), 1.0) * cfg.grid_rows).astype(int),
        0,
        cfg.grid_rows - 1,
    )
    cols = np.clip(
        (points[:, 0] / max(float(image_width), 1.0) * cfg.grid_cols).astype(int),
        0,
        cfg.grid_cols - 1,
    )
    occupied = int(len(set(zip(rows.tolist(), cols.tolist()))))

    ok = (
        span_x >= cfg.minimum_inlier_span_x_fraction
        and span_y >= cfg.minimum_inlier_span_y_fraction
        and hull_fraction >= cfg.minimum_inlier_hull_area_fraction
        and occupied >= cfg.minimum_occupied_grid_cells
    )
    return ok, {
        "span_x_fraction": span_x,
        "span_y_fraction": span_y,
        "hull_area_fraction": hull_fraction,
        "occupied_grid_cells": occupied,
    }


def _project_points(points: np.ndarray, h: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    q = np.column_stack((pts, np.ones(len(pts), dtype=np.float64))) @ h.T
    denominator = q[:, 2]
    valid = (
        np.isfinite(q).all(axis=1)
        & np.isfinite(denominator)
        & (np.abs(denominator) > EPS)
    )
    out = np.full((len(pts), 2), np.nan, dtype=np.float64)
    out[valid] = q[valid, :2] / denominator[valid, None]
    valid &= np.isfinite(out).all(axis=1)
    return out, valid


def _symmetric_errors(
    source: np.ndarray,
    target: np.ndarray,
    h: np.ndarray,
) -> np.ndarray:
    projected_forward, valid_forward = _project_points(source, h)
    try:
        inverse = np.linalg.inv(h)
    except np.linalg.LinAlgError:
        return np.asarray([], dtype=np.float64)
    projected_backward, valid_backward = _project_points(target, inverse)

    valid = valid_forward & valid_backward
    if not valid.any():
        return np.asarray([], dtype=np.float64)

    forward = np.linalg.norm(projected_forward[valid] - target[valid], axis=1)
    backward = np.linalg.norm(projected_backward[valid] - source[valid], axis=1)
    return 0.5 * (forward + backward)


def _work_transform_to_native(g_work: np.ndarray, scale: float) -> np.ndarray | None:
    if scale <= 0:
        return None
    s = np.asarray(
        [[scale, 0.0, 0.0], [0.0, scale, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    s_inv = np.asarray(
        [[1.0 / scale, 0.0, 0.0], [0.0, 1.0 / scale, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return _normalise_h(s_inv @ g_work @ s)


def _normalised_image_transform(
    h_native: np.ndarray,
    image_width: int,
    image_height: int,
) -> np.ndarray | None:
    """Express pixel homography in centred width/height-normalised coordinates."""

    width = max(float(image_width), 1.0)
    height = max(float(image_height), 1.0)
    a = np.asarray(
        [
            [width, 0.0, width / 2.0],
            [0.0, height, height / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    try:
        normalized = np.linalg.inv(a) @ h_native @ a
    except np.linalg.LinAlgError:
        return None
    return _normalise_h(normalized)


def camera_motion_descriptor(
    h_native: Any,
    *,
    image_width: int,
    image_height: int,
) -> dict[str, Any] | None:
    """Return exact projective coordinates plus interpretable local observables.

    The observables are intentionally NOT called physical pan/tilt/zoom. They
    are stable image-space proxies that can later form a Camera Atlas.
    """

    h = _normalise_h(h_native)
    if h is None:
        return None
    g = _normalised_image_transform(h, image_width, image_height)
    if g is None:
        return None

    a, b, c = g[0]
    d, e, f = g[1]
    p, q, r = g[2]
    if abs(float(r)) < EPS:
        return None

    g = g / float(r)
    a, b, c = g[0]
    d, e, f = g[1]
    p, q, _ = g[2]

    jacobian = np.asarray(
        [
            [a - c * p, b - c * q],
            [d - f * p, e - f * q],
        ],
        dtype=np.float64,
    )
    if not np.isfinite(jacobian).all():
        return None

    try:
        u, singular_values, vt = np.linalg.svd(jacobian)
    except np.linalg.LinAlgError:
        return None

    if len(singular_values) != 2 or np.min(singular_values) <= EPS:
        return None

    rotation_matrix = u @ vt
    if np.linalg.det(rotation_matrix) < 0:
        u[:, -1] *= -1.0
        rotation_matrix = u @ vt

    rotation_rad = math.atan2(
        float(rotation_matrix[1, 0]),
        float(rotation_matrix[0, 0]),
    )
    log_scale = 0.5 * float(np.log(singular_values[0] * singular_values[1]))
    anisotropy_log = float(np.log(singular_values[0] / singular_values[1]))

    exact_vector = np.asarray(
        [
            g[0, 0],
            g[0, 1],
            g[0, 2],
            g[1, 0],
            g[1, 1],
            g[1, 2],
            g[2, 0],
            g[2, 1],
        ],
        dtype=np.float64,
    )

    return {
        "projective_vector": exact_vector.tolist(),
        "observables": {
            "center_dx_fraction": float(c),
            "center_dy_fraction": float(f),
            "log_area_scale": log_scale,
            "rotation_rad": float(rotation_rad),
            "rotation_deg": float(math.degrees(rotation_rad)),
            "anisotropy_log": anisotropy_log,
            "projective_x": float(p),
            "projective_y": float(q),
        },
        "normalized_homography": g.tolist(),
    }


def estimate_global_camera_step(
    previous_frame: np.ndarray,
    current_frame: np.ndarray,
    *,
    config: GlobalCameraMotionConfig | None = None,
) -> dict[str, Any]:
    """Estimate the global static-scene homography previous -> current."""

    cfg = config or GlobalCameraMotionConfig()

    if (
        not isinstance(previous_frame, np.ndarray)
        or not isinstance(current_frame, np.ndarray)
        or previous_frame.ndim != 3
        or current_frame.ndim != 3
    ):
        return {"status": "invalid_frame"}

    previous_work, scale_previous = _resize_frame(previous_frame, cfg.analysis_width)
    current_work, scale_current = _resize_frame(current_frame, cfg.analysis_width)

    if previous_work.shape[:2] != current_work.shape[:2]:
        return {"status": "inconsistent_frame_shape"}
    if abs(scale_previous - scale_current) > 1e-9:
        return {"status": "inconsistent_resize_scale"}

    scale = scale_previous
    previous_gray = _gray(previous_work)
    current_gray = _gray(current_work)
    height, width = previous_gray.shape

    points = _grid_features(previous_gray, cfg)
    if points is None or len(points) < cfg.minimum_tracks:
        return {
            "status": "insufficient_features",
            "feature_count": 0 if points is None else int(len(points)),
        }

    tracks = _forward_backward_tracks(previous_gray, current_gray, points, cfg)
    if tracks is None:
        return {"status": "flow_failed"}

    source, target, fb_error = tracks
    if len(source) < cfg.minimum_tracks:
        return {
            "status": "insufficient_tracks",
            "feature_count": int(len(points)),
            "track_count": int(len(source)),
        }

    matrix, inlier_mask = cv2.findHomography(
        source,
        target,
        method=cv2.RANSAC,
        ransacReprojThreshold=cfg.ransac_reprojection_threshold_px,
        maxIters=cfg.ransac_max_iters,
        confidence=cfg.ransac_confidence,
    )
    if matrix is None or inlier_mask is None:
        return {
            "status": "homography_ransac_failed",
            "track_count": int(len(source)),
        }

    matrix = _normalise_h(matrix)
    if matrix is None:
        return {"status": "invalid_step_homography"}

    inliers = inlier_mask.ravel() > 0
    inlier_source = source[inliers]
    inlier_target = target[inliers]
    inlier_count = int(np.count_nonzero(inliers))
    inlier_ratio = inlier_count / max(float(len(source)), 1.0)

    if inlier_count < cfg.minimum_inliers:
        return {
            "status": "insufficient_inliers",
            "track_count": int(len(source)),
            "inlier_count": inlier_count,
            "inlier_ratio": float(inlier_ratio),
        }
    if inlier_ratio < cfg.minimum_inlier_ratio:
        return {
            "status": "low_inlier_ratio",
            "track_count": int(len(source)),
            "inlier_count": inlier_count,
            "inlier_ratio": float(inlier_ratio),
        }

    distribution_ok, distribution = _distribution_metrics(
        inlier_source,
        width,
        height,
        cfg,
    )
    if not distribution_ok:
        return {
            "status": "poor_inlier_distribution",
            "track_count": int(len(source)),
            "inlier_count": inlier_count,
            "inlier_ratio": float(inlier_ratio),
            **distribution,
        }

    projected, valid_projection = _project_points(inlier_source, matrix)
    reprojection = np.linalg.norm(
        projected[valid_projection] - inlier_target[valid_projection],
        axis=1,
    )
    median_reprojection = (
        float(np.median(reprojection))
        if len(reprojection)
        else float("inf")
    )
    if (
        not math.isfinite(median_reprojection)
        or median_reprojection > cfg.maximum_median_reprojection_px
    ):
        return {
            "status": "high_reprojection_error",
            "median_reprojection_px": median_reprojection,
            "inlier_count": inlier_count,
            "inlier_ratio": float(inlier_ratio),
        }

    symmetric = _symmetric_errors(inlier_source, inlier_target, matrix)
    median_symmetric = (
        float(np.median(symmetric))
        if len(symmetric)
        else float("inf")
    )
    if (
        not math.isfinite(median_symmetric)
        or median_symmetric > cfg.maximum_median_symmetric_error_px
    ):
        return {
            "status": "high_symmetric_error",
            "median_symmetric_error_px": median_symmetric,
            "inlier_count": inlier_count,
            "inlier_ratio": float(inlier_ratio),
        }

    try:
        condition = float(np.linalg.cond(matrix))
    except np.linalg.LinAlgError:
        condition = float("inf")
    if (
        not math.isfinite(condition)
        or condition > cfg.maximum_step_condition
    ):
        return {
            "status": "ill_conditioned_step",
            "condition": condition,
        }

    native_matrix = _work_transform_to_native(matrix, scale)
    if native_matrix is None:
        return {"status": "native_step_conversion_failed"}

    descriptor = camera_motion_descriptor(
        native_matrix,
        image_width=previous_frame.shape[1],
        image_height=previous_frame.shape[0],
    )
    if descriptor is None:
        return {"status": "descriptor_failed"}

    observables = descriptor["observables"]
    if (
        abs(float(observables["projective_x"])) > cfg.maximum_projective_term
        or abs(float(observables["projective_y"])) > cfg.maximum_projective_term
    ):
        return {
            "status": "implausible_projective_step",
            "projective_x": observables["projective_x"],
            "projective_y": observables["projective_y"],
        }

    return {
        "status": "ok",
        "matrix": native_matrix,
        "feature_count": int(len(points)),
        "track_count": int(len(source)),
        "inlier_count": inlier_count,
        "inlier_ratio": float(inlier_ratio),
        "median_forward_backward_error_px": (
            float(np.median(fb_error))
            if len(fb_error)
            else None
        ),
        "median_reprojection_px": median_reprojection,
        "median_symmetric_error_px": median_symmetric,
        "condition": condition,
        "analysis_scale": float(scale),
        **distribution,
        "descriptor": descriptor,
    }


def _frame_sequence(start_frame: int, end_frame: int, step: int) -> list[int]:
    start = int(start_frame)
    end = int(end_frame)
    if start == end:
        return [start]

    direction = 1 if end > start else -1
    stride = max(1, int(step)) * direction
    frames = list(range(start, end, stride))
    if not frames or frames[0] != start:
        frames.insert(0, start)
    if frames[-1] != end:
        frames.append(end)
    return frames


def _read_frame(capture: cv2.VideoCapture, frame_index: int) -> np.ndarray | None:
    capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
    ok, frame = capture.read()
    return frame if ok and frame is not None else None


def _aggregate_path(
    *,
    start_frame: int,
    end_frame: int,
    cumulative_transform: np.ndarray,
    steps: list[dict[str, Any]],
    cfg: GlobalCameraMotionConfig,
) -> dict[str, Any]:
    inlier_ratios = [
        float(step["inlier_ratio"])
        for step in steps
        if step.get("status") == "ok" and step.get("inlier_ratio") is not None
    ]
    reprojections = [
        float(step["median_reprojection_px"])
        for step in steps
        if step.get("status") == "ok" and step.get("median_reprojection_px") is not None
    ]
    symmetric = [
        float(step["median_symmetric_error_px"])
        for step in steps
        if step.get("status") == "ok" and step.get("median_symmetric_error_px") is not None
    ]
    tracks = [
        int(step["inlier_count"])
        for step in steps
        if step.get("status") == "ok" and step.get("inlier_count") is not None
    ]

    return {
        "status": "ok",
        "start_frame": int(start_frame),
        "end_frame": int(end_frame),
        "frame_span": abs(int(end_frame) - int(start_frame)),
        "step_count": len(steps),
        "cumulative_transform": cumulative_transform.tolist(),
        "median_inlier_ratio": float(np.median(inlier_ratios)) if inlier_ratios else None,
        "minimum_inlier_ratio": float(np.min(inlier_ratios)) if inlier_ratios else None,
        "median_reprojection_px": float(np.median(reprojections)) if reprojections else None,
        "median_symmetric_error_px": float(np.median(symmetric)) if symmetric else None,
        "median_inlier_count": float(np.median(tracks)) if tracks else None,
        "steps": steps,
        "config": asdict(cfg),
    }


def analyse_video_camera_motion(
    video_path: str | Path,
    start_frame: int,
    end_frame: int,
    *,
    config: GlobalCameraMotionConfig | None = None,
) -> dict[str, Any]:
    """Analyse a video interval without any pitch calibration."""

    cfg = config or GlobalCameraMotionConfig()
    path = Path(video_path)
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return {"status": "video_open_failed", "video": str(path)}

    frames = _frame_sequence(start_frame, end_frame, cfg.sample_step_frames)
    steps: list[dict[str, Any]] = []
    cumulative = np.eye(3, dtype=np.float64)

    try:
        previous_index = frames[0]
        previous_frame = _read_frame(capture, previous_index)
        if previous_frame is None:
            return {"status": "frame_read_failed", "frame": previous_index}

        for current_index in frames[1:]:
            current_frame = _read_frame(capture, current_index)
            if current_frame is None:
                return {
                    "status": "frame_read_failed",
                    "frame": current_index,
                    "steps": steps,
                }

            result = estimate_global_camera_step(
                previous_frame,
                current_frame,
                config=cfg,
            )
            step_record = {
                "from_frame": int(previous_index),
                "to_frame": int(current_index),
                **{key: value for key, value in result.items() if key != "matrix"},
            }
            if result.get("status") != "ok":
                return {
                    "status": "path_failed",
                    "failed_from_frame": int(previous_index),
                    "failed_to_frame": int(current_index),
                    "failure": result,
                    "steps": [*steps, step_record],
                    "cumulative_transform": cumulative.tolist(),
                    "config": asdict(cfg),
                }

            step_matrix = _normalise_h(result["matrix"])
            cumulative = _normalise_h(step_matrix @ cumulative)
            if cumulative is None:
                return {
                    "status": "composition_failed",
                    "failed_from_frame": int(previous_index),
                    "failed_to_frame": int(current_index),
                    "steps": [*steps, step_record],
                    "config": asdict(cfg),
                }

            steps.append(step_record)
            previous_index = current_index
            previous_frame = current_frame

        return _aggregate_path(
            start_frame=start_frame,
            end_frame=end_frame,
            cumulative_transform=cumulative,
            steps=steps,
            cfg=cfg,
        )
    finally:
        capture.release()


def propagate_homography_with_global_motion(
    video_path: str | Path,
    start_frame: int,
    end_frame: int,
    start_homography: Any,
    *,
    config: GlobalCameraMotionConfig | None = None,
) -> dict[str, Any]:
    """Propagate an H using image-only camera motion.

    `start_homography` is used only after the image-motion path has been
    estimated. It does not influence feature detection or motion estimation.
    """

    start_h = _normalise_h(start_homography)
    if start_h is None:
        return {"status": "invalid_start_homography"}

    path = analyse_video_camera_motion(
        video_path,
        start_frame,
        end_frame,
        config=config,
    )
    if path.get("status") != "ok":
        return path

    cumulative = _normalise_h(path.get("cumulative_transform"))
    if cumulative is None:
        return {**path, "status": "invalid_cumulative_transform"}

    propagated = _normalise_h(cumulative @ start_h)
    if propagated is None:
        return {**path, "status": "propagated_homography_failed"}

    return {
        **path,
        "homography": propagated.tolist(),
        "source": "global_camera_motion_v1",
    }
