"""Ground-plane temporal registration for FIELD cameras.

This is NOT the discarded "optical-flow motion alpha" observer.

For two views of the same planar football pitch there is an image-to-image
homography G such that

    H_current ~= G_previous_to_current @ H_previous

where H maps pitch-world coordinates to image coordinates.

We estimate G from field-masked KLT tracks + full projective RANSAC homography,
re-initialising tracks every short step. Moving players are outliers. The current
pitch H is used only to decide which previous-frame pixels belong to the ground
plane.

No target-frame HUMAN geometry is used.
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
class PlanarFlowConfig:
    analysis_width: int = 720
    sample_step_frames: int = 2

    max_features: int = 900
    feature_quality_level: float = 0.006
    feature_min_distance_px: float = 6.0
    feature_block_size: int = 7

    lk_window_px: int = 25
    lk_max_level: int = 4
    forward_backward_max_error_px: float = 1.5

    ransac_reprojection_threshold_px: float = 2.5
    minimum_tracks: int = 24
    minimum_inliers: int = 18
    minimum_inlier_ratio: float = 0.35
    maximum_median_reprojection_px: float = 2.2

    minimum_inlier_span_x_fraction: float = 0.16
    minimum_inlier_span_y_fraction: float = 0.12
    minimum_inlier_hull_area_fraction: float = 0.012

    pitch_mask_world_margin_m: float = 1.5
    pitch_mask_erode_px: int = 5
    maximum_step_condition: float = 1e8


def _normalise_h(value: Any) -> np.ndarray | None:
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
        or condition > 1e14
        or not math.isfinite(determinant)
        or abs(determinant) < 1e-12
    ):
        return None
    return h


def _resize_frame(frame: np.ndarray, analysis_width: int):
    height, width = frame.shape[:2]
    if width <= analysis_width:
        return frame, 1.0

    scale = float(analysis_width) / float(width)
    resized = cv2.resize(
        frame,
        (
            int(round(width * scale)),
            int(round(height * scale)),
        ),
        interpolation=cv2.INTER_AREA,
    )
    return resized, scale


def _gray(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.createCLAHE(
        clipLimit=1.5,
        tileGridSize=(8, 8),
    ).apply(gray)


def _scale_h_world_to_image(h_native: np.ndarray, scale: float) -> np.ndarray:
    s = np.asarray(
        [
            [scale, 0.0, 0.0],
            [0.0, scale, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return s @ h_native


def _work_transform_to_native(g_work: np.ndarray, scale: float) -> np.ndarray:
    s = np.asarray(
        [
            [scale, 0.0, 0.0],
            [0.0, scale, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    s_inv = np.asarray(
        [
            [1.0 / scale, 0.0, 0.0],
            [0.0, 1.0 / scale, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return s_inv @ g_work @ s


def _pitch_mask_from_inverse_h(
    h_world_to_work: np.ndarray,
    image_width: int,
    image_height: int,
    pitch_length_m: float,
    pitch_width_m: float,
    cfg: PlanarFlowConfig,
) -> np.ndarray | None:
    """Classify work-image pixels by back-projecting them to pitch world.

    This remains valid even when the projective pole crosses the finite field,
    unlike projecting the four pitch corners and filling one polygon.
    """

    try:
        inverse = np.linalg.inv(h_world_to_work)
    except np.linalg.LinAlgError:
        return None

    yy, xx = np.indices(
        (image_height, image_width),
        dtype=np.float64,
    )
    homogeneous = np.stack(
        (
            xx.ravel(),
            yy.ravel(),
            np.ones(xx.size, dtype=np.float64),
        ),
        axis=1,
    )

    world_h = homogeneous @ inverse.T
    denominator = world_h[:, 2]
    valid = (
        np.isfinite(world_h).all(axis=1)
        & np.isfinite(denominator)
        & (np.abs(denominator) > 1e-9)
    )

    world_x = np.full(len(homogeneous), np.nan, dtype=np.float64)
    world_y = np.full(len(homogeneous), np.nan, dtype=np.float64)
    world_x[valid] = world_h[valid, 0] / denominator[valid]
    world_y[valid] = world_h[valid, 1] / denominator[valid]

    half_length = 0.5 * float(pitch_length_m) + cfg.pitch_mask_world_margin_m
    half_width = 0.5 * float(pitch_width_m) + cfg.pitch_mask_world_margin_m

    inside = (
        valid
        & (np.abs(world_x) <= half_length)
        & (np.abs(world_y) <= half_width)
    )

    mask = (
        inside.reshape(image_height, image_width).astype(np.uint8) * 255
    )

    if cfg.pitch_mask_erode_px > 0:
        radius = int(cfg.pitch_mask_erode_px)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * radius + 1, 2 * radius + 1),
        )
        mask = cv2.erode(mask, kernel)

    if int(np.count_nonzero(mask)) < cfg.minimum_tracks * 20:
        return None

    return mask


def _forward_backward_tracks(
    previous_gray: np.ndarray,
    current_gray: np.ndarray,
    points: np.ndarray,
    cfg: PlanarFlowConfig,
):
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

    return (
        source[valid],
        target[valid],
        fb_error[valid],
    )


def _spatially_distributed(
    points: np.ndarray,
    image_width: int,
    image_height: int,
    cfg: PlanarFlowConfig,
) -> tuple[bool, dict[str, float]]:
    if len(points) < 3:
        return False, {}

    span_x = float(np.ptp(points[:, 0])) / max(float(image_width), 1.0)
    span_y = float(np.ptp(points[:, 1])) / max(float(image_height), 1.0)

    hull = cv2.convexHull(points.astype(np.float32))
    hull_area = float(cv2.contourArea(hull))
    hull_fraction = hull_area / max(float(image_width * image_height), 1.0)

    ok = (
        span_x >= cfg.minimum_inlier_span_x_fraction
        and span_y >= cfg.minimum_inlier_span_y_fraction
        and hull_fraction >= cfg.minimum_inlier_hull_area_fraction
    )

    return ok, {
        "span_x_fraction": span_x,
        "span_y_fraction": span_y,
        "hull_area_fraction": hull_fraction,
    }


def estimate_ground_plane_step(
    previous_frame: np.ndarray,
    current_frame: np.ndarray,
    previous_pitch_homography: Any,
    *,
    pitch_length_m: float,
    pitch_width_m: float,
    config: PlanarFlowConfig | None = None,
) -> dict[str, Any]:
    """Estimate image homography previous -> current on the pitch plane."""

    cfg = config or PlanarFlowConfig()
    h_native = _normalise_h(previous_pitch_homography)
    if h_native is None:
        return {"status": "invalid_previous_pitch_h"}

    previous_work, scale_previous = _resize_frame(
        previous_frame,
        cfg.analysis_width,
    )
    current_work, scale_current = _resize_frame(
        current_frame,
        cfg.analysis_width,
    )

    if abs(scale_previous - scale_current) > 1e-9:
        return {"status": "inconsistent_resize_scale"}

    scale = scale_previous
    previous_gray = _gray(previous_work)
    current_gray = _gray(current_work)
    height, width = previous_gray.shape

    h_work = _scale_h_world_to_image(h_native, scale)
    mask = _pitch_mask_from_inverse_h(
        h_work,
        width,
        height,
        pitch_length_m,
        pitch_width_m,
        cfg,
    )
    if mask is None:
        return {"status": "pitch_mask_failed"}

    points = cv2.goodFeaturesToTrack(
        previous_gray,
        mask=mask,
        maxCorners=cfg.max_features,
        qualityLevel=cfg.feature_quality_level,
        minDistance=cfg.feature_min_distance_px,
        blockSize=cfg.feature_block_size,
        useHarrisDetector=False,
    )

    if points is None or len(points) < cfg.minimum_tracks:
        return {
            "status": "insufficient_features",
            "feature_count": 0 if points is None else int(len(points)),
        }

    tracks = _forward_backward_tracks(
        previous_gray,
        current_gray,
        points,
        cfg,
    )
    if tracks is None:
        return {"status": "flow_failed"}

    source, target, fb_error = tracks
    if len(source) < cfg.minimum_tracks:
        return {
            "status": "insufficient_tracks",
            "track_count": int(len(source)),
        }

    matrix, inlier_mask = cv2.findHomography(
        source,
        target,
        method=cv2.RANSAC,
        ransacReprojThreshold=cfg.ransac_reprojection_threshold_px,
        maxIters=3000,
        confidence=0.995,
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

    distribution_ok, distribution = _spatially_distributed(
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

    source_h = np.column_stack(
        (
            inlier_source,
            np.ones(inlier_count, dtype=np.float64),
        )
    )
    projected_h = source_h @ matrix.T
    valid = np.abs(projected_h[:, 2]) > 1e-9
    projected = np.full_like(inlier_target, np.nan, dtype=np.float64)
    projected[valid] = projected_h[valid, :2] / projected_h[valid, 2, None]

    reprojection = np.linalg.norm(
        projected[valid] - inlier_target[valid],
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

    try:
        step_condition = float(np.linalg.cond(matrix))
    except np.linalg.LinAlgError:
        step_condition = float("inf")

    if (
        not math.isfinite(step_condition)
        or step_condition > cfg.maximum_step_condition
    ):
        return {
            "status": "ill_conditioned_step",
            "condition": step_condition,
        }

    native_matrix = _work_transform_to_native(
        matrix,
        scale,
    )
    native_matrix = _normalise_h(native_matrix)
    if native_matrix is None:
        return {"status": "native_step_conversion_failed"}

    return {
        "status": "ok",
        "matrix": native_matrix,
        "track_count": int(len(source)),
        "inlier_count": inlier_count,
        "inlier_ratio": float(inlier_ratio),
        "median_forward_backward_error_px": (
            float(np.median(fb_error))
            if len(fb_error)
            else None
        ),
        "median_reprojection_px": median_reprojection,
        "analysis_scale": float(scale),
        **distribution,
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


def _read_frame(capture: cv2.VideoCapture, frame_index: int):
    capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
    ok, frame = capture.read()
    return frame if ok and frame is not None else None


def propagate_pitch_homography(
    video_path: str | Path,
    start_frame: int,
    end_frame: int,
    start_homography: Any,
    *,
    pitch_length_m: float,
    pitch_width_m: float,
    config: PlanarFlowConfig | None = None,
) -> dict[str, Any]:
    """Sequentially compose ground-plane image homographies."""

    cfg = config or PlanarFlowConfig()
    h = _normalise_h(start_homography)
    if h is None:
        return {"status": "invalid_start_homography"}

    path = Path(video_path)
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return {"status": "video_open_failed", "video": str(path)}

    try:
        frames = _frame_sequence(
            start_frame,
            end_frame,
            cfg.sample_step_frames,
        )

        previous_frame = _read_frame(
            capture,
            frames[0],
        )
        if previous_frame is None:
            return {
                "status": "frame_read_failed",
                "frame": frames[0],
            }

        steps = []

        for current_index in frames[1:]:
            current_frame = _read_frame(
                capture,
                current_index,
            )
            if current_frame is None:
                return {
                    "status": "frame_read_failed",
                    "frame": current_index,
                    "steps": steps,
                }

            step = estimate_ground_plane_step(
                previous_frame,
                current_frame,
                h,
                pitch_length_m=pitch_length_m,
                pitch_width_m=pitch_width_m,
                config=cfg,
            )

            step_record = {
                "from_frame": int(frames[len(steps)]),
                "to_frame": int(current_index),
                **{
                    key: value
                    for key, value in step.items()
                    if key != "matrix"
                },
            }
            steps.append(step_record)

            if step.get("status") != "ok":
                return {
                    "status": "path_failed",
                    "failed_from_frame": step_record["from_frame"],
                    "failed_to_frame": step_record["to_frame"],
                    "failure": step,
                    "steps": steps,
                    "homography": h.tolist(),
                    "config": asdict(cfg),
                }

            g = np.asarray(
                step["matrix"],
                dtype=np.float64,
            )

            h = _normalise_h(g @ h)
            if h is None:
                return {
                    "status": "composition_failed",
                    "steps": steps,
                    "config": asdict(cfg),
                }

            previous_frame = current_frame

        inlier_ratios = [
            float(step["inlier_ratio"])
            for step in steps
            if step.get("status") == "ok"
            and step.get("inlier_ratio") is not None
        ]
        reprojections = [
            float(step["median_reprojection_px"])
            for step in steps
            if step.get("status") == "ok"
            and step.get("median_reprojection_px") is not None
        ]
        tracks = [
            int(step["inlier_count"])
            for step in steps
            if step.get("status") == "ok"
            and step.get("inlier_count") is not None
        ]

        return {
            "status": "ok",
            "start_frame": int(start_frame),
            "end_frame": int(end_frame),
            "frame_span": abs(int(end_frame) - int(start_frame)),
            "step_count": len(steps),
            "homography": h.tolist(),
            "median_inlier_ratio": (
                float(np.median(inlier_ratios))
                if inlier_ratios
                else None
            ),
            "minimum_inlier_ratio": (
                float(np.min(inlier_ratios))
                if inlier_ratios
                else None
            ),
            "median_reprojection_px": (
                float(np.median(reprojections))
                if reprojections
                else None
            ),
            "median_inlier_count": (
                float(np.median(tracks))
                if tracks
                else None
            ),
            "steps": steps,
            "config": asdict(cfg),
        }

    finally:
        capture.release()


def homography_disagreement_px(
    first_h: Any,
    second_h: Any,
    *,
    pitch_length_m: float,
    pitch_width_m: float,
    image_width: int,
    image_height: int,
    x_samples: int = 13,
    y_samples: int = 9,
    margin_px: float = 16.0,
) -> dict[str, Any]:
    """Compare two H candidates without any target-frame HUMAN geometry."""

    first = _normalise_h(first_h)
    second = _normalise_h(second_h)
    if first is None or second is None:
        return {"status": "invalid_homography"}

    xs = np.linspace(
        -0.5 * pitch_length_m,
        0.5 * pitch_length_m,
        x_samples,
    )
    ys = np.linspace(
        -0.5 * pitch_width_m,
        0.5 * pitch_width_m,
        y_samples,
    )
    world = np.asarray(
        [(x, y) for y in ys for x in xs],
        dtype=np.float64,
    )
    homogeneous = np.column_stack(
        (
            world,
            np.ones(len(world), dtype=np.float64),
        )
    )

    def project(h):
        q = homogeneous @ h.T
        d = q[:, 2]
        valid = (
            np.isfinite(q).all(axis=1)
            & np.isfinite(d)
            & (np.abs(d) > 1e-9)
        )
        image = np.full((len(world), 2), np.nan, dtype=np.float64)
        image[valid] = q[valid, :2] / d[valid, None]
        valid &= np.isfinite(image).all(axis=1)
        return image, valid

    a, va = project(first)
    b, vb = project(second)

    inside_a = (
        va
        & (a[:, 0] >= margin_px)
        & (a[:, 0] <= image_width - margin_px)
        & (a[:, 1] >= margin_px)
        & (a[:, 1] <= image_height - margin_px)
    )
    inside_b = (
        vb
        & (b[:, 0] >= margin_px)
        & (b[:, 0] <= image_width - margin_px)
        & (b[:, 1] >= margin_px)
        & (b[:, 1] <= image_height - margin_px)
    )

    common = inside_a & inside_b
    if int(np.count_nonzero(common)) < 4:
        common = va & vb

    if int(np.count_nonzero(common)) < 4:
        return {"status": "not_enough_common_points"}

    errors = np.linalg.norm(
        a[common] - b[common],
        axis=1,
    )

    return {
        "status": "ok",
        "count": int(len(errors)),
        "median_px": float(np.median(errors)),
        "p90_px": float(np.percentile(errors, 90)),
        "max_px": float(np.max(errors)),
    }
