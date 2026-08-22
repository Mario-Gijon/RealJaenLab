"""Temporal FIELD propagation for RealJaen Supervisor.

Propagates a manually calibrated world->image pitch homography across nearby
video frames using pitch-masked KLT optical flow and a robust frame-to-frame
image homography.

This module intentionally contains no FastAPI/UI code. The Supervisor backend
can call `propagate_field_homography()` and store the returned per-frame
homographies as propagated FIELD state.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np


PITCH_REFERENCE_POINTS = np.array(
    [
        [-52.5, -34.0],
        [52.5, -34.0],
        [52.5, 34.0],
        [-52.5, 34.0],
        [0.0, 0.0],
        [0.0, -34.0],
        [0.0, 34.0],
        [-36.0, -20.16],
        [36.0, 20.16],
    ],
    dtype=np.float64,
)


@dataclass(frozen=True)
class PropagationConfig:
    """Tunable thresholds expressed in native-video pixels where applicable."""

    working_scale: float = 0.5
    max_corners: int = 900
    quality_level: float = 0.01
    min_distance: float = 7.0
    block_size: int = 7

    lk_window: int = 23
    lk_max_level: int = 3
    fb_max_error_px: float = 3.0

    ransac_reproj_threshold_px: float = 3.5
    min_tracks: int = 20
    min_inliers: int = 12
    min_inlier_ratio: float = 0.35
    max_median_reprojection_error_px: float = 4.5

    max_median_frame_motion_px: float = 120.0
    max_reference_motion_px: float = 320.0
    min_pitch_area_ratio: float = 0.45
    max_pitch_area_ratio: float = 2.2

    min_confidence: float = 0.30


def _normalise_homography(matrix: np.ndarray) -> np.ndarray | None:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        return None
    scale = matrix[2, 2]
    if abs(scale) < 1e-12:
        norm = np.linalg.norm(matrix)
        if not np.isfinite(norm) or norm < 1e-12:
            return None
        matrix = matrix / norm
    else:
        matrix = matrix / scale
    if not np.isfinite(matrix).all():
        return None
    try:
        if abs(np.linalg.det(matrix)) < 1e-12:
            return None
    except np.linalg.LinAlgError:
        return None
    return matrix


def _project_points(points_xy: np.ndarray, homography: np.ndarray) -> np.ndarray | None:
    points_xy = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    homogeneous = np.column_stack((points_xy, np.ones(len(points_xy), dtype=np.float64)))
    projected_h = homogeneous @ homography.T
    denominators = projected_h[:, 2]
    if np.any(~np.isfinite(projected_h)) or np.any(np.abs(denominators) < 1e-10):
        return None
    projected = projected_h[:, :2] / denominators[:, None]
    return projected if np.isfinite(projected).all() else None


def _pitch_polygon(homography: np.ndarray) -> np.ndarray | None:
    corners = PITCH_REFERENCE_POINTS[:4]
    projected = _project_points(corners, homography)
    if projected is None:
        return None
    return projected


def _polygon_area(points: np.ndarray) -> float:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    x = points[:, 0]
    y = points[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


def _scaled_homography(homography_native: np.ndarray, scale: float) -> np.ndarray:
    scaling = np.array([[scale, 0.0, 0.0], [0.0, scale, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return scaling @ homography_native


def _motion_to_native(motion_scaled: np.ndarray, scale: float) -> np.ndarray:
    scaling = np.array([[scale, 0.0, 0.0], [0.0, scale, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    inverse_scaling = np.array(
        [[1.0 / scale, 0.0, 0.0], [0.0, 1.0 / scale, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return inverse_scaling @ motion_scaled @ scaling


def _make_pitch_mask(
    image_shape: tuple[int, int],
    homography_scaled: np.ndarray,
) -> np.ndarray | None:
    """Rasterise only the *visible* part of the 105x68 m pitch.

    Do not build this mask from the four projected pitch corners. In real TV/Veo
    views, especially close to one goal, one or more physical corners can lie
    extremely far outside the image even when the visible field geometry is
    perfectly valid. A corner polygon therefore gives false failures.

    Instead we create a canonical top-down binary pitch raster and warp it
    directly into the current image with the world->image homography. OpenCV
    clips the result to the viewport, so only the actually visible pitch area
    matters.
    """
    height, width = image_shape
    if height <= 0 or width <= 0:
        return None

    h = _normalise_homography(homography_scaled)
    if h is None:
        return None

    # 8 px / metre gives a 840x544 source mask: detailed enough for clean
    # boundaries while remaining tiny compared with a video frame.
    pixels_per_metre = 8.0
    canonical_width = int(round(105.0 * pixels_per_metre)) + 1
    canonical_height = int(round(68.0 * pixels_per_metre)) + 1
    canonical_mask = np.full((canonical_height, canonical_width), 255, dtype=np.uint8)

    # Source-raster pixel -> canonical world coordinate.
    # u=0 -> x=-52.5, u=840 -> x=+52.5
    # v=0 -> y=-34,   v=544 -> y=+34
    raster_to_world = np.array(
        [
            [1.0 / pixels_per_metre, 0.0, -52.5],
            [0.0, 1.0 / pixels_per_metre, -34.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    raster_to_image = h @ raster_to_world
    raster_to_image = _normalise_homography(raster_to_image)
    if raster_to_image is None:
        return None

    try:
        mask = cv2.warpPerspective(
            canonical_mask,
            raster_to_image,
            (width, height),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
    except cv2.error:
        return None

    # Stay a few pixels inside the projected touch/goal lines. This reduces
    # features from advertising boards, benches or spectators exactly on the
    # boundary without requiring perfect manual calibration.
    erosion_radius = max(1, int(round(3 * max(0.5, min(1.0, width / 960.0)))))
    kernel_size = erosion_radius * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask = cv2.erode(mask, kernel, iterations=1)

    visible_pixels = int(cv2.countNonZero(mask))
    # A near-goal shot may show only a modest fraction of the full pitch, so
    # use a deliberately low viewport-area threshold.
    minimum_pixels = max(250, int(width * height * 0.0025))
    if visible_pixels < minimum_pixels:
        return None
    return mask


def _prepare_frame(frame_bgr: np.ndarray, scale: float) -> np.ndarray:
    if scale != 1.0:
        frame_bgr = cv2.resize(
            frame_bgr,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA,
        )
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    # Small blur removes codec noise but keeps painted-line / grass texture.
    return cv2.GaussianBlur(gray, (3, 3), 0)


def _detect_features(gray: np.ndarray, mask: np.ndarray, config: PropagationConfig) -> np.ndarray | None:
    points = cv2.goodFeaturesToTrack(
        gray,
        mask=mask,
        maxCorners=config.max_corners,
        qualityLevel=config.quality_level,
        minDistance=config.min_distance * config.working_scale,
        blockSize=config.block_size,
        useHarrisDetector=False,
    )
    if points is None:
        return None
    return points.astype(np.float32)


def _track_features(
    previous_gray: np.ndarray,
    current_gray: np.ndarray,
    previous_points: np.ndarray,
    config: PropagationConfig,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | int]] | None:
    window = max(9, int(round(config.lk_window * config.working_scale)))
    if window % 2 == 0:
        window += 1

    lk_params = dict(
        winSize=(window, window),
        maxLevel=config.lk_max_level,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    forward, status_forward, _ = cv2.calcOpticalFlowPyrLK(
        previous_gray,
        current_gray,
        previous_points,
        None,
        **lk_params,
    )
    if forward is None or status_forward is None:
        return None

    backward, status_backward, _ = cv2.calcOpticalFlowPyrLK(
        current_gray,
        previous_gray,
        forward,
        None,
        **lk_params,
    )
    if backward is None or status_backward is None:
        return None

    previous_flat = previous_points.reshape(-1, 2)
    forward_flat = forward.reshape(-1, 2)
    backward_flat = backward.reshape(-1, 2)
    valid = (
        status_forward.reshape(-1).astype(bool)
        & status_backward.reshape(-1).astype(bool)
        & np.isfinite(forward_flat).all(axis=1)
        & np.isfinite(backward_flat).all(axis=1)
    )

    fb_error_scaled = np.linalg.norm(previous_flat - backward_flat, axis=1)
    fb_limit_scaled = config.fb_max_error_px * config.working_scale
    valid &= fb_error_scaled <= fb_limit_scaled

    previous_valid = previous_flat[valid]
    current_valid = forward_flat[valid]
    if len(previous_valid) == 0:
        return None

    diagnostics = {
        "detected_features": int(len(previous_flat)),
        "tracked_features": int(len(previous_valid)),
        "median_fb_error_px": float(np.median(fb_error_scaled[valid]) / config.working_scale),
    }
    return previous_valid, current_valid, diagnostics


def _estimate_frame_motion(
    previous_points: np.ndarray,
    current_points: np.ndarray,
    config: PropagationConfig,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | int]] | None:
    if len(previous_points) < 4:
        return None

    threshold_scaled = config.ransac_reproj_threshold_px * config.working_scale
    motion, inlier_mask = cv2.findHomography(
        previous_points,
        current_points,
        method=cv2.RANSAC,
        ransacReprojThreshold=threshold_scaled,
        maxIters=3000,
        confidence=0.995,
    )
    motion = _normalise_homography(motion) if motion is not None else None
    if motion is None or inlier_mask is None:
        return None

    inliers = inlier_mask.reshape(-1).astype(bool)
    inlier_count = int(np.count_nonzero(inliers))
    inlier_ratio = float(inlier_count / max(len(inliers), 1))
    if inlier_count:
        predicted = cv2.perspectiveTransform(
            previous_points.reshape(-1, 1, 2).astype(np.float32),
            motion.astype(np.float64),
        ).reshape(-1, 2)
        reprojection_scaled = np.linalg.norm(predicted - current_points, axis=1)
        median_reprojection = float(np.median(reprojection_scaled[inliers]) / config.working_scale)
    else:
        median_reprojection = float("inf")

    diagnostics = {
        "inliers": inlier_count,
        "inlier_ratio": inlier_ratio,
        "median_reprojection_error_px": median_reprojection,
    }
    return motion, inliers, diagnostics


def _validate_candidate(
    previous_h: np.ndarray,
    next_h: np.ndarray,
    native_width: int,
    native_height: int,
    config: PropagationConfig,
) -> tuple[bool, str | None, dict[str, float]]:
    previous_polygon = _pitch_polygon(previous_h)
    next_polygon = _pitch_polygon(next_h)
    if previous_polygon is None or next_polygon is None:
        return False, "La proyección del campo dejó de ser finita.", {}

    coordinate_limit_x = native_width * 12.0
    coordinate_limit_y = native_height * 12.0
    if (
        np.max(np.abs(next_polygon[:, 0])) > coordinate_limit_x
        or np.max(np.abs(next_polygon[:, 1])) > coordinate_limit_y
    ):
        return False, "La proyección del campo explotó numéricamente.", {}

    previous_area = _polygon_area(previous_polygon)
    next_area = _polygon_area(next_polygon)
    if previous_area < 1e-3 or next_area < 1e-3:
        return False, "La proyección del campo se volvió degenerada.", {}

    area_ratio = next_area / previous_area
    if not (config.min_pitch_area_ratio <= area_ratio <= config.max_pitch_area_ratio):
        return False, "El cambio de escala entre fotogramas es demasiado grande.", {"pitch_area_ratio": area_ratio}

    previous_refs = _project_points(PITCH_REFERENCE_POINTS, previous_h)
    next_refs = _project_points(PITCH_REFERENCE_POINTS, next_h)
    if previous_refs is None or next_refs is None:
        return False, "No se pudieron proyectar referencias del campo.", {"pitch_area_ratio": area_ratio}

    # Judge camera motion using references reasonably close to the current
    # viewport. Distant off-screen corners can move many pixels harmlessly.
    margin_x = native_width * 0.5
    margin_y = native_height * 0.5
    near_view = (
        (previous_refs[:, 0] >= -margin_x)
        & (previous_refs[:, 0] <= native_width + margin_x)
        & (previous_refs[:, 1] >= -margin_y)
        & (previous_refs[:, 1] <= native_height + margin_y)
    )
    displacements = np.linalg.norm(next_refs - previous_refs, axis=1)
    judged = displacements[near_view] if np.any(near_view) else displacements
    median_motion = float(np.median(judged))
    max_motion = float(np.max(judged))

    if median_motion > config.max_median_frame_motion_px or max_motion > config.max_reference_motion_px:
        return (
            False,
            "El movimiento estimado de cámara entre fotogramas es poco plausible.",
            {
                "pitch_area_ratio": area_ratio,
                "median_reference_motion_px": median_motion,
                "max_reference_motion_px": max_motion,
            },
        )

    try:
        condition = float(np.linalg.cond(next_h))
    except np.linalg.LinAlgError:
        return False, "La homografía propagada es singular.", {}

    return True, None, {
        "pitch_area_ratio": float(area_ratio),
        "median_reference_motion_px": median_motion,
        "max_reference_motion_px": max_motion,
        "homography_condition": condition,
    }


def _confidence(
    tracked: int,
    inliers: int,
    inlier_ratio: float,
    median_reprojection_error_px: float,
    median_fb_error_px: float,
) -> float:
    track_score = min(1.0, tracked / 80.0)
    inlier_score = min(1.0, inliers / 50.0)
    ratio_score = min(1.0, max(0.0, (inlier_ratio - 0.2) / 0.6))
    reprojection_score = float(np.exp(-max(median_reprojection_error_px, 0.0) / 4.0))
    fb_score = float(np.exp(-max(median_fb_error_px, 0.0) / 3.0))
    score = (
        0.15 * track_score
        + 0.20 * inlier_score
        + 0.30 * ratio_score
        + 0.20 * reprojection_score
        + 0.15 * fb_score
    )
    return float(np.clip(score, 0.0, 1.0))


def _read_frame_at(cap: cv2.VideoCapture, frame_index: int) -> np.ndarray | None:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
    ok, frame = cap.read()
    return frame if ok and frame is not None else None


def propagate_field_homography(
    video_path: str | Path,
    anchor_frame: int,
    anchor_homography: list[list[float]] | np.ndarray,
    *,
    max_frames: int = 150,
    direction: int = 1,
    config: PropagationConfig | None = None,
) -> dict[str, Any]:
    """Propagate one manual pitch calibration through nearby video frames.

    Parameters
    ----------
    video_path:
        Source MP4 path.
    anchor_frame:
        Exact frame that owns the manually validated homography.
    anchor_homography:
        3x3 world-pitch -> native-video homography.
    max_frames:
        Number of *additional* frames to attempt. 150 is roughly five seconds
        at 30 fps and is deliberately conservative for the first Supervisor UI.
    direction:
        +1 forward, -1 backward.
    config:
        Optional thresholds.

    Returns
    -------
    A JSON-serialisable dict. Propagation stops at the first unreliable frame
    rather than continuing with a known-bad geometry.
    """

    cfg = config or PropagationConfig()
    if direction not in (-1, 1):
        return {"status": "invalid", "detail": "direction debe ser 1 o -1.", "frames": []}
    if max_frames < 1:
        return {"status": "invalid", "detail": "max_frames debe ser mayor que cero.", "frames": []}
    if not (0.2 <= cfg.working_scale <= 1.0):
        return {"status": "invalid", "detail": "working_scale debe estar entre 0.2 y 1.0.", "frames": []}

    video_path = Path(video_path)
    if not video_path.exists():
        return {"status": "invalid", "detail": f"No existe el vídeo: {video_path}", "frames": []}

    current_h = _normalise_homography(np.asarray(anchor_homography, dtype=np.float64))
    if current_h is None:
        return {"status": "invalid", "detail": "La homografía ancla no es válida.", "frames": []}

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"status": "invalid", "detail": "OpenCV no pudo abrir el vídeo.", "frames": []}

    try:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        native_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        native_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)

        if frame_count and not (0 <= anchor_frame < frame_count):
            return {"status": "invalid", "detail": "El fotograma ancla está fuera del vídeo.", "frames": []}
        if native_width <= 0 or native_height <= 0:
            return {"status": "invalid", "detail": "No se pudo leer la resolución del vídeo.", "frames": []}

        previous_bgr = _read_frame_at(cap, anchor_frame)
        if previous_bgr is None:
            return {"status": "invalid", "detail": "No se pudo leer el fotograma ancla.", "frames": []}
        previous_gray = _prepare_frame(previous_bgr, cfg.working_scale)

        results: list[dict[str, Any]] = [
            {
                "frame": int(anchor_frame),
                "homography": current_h.tolist(),
                "source": "manual_anchor",
                "confidence": 1.0,
            }
        ]

        current_index = int(anchor_frame)
        stopped_reason: str | None = None
        stopped_frame: int | None = None

        for _ in range(max_frames):
            target_index = current_index + direction
            if target_index < 0 or (frame_count and target_index >= frame_count):
                break

            if direction > 0:
                ok, current_bgr = cap.read()
                if not ok:
                    current_bgr = None
            else:
                current_bgr = _read_frame_at(cap, target_index)
            if current_bgr is None:
                stopped_reason = "No se pudo leer el siguiente fotograma."
                stopped_frame = target_index
                break
            current_gray = _prepare_frame(current_bgr, cfg.working_scale)

            scaled_h = _scaled_homography(current_h, cfg.working_scale)
            pitch_mask = _make_pitch_mask(previous_gray.shape, scaled_h)
            if pitch_mask is None:
                stopped_reason = "La zona proyectada del campo no es utilizable para tracking."
                stopped_frame = target_index
                break

            previous_points = _detect_features(previous_gray, pitch_mask, cfg)
            if previous_points is None or len(previous_points) < cfg.min_tracks:
                stopped_reason = "No hay suficientes puntos visuales fiables dentro del campo."
                stopped_frame = target_index
                break

            tracked_result = _track_features(previous_gray, current_gray, previous_points, cfg)
            if tracked_result is None:
                stopped_reason = "El flujo óptico no produjo suficientes correspondencias."
                stopped_frame = target_index
                break
            tracked_previous, tracked_current, track_diag = tracked_result

            if len(tracked_previous) < cfg.min_tracks:
                stopped_reason = "Se perdieron demasiados puntos durante el flujo óptico."
                stopped_frame = target_index
                break

            motion_result = _estimate_frame_motion(tracked_previous, tracked_current, cfg)
            if motion_result is None:
                stopped_reason = "No se pudo estimar el movimiento de cámara."
                stopped_frame = target_index
                break
            motion_scaled, _, motion_diag = motion_result

            if motion_diag["inliers"] < cfg.min_inliers:
                stopped_reason = "Muy pocos puntos apoyan el movimiento de cámara estimado."
                stopped_frame = target_index
                break
            if motion_diag["inlier_ratio"] < cfg.min_inlier_ratio:
                stopped_reason = "El movimiento estimado tiene demasiados outliers."
                stopped_frame = target_index
                break
            if motion_diag["median_reprojection_error_px"] > cfg.max_median_reprojection_error_px:
                stopped_reason = "El movimiento estimado tiene demasiado error geométrico."
                stopped_frame = target_index
                break

            motion_native = _motion_to_native(motion_scaled, cfg.working_scale)
            candidate_h = _normalise_homography(motion_native @ current_h)
            if candidate_h is None:
                stopped_reason = "La homografía propagada dejó de ser válida."
                stopped_frame = target_index
                break

            valid, validation_reason, validation_diag = _validate_candidate(
                current_h,
                candidate_h,
                native_width,
                native_height,
                cfg,
            )
            if not valid:
                stopped_reason = validation_reason
                stopped_frame = target_index
                break

            confidence = _confidence(
                int(track_diag["tracked_features"]),
                int(motion_diag["inliers"]),
                float(motion_diag["inlier_ratio"]),
                float(motion_diag["median_reprojection_error_px"]),
                float(track_diag["median_fb_error_px"]),
            )
            if confidence < cfg.min_confidence:
                stopped_reason = "La confianza de propagación cayó por debajo del mínimo."
                stopped_frame = target_index
                break

            results.append(
                {
                    "frame": int(target_index),
                    "homography": candidate_h.tolist(),
                    "source": "propagated_optical_flow",
                    "confidence": confidence,
                    "tracking": {
                        **track_diag,
                        **motion_diag,
                        **validation_diag,
                    },
                }
            )

            current_h = candidate_h
            current_index = target_index
            previous_gray = current_gray

        propagated_count = max(0, len(results) - 1)
        status = "ok" if stopped_reason is None else ("partial" if propagated_count else "lost")
        direction_label = "forward" if direction > 0 else "backward"
        return {
            "status": status,
            "detail": stopped_reason,
            "anchor_frame": int(anchor_frame),
            "direction": direction_label,
            "fps": fps or None,
            "requested_frames": int(max_frames),
            "propagated_frames": propagated_count,
            "stopped_frame": stopped_frame,
            "config": asdict(cfg),
            "frames": results,
        }
    finally:
        cap.release()