"""Anchor-relative temporal FIELD propagation for RealJaen Supervisor.

Why this version exists
-----------------------
A valid world->image homography may project *off-screen* parts of the 105x68 m
pitch extremely far away, or even cross projective infinity, while the visible
part of the pitch is perfectly usable. Therefore this module deliberately does
NOT validate propagation from the four full-pitch corner projections.

It also avoids chaining frame-to-frame field homographies:

    H_t = M_t @ H_(t-1)

because small projective errors accumulate.

Instead, feature identities are retained from the manual anchor frame. Each
target frame estimates the image motion directly from ANCHOR feature positions
to CURRENT feature positions:

    H_t = M_anchor_to_t @ H_anchor

The optical-flow tracking itself is sequential (for robustness), but the FIELD
homography is always reconstructed from the original manual anchor.

Public API is intentionally compatible with the existing Supervisor endpoint:

    propagate_field_homography(
        video_path,
        anchor_frame,
        anchor_homography,
        max_frames=150,
        direction=1,
    )
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


# A dense set of virtual pitch-plane references. They are used ONLY for
# propagation validation. References whose ANCHOR projection is far outside the
# viewport are ignored, so an off-screen projective pole does not invalidate an
# otherwise useful camera view.
_REFERENCE_X = np.array(
    [-52.5, -47.0, -41.5, -36.0, -18.0, 0.0, 18.0, 36.0, 41.5, 47.0, 52.5],
    dtype=np.float64,
)
_REFERENCE_Y = np.array(
    [-34.0, -20.16, -9.16, 0.0, 9.16, 20.16, 34.0],
    dtype=np.float64,
)
PITCH_REFERENCE_POINTS = np.array(
    [(x, y) for y in _REFERENCE_Y for x in _REFERENCE_X],
    dtype=np.float64,
)


@dataclass(frozen=True)
class PropagationConfig:
    """Conservative defaults for a 1920x1080-ish Veo export."""

    working_scale: float = 0.5

    # Feature detection. Grid sampling prevents one textured stand/advert from
    # dominating all tracks.
    max_corners: int = 1400
    quality_level: float = 0.006
    min_distance_px: float = 7.0
    block_size: int = 7
    grid_columns: int = 10
    grid_rows: int = 6
    max_features_per_cell: int = 28

    # KLT tracking.
    lk_window_px: int = 27
    lk_max_level: int = 4
    fb_max_error_px: float = 2.5
    max_step_motion_px: float = 90.0

    # Anchor -> current robust motion.
    ransac_reproj_threshold_px: float = 3.5
    min_tracks: int = 35
    min_inliers: int = 24
    min_inlier_ratio: float = 0.45
    max_median_reprojection_error_px: float = 4.0

    # Candidate FIELD validation using only references near the anchor viewport.
    reference_margin_ratio: float = 0.55
    min_visible_references: int = 6
    max_anchor_median_motion_px: float = 900.0
    max_visible_reference_displacement_px: float = 1400.0
    max_homography_condition: float = 5e8

    # Compare the *visible projected pitch mask* rather than the polygon made by
    # four full-pitch corners.
    min_visible_mask_ratio: float = 0.28
    max_visible_mask_ratio: float = 3.5
    min_visible_pitch_ratio: float = 0.0025

    min_confidence: float = 0.32


def _normalise_homography(matrix: Any) -> np.ndarray | None:
    try:
        h = np.asarray(matrix, dtype=np.float64)
    except (TypeError, ValueError):
        return None

    if h.shape != (3, 3) or not np.isfinite(h).all():
        return None

    scale = float(h[2, 2])
    if abs(scale) < 1e-12:
        norm = float(np.linalg.norm(h))
        if not np.isfinite(norm) or norm < 1e-12:
            return None
        h = h / norm
    else:
        h = h / scale

    if not np.isfinite(h).all():
        return None

    try:
        det = float(np.linalg.det(h))
        cond = float(np.linalg.cond(h))
    except np.linalg.LinAlgError:
        return None

    if abs(det) < 1e-12 or not np.isfinite(cond):
        return None
    return h


def _project_points_partial(
    points_xy: np.ndarray,
    homography: np.ndarray,
    *,
    denominator_epsilon: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """Project points without failing because ONE off-screen point hits infinity.

    Returns
    -------
    projected:
        Nx2 array. Invalid rows contain NaN.
    valid:
        Boolean mask for finite, non-singular rows.
    """

    points = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    homogeneous = np.column_stack(
        (points, np.ones(len(points), dtype=np.float64))
    )
    projected_h = homogeneous @ homography.T
    denominator = projected_h[:, 2]

    valid = (
        np.isfinite(projected_h).all(axis=1)
        & np.isfinite(denominator)
        & (np.abs(denominator) > denominator_epsilon)
    )

    projected = np.full((len(points), 2), np.nan, dtype=np.float64)
    projected[valid] = (
        projected_h[valid, :2] / denominator[valid, None]
    )
    valid &= np.isfinite(projected).all(axis=1)
    return projected, valid


def _scaled_homography(homography_native: np.ndarray, scale: float) -> np.ndarray:
    scaling = np.array(
        [[scale, 0.0, 0.0], [0.0, scale, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return scaling @ homography_native


def _motion_to_native(motion_scaled: np.ndarray, scale: float) -> np.ndarray:
    scaling = np.array(
        [[scale, 0.0, 0.0], [0.0, scale, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    inverse_scaling = np.array(
        [
            [1.0 / scale, 0.0, 0.0],
            [0.0, 1.0 / scale, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return inverse_scaling @ motion_scaled @ scaling


def _make_pitch_mask(
    image_shape: tuple[int, int],
    homography_scaled: np.ndarray,
    config: PropagationConfig,
) -> np.ndarray | None:
    """Rasterise only the visible part of the canonical pitch.

    This is intentionally inverse-mapped by OpenCV and clipped to the viewport.
    It remains useful even if an invisible part of the full 105x68 plane is
    extremely far away in the current perspective.
    """

    height, width = image_shape
    if height <= 0 or width <= 0:
        return None

    h = _normalise_homography(homography_scaled)
    if h is None:
        return None

    pixels_per_metre = 8.0
    canonical_width = int(round(105.0 * pixels_per_metre)) + 1
    canonical_height = int(round(68.0 * pixels_per_metre)) + 1
    canonical_mask = np.full(
        (canonical_height, canonical_width),
        255,
        dtype=np.uint8,
    )

    raster_to_world = np.array(
        [
            [1.0 / pixels_per_metre, 0.0, -52.5],
            [0.0, 1.0 / pixels_per_metre, -34.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    raster_to_image = _normalise_homography(h @ raster_to_world)
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

    # Stay slightly inside the manually calibrated boundary.
    erosion_radius = max(
        1,
        int(round(3 * max(0.5, min(1.0, width / 960.0)))),
    )
    kernel_size = erosion_radius * 2 + 1
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size),
    )
    mask = cv2.erode(mask, kernel, iterations=1)

    visible_pixels = int(cv2.countNonZero(mask))
    minimum_pixels = max(
        250,
        int(width * height * config.min_visible_pitch_ratio),
    )
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
    return cv2.GaussianBlur(gray, (3, 3), 0)


def _detect_features_grid(
    gray: np.ndarray,
    mask: np.ndarray,
    config: PropagationConfig,
) -> np.ndarray | None:
    """Detect spatially distributed anchor features inside the visible pitch."""

    height, width = gray.shape[:2]
    collected: list[np.ndarray] = []

    cell_width = width / config.grid_columns
    cell_height = height / config.grid_rows
    min_distance_scaled = max(
        2.0,
        config.min_distance_px * config.working_scale,
    )

    for row in range(config.grid_rows):
        y0 = int(round(row * cell_height))
        y1 = int(round((row + 1) * cell_height))
        for column in range(config.grid_columns):
            x0 = int(round(column * cell_width))
            x1 = int(round((column + 1) * cell_width))
            if x1 <= x0 or y1 <= y0:
                continue

            cell_mask = np.zeros_like(mask)
            cell_mask[y0:y1, x0:x1] = mask[y0:y1, x0:x1]
            if cv2.countNonZero(cell_mask) < 20:
                continue

            points = cv2.goodFeaturesToTrack(
                gray,
                mask=cell_mask,
                maxCorners=config.max_features_per_cell,
                qualityLevel=config.quality_level,
                minDistance=min_distance_scaled,
                blockSize=config.block_size,
                useHarrisDetector=False,
            )
            if points is not None:
                collected.append(points.reshape(-1, 2))

    if not collected:
        return None

    points = np.concatenate(collected, axis=0).astype(np.float32)
    if len(points) > config.max_corners:
        points = points[: config.max_corners]
    return points.reshape(-1, 1, 2)


def _track_features(
    previous_gray: np.ndarray,
    current_gray: np.ndarray,
    previous_points: np.ndarray,
    config: PropagationConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float | int]] | None:
    """Track points one step and return the validity mask for anchor identity."""

    window = max(
        9,
        int(round(config.lk_window_px * config.working_scale)),
    )
    if window % 2 == 0:
        window += 1

    lk_params = dict(
        winSize=(window, window),
        maxLevel=config.lk_max_level,
        criteria=(
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            30,
            0.01,
        ),
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

    fb_error_scaled = np.linalg.norm(
        previous_flat - backward_flat,
        axis=1,
    )
    valid &= (
        fb_error_scaled
        <= config.fb_max_error_px * config.working_scale
    )

    previous_valid = previous_flat[valid]
    current_valid = forward_flat[valid]
    if len(previous_valid) == 0:
        return None

    step_motion_native = (
        np.linalg.norm(current_valid - previous_valid, axis=1)
        / config.working_scale
    )

    diagnostics = {
        "tracked_features": int(len(current_valid)),
        "median_fb_error_px": float(
            np.median(fb_error_scaled[valid]) / config.working_scale
        ),
        "median_step_motion_px": float(np.median(step_motion_native)),
    }
    return previous_valid, current_valid, valid, diagnostics


def _estimate_anchor_motion(
    anchor_points: np.ndarray,
    current_points: np.ndarray,
    config: PropagationConfig,
) -> tuple[np.ndarray, dict[str, float | int]] | None:
    """Estimate ANCHOR-image -> CURRENT-image motion directly."""

    if len(anchor_points) < 4:
        return None

    threshold_scaled = (
        config.ransac_reproj_threshold_px * config.working_scale
    )
    motion, inlier_mask = cv2.findHomography(
        anchor_points.reshape(-1, 2),
        current_points.reshape(-1, 2),
        method=cv2.RANSAC,
        ransacReprojThreshold=threshold_scaled,
        maxIters=4000,
        confidence=0.995,
    )
    motion = (
        _normalise_homography(motion)
        if motion is not None
        else None
    )
    if motion is None or inlier_mask is None:
        return None

    inliers = inlier_mask.reshape(-1).astype(bool)
    inlier_count = int(np.count_nonzero(inliers))
    inlier_ratio = float(inlier_count / max(len(inliers), 1))

    if inlier_count:
        predicted = cv2.perspectiveTransform(
            anchor_points.reshape(-1, 1, 2).astype(np.float32),
            motion.astype(np.float64),
        ).reshape(-1, 2)
        errors_scaled = np.linalg.norm(
            predicted - current_points.reshape(-1, 2),
            axis=1,
        )
        median_error = float(
            np.median(errors_scaled[inliers]) / config.working_scale
        )
        p90_error = float(
            np.percentile(errors_scaled[inliers], 90)
            / config.working_scale
        )
    else:
        median_error = float("inf")
        p90_error = float("inf")

    diagnostics = {
        "tracks_for_model": int(len(anchor_points)),
        "inliers": inlier_count,
        "inlier_ratio": inlier_ratio,
        "median_reprojection_error_px": median_error,
        "p90_reprojection_error_px": p90_error,
    }
    return motion, diagnostics


def _visible_reference_mask(
    anchor_h: np.ndarray,
    native_width: int,
    native_height: int,
    config: PropagationConfig,
) -> tuple[np.ndarray, np.ndarray]:
    anchor_projected, valid = _project_points_partial(
        PITCH_REFERENCE_POINTS,
        anchor_h,
    )

    margin_x = native_width * config.reference_margin_ratio
    margin_y = native_height * config.reference_margin_ratio

    near = (
        valid
        & (anchor_projected[:, 0] >= -margin_x)
        & (anchor_projected[:, 0] <= native_width + margin_x)
        & (anchor_projected[:, 1] >= -margin_y)
        & (anchor_projected[:, 1] <= native_height + margin_y)
    )
    return anchor_projected, near


def _validate_candidate(
    anchor_h: np.ndarray,
    candidate_h: np.ndarray,
    *,
    anchor_reference_projection: np.ndarray,
    anchor_reference_mask: np.ndarray,
    previous_mask_pixels: int,
    current_mask_pixels: int,
    native_width: int,
    native_height: int,
    config: PropagationConfig,
) -> tuple[bool, str | None, dict[str, float | int]]:
    """Validate only geometry relevant to the current visible camera view."""

    try:
        condition = float(np.linalg.cond(candidate_h))
    except np.linalg.LinAlgError:
        return False, "La homografía propagada es singular.", {}

    if (
        not np.isfinite(condition)
        or condition > config.max_homography_condition
    ):
        return (
            False,
            "La homografía propagada es numéricamente inestable.",
            {"homography_condition": condition},
        )

    candidate_projected, candidate_valid = _project_points_partial(
        PITCH_REFERENCE_POINTS,
        candidate_h,
    )
    judged = anchor_reference_mask & candidate_valid

    if int(np.count_nonzero(judged)) < config.min_visible_references:
        return (
            False,
            "Quedaron muy pocas referencias visibles para validar el campo.",
            {"homography_condition": condition},
        )

    displacement = np.linalg.norm(
        candidate_projected[judged]
        - anchor_reference_projection[judged],
        axis=1,
    )
    median_displacement = float(np.median(displacement))
    max_displacement = float(np.max(displacement))

    if (
        median_displacement > config.max_anchor_median_motion_px
        or max_displacement
        > config.max_visible_reference_displacement_px
    ):
        return (
            False,
            "El movimiento acumulado de cámara es poco plausible.",
            {
                "homography_condition": condition,
                "median_reference_displacement_px": median_displacement,
                "max_reference_displacement_px": max_displacement,
            },
        )

    mask_ratio = float(
        current_mask_pixels / max(previous_mask_pixels, 1)
    )
    if not (
        config.min_visible_mask_ratio
        <= mask_ratio
        <= config.max_visible_mask_ratio
    ):
        return (
            False,
            "La zona visible del campo cambió de forma poco plausible.",
            {
                "homography_condition": condition,
                "visible_mask_ratio": mask_ratio,
                "median_reference_displacement_px": median_displacement,
                "max_reference_displacement_px": max_displacement,
            },
        )

    return True, None, {
        "homography_condition": condition,
        "visible_mask_ratio": mask_ratio,
        "visible_reference_count": int(np.count_nonzero(judged)),
        "median_reference_displacement_px": median_displacement,
        "max_reference_displacement_px": max_displacement,
    }


def _confidence(
    remaining_anchor_tracks: int,
    inliers: int,
    inlier_ratio: float,
    median_reprojection_error_px: float,
    median_fb_error_px: float,
) -> float:
    track_score = min(1.0, remaining_anchor_tracks / 120.0)
    inlier_score = min(1.0, inliers / 80.0)
    ratio_score = min(
        1.0,
        max(0.0, (inlier_ratio - 0.25) / 0.65),
    )
    reprojection_score = float(
        np.exp(-max(median_reprojection_error_px, 0.0) / 3.5)
    )
    fb_score = float(
        np.exp(-max(median_fb_error_px, 0.0) / 2.5)
    )

    score = (
        0.18 * track_score
        + 0.18 * inlier_score
        + 0.28 * ratio_score
        + 0.22 * reprojection_score
        + 0.14 * fb_score
    )
    return float(np.clip(score, 0.0, 1.0))


def _read_frame_at(
    cap: cv2.VideoCapture,
    frame_index: int,
) -> np.ndarray | None:
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
    """Propagate one manual FIELD anchor without chaining field homographies."""

    cfg = config or PropagationConfig()

    if direction not in (-1, 1):
        return {
            "status": "invalid",
            "detail": "direction debe ser 1 o -1.",
            "frames": [],
        }
    if max_frames < 1:
        return {
            "status": "invalid",
            "detail": "max_frames debe ser mayor que cero.",
            "frames": [],
        }
    if not (0.2 <= cfg.working_scale <= 1.0):
        return {
            "status": "invalid",
            "detail": "working_scale debe estar entre 0.2 y 1.0.",
            "frames": [],
        }

    path = Path(video_path)
    if not path.exists():
        return {
            "status": "invalid",
            "detail": f"No existe el vídeo: {path}",
            "frames": [],
        }

    anchor_h = _normalise_homography(anchor_homography)
    if anchor_h is None:
        return {
            "status": "invalid",
            "detail": "La homografía ancla no es válida.",
            "frames": [],
        }

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {
            "status": "invalid",
            "detail": "OpenCV no pudo abrir el vídeo.",
            "frames": [],
        }

    try:
        frame_count = int(
            cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        )
        native_width = int(
            cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0
        )
        native_height = int(
            cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0
        )
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)

        if (
            frame_count
            and not (0 <= int(anchor_frame) < frame_count)
        ):
            return {
                "status": "invalid",
                "detail": "El fotograma ancla está fuera del vídeo.",
                "frames": [],
            }
        if native_width <= 0 or native_height <= 0:
            return {
                "status": "invalid",
                "detail": "No se pudo leer la resolución del vídeo.",
                "frames": [],
            }

        anchor_bgr = _read_frame_at(cap, int(anchor_frame))
        if anchor_bgr is None:
            return {
                "status": "invalid",
                "detail": "No se pudo leer el fotograma ancla.",
                "frames": [],
            }

        anchor_gray = _prepare_frame(
            anchor_bgr,
            cfg.working_scale,
        )
        anchor_h_scaled = _scaled_homography(
            anchor_h,
            cfg.working_scale,
        )
        anchor_mask = _make_pitch_mask(
            anchor_gray.shape,
            anchor_h_scaled,
            cfg,
        )
        if anchor_mask is None:
            return {
                "status": "lost",
                "detail": (
                    "La zona visible del campo del fotograma ancla "
                    "no es utilizable para tracking."
                ),
                "frames": [],
            }

        anchor_points = _detect_features_grid(
            anchor_gray,
            anchor_mask,
            cfg,
        )
        if (
            anchor_points is None
            or len(anchor_points) < cfg.min_tracks
        ):
            return {
                "status": "lost",
                "detail": (
                    "No hay suficientes puntos visuales fiables "
                    "dentro del campo del ancla."
                ),
                "frames": [],
            }

        # These two arrays always share identity. `origin_points` NEVER moves;
        # `previous_points` advances one frame at a time.
        origin_points = anchor_points.reshape(-1, 2).copy()
        previous_points = anchor_points.copy()
        previous_gray = anchor_gray
        previous_mask_pixels = int(cv2.countNonZero(anchor_mask))

        anchor_ref_projection, anchor_ref_mask = (
            _visible_reference_mask(
                anchor_h,
                native_width,
                native_height,
                cfg,
            )
        )
        if (
            int(np.count_nonzero(anchor_ref_mask))
            < cfg.min_visible_references
        ):
            return {
                "status": "lost",
                "detail": (
                    "La calibración ancla no deja suficientes "
                    "referencias cerca de la vista actual."
                ),
                "frames": [],
            }

        results: list[dict[str, Any]] = [
            {
                "frame": int(anchor_frame),
                "homography": anchor_h.tolist(),
                "source": "manual_anchor",
                "confidence": 1.0,
            }
        ]

        current_index = int(anchor_frame)
        stopped_reason: str | None = None
        stopped_frame: int | None = None

        for _ in range(max_frames):
            target_index = current_index + direction
            if target_index < 0 or (
                frame_count and target_index >= frame_count
            ):
                break

            current_bgr = _read_frame_at(cap, target_index)
            if current_bgr is None:
                stopped_reason = (
                    "No se pudo leer el siguiente fotograma."
                )
                stopped_frame = target_index
                break

            current_gray = _prepare_frame(
                current_bgr,
                cfg.working_scale,
            )

            tracked = _track_features(
                previous_gray,
                current_gray,
                previous_points,
                cfg,
            )
            if tracked is None:
                stopped_reason = (
                    "El flujo óptico no produjo correspondencias."
                )
                stopped_frame = target_index
                break

            (
                _previous_valid,
                current_valid,
                valid_mask,
                track_diag,
            ) = tracked

            origin_points = origin_points[valid_mask]
            previous_points = current_valid.reshape(-1, 1, 2)

            if len(origin_points) < cfg.min_tracks:
                stopped_reason = (
                    "Quedan muy pocos tracks del fotograma ancla."
                )
                stopped_frame = target_index
                break

            if (
                float(track_diag["median_step_motion_px"])
                > cfg.max_step_motion_px
            ):
                stopped_reason = (
                    "El movimiento entre fotogramas es demasiado grande."
                )
                stopped_frame = target_index
                break

            motion_result = _estimate_anchor_motion(
                origin_points,
                current_valid,
                cfg,
            )
            if motion_result is None:
                stopped_reason = (
                    "No se pudo estimar el movimiento desde el ancla."
                )
                stopped_frame = target_index
                break

            motion_scaled, motion_diag = motion_result

            if int(motion_diag["inliers"]) < cfg.min_inliers:
                stopped_reason = (
                    "Muy pocos tracks apoyan el movimiento desde el ancla."
                )
                stopped_frame = target_index
                break
            if (
                float(motion_diag["inlier_ratio"])
                < cfg.min_inlier_ratio
            ):
                stopped_reason = (
                    "El movimiento desde el ancla tiene demasiados outliers."
                )
                stopped_frame = target_index
                break
            if (
                float(
                    motion_diag[
                        "median_reprojection_error_px"
                    ]
                )
                > cfg.max_median_reprojection_error_px
            ):
                stopped_reason = (
                    "El movimiento desde el ancla tiene demasiado "
                    "error geométrico."
                )
                stopped_frame = target_index
                break

            motion_native = _motion_to_native(
                motion_scaled,
                cfg.working_scale,
            )
            candidate_h = _normalise_homography(
                motion_native @ anchor_h
            )
            if candidate_h is None:
                stopped_reason = (
                    "La homografía propagada dejó de ser válida."
                )
                stopped_frame = target_index
                break

            candidate_scaled = _scaled_homography(
                candidate_h,
                cfg.working_scale,
            )
            current_mask = _make_pitch_mask(
                current_gray.shape,
                candidate_scaled,
                cfg,
            )
            if current_mask is None:
                stopped_reason = (
                    "La zona proyectada del campo dejó de ser "
                    "utilizable para tracking."
                )
                stopped_frame = target_index
                break

            current_mask_pixels = int(
                cv2.countNonZero(current_mask)
            )

            valid, reason, validation_diag = (
                _validate_candidate(
                    anchor_h,
                    candidate_h,
                    anchor_reference_projection=(
                        anchor_ref_projection
                    ),
                    anchor_reference_mask=anchor_ref_mask,
                    previous_mask_pixels=previous_mask_pixels,
                    current_mask_pixels=current_mask_pixels,
                    native_width=native_width,
                    native_height=native_height,
                    config=cfg,
                )
            )
            if not valid:
                stopped_reason = reason
                stopped_frame = target_index
                break

            confidence = _confidence(
                len(origin_points),
                int(motion_diag["inliers"]),
                float(motion_diag["inlier_ratio"]),
                float(
                    motion_diag[
                        "median_reprojection_error_px"
                    ]
                ),
                float(track_diag["median_fb_error_px"]),
            )
            if confidence < cfg.min_confidence:
                stopped_reason = (
                    "La confianza de propagación cayó por debajo "
                    "del mínimo."
                )
                stopped_frame = target_index
                break

            results.append(
                {
                    "frame": int(target_index),
                    "homography": candidate_h.tolist(),
                    "source": "propagated_optical_flow_anchor",
                    "confidence": confidence,
                    "tracking": {
                        "remaining_anchor_tracks": int(
                            len(origin_points)
                        ),
                        **track_diag,
                        **motion_diag,
                        **validation_diag,
                    },
                }
            )

            current_index = target_index
            previous_gray = current_gray
            previous_mask_pixels = current_mask_pixels

        propagated_count = max(0, len(results) - 1)
        status = (
            "ok"
            if stopped_reason is None
            else ("partial" if propagated_count else "lost")
        )
        direction_label = (
            "forward" if direction > 0 else "backward"
        )

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