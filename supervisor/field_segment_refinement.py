"""Bidirectional refinement of FIELD homographies between manual anchors.

This module is designed for the post-match RealJaen Supervisor workflow:

    manual anchor A
        -> forward propagation
        -> ...
        <- backward propagation
    manual anchor B

Instead of treating a new manual calibration only as a new starting point,
this module uses BOTH anchors to rebuild the complete interval between them.

Important design choices
------------------------
* Manual anchors are immutable ground truth for their exact frames.
* Forward propagation comes from the left anchor.
* Backward propagation comes from the right anchor.
* We never linearly interpolate 3x3 homography matrices.
* When both estimates exist, several canonical pitch-plane points are projected
  by both homographies. Their IMAGE positions are blended according to temporal
  distance and propagation confidence, then a fresh homography is fitted.
* The disagreement between both propagated estimates is kept as a diagnostic.
  Large disagreement means the segment deserves review even if a numerically
  valid blended homography can still be produced.
* If only one propagation reaches a frame, that estimate can still be used.
* If neither side reaches a frame, the gap is reported instead of inventing H.

The public API is JSON-friendly so FastAPI can return it directly.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from supervisor.field_propagation import propagate_field_homography


# Spread across the full 105 x 68 m pitch plane. These are virtual fitting
# points; they are not assumed to be painted/visible landmarks in the image.
_XS = (-52.5, -47.0, -36.0, -18.0, 0.0, 18.0, 36.0, 47.0, 52.5)
_YS = (-34.0, -20.16, -9.16, 0.0, 9.16, 20.16, 34.0)
CANONICAL_BLEND_POINTS = np.asarray(
    [(x, y) for y in _YS for x in _XS],
    dtype=np.float64,
)


@dataclass(frozen=True)
class SegmentRefinementConfig:
    """Conservative defaults for post-match anchor-to-anchor refinement."""

    # Smooth temporal transition between anchors.
    use_smoothstep: bool = True

    # Propagation confidence influences the left/right blend, but temporal
    # position remains the main prior.
    confidence_power: float = 1.35
    min_source_confidence: float = 0.05

    # Fit mainly from virtual world points that are not absurdly far outside
    # the current viewport. This avoids distant projective geometry dominating
    # the compromise when the two estimates disagree.
    viewport_margin_ratio: float = 0.85
    min_fit_points: int = 12

    # Diagnostics / review thresholds.
    disagreement_review_px: float = 35.0
    disagreement_severe_px: float = 120.0
    disagreement_confidence_scale_px: float = 90.0
    max_fit_error_px: float = 8.0

    # Numerical validation.
    max_homography_condition: float = 5e8


def _normalise_homography(matrix: Any) -> np.ndarray | None:
    try:
        h = np.asarray(matrix, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if h.shape != (3, 3) or not np.isfinite(h).all():
        return None
    if abs(float(h[2, 2])) < 1e-12:
        return None
    h = h / float(h[2, 2])
    if not np.isfinite(h).all():
        return None
    try:
        determinant = float(np.linalg.det(h))
        condition = float(np.linalg.cond(h))
    except np.linalg.LinAlgError:
        return None
    if abs(determinant) < 1e-12 or not np.isfinite(condition):
        return None
    return h


def _project_points(points_xy: np.ndarray, h: np.ndarray) -> np.ndarray | None:
    points = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    homogeneous = np.column_stack((points, np.ones(len(points), dtype=np.float64)))
    projected_h = homogeneous @ h.T
    if not np.isfinite(projected_h).all():
        return None
    denominator = projected_h[:, 2]
    if np.any(np.abs(denominator) < 1e-10):
        return None
    projected = projected_h[:, :2] / denominator[:, None]
    return projected if np.isfinite(projected).all() else None


def _smoothstep(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def _frame_map(propagation: dict[str, Any]) -> dict[int, dict[str, Any]]:
    mapped: dict[int, dict[str, Any]] = {}
    for item in propagation.get("frames", []) or []:
        try:
            frame = int(item["frame"])
        except (KeyError, TypeError, ValueError):
            continue
        if _normalise_homography(item.get("homography")) is not None:
            mapped[frame] = item
    return mapped


def _video_size(video_path: Path) -> tuple[int, int] | None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if width <= 0 or height <= 0:
            return None
        return width, height
    finally:
        cap.release()


def _near_viewport_mask(
    left_points: np.ndarray,
    right_points: np.ndarray,
    width: int,
    height: int,
    margin_ratio: float,
) -> np.ndarray:
    margin_x = width * margin_ratio
    margin_y = height * margin_ratio

    def near(points: np.ndarray) -> np.ndarray:
        return (
            (points[:, 0] >= -margin_x)
            & (points[:, 0] <= width + margin_x)
            & (points[:, 1] >= -margin_y)
            & (points[:, 1] <= height + margin_y)
        )

    return near(left_points) | near(right_points)


def _fit_homography(
    world_points: np.ndarray,
    image_points: np.ndarray,
    config: SegmentRefinementConfig,
) -> tuple[np.ndarray | None, float | None]:
    if len(world_points) < 4 or len(image_points) != len(world_points):
        return None, None

    h, _ = cv2.findHomography(
        world_points.astype(np.float64),
        image_points.astype(np.float64),
        method=0,
    )
    h = _normalise_homography(h)
    if h is None:
        return None, None

    try:
        condition = float(np.linalg.cond(h))
    except np.linalg.LinAlgError:
        return None, None
    if not np.isfinite(condition) or condition > config.max_homography_condition:
        return None, None

    reprojection = _project_points(world_points, h)
    if reprojection is None:
        return None, None
    errors = np.linalg.norm(reprojection - image_points, axis=1)
    fit_error = float(np.median(errors))
    if not np.isfinite(fit_error):
        return None, None
    return h, fit_error


def _blend_homographies_via_pitch_points(
    left_h: np.ndarray,
    right_h: np.ndarray,
    *,
    temporal_t: float,
    left_confidence: float,
    right_confidence: float,
    width: int,
    height: int,
    config: SegmentRefinementConfig,
) -> tuple[np.ndarray | None, dict[str, float | int | bool]]:
    left_projected = _project_points(CANONICAL_BLEND_POINTS, left_h)
    right_projected = _project_points(CANONICAL_BLEND_POINTS, right_h)
    if left_projected is None or right_projected is None:
        return None, {}

    viewport_mask = _near_viewport_mask(
        left_projected,
        right_projected,
        width,
        height,
        config.viewport_margin_ratio,
    )
    if int(np.count_nonzero(viewport_mask)) < config.min_fit_points:
        viewport_mask = np.ones(len(CANONICAL_BLEND_POINTS), dtype=bool)

    world = CANONICAL_BLEND_POINTS[viewport_mask]
    left_img = left_projected[viewport_mask]
    right_img = right_projected[viewport_mask]

    t = _smoothstep(temporal_t) if config.use_smoothstep else float(np.clip(temporal_t, 0.0, 1.0))
    lc = max(config.min_source_confidence, float(left_confidence))
    rc = max(config.min_source_confidence, float(right_confidence))

    left_raw = max(1e-9, (1.0 - t) * (lc ** config.confidence_power))
    right_raw = max(1e-9, t * (rc ** config.confidence_power))
    weight_sum = left_raw + right_raw
    left_weight = left_raw / weight_sum
    right_weight = right_raw / weight_sum

    disagreement = np.linalg.norm(left_img - right_img, axis=1)
    median_disagreement = float(np.median(disagreement))
    p90_disagreement = float(np.percentile(disagreement, 90))

    blended = left_weight * left_img + right_weight * right_img
    h, fit_error = _fit_homography(world, blended, config)
    if h is None or fit_error is None:
        return None, {
            "temporal_t": float(temporal_t),
            "left_weight": float(left_weight),
            "right_weight": float(right_weight),
            "median_disagreement_px": median_disagreement,
            "p90_disagreement_px": p90_disagreement,
        }

    diagnostics: dict[str, float | int | bool] = {
        "temporal_t": float(temporal_t),
        "left_weight": float(left_weight),
        "right_weight": float(right_weight),
        "left_confidence": float(left_confidence),
        "right_confidence": float(right_confidence),
        "fit_points": int(len(world)),
        "fit_error_px": float(fit_error),
        "median_disagreement_px": median_disagreement,
        "p90_disagreement_px": p90_disagreement,
        "needs_review": bool(
            median_disagreement > config.disagreement_review_px
            or fit_error > config.max_fit_error_px
        ),
        "severe_disagreement": bool(median_disagreement > config.disagreement_severe_px),
    }
    return h, diagnostics


def _source_confidence(item: dict[str, Any] | None) -> float:
    if not item:
        return 0.0
    try:
        return float(np.clip(float(item.get("confidence", 0.0)), 0.0, 1.0))
    except (TypeError, ValueError):
        return 0.0


def _summary(propagation: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": propagation.get("status"),
        "detail": propagation.get("detail"),
        "anchor_frame": propagation.get("anchor_frame"),
        "direction": propagation.get("direction"),
        "propagated_frames": propagation.get("propagated_frames"),
        "stopped_frame": propagation.get("stopped_frame"),
    }


def _compress_ranges(frames: list[int]) -> list[dict[str, int]]:
    if not frames:
        return []
    ordered = sorted(set(int(frame) for frame in frames))
    ranges: list[dict[str, int]] = []
    start = previous = ordered[0]
    for frame in ordered[1:]:
        if frame == previous + 1:
            previous = frame
            continue
        ranges.append({"start": start, "end": previous})
        start = previous = frame
    ranges.append({"start": start, "end": previous})
    return ranges


def refine_field_segment(
    video_path: str | Path,
    left_anchor_frame: int,
    left_anchor_homography: list[list[float]] | np.ndarray,
    right_anchor_frame: int,
    right_anchor_homography: list[list[float]] | np.ndarray,
    *,
    propagation_config: Any | None = None,
    config: SegmentRefinementConfig | None = None,
) -> dict[str, Any]:
    """Refine all FIELD homographies between two manual calibration anchors.

    The interval is solved bidirectionally:
      * left anchor -> forward
      * right anchor -> backward

    Frames reached by both sides are fused through projected pitch-plane
    reference points. Exact anchor frames always keep the manual homography.
    """

    cfg = config or SegmentRefinementConfig()

    try:
        left_frame = int(left_anchor_frame)
        right_frame = int(right_anchor_frame)
    except (TypeError, ValueError):
        return {"status": "invalid", "detail": "Los fotogramas ancla no son válidos.", "frames": []}

    if right_frame <= left_frame:
        return {
            "status": "invalid",
            "detail": "El segundo ancla debe estar después del primero.",
            "frames": [],
        }

    left_h = _normalise_homography(left_anchor_homography)
    right_h = _normalise_homography(right_anchor_homography)
    if left_h is None or right_h is None:
        return {
            "status": "invalid",
            "detail": "Una de las homografías manuales no es válida.",
            "frames": [],
        }

    path = Path(video_path)
    if not path.exists():
        return {"status": "invalid", "detail": f"No existe el vídeo: {path}", "frames": []}

    size = _video_size(path)
    if size is None:
        return {
            "status": "invalid",
            "detail": "No se pudo leer la resolución del vídeo.",
            "frames": [],
        }
    width, height = size

    interval = right_frame - left_frame

    forward_kwargs: dict[str, Any] = {
        "max_frames": interval,
        "direction": 1,
    }
    backward_kwargs: dict[str, Any] = {
        "max_frames": interval,
        "direction": -1,
    }
    if propagation_config is not None:
        forward_kwargs["config"] = propagation_config
        backward_kwargs["config"] = propagation_config

    forward = propagate_field_homography(
        path,
        left_frame,
        left_h,
        **forward_kwargs,
    )
    backward = propagate_field_homography(
        path,
        right_frame,
        right_h,
        **backward_kwargs,
    )

    forward_map = _frame_map(forward)
    backward_map = _frame_map(backward)

    # Always trust explicitly supplied anchors, even if propagation itself
    # could not initialise from one side.
    forward_map[left_frame] = {
        "frame": left_frame,
        "homography": left_h.tolist(),
        "confidence": 1.0,
        "source": "manual_anchor",
    }
    backward_map[right_frame] = {
        "frame": right_frame,
        "homography": right_h.tolist(),
        "confidence": 1.0,
        "source": "manual_anchor",
    }

    results: list[dict[str, Any]] = []
    missing_frames: list[int] = []
    review_frames: list[int] = []

    for frame in range(left_frame, right_frame + 1):
        if frame == left_frame:
            results.append(
                {
                    "frame": frame,
                    "homography": left_h.tolist(),
                    "source": "manual_anchor",
                    "confidence": 1.0,
                    "anchor": "left",
                }
            )
            continue

        if frame == right_frame:
            results.append(
                {
                    "frame": frame,
                    "homography": right_h.tolist(),
                    "source": "manual_anchor",
                    "confidence": 1.0,
                    "anchor": "right",
                }
            )
            continue

        left_item = forward_map.get(frame)
        right_item = backward_map.get(frame)

        left_candidate = _normalise_homography(left_item.get("homography")) if left_item else None
        right_candidate = _normalise_homography(right_item.get("homography")) if right_item else None

        if left_candidate is not None and right_candidate is not None:
            temporal_t = (frame - left_frame) / float(interval)
            left_confidence = _source_confidence(left_item)
            right_confidence = _source_confidence(right_item)

            blended_h, diagnostics = _blend_homographies_via_pitch_points(
                left_candidate,
                right_candidate,
                temporal_t=temporal_t,
                left_confidence=left_confidence,
                right_confidence=right_confidence,
                width=width,
                height=height,
                config=cfg,
            )
            if blended_h is None:
                missing_frames.append(frame)
                continue

            disagreement = float(diagnostics.get("median_disagreement_px", 0.0))
            agreement_penalty = float(
                np.exp(-max(disagreement, 0.0) / cfg.disagreement_confidence_scale_px)
            )
            source_confidence = (
                float(diagnostics.get("left_weight", 0.5)) * left_confidence
                + float(diagnostics.get("right_weight", 0.5)) * right_confidence
            )
            confidence = float(np.clip(source_confidence * agreement_penalty, 0.0, 1.0))

            needs_review = bool(diagnostics.get("needs_review", False))
            if needs_review:
                review_frames.append(frame)

            results.append(
                {
                    "frame": frame,
                    "homography": blended_h.tolist(),
                    "source": "refined_bidirectional",
                    "confidence": confidence,
                    "left_source": left_item.get("source"),
                    "right_source": right_item.get("source"),
                    "refinement": diagnostics,
                }
            )
            continue

        # If only one direction reached this frame, preserve useful coverage but
        # label it clearly as single-sided rather than pretending it was refined.
        single_item = left_item if left_candidate is not None else right_item
        single_h = left_candidate if left_candidate is not None else right_candidate
        if single_item is not None and single_h is not None:
            side = "left" if left_candidate is not None else "right"
            confidence = _source_confidence(single_item)
            review_frames.append(frame)
            results.append(
                {
                    "frame": frame,
                    "homography": single_h.tolist(),
                    "source": "propagated_single_side",
                    "confidence": confidence,
                    "side": side,
                    "needs_review": True,
                }
            )
            continue

        missing_frames.append(frame)

    expected = interval + 1
    produced_frames = {int(item["frame"]) for item in results}
    complete = len(produced_frames) == expected

    if complete and not review_frames:
        status = "ok"
        detail = None
    elif results:
        status = "partial"
        if missing_frames:
            detail = "Hay fotogramas sin cobertura desde ninguno de los dos anclajes."
        elif review_frames:
            detail = "El tramo está cubierto, pero algunos fotogramas necesitan revisión por desacuerdo o propagación unilateral."
        else:
            detail = "El tramo se refinó parcialmente."
    else:
        status = "lost"
        detail = "No se pudo reconstruir el tramo entre las dos calibraciones."

    return {
        "status": status,
        "detail": detail,
        "left_anchor_frame": left_frame,
        "right_anchor_frame": right_frame,
        "interval_frames": interval,
        "produced_frames": len(results),
        "missing_ranges": _compress_ranges(missing_frames),
        "review_ranges": _compress_ranges(review_frames),
        "forward": _summary(forward),
        "backward": _summary(backward),
        "config": asdict(cfg),
        "frames": results,
    }
