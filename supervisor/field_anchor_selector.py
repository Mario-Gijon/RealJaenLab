"""Active HUMAN anchor selector for RealJaen FIELD.

This module turns Camera Atlas Coverage diagnostics into a practical question:

    Which frames should the human calibrate next to improve FIELD coverage
    with the least manual work?

The selector is deliberately conservative:

- HUMAN anchors remain the only Camera Atlas truth.
- propagated/refined homographies are used only as approximate state probes;
- actual video pixels are inspected before recommending a frame;
- several frames from the same temporal/state hole are collapsed;
- within each useful camera-state neighborhood, the selector prefers the frame
  that is easiest to calibrate visually (sharp, pitch visible, useful white
  field-marking evidence, sensible exposure);
- isolated existing HUMAN anchors are reported for review rather than silently
  accepted as proof that the surrounding camera state is well represented.

This V1 still uses Coverage V3 as candidate generation.  It is intentionally a
bridge toward a future propagation-independent view-state scanner.

No review is modified and no candidate is promoted automatically.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from supervisor.field_camera_atlas_coverage import (
    CameraAtlasCoverageConfig,
    robust_camera_distance,
)
from supervisor.field_visibility import compute_visible_pitch_footprint


@dataclass(frozen=True)
class AnchorSelectorConfig:
    max_recommendations: int = 6

    # Start from the worst coverage regions, then search a short neighborhood
    # for a more calibratable representative of the same camera state.
    seed_pool_size: int = 30
    seed_percentile: float = 65.0
    search_radius_frames: int = 14
    search_step_frames: int = 2

    # Do not replace a seed by a visually nicer frame whose approximate camera
    # state has already moved too far away from the original coverage hole.
    max_seed_state_distance_px: float = 100.0

    # Global de-duplication.  Two recommendations should represent genuinely
    # different pieces of missing camera-state coverage.
    min_temporal_separation_frames: int = 25
    min_state_separation_px: float = 85.0
    min_visual_descriptor_distance: float = 0.045

    # Existing HUMAN anchors this isolated are worth explicitly checking.
    isolated_anchor_review_threshold_px: float = 500.0

    # Candidate ranking weights.
    coverage_weight: float = 0.58
    visual_quality_weight: float = 0.30
    local_peak_bonus: float = 0.07
    reliability_bonus: float = 0.05

    # View tags are diagnostics, not hard semantic truth.
    left_right_centroid_threshold_m: float = 10.0
    zoom_open_visible_fraction: float = 0.72
    zoom_close_visible_fraction: float = 0.48


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
        if abs(float(np.linalg.det(h))) < 1e-12:
            return None
    except np.linalg.LinAlgError:
        return None
    return h


def _valid_frame(value: Any) -> int | None:
    try:
        frame = int(value)
    except (TypeError, ValueError):
        return None
    return frame if frame >= 0 else None


def _extract_probe_lookup(
    propagations: list[dict[str, Any]] | None,
    refined_segments: list[dict[str, Any]] | None,
    *,
    video_name: str | None,
) -> dict[int, dict[str, Any]]:
    """Return one diagnostic H per frame; refined wins over propagated."""
    by_frame: dict[int, dict[str, Any]] = {}

    def consume(
        containers: list[dict[str, Any]] | None,
        provenance: str,
        priority: int,
    ) -> None:
        for container in containers or []:
            if not isinstance(container, dict):
                continue

            container_video = container.get("video")
            if (
                video_name is not None
                and container_video is not None
                and container_video != video_name
            ):
                continue

            frames = container.get("frames", [])
            if not isinstance(frames, list):
                continue

            for item in frames:
                if not isinstance(item, dict):
                    continue
                frame = _valid_frame(item.get("frame"))
                h = _as_h(item.get("homography"))
                if frame is None or h is None:
                    continue

                previous = by_frame.get(frame)
                if (
                    previous is None
                    or priority > int(previous["_priority"])
                ):
                    by_frame[frame] = {
                        "frame": frame,
                        "homography": h.tolist(),
                        "provenance": provenance,
                        "_priority": priority,
                    }

    consume(propagations, "propagated", 1)
    consume(refined_segments, "refined", 2)

    for record in by_frame.values():
        record.pop("_priority", None)

    return by_frame


def _coverage_probe_lookup(
    coverage_analysis: dict[str, Any],
) -> dict[int, dict[str, Any]]:
    curve = (
        coverage_analysis.get("probe_curve", [])
        if isinstance(coverage_analysis, dict)
        else []
    )
    output = {}
    if not isinstance(curve, list):
        return output

    for item in curve:
        if not isinstance(item, dict):
            continue
        frame = _valid_frame(item.get("frame"))
        distance = item.get("coverage_distance_px")
        if frame is None or not isinstance(distance, (int, float)):
            continue
        output[frame] = item
    return output


def _seed_frames(
    coverage_analysis: dict[str, Any],
    cfg: AnchorSelectorConfig,
) -> list[int]:
    probe_lookup = _coverage_probe_lookup(coverage_analysis)
    if not probe_lookup:
        return []

    stable = [
        item
        for item in probe_lookup.values()
        if item.get("diagnostic_reliability") != "unstable"
    ]
    if not stable:
        return []

    values = np.asarray(
        [float(item["coverage_distance_px"]) for item in stable],
        dtype=np.float64,
    )
    threshold = float(np.percentile(values, cfg.seed_percentile))

    pool = [
        item
        for item in stable
        if float(item["coverage_distance_px"]) >= threshold
    ]

    recommended = coverage_analysis.get(
        "recommended_anchor_candidates",
        [],
    )
    preferred_frames = {
        int(item["frame"])
        for item in recommended
        if isinstance(item, dict) and _valid_frame(item.get("frame")) is not None
    }

    pool.sort(
        key=lambda item: (
            int(item["frame"]) in preferred_frames,
            bool(item.get("local_peak", False)),
            float(item["coverage_distance_px"]),
        ),
        reverse=True,
    )

    seeds = []
    for item in pool:
        frame = int(item["frame"])

        # Keep one seed per local temporal peak first.  State de-duplication is
        # applied later using the actual probe homographies.
        if any(
            abs(frame - existing) < cfg.min_temporal_separation_frames
            for existing in seeds
        ):
            continue

        seeds.append(frame)
        if len(seeds) >= cfg.seed_pool_size:
            break

    return seeds


def _read_frame(capture: cv2.VideoCapture, frame_index: int):
    capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
    ok, frame = capture.read()
    return frame if ok and frame is not None else None


def _frame_visual_analysis(frame_bgr: np.ndarray) -> dict[str, Any]:
    """Cheap per-frame quality + view descriptor.

    The feature is intentionally appearance-based and match-local.  It is not
    a field detector and does not attempt to infer a homography.
    """
    height, width = frame_bgr.shape[:2]
    if height <= 0 or width <= 0:
        return {
            "quality_score": 0.0,
            "descriptor": None,
            "diagnostics": {},
        }

    scale = min(1.0, 640.0 / width)
    if scale < 1.0:
        image = cv2.resize(
            frame_bgr,
            (
                max(1, int(round(width * scale))),
                max(1, int(round(height * scale))),
            ),
            interpolation=cv2.INTER_AREA,
        )
    else:
        image = frame_bgr

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    # Broad football-pitch green.  This is only a quality/context cue.
    green = (
        (h >= 25)
        & (h <= 95)
        & (s >= 45)
        & (v >= 35)
    ).astype(np.uint8)

    # White/bright markings become more useful when they touch green context.
    white = (
        (s <= 105)
        & (v >= 145)
    ).astype(np.uint8)

    green_dilated = cv2.dilate(
        green,
        np.ones((9, 9), dtype=np.uint8),
        iterations=1,
    )
    line_like = white & green_dilated

    laplacian = cv2.Laplacian(gray, cv2.CV_32F)
    sharpness_raw = float(np.var(laplacian))
    sharpness = float(
        1.0 - math.exp(-sharpness_raw / 220.0)
    )

    green_fraction = float(np.mean(green))
    line_fraction = float(np.mean(line_like))

    # Sobel edge strength inside or immediately around the pitch context.
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(gx, gy)
    context_mask = green_dilated.astype(bool)
    if np.any(context_mask):
        edge_strength_raw = float(
            np.mean(gradient[context_mask])
        )
    else:
        edge_strength_raw = 0.0
    edge_strength = float(
        1.0 - math.exp(-edge_strength_raw / 45.0)
    )

    dark_fraction = float(np.mean(v <= 18))
    clipped_fraction = float(np.mean(v >= 248))
    exposure_quality = max(
        0.0,
        1.0 - 1.8 * dark_fraction - 1.4 * clipped_fraction,
    )

    pitch_quality = min(1.0, green_fraction / 0.42)
    marking_quality = min(1.0, line_fraction / 0.018)

    quality = (
        0.30 * pitch_quality
        + 0.25 * marking_quality
        + 0.22 * sharpness
        + 0.13 * edge_strength
        + 0.10 * exposure_quality
    )
    quality = float(max(0.0, min(1.0, quality)))

    # Coarse appearance descriptor: HSV + pitch + edge energy in a 6x4 grid.
    rows = 4
    columns = 6
    descriptor_parts = []

    gradient_norm = np.clip(gradient / 255.0, 0.0, 1.0)
    image_h, image_w = gray.shape

    for row in range(rows):
        y0 = int(round(row * image_h / rows))
        y1 = int(round((row + 1) * image_h / rows))
        for column in range(columns):
            x0 = int(round(column * image_w / columns))
            x1 = int(round((column + 1) * image_w / columns))

            cell_h = h[y0:y1, x0:x1]
            cell_s = s[y0:y1, x0:x1]
            cell_v = v[y0:y1, x0:x1]
            cell_green = green[y0:y1, x0:x1]
            cell_line = line_like[y0:y1, x0:x1]
            cell_gradient = gradient_norm[y0:y1, x0:x1]

            if cell_h.size == 0:
                descriptor_parts.extend([0.0] * 6)
                continue

            descriptor_parts.extend(
                [
                    float(np.mean(cell_h) / 179.0),
                    float(np.mean(cell_s) / 255.0),
                    float(np.mean(cell_v) / 255.0),
                    float(np.mean(cell_green)),
                    float(np.mean(cell_line)),
                    float(np.mean(cell_gradient)),
                ]
            )

    descriptor = np.asarray(descriptor_parts, dtype=np.float64)
    norm = float(np.linalg.norm(descriptor))
    if norm > 1e-12:
        descriptor /= norm

    return {
        "quality_score": quality,
        "descriptor": descriptor,
        "diagnostics": {
            "green_fraction": green_fraction,
            "line_like_fraction": line_fraction,
            "sharpness": sharpness,
            "edge_strength": edge_strength,
            "exposure_quality": exposure_quality,
        },
    }


def _visual_distance(
    first: np.ndarray | None,
    second: np.ndarray | None,
) -> float | None:
    if first is None or second is None:
        return None
    if first.shape != second.shape:
        return None
    # Unit-normalised descriptor: cosine distance / 2 in roughly [0, 1].
    cosine = float(np.dot(first, second))
    return float(max(0.0, min(1.0, (1.0 - cosine) * 0.5)))


def _view_tags(
    homography: Any,
    image_width: int,
    image_height: int,
    cfg: AnchorSelectorConfig,
) -> dict[str, Any]:
    visibility = compute_visible_pitch_footprint(
        homography,
        image_width,
        image_height,
    )
    if visibility.get("status") != "ok":
        return {
            "side": "desconocida",
            "zoom": "desconocido",
            "visible_fraction": None,
            "visible_lines": 0,
        }

    centroid = visibility.get("world_centroid")
    visible_fraction = visibility.get("visible_fraction")
    visible_lines = visibility.get("visible_lines", [])

    side = "centro"
    if (
        isinstance(centroid, list)
        and len(centroid) >= 1
        and isinstance(centroid[0], (int, float))
    ):
        if centroid[0] < -cfg.left_right_centroid_threshold_m:
            side = "izquierda"
        elif centroid[0] > cfg.left_right_centroid_threshold_m:
            side = "derecha"

    zoom = "medio"
    if isinstance(visible_fraction, (int, float)):
        if visible_fraction >= cfg.zoom_open_visible_fraction:
            zoom = "abierto"
        elif visible_fraction <= cfg.zoom_close_visible_fraction:
            zoom = "cerrado"

    line_names = [
        item.get("semantic")
        for item in visible_lines
        if isinstance(item, dict) and item.get("semantic")
    ]

    return {
        "side": side,
        "zoom": zoom,
        "visible_fraction": visible_fraction,
        "visible_lines": len(line_names),
        "visible_line_semantics": line_names,
    }


def _window_frames(
    seed_frame: int,
    probe_lookup: dict[int, dict[str, Any]],
    cfg: AnchorSelectorConfig,
) -> list[int]:
    candidates = []
    for frame in range(
        max(0, seed_frame - cfg.search_radius_frames),
        seed_frame + cfg.search_radius_frames + 1,
        max(1, cfg.search_step_frames),
    ):
        if frame in probe_lookup:
            candidates.append(frame)

    if seed_frame in probe_lookup and seed_frame not in candidates:
        candidates.append(seed_frame)

    return sorted(set(candidates))


def _normalise_coverage(
    value: float,
    coverage_values: np.ndarray,
) -> float:
    if len(coverage_values) == 0:
        return 0.0
    low = float(np.percentile(coverage_values, 20))
    high = float(np.percentile(coverage_values, 95))
    if high <= low + 1e-9:
        return 1.0
    return float(max(0.0, min(1.0, (value - low) / (high - low))))


def _best_representative_for_seed(
    seed_frame: int,
    *,
    capture: cv2.VideoCapture,
    probe_lookup: dict[int, dict[str, Any]],
    coverage_lookup: dict[int, dict[str, Any]],
    coverage_values: np.ndarray,
    image_width: int,
    image_height: int,
    coverage_config: CameraAtlasCoverageConfig,
    cfg: AnchorSelectorConfig,
) -> dict[str, Any] | None:
    seed_probe = probe_lookup.get(seed_frame)
    if seed_probe is None:
        return None

    seed_h = seed_probe["homography"]
    best = None

    for frame in _window_frames(seed_frame, probe_lookup, cfg):
        probe = probe_lookup[frame]
        coverage_item = coverage_lookup.get(frame)
        if coverage_item is None:
            continue
        if coverage_item.get("diagnostic_reliability") == "unstable":
            continue

        state_from_seed = robust_camera_distance(
            seed_h,
            probe["homography"],
            image_width,
            image_height,
            config=coverage_config,
        )
        if state_from_seed is None:
            continue
        if (
            frame != seed_frame
            and state_from_seed["median_px"]
            > cfg.max_seed_state_distance_px
        ):
            continue

        frame_bgr = _read_frame(capture, frame)
        if frame_bgr is None:
            continue

        visual = _frame_visual_analysis(frame_bgr)
        coverage_distance = float(
            coverage_item["coverage_distance_px"]
        )
        coverage_score = _normalise_coverage(
            coverage_distance,
            coverage_values,
        )

        local_peak = bool(coverage_item.get("local_peak", False))
        reliability = coverage_item.get("diagnostic_reliability")
        reliability_score = (
            1.0 if reliability == "stable"
            else 0.45 if reliability == "uncertain"
            else 0.15
        )

        score = (
            cfg.coverage_weight * coverage_score
            + cfg.visual_quality_weight * visual["quality_score"]
            + cfg.local_peak_bonus * float(local_peak)
            + cfg.reliability_bonus * reliability_score
        )

        # Small preference for staying near the seed state.
        score -= min(
            0.08,
            0.08
            * float(state_from_seed["median_px"])
            / max(cfg.max_seed_state_distance_px, 1.0),
        )

        candidate = {
            "seed_frame": int(seed_frame),
            "frame": int(frame),
            "provenance": probe["provenance"],
            "homography": probe["homography"],
            "coverage_distance_px": coverage_distance,
            "nearest_anchor": coverage_item.get("nearest_anchor"),
            "diagnostic_reliability": reliability,
            "local_peak": local_peak,
            "state_distance_from_seed_px": float(
                state_from_seed["median_px"]
            ),
            "quality_score": float(visual["quality_score"]),
            "visual_diagnostics": visual["diagnostics"],
            "_visual_descriptor": visual["descriptor"],
            "_score": float(score),
            "view": _view_tags(
                probe["homography"],
                image_width,
                image_height,
                cfg,
            ),
        }

        if best is None or candidate["_score"] > best["_score"]:
            best = candidate

    return best


def _isolated_anchor_reviews(
    coverage_analysis: dict[str, Any],
    cfg: AnchorSelectorConfig,
) -> list[dict[str, Any]]:
    output = []
    isolation = (
        coverage_analysis.get("anchor_isolation", [])
        if isinstance(coverage_analysis, dict)
        else []
    )
    if not isinstance(isolation, list):
        return output

    for item in isolation:
        if not isinstance(item, dict):
            continue
        nearest = item.get("nearest_anchor")
        if not isinstance(nearest, dict):
            continue
        distance = nearest.get("median_px")
        if not isinstance(distance, (int, float)):
            continue
        if distance < cfg.isolated_anchor_review_threshold_px:
            continue

        output.append({
            "frame": item.get("frame"),
            "nearest_anchor_frame": nearest.get("frame"),
            "nearest_anchor_distance_px": float(distance),
            "reason": "isolated_existing_human_anchor",
            "action": "review_manual_calibration_or_confirm_unique_view",
        })

    output.sort(
        key=lambda item: item["nearest_anchor_distance_px"],
        reverse=True,
    )
    return output


def select_manual_anchor_frames(
    video_path: str | Path,
    atlas: dict[str, Any],
    coverage_analysis: dict[str, Any],
    propagations: list[dict[str, Any]] | None,
    refined_segments: list[dict[str, Any]] | None,
    image_width: int,
    image_height: int,
    *,
    video_name: str | None = None,
    selector_config: AnchorSelectorConfig | None = None,
    coverage_config: CameraAtlasCoverageConfig | None = None,
) -> dict[str, Any]:
    """Recommend the next HUMAN calibration frames.

    This function reads pixels from a small set of candidate windows only.
    It never edits review data.
    """
    cfg = selector_config or AnchorSelectorConfig()
    distance_cfg = coverage_config or CameraAtlasCoverageConfig()

    path = Path(video_path)
    if not path.exists():
        return {
            "status": "video_not_found",
            "recommendations": [],
            "config": asdict(cfg),
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
            "recommendations": [],
            "config": asdict(cfg),
        }

    probe_lookup = _extract_probe_lookup(
        propagations,
        refined_segments,
        video_name=video_name,
    )
    coverage_lookup = _coverage_probe_lookup(
        coverage_analysis,
    )
    seeds = _seed_frames(coverage_analysis, cfg)

    if not seeds:
        return {
            "status": "no_candidate_seeds",
            "recommendations": [],
            "isolated_anchor_reviews": _isolated_anchor_reviews(
                coverage_analysis,
                cfg,
            ),
            "config": asdict(cfg),
        }

    coverage_values = np.asarray(
        [
            float(item["coverage_distance_px"])
            for item in coverage_lookup.values()
            if isinstance(item.get("coverage_distance_px"), (int, float))
        ],
        dtype=np.float64,
    )

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        return {
            "status": "video_open_failed",
            "recommendations": [],
            "config": asdict(cfg),
        }

    representatives = []

    try:
        for seed in seeds:
            representative = _best_representative_for_seed(
                seed,
                capture=capture,
                probe_lookup=probe_lookup,
                coverage_lookup=coverage_lookup,
                coverage_values=coverage_values,
                image_width=width,
                image_height=height,
                coverage_config=distance_cfg,
                cfg=cfg,
            )
            if representative is not None:
                representatives.append(representative)
    finally:
        capture.release()

    representatives.sort(
        key=lambda item: (
            item["_score"],
            item["coverage_distance_px"],
        ),
        reverse=True,
    )

    selected = []

    for candidate in representatives:
        frame = int(candidate["frame"])
        accept = True

        for chosen in selected:
            chosen_frame = int(chosen["frame"])

            if (
                abs(frame - chosen_frame)
                < cfg.min_temporal_separation_frames
            ):
                accept = False
                break

            state_distance = robust_camera_distance(
                candidate["homography"],
                chosen["homography"],
                width,
                height,
                config=distance_cfg,
            )
            visual_distance = _visual_distance(
                candidate["_visual_descriptor"],
                chosen["_visual_descriptor"],
            )

            state_too_close = (
                state_distance is not None
                and state_distance["median_px"]
                < cfg.min_state_separation_px
            )
            visual_too_close = (
                visual_distance is not None
                and visual_distance
                < cfg.min_visual_descriptor_distance
            )

            # State similarity is primary. Visual similarity makes the
            # duplicate decision more conservative rather than replacing H.
            if state_too_close and visual_too_close:
                accept = False
                break

        if not accept:
            continue

        selected.append(candidate)
        if len(selected) >= cfg.max_recommendations:
            break

    public = []
    for rank, item in enumerate(selected, start=1):
        public_item = {
            key: value
            for key, value in item.items()
            if key not in {
                "homography",
                "_visual_descriptor",
                "_score",
            }
        }
        public_item["rank"] = rank
        public_item["selection_score"] = float(item["_score"])
        public_item["reason"] = (
            "uncovered_camera_state_with_calibratable_frame"
        )
        public.append(public_item)

    strong_anchor_count = (
        int(atlas.get("anchor_count", 0) or 0)
        if isinstance(atlas, dict)
        else 0
    )

    return {
        "status": "ok",
        "policy": {
            "human_only_training": True,
            "auto_frames_are_candidates_only": True,
            "pixels_verified_before_recommendation": True,
            "automatic_promotion": False,
        },
        "strong_anchor_count": strong_anchor_count,
        "seed_count": len(seeds),
        "representative_count": len(representatives),
        "recommendations": public,
        "isolated_anchor_reviews": _isolated_anchor_reviews(
            coverage_analysis,
            cfg,
        ),
        "config": asdict(cfg),
    }


def summarize_anchor_selection(
    selection: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(selection, dict):
        return {"status": "invalid_selection"}

    recommendations = selection.get("recommendations", [])
    reviews = selection.get("isolated_anchor_reviews", [])

    return {
        "status": selection.get("status"),
        "strong_anchor_count": selection.get(
            "strong_anchor_count",
            0,
        ),
        "recommended_count": len(recommendations),
        "recommended_frames": [
            item.get("frame")
            for item in recommendations
            if isinstance(item, dict)
        ],
        "isolated_anchor_review_frames": [
            item.get("frame")
            for item in reviews
            if isinstance(item, dict)
        ],
    }
