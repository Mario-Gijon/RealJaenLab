"""Camera Atlas coverage diagnostics for RealJaen FIELD.

This module measures how well the strong HUMAN Camera Atlas covers the camera
states seen in legacy propagated/refined material.

Hard rule:
    HUMAN anchors define the atlas.
    Propagated/refined frames are probes only.
    Probe homographies never become training truth and never modify the atlas.

The main outputs are:
- a robust camera-state distance that does not let geometry many screens outside
  the viewport dominate;
- nearest HUMAN anchor for every diagnostic probe;
- temporal-consistency flags for potentially broken legacy propagation;
- robust anchor-to-anchor isolation diagnostics;
- a small set of recommended frames where a new HUMAN anchor would add the most
  camera-state coverage.

This is offline/read-only analysis.  It does not decode video, detect lines,
track motion, persist reviews or estimate production per-frame homographies.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Iterable

import numpy as np


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
class CameraAtlasCoverageConfig:
    # Only support points lying in this expanded viewport for at least one of
    # the two cameras participate in the primary distance.
    extended_viewport_margin: float = 0.75
    minimum_support_points: int = 6

    # Even in fallback mode, coordinates are winsorised so a point projected
    # ten screens away cannot dominate a camera-state distance.
    fallback_clip_margin: float = 1.50

    # Candidate selection: avoid asking the user to calibrate several adjacent
    # frames from the same temporal gap or the same camera-state hole.
    recommended_candidates: int = 6
    candidate_min_frame_separation: int = 30
    candidate_min_state_separation_px: float = 90.0

    # Local maxima are preferred, but broad uncovered plateaus still need a
    # representative candidate.
    candidate_percentile: float = 70.0

    # Diagnostic continuity only. Threshold is learned from probe transitions
    # using median/MAD, with this absolute floor preventing an unrealistically
    # tiny threshold on almost-static synthetic data.
    continuity_mad_multiplier: float = 6.0
    continuity_min_outlier_px_per_frame: float = 45.0

    # Reporting thresholds; these do not affect training or candidate choice.
    coverage_thresholds_px: tuple[float, ...] = (50.0, 100.0, 200.0, 400.0)


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
    except np.linalg.LinAlgError:
        return None

    if not np.isfinite(determinant) or abs(determinant) < 1e-12:
        return None
    return h


def _project_points(points_xy: np.ndarray, h: np.ndarray):
    points = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    homogeneous = np.column_stack((points, np.ones(len(points))))
    projected_h = homogeneous @ h.T
    denominator = projected_h[:, 2]

    valid = (
        np.isfinite(projected_h).all(axis=1)
        & np.isfinite(denominator)
        & (np.abs(denominator) > 1e-10)
    )

    projected = np.full((len(points), 2), np.nan, dtype=np.float64)
    projected[valid] = (
        projected_h[valid, :2] / denominator[valid, None]
    )
    valid &= np.isfinite(projected).all(axis=1)
    return projected, valid


def _normalised_projection(
    homography: Any,
    image_width: int,
    image_height: int,
):
    h = _as_h(homography)
    if h is None:
        return None, None

    projected, valid = _project_points(_SIGNATURE_WORLD, h)
    normalised = np.full_like(projected, np.nan)

    width = max(float(image_width), 1.0)
    height = max(float(image_height), 1.0)

    normalised[valid, 0] = projected[valid, 0] / width
    normalised[valid, 1] = projected[valid, 1] / height
    return normalised, valid


def robust_camera_distance(
    first_homography: Any,
    second_homography: Any,
    image_width: int,
    image_height: int,
    *,
    config: CameraAtlasCoverageConfig | None = None,
) -> dict[str, Any] | None:
    """Robust camera-state distance measured on canonical pitch projections.

    Primary support is restricted to geometry visible or reasonably near the
    viewport in at least one camera. This avoids distant extrapolated geometry
    dominating the metric.

    Returned distances are in image-pixel units:
        sqrt((dx_norm * width)^2 + (dy_norm * height)^2)
    """
    cfg = config or CameraAtlasCoverageConfig()

    first, first_valid = _normalised_projection(
        first_homography,
        image_width,
        image_height,
    )
    second, second_valid = _normalised_projection(
        second_homography,
        image_width,
        image_height,
    )
    if first is None or second is None:
        return None

    common = first_valid & second_valid
    if int(np.count_nonzero(common)) < cfg.minimum_support_points:
        return None

    margin = float(cfg.extended_viewport_margin)

    def near_viewport(points: np.ndarray) -> np.ndarray:
        return (
            (points[:, 0] >= -margin)
            & (points[:, 0] <= 1.0 + margin)
            & (points[:, 1] >= -margin)
            & (points[:, 1] <= 1.0 + margin)
        )

    support = common & (near_viewport(first) | near_viewport(second))
    used_fallback = False

    if int(np.count_nonzero(support)) < cfg.minimum_support_points:
        support = common
        used_fallback = True

    clip_margin = (
        float(cfg.fallback_clip_margin)
        if used_fallback
        else margin
    )

    lower = -clip_margin
    upper = 1.0 + clip_margin

    first_clipped = np.clip(first[support], lower, upper)
    second_clipped = np.clip(second[support], lower, upper)

    delta = first_clipped - second_clipped
    dx = delta[:, 0] * float(image_width)
    dy = delta[:, 1] * float(image_height)
    distances = np.sqrt(dx * dx + dy * dy)

    if len(distances) == 0 or not np.isfinite(distances).all():
        return None

    return {
        "median_px": float(np.median(distances)),
        "p90_px": float(np.percentile(distances, 90)),
        "max_px": float(np.max(distances)),
        "support_points": int(len(distances)),
        "support_fraction": float(
            len(distances) / len(_SIGNATURE_WORLD)
        ),
        "used_fallback_support": bool(used_fallback),
    }


def _valid_frame(frame: Any) -> int | None:
    try:
        value = int(frame)
    except (TypeError, ValueError):
        return None
    return value if value >= 0 else None


def _extract_frames_from_container(
    containers: Iterable[dict[str, Any]] | None,
    *,
    provenance: str,
    priority: int,
    video_name: str | None,
) -> list[dict[str, Any]]:
    output = []

    for container_index, container in enumerate(containers or []):
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

            output.append({
                "frame": frame,
                "homography": h.tolist(),
                "provenance": provenance,
                "_priority": priority,
                "_container_index": container_index,
            })

    return output


def collect_diagnostic_probes(
    propagations: list[dict[str, Any]] | None,
    refined_segments: list[dict[str, Any]] | None,
    *,
    video_name: str | None = None,
    strong_anchor_frames: set[int] | None = None,
) -> list[dict[str, Any]]:
    """Collect legacy frames as probes, never as atlas training examples.

    If a frame exists in both sources, refined wins over propagated only for the
    diagnostic probe representation.
    """
    strong_frames = strong_anchor_frames or set()

    records = []
    records.extend(
        _extract_frames_from_container(
            propagations,
            provenance="propagated",
            priority=1,
            video_name=video_name,
        )
    )
    records.extend(
        _extract_frames_from_container(
            refined_segments,
            provenance="refined",
            priority=2,
            video_name=video_name,
        )
    )

    by_frame: dict[int, dict[str, Any]] = {}

    for record in records:
        frame = int(record["frame"])
        if frame in strong_frames:
            continue

        current = by_frame.get(frame)
        if (
            current is None
            or int(record["_priority"]) > int(current["_priority"])
        ):
            by_frame[frame] = record

    output = []
    for frame in sorted(by_frame):
        record = dict(by_frame[frame])
        record.pop("_priority", None)
        record.pop("_container_index", None)
        output.append(record)

    return output


def _strong_anchors(atlas: dict[str, Any]) -> list[dict[str, Any]]:
    anchors = atlas.get("anchors", []) if isinstance(atlas, dict) else []
    output = []

    if not isinstance(anchors, list):
        return output

    for anchor in anchors:
        if not isinstance(anchor, dict):
            continue
        frame = _valid_frame(anchor.get("frame"))
        h = _as_h(anchor.get("homography"))
        if frame is None or h is None:
            continue
        output.append({
            "frame": frame,
            "homography": h.tolist(),
        })

    output.sort(key=lambda item: item["frame"])
    return output


def _nearest_anchor(
    homography: Any,
    anchors: list[dict[str, Any]],
    image_width: int,
    image_height: int,
    cfg: CameraAtlasCoverageConfig,
):
    candidates = []

    for anchor in anchors:
        distance = robust_camera_distance(
            homography,
            anchor["homography"],
            image_width,
            image_height,
            config=cfg,
        )
        if distance is None:
            continue
        candidates.append({
            "frame": int(anchor["frame"]),
            **distance,
        })

    candidates.sort(key=lambda item: item["median_px"])

    return {
        "nearest": candidates[0] if candidates else None,
        "second_nearest": candidates[1] if len(candidates) > 1 else None,
        "all": candidates,
    }


def _anchor_isolation(
    anchors: list[dict[str, Any]],
    image_width: int,
    image_height: int,
    cfg: CameraAtlasCoverageConfig,
):
    output = []

    for index, anchor in enumerate(anchors):
        candidates = []
        for other_index, other in enumerate(anchors):
            if index == other_index:
                continue

            distance = robust_camera_distance(
                anchor["homography"],
                other["homography"],
                image_width,
                image_height,
                config=cfg,
            )
            if distance is None:
                continue

            candidates.append({
                "frame": int(other["frame"]),
                **distance,
            })

        candidates.sort(key=lambda item: item["median_px"])

        output.append({
            "frame": int(anchor["frame"]),
            "nearest_anchor": candidates[0] if candidates else None,
        })

    return output


def _continuity_diagnostics(
    probes: list[dict[str, Any]],
    image_width: int,
    image_height: int,
    cfg: CameraAtlasCoverageConfig,
):
    transitions = []

    for left, right in zip(probes, probes[1:]):
        frame_gap = int(right["frame"] - left["frame"])
        if frame_gap <= 0:
            continue

        distance = robust_camera_distance(
            left["homography"],
            right["homography"],
            image_width,
            image_height,
            config=cfg,
        )
        if distance is None:
            continue

        transitions.append({
            "frame_a": int(left["frame"]),
            "frame_b": int(right["frame"]),
            "frame_gap": frame_gap,
            "median_px": float(distance["median_px"]),
            "median_px_per_frame": float(
                distance["median_px"] / frame_gap
            ),
        })

    speeds = np.asarray(
        [item["median_px_per_frame"] for item in transitions],
        dtype=np.float64,
    )

    if len(speeds) == 0:
        threshold = float(cfg.continuity_min_outlier_px_per_frame)
        median_speed = None
        mad = None
    else:
        median_speed = float(np.median(speeds))
        mad = float(np.median(np.abs(speeds - median_speed)))
        robust_sigma = 1.4826 * mad
        threshold = max(
            float(cfg.continuity_min_outlier_px_per_frame),
            median_speed
            + cfg.continuity_mad_multiplier * robust_sigma,
        )

    transition_outlier_by_pair = {}
    for transition in transitions:
        is_outlier = (
            transition["median_px_per_frame"] > threshold
        )
        transition["outlier"] = bool(is_outlier)
        transition_outlier_by_pair[
            (transition["frame_a"], transition["frame_b"])
        ] = bool(is_outlier)

    reliability = {}

    for index, probe in enumerate(probes):
        frame = int(probe["frame"])
        adjacent_flags = []

        if index > 0:
            previous_frame = int(probes[index - 1]["frame"])
            key = (previous_frame, frame)
            if key in transition_outlier_by_pair:
                adjacent_flags.append(
                    transition_outlier_by_pair[key]
                )

        if index + 1 < len(probes):
            next_frame = int(probes[index + 1]["frame"])
            key = (frame, next_frame)
            if key in transition_outlier_by_pair:
                adjacent_flags.append(
                    transition_outlier_by_pair[key]
                )

        if not adjacent_flags:
            status = "unknown"
        elif all(adjacent_flags):
            status = "unstable"
        elif any(adjacent_flags):
            status = "uncertain"
        else:
            status = "stable"

        reliability[frame] = status

    return {
        "median_speed_px_per_frame": median_speed,
        "mad_speed_px_per_frame": mad,
        "outlier_threshold_px_per_frame": float(threshold),
        "transitions": transitions,
        "reliability_by_frame": reliability,
    }


def _coverage_distribution(
    probe_results: list[dict[str, Any]],
    cfg: CameraAtlasCoverageConfig,
):
    values = np.asarray(
        [
            item["coverage_distance_px"]
            for item in probe_results
            if isinstance(item.get("coverage_distance_px"), (int, float))
        ],
        dtype=np.float64,
    )

    if len(values) == 0:
        return {
            "count": 0,
            "min_px": None,
            "median_px": None,
            "p90_px": None,
            "max_px": None,
            "threshold_coverage": {},
        }

    threshold_coverage = {}
    for threshold in cfg.coverage_thresholds_px:
        threshold_coverage[str(int(threshold))] = float(
            np.mean(values <= threshold)
        )

    return {
        "count": int(len(values)),
        "min_px": float(np.min(values)),
        "median_px": float(np.median(values)),
        "p90_px": float(np.percentile(values, 90)),
        "max_px": float(np.max(values)),
        "threshold_coverage": threshold_coverage,
    }


def _is_local_peak(
    probe_results: list[dict[str, Any]],
    index: int,
) -> bool:
    value = probe_results[index].get("coverage_distance_px")
    if not isinstance(value, (int, float)):
        return False

    left = (
        probe_results[index - 1].get("coverage_distance_px")
        if index > 0
        else None
    )
    right = (
        probe_results[index + 1].get("coverage_distance_px")
        if index + 1 < len(probe_results)
        else None
    )

    left_ok = not isinstance(left, (int, float)) or value >= left
    right_ok = not isinstance(right, (int, float)) or value >= right
    return bool(left_ok and right_ok)


def _candidate_frames(
    probe_results: list[dict[str, Any]],
    image_width: int,
    image_height: int,
    cfg: CameraAtlasCoverageConfig,
):
    valid_values = np.asarray(
        [
            item["coverage_distance_px"]
            for item in probe_results
            if (
                isinstance(item.get("coverage_distance_px"), (int, float))
                and item.get("diagnostic_reliability") != "unstable"
            )
        ],
        dtype=np.float64,
    )

    if len(valid_values) == 0:
        return []

    threshold = float(
        np.percentile(valid_values, cfg.candidate_percentile)
    )

    pool = []

    for index, item in enumerate(probe_results):
        distance = item.get("coverage_distance_px")
        if not isinstance(distance, (int, float)):
            continue
        if item.get("diagnostic_reliability") == "unstable":
            continue
        if distance < threshold:
            continue

        pool.append({
            **item,
            "_local_peak": _is_local_peak(probe_results, index),
        })

    pool.sort(
        key=lambda item: (
            bool(item["_local_peak"]),
            float(item["coverage_distance_px"]),
        ),
        reverse=True,
    )

    selected = []

    for candidate in pool:
        frame = int(candidate["frame"])

        if any(
            abs(frame - int(chosen["frame"]))
            < cfg.candidate_min_frame_separation
            for chosen in selected
        ):
            continue

        too_similar_state = False
        for chosen in selected:
            state_distance = robust_camera_distance(
                candidate["homography"],
                chosen["homography"],
                image_width,
                image_height,
                config=cfg,
            )
            if (
                state_distance is not None
                and state_distance["median_px"]
                < cfg.candidate_min_state_separation_px
            ):
                too_similar_state = True
                break

        if too_similar_state:
            continue

        selected_record = dict(candidate)
        selected_record["local_peak"] = bool(candidate["_local_peak"])
        selected_record["reason"] = "large_camera_state_coverage_gap"
        selected.append(selected_record)

        if len(selected) >= cfg.recommended_candidates:
            break

    return [
        {
            key: value
            for key, value in item.items()
            if key not in {"homography", "_local_peak"}
        }
        for item in selected
    ]


def analyze_camera_atlas_coverage(
    atlas: dict[str, Any],
    propagations: list[dict[str, Any]] | None,
    refined_segments: list[dict[str, Any]] | None,
    image_width: int,
    image_height: int,
    *,
    video_name: str | None = None,
    config: CameraAtlasCoverageConfig | None = None,
) -> dict[str, Any]:
    """Measure HUMAN-atlas coverage using legacy auto frames as probes only."""
    cfg = config or CameraAtlasCoverageConfig()

    try:
        width = int(image_width)
        height = int(image_height)
    except (TypeError, ValueError):
        width = 0
        height = 0

    if width <= 0 or height <= 0:
        return {
            "status": "invalid_image_size",
            "config": asdict(cfg),
        }

    anchors = _strong_anchors(atlas)

    if not anchors:
        return {
            "status": "no_strong_anchors",
            "anchor_count": 0,
            "probe_count": 0,
            "config": asdict(cfg),
        }

    strong_frames = {int(anchor["frame"]) for anchor in anchors}

    probes = collect_diagnostic_probes(
        propagations,
        refined_segments,
        video_name=video_name,
        strong_anchor_frames=strong_frames,
    )

    continuity = _continuity_diagnostics(
        probes,
        width,
        height,
        cfg,
    )

    probe_results = []

    for probe in probes:
        nearest = _nearest_anchor(
            probe["homography"],
            anchors,
            width,
            height,
            cfg,
        )

        nearest_anchor = nearest["nearest"]
        second_anchor = nearest["second_nearest"]

        nearest_distance = (
            float(nearest_anchor["median_px"])
            if nearest_anchor is not None
            else None
        )

        probe_results.append({
            "frame": int(probe["frame"]),
            "provenance": probe["provenance"],
            "homography": probe["homography"],
            "coverage_distance_px": nearest_distance,
            "nearest_anchor": nearest_anchor,
            "second_nearest_anchor": second_anchor,
            "diagnostic_reliability": continuity[
                "reliability_by_frame"
            ].get(int(probe["frame"]), "unknown"),
        })

    distribution = _coverage_distribution(
        probe_results,
        cfg,
    )

    candidates = _candidate_frames(
        probe_results,
        width,
        height,
        cfg,
    )

    anchor_isolation = _anchor_isolation(
        anchors,
        width,
        height,
        cfg,
    )

    public_probes = [
        {
            key: value
            for key, value in item.items()
            if key != "homography"
        }
        for item in probe_results
    ]

    return {
        "status": "ok",
        "policy": {
            "human_anchors_define_atlas": True,
            "propagated_is_probe_only": True,
            "refined_is_probe_only": True,
            "probe_frames_promoted_automatically": False,
        },
        "anchor_count": len(anchors),
        "probe_count": len(probes),
        "anchor_isolation": anchor_isolation,
        "coverage": distribution,
        "recommended_anchor_candidates": candidates,
        "probe_curve": public_probes,
        "continuity": {
            key: value
            for key, value in continuity.items()
            if key != "reliability_by_frame"
        },
        "config": asdict(cfg),
    }


def summarize_camera_atlas_coverage(
    analysis: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(analysis, dict):
        return {"status": "invalid_analysis"}

    coverage = analysis.get("coverage", {})
    candidates = analysis.get(
        "recommended_anchor_candidates",
        [],
    )
    anchor_isolation = analysis.get("anchor_isolation", [])

    worst_anchor = None
    valid_isolation = [
        item
        for item in anchor_isolation
        if isinstance(
            item.get("nearest_anchor", {}).get("median_px")
            if isinstance(item.get("nearest_anchor"), dict)
            else None,
            (int, float),
        )
    ]
    if valid_isolation:
        worst_anchor = max(
            valid_isolation,
            key=lambda item: item["nearest_anchor"]["median_px"],
        )

    return {
        "status": analysis.get("status"),
        "anchor_count": analysis.get("anchor_count", 0),
        "probe_count": analysis.get("probe_count", 0),
        "coverage_median_px": coverage.get("median_px"),
        "coverage_p90_px": coverage.get("p90_px"),
        "coverage_max_px": coverage.get("max_px"),
        "threshold_coverage": coverage.get(
            "threshold_coverage",
            {},
        ),
        "recommended_frames": [
            item.get("frame") for item in candidates
        ],
        "worst_isolated_anchor": worst_anchor,
    }
