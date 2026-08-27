"""Dense Camera Atlas V1.

Turn directly relocalized Camera-Atlas keyframes into a dense per-frame pitch
homography sequence.

Architecture
------------
1. HUMAN anchors are exact pitch calibrations.
2. Anchor Relocalization directly transfers those calibrations to sparse camera
   keyframes anywhere in the video.
3. Those relocalized keyframes are SOFT anchors: useful geometric observations,
   but never HUMAN truth and never training labels.
4. The already validated Camera H_t planar-flow bridge/fusion fills every frame
   between consecutive soft anchors.

CAT02 change
------------
The one-sided start/end tails now reuse the validated cached directional-flow
implementation from `field_camera_state_pipeline`.

CAT01 called `propagate_pitch_homography(anchor -> target)` independently for
every tail target. That repeatedly recomputed all prefixes and was O(n^2) in
the tail length.

CAT02 builds the directional path once and then queries cached prefixes, making
tail recovery approximately O(n).

No camera mathematics or provenance policy has changed.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np

from supervisor.field_camera_state_pipeline import (
    _CachedDirectionalFlow,
    _IntervalFrameCache,
    _camera_ht_between_anchors,
)
from supervisor.field_camera_state_planar_flow import PlanarFlowConfig


@dataclass(frozen=True)
class DenseCameraAtlasConfig:
    minimum_soft_anchor_confidence: str = "verified"
    tail_flow_step_frames: int = 2


def _normalise_h(value: Any) -> np.ndarray | None:
    try:
        h = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if h.shape != (3, 3) or not np.isfinite(h).all():
        return None
    if abs(float(h[2, 2])) < 1e-9:
        return None
    h = h / float(h[2, 2])
    try:
        det = float(np.linalg.det(h))
        cond = float(np.linalg.cond(h))
    except np.linalg.LinAlgError:
        return None
    if (
        not np.isfinite(det)
        or abs(det) < 1e-12
        or not np.isfinite(cond)
        or cond > 1e14
    ):
        return None
    return h


def _confidence_rank(value: str | None) -> int:
    if value == "human":
        return 3
    if value == "high":
        return 2
    if value == "verified":
        return 1
    return 0


def _valid_soft_anchors(
    anchors: list[dict[str, Any]],
    *,
    config: DenseCameraAtlasConfig,
) -> list[dict[str, Any]]:
    minimum = _confidence_rank(config.minimum_soft_anchor_confidence)
    output = []

    for anchor in anchors:
        try:
            frame = int(anchor["frame"])
        except (KeyError, TypeError, ValueError):
            continue

        h = _normalise_h(anchor.get("homography"))
        if h is None:
            continue

        tier = str(anchor.get("confidence_tier") or "verified")
        if _confidence_rank(tier) < minimum:
            continue

        output.append(
            {
                **anchor,
                "frame": frame,
                "homography": h.tolist(),
                "confidence_tier": tier,
            }
        )

    deduplicated: dict[int, dict[str, Any]] = {}
    for anchor in output:
        frame = int(anchor["frame"])
        current = deduplicated.get(frame)
        if current is None:
            deduplicated[frame] = anchor
            continue
        if _confidence_rank(anchor["confidence_tier"]) > _confidence_rank(
            current["confidence_tier"]
        ):
            deduplicated[frame] = anchor

    return sorted(
        deduplicated.values(),
        key=lambda item: int(item["frame"]),
    )


def _soft_anchor_record(anchor: dict[str, Any]) -> dict[str, Any]:
    return {
        "frame": int(anchor["frame"]),
        "homography": anchor["homography"],
        "source": "camera_atlas_keyframe_v1",
        "purpose": "derived_camera",
        "method": "direct_human_anchor_relocalization",
        "confidence_tier": anchor.get("confidence_tier", "verified"),
        "origin_human_anchor_frame": anchor.get("anchor_frame"),
        "relocalization": {
            "score": anchor.get("score"),
            "match_count": anchor.get("match_count"),
            "inlier_count": anchor.get("inlier_count"),
            "inlier_ratio": anchor.get("inlier_ratio"),
            "median_reprojection_px": anchor.get(
                "median_reprojection_px"
            ),
            "direct_state_distance": anchor.get(
                "direct_state_distance"
            ),
        },
        "policy": {
            "is_human_anchor": False,
            "may_train_camera_state": False,
            "may_seed_dense_camera_ht": True,
        },
    }


def _dense_record(
    record: dict[str, Any],
    *,
    left_anchor: dict[str, Any],
    right_anchor: dict[str, Any],
) -> dict[str, Any]:
    output = dict(record)
    output["source"] = "camera_atlas_ht_v1"
    output["purpose"] = "derived_camera"
    output["method"] = f"atlas_{record.get('method', 'camera_ht')}"
    output["atlas_left_keyframe"] = int(left_anchor["frame"])
    output["atlas_right_keyframe"] = int(right_anchor["frame"])
    output["atlas_left_confidence"] = left_anchor.get(
        "confidence_tier",
        "verified",
    )
    output["atlas_right_confidence"] = right_anchor.get(
        "confidence_tier",
        "verified",
    )
    output["atlas_left_human_origin"] = left_anchor.get("anchor_frame")
    output["atlas_right_human_origin"] = right_anchor.get(
        "anchor_frame"
    )
    output["policy"] = {
        "is_human_anchor": False,
        "may_train_camera_state": False,
        "uses_only_derived_keyframe_endpoints": True,
    }
    return output


def _tail_record(
    *,
    frame: int,
    path: dict[str, Any],
    anchor: dict[str, Any],
) -> dict[str, Any]:
    return {
        "frame": int(frame),
        "homography": path["homography"],
        "source": "camera_atlas_ht_v1",
        "purpose": "derived_camera",
        "method": "atlas_one_sided_tail_flow_cached",
        "confidence_tier": anchor.get(
            "confidence_tier",
            "verified",
        ),
        "atlas_anchor_frame": int(anchor["frame"]),
        "atlas_human_origin": anchor.get("anchor_frame"),
        "flow_quality": {
            "step_count": path.get("step_count"),
            "median_inlier_ratio": path.get(
                "median_inlier_ratio"
            ),
            "median_reprojection_px": path.get(
                "median_reprojection_px"
            ),
        },
        "policy": {
            "is_human_anchor": False,
            "may_train_camera_state": False,
        },
    }


def _one_sided_tail(
    video_path: str | Path,
    *,
    anchor: dict[str, Any],
    target_frames: list[int],
    pitch_length_m: float,
    pitch_width_m: float,
    config: DenseCameraAtlasConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Recover one tail using one cached directional-flow path.

    This preserves the same PlanarFlowConfig and the same per-target path
    semantics as CAT01, but avoids re-decoding/re-estimating the common prefix
    independently for every target.
    """

    if not target_frames:
        return [], []

    derived: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    flow_cfg = PlanarFlowConfig(
        sample_step_frames=max(
            1,
            int(config.tail_flow_step_frames),
        )
    )
    anchor_frame = int(anchor["frame"])

    ordered_targets = sorted(
        {int(frame) for frame in target_frames},
        key=lambda frame: abs(frame - anchor_frame),
    )

    furthest_target = max(
        ordered_targets,
        key=lambda frame: abs(frame - anchor_frame),
    )

    frame_cache = _IntervalFrameCache(video_path)
    try:
        cached_flow = _CachedDirectionalFlow(
            frame_cache,
            anchor_frame,
            furthest_target,
            anchor["homography"],
            pitch_length_m=pitch_length_m,
            pitch_width_m=pitch_width_m,
            config=flow_cfg,
        )

        for frame in ordered_targets:
            path = cached_flow.path_to(frame)

            if (
                path.get("status") != "ok"
                or not isinstance(path.get("homography"), list)
            ):
                failures.append(
                    {
                        "frame": int(frame),
                        "reason": "atlas_tail_flow_failed",
                        "anchor_frame": anchor_frame,
                        "flow_status": path.get("status"),
                    }
                )
                continue

            derived.append(
                _tail_record(
                    frame=frame,
                    path=path,
                    anchor=anchor,
                )
            )
    finally:
        frame_cache.close()

    derived.sort(key=lambda record: int(record["frame"]))
    failures.sort(key=lambda record: int(record["frame"]))
    return derived, failures


def build_dense_camera_atlas(
    video_path: str | Path,
    *,
    soft_anchors: list[dict[str, Any]],
    image_width: int,
    image_height: int,
    pitch_length_m: float,
    pitch_width_m: float,
    start_frame: int,
    end_frame: int,
    config: DenseCameraAtlasConfig | None = None,
    progress_callback=None,
) -> dict[str, Any]:
    """Densify relocalized keyframes into per-frame pitch homographies."""

    cfg = config or DenseCameraAtlasConfig()
    anchors = _valid_soft_anchors(
        soft_anchors,
        config=cfg,
    )

    if not anchors:
        return {"status": "no_valid_soft_anchors"}

    start = int(start_frame)
    end = int(end_frame)
    if end < start:
        return {"status": "invalid_frame_range"}

    records_by_frame: dict[int, dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []

    for anchor in anchors:
        frame = int(anchor["frame"])
        if start <= frame <= end:
            records_by_frame[frame] = _soft_anchor_record(anchor)

    interval_count = max(0, len(anchors) - 1)
    for index, (left, right) in enumerate(
        zip(anchors[:-1], anchors[1:])
    ):
        left_frame = int(left["frame"])
        right_frame = int(right["frame"])

        if right_frame <= start or left_frame >= end:
            continue

        if right_frame <= left_frame + 1:
            if progress_callback is not None:
                progress_callback(
                    "intervals",
                    index + 1,
                    interval_count,
                    left_frame,
                    right_frame,
                    len(records_by_frame),
                )
            continue

        derived, interval_failures = _camera_ht_between_anchors(
            video_path,
            left_frame,
            left["homography"],
            right_frame,
            right["homography"],
            image_width=image_width,
            image_height=image_height,
            pitch_length_m=pitch_length_m,
            pitch_width_m=pitch_width_m,
        )

        for record in derived:
            frame = int(record["frame"])
            if start <= frame <= end:
                records_by_frame[frame] = _dense_record(
                    record,
                    left_anchor=left,
                    right_anchor=right,
                )

        for failure in interval_failures:
            frame = int(failure.get("frame", -1))
            if start <= frame <= end:
                failures.append(
                    {
                        **failure,
                        "atlas_left_keyframe": left_frame,
                        "atlas_right_keyframe": right_frame,
                    }
                )

        if progress_callback is not None:
            progress_callback(
                "intervals",
                index + 1,
                interval_count,
                left_frame,
                right_frame,
                len(records_by_frame),
            )

    first_anchor = anchors[0]
    first_anchor_frame = int(first_anchor["frame"])
    if start < first_anchor_frame:
        target_frames = list(
            range(start, first_anchor_frame)
        )
        tail, tail_failures = _one_sided_tail(
            video_path,
            anchor=first_anchor,
            target_frames=target_frames,
            pitch_length_m=pitch_length_m,
            pitch_width_m=pitch_width_m,
            config=cfg,
        )
        for record in tail:
            records_by_frame[int(record["frame"])] = record
        failures.extend(tail_failures)

        if progress_callback is not None:
            progress_callback(
                "tail_start",
                len(target_frames),
                len(target_frames),
                start,
                first_anchor_frame,
                len(records_by_frame),
            )

    last_anchor = anchors[-1]
    last_anchor_frame = int(last_anchor["frame"])
    if end > last_anchor_frame:
        target_frames = list(
            range(last_anchor_frame + 1, end + 1)
        )
        tail, tail_failures = _one_sided_tail(
            video_path,
            anchor=last_anchor,
            target_frames=target_frames,
            pitch_length_m=pitch_length_m,
            pitch_width_m=pitch_width_m,
            config=cfg,
        )
        for record in tail:
            records_by_frame[int(record["frame"])] = record
        failures.extend(tail_failures)

        if progress_callback is not None:
            progress_callback(
                "tail_end",
                len(target_frames),
                len(target_frames),
                last_anchor_frame,
                end,
                len(records_by_frame),
            )

    expected_frames = end - start + 1
    records = [
        records_by_frame[frame]
        for frame in sorted(records_by_frame)
        if start <= frame <= end
    ]
    missing = [
        frame
        for frame in range(start, end + 1)
        if frame not in records_by_frame
    ]

    source_counts: dict[str, int] = {}
    for record in records:
        source = str(record.get("source") or "unknown")
        source_counts[source] = (
            source_counts.get(source, 0) + 1
        )

    high_endpoint_frames = sum(
        1
        for record in records
        if (
            record.get("confidence_tier") == "high"
            or (
                record.get("atlas_left_confidence") == "high"
                and record.get("atlas_right_confidence") == "high"
            )
        )
    )

    return {
        "status": "ok",
        "version": 1,
        "algorithm": "dense_camera_atlas_v1",
        "implementation": "CAT02_cached_tails",
        "config": asdict(cfg),
        "frame_range": {
            "start": start,
            "end": end,
            "expected_count": expected_frames,
        },
        "soft_anchor_count": len(anchors),
        "derived_frame_count": len(records),
        "coverage_fraction": len(records)
        / max(float(expected_frames), 1.0),
        "missing_frame_count": len(missing),
        "high_endpoint_frame_count": high_endpoint_frames,
        "source_counts": source_counts,
        "soft_anchors": anchors,
        "records": records,
        "failures": failures,
        "missing_frames": missing,
    }
