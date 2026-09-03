"""Candidate Global Hub Probe V1.

Read-only diagnostic for CAMERA Anchor planning.

The sparse Planner graph deliberately verifies only a few plausible neighbors
per state. That is appropriate for an O(Nk) graph, but it can underestimate a
different property:

    GLOBAL HUB REACH:
    "If this frame became HUMAN, to how many representative keyframes could it
     directly relocalize?"

This module takes only a SHORTLIST of candidate frames and verifies each one
against every trajectory keyframe. Cost is O(MN), where M is deliberately
small. It does NOT need any candidate homography.

It also measures reach over counterfactual conflict evidence from Sparse
Validator, so a candidate that reaches many irrelevant states does not look
equivalent to a candidate that reaches states where current HUMAN predictions
disagree.

No production score/threshold is defined here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import json
import math
import time

import cv2
import numpy as np

from supervisor.field_camera_loop_closure import (
    LoopClosureConfig,
    extract_keyframe_features,
    verify_loop_candidate,
)


EPS = 1e-12


@dataclass(frozen=True)
class CandidateHubProbeConfig:
    analysis_width: int = 960
    minimum_support_for_conflict: int = 2
    progress_every_edges: int = 250


def _load(value: dict[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return json.loads(Path(value).read_text(encoding="utf-8"))


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _percentile_rank(values: list[float]) -> list[float]:
    if not values:
        return []

    array = np.asarray(values, dtype=np.float64)
    order = np.argsort(array, kind="mergesort")
    ranks = np.zeros(len(array), dtype=np.float64)

    pos = 0
    while pos < len(array):
        end = pos + 1
        current = array[order[pos]]

        while (
            end < len(array)
            and np.isclose(
                array[order[end]],
                current,
                rtol=0.0,
                atol=1e-12,
            )
        ):
            end += 1

        average = 0.5 * (pos + end - 1)
        rank = (
            1.0
            if len(array) == 1
            else average / (len(array) - 1)
        )

        for index in order[pos:end]:
            ranks[int(index)] = rank

        pos = end

    return [float(value) for value in ranks]


def _trajectory_frames(v1: dict[str, Any]) -> list[int]:
    frames = []

    for state in v1.get("states", []):
        if not isinstance(state, dict):
            continue
        try:
            frames.append(int(state["frame"]))
        except (KeyError, TypeError, ValueError):
            pass

    return sorted(set(frames))


def _conflict_mass(
    sparse_before: dict[str, Any],
    minimum_support: int,
) -> tuple[dict[int, float], dict[int, dict[str, Any]]]:
    """Empirical pair-evidence mass, no fixed px threshold."""
    pair_rows: list[tuple[int, float]] = []
    target_map: dict[int, dict[str, Any]] = {}

    for target in sparse_before.get("targets", []):
        if not isinstance(target, dict):
            continue

        try:
            frame = int(target["frame"])
            support = int(target.get("support_count") or 0)
        except (KeyError, TypeError, ValueError):
            continue

        target_map[frame] = target

        if target.get("is_exact_human"):
            continue
        if support < int(minimum_support):
            continue

        for pair in target.get("pair_details", []):
            if (
                not isinstance(pair, dict)
                or pair.get("status") != "ok"
            ):
                continue

            value = _finite(pair.get("median_px"))
            if value is None:
                continue

            pair_rows.append((frame, max(0.0, value)))

    if not pair_rows:
        return {}, target_map

    ranks = _percentile_rank(
        [value for _, value in pair_rows]
    )

    mass: dict[int, float] = {}

    for (frame, _), rank in zip(pair_rows, ranks):
        # Tiny floor only prevents the least-disagreeing comparable pair from
        # contributing literal zero. Evidence still remains empirical.
        contribution = 0.04 + 0.96 * rank
        mass[frame] = mass.get(frame, 0.0) + contribution

    return mass, target_map


def _region_map(planner_v2: dict[str, Any]) -> dict[int, int]:
    result: dict[int, int] = {}

    for region in planner_v2.get("recommended_regions", []):
        if not isinstance(region, dict):
            continue

        try:
            region_id = int(
                region.get(
                    "temporal_rank",
                    region.get("region"),
                )
            )
        except (TypeError, ValueError):
            continue

        for frame in region.get("frames", []):
            try:
                result[int(frame)] = region_id
            except (TypeError, ValueError):
                pass

    return result


def _read_frame(
    capture: cv2.VideoCapture,
    frame: int,
) -> np.ndarray | None:
    capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame))
    ok, image = capture.read()
    if not ok or image is None:
        return None
    return image


def _distribution(values: list[float]) -> dict[str, Any]:
    clean = np.asarray(
        [
            float(value)
            for value in values
            if value is not None
            and math.isfinite(float(value))
        ],
        dtype=np.float64,
    )

    if not len(clean):
        return {
            "count": 0,
            "median": None,
            "p75": None,
            "p90": None,
            "max": None,
        }

    return {
        "count": int(len(clean)),
        "median": float(np.median(clean)),
        "p75": float(np.percentile(clean, 75)),
        "p90": float(np.percentile(clean, 90)),
        "max": float(np.max(clean)),
    }


def analyse_candidate_hub_probe(
    video_path: str | Path,
    planner_v1: dict[str, Any] | str | Path,
    sparse_before: dict[str, Any] | str | Path,
    planner_v2: dict[str, Any] | str | Path,
    *,
    candidate_frames: list[int],
    config: CandidateHubProbeConfig | None = None,
) -> dict[str, Any]:
    cfg = config or CandidateHubProbeConfig()

    try:
        v1 = _load(planner_v1)
        sparse = _load(sparse_before)
        regions = _load(planner_v2)
    except (OSError, json.JSONDecodeError) as error:
        return {
            "status": "input_read_failed",
            "error": str(error),
        }

    for label, value in (
        ("planner_v1", v1),
        ("sparse_before", sparse),
        ("planner_v2", regions),
    ):
        if value.get("status") != "ok":
            return {
                "status": f"invalid_{label}",
                "input_status": value.get("status"),
            }

    video = Path(video_path)
    if not video.exists():
        return {
            "status": "video_not_found",
            "video": str(video),
        }

    targets = _trajectory_frames(v1)
    candidates = sorted(
        set(int(frame) for frame in candidate_frames)
    )

    if not targets or not candidates:
        return {"status": "no_frames"}

    conflict_mass, target_map = _conflict_mass(
        sparse,
        cfg.minimum_support_for_conflict,
    )
    frame_to_region = _region_map(regions)

    loop_cfg = LoopClosureConfig(
        analysis_width=cfg.analysis_width
    )

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        return {"status": "video_open_failed"}

    native_width = int(
        capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0
    )
    native_height = int(
        capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0
    )

    unique_frames = sorted(set(targets) | set(candidates))
    features: dict[int, dict[str, Any]] = {}
    failed_feature_frames: list[int] = []

    try:
        for index, frame in enumerate(unique_frames, start=1):
            image = _read_frame(capture, frame)
            if image is None:
                failed_feature_frames.append(frame)
                continue

            feature = extract_keyframe_features(
                image,
                config=loop_cfg,
            )

            if feature.get("status") != "ok":
                failed_feature_frames.append(frame)
                continue

            features[frame] = feature

            if (
                index % 25 == 0
                or index == len(unique_frames)
            ):
                print(
                    f"[hub-probe] features "
                    f"{index}/{len(unique_frames)}"
                )
    finally:
        capture.release()

    total_edges = sum(
        1
        for candidate in candidates
        if candidate in features
        for target in targets
        if target in features
        and target != candidate
    )

    done_edges = 0
    started = time.perf_counter()
    candidate_rows: list[dict[str, Any]] = []

    for candidate in candidates:
        if candidate not in features:
            candidate_rows.append(
                {
                    "frame": candidate,
                    "status": "candidate_features_unavailable",
                }
            )
            continue

        verified_targets: list[int] = []
        conflict_targets: list[int] = []
        conflict_mass_sum = 0.0
        direct_distances_median: list[float] = []
        direct_distances_p90: list[float] = []
        regions_reached: set[int] = set()
        support_patterns: dict[tuple[int, ...], int] = {}

        verification_metrics = {
            "inlier_count": [],
            "inlier_ratio": [],
            "median_reprojection_px": [],
        }

        for target in targets:
            if target == candidate or target not in features:
                continue

            verification = verify_loop_candidate(
                features[candidate],
                features[target],
                native_width=native_width,
                native_height=native_height,
                config=loop_cfg,
            )

            done_edges += 1

            if (
                done_edges % max(
                    1,
                    int(cfg.progress_every_edges),
                )
                == 0
                or done_edges == total_edges
            ):
                elapsed = time.perf_counter() - started
                print(
                    f"[hub-probe] edges "
                    f"{done_edges}/{total_edges}"
                    f" · elapsed={elapsed:.1f}s"
                )

            if verification.get("status") != "ok":
                continue

            verified_targets.append(target)

            region_id = frame_to_region.get(target)
            if region_id is not None:
                regions_reached.add(region_id)

            distance = verification.get(
                "direct_state_distance"
            ) or {}

            median = _finite(distance.get("median_px"))
            p90 = _finite(distance.get("p90_px"))

            if median is not None:
                direct_distances_median.append(median)
            if p90 is not None:
                direct_distances_p90.append(p90)

            for key in verification_metrics:
                value = _finite(verification.get(key))
                if value is not None:
                    verification_metrics[key].append(value)

            if target in conflict_mass:
                conflict_targets.append(target)
                conflict_mass_sum += float(
                    conflict_mass[target]
                )

                target_info = target_map.get(target, {})
                pattern = tuple(
                    sorted(
                        int(frame)
                        for frame in target_info.get(
                            "support_anchor_frames",
                            [],
                        )
                    )
                )
                support_patterns[pattern] = (
                    support_patterns.get(pattern, 0) + 1
                )

        total_target_opportunities = max(
            1,
            sum(
                1
                for target in targets
                if target in features
                and target != candidate
            ),
        )

        total_conflicted = len(conflict_mass)
        total_conflict_mass = float(
            sum(conflict_mass.values())
        )

        candidate_rows.append(
            {
                "frame": candidate,
                "status": "ok",
                "verified_count": len(verified_targets),
                "verified_fraction": (
                    len(verified_targets)
                    / total_target_opportunities
                ),
                "verified_targets": verified_targets,
                "conflicted_verified_count": len(
                    conflict_targets
                ),
                "conflicted_verified_fraction": (
                    len(conflict_targets)
                    / max(total_conflicted, 1)
                ),
                "conflict_mass_reached": float(
                    conflict_mass_sum
                ),
                "conflict_mass_fraction": (
                    conflict_mass_sum
                    / max(total_conflict_mass, EPS)
                ),
                "camera_regions_reached": sorted(
                    int(value)
                    for value in regions_reached
                ),
                "camera_region_count": len(regions_reached),
                "dominant_support_patterns": [
                    {
                        "support_anchor_frames": list(pattern),
                        "count": count,
                    }
                    for pattern, count in sorted(
                        support_patterns.items(),
                        key=lambda item: (
                            -item[1],
                            item[0],
                        ),
                    )
                ][:10],
                "direct_state_median_distribution_px": (
                    _distribution(
                        direct_distances_median
                    )
                ),
                "direct_state_p90_distribution_px": (
                    _distribution(
                        direct_distances_p90
                    )
                ),
                "verification_metrics": {
                    key: _distribution(values)
                    for key, values
                    in verification_metrics.items()
                },
            }
        )

    valid = [
        row
        for row in candidate_rows
        if row.get("status") == "ok"
    ]

    by_global_reach = sorted(
        valid,
        key=lambda row: (
            -row["verified_count"],
            -row["conflict_mass_reached"],
            row["frame"],
        ),
    )

    by_conflict_reach = sorted(
        valid,
        key=lambda row: (
            -row["conflicted_verified_count"],
            -row["conflict_mass_reached"],
            -row["verified_count"],
            row["frame"],
        ),
    )

    by_conflict_mass = sorted(
        valid,
        key=lambda row: (
            -row["conflict_mass_reached"],
            -row["conflicted_verified_count"],
            -row["verified_count"],
            row["frame"],
        ),
    )

    def rank_map(rows):
        return {
            int(row["frame"]): index
            for index, row in enumerate(rows, start=1)
        }

    global_rank = rank_map(by_global_reach)
    conflict_rank = rank_map(by_conflict_reach)
    mass_rank = rank_map(by_conflict_mass)

    for row in valid:
        frame = int(row["frame"])
        row["global_reach_rank"] = global_rank[frame]
        row["conflict_reach_rank"] = conflict_rank[frame]
        row["conflict_mass_rank"] = mass_rank[frame]

    return {
        "status": "ok",
        "version": 1,
        "algorithm": "candidate_global_hub_probe_v1",
        "policy": {
            "candidate_homographies_used": False,
            "human_labels_used_for_candidate_geometry": False,
            "global_direct_verification": True,
            "production_score_defined": False,
            "conflict_threshold_px": None,
        },
        "config": asdict(cfg),
        "candidate_frames": candidates,
        "target_count": len(targets),
        "conflicted_target_count": len(conflict_mass),
        "failed_feature_frames": failed_feature_frames,
        "candidate_results": candidate_rows,
        "by_global_reach": by_global_reach,
        "by_conflict_reach": by_conflict_reach,
        "by_conflict_mass": by_conflict_mass,
    }
