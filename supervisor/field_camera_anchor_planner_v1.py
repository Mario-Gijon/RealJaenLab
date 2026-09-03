"""Offline CAMERA Anchor Planner V1.

Experimental/read-only planner. It does NOT consume HUMAN homographies, Dense,
Relocalization outputs or review calibrations.

It uses:
- Camera Trajectory keyframes as the planning universe.
- FIELD view geometry descriptors as cheap neighborhood proposals.
- Direct SIFT + robust projective verification as sparse graph edges.

The returned 1x/2x plans are preflight HUMAN proposals, not CAMERA acceptance.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import json
import math

import cv2
import numpy as np

from supervisor.field_view_scanner import (
    FieldViewScannerConfig,
    _descriptor as field_view_descriptor,
    field_view_distance,
)
from supervisor.field_camera_loop_closure import (
    LoopClosureConfig,
    extract_keyframe_features,
    verify_loop_candidate,
)


@dataclass(frozen=True)
class CameraAnchorPlannerConfig:
    analysis_width: int = 960

    # Sparse graph proposal:
    # cheap geometry descriptor -> a few plausible neighbors -> direct SIFT check.
    geometry_neighbors_per_state: int = 8
    temporal_neighbors_each_side: int = 2

    # Only states that are realistic HUMAN calibration candidates participate.
    minimum_calibratability: float = 0.16

    # Safety cap; not a target number of HUMAN anchors.
    maximum_plan_anchors: int = 80

    # Greedy multicover ranking.
    gain_weight: float = 1.0
    quality_weight: float = 0.30
    novelty_weight: float = 0.20


def _read_frame(
    capture: cv2.VideoCapture,
    frame: int,
) -> np.ndarray | None:
    capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame))
    ok, image = capture.read()
    return image if ok and image is not None else None


def _keyframe_frames(trajectory: dict[str, Any]) -> list[int]:
    frames: list[int] = []
    for item in trajectory.get("keyframes", []):
        if not isinstance(item, dict):
            continue
        try:
            frames.append(int(item["frame"]))
        except (KeyError, TypeError, ValueError):
            continue
    return sorted(set(frames))


def _candidate_pairs(
    states: list[dict[str, Any]],
    cfg: CameraAnchorPlannerConfig,
) -> list[tuple[int, int]]:
    """Propose a sparse graph from geometry-nearest + temporal neighbors."""
    if not states:
        return []

    descriptors = np.asarray(
        [state["descriptor"] for state in states],
        dtype=np.float64,
    )
    norms = np.linalg.norm(descriptors, axis=1, keepdims=True)
    unit = descriptors / np.maximum(norms, 1e-12)

    # field_view_distance() is (1-cosine)/2 for normalized descriptors.
    distances = np.clip(
        (1.0 - unit @ unit.T) * 0.5,
        0.0,
        1.0,
    )

    pairs: set[tuple[int, int]] = set()
    count = len(states)
    k = max(
        0,
        min(cfg.geometry_neighbors_per_state, count - 1),
    )

    for index in range(count):
        if k:
            added = 0
            for other_raw in np.argsort(distances[index]):
                other = int(other_raw)
                if other == index:
                    continue
                pairs.add(
                    (min(index, other), max(index, other))
                )
                added += 1
                if added >= k:
                    break

        # Temporal edges protect us when the geometry descriptor is imperfect
        # during pans/zooms: adjacent camera states are still worth verifying.
        for delta in range(
            1,
            cfg.temporal_neighbors_each_side + 1,
        ):
            for other in (index - delta, index + delta):
                if 0 <= other < count:
                    pairs.add(
                        (min(index, other), max(index, other))
                    )

    return sorted(pairs)


def _coverage_sets(
    states: list[dict[str, Any]],
    edges: list[dict[str, Any]],
) -> dict[int, set[int]]:
    """Direct one-hop support only. No transitive closure."""
    coverage = {
        int(state["frame"]): {int(state["frame"])}
        for state in states
    }

    for edge in edges:
        if not edge.get("usable"):
            continue

        left = int(edge["left_frame"])
        right = int(edge["right_frame"])

        coverage.setdefault(left, {left}).add(right)
        coverage.setdefault(right, {right}).add(left)

    return coverage


def _greedy_multicover_plan(
    states: list[dict[str, Any]],
    coverage: dict[int, set[int]],
    multiplicity: int,
    cfg: CameraAnchorPlannerConfig,
) -> dict[str, Any]:
    """Greedy HUMAN proposal for direct 1x or 2x support.

    1x:
        each state should have at least one selected HUMAN that directly
        verifies against it (or be HUMAN itself).

    2x:
        each state should ideally have support from two selected HUMAN states.

    This is deliberately planning-only. Sparse Validator will later decide
    whether the transferred field homographies actually agree.
    """
    frames = [int(state["frame"]) for state in states]

    quality = {
        int(state["frame"]): float(state["quality_score"])
        for state in states
    }

    descriptor = {
        int(state["frame"]): np.asarray(
            state["descriptor"],
            dtype=float,
        )
        for state in states
    }

    selected: list[int] = []
    support = {frame: 0 for frame in frames}
    recommendations: list[dict[str, Any]] = []

    while len(selected) < cfg.maximum_plan_anchors:
        deficits = {
            frame: max(0, multiplicity - count)
            for frame, count in support.items()
        }

        if not any(deficits.values()):
            break

        best: int | None = None
        best_score = -float("inf")
        best_gain = -1

        for candidate in frames:
            if candidate in selected:
                continue

            covered = coverage.get(candidate, {candidate})
            gain = sum(
                1
                for frame in covered
                if deficits.get(frame, 0) > 0
            )
            if gain <= 0:
                continue

            if selected:
                novelty = min(
                    field_view_distance(
                        descriptor[candidate],
                        descriptor[anchor],
                    )
                    for anchor in selected
                )
            else:
                novelty = 1.0

            score = (
                cfg.gain_weight * gain
                + cfg.quality_weight * quality[candidate]
                + cfg.novelty_weight * novelty
            )

            if (
                score > best_score
                or (
                    math.isclose(
                        score,
                        best_score,
                        abs_tol=1e-12,
                    )
                    and gain > best_gain
                )
                or (
                    math.isclose(
                        score,
                        best_score,
                        abs_tol=1e-12,
                    )
                    and gain == best_gain
                    and (best is None or candidate < best)
                )
            ):
                best = candidate
                best_score = score
                best_gain = gain

        if best is None:
            break

        selected.append(best)
        newly_helped: list[int] = []

        for frame in coverage.get(best, {best}):
            if (
                frame in support
                and support[frame] < multiplicity
            ):
                support[frame] += 1
                newly_helped.append(frame)

        recommendations.append(
            {
                "rank": len(selected),
                "frame": int(best),
                "quality_score": quality[best],
                "gain_count": len(newly_helped),
                "covered_frames": sorted(
                    int(frame)
                    for frame in coverage.get(
                        best,
                        {best},
                    )
                ),
                "newly_helped_frames": sorted(
                    int(frame)
                    for frame in newly_helped
                ),
            }
        )

    uncovered = [
        frame
        for frame in frames
        if support[frame] < multiplicity
    ]

    histogram: dict[int, int] = {}
    for value in support.values():
        histogram[int(value)] = (
            histogram.get(int(value), 0) + 1
        )

    return {
        "multiplicity": int(multiplicity),
        "anchor_count": len(selected),
        "anchors": selected,
        "recommendations": recommendations,
        "covered_state_count": len(frames) - len(uncovered),
        "total_state_count": len(frames),
        "coverage_fraction": (
            (len(frames) - len(uncovered)) / len(frames)
            if frames
            else 0.0
        ),
        "uncovered_frames": uncovered,
        "support_histogram": histogram,
    }


def analyse_camera_anchor_planner(
    video_path: str | Path,
    trajectory: dict[str, Any] | str | Path,
    *,
    config: CameraAnchorPlannerConfig | None = None,
) -> dict[str, Any]:
    cfg = config or CameraAnchorPlannerConfig()
    path = Path(video_path)

    if not path.exists():
        return {
            "status": "video_not_found",
            "video": str(path),
        }

    if isinstance(trajectory, (str, Path)):
        try:
            trajectory = json.loads(
                Path(trajectory).read_text(
                    encoding="utf-8"
                )
            )
        except (OSError, json.JSONDecodeError) as error:
            return {
                "status": "trajectory_read_failed",
                "error": str(error),
            }

    frames = _keyframe_frames(trajectory)

    if not frames:
        return {"status": "no_keyframes"}

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return {
            "status": "video_open_failed",
            "video": str(path),
        }

    width = int(
        capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0
    )
    height = int(
        capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0
    )
    fps = float(
        capture.get(cv2.CAP_PROP_FPS) or 0.0
    )

    # Geometry descriptors are intentionally cheap; SIFT uses the existing
    # LoopClosure 960px policy.
    view_cfg = FieldViewScannerConfig(
        analysis_width=min(cfg.analysis_width, 640)
    )
    loop_cfg = LoopClosureConfig(
        analysis_width=cfg.analysis_width
    )

    states: list[dict[str, Any]] = []
    failed_frames: list[int] = []

    try:
        for index, frame in enumerate(frames, start=1):
            image = _read_frame(capture, frame)

            if image is None:
                failed_frames.append(int(frame))
                continue

            view = field_view_descriptor(
                image,
                view_cfg,
            )
            features = extract_keyframe_features(
                image,
                config=loop_cfg,
            )

            if (
                float(view.get("quality_score", 0.0))
                < cfg.minimum_calibratability
                or features.get("status") != "ok"
            ):
                failed_frames.append(int(frame))
                continue

            states.append(
                {
                    "frame": int(frame),
                    "quality_score": float(
                        view["quality_score"]
                    ),
                    "geometry": view["geometry"],
                    "descriptor": np.asarray(
                        view["descriptor"],
                        dtype=float,
                    ),
                    "features": features,
                }
            )

            if index % 25 == 0 or index == len(frames):
                print(
                    f"[planner] features "
                    f"{index}/{len(frames)}"
                )
    finally:
        capture.release()

    if not states:
        return {"status": "no_viable_states"}

    pair_indices = _candidate_pairs(states, cfg)
    edges: list[dict[str, Any]] = []

    total_pairs = len(pair_indices)

    for pair_number, (
        left_index,
        right_index,
    ) in enumerate(pair_indices, start=1):
        left = states[left_index]
        right = states[right_index]

        verification = verify_loop_candidate(
            left["features"],
            right["features"],
            native_width=width,
            native_height=height,
            config=loop_cfg,
        )

        edges.append(
            {
                "left_frame": int(left["frame"]),
                "right_frame": int(right["frame"]),
                "usable": (
                    verification.get("status") == "ok"
                ),
                "verification": {
                    key: value
                    for key, value
                    in verification.items()
                    if key != "homography"
                },
            }
        )

        if (
            pair_number % 100 == 0
            or pair_number == total_pairs
        ):
            verified = sum(
                1
                for edge in edges
                if edge["usable"]
            )
            print(
                f"[planner] edges "
                f"{pair_number}/{total_pairs} "
                f"· verified={verified}"
            )

    coverage = _coverage_sets(states, edges)

    plan_1x = _greedy_multicover_plan(
        states,
        coverage,
        1,
        cfg,
    )
    plan_2x = _greedy_multicover_plan(
        states,
        coverage,
        2,
        cfg,
    )

    degrees = {
        frame: max(0, len(neighbors) - 1)
        for frame, neighbors in coverage.items()
    }

    quality_by_frame = {
        int(state["frame"]): float(
            state["quality_score"]
        )
        for state in states
    }

    weak_states = sorted(
        (
            {
                "frame": int(frame),
                "direct_neighbor_count": int(degree),
                "quality_score": quality_by_frame[frame],
            }
            for frame, degree in degrees.items()
        ),
        key=lambda item: (
            item["direct_neighbor_count"],
            item["quality_score"],
            item["frame"],
        ),
    )

    public_states = [
        {
            "frame": int(state["frame"]),
            "quality_score": float(
                state["quality_score"]
            ),
            "geometry": state["geometry"],
            "direct_neighbor_count": int(
                degrees.get(
                    int(state["frame"]),
                    0,
                )
            ),
        }
        for state in states
    ]

    return {
        "status": "ok",
        "version": 1,
        "algorithm": "camera_anchor_planner_v1",
        "policy": {
            "uses_human_homographies": False,
            "uses_dense": False,
            "uses_relocalization_outputs": False,
            "uses_review_calibrations": False,
            "sparse_graph": True,
            "direct_visual_verification": True,
            "transitive_coverage": False,
        },
        "video": {
            "path": str(path),
            "width": width,
            "height": height,
            "fps": fps,
        },
        "trajectory_keyframe_count": len(frames),
        "viable_state_count": len(states),
        "failed_state_frames": failed_frames,
        "candidate_pair_count": len(pair_indices),
        "verified_edge_count": sum(
            1
            for edge in edges
            if edge["usable"]
        ),
        "config": asdict(cfg),
        "states": public_states,
        "edges": edges,
        "plan_1x": plan_1x,
        "plan_2x": plan_2x,
        "weak_states": weak_states[
            : min(30, len(weak_states))
        ],
    }
