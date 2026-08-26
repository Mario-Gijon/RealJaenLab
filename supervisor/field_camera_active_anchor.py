"""Active Anchor Selector V1 for genuinely unresolved camera states.

Input:
    - one target video
    - cross-video transfer results containing unresolved keyframes

Goal:
    choose ONE HUMAN frame that is expected to resolve the largest number of
    currently unresolved keyframes.

Important:
    We do not cluster by time alone and we do not use transitive union-find.
    Every candidate HUMAN is scored by DIRECT image-to-image geometric overlap
    with every other unresolved keyframe.

If candidate A directly verifies against B, C and D, then one HUMAN at A can in
principle transfer its exact pitch H to B/C/D.

This directly measures expected marginal coverage gain per HUMAN.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from supervisor.field_camera_loop_closure import (
    LoopClosureConfig,
    extract_keyframe_features,
    verify_loop_candidate,
)


@dataclass(frozen=True)
class ActiveAnchorConfig:
    analysis_width: int = 960

    # Direct overlap already passed the robust LoopClosure verifier. These
    # additional limits avoid calling an extremely weak overlap "covered".
    minimum_pair_inliers: int = 22
    minimum_pair_inlier_ratio: float = 0.36
    maximum_pair_reprojection_px: float = 2.6


def _read_features(
    capture: cv2.VideoCapture,
    frame: int,
    loop_config: LoopClosureConfig,
) -> dict[str, Any]:
    capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame))
    ok, image = capture.read()
    if not ok or image is None:
        return {"status": "frame_read_failed", "frame": int(frame)}
    return extract_keyframe_features(image, config=loop_config)


def _pair_is_usable(
    verification: dict[str, Any],
    cfg: ActiveAnchorConfig,
) -> bool:
    return bool(
        verification.get("status") == "ok"
        and int(verification.get("inlier_count") or 0)
        >= cfg.minimum_pair_inliers
        and float(verification.get("inlier_ratio") or 0.0)
        >= cfg.minimum_pair_inlier_ratio
        and float(
            verification.get("median_reprojection_px") or float("inf")
        )
        <= cfg.maximum_pair_reprojection_px
    )


def select_next_active_anchor(
    video_path: str | Path,
    unresolved_frames: list[int],
    *,
    config: ActiveAnchorConfig | None = None,
) -> dict[str, Any]:
    cfg = config or ActiveAnchorConfig()
    loop_cfg = LoopClosureConfig(analysis_width=cfg.analysis_width)

    frames = sorted({int(frame) for frame in unresolved_frames})
    if not frames:
        return {
            "status": "complete",
            "recommendations": [],
            "unresolved_count": 0,
        }

    path = Path(video_path)
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return {"status": "video_open_failed", "video": str(path)}

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)

    features: dict[int, dict[str, Any]] = {}
    try:
        for frame in frames:
            features[frame] = _read_features(
                capture,
                frame,
                loop_cfg,
            )
    finally:
        capture.release()

    # Direct pairwise overlap matrix. No transitive closure is used for scoring.
    pair_results: dict[tuple[int, int], dict[str, Any]] = {}

    for left_index, left in enumerate(frames):
        for right in frames[left_index + 1 :]:
            verification = verify_loop_candidate(
                features[left],
                features[right],
                native_width=width,
                native_height=height,
                config=loop_cfg,
            )
            pair_results[(left, right)] = {
                "usable": _pair_is_usable(verification, cfg),
                "verification": {
                    key: value
                    for key, value in verification.items()
                    if key != "homography"
                },
            }

    recommendations = []

    for candidate in frames:
        covered = [candidate]
        pair_quality = []

        for other in frames:
            if other == candidate:
                continue
            key = (
                (candidate, other)
                if candidate < other
                else (other, candidate)
            )
            pair = pair_results.get(key)
            if not pair or not pair["usable"]:
                continue

            covered.append(other)
            verification = pair["verification"]
            pair_quality.append(
                {
                    "frame": int(other),
                    "inlier_count": int(
                        verification.get("inlier_count") or 0
                    ),
                    "inlier_ratio": float(
                        verification.get("inlier_ratio") or 0.0
                    ),
                    "median_reprojection_px": float(
                        verification.get("median_reprojection_px")
                        or 999.0
                    ),
                    "direct_state_distance": verification.get(
                        "direct_state_distance"
                    ),
                }
            )

        if pair_quality:
            median_inliers = float(
                np.median(
                    [item["inlier_count"] for item in pair_quality]
                )
            )
            median_ratio = float(
                np.median(
                    [item["inlier_ratio"] for item in pair_quality]
                )
            )
            median_reproj = float(
                np.median(
                    [
                        item["median_reprojection_px"]
                        for item in pair_quality
                    ]
                )
            )
        else:
            median_inliers = 0.0
            median_ratio = 0.0
            median_reproj = float("inf")

        recommendations.append(
            {
                "frame": int(candidate),
                "expected_unresolved_covered": sorted(
                    int(frame) for frame in covered
                ),
                "expected_gain_count": len(covered),
                "expected_remaining_count": len(frames) - len(covered),
                "median_pair_inliers": median_inliers,
                "median_pair_inlier_ratio": median_ratio,
                "median_pair_reprojection_px": (
                    median_reproj
                    if np.isfinite(median_reproj)
                    else None
                ),
                "pair_quality": pair_quality,
            }
        )

    recommendations.sort(
        key=lambda item: (
            -int(item["expected_gain_count"]),
            -float(item["median_pair_inliers"]),
            -float(item["median_pair_inlier_ratio"]),
            float(
                item["median_pair_reprojection_px"]
                if item["median_pair_reprojection_px"] is not None
                else 999.0
            ),
            int(item["frame"]),
        )
    )

    for rank, recommendation in enumerate(recommendations, start=1):
        recommendation["rank"] = rank

    # Greedy preview of how many HUMAN anchors would be needed if each selected
    # candidate removed exactly the states it directly covers. This is only a
    # planning estimate; the real loop must recompute after every HUMAN.
    remaining = set(frames)
    greedy = []

    while remaining:
        best = None
        best_cover = set()

        for recommendation in recommendations:
            candidate = int(recommendation["frame"])
            if candidate not in remaining:
                continue
            coverage = set(
                int(frame)
                for frame in recommendation[
                    "expected_unresolved_covered"
                ]
            ) & remaining
            if (
                best is None
                or len(coverage) > len(best_cover)
                or (
                    len(coverage) == len(best_cover)
                    and int(recommendation["rank"])
                    < int(best["rank"])
                )
            ):
                best = recommendation
                best_cover = coverage

        if best is None:
            break

        greedy.append(
            {
                "frame": int(best["frame"]),
                "covers": sorted(best_cover),
                "gain_count": len(best_cover),
            }
        )
        remaining -= best_cover

    return {
        "status": "ok",
        "version": 1,
        "algorithm": "active_anchor_selector_v1",
        "video": {
            "path": str(path),
            "width": width,
            "height": height,
            "fps": fps,
        },
        "config": asdict(cfg),
        "unresolved_count": len(frames),
        "unresolved_frames": frames,
        "recommendations": recommendations,
        "greedy_plan_estimate": greedy,
        "pair_results": [
            {
                "left_frame": left,
                "right_frame": right,
                **result,
            }
            for (left, right), result in sorted(pair_results.items())
        ],
    }
