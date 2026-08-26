"""Anchor Relocalization V1.

Use a small set of exact HUMAN pitch homographies as reusable camera-state
anchors. A HUMAN anchor is NOT limited to nearby time: it can relocalize any
distant frame whose camera view has enough direct static-scene overlap.

For HUMAN anchor A and target frame T:

    G_A_to_T = direct image homography estimated from SIFT + robust geometry
    H_A      = exact HUMAN pitch-world -> anchor-image homography

therefore:

    H_T ~= G_A_to_T @ H_A

No target-frame pitch lines, pitch points, or target HUMAN H are used.

This module intentionally tests direct HUMAN reuse before introducing graph
propagation. If direct anchor reuse already covers most keyframes, the Camera
Atlas can be dramatically simpler.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from supervisor.field_camera_loop_closure import (
    LoopClosureConfig,
    extract_keyframe_features,
    verify_loop_candidate,
)


EPS = 1e-9


@dataclass(frozen=True)
class AnchorRelocalizationConfig:
    analysis_width: int = 960

    # Candidate acceptance is inherited from LoopClosureConfig. These thresholds
    # only classify already-verified candidates into a high-confidence tier.
    high_confidence_min_inliers: int = 250
    high_confidence_min_inlier_ratio: float = 0.90
    high_confidence_max_reprojection_px: float = 0.55

    # Soft score prefers large stable overlap and small local reprojection error.
    displacement_soft_scale_px: float = 600.0


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


def _candidate_score(
    verification: dict[str, Any],
    cfg: AnchorRelocalizationConfig,
) -> float:
    """Score only candidates that already passed direct geometric verification."""

    if verification.get("status") != "ok":
        return float("-inf")

    inliers = max(float(verification.get("inlier_count") or 0.0), 0.0)
    ratio = max(float(verification.get("inlier_ratio") or 0.0), 0.0)
    reproj = max(float(verification.get("median_reprojection_px") or 99.0), 0.0)
    direct = verification.get("direct_state_distance") or {}
    displacement = max(float(direct.get("median_px") or 0.0), 0.0)

    overlap = math.log1p(inliers) * max(ratio, 1e-6)
    local_precision = 1.0 / (1.0 + reproj)
    displacement_soft = 1.0 / (
        1.0 + displacement / max(cfg.displacement_soft_scale_px, 1.0)
    )
    return float(overlap * local_precision * displacement_soft)


def _high_confidence(
    verification: dict[str, Any],
    cfg: AnchorRelocalizationConfig,
) -> bool:
    return bool(
        verification.get("status") == "ok"
        and int(verification.get("inlier_count") or 0)
        >= cfg.high_confidence_min_inliers
        and float(verification.get("inlier_ratio") or 0.0)
        >= cfg.high_confidence_min_inlier_ratio
        and float(verification.get("median_reprojection_px") or float("inf"))
        <= cfg.high_confidence_max_reprojection_px
    )


def derive_target_homography(
    anchor_homography: Any,
    anchor_to_target_image_homography: Any,
) -> np.ndarray | None:
    anchor_h = _normalise_h(anchor_homography)
    image_h = _normalise_h(anchor_to_target_image_homography)
    if anchor_h is None or image_h is None:
        return None
    return _normalise_h(image_h @ anchor_h)


def _read_features(
    capture: cv2.VideoCapture,
    frame_index: int,
    *,
    loop_config: LoopClosureConfig,
) -> dict[str, Any]:
    capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
    ok, frame = capture.read()
    if not ok or frame is None:
        return {
            "status": "frame_read_failed",
            "frame": int(frame_index),
        }
    return extract_keyframe_features(
        frame,
        config=loop_config,
    )


def analyse_anchor_relocalization(
    video_path: str | Path,
    *,
    anchors: list[dict[str, Any]],
    targets: list[dict[str, Any]],
    config: AnchorRelocalizationConfig | None = None,
    exclude_same_frame: bool = False,
    progress_callback=None,
) -> dict[str, Any]:
    """Relocalize target camera states directly from reusable HUMAN anchors.

    anchors:
        [{"frame": int, "homography": 3x3, ...}, ...]

    targets:
        [{"frame": int, ...}, ...]

    The output includes one derived H per successfully relocalized target and
    all verified anchor candidates for diagnostics.
    """

    cfg = config or AnchorRelocalizationConfig()
    loop_cfg = LoopClosureConfig(
        analysis_width=cfg.analysis_width,
    )

    path = Path(video_path)
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return {"status": "video_open_failed", "video": str(path)}

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)

    valid_anchors = []
    for anchor in anchors:
        try:
            frame = int(anchor["frame"])
        except (KeyError, TypeError, ValueError):
            continue
        h = _normalise_h(anchor.get("homography"))
        if h is None:
            continue
        valid_anchors.append(
            {
                **anchor,
                "frame": frame,
                "homography": h.tolist(),
            }
        )

    valid_targets = []
    for target in targets:
        try:
            frame = int(target["frame"])
        except (KeyError, TypeError, ValueError):
            continue
        valid_targets.append({**target, "frame": frame})

    if not valid_anchors:
        capture.release()
        return {"status": "no_valid_anchors"}
    if not valid_targets:
        capture.release()
        return {"status": "no_valid_targets"}

    needed_frames = sorted(
        {int(anchor["frame"]) for anchor in valid_anchors}
        | {int(target["frame"]) for target in valid_targets}
    )

    features: dict[int, dict[str, Any]] = {}
    try:
        for index, frame in enumerate(needed_frames):
            features[frame] = _read_features(
                capture,
                frame,
                loop_config=loop_cfg,
            )
            if progress_callback is not None:
                progress_callback(
                    "features",
                    index + 1,
                    len(needed_frames),
                    frame,
                    0,
                )
    finally:
        capture.release()

    results = []
    verified_pair_count = 0
    high_confidence_pair_count = 0

    for target_index, target in enumerate(valid_targets):
        target_frame = int(target["frame"])
        candidates = []

        for anchor in valid_anchors:
            anchor_frame = int(anchor["frame"])
            if exclude_same_frame and anchor_frame == target_frame:
                continue

            verification = verify_loop_candidate(
                features.get(anchor_frame, {"status": "missing"}),
                features.get(target_frame, {"status": "missing"}),
                native_width=width,
                native_height=height,
                config=loop_cfg,
            )

            candidate = {
                "anchor_frame": anchor_frame,
                "target_frame": target_frame,
                "verification": {
                    key: value
                    for key, value in verification.items()
                    if key != "homography"
                },
            }

            if verification.get("status") == "ok":
                verified_pair_count += 1

                predicted_h = derive_target_homography(
                    anchor["homography"],
                    verification["homography"],
                )
                if predicted_h is not None:
                    candidate["homography"] = predicted_h.tolist()
                    candidate["image_homography"] = verification["homography"]
                    candidate["score"] = _candidate_score(verification, cfg)
                    candidate["high_confidence"] = _high_confidence(
                        verification,
                        cfg,
                    )
                    if candidate["high_confidence"]:
                        high_confidence_pair_count += 1

            candidates.append(candidate)

        valid_candidates = [
            item
            for item in candidates
            if isinstance(item.get("homography"), list)
            and math.isfinite(float(item.get("score", float("-inf"))))
        ]
        valid_candidates.sort(
            key=lambda item: (
                -float(item["score"]),
                -int(item["verification"].get("inlier_count") or 0),
                float(
                    item["verification"].get("median_reprojection_px")
                    or float("inf")
                ),
            )
        )

        best = valid_candidates[0] if valid_candidates else None

        result = {
            "frame": target_frame,
            "target": target,
            "candidate_count": len(candidates),
            "verified_candidate_count": len(valid_candidates),
            "candidates": candidates,
        }

        if best is None:
            result["status"] = "unrelocalized"
        else:
            result.update(
                {
                    "status": "ok",
                    "homography": best["homography"],
                    "source": "human_anchor_direct_relocalization_v1",
                    "anchor_frame": int(best["anchor_frame"]),
                    "confidence_tier": (
                        "high"
                        if bool(best.get("high_confidence"))
                        else "verified"
                    ),
                    "score": float(best["score"]),
                    "direct_state_distance": best["verification"].get(
                        "direct_state_distance"
                    ),
                    "match_count": best["verification"].get("match_count"),
                    "inlier_count": best["verification"].get("inlier_count"),
                    "inlier_ratio": best["verification"].get("inlier_ratio"),
                    "median_reprojection_px": best["verification"].get(
                        "median_reprojection_px"
                    ),
                }
            )

        results.append(result)

        if progress_callback is not None:
            covered = sum(1 for item in results if item.get("status") == "ok")
            progress_callback(
                "targets",
                target_index + 1,
                len(valid_targets),
                target_frame,
                covered,
            )

    covered = [item for item in results if item.get("status") == "ok"]
    high = [
        item
        for item in covered
        if item.get("confidence_tier") == "high"
    ]

    return {
        "status": "ok",
        "version": 1,
        "algorithm": "human_anchor_direct_relocalization_v1",
        "video": {
            "path": str(path),
            "width": width,
            "height": height,
            "fps": fps,
        },
        "config": asdict(cfg),
        "anchor_count": len(valid_anchors),
        "target_count": len(valid_targets),
        "covered_target_count": len(covered),
        "high_confidence_target_count": len(high),
        "coverage_fraction": len(covered) / max(float(len(valid_targets)), 1.0),
        "high_confidence_fraction": len(high) / max(float(len(valid_targets)), 1.0),
        "verified_pair_count": verified_pair_count,
        "high_confidence_pair_count": high_confidence_pair_count,
        "anchors": valid_anchors,
        "results": results,
    }
