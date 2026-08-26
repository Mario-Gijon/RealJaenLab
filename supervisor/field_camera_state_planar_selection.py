"""Reliability-based selection between left/right planar-flow cameras.

This keeps the strongest part of planar flow: each propagated H is a direct
image measurement of the ground plane.

It deliberately does NOT project those H estimates into the 5-D PCA manifold
and does NOT average raw H coefficients. Instead it selects one complete
propagated H using only observable path quality:

- accumulated step count,
- median RANSAC inlier ratio,
- median reprojection error.

The target-frame HUMAN H is never used.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PlanarSelectionConfig:
    step_decay_tau: float = 40.0
    inlier_ratio_power: float = 4.0
    reprojection_decay_px: float = 1.5

    minimum_reliability: float = 1e-6
    minimum_reliability_ratio_for_strong_choice: float = 1.20

    # Bidirectional disagreement is a confidence signal, not a reason to invent
    # a fused H. A selected direct path remains derived.
    good_disagreement_px: float = 80.0
    weak_disagreement_px: float = 180.0


def path_reliability(
    path_result: dict[str, Any] | None,
    *,
    config: PlanarSelectionConfig | None = None,
) -> float:
    cfg = config or PlanarSelectionConfig()

    if not isinstance(path_result, dict) or path_result.get("status") != "ok":
        return 0.0

    try:
        steps = max(0.0, float(path_result.get("step_count") or 0.0))
        inlier_ratio = float(path_result.get("median_inlier_ratio") or 0.0)
        reprojection = float(path_result.get("median_reprojection_px") or 0.0)
    except (TypeError, ValueError):
        return 0.0

    inlier_ratio = float(np.clip(inlier_ratio, 0.0, 1.0))
    tau = max(float(cfg.step_decay_tau), 1e-6)
    reproj_scale = max(float(cfg.reprojection_decay_px), 1e-6)

    reliability = (
        math.exp(-steps / tau)
        * (inlier_ratio ** float(cfg.inlier_ratio_power))
        * math.exp(-reprojection / reproj_scale)
    )

    return float(reliability) if math.isfinite(reliability) else 0.0


def select_planar_flow_camera(
    left_path: dict[str, Any] | None,
    right_path: dict[str, Any] | None,
    *,
    disagreement_px: float | None = None,
    config: PlanarSelectionConfig | None = None,
) -> dict[str, Any]:
    """Select one complete propagated H. Never average homography coefficients."""

    cfg = config or PlanarSelectionConfig()

    candidates = []

    for side, path in (("left", left_path), ("right", right_path)):
        reliability = path_reliability(path, config=cfg)

        if (
            reliability < cfg.minimum_reliability
            or not isinstance(path, dict)
            or path.get("status") != "ok"
            or not isinstance(path.get("homography"), list)
        ):
            continue

        candidates.append(
            {
                "side": side,
                "reliability": reliability,
                "path": path,
            }
        )

    if not candidates:
        return {
            "status": "no_usable_planar_flow",
            "confidence": "poor",
            "config": asdict(cfg),
        }

    candidates.sort(
        key=lambda item: item["reliability"],
        reverse=True,
    )

    best = candidates[0]
    second = candidates[1] if len(candidates) > 1 else None

    ratio = (
        best["reliability"] / max(second["reliability"], 1e-12)
        if second is not None
        else float("inf")
    )

    if disagreement_px is None or not math.isfinite(float(disagreement_px)):
        disagreement = None
    else:
        disagreement = float(disagreement_px)

    if second is None:
        confidence = "weak"
        reason = "only_one_valid_path"
    elif (
        disagreement is not None
        and disagreement <= cfg.good_disagreement_px
        and ratio >= 1.05
    ):
        confidence = "good"
        reason = "paths_broadly_agree_and_best_is_more_reliable"
    elif ratio >= cfg.minimum_reliability_ratio_for_strong_choice:
        confidence = "good"
        reason = "one_path_is_materially_more_reliable"
    elif (
        disagreement is not None
        and disagreement <= cfg.weak_disagreement_px
    ):
        confidence = "weak"
        reason = "paths_disagree_but_choice_is_not_decisive"
    else:
        confidence = "poor"
        reason = "large_disagreement_and_similar_reliability"

    return {
        "status": "ok",
        "confidence": confidence,
        "selected_side": best["side"],
        "homography": best["path"]["homography"],
        "selected_reliability": best["reliability"],
        "reliability_ratio": (
            float(ratio)
            if math.isfinite(ratio)
            else None
        ),
        "disagreement_px": disagreement,
        "reason": reason,
        "candidates": [
            {
                "side": item["side"],
                "reliability": item["reliability"],
                "step_count": item["path"].get("step_count"),
                "median_inlier_ratio": item["path"].get("median_inlier_ratio"),
                "median_reprojection_px": item["path"].get("median_reprojection_px"),
            }
            for item in candidates
        ],
        "config": asdict(cfg),
        "policy": {
            "source": "planar_flow_selected",
            "purpose": "derived_camera",
            "averages_raw_h_coefficients": False,
            "projects_into_5d_before_selection": False,
            "is_human_anchor": False,
            "may_train_camera_state": False,
        },
    }
