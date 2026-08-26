"""Fuse endpoint-corrected planar-flow cameras in exact 8-D projective-jet space.

Only fuse when the two corrected cameras already agree sufficiently in image
space. Otherwise keep the normal reliability-selected complete H.

No raw-H coefficient averaging. No 5-D PCA projection. No target-frame HUMAN.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import numpy as np

from supervisor.field_camera_state_planar_selection import (
    PlanarSelectionConfig,
    path_reliability,
    select_planar_flow_camera,
)
from supervisor.field_camera_state_projective import (
    homography_from_projective_jet,
    projective_jet_from_homography,
)


@dataclass(frozen=True)
class PlanarBridgeFusionConfig:
    maximum_fusion_disagreement_px: float = 70.0
    reliability_power: float = 1.0
    maximum_condition: float = 1e14


def _jet(
    homography: Any,
    *,
    image_width: int,
    image_height: int,
    pitch_length_m: float,
    pitch_width_m: float,
    maximum_condition: float,
) -> np.ndarray | None:
    feature = projective_jet_from_homography(
        homography,
        image_width=image_width,
        image_height=image_height,
        pitch_length_m=pitch_length_m,
        pitch_width_m=pitch_width_m,
        maximum_condition=maximum_condition,
    )
    if feature is None:
        return None

    feature = np.asarray(feature, dtype=np.float64).reshape(-1)
    if feature.shape != (8,) or not np.isfinite(feature).all():
        return None
    return feature


def fuse_corrected_planar_paths(
    left_path: dict[str, Any] | None,
    right_path: dict[str, Any] | None,
    *,
    disagreement_px: float | None,
    image_width: int,
    image_height: int,
    pitch_length_m: float,
    pitch_width_m: float,
    selection_config: PlanarSelectionConfig | None = None,
    fusion_config: PlanarBridgeFusionConfig | None = None,
) -> dict[str, Any]:
    select_cfg = selection_config or PlanarSelectionConfig()
    fuse_cfg = fusion_config or PlanarBridgeFusionConfig()

    selected = select_planar_flow_camera(
        left_path,
        right_path,
        disagreement_px=disagreement_px,
        config=select_cfg,
    )
    if selected.get("status") != "ok" or not isinstance(selected.get("homography"), list):
        return {
            "status": selected.get("status", "selection_failed"),
            "selection": selected,
        }

    def selected_only(reason: str) -> dict[str, Any]:
        return {
            "status": "ok",
            "mode": "selected_only",
            "reason": reason,
            "homography": selected["homography"],
            "selection": selected,
            "weights": {selected.get("selected_side"): 1.0},
            "disagreement_px": (
                float(disagreement_px)
                if disagreement_px is not None and math.isfinite(float(disagreement_px))
                else None
            ),
            "config": asdict(fuse_cfg),
        }

    if (
        not isinstance(left_path, dict)
        or left_path.get("status") != "ok"
        or not isinstance(left_path.get("homography"), list)
        or not isinstance(right_path, dict)
        or right_path.get("status") != "ok"
        or not isinstance(right_path.get("homography"), list)
    ):
        return selected_only("one_path_unavailable")

    if disagreement_px is None or not math.isfinite(float(disagreement_px)):
        return selected_only("no_disagreement_measure")

    disagreement = float(disagreement_px)
    if disagreement > float(fuse_cfg.maximum_fusion_disagreement_px):
        return selected_only("disagreement_too_large")

    left_jet = _jet(
        left_path["homography"],
        image_width=image_width,
        image_height=image_height,
        pitch_length_m=pitch_length_m,
        pitch_width_m=pitch_width_m,
        maximum_condition=fuse_cfg.maximum_condition,
    )
    right_jet = _jet(
        right_path["homography"],
        image_width=image_width,
        image_height=image_height,
        pitch_length_m=pitch_length_m,
        pitch_width_m=pitch_width_m,
        maximum_condition=fuse_cfg.maximum_condition,
    )
    if left_jet is None or right_jet is None:
        return selected_only("jet_conversion_failed")

    left_rel = path_reliability(left_path, config=select_cfg)
    right_rel = path_reliability(right_path, config=select_cfg)

    power = max(0.0, float(fuse_cfg.reliability_power))
    if power == 0.0:
        lw = rw = 1.0
    else:
        lw = max(left_rel, 1e-12) ** power
        rw = max(right_rel, 1e-12) ** power

    total = lw + rw
    if total <= 0.0:
        return selected_only("invalid_weights")

    lf = lw / total
    rf = rw / total
    fused_jet = lf * left_jet + rf * right_jet

    fused_h = homography_from_projective_jet(
        fused_jet,
        image_width=image_width,
        image_height=image_height,
        pitch_length_m=pitch_length_m,
        pitch_width_m=pitch_width_m,
        maximum_condition=fuse_cfg.maximum_condition,
    )
    if fused_h is None:
        return selected_only("fused_jet_to_h_failed")

    return {
        "status": "ok",
        "mode": "fused_8d_jet",
        "homography": fused_h.tolist(),
        "selection": selected,
        "weights": {"left": float(lf), "right": float(rf)},
        "left_reliability": float(left_rel),
        "right_reliability": float(right_rel),
        "disagreement_px": disagreement,
        "config": asdict(fuse_cfg),
        "policy": {
            "uses_target_human_geometry": False,
            "uses_complete_projective_jet_8d": True,
            "uses_pca_5d": False,
            "averages_raw_h_coefficients": False,
            "is_human_anchor": False,
            "may_train_camera_state": False,
        },
    }
