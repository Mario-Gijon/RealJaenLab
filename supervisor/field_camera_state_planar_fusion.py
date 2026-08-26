"""Bidirectional fusion of ground-plane planar-flow camera estimates.

The planar-flow module can independently propagate a strong HUMAN pitch H from
the left anchor and from the right anchor to the same target frame.

This module converts both propagated H estimates into the validated 5-D
Projective Camera State, assigns reliability from path length + RANSAC quality,
and either:

- reliability-weights the two states when they broadly agree, or
- selects the more reliable side when their image-space disagreement is large.

The linear temporal Camera State prior is an optional weak regularizer only.
Raw H coefficients are never averaged.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import numpy as np

from supervisor.field_camera_state_projective import (
    homography_from_state,
    projective_jet_from_homography,
)


@dataclass(frozen=True)
class PlanarFusionConfig:
    step_decay_tau: float = 40.0
    inlier_ratio_power: float = 4.0
    reprojection_decay_px: float = 1.5
    temporal_prior_weight: float = 0.0
    hard_select_disagreement_px: float = 160.0
    maximum_state_radius_from_prior: float = 3.2
    minimum_reliability: float = 1e-5


def state_from_homography(
    projective_state_model: dict[str, Any],
    homography: Any,
) -> np.ndarray | None:
    """Project an arbitrary H into the fitted 5-D Projective Camera State."""

    if projective_state_model.get("status") != "ok":
        return None

    feature = projective_jet_from_homography(
        homography,
        image_width=projective_state_model["image_width"],
        image_height=projective_state_model["image_height"],
        pitch_length_m=projective_state_model["pitch_length_m"],
        pitch_width_m=projective_state_model["pitch_width_m"],
        maximum_condition=projective_state_model["config"]["maximum_h_condition"],
    )
    if feature is None:
        return None

    feature = np.asarray(feature, dtype=np.float64)
    mean = np.asarray(projective_state_model["feature_mean"], dtype=np.float64)
    std = np.asarray(projective_state_model["feature_std"], dtype=np.float64)
    pca_mean = np.asarray(projective_state_model["pca_mean"], dtype=np.float64)
    components = np.asarray(
        projective_state_model["pca_components"],
        dtype=np.float64,
    )

    z = (feature - mean) / std
    state = (z - pca_mean) @ components.T

    return state if np.isfinite(state).all() else None


def _anchor_state_scale(
    projective_state_model: dict[str, Any],
) -> np.ndarray:
    values = np.asarray(
        [
            anchor["state"]
            for anchor in projective_state_model.get("anchors", [])
            if isinstance(anchor, dict)
            and anchor.get("state") is not None
        ],
        dtype=np.float64,
    )

    if values.ndim != 2 or len(values) == 0:
        return np.ones(
            int(projective_state_model.get("dimensions") or 1),
            dtype=np.float64,
        )

    return np.maximum(
        np.std(values, axis=0),
        0.30,
    )


def state_radius(
    projective_state_model: dict[str, Any],
    state: Any,
    prior_state: Any,
) -> float | None:
    try:
        state = np.asarray(state, dtype=np.float64).reshape(-1)
        prior = np.asarray(prior_state, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None

    if state.shape != prior.shape or not np.isfinite(state).all() or not np.isfinite(prior).all():
        return None

    scale = _anchor_state_scale(projective_state_model)

    if scale.shape != state.shape:
        return None

    return float(
        np.sqrt(
            np.mean(
                ((state - prior) / scale) ** 2
            )
        )
    )


def path_reliability(
    path_result: dict[str, Any] | None,
    *,
    config: PlanarFusionConfig | None = None,
) -> float:
    cfg = config or PlanarFusionConfig()

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

    return (
        float(reliability)
        if math.isfinite(reliability)
        else 0.0
    )


def _decode_state(
    model: dict[str, Any],
    state: np.ndarray,
) -> np.ndarray | None:
    decoded = homography_from_state(
        model,
        state,
    )

    if decoded.get("status") != "ok":
        return None

    try:
        homography = np.asarray(
            decoded["homography"],
            dtype=np.float64,
        )
    except (KeyError, TypeError, ValueError):
        return None

    if homography.shape != (3, 3) or not np.isfinite(homography).all():
        return None

    return homography


def fuse_bidirectional_planar_flow(
    projective_state_model: dict[str, Any],
    temporal_prior_state: Any,
    left_path: dict[str, Any] | None,
    right_path: dict[str, Any] | None,
    *,
    disagreement_px: float | None = None,
    config: PlanarFusionConfig | None = None,
) -> dict[str, Any]:
    """Fuse left/right propagated H estimates in Projective State space."""

    cfg = config or PlanarFusionConfig()

    if projective_state_model.get("status") != "ok":
        return {"status": "invalid_model"}

    try:
        prior = np.asarray(
            temporal_prior_state,
            dtype=np.float64,
        ).reshape(-1)
    except (TypeError, ValueError):
        return {"status": "invalid_temporal_prior"}

    if (
        prior.shape != (
            int(projective_state_model["dimensions"]),
        )
        or not np.isfinite(prior).all()
    ):
        return {"status": "invalid_temporal_prior"}

    candidates = []

    for side, path in (
        ("left", left_path),
        ("right", right_path),
    ):
        if (
            not isinstance(path, dict)
            or path.get("status") != "ok"
            or not isinstance(path.get("homography"), list)
        ):
            continue

        state = state_from_homography(
            projective_state_model,
            path["homography"],
        )

        if state is None:
            continue

        radius = state_radius(
            projective_state_model,
            state,
            prior,
        )

        reliability = path_reliability(
            path,
            config=cfg,
        )

        if (
            radius is None
            or radius > cfg.maximum_state_radius_from_prior
            or reliability < cfg.minimum_reliability
        ):
            continue

        candidates.append(
            {
                "side": side,
                "state": state,
                "reliability": reliability,
                "radius_from_prior": radius,
                "path": path,
            }
        )

    if not candidates:
        return {
            "status": "no_usable_planar_flow",
            "config": asdict(cfg),
        }

    if len(candidates) == 1:
        selected = candidates[0]
        state = selected["state"].copy()
        mode = f"{selected['side']}_only"
        weights = {
            selected["side"]: 1.0,
            "temporal_prior": 0.0,
        }

    else:
        left = next(
            item
            for item in candidates
            if item["side"] == "left"
        )
        right = next(
            item
            for item in candidates
            if item["side"] == "right"
        )

        hard_select = (
            disagreement_px is not None
            and math.isfinite(float(disagreement_px))
            and float(disagreement_px)
            >= float(cfg.hard_select_disagreement_px)
        )

        if hard_select:
            selected = max(
                (left, right),
                key=lambda item: item["reliability"],
            )
            state = selected["state"].copy()
            mode = "hard_select_" + selected["side"]
            weights = {
                "left": (
                    1.0
                    if selected["side"] == "left"
                    else 0.0
                ),
                "right": (
                    1.0
                    if selected["side"] == "right"
                    else 0.0
                ),
                "temporal_prior": 0.0,
            }
        else:
            left_weight = left["reliability"]
            right_weight = right["reliability"]
            prior_weight = max(
                0.0,
                float(cfg.temporal_prior_weight),
            )

            denominator = (
                left_weight
                + right_weight
                + prior_weight
            )

            if denominator <= 0.0:
                return {
                    "status": "invalid_weights",
                    "config": asdict(cfg),
                }

            state = (
                left_weight * left["state"]
                + right_weight * right["state"]
                + prior_weight * prior
            ) / denominator

            mode = "weighted_state_fusion"

            weights = {
                "left": float(
                    left_weight / denominator
                ),
                "right": float(
                    right_weight / denominator
                ),
                "temporal_prior": float(
                    prior_weight / denominator
                ),
            }

    radius = state_radius(
        projective_state_model,
        state,
        prior,
    )

    if (
        radius is None
        or radius > cfg.maximum_state_radius_from_prior
    ):
        return {
            "status": "fused_state_outside_prior_tube",
            "radius_from_prior": radius,
            "config": asdict(cfg),
        }

    homography = _decode_state(
        projective_state_model,
        state,
    )

    if homography is None:
        return {
            "status": "state_decode_failed",
            "config": asdict(cfg),
        }

    return {
        "status": "ok",
        "mode": mode,
        "state": state.tolist(),
        "homography": homography.tolist(),
        "radius_from_prior": radius,
        "weights": weights,
        "disagreement_px": (
            float(disagreement_px)
            if disagreement_px is not None
            and math.isfinite(float(disagreement_px))
            else None
        ),
        "candidates": [
            {
                "side": item["side"],
                "reliability": item["reliability"],
                "radius_from_prior": item["radius_from_prior"],
                "step_count": item["path"].get("step_count"),
                "median_inlier_ratio": item["path"].get("median_inlier_ratio"),
                "median_reprojection_px": item["path"].get("median_reprojection_px"),
            }
            for item in candidates
        ],
        "config": asdict(cfg),
        "policy": {
            "optimizes_raw_h_coefficients": False,
            "fuses_projective_state": True,
            "is_human_anchor": False,
            "may_train_camera_state": False,
        },
    }
