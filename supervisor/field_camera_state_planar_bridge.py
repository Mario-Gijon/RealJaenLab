"""Endpoint-closed bridge correction for ground-plane planar flow.

A propagated planar-flow H is a direct image measurement, but sequential
composition accumulates drift. Between two strong HUMAN anchors we know the
exact H at BOTH ends, so the accumulated endpoint error is observable.

This module removes that drift in the COMPLETE 8-D projective jet, not in the
5-D PCA manifold and not by averaging raw H coefficients.

For a left->target path inside [L,R]:

    J_raw(t)      = exact projective jet of propagated H_t
    J_raw(R)      = exact projective jet of propagated H_R
    J_true(R)     = exact projective jet of strong HUMAN H_R
    delta_R       = J_true(R) - J_raw(R)
    J_corrected(t)= J_raw(t) + w(alpha) * delta_R

where alpha=(t-L)/(R-L). The right->target path is corrected symmetrically
towards the exact left endpoint.

Target-frame HUMAN geometry is never used.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import numpy as np

from supervisor.field_camera_state_projective import (
    homography_from_projective_jet,
    projective_jet_from_homography,
)


@dataclass(frozen=True)
class PlanarBridgeConfig:
    correction_exponent: float = 1.0
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

    return (
        feature
        if feature.shape == (8,)
        and np.isfinite(feature).all()
        else None
    )


def _weight(
    progress: float,
    exponent: float,
) -> float:
    progress = float(np.clip(progress, 0.0, 1.0))
    exponent = max(float(exponent), 1e-6)
    return float(progress ** exponent)


def bridge_correct_homography(
    target_path: dict[str, Any] | None,
    closure_path: dict[str, Any] | None,
    true_endpoint_homography: Any,
    *,
    progress_to_endpoint: float,
    image_width: int,
    image_height: int,
    pitch_length_m: float,
    pitch_width_m: float,
    config: PlanarBridgeConfig | None = None,
) -> dict[str, Any]:
    """Correct one propagated target H using its known endpoint closure error."""

    cfg = config or PlanarBridgeConfig()

    if (
        not isinstance(target_path, dict)
        or target_path.get("status") != "ok"
        or not isinstance(target_path.get("homography"), list)
    ):
        return {"status": "invalid_target_path"}

    if (
        not isinstance(closure_path, dict)
        or closure_path.get("status") != "ok"
        or not isinstance(closure_path.get("homography"), list)
    ):
        return {"status": "invalid_closure_path"}

    raw_target_jet = _jet(
        target_path["homography"],
        image_width=image_width,
        image_height=image_height,
        pitch_length_m=pitch_length_m,
        pitch_width_m=pitch_width_m,
        maximum_condition=cfg.maximum_condition,
    )
    raw_endpoint_jet = _jet(
        closure_path["homography"],
        image_width=image_width,
        image_height=image_height,
        pitch_length_m=pitch_length_m,
        pitch_width_m=pitch_width_m,
        maximum_condition=cfg.maximum_condition,
    )
    true_endpoint_jet = _jet(
        true_endpoint_homography,
        image_width=image_width,
        image_height=image_height,
        pitch_length_m=pitch_length_m,
        pitch_width_m=pitch_width_m,
        maximum_condition=cfg.maximum_condition,
    )

    if (
        raw_target_jet is None
        or raw_endpoint_jet is None
        or true_endpoint_jet is None
    ):
        return {"status": "jet_conversion_failed"}

    endpoint_delta = (
        true_endpoint_jet
        - raw_endpoint_jet
    )

    weight = _weight(
        progress_to_endpoint,
        cfg.correction_exponent,
    )

    corrected_jet = (
        raw_target_jet
        + weight * endpoint_delta
    )

    corrected_h = homography_from_projective_jet(
        corrected_jet,
        image_width=image_width,
        image_height=image_height,
        pitch_length_m=pitch_length_m,
        pitch_width_m=pitch_width_m,
        maximum_condition=cfg.maximum_condition,
    )

    if corrected_h is None:
        return {"status": "corrected_jet_to_h_failed"}

    return {
        "status": "ok",
        "homography": corrected_h.tolist(),
        "progress_to_endpoint": float(
            np.clip(
                progress_to_endpoint,
                0.0,
                1.0,
            )
        ),
        "correction_weight": weight,
        "endpoint_jet_delta_rms": float(
            np.sqrt(
                np.mean(
                    endpoint_delta * endpoint_delta
                )
            )
        ),
        "target_jet_change_rms": float(
            np.sqrt(
                np.mean(
                    (
                        corrected_jet
                        - raw_target_jet
                    )
                    ** 2
                )
            )
        ),
        "raw_target_jet": raw_target_jet.tolist(),
        "corrected_target_jet": corrected_jet.tolist(),
        "config": asdict(cfg),
        "policy": {
            "uses_target_human_geometry": False,
            "uses_endpoint_human_geometry": True,
            "uses_complete_projective_jet_8d": True,
            "uses_pca_5d": False,
            "averages_raw_h_coefficients": False,
            "is_human_anchor": False,
            "may_train_camera_state": False,
        },
    }


def corrected_path_result(
    original_target_path: dict[str, Any],
    correction_result: dict[str, Any],
) -> dict[str, Any]:
    """Keep original flow-quality diagnostics but substitute the corrected H."""

    if correction_result.get("status") != "ok":
        return {
            "status": correction_result.get(
                "status",
                "bridge_correction_failed",
            )
        }

    output = dict(
        original_target_path
    )
    output["homography"] = correction_result[
        "homography"
    ]
    output["bridge_correction"] = {
        key: value
        for key, value in correction_result.items()
        if key not in {
            "homography",
            "raw_target_jet",
            "corrected_target_jet",
        }
    }

    return output
