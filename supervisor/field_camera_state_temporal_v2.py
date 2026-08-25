"""Temporal Camera State Observer V2 for RealJaen FIELD.

V1 taught us an important lesson:

    accumulated affine optical-flow motion is useful as a continuity signal,
    but its net-motion alpha is NOT a reliable linear coordinate of the learned
    3-D Camera State trajectory.

V2 therefore separates two responsibilities:

1. Camera State trajectory:
   interpolate the HUMAN 3-D states with a local cubic Hermite curve in time.

2. Raw-video optical flow:
   validate that the interval is visually continuous and free of obvious
   tracking failure / shot discontinuity. It is a confidence signal, not the
   state coordinate itself.

This module is still analysis-only:
- no homography reconstruction;
- no review mutation;
- no PEOPLE;
- no propagation/refinement;
- no legacy Atlas.

It intentionally keeps V1 available so benchmarks can compare:
    linear time vs V1 motion-alpha vs V2 Hermite trajectory.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Any

import numpy as np

from supervisor.field_camera_state_temporal import (
    TemporalCameraStateConfig,
    estimate_bracketed_camera_state as estimate_v1_motion_state,
    measure_temporal_motion_path,
)


EPS = 1e-10


@dataclass(frozen=True)
class TemporalCameraStateV2Config:
    # Hermite tangent strength. 1.0 = standard centered finite-difference
    # tangent, lower values reduce overshoot.
    tangent_tension: float = 0.65

    # Optical flow is a confidence gate only.
    good_valid_step_fraction: float = 0.85
    weak_valid_step_fraction: float = 0.55

    # Compare the direct left->right accumulated affine with the composition
    # of left->query and query->right. This is not used to move state; it only
    # detects inconsistent flow accumulation.
    good_composition_residual: float = 0.10
    weak_composition_residual: float = 0.25

    # Reuse the already-tested V1 raw-motion parameters.
    motion: TemporalCameraStateConfig = TemporalCameraStateConfig()


def _state_map(camera_state_model: dict[str, Any]) -> dict[int, np.ndarray]:
    return {
        int(anchor["frame"]): np.asarray(
            anchor["state"],
            dtype=np.float64,
        )
        for anchor in camera_state_model.get("anchors", [])
        if isinstance(anchor, dict)
        and anchor.get("state") is not None
    }


def _linear_state(
    left_frame: int,
    right_frame: int,
    query_frame: int,
    left_state: np.ndarray,
    right_state: np.ndarray,
) -> tuple[np.ndarray, float]:
    alpha = (
        int(query_frame) - int(left_frame)
    ) / float(
        int(right_frame) - int(left_frame)
    )

    state = (
        (1.0 - alpha) * left_state
        + alpha * right_state
    )

    return state, float(alpha)


def _finite_difference_tangent(
    frame_prev: int | None,
    state_prev: np.ndarray | None,
    frame_current: int,
    state_current: np.ndarray,
    frame_next: int | None,
    state_next: np.ndarray | None,
    tension: float,
) -> np.ndarray:
    """Estimate ds/df at one HUMAN anchor."""

    if (
        frame_prev is not None
        and state_prev is not None
        and frame_next is not None
        and state_next is not None
    ):
        denominator = float(
            frame_next - frame_prev
        )
        if abs(denominator) > EPS:
            tangent = (
                state_next - state_prev
            ) / denominator
            return float(tension) * tangent

    if (
        frame_next is not None
        and state_next is not None
    ):
        denominator = float(
            frame_next - frame_current
        )
        if abs(denominator) > EPS:
            tangent = (
                state_next - state_current
            ) / denominator
            return float(tension) * tangent

    if (
        frame_prev is not None
        and state_prev is not None
    ):
        denominator = float(
            frame_current - frame_prev
        )
        if abs(denominator) > EPS:
            tangent = (
                state_current - state_prev
            ) / denominator
            return float(tension) * tangent

    return np.zeros_like(
        state_current,
        dtype=np.float64,
    )


def _hermite_state(
    left_frame: int,
    right_frame: int,
    query_frame: int,
    left_state: np.ndarray,
    right_state: np.ndarray,
    left_tangent_per_frame: np.ndarray,
    right_tangent_per_frame: np.ndarray,
) -> tuple[np.ndarray, float]:
    interval = float(
        right_frame - left_frame
    )

    if interval <= 0:
        raise ValueError(
            "Hermite interval must be positive."
        )

    alpha = (
        query_frame - left_frame
    ) / interval

    t = float(alpha)
    t2 = t * t
    t3 = t2 * t

    h00 = 2.0 * t3 - 3.0 * t2 + 1.0
    h10 = t3 - 2.0 * t2 + t
    h01 = -2.0 * t3 + 3.0 * t2
    h11 = t3 - t2

    state = (
        h00 * left_state
        + h10 * interval * left_tangent_per_frame
        + h01 * right_state
        + h11 * interval * right_tangent_per_frame
    )

    return (
        np.asarray(
            state,
            dtype=np.float64,
        ),
        float(alpha),
    )


def _normalise_affine_for_comparison(
    transform: Any,
    width: int,
    height: int,
) -> np.ndarray | None:
    try:
        matrix = np.asarray(
            transform,
            dtype=np.float64,
        )
    except (TypeError, ValueError):
        return None

    if (
        matrix.shape != (3, 3)
        or not np.isfinite(matrix).all()
    ):
        return None

    result = matrix.copy()
    result[0, 2] /= max(
        float(width),
        1.0,
    )
    result[1, 2] /= max(
        float(height),
        1.0,
    )

    return result


def _composition_residual(
    left_to_query: dict[str, Any],
    query_to_right: dict[str, Any],
    left_to_right: dict[str, Any],
) -> float | None:
    """Affine-composition consistency diagnostic.

    direct: left -> right
    composed: (query -> right) @ (left -> query)

    Translation is normalized by analysis image dimensions before the matrix
    Frobenius residual is measured.
    """

    width = int(
        left_to_right.get(
            "analysis_width",
            0,
        )
        or 0
    )
    height = int(
        left_to_right.get(
            "analysis_height",
            0,
        )
        or 0
    )

    direct = _normalise_affine_for_comparison(
        left_to_right.get(
            "final_transform"
        ),
        width,
        height,
    )
    first = _normalise_affine_for_comparison(
        left_to_query.get(
            "final_transform"
        ),
        width,
        height,
    )
    second = _normalise_affine_for_comparison(
        query_to_right.get(
            "final_transform"
        ),
        width,
        height,
    )

    if (
        direct is None
        or first is None
        or second is None
    ):
        return None

    composed = second @ first

    denominator = float(
        np.linalg.norm(
            direct,
        )
    )

    if denominator < EPS:
        return None

    return float(
        np.linalg.norm(
            direct - composed
        )
        / denominator
    )


def _motion_continuity(
    video_path: str | Path,
    *,
    left_frame: int,
    query_frame: int,
    right_frame: int,
    config: TemporalCameraStateV2Config,
) -> dict[str, Any]:
    motion_cfg = config.motion

    left_to_query = measure_temporal_motion_path(
        video_path,
        left_frame,
        query_frame,
        config=motion_cfg,
    )
    query_to_right = measure_temporal_motion_path(
        video_path,
        query_frame,
        right_frame,
        config=motion_cfg,
    )
    left_to_right = measure_temporal_motion_path(
        video_path,
        left_frame,
        right_frame,
        config=motion_cfg,
    )

    paths = [
        left_to_query,
        query_to_right,
        left_to_right,
    ]

    acceptable = {
        "ok",
        "weak_motion_path",
    }

    if any(
        path.get("status")
        not in acceptable
        for path in paths
    ):
        return {
            "status": "motion_measurement_failed",
            "confidence": "poor",
            "path_statuses": [
                path.get("status")
                for path in paths
            ],
        }

    valid_fraction = min(
        float(
            path.get(
                "valid_step_fraction",
                0.0,
            )
        )
        for path in paths
    )

    successful_fraction = min(
        float(
            path.get(
                "successful_step_fraction",
                0.0,
            )
        )
        for path in paths
    )

    residual = _composition_residual(
        left_to_query,
        query_to_right,
        left_to_right,
    )

    if (
        valid_fraction
        >= config.good_valid_step_fraction
        and (
            residual is None
            or residual
            <= config.good_composition_residual
        )
    ):
        confidence = "good"
    elif (
        successful_fraction
        >= config.weak_valid_step_fraction
        and (
            residual is None
            or residual
            <= config.weak_composition_residual
        )
    ):
        confidence = "weak"
    else:
        confidence = "poor"

    return {
        "status": "ok",
        "confidence": confidence,
        "valid_step_fraction": valid_fraction,
        "successful_step_fraction": successful_fraction,
        "composition_residual": residual,
        "left_to_query": {
            "valid_step_fraction": left_to_query.get(
                "valid_step_fraction"
            ),
            "median_inlier_ratio": left_to_query.get(
                "median_inlier_ratio"
            ),
        },
        "query_to_right": {
            "valid_step_fraction": query_to_right.get(
                "valid_step_fraction"
            ),
            "median_inlier_ratio": query_to_right.get(
                "median_inlier_ratio"
            ),
        },
        "left_to_right": {
            "valid_step_fraction": left_to_right.get(
                "valid_step_fraction"
            ),
            "median_inlier_ratio": left_to_right.get(
                "median_inlier_ratio"
            ),
        },
    }


def estimate_temporal_camera_state_v2(
    camera_state_model: dict[str, Any],
    video_path: str | Path,
    *,
    query_frame: int,
    excluded_frames: set[int] | None = None,
    config: TemporalCameraStateV2Config | None = None,
) -> dict[str, Any]:
    """Estimate one interior frame from surrounding HUMAN Camera State anchors."""

    cfg = config or TemporalCameraStateV2Config()

    if camera_state_model.get(
        "status"
    ) != "ok":
        return {
            "status": "invalid_camera_state_model",
        }

    query = int(
        query_frame
    )
    excluded = {
        int(frame)
        for frame in (
            excluded_frames
            or set()
        )
    }

    states = _state_map(
        camera_state_model
    )

    known_frames = sorted(
        frame
        for frame in states
        if frame not in excluded
        and frame != query
    )

    left_candidates = [
        frame
        for frame in known_frames
        if frame < query
    ]
    right_candidates = [
        frame
        for frame in known_frames
        if frame > query
    ]

    if (
        not left_candidates
        or not right_candidates
    ):
        return {
            "status": "not_bracketed",
            "query_frame": query,
        }

    left = left_candidates[
        -1
    ]
    right = right_candidates[
        0
    ]

    left_outer = (
        left_candidates[
            -2
        ]
        if len(
            left_candidates
        )
        >= 2
        else None
    )
    right_outer = (
        right_candidates[
            1
        ]
        if len(
            right_candidates
        )
        >= 2
        else None
    )

    left_state = states[
        left
    ]
    right_state = states[
        right
    ]

    left_tangent = (
        _finite_difference_tangent(
            left_outer,
            (
                states[
                    left_outer
                ]
                if left_outer
                is not None
                else None
            ),
            left,
            left_state,
            right,
            right_state,
            cfg.tangent_tension,
        )
    )

    right_tangent = (
        _finite_difference_tangent(
            left,
            left_state,
            right,
            right_state,
            right_outer,
            (
                states[
                    right_outer
                ]
                if right_outer
                is not None
                else None
            ),
            cfg.tangent_tension,
        )
    )

    hermite_state, alpha = (
        _hermite_state(
            left,
            right,
            query,
            left_state,
            right_state,
            left_tangent,
            right_tangent,
        )
    )

    linear_state, linear_alpha = (
        _linear_state(
            left,
            right,
            query,
            left_state,
            right_state,
        )
    )

    motion = _motion_continuity(
        video_path,
        left_frame=left,
        query_frame=query,
        right_frame=right,
        config=cfg,
    )

    return {
        "status": "ok",
        "method": "local_hermite_state_with_flow_confidence",
        "query_frame": query,
        "left_frame": left,
        "right_frame": right,
        "left_outer_frame": left_outer,
        "right_outer_frame": right_outer,
        "state": hermite_state.tolist(),
        "linear_time_state": linear_state.tolist(),
        "alpha_time": float(
            alpha
        ),
        "linear_alpha_time": float(
            linear_alpha
        ),
        "left_tangent_per_frame": (
            left_tangent.tolist()
        ),
        "right_tangent_per_frame": (
            right_tangent.tolist()
        ),
        "confidence": motion.get(
            "confidence",
            "poor",
        ),
        "motion_continuity": motion,
        "policy": {
            "optical_flow_moves_state": False,
            "optical_flow_is_confidence_only": True,
            "returns_homography": False,
            "requires_temporal_bracket": True,
        },
    }


def _state_rmse(
    first: np.ndarray,
    second: np.ndarray,
) -> float:
    delta = (
        np.asarray(
            first,
            dtype=np.float64,
        )
        - np.asarray(
            second,
            dtype=np.float64,
        )
    )

    return float(
        np.linalg.norm(
            delta
        )
        / math.sqrt(
            len(
                delta
            )
        )
    )


def temporal_camera_state_v2_leave_one_out(
    camera_state_model: dict[str, Any],
    video_path: str | Path,
    *,
    config: TemporalCameraStateV2Config | None = None,
) -> dict[str, Any]:
    """LOO benchmark for interior HUMAN anchors only."""

    cfg = config or TemporalCameraStateV2Config()

    if camera_state_model.get(
        "status"
    ) != "ok":
        return {
            "status": "invalid_camera_state_model",
        }

    frames = sorted(
        int(
            frame
        )
        for frame in camera_state_model.get(
            "anchor_frames",
            [],
        )
    )

    states = _state_map(
        camera_state_model
    )

    if len(
        frames
    ) < 5:
        return {
            "status": "not_enough_anchors",
        }

    results = []

    for index in range(
        1,
        len(
            frames
        )
        - 1,
    ):
        frame = frames[
            index
        ]

        estimate = (
            estimate_temporal_camera_state_v2(
                camera_state_model,
                video_path,
                query_frame=frame,
                excluded_frames={
                    frame
                },
                config=cfg,
            )
        )

        if estimate.get(
            "status"
        ) != "ok":
            results.append(
                {
                    "frame": frame,
                    "status": estimate.get(
                        "status"
                    ),
                }
            )
            continue

        target = states[
            frame
        ]
        hermite = np.asarray(
            estimate[
                "state"
            ],
            dtype=np.float64,
        )
        linear = np.asarray(
            estimate[
                "linear_time_state"
            ],
            dtype=np.float64,
        )

        v1 = (
            estimate_v1_motion_state(
                camera_state_model,
                video_path,
                left_frame=int(
                    estimate[
                        "left_frame"
                    ]
                ),
                query_frame=frame,
                right_frame=int(
                    estimate[
                        "right_frame"
                    ]
                ),
                config=cfg.motion,
            )
        )

        v1_error = None

        if v1.get(
            "status"
        ) == "ok":
            v1_error = (
                _state_rmse(
                    np.asarray(
                        v1[
                            "state"
                        ],
                        dtype=np.float64,
                    ),
                    target,
                )
            )

        results.append(
            {
                "frame": frame,
                "status": "ok",
                "left_frame": estimate[
                    "left_frame"
                ],
                "right_frame": estimate[
                    "right_frame"
                ],
                "hermite_state_rmse": _state_rmse(
                    hermite,
                    target,
                ),
                "linear_time_state_rmse": _state_rmse(
                    linear,
                    target,
                ),
                "v1_motion_state_rmse": v1_error,
                "confidence": estimate[
                    "confidence"
                ],
                "composition_residual": (
                    estimate[
                        "motion_continuity"
                    ].get(
                        "composition_residual"
                    )
                ),
                "valid_step_fraction": (
                    estimate[
                        "motion_continuity"
                    ].get(
                        "valid_step_fraction"
                    )
                ),
            }
        )

    valid = [
        item
        for item in results
        if item.get(
            "status"
        )
        == "ok"
    ]

    if not valid:
        return {
            "status": "no_valid_holdouts",
            "per_anchor": results,
        }

    hermite_errors = np.asarray(
        [
            item[
                "hermite_state_rmse"
            ]
            for item in valid
        ],
        dtype=np.float64,
    )

    linear_errors = np.asarray(
        [
            item[
                "linear_time_state_rmse"
            ]
            for item in valid
        ],
        dtype=np.float64,
    )

    v1_errors = np.asarray(
        [
            item[
                "v1_motion_state_rmse"
            ]
            for item in valid
            if item.get(
                "v1_motion_state_rmse"
            )
            is not None
        ],
        dtype=np.float64,
    )

    summary = {
        "count": len(
            valid
        ),
        "hermite_median_rmse": float(
            np.median(
                hermite_errors
            )
        ),
        "hermite_p90_rmse": float(
            np.percentile(
                hermite_errors,
                90,
            )
        ),
        "hermite_max_rmse": float(
            np.max(
                hermite_errors
            )
        ),
        "linear_median_rmse": float(
            np.median(
                linear_errors
            )
        ),
        "linear_p90_rmse": float(
            np.percentile(
                linear_errors,
                90,
            )
        ),
        "linear_max_rmse": float(
            np.max(
                linear_errors
            )
        ),
        "median_improvement_vs_linear": float(
            np.median(
                linear_errors
                - hermite_errors
            )
        ),
    }

    if len(
        v1_errors
    ):
        summary.update(
            {
                "v1_motion_median_rmse": float(
                    np.median(
                        v1_errors
                    )
                ),
                "v1_motion_p90_rmse": float(
                    np.percentile(
                        v1_errors,
                        90,
                    )
                ),
                "v1_motion_max_rmse": float(
                    np.max(
                        v1_errors
                    )
                ),
            }
        )

    return {
        "status": "ok",
        "summary": summary,
        "per_anchor": results,
        "config": {
            "tangent_tension": cfg.tangent_tension,
            "good_valid_step_fraction": (
                cfg.good_valid_step_fraction
            ),
            "weak_valid_step_fraction": (
                cfg.weak_valid_step_fraction
            ),
            "good_composition_residual": (
                cfg.good_composition_residual
            ),
            "weak_composition_residual": (
                cfg.weak_composition_residual
            ),
            "motion": asdict(
                cfg.motion
            ),
        },
    }
