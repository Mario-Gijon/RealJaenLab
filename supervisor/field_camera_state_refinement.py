"""Production Projective Camera State refinement V2 for RealJaen FIELD.

Validated weak-geometry recovery path:

    temporal Projective Camera State prior
    + partial HUMAN geometry
    -> optimize ONLY Projective State coordinates
    -> decode complete H

V2 fixes a semantic-weighting bug found by the frame-115 ablation.

Why V1 failed with one arc
--------------------------
The existing camera solver represents:
- one point as 2 residuals (u, v);
- one line as 2 endpoint-to-line residuals;
- one arc as many radial residual samples (often 24-32).

That is useful inside the independent solver, but in a prior-regularized state
refinement it allowed ONE arc to have the effective weight of dozens of HUMAN
references. Frame 115 proved this directly:

    points only     -> good, shift-z ~0.305
    points + 1 arc -> poor, shift-z ~5.257
    arc only        -> poor, shift-z ~10.778

V2 preserves all arc samples for geometry, but normalizes the TOTAL base weight
of each semantic arc to the equivalent of two scalar residuals, matching one
point (u,v) or one semantic line (two endpoints).

Important invariants
--------------------
- raw H coefficients are never optimized;
- derived H is never HUMAN evidence;
- lines keep the exact semantic projective treatment from the FIELD solver;
- arcs remain usable, but sample density no longer multiplies semantic weight;
- confidence uses semantic-balanced fit quality, not raw residual count.

No review I/O and no UI code lives here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import numpy as np

from supervisor.field_camera_calibration import (
    CameraCalibrationConfig,
    _line_diagnostics,
    _local_pixel_scale,
    _parse_arcs,
    _parse_lines,
    _parse_points,
    _point_diagnostics,
    _project_points,
    _residual_vector,
    _robust_weights,
)
from supervisor.field_camera_state_projective import (
    homography_from_state,
)


_EPS = 1e-10


@dataclass(frozen=True)
class ProjectiveStateRefinementConfig:
    # Empirically validated on the current match benchmark.
    prior_sigma_state: float = 3.0

    max_iterations: int = 70
    initial_damping: float = 1e-3
    finite_difference_step: float = 1e-5
    convergence_delta: float = 1e-6

    # One point contributes 2 scalar residuals. One semantic line contributes
    # 2 endpoint residuals. Give one entire arc the same base residual mass,
    # independently of whether the mouse trace contains 8, 24 or 32 samples.
    arc_equivalent_scalar_residuals: float = 2.0

    # Acceptance guardrails. These are diagnostic policy values, not universal
    # physical constants.
    good_fit_rms_sigma: float = 1.0
    warning_fit_rms_sigma: float = 2.0
    maximum_fit_rms_sigma: float = 3.0

    good_state_shift_z_rms: float = 1.5
    warning_state_shift_z_rms: float = 2.5
    maximum_state_shift_z_rms: float = 3.0

    camera: CameraCalibrationConfig = CameraCalibrationConfig()


def _as_state(
    value: Any,
    dimensions: int,
) -> np.ndarray | None:
    try:
        state = np.asarray(
            value,
            dtype=np.float64,
        ).reshape(-1)
    except (TypeError, ValueError):
        return None

    if (
        state.shape != (dimensions,)
        or not np.isfinite(state).all()
    ):
        return None

    return state


def _state_scale(
    model: dict[str, Any],
) -> np.ndarray:
    values = []

    for anchor in model.get(
        "anchors",
        [],
    ):
        if (
            not isinstance(anchor, dict)
            or anchor.get("state") is None
        ):
            continue

        try:
            state = np.asarray(
                anchor["state"],
                dtype=np.float64,
            ).reshape(-1)
        except (TypeError, ValueError):
            continue

        if np.isfinite(state).all():
            values.append(state)

    dimensions = int(
        model.get(
            "dimensions",
            0,
        )
        or 0
    )

    if (
        dimensions <= 0
        or len(values) < 2
    ):
        return np.ones(
            max(dimensions, 1),
            dtype=np.float64,
        )

    scale = np.std(
        np.asarray(
            values,
            dtype=np.float64,
        ),
        axis=0,
        ddof=0,
    )

    return np.where(
        scale < 1e-8,
        1.0,
        scale,
    )


def _decode_h(
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
        h = np.asarray(
            decoded["homography"],
            dtype=np.float64,
        )
    except (
        KeyError,
        TypeError,
        ValueError,
    ):
        return None

    if (
        h.shape != (3, 3)
        or not np.isfinite(h).all()
    ):
        return None

    return h


def _arc_layout(
    h: np.ndarray,
    arcs: list[dict[str, Any]],
) -> list[dict[str, Any]] | None:
    """Mirror the arc layout produced by field_camera_calibration._residual_vector."""

    if not arcs:
        return []

    try:
        inverse_h = np.linalg.inv(h)
    except np.linalg.LinAlgError:
        return None

    layout = []

    for item in arcs:
        image_samples = item["image_samples"]

        _, valid = _project_points(
            image_samples,
            inverse_h,
        )

        valid_count = int(
            np.count_nonzero(
                valid
            )
        )

        # _residual_vector fails the whole evaluation in this case.
        if valid_count < 4:
            return None

        px_per_metre = _local_pixel_scale(
            item["world_center"],
            item["world_radius"],
            h,
        )

        # _residual_vector silently skips this arc if scale cannot be measured.
        if px_per_metre is None:
            continue

        layout.append(
            {
                "semantic": item.get("semantic"),
                "residual_count": valid_count,
            }
        )

    return layout


def _semantic_base_weights(
    h: np.ndarray,
    points: list[dict[str, Any]],
    lines: list[dict[str, Any]],
    arcs: list[dict[str, Any]],
    data_residual_count: int,
    config: ProjectiveStateRefinementConfig,
) -> tuple[np.ndarray, list[dict[str, Any]]] | None:
    """Return per-scalar base weights before robust Huber weighting.

    Points and lines retain weight 1 per scalar residual:
        point -> total base mass 2
        line  -> total base mass 2

    Each arc gets:
        per_sample_weight = arc_equivalent_scalar_residuals / N_valid

    so the full arc has fixed total semantic mass regardless of trace density.
    """

    weights = np.ones(
        data_residual_count,
        dtype=np.float64,
    )

    offset = (
        2 * len(points)
        + 2 * len(lines)
    )

    layout = _arc_layout(
        h,
        arcs,
    )

    if layout is None:
        return None

    diagnostics = []

    target_mass = max(
        float(
            config.arc_equivalent_scalar_residuals
        ),
        _EPS,
    )

    for item in layout:
        count = int(
            item["residual_count"]
        )

        if (
            count <= 0
            or offset + count
            > data_residual_count
        ):
            return None

        per_sample = (
            target_mass
            / float(count)
        )

        weights[
            offset : offset + count
        ] = per_sample

        diagnostics.append(
            {
                "semantic": item.get("semantic"),
                "sample_count": count,
                "per_sample_base_weight": float(
                    per_sample
                ),
                "total_base_weight": float(
                    target_mass
                ),
            }
        )

        offset += count

    if offset != data_residual_count:
        # This indicates our layout no longer matches the solver's residual
        # construction; fail explicitly instead of silently mis-weighting.
        return None

    return (
        weights,
        diagnostics,
    )


def _evaluate(
    model: dict[str, Any],
    state: np.ndarray,
    prior_state: np.ndarray,
    state_scale: np.ndarray,
    points: list[dict[str, Any]],
    lines: list[dict[str, Any]],
    arcs: list[dict[str, Any]],
    config: ProjectiveStateRefinementConfig,
) -> dict[str, Any] | None:
    h = _decode_h(
        model,
        state,
    )

    if h is None:
        return None

    data = _residual_vector(
        h,
        points,
        lines,
        arcs,
        config.camera,
    )

    if data is None:
        return None

    data_residuals, data_kinds = data

    semantic_weights_result = (
        _semantic_base_weights(
            h,
            points,
            lines,
            arcs,
            len(data_residuals),
            config,
        )
    )

    if semantic_weights_result is None:
        return None

    semantic_weights, arc_weight_diagnostics = (
        semantic_weights_result
    )

    robust_weights = _robust_weights(
        data_residuals,
        data_kinds,
        config.camera,
    )

    data_weights = (
        semantic_weights
        * robust_weights
    )

    prior_sigma = max(
        float(
            config.prior_sigma_state
        ),
        1e-6,
    )

    prior_residuals = (
        state - prior_state
    ) / (
        state_scale
        * prior_sigma
    )

    residuals = np.concatenate(
        (
            data_residuals,
            prior_residuals,
        )
    )

    weights = np.concatenate(
        (
            data_weights,
            np.ones(
                len(prior_residuals),
                dtype=np.float64,
            ),
        )
    )

    return {
        "homography": h,
        "data_residuals": data_residuals,
        "data_kinds": data_kinds,
        "semantic_base_weights": semantic_weights,
        "data_weights": data_weights,
        "prior_residuals": prior_residuals,
        "residuals": residuals,
        "weights": weights,
        "arc_weight_diagnostics": arc_weight_diagnostics,
    }


def _weighted_cost(
    evaluated: dict[str, Any],
) -> float:
    residuals = evaluated[
        "residuals"
    ]
    weights = evaluated[
        "weights"
    ]

    return float(
        0.5
        * np.sum(
            weights
            * residuals
            * residuals
        )
    )


def _balanced_fit_rms_sigma(
    evaluated: dict[str, Any],
) -> float:
    """Raw geometric fit RMS with semantic sample-density normalization.

    We intentionally do NOT apply robust Huber weights here. A bad semantic
    reference should still be visible in diagnostics. Only the arc sample-count
    normalization is applied.
    """

    residuals = evaluated[
        "data_residuals"
    ]
    base_weights = evaluated[
        "semantic_base_weights"
    ]

    denominator = float(
        np.sum(
            base_weights
        )
    )

    if denominator <= _EPS:
        return float(
            "inf"
        )

    return float(
        math.sqrt(
            np.sum(
                base_weights
                * residuals
                * residuals
            )
            / denominator
        )
    )


def _confidence(
    *,
    fit_rms_sigma: float,
    state_shift_z_rms: float,
    prior_confidence: str | None,
    config: ProjectiveStateRefinementConfig,
) -> str:
    if prior_confidence in {
        "poor",
        "outside_coverage",
        "invalid",
    }:
        return "poor"

    if (
        fit_rms_sigma
        > config.maximum_fit_rms_sigma
        or state_shift_z_rms
        > config.maximum_state_shift_z_rms
    ):
        return "poor"

    if (
        fit_rms_sigma
        <= config.good_fit_rms_sigma
        and state_shift_z_rms
        <= config.warning_state_shift_z_rms
        and prior_confidence
        not in {
            "weak",
        }
    ):
        return "good"

    if (
        fit_rms_sigma
        <= config.warning_fit_rms_sigma
        and state_shift_z_rms
        <= config.maximum_state_shift_z_rms
    ):
        return "weak"

    return "poor"


def refine_projective_state_with_constraints(
    projective_state_model: dict[str, Any],
    prior_state: Any,
    point_constraints: list[dict[str, Any]],
    line_constraints: list[dict[str, Any]],
    arc_constraints: list[dict[str, Any]],
    *,
    prior_confidence: str | None = None,
    config: ProjectiveStateRefinementConfig | None = None,
) -> dict[str, Any]:
    """Refine a temporal Projective State prior with partial HUMAN geometry.

    The returned H is DERIVED and may never be promoted to a HUMAN anchor.
    """

    cfg = (
        config
        or ProjectiveStateRefinementConfig()
    )

    if projective_state_model.get(
        "status"
    ) != "ok":
        return {
            "status": "invalid_projective_state_model",
        }

    dimensions = int(
        projective_state_model.get(
            "dimensions",
            0,
        )
        or 0
    )

    if dimensions <= 0:
        return {
            "status": "invalid_projective_state_dimensions",
        }

    prior = _as_state(
        prior_state,
        dimensions,
    )

    if prior is None:
        return {
            "status": "invalid_prior_state",
        }

    points = _parse_points(
        point_constraints
    )
    lines = _parse_lines(
        line_constraints
    )
    arcs = _parse_arcs(
        arc_constraints
    )

    if not points and not lines and not arcs:
        return {
            "status": "no_constraints",
            "diagnostics": {
                "points": 0,
                "lines": 0,
                "arcs": 0,
            },
        }

    scale = _state_scale(
        projective_state_model
    )

    state = prior.copy()
    damping = float(
        cfg.initial_damping
    )

    initial = _evaluate(
        projective_state_model,
        state,
        prior,
        scale,
        points,
        lines,
        arcs,
        cfg,
    )

    if initial is None:
        return {
            "status": "initial_evaluation_failed",
        }

    initial_cost = _weighted_cost(
        initial
    )
    best_cost = initial_cost

    iterations = 0

    for iteration in range(
        cfg.max_iterations
    ):
        iterations = (
            iteration + 1
        )

        current = _evaluate(
            projective_state_model,
            state,
            prior,
            scale,
            points,
            lines,
            arcs,
            cfg,
        )

        if current is None:
            break

        residuals = current[
            "residuals"
        ]
        weights = current[
            "weights"
        ]

        jacobian = np.zeros(
            (
                len(residuals),
                dimensions,
            ),
            dtype=np.float64,
        )

        for column in range(
            dimensions
        ):
            step = (
                cfg.finite_difference_step
                * max(
                    1.0,
                    abs(
                        float(
                            state[column]
                        )
                    ),
                )
            )

            candidate_state = (
                state.copy()
            )
            candidate_state[
                column
            ] += step

            candidate = _evaluate(
                projective_state_model,
                candidate_state,
                prior,
                scale,
                points,
                lines,
                arcs,
                cfg,
            )

            if candidate is None:
                continue

            candidate_residuals = (
                candidate[
                    "residuals"
                ]
            )

            if len(
                candidate_residuals
            ) != len(
                residuals
            ):
                return {
                    "status": "residual_size_changed",
                }

            jacobian[
                :,
                column
            ] = (
                candidate_residuals
                - residuals
            ) / step

        sqrt_weights = np.sqrt(
            weights
        )

        weighted_j = (
            jacobian
            * sqrt_weights[
                :,
                None,
            ]
        )
        weighted_r = (
            residuals
            * sqrt_weights
        )

        lhs = (
            weighted_j.T
            @ weighted_j
            + damping
            * np.eye(
                dimensions,
                dtype=np.float64,
            )
        )

        rhs = -(
            weighted_j.T
            @ weighted_r
        )

        try:
            delta = np.linalg.solve(
                lhs,
                rhs,
            )
        except np.linalg.LinAlgError:
            damping *= 10.0
            continue

        if not np.isfinite(
            delta
        ).all():
            break

        candidate_state = (
            state + delta
        )

        candidate = _evaluate(
            projective_state_model,
            candidate_state,
            prior,
            scale,
            points,
            lines,
            arcs,
            cfg,
        )

        if candidate is None:
            damping *= 5.0
            continue

        candidate_cost = _weighted_cost(
            candidate
        )

        if candidate_cost < best_cost:
            state = candidate_state
            best_cost = candidate_cost
            damping = max(
                1e-8,
                damping * 0.35,
            )

            if (
                float(
                    np.linalg.norm(
                        delta
                    )
                )
                < cfg.convergence_delta
            ):
                break
        else:
            damping = min(
                1e8,
                damping * 4.0,
            )

    final = _evaluate(
        projective_state_model,
        state,
        prior,
        scale,
        points,
        lines,
        arcs,
        cfg,
    )

    if final is None:
        return {
            "status": "final_evaluation_failed",
        }

    final_h = final[
        "homography"
    ]

    fit_rms_sigma = (
        _balanced_fit_rms_sigma(
            final
        )
    )

    state_shift = (
        state - prior
    )
    state_shift_z = (
        state_shift / scale
    )

    state_shift_z_rms = float(
        np.linalg.norm(
            state_shift_z
        )
        / math.sqrt(
            dimensions
        )
    )

    confidence = _confidence(
        fit_rms_sigma=fit_rms_sigma,
        state_shift_z_rms=state_shift_z_rms,
        prior_confidence=prior_confidence,
        config=cfg,
    )

    initial_point_error, _ = (
        _point_diagnostics(
            initial["homography"],
            points,
        )
    )
    initial_line_error, _ = (
        _line_diagnostics(
            initial["homography"],
            lines,
        )
    )

    final_point_error, point_details = (
        _point_diagnostics(
            final_h,
            points,
        )
    )
    final_line_error, line_details = (
        _line_diagnostics(
            final_h,
            lines,
        )
    )

    try:
        condition = float(
            np.linalg.cond(
                final_h
            )
        )
    except np.linalg.LinAlgError:
        condition = float(
            "inf"
        )

    return {
        "status": "ok",
        "confidence": confidence,
        "homography": final_h.tolist(),
        "state": state.tolist(),
        "prior_state": prior.tolist(),
        "state_shift": state_shift.tolist(),
        "state_shift_z": state_shift_z.tolist(),
        "state_shift_z_rms": state_shift_z_rms,
        "fit_rms_sigma": fit_rms_sigma,
        "initial_cost": float(
            initial_cost
        ),
        "final_cost": float(
            _weighted_cost(
                final
            )
        ),
        "iterations": iterations,
        "diagnostics": {
            "mode": "projective_camera_state_refinement_v2_semantic_balanced",
            "projective_state_dimensions": dimensions,
            "prior_confidence": prior_confidence,
            "points": len(points),
            "lines": len(lines),
            "arcs": len(arcs),
            "initial_point_error_px": initial_point_error,
            "initial_line_error_px": initial_line_error,
            "point_error_px": final_point_error,
            "line_error_px": final_line_error,
            "point_errors": point_details,
            "line_errors": line_details,
            "arc_weighting": {
                "policy": "fixed_total_semantic_weight_per_arc",
                "equivalent_scalar_residuals_per_arc": float(
                    cfg.arc_equivalent_scalar_residuals
                ),
                "groups": final[
                    "arc_weight_diagnostics"
                ],
            },
            "homography_condition": condition,
            "fit_rms_sigma": fit_rms_sigma,
            "state_shift_z_rms": state_shift_z_rms,
            "config": {
                "prior_sigma_state": cfg.prior_sigma_state,
                "max_iterations": cfg.max_iterations,
                "arc_equivalent_scalar_residuals": (
                    cfg.arc_equivalent_scalar_residuals
                ),
                "good_fit_rms_sigma": cfg.good_fit_rms_sigma,
                "warning_fit_rms_sigma": cfg.warning_fit_rms_sigma,
                "maximum_fit_rms_sigma": cfg.maximum_fit_rms_sigma,
                "good_state_shift_z_rms": cfg.good_state_shift_z_rms,
                "warning_state_shift_z_rms": cfg.warning_state_shift_z_rms,
                "maximum_state_shift_z_rms": cfg.maximum_state_shift_z_rms,
                "camera": asdict(
                    cfg.camera
                ),
            },
        },
        "policy": {
            "source": "camera_state_refined",
            "purpose": "derived_camera",
            "is_human_anchor": False,
            "may_train_camera_state": False,
            "optimizes_raw_h_coefficients": False,
            "optimizes_projective_state": True,
            "uses_temporal_prior": True,
            "semantic_sample_density_balanced": True,
        },
    }


def should_accept_refined_camera(
    result: dict[str, Any],
) -> bool:
    return (
        isinstance(result, dict)
        and result.get("status") == "ok"
        and result.get("confidence")
        in {
            "good",
            "weak",
        }
    )
