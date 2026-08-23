"""Conservative automatic FIELD homography refinement preview.

This module turns *already gated and visually confirmed* semantic line
observations into a proposed homography correction.

It is deliberately conservative and side-effect free:

    current H
      + finite local line observations
      + semantic consensus
      + strong camera prior
        -> proposed H

The proposal is accepted only when:
- enough independent visual evidence exists;
- the observed lines are still near the current FIELD prediction;
- a combined refinement improves the visual line errors;
- the proposed camera movement stays small and plausible.

Nothing is persisted here.  The intended first integration is DEBUG/PREVIEW:
draw the proposed pitch in yellow and compare it with the current blue pitch.

Later, once validated on real clips, the exact same proposal can become an
automatic per-frame correction policy.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
from typing import Any

import numpy as np

from supervisor.field_camera_refinement import (
    FieldCameraRefinementConfig,
    refine_field_camera_calibration,
)


# Keep the semantic IDs identical to field_online_adaptation.py / pitch.js.
LINES: dict[str, tuple[tuple[float, float], tuple[float, float]]] = {
    "halfway_line": ((0.0, -34.0), (0.0, 34.0)),
    "top_touchline": ((-52.5, -34.0), (52.5, -34.0)),
    "bottom_touchline": ((-52.5, 34.0), (52.5, 34.0)),
    "left_goal_line": ((-52.5, -34.0), (-52.5, 34.0)),
    "right_goal_line": ((52.5, -34.0), (52.5, 34.0)),
    "left_penalty_top": ((-52.5, -20.16), (-36.0, -20.16)),
    "left_penalty_front": ((-36.0, -20.16), (-36.0, 20.16)),
    "left_penalty_bottom": ((-36.0, 20.16), (-52.5, 20.16)),
    "right_penalty_top": ((52.5, -20.16), (36.0, -20.16)),
    "right_penalty_front": ((36.0, -20.16), (36.0, 20.16)),
    "right_penalty_bottom": ((36.0, 20.16), (52.5, 20.16)),
    "left_goal_top": ((-52.5, -9.16), (-47.0, -9.16)),
    "left_goal_front": ((-47.0, -9.16), (-47.0, 9.16)),
    "left_goal_bottom": ((-47.0, 9.16), (-52.5, 9.16)),
    "right_goal_top": ((52.5, -9.16), (47.0, -9.16)),
    "right_goal_front": ((47.0, -9.16), (47.0, 9.16)),
    "right_goal_bottom": ((47.0, 9.16), (52.5, 9.16)),
}


_SUPPORT_WORLD = np.asarray(
    [
        (-52.5, -34.0), (0.0, -34.0), (52.5, -34.0),
        (-52.5, -20.16), (-36.0, -20.16), (0.0, -20.16),
        (36.0, -20.16), (52.5, -20.16),
        (-52.5, 0.0), (-36.0, 0.0), (0.0, 0.0), (36.0, 0.0), (52.5, 0.0),
        (-52.5, 20.16), (-36.0, 20.16), (0.0, 20.16),
        (36.0, 20.16), (52.5, 20.16),
        (-52.5, 34.0), (0.0, 34.0), (52.5, 34.0),
        (-47.0, -9.16), (-47.0, 9.16),
        (47.0, -9.16), (47.0, 9.16),
    ],
    dtype=np.float64,
)


@dataclass(frozen=True)
class FieldAutoRefinementConfig:
    # V2 observer already rejects weak evidence below ~0.50.  We still require
    # a little more agreement before proposing a camera update.
    min_observation_confidence: float = 0.50
    strong_observation_confidence: float = 0.60

    # Two independent world-line families are the preferred minimum.  Three
    # same-family lines can also propose a cautious update.
    min_observations: int = 2
    min_same_family_observations: int = 3

    # Local visual observation must still be near the predicted search band.
    max_observation_offset_px: float = 42.0
    max_observation_angle_deg: float = 5.0

    # Automatic evidence must pull less strongly than manual evidence.
    auto_line_sigma_px: float = 2.8
    auto_prior_sigma_px: float = 16.0

    # Post-refinement visual acceptance.
    max_final_line_error_px: float = 6.0
    max_worst_final_line_error_px: float = 9.0
    min_relative_error_improvement: float = 0.12
    min_absolute_error_improvement_px: float = 0.8

    # Camera-motion safety, measured on canonical support points that are near
    # the current image. These are intentionally much stricter than manual
    # refinement's emergency limits.
    support_view_margin_ratio: float = 0.18
    max_support_median_displacement_px: float = 14.0
    max_support_p90_displacement_px: float = 30.0
    max_support_displacement_px: float = 48.0

    # Do not bother displaying a different H when the proposal barely moves.
    min_material_support_median_displacement_px: float = 0.35


def _finite_h(matrix: Any) -> np.ndarray | None:
    try:
        h = np.asarray(matrix, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if h.shape != (3, 3) or not np.isfinite(h).all():
        return None
    if abs(float(h[2, 2])) < 1e-12:
        return None
    h = h / h[2, 2]
    try:
        if abs(float(np.linalg.det(h))) < 1e-12:
            return None
    except np.linalg.LinAlgError:
        return None
    return h


def _project_points(world_xy: np.ndarray, h: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(world_xy, dtype=np.float64).reshape(-1, 2)
    hom = np.column_stack((points, np.ones(len(points), dtype=np.float64)))
    q = hom @ h.T
    denom = q[:, 2]
    valid = (
        np.isfinite(q).all(axis=1)
        & np.isfinite(denom)
        & (np.abs(denom) > 1e-9)
    )
    out = np.full((len(points), 2), np.nan, dtype=np.float64)
    out[valid] = q[valid, :2] / denom[valid, None]
    valid &= np.isfinite(out).all(axis=1)
    return out, valid


def _line_from_points(start: np.ndarray, end: np.ndarray) -> np.ndarray | None:
    line = np.cross(
        np.asarray([start[0], start[1], 1.0], dtype=np.float64),
        np.asarray([end[0], end[1], 1.0], dtype=np.float64),
    )
    norm = math.hypot(float(line[0]), float(line[1]))
    if norm < 1e-9 or not np.isfinite(line).all():
        return None
    return line / norm


def _project_world_line(
    world_start: tuple[float, float],
    world_end: tuple[float, float],
    h: np.ndarray,
) -> np.ndarray | None:
    projected, valid = _project_points(
        np.asarray([world_start, world_end], dtype=np.float64),
        h,
    )
    if not np.all(valid):
        return None
    return _line_from_points(projected[0], projected[1])


def _point(value: Any) -> np.ndarray | None:
    try:
        point = np.asarray(
            [float(value["x"]), float(value["y"])],
            dtype=np.float64,
        )
    except (KeyError, TypeError, ValueError):
        return None
    return point if np.isfinite(point).all() else None


def _line_family(world_segment) -> str:
    start = np.asarray(world_segment[0], dtype=np.float64)
    end = np.asarray(world_segment[1], dtype=np.float64)
    delta = end - start
    return "x" if abs(float(delta[0])) >= abs(float(delta[1])) else "y"


def _observation_to_constraint(
    observation: dict[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    semantic = str(observation.get("semantic", ""))
    world_segment = LINES.get(semantic)

    diagnostic = {
        "semantic": semantic,
        "status": observation.get("status"),
        "confidence": float(observation.get("confidence", 0.0) or 0.0),
        "normal_offset_px": observation.get("normal_offset_px"),
        "angle_offset_deg": observation.get("angle_offset_deg"),
    }

    if world_segment is None:
        diagnostic["rejected_reason"] = "semantic_not_supported"
        return None, diagnostic

    start = _point(observation.get("observed_segment_start", {}))
    end = _point(observation.get("observed_segment_end", {}))
    if start is None or end is None or float(np.linalg.norm(end - start)) < 8.0:
        diagnostic["rejected_reason"] = "missing_finite_observed_segment"
        return None, diagnostic

    world_start, world_end = world_segment
    constraint = {
        "type": "line",
        "semantic": semantic,
        "source": "auto_observer",
        "confidence": diagnostic["confidence"],
        "world": {
            "start": {"x": float(world_start[0]), "y": float(world_start[1])},
            "end": {"x": float(world_end[0]), "y": float(world_end[1])},
        },
        "image": {
            "start": {"x": float(start[0]), "y": float(start[1])},
            "end": {"x": float(end[0]), "y": float(end[1])},
        },
    }
    diagnostic["family"] = _line_family(world_segment)
    return constraint, diagnostic


def _line_error(
    h: np.ndarray,
    constraint: dict[str, Any],
) -> float | None:
    semantic = str(constraint.get("semantic", ""))
    world_segment = LINES.get(semantic)
    if world_segment is None:
        return None

    predicted = _project_world_line(world_segment[0], world_segment[1], h)
    if predicted is None:
        return None

    image = constraint.get("image", {})
    start = _point(image.get("start", {}))
    end = _point(image.get("end", {}))
    if start is None or end is None:
        return None

    errors = [
        abs(float(predicted[0] * p[0] + predicted[1] * p[1] + predicted[2]))
        for p in (start, end)
    ]
    return float(np.mean(errors))


def _support_displacement(
    current_h: np.ndarray,
    proposed_h: np.ndarray,
    image_width: int,
    image_height: int,
    margin_ratio: float,
) -> dict[str, Any] | None:
    current, current_valid = _project_points(_SUPPORT_WORLD, current_h)
    proposed, proposed_valid = _project_points(_SUPPORT_WORLD, proposed_h)

    margin_x = float(image_width) * margin_ratio
    margin_y = float(image_height) * margin_ratio
    nearby = (
        current_valid
        & proposed_valid
        & (current[:, 0] >= -margin_x)
        & (current[:, 0] <= image_width + margin_x)
        & (current[:, 1] >= -margin_y)
        & (current[:, 1] <= image_height + margin_y)
    )
    if int(np.count_nonzero(nearby)) < 4:
        return None

    displacement = np.linalg.norm(proposed[nearby] - current[nearby], axis=1)
    return {
        "support_points": int(len(displacement)),
        "median_px": float(np.median(displacement)),
        "p90_px": float(np.percentile(displacement, 90)),
        "max_px": float(np.max(displacement)),
    }


def _refinement_config(
    config: FieldAutoRefinementConfig,
) -> FieldCameraRefinementConfig:
    base = FieldCameraRefinementConfig()
    return replace(
        base,
        line_sigma_px=float(config.auto_line_sigma_px),
        prior_sigma_px=float(config.auto_prior_sigma_px),
        # Keep the internal emergency limits out of the way; the much stricter
        # auto-specific support-displacement checks below decide acceptance.
        max_prior_median_displacement_px=max(
            base.max_prior_median_displacement_px,
            config.max_support_median_displacement_px * 3.0,
        ),
        max_prior_p90_displacement_px=max(
            base.max_prior_p90_displacement_px,
            config.max_support_p90_displacement_px * 3.0,
        ),
    )


def propose_field_auto_refinement(
    initial_homography: Any,
    observer_result: dict[str, Any],
    image_width: int,
    image_height: int,
    *,
    config: FieldAutoRefinementConfig | None = None,
) -> dict[str, Any]:
    """Return a conservative DEBUG proposal for an automatically refined H."""

    cfg = config or FieldAutoRefinementConfig()
    current_h = _finite_h(initial_homography)
    if current_h is None:
        return {
            "status": "invalid_initial_homography",
            "proposal": None,
        }

    try:
        width = int(image_width)
        height = int(image_height)
    except (TypeError, ValueError):
        width = 0
        height = 0
    if width <= 0 or height <= 0:
        return {"status": "invalid_image_size", "proposal": None}

    observations = observer_result.get("observations", [])
    accepted_constraints: list[dict[str, Any]] = []
    accepted_meta: list[dict[str, Any]] = []
    rejected_meta: list[dict[str, Any]] = []

    for observation in observations if isinstance(observations, list) else []:
        if not isinstance(observation, dict) or observation.get("status") != "found":
            continue

        constraint, meta = _observation_to_constraint(observation)
        if constraint is None:
            rejected_meta.append(meta)
            continue

        confidence = float(meta["confidence"])
        offset = abs(float(meta.get("normal_offset_px") or 0.0))
        angle = abs(float(meta.get("angle_offset_deg") or 0.0))

        if confidence < cfg.min_observation_confidence:
            meta["rejected_reason"] = "confidence"
            rejected_meta.append(meta)
            continue
        if offset > cfg.max_observation_offset_px:
            meta["rejected_reason"] = "offset"
            rejected_meta.append(meta)
            continue
        if angle > cfg.max_observation_angle_deg:
            meta["rejected_reason"] = "angle"
            rejected_meta.append(meta)
            continue

        initial_error = _line_error(current_h, constraint)
        if initial_error is None:
            meta["rejected_reason"] = "cannot_measure_initial_error"
            rejected_meta.append(meta)
            continue

        meta["initial_error_px"] = float(initial_error)
        accepted_constraints.append(constraint)
        accepted_meta.append(meta)

    families = sorted({
        str(item.get("family"))
        for item in accepted_meta
        if item.get("family") in {"x", "y"}
    })
    strong_count = sum(
        1
        for item in accepted_meta
        if float(item["confidence"]) >= cfg.strong_observation_confidence
    )

    independent_enough = (
        len(accepted_constraints) >= cfg.min_observations
        and (
            len(families) >= 2
            or len(accepted_constraints) >= cfg.min_same_family_observations
        )
    )

    # With only two observations, ask that at least one is clearly stronger.
    confidence_enough = (
        len(accepted_constraints) >= 3
        or strong_count >= 1
    )

    evidence = {
        "found_observations": int(sum(
            1
            for item in observations
            if isinstance(item, dict) and item.get("status") == "found"
        )) if isinstance(observations, list) else 0,
        "usable_observations": len(accepted_constraints),
        "strong_observations": strong_count,
        "families": families,
        "used": accepted_meta,
        "rejected": rejected_meta,
    }

    if not independent_enough or not confidence_enough:
        return {
            "status": "insufficient_consensus",
            "proposal": None,
            "evidence": evidence,
            "detail": (
                "No hay todavía suficientes líneas visuales independientes "
                "y coherentes para mover FIELD automáticamente."
            ),
            "config": asdict(cfg),
        }

    refined = refine_field_camera_calibration(
        current_h,
        [],
        accepted_constraints,
        [],
        width,
        height,
        config=_refinement_config(cfg),
    )
    if refined.get("status") not in {"solved", "solved_with_warnings"}:
        return {
            "status": "refinement_rejected",
            "proposal": None,
            "evidence": evidence,
            "refinement": refined,
            "config": asdict(cfg),
        }

    proposed_h = _finite_h(refined.get("homography"))
    if proposed_h is None:
        return {
            "status": "refinement_rejected",
            "proposal": None,
            "evidence": evidence,
            "detail": "El refinamiento devolvió una homografía no válida.",
            "config": asdict(cfg),
        }

    initial_errors = []
    final_errors = []
    per_line = []

    for constraint, meta in zip(accepted_constraints, accepted_meta):
        before = _line_error(current_h, constraint)
        after = _line_error(proposed_h, constraint)
        if before is None or after is None:
            continue
        initial_errors.append(before)
        final_errors.append(after)
        per_line.append({
            "semantic": meta["semantic"],
            "confidence": meta["confidence"],
            "family": meta.get("family"),
            "before_error_px": float(before),
            "after_error_px": float(after),
            "improvement_px": float(before - after),
        })

    if len(final_errors) < cfg.min_observations:
        return {
            "status": "refinement_rejected",
            "proposal": None,
            "evidence": evidence,
            "detail": "No se pudo validar el error visual después del refinamiento.",
            "config": asdict(cfg),
        }

    before_mean = float(np.mean(initial_errors))
    after_mean = float(np.mean(final_errors))
    absolute_improvement = before_mean - after_mean
    relative_improvement = (
        absolute_improvement / max(before_mean, 1e-6)
    )

    movement = _support_displacement(
        current_h,
        proposed_h,
        width,
        height,
        cfg.support_view_margin_ratio,
    )
    if movement is None:
        return {
            "status": "refinement_rejected",
            "proposal": None,
            "evidence": evidence,
            "detail": "No hay suficientes puntos visibles para validar el movimiento de cámara.",
            "config": asdict(cfg),
        }

    line_quality_ok = (
        after_mean <= cfg.max_final_line_error_px
        and max(final_errors) <= cfg.max_worst_final_line_error_px
    )
    improvement_ok = (
        absolute_improvement >= cfg.min_absolute_error_improvement_px
        or relative_improvement >= cfg.min_relative_error_improvement
    )
    movement_ok = (
        movement["median_px"] <= cfg.max_support_median_displacement_px
        and movement["p90_px"] <= cfg.max_support_p90_displacement_px
        and movement["max_px"] <= cfg.max_support_displacement_px
    )

    validation = {
        "before_mean_line_error_px": before_mean,
        "after_mean_line_error_px": after_mean,
        "absolute_improvement_px": float(absolute_improvement),
        "relative_improvement": float(relative_improvement),
        "line_quality_ok": bool(line_quality_ok),
        "improvement_ok": bool(improvement_ok),
        "movement_ok": bool(movement_ok),
        "support_displacement": movement,
        "lines": per_line,
    }

    if movement["median_px"] < cfg.min_material_support_median_displacement_px:
        return {
            "status": "no_material_change",
            "proposal": None,
            "evidence": evidence,
            "validation": validation,
            "detail": "FIELD ya está prácticamente donde lo dejaría la autocorrección.",
            "config": asdict(cfg),
        }

    if not (line_quality_ok and improvement_ok and movement_ok):
        return {
            "status": "proposal_rejected",
            "proposal": None,
            "evidence": evidence,
            "validation": validation,
            "detail": (
                "Las observaciones visuales no justifican una corrección "
                "automática suficientemente segura."
            ),
            "config": asdict(cfg),
        }

    confidence_values = [
        float(item["confidence"])
        for item in accepted_meta
    ]
    mean_confidence = float(np.mean(confidence_values))
    quality = "strong" if (
        len(families) >= 2
        and strong_count >= 2
        and mean_confidence >= cfg.strong_observation_confidence
    ) else "cautious"

    return {
        "status": "proposed",
        "proposal": {
            "homography": proposed_h.tolist(),
            "quality": quality,
            "mean_observation_confidence": mean_confidence,
            "used_semantics": [item["semantic"] for item in accepted_meta],
        },
        "evidence": evidence,
        "validation": validation,
        "refinement_diagnostics": refined.get("diagnostics"),
        "detail": (
            "Propuesta FIELD calculada solo para depuración; "
            "no se ha aplicado ni guardado."
        ),
        "config": asdict(cfg),
    }