"""Safe production recovery policy for derived FIELD cameras.

This module sits ABOVE field_camera_state_refinement.py.

Validated policy
----------------
1. Points + semantic lines are the primary partial-HUMAN geometry.
2. Arcs are optional rescue evidence.
3. If points/lines already produce an acceptable Projective-State refinement,
   keep that result and do not let an arc move the camera.
4. Only use arcs when the primary candidate is unavailable or rejected and the
   arc-inclusive candidate itself passes the normal acceptance guardrails.
5. A recovered camera is always DERIVED, never a HUMAN anchor.

Why
---
Frame 115 proved that its centre-circle trace is inconsistent enough to move
the otherwise-good temporal/projective solution far off the validated state
manifold even after sample-count balancing:

    points only     -> good, shift-z ~0.305
    points + arc    -> poor, shift-z ~2.707
    arc only        -> poor

So sample balancing was necessary, but semantic candidate selection is also
needed. This wrapper implements that policy without weakening the low-level
refinement mathematics.
"""

from __future__ import annotations

from typing import Any

from supervisor.field_camera_state_refinement import (
    ProjectiveStateRefinementConfig,
    refine_projective_state_with_constraints,
    should_accept_refined_camera,
)


def _summary(
    result: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not isinstance(result, dict):
        return None

    return {
        "status": result.get("status"),
        "confidence": result.get("confidence"),
        "accepted": should_accept_refined_camera(result),
        "state_shift_z_rms": result.get("state_shift_z_rms"),
        "fit_rms_sigma": result.get("fit_rms_sigma"),
        "point_error_px": (
            result.get("diagnostics", {})
            .get("point_error_px")
        ),
        "line_error_px": (
            result.get("diagnostics", {})
            .get("line_error_px")
        ),
    }


def recover_camera_from_projective_state(
    projective_state_model: dict[str, Any],
    prior_state: Any,
    point_constraints: list[dict[str, Any]],
    line_constraints: list[dict[str, Any]],
    arc_constraints: list[dict[str, Any]],
    *,
    prior_confidence: str | None = None,
    config: ProjectiveStateRefinementConfig | None = None,
) -> dict[str, Any]:
    """Recover one derived camera with conservative semantic fallback.

    Selection policy:
    - Try points + lines first when either exists.
    - If that candidate is accepted, return it.
    - Otherwise try the same geometry plus arcs.
    - If the arc-inclusive candidate is accepted, return it as an arc rescue.
    - Otherwise return an explicit rejected result.

    This deliberately prefers trusted non-arc geometry over an arc that happens
    to reduce its own residual while pulling the camera away from the temporal
    Projective-State prior.
    """

    cfg = config or ProjectiveStateRefinementConfig()

    points = list(point_constraints or [])
    lines = list(line_constraints or [])
    arcs = list(arc_constraints or [])

    primary = None
    inclusive = None

    has_primary_geometry = bool(points or lines)

    if has_primary_geometry:
        primary = refine_projective_state_with_constraints(
            projective_state_model,
            prior_state,
            points,
            lines,
            [],
            prior_confidence=prior_confidence,
            config=cfg,
        )

        if should_accept_refined_camera(primary):
            selected = dict(primary)
            selected["recovery"] = {
                "mode": "primary_points_lines",
                "used_points": len(points),
                "used_lines": len(lines),
                "used_arcs": 0,
                "ignored_arcs": len(arcs),
                "reason": (
                    "La geometría primaria ya produce una cámara derivada "
                    "aceptable; los arcos quedan como diagnóstico opcional."
                ),
                "candidates": {
                    "primary": _summary(primary),
                    "arc_inclusive": None,
                },
            }
            return selected

    if arcs:
        inclusive = refine_projective_state_with_constraints(
            projective_state_model,
            prior_state,
            points,
            lines,
            arcs,
            prior_confidence=prior_confidence,
            config=cfg,
        )

        if should_accept_refined_camera(inclusive):
            selected = dict(inclusive)
            selected["recovery"] = {
                "mode": (
                    "arc_rescue"
                    if has_primary_geometry
                    else "arc_only"
                ),
                "used_points": len(points),
                "used_lines": len(lines),
                "used_arcs": len(arcs),
                "ignored_arcs": 0,
                "reason": (
                    "La geometría primaria no fue suficiente; "
                    "el candidato con arcos sí supera las guardas."
                ),
                "candidates": {
                    "primary": _summary(primary),
                    "arc_inclusive": _summary(inclusive),
                },
            }
            return selected

    return {
        "status": "no_acceptable_camera",
        "confidence": "poor",
        "detail": (
            "Ningún candidato de Projective Camera State supera "
            "las guardas de aceptación."
        ),
        "recovery": {
            "mode": "rejected",
            "used_points": len(points),
            "used_lines": len(lines),
            "used_arcs": len(arcs),
            "candidates": {
                "primary": _summary(primary),
                "arc_inclusive": _summary(inclusive),
            },
        },
        "policy": {
            "is_human_anchor": False,
            "may_train_camera_state": False,
        },
    }
