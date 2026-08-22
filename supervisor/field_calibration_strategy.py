"""Automatic strategy selector for manual FIELD calibration.

Why this module exists
----------------------
The Supervisor has two different geometric operations:

1. ``solve_field_camera_calibration``
   Solve a camera homography from manual evidence itself.

2. ``refine_field_camera_calibration``
   Correct an already reasonable propagated/refined homography while keeping
   unobserved degrees of freedom close to that incoming camera.

The second operation is essential when the user only marks sparse or
degenerate evidence (for example several points on the halfway line).  It is
the wrong default once the user has supplied enough independent, well-spread
manual evidence: in that case the incoming H can fight the human annotations.

This module chooses between both operations based on the *geometric strength*
of the manual evidence.  It does not modify either solver.

Principle:
    sparse / degenerate evidence -> REFINE existing H
    strong / well-spread evidence -> FULL solve from human constraints

If a requested full solve cannot be obtained, we safely fall back to
refinement when an initial homography exists.

No FastAPI or UI code lives here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Literal

import numpy as np

from supervisor.field_camera_calibration import solve_field_camera_calibration
from supervisor.field_camera_refinement import refine_field_camera_calibration


CalibrationMode = Literal["auto", "full", "refine"]


@dataclass(frozen=True)
class CalibrationStrategyConfig:
    # A point-only full solve needs at least four non-collinear points, but in
    # practice we want more evidence / spread before abandoning the incoming H.
    min_points_for_strong_full: int = 6
    min_points_for_possible_full: int = 4

    # Relative coverage of the known 105 x 68 pitch.
    strong_world_span_x_ratio: float = 0.18
    strong_world_span_y_ratio: float = 0.18
    moderate_world_span_x_ratio: float = 0.10
    moderate_world_span_y_ratio: float = 0.10

    # Relative coverage in the current image.
    strong_image_span_x_ratio: float = 0.18
    strong_image_span_y_ratio: float = 0.16
    moderate_image_span_x_ratio: float = 0.10
    moderate_image_span_y_ratio: float = 0.09

    # A mix of two independent world-line families is strong evidence because
    # it constrains both projective directions.
    min_lines_for_mixed_full: int = 2

    # Numerical collinearity threshold for manually clicked point coordinates.
    non_collinear_ratio: float = 2e-3


def _finite_xy(value: Any) -> np.ndarray | None:
    try:
        point = np.asarray(
            [float(value["x"]), float(value["y"])],
            dtype=np.float64,
        )
    except (KeyError, TypeError, ValueError):
        return None
    return point if np.isfinite(point).all() else None


def _parse_point_pairs(
    constraints: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray]:
    world: list[np.ndarray] = []
    image: list[np.ndarray] = []

    for item in constraints:
        w = _finite_xy(item.get("world", {}))
        i = _finite_xy(item.get("image", {}))
        if w is None or i is None:
            continue
        world.append(w)
        image.append(i)

    if not world:
        empty = np.empty((0, 2), dtype=np.float64)
        return empty, empty.copy()

    return (
        np.asarray(world, dtype=np.float64),
        np.asarray(image, dtype=np.float64),
    )


def _line_support(
    constraints: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, set[str]]:
    """Return endpoint clouds for coverage diagnostics only.

    Important: semantic line endpoints are NOT treated as point
    correspondences.  They are used here solely to estimate how much of the
    pitch/image the manual evidence spans.
    """

    world: list[np.ndarray] = []
    image: list[np.ndarray] = []
    families: set[str] = set()

    for item in constraints:
        w = item.get("world", {})
        i = item.get("image", {})
        ws = _finite_xy(w.get("start", {}))
        we = _finite_xy(w.get("end", {}))
        ins = _finite_xy(i.get("start", {}))
        ine = _finite_xy(i.get("end", {}))

        if ws is None or we is None or ins is None or ine is None:
            continue

        world.extend((ws, we))
        image.extend((ins, ine))

        direction = we - ws
        if abs(float(direction[0])) >= abs(float(direction[1])):
            families.add("longitudinal_x")
        else:
            families.add("transversal_y")

    if not world:
        empty = np.empty((0, 2), dtype=np.float64)
        return empty, empty.copy(), families

    return (
        np.asarray(world, dtype=np.float64),
        np.asarray(image, dtype=np.float64),
        families,
    )


def _non_collinear(points: np.ndarray, ratio_threshold: float) -> bool:
    if len(points) < 3:
        return False

    centered = points - np.mean(points, axis=0)
    try:
        singular = np.linalg.svd(centered, compute_uv=False)
    except np.linalg.LinAlgError:
        return False

    if len(singular) < 2 or singular[0] <= 1e-12:
        return False

    return bool(singular[1] / singular[0] >= ratio_threshold)


def _span_ratio(
    points: np.ndarray,
    axis: int,
    denominator: float,
) -> float:
    if len(points) == 0 or denominator <= 0.0:
        return 0.0
    return float(np.ptp(points[:, axis]) / denominator)


def _combined_cloud(
    point_cloud: np.ndarray,
    line_cloud: np.ndarray,
) -> np.ndarray:
    if len(point_cloud) and len(line_cloud):
        return np.vstack((point_cloud, line_cloud))
    if len(point_cloud):
        return point_cloud.copy()
    if len(line_cloud):
        return line_cloud.copy()
    return np.empty((0, 2), dtype=np.float64)


def assess_manual_calibration_evidence(
    point_constraints: list[dict[str, Any]],
    line_constraints: list[dict[str, Any]],
    arc_constraints: list[dict[str, Any]],
    image_width: int | None,
    image_height: int | None,
    *,
    config: CalibrationStrategyConfig | None = None,
) -> dict[str, Any]:
    """Describe whether current manual evidence can safely dominate the prior."""

    cfg = config or CalibrationStrategyConfig()

    width = max(float(image_width or 0), 1.0)
    height = max(float(image_height or 0), 1.0)

    point_world, point_image = _parse_point_pairs(point_constraints)
    line_world, line_image, line_families = _line_support(line_constraints)

    combined_world = _combined_cloud(point_world, line_world)
    combined_image = _combined_cloud(point_image, line_image)

    point_world_non_collinear = _non_collinear(
        point_world,
        cfg.non_collinear_ratio,
    )
    point_image_non_collinear = _non_collinear(
        point_image,
        cfg.non_collinear_ratio,
    )
    point_geometry_independent = (
        point_world_non_collinear and point_image_non_collinear
    )

    world_span_x = _span_ratio(combined_world, 0, 105.0)
    world_span_y = _span_ratio(combined_world, 1, 68.0)
    image_span_x = _span_ratio(combined_image, 0, width)
    image_span_y = _span_ratio(combined_image, 1, height)

    point_count = len(point_world)
    line_count = len(line_world) // 2
    arc_count = len(arc_constraints)
    line_family_count = len(line_families)

    strong_world_spread = (
        world_span_x >= cfg.strong_world_span_x_ratio
        and world_span_y >= cfg.strong_world_span_y_ratio
    )
    moderate_world_spread = (
        world_span_x >= cfg.moderate_world_span_x_ratio
        and world_span_y >= cfg.moderate_world_span_y_ratio
    )
    strong_image_spread = (
        image_span_x >= cfg.strong_image_span_x_ratio
        and image_span_y >= cfg.strong_image_span_y_ratio
    )
    moderate_image_spread = (
        image_span_x >= cfg.moderate_image_span_x_ratio
        and image_span_y >= cfg.moderate_image_span_y_ratio
    )

    # Case A: many independent point correspondences spread through the view.
    strong_points = (
        point_count >= cfg.min_points_for_strong_full
        and point_geometry_independent
        and moderate_world_spread
        and moderate_image_spread
    )

    # Case B: four+ independent points plus broad spatial support.
    broad_points = (
        point_count >= cfg.min_points_for_possible_full
        and point_geometry_independent
        and strong_world_spread
        and strong_image_spread
    )

    # Case C: mixed points + lines from both field directions. This is often
    # more informative than adding many nearby points.
    strong_mixed = (
        line_count >= cfg.min_lines_for_mixed_full
        and line_family_count >= 2
        and point_count >= 2
        and moderate_world_spread
        and moderate_image_spread
    )

    # Case D: multiple independent semantic lines plus at least one extra
    # geometric cue (point or arc). Conservative, because line-only cases can
    # still leave projective ambiguity depending on layout.
    line_dominant = (
        line_count >= 4
        and line_family_count >= 2
        and (point_count >= 1 or arc_count >= 1)
        and strong_world_spread
        and moderate_image_spread
    )

    strong_for_full = bool(
        strong_points
        or broad_points
        or strong_mixed
        or line_dominant
    )

    reasons: list[str] = []
    if strong_points:
        reasons.append("muchos_puntos_independientes")
    if broad_points:
        reasons.append("puntos_bien_repartidos")
    if strong_mixed:
        reasons.append("puntos_y_lineas_de_dos_familias")
    if line_dominant:
        reasons.append("lineas_semanticas_fuertes_y_repartidas")

    if not strong_for_full:
        if not point_geometry_independent and point_count >= 3:
            reasons.append("puntos_colineales_o_casi_colineales")
        if line_family_count < 2 and line_count:
            reasons.append("lineas_de_una_sola_familia")
        if not moderate_world_spread:
            reasons.append("poca_cobertura_del_campo")
        if not moderate_image_spread:
            reasons.append("poca_cobertura_de_la_imagen")
        if point_count < 4 and line_count < 2:
            reasons.append("evidencia_manual_escasa")

    # A compact score is useful for UI/debug only; the boolean decision above
    # remains rule based and interpretable.
    score = 0.0
    score += min(1.0, point_count / 8.0) * 0.26
    score += min(1.0, line_count / 4.0) * 0.20
    score += min(1.0, arc_count / 2.0) * 0.06
    score += (1.0 if point_geometry_independent else 0.0) * 0.14
    score += min(1.0, line_family_count / 2.0) * 0.12
    score += min(1.0, world_span_x / 0.45) * 0.06
    score += min(1.0, world_span_y / 0.45) * 0.06
    score += min(1.0, image_span_x / 0.45) * 0.05
    score += min(1.0, image_span_y / 0.45) * 0.05

    return {
        "strong_for_full_solve": strong_for_full,
        "evidence_score": float(max(0.0, min(1.0, score))),
        "reasons": reasons,
        "points": point_count,
        "lines": line_count,
        "arcs": arc_count,
        "line_families": sorted(line_families),
        "point_geometry_independent": point_geometry_independent,
        "coverage": {
            "world_span_x_ratio": world_span_x,
            "world_span_y_ratio": world_span_y,
            "image_span_x_ratio": image_span_x,
            "image_span_y_ratio": image_span_y,
        },
    }


def _is_solved(result: dict[str, Any] | None) -> bool:
    return bool(
        isinstance(result, dict)
        and result.get("status") in {"solved", "solved_with_warnings"}
        and result.get("homography") is not None
    )


def _attach_strategy(
    result: dict[str, Any],
    *,
    requested_mode: str,
    selected_mode: str,
    evidence: dict[str, Any],
    fallback_from: str | None = None,
) -> dict[str, Any]:
    output = dict(result)
    diagnostics = dict(output.get("diagnostics") or {})
    diagnostics["calibration_strategy"] = {
        "requested_mode": requested_mode,
        "selected_mode": selected_mode,
        "fallback_from": fallback_from,
        "evidence": evidence,
    }
    output["diagnostics"] = diagnostics
    return output


def solve_field_camera_adaptive(
    point_constraints: list[dict[str, Any]],
    line_constraints: list[dict[str, Any]],
    arc_constraints: list[dict[str, Any]],
    image_width: int | None,
    image_height: int | None,
    *,
    initial_homography: Any | None = None,
    mode: CalibrationMode = "auto",
    strategy_config: CalibrationStrategyConfig | None = None,
) -> dict[str, Any]:
    """Choose full calibration or partial refinement automatically.

    ``mode="auto"`` is the intended production mode.

    ``mode="full"`` and ``mode="refine"`` exist for debugging / controlled
    comparison, not as a requirement for the normal UI.
    """

    if mode not in {"auto", "full", "refine"}:
        return {
            "status": "invalid_strategy",
            "detail": f"Modo de calibración no válido: {mode}",
        }

    evidence = assess_manual_calibration_evidence(
        point_constraints,
        line_constraints,
        arc_constraints,
        image_width,
        image_height,
        config=strategy_config,
    )

    has_initial = initial_homography is not None

    if mode == "full":
        selected = "full"
    elif mode == "refine":
        if not has_initial:
            # Refinement is impossible without a prior H. Full solve is the
            # only meaningful interpretation.
            selected = "full"
        else:
            selected = "refine"
    else:
        if not has_initial:
            selected = "full"
        elif evidence["strong_for_full_solve"]:
            selected = "full"
        else:
            selected = "refine"

    if selected == "full":
        full = solve_field_camera_calibration(
            point_constraints,
            line_constraints,
            arc_constraints,
            image_width,
            image_height,
        )
        if _is_solved(full):
            return _attach_strategy(
                full,
                requested_mode=mode,
                selected_mode="full",
                evidence=evidence,
            )

        # If AUTO chose a full solve but the evidence was numerically awkward,
        # preserve usability by falling back to the old refinement behaviour.
        if has_initial and mode == "auto":
            refined = refine_field_camera_calibration(
                initial_homography,
                point_constraints,
                line_constraints,
                arc_constraints,
                image_width,
                image_height,
            )
            return _attach_strategy(
                refined,
                requested_mode=mode,
                selected_mode="refine",
                evidence=evidence,
                fallback_from="full",
            )

        return _attach_strategy(
            full,
            requested_mode=mode,
            selected_mode="full",
            evidence=evidence,
        )

    refined = refine_field_camera_calibration(
        initial_homography,
        point_constraints,
        line_constraints,
        arc_constraints,
        image_width,
        image_height,
    )
    return _attach_strategy(
        refined,
        requested_mode=mode,
        selected_mode="refine",
        evidence=evidence,
    )