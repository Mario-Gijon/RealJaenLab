"""Automatic visual line evidence for Projective Camera State.

This module is analysis/core geometry only. It does not read reviews, videos or
persist anything.

The camera prior comes from the validated 5-D Projective Camera State. The prior
homography is used only to define small local searches for real pitch markings.
Image pixels then provide the observed line geometry.

All physical pitch geometry comes from ``build_pitch_model(length_m, width_m)``.
Nothing here uses the old 105x68 constants from online-adaptation/spatial-gating.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import numpy as np

from supervisor.field_local_observer import (
    LocalLineObserverConfig,
    observe_field_lines,
)


EPS = 1e-9


@dataclass(frozen=True)
class AutoLineEvidenceConfig:
    search_half_width_px: float = 60.0
    minimum_observer_confidence: float = 0.65
    observer_max_angle_delta_deg: float = 12.0

    minimum_projected_length_px: float = 18.0
    longitudinal_margin_px: float = 30.0
    pitch_region_extra_margin_px: float = 16.0

    projective_denominator_epsilon: float = 1e-8
    maximum_pitch_polygon_extent_factor: float = 5.0


def _as_h(value: Any) -> np.ndarray | None:
    try:
        h = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if h.shape != (3, 3) or not np.isfinite(h).all():
        return None
    if abs(float(h[2, 2])) < EPS:
        return None
    h = h / float(h[2, 2])
    return h


def _world_point(value: Any) -> np.ndarray | None:
    try:
        point = np.asarray(
            [float(value["x"]), float(value["y"])],
            dtype=np.float64,
        )
    except (KeyError, TypeError, ValueError):
        return None
    return point if np.isfinite(point).all() else None


def _json_point(point: np.ndarray) -> dict[str, float]:
    return {
        "x": float(point[0]),
        "y": float(point[1]),
    }


def _project_with_denominator(
    world_xy: np.ndarray,
    homography: np.ndarray,
) -> tuple[np.ndarray | None, float | None]:
    q = homography @ np.asarray(
        [float(world_xy[0]), float(world_xy[1]), 1.0],
        dtype=np.float64,
    )
    denominator = float(q[2])
    if (
        not np.isfinite(q).all()
        or not math.isfinite(denominator)
        or abs(denominator) < EPS
    ):
        return None, None
    point = q[:2] / denominator
    if not np.isfinite(point).all():
        return None, None
    return point, denominator


def _normalise_line(line: Any) -> np.ndarray | None:
    try:
        value = np.asarray(line, dtype=np.float64).reshape(3)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(value).all():
        return None
    norm = math.hypot(float(value[0]), float(value[1]))
    if norm < EPS:
        return None
    return value / norm


def _line_from_points(
    first: np.ndarray,
    second: np.ndarray,
) -> np.ndarray | None:
    delta = second - first
    if float(np.linalg.norm(delta)) < EPS:
        return None
    return _normalise_line(
        np.cross(
            np.asarray([first[0], first[1], 1.0], dtype=np.float64),
            np.asarray([second[0], second[1], 1.0], dtype=np.float64),
        )
    )


def _clip_segment_to_rect(
    start: np.ndarray,
    end: np.ndarray,
    left: float,
    top: float,
    right: float,
    bottom: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Liang-Barsky clip of a finite image segment."""

    delta = end - start
    p = (
        -float(delta[0]),
        float(delta[0]),
        -float(delta[1]),
        float(delta[1]),
    )
    q = (
        float(start[0] - left),
        float(right - start[0]),
        float(start[1] - top),
        float(bottom - start[1]),
    )

    t0 = 0.0
    t1 = 1.0

    for pi, qi in zip(p, q):
        if abs(pi) < EPS:
            if qi < 0.0:
                return None
            continue

        ratio = qi / pi
        if pi < 0.0:
            t0 = max(t0, ratio)
        else:
            t1 = min(t1, ratio)

        if t0 > t1:
            return None

    return (
        start + t0 * delta,
        start + t1 * delta,
    )


def _orientation_family(
    world_start: np.ndarray,
    world_end: np.ndarray,
) -> str:
    dx = abs(float(world_end[0] - world_start[0]))
    dy = abs(float(world_end[1] - world_start[1]))

    if dx <= 1e-6 and dy > 1e-6:
        return "x_constant"
    if dy <= 1e-6 and dx > 1e-6:
        return "y_constant"
    return "other"


def _project_pitch_region(
    pitch_model: dict[str, Any],
    homography: np.ndarray,
    image_width: int,
    image_height: int,
    config: AutoLineEvidenceConfig,
) -> dict[str, Any] | None:
    polygon = pitch_model.get("pitch_polygon")
    if not isinstance(polygon, list) or len(polygon) < 3:
        return None

    projected = []
    denominators = []

    for value in polygon:
        world = _world_point(value)
        if world is None:
            return None
        point, denominator = _project_with_denominator(world, homography)
        if point is None or denominator is None:
            return None
        if abs(denominator) < config.projective_denominator_epsilon:
            return None
        projected.append(point)
        denominators.append(denominator)

    # A convex pitch polygon crossing the projective pole is not a meaningful
    # image mask. In that case the local observer simply runs without a mask.
    signs = np.sign(np.asarray(denominators, dtype=np.float64))
    if np.any(signs != signs[0]):
        return None

    factor = max(1.0, float(config.maximum_pitch_polygon_extent_factor))
    min_x = -factor * float(image_width)
    max_x = (1.0 + factor) * float(image_width)
    min_y = -factor * float(image_height)
    max_y = (1.0 + factor) * float(image_height)

    safe = [
        np.asarray(
            [
                np.clip(point[0], min_x, max_x),
                np.clip(point[1], min_y, max_y),
            ],
            dtype=np.float64,
        )
        for point in projected
    ]

    return {
        "polygon": [_json_point(point) for point in safe],
        "margin_px": float(
            max(
                config.search_half_width_px
                + config.pitch_region_extra_margin_px,
                config.search_half_width_px,
            )
        ),
    }


def build_auto_line_guidance(
    prior_homography: Any,
    pitch_model: dict[str, Any],
    image_width: int,
    image_height: int,
    *,
    config: AutoLineEvidenceConfig | None = None,
) -> dict[str, Any]:
    """Build finite semantic line searches around a prior H."""

    cfg = config or AutoLineEvidenceConfig()
    h = _as_h(prior_homography)

    try:
        width = int(image_width)
        height = int(image_height)
    except (TypeError, ValueError):
        width = 0
        height = 0

    if h is None or width <= 0 or height <= 0:
        return {
            "status": "invalid_input",
            "searches": [],
            "pitch_region": None,
            "config": asdict(cfg),
        }

    half_width = max(2.0, float(cfg.search_half_width_px))
    expanded = (
        -half_width,
        -half_width,
        width + half_width,
        height + half_width,
    )

    searches = []
    rejected = []

    for line_definition in pitch_model.get("calibration_lines", []):
        if not isinstance(line_definition, dict):
            continue

        semantic = str(line_definition.get("id") or "")
        world = line_definition.get("world") or {}
        world_start = _world_point(world.get("start"))
        world_end = _world_point(world.get("end"))

        if not semantic or world_start is None or world_end is None:
            continue

        projected_start, denominator_start = _project_with_denominator(
            world_start,
            h,
        )
        projected_end, denominator_end = _project_with_denominator(
            world_end,
            h,
        )

        if (
            projected_start is None
            or projected_end is None
            or denominator_start is None
            or denominator_end is None
        ):
            rejected.append(
                {"semantic": semantic, "reason": "projection_failed"}
            )
            continue

        if (
            abs(denominator_start) < cfg.projective_denominator_epsilon
            or abs(denominator_end) < cfg.projective_denominator_epsilon
            or math.copysign(1.0, denominator_start)
            != math.copysign(1.0, denominator_end)
        ):
            rejected.append(
                {"semantic": semantic, "reason": "projective_pole_crossing"}
            )
            continue

        clipped = _clip_segment_to_rect(
            projected_start,
            projected_end,
            *expanded,
        )

        if clipped is None:
            rejected.append(
                {"semantic": semantic, "reason": "outside_searchable_image"}
            )
            continue

        physical_start, physical_end = clipped
        visible_length = float(
            np.linalg.norm(physical_end - physical_start)
        )

        if visible_length < cfg.minimum_projected_length_px:
            rejected.append(
                {"semantic": semantic, "reason": "too_short"}
            )
            continue

        expected_line = _line_from_points(
            physical_start,
            physical_end,
        )

        if expected_line is None:
            rejected.append(
                {"semantic": semantic, "reason": "line_failed"}
            )
            continue

        tangent = (
            physical_end - physical_start
        ) / max(visible_length, EPS)

        longitudinal_margin = max(
            0.0,
            float(cfg.longitudinal_margin_px),
        )

        search_start = (
            physical_start
            - tangent * longitudinal_margin
        )
        search_end = (
            physical_end
            + tangent * longitudinal_margin
        )

        searches.append(
            {
                "semantic": semantic,
                "type": "line",
                "world": {
                    "start": _json_point(world_start),
                    "end": _json_point(world_end),
                },
                "orientation_family": _orientation_family(
                    world_start,
                    world_end,
                ),
                "expected": {
                    "line": expected_line.tolist(),
                    "visible_start": _json_point(physical_start),
                    "visible_end": _json_point(physical_end),
                },
                "predicted": {
                    "line": expected_line.tolist(),
                },
                "search": {
                    "kind": "finite_line_capsule",
                    "half_width_px": half_width,
                    "finite_segment": {
                        "physical_start": _json_point(physical_start),
                        "physical_end": _json_point(physical_end),
                        "search_start": _json_point(search_start),
                        "search_end": _json_point(search_end),
                    },
                },
            }
        )

    pitch_region = _project_pitch_region(
        pitch_model,
        h,
        width,
        height,
        cfg,
    )

    if pitch_region is not None:
        for item in searches:
            item["pitch_region"] = pitch_region

    return {
        "status": "ok",
        "searches": searches,
        "pitch_region": pitch_region,
        "rejected_searches": rejected,
        "config": asdict(cfg),
    }


def automatic_constraints_from_observations(
    observer_result: dict[str, Any],
    guidance: dict[str, Any],
    *,
    minimum_confidence: float,
) -> dict[str, Any]:
    """Convert confident visual line observations to temporary constraints."""

    searches = {
        item.get("semantic"): item
        for item in guidance.get("searches", [])
        if isinstance(item, dict) and item.get("semantic")
    }

    constraints = []
    observations = []

    for observation in observer_result.get("observations", []):
        if (
            not isinstance(observation, dict)
            or observation.get("status") != "found"
        ):
            continue

        confidence = float(observation.get("confidence") or 0.0)
        if confidence < float(minimum_confidence):
            continue

        semantic = observation.get("semantic")
        search = searches.get(semantic)

        if not isinstance(search, dict):
            continue

        start = observation.get("observed_segment_start")
        end = observation.get("observed_segment_end")

        if not isinstance(start, dict) or not isinstance(end, dict):
            continue

        constraint = {
            "type": "line",
            "semantic": semantic,
            "world": search["world"],
            "image": {
                "start": {
                    "x": float(start["x"]),
                    "y": float(start["y"]),
                },
                "end": {
                    "x": float(end["x"]),
                    "y": float(end["y"]),
                },
            },
            "source": "automatic_visual_line",
            "confidence": confidence,
            "orientation_family": search.get("orientation_family"),
        }

        constraints.append(constraint)
        observations.append(
            {
                "semantic": semantic,
                "confidence": confidence,
                "orientation_family": search.get("orientation_family"),
                "observed_line": observation.get("observed_line"),
                "normal_offset_px": observation.get("normal_offset_px"),
                "angle_offset_deg": observation.get("angle_offset_deg"),
                "support_length_px": observation.get("support_length_px"),
                "paint_score": observation.get("paint_score"),
            }
        )

    families = {
        "x_constant": sum(
            1
            for item in constraints
            if item.get("orientation_family") == "x_constant"
        ),
        "y_constant": sum(
            1
            for item in constraints
            if item.get("orientation_family") == "y_constant"
        ),
    }

    eligible = (
        len(constraints) >= 2
        and families["x_constant"] >= 1
        and families["y_constant"] >= 1
    )

    return {
        "status": "ok",
        "eligible": eligible,
        "constraints": constraints,
        "observations": observations,
        "family_counts": families,
        "reason": (
            None
            if eligible
            else "need_two_lines_and_both_orientation_families"
        ),
    }


def observe_automatic_pitch_lines(
    frame_bgr: np.ndarray,
    prior_homography: Any,
    pitch_model: dict[str, Any],
    image_width: int,
    image_height: int,
    *,
    config: AutoLineEvidenceConfig | None = None,
) -> dict[str, Any]:
    """Observe real pitch lines near a prior H and return temporary constraints."""

    cfg = config or AutoLineEvidenceConfig()

    guidance = build_auto_line_guidance(
        prior_homography,
        pitch_model,
        image_width,
        image_height,
        config=cfg,
    )

    if guidance.get("status") != "ok":
        return {
            "status": guidance.get("status"),
            "eligible": False,
            "guidance": guidance,
            "observer": None,
            "constraints": [],
            "observations": [],
            "family_counts": {
                "x_constant": 0,
                "y_constant": 0,
            },
        }

    observer_config = LocalLineObserverConfig(
        max_angle_delta_deg=float(
            cfg.observer_max_angle_delta_deg
        ),
        min_confidence=float(
            cfg.minimum_observer_confidence
        ),
    )

    observer = observe_field_lines(
        frame_bgr,
        guidance,
        config=observer_config,
    )

    converted = automatic_constraints_from_observations(
        observer,
        guidance,
        minimum_confidence=cfg.minimum_observer_confidence,
    )

    return {
        "status": "ok",
        "eligible": converted["eligible"],
        "guidance": guidance,
        "observer": observer,
        "constraints": converted["constraints"],
        "observations": converted["observations"],
        "family_counts": converted["family_counts"],
        "reason": converted["reason"],
        "config": asdict(cfg),
    }
