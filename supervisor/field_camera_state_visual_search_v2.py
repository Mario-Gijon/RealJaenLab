"""Visual Projective Camera State search V2: semantic topology.

V1 proved that raw white-paint score is strongly aliased: a wrong camera can
align a few semantic pitch lines with other real white lines.

V2 keeps circles/arcs OUT of the search. It uses only straight pitch markings,
but scores the COMPLETE visible line network rather than cherry-picking the best
parallel lines:

- all visible finite semantic lines contribute;
- missing expected paint is penalised through coverage/median terms;
- x-constant and y-constant families must both score;
- real line-line intersections are scored as two-direction paint crosses;
- search stays inside a finite tube around the temporal 5-D prior.

The target-frame HUMAN geometry is never consumed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import numpy as np

from supervisor.field_camera_state_auto_lines import (
    AutoLineEvidenceConfig,
    observe_automatic_pitch_lines,
)
from supervisor.field_camera_state_projective import homography_from_state
from supervisor.field_camera_state_recovery import recover_camera_from_projective_state
from supervisor.field_camera_state_visual_search import (
    VisualStateSearchConfig,
    _line_family,
    _project_segment,
    _score_one_projected_line,
    prepare_paint_image,
)


@dataclass(frozen=True)
class VisualTopologySearchConfig:
    random_seed: int = 20260826
    maximum_prior_radius: float = 2.8
    round_scales: tuple[float, ...] = (1.5, 0.8, 0.4, 0.2)
    population_per_round: int = 90
    beam_size: int = 5
    final_candidate_count: int = 10
    minimum_state_scale: float = 0.30
    temporal_penalty: float = 0.02

    line_good_threshold: float = 0.18
    intersection_good_threshold: float = 0.16
    intersection_half_length_px: float = 13.0
    intersection_margin_px: float = 10.0

    observer_search_half_width_px: float = 12.0
    observer_minimum_confidence: float = 0.50
    observer_max_angle_delta_deg: float = 12.0


def _coarse_paint_config(cfg: VisualTopologySearchConfig) -> VisualStateSearchConfig:
    return VisualStateSearchConfig(
        random_seed=cfg.random_seed,
        round_scales=(1.0,),
        population_per_round=1,
        beam_size=1,
        final_candidate_count=1,
        minimum_state_scale=cfg.minimum_state_scale,
        maximum_prior_radius=cfg.maximum_prior_radius,
        samples_per_line_min=8,
        samples_per_line_max=24,
        target_sample_spacing_px=20.0,
        normal_offsets_px=(-4.0, 0.0, 4.0),
        side_sample_distance_px=4.0,
        minimum_visible_line_px=22.0,
        temporal_penalty=cfg.temporal_penalty,
        observer_search_half_width_px=cfg.observer_search_half_width_px,
        observer_minimum_confidence=cfg.observer_minimum_confidence,
        observer_max_angle_delta_deg=cfg.observer_max_angle_delta_deg,
    )


def _decode_h(model: dict[str, Any], state: np.ndarray) -> np.ndarray | None:
    result = homography_from_state(model, state)
    if result.get("status") != "ok":
        return None
    try:
        h = np.asarray(result["homography"], dtype=np.float64)
    except (KeyError, TypeError, ValueError):
        return None
    return h if h.shape == (3, 3) and np.isfinite(h).all() else None


def _anchor_states(model: dict[str, Any]) -> dict[int, np.ndarray]:
    return {
        int(anchor["frame"]): np.asarray(anchor["state"], dtype=np.float64)
        for anchor in model.get("anchors", [])
        if isinstance(anchor, dict) and anchor.get("state") is not None
    }


def _state_scale(model: dict[str, Any], minimum: float) -> np.ndarray:
    values = np.asarray(list(_anchor_states(model).values()), dtype=np.float64)
    scale = np.std(values, axis=0) if len(values) else np.ones(model["dimensions"])
    return np.maximum(scale, float(minimum))


def _radius(state: np.ndarray, prior: np.ndarray, scale: np.ndarray) -> float:
    return float(np.sqrt(np.mean(((state - prior) / scale) ** 2)))


def _key(state: np.ndarray) -> tuple[float, ...]:
    return tuple(np.round(state, 6).tolist())


def _project_point(world: np.ndarray, h: np.ndarray) -> np.ndarray | None:
    q = np.asarray([world[0], world[1], 1.0], dtype=np.float64) @ h.T
    if not np.isfinite(q).all() or abs(float(q[2])) < 1e-9:
        return None
    point = q[:2] / q[2]
    return point if np.isfinite(point).all() else None


def _world_intersections(pitch_model: dict[str, Any]) -> list[dict[str, Any]]:
    lines = []
    for definition in pitch_model.get("calibration_lines", []):
        if not isinstance(definition, dict):
            continue
        world = definition.get("world") or {}
        start = world.get("start")
        end = world.get("end")
        if not isinstance(start, dict) or not isinstance(end, dict):
            continue
        family = _line_family(start, end)
        if family not in {"x_constant", "y_constant"}:
            continue
        lines.append((definition.get("id"), family, start, end))

    output = []
    for semantic_a, family_a, start_a, end_a in lines:
        for semantic_b, family_b, start_b, end_b in lines:
            if family_a != "x_constant" or family_b != "y_constant":
                continue

            x = float(start_a["x"])
            y = float(start_b["y"])

            ay0, ay1 = sorted((float(start_a["y"]), float(end_a["y"])))
            bx0, bx1 = sorted((float(start_b["x"]), float(end_b["x"])))

            if not (ay0 - 1e-6 <= y <= ay1 + 1e-6):
                continue
            if not (bx0 - 1e-6 <= x <= bx1 + 1e-6):
                continue

            output.append(
                {
                    "x_semantic": semantic_a,
                    "y_semantic": semantic_b,
                    "world": np.asarray([x, y], dtype=np.float64),
                }
            )
    return output


def _short_segment(center: np.ndarray, direction: np.ndarray, half_length: float):
    norm = float(np.linalg.norm(direction))
    if norm < 1e-9:
        return None
    unit = direction / norm
    return center - unit * half_length, center + unit * half_length


def score_homography_topology(
    paint: dict[str, np.ndarray],
    homography: np.ndarray,
    pitch_model: dict[str, Any],
    image_width: int,
    image_height: int,
    *,
    config: VisualTopologySearchConfig | None = None,
) -> dict[str, Any]:
    cfg = config or VisualTopologySearchConfig()
    coarse = _coarse_paint_config(cfg)

    visible = {}
    scores = []

    for definition in pitch_model.get("calibration_lines", []):
        if not isinstance(definition, dict):
            continue
        world = definition.get("world") or {}
        start_world = world.get("start")
        end_world = world.get("end")
        if not isinstance(start_world, dict) or not isinstance(end_world, dict):
            continue

        projected = _project_segment(
            start_world,
            end_world,
            homography,
            image_width,
            image_height,
        )
        if projected is None:
            continue

        line_score = _score_one_projected_line(
            paint,
            projected[0],
            projected[1],
            coarse,
        )
        if line_score is None:
            continue

        family = _line_family(start_world, end_world)
        semantic = definition.get("id")
        item = {
            "semantic": semantic,
            "family": family,
            "score": float(line_score["score"]),
            "start": projected[0],
            "end": projected[1],
        }
        visible[semantic] = item
        scores.append(item)

    if not scores:
        return {
            "status": "ok",
            "score": 0.0,
            "visible_line_count": 0,
            "line_coverage": 0.0,
            "intersection_count": 0,
            "intersection_coverage": 0.0,
            "line_scores": [],
            "intersection_scores": [],
        }

    values = np.asarray([item["score"] for item in scores], dtype=np.float64)
    line_mean = float(np.mean(values))
    line_median = float(np.median(values))
    line_coverage = float(np.mean(values >= cfg.line_good_threshold))

    x_values = [
        item["score"]
        for item in scores
        if item["family"] == "x_constant"
    ]
    y_values = [
        item["score"]
        for item in scores
        if item["family"] == "y_constant"
    ]
    x_mean = float(np.mean(x_values)) if x_values else 0.0
    y_mean = float(np.mean(y_values)) if y_values else 0.0
    family_balance = min(x_mean, y_mean)

    intersection_scores = []
    for intersection in _world_intersections(pitch_model):
        x_item = visible.get(intersection["x_semantic"])
        y_item = visible.get(intersection["y_semantic"])
        if x_item is None or y_item is None:
            continue

        point = _project_point(intersection["world"], homography)
        if point is None:
            continue

        margin = cfg.intersection_margin_px
        if not (
            margin <= point[0] <= image_width - margin
            and margin <= point[1] <= image_height - margin
        ):
            continue

        x_direction = x_item["end"] - x_item["start"]
        y_direction = y_item["end"] - y_item["start"]

        x_segment = _short_segment(
            point,
            x_direction,
            cfg.intersection_half_length_px,
        )
        y_segment = _short_segment(
            point,
            y_direction,
            cfg.intersection_half_length_px,
        )
        if x_segment is None or y_segment is None:
            continue

        x_score = _score_one_projected_line(
            paint,
            x_segment[0],
            x_segment[1],
            coarse,
        )
        y_score = _score_one_projected_line(
            paint,
            y_segment[0],
            y_segment[1],
            coarse,
        )
        if x_score is None or y_score is None:
            continue

        cross_score = min(
            float(x_score["score"]),
            float(y_score["score"]),
        )
        intersection_scores.append(
            {
                "x_semantic": intersection["x_semantic"],
                "y_semantic": intersection["y_semantic"],
                "score": cross_score,
            }
        )

    if intersection_scores:
        cross_values = np.asarray(
            [item["score"] for item in intersection_scores],
            dtype=np.float64,
        )
        intersection_mean = float(np.mean(cross_values))
        intersection_coverage = float(
            np.mean(cross_values >= cfg.intersection_good_threshold)
        )
    else:
        intersection_mean = 0.0
        intersection_coverage = 0.0

    topology_score = (
        0.24 * line_mean
        + 0.14 * line_median
        + 0.18 * line_coverage
        + 0.14 * family_balance
        + 0.20 * intersection_mean
        + 0.10 * intersection_coverage
    )

    if not x_values or not y_values:
        topology_score *= 0.55

    return {
        "status": "ok",
        "score": float(topology_score),
        "visible_line_count": len(scores),
        "line_mean": line_mean,
        "line_median": line_median,
        "line_coverage": line_coverage,
        "x_family_mean": x_mean,
        "y_family_mean": y_mean,
        "family_balance": family_balance,
        "intersection_count": len(intersection_scores),
        "intersection_mean": intersection_mean,
        "intersection_coverage": intersection_coverage,
        "line_scores": sorted(
            scores,
            key=lambda item: item["score"],
            reverse=True,
        ),
        "intersection_scores": sorted(
            intersection_scores,
            key=lambda item: item["score"],
            reverse=True,
        ),
    }


def search_projective_state_topology(
    model: dict[str, Any],
    prior_state: Any,
    frame_bgr: np.ndarray,
    pitch_model: dict[str, Any],
    *,
    query_frame: int | None = None,
    config: VisualTopologySearchConfig | None = None,
) -> dict[str, Any]:
    cfg = config or VisualTopologySearchConfig()
    if model.get("status") != "ok":
        return {"status": "invalid_model"}

    dimensions = int(model.get("dimensions") or 0)
    try:
        prior = np.asarray(prior_state, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return {"status": "invalid_prior_state"}
    if prior.shape != (dimensions,) or not np.isfinite(prior).all():
        return {"status": "invalid_prior_state"}

    height, width = frame_bgr.shape[:2]
    paint = prepare_paint_image(frame_bgr)
    scale = _state_scale(model, cfg.minimum_state_scale)
    rng = np.random.default_rng(cfg.random_seed + int(query_frame or 0))
    cache = {}

    def evaluate(state):
        radius = _radius(state, prior, scale)
        if radius > cfg.maximum_prior_radius:
            return None
        key = _key(state)
        if key in cache:
            return cache[key]

        h = _decode_h(model, state)
        if h is None:
            return None

        topology = score_homography_topology(
            paint,
            h,
            pitch_model,
            width,
            height,
            config=cfg,
        )
        objective = float(
            topology["score"] - cfg.temporal_penalty * radius
        )
        result = {
            "state": state.copy(),
            "homography": h,
            "prior_radius": radius,
            "topology": topology,
            "objective": objective,
        }
        cache[key] = result
        return result

    initial = [prior.copy()]
    anchors = _anchor_states(model)
    for state in anchors.values():
        if _radius(state, prior, scale) <= cfg.maximum_prior_radius:
            initial.append(state.copy())

    if query_frame is not None and anchors:
        frames = sorted(anchors)
        left = [frame for frame in frames if frame < query_frame]
        right = [frame for frame in frames if frame > query_frame]
        if left and right:
            a = anchors[left[-1]]
            b = anchors[right[0]]
            for alpha in np.linspace(0.0, 1.0, 17):
                initial.append((1.0 - alpha) * a + alpha * b)

    beam = [
        result
        for state in initial
        if (result := evaluate(state)) is not None
    ]
    beam.sort(key=lambda item: item["objective"], reverse=True)
    beam = beam[: cfg.beam_size]

    rounds = []
    for round_index, sigma in enumerate(cfg.round_scales):
        candidates = list(beam)
        centers = beam if beam else [evaluate(prior)]
        centers = [item for item in centers if item is not None]
        per_center = max(
            1,
            int(math.ceil(cfg.population_per_round / max(1, len(centers)))),
        )

        for center in centers:
            for _ in range(per_center):
                proposal = (
                    center["state"]
                    + rng.normal(0.0, float(sigma), size=dimensions) * scale
                )
                result = evaluate(proposal)
                if result is not None:
                    candidates.append(result)

        candidates.sort(key=lambda item: item["objective"], reverse=True)
        unique = []
        seen = set()
        for item in candidates:
            key = _key(item["state"])
            if key in seen:
                continue
            seen.add(key)
            unique.append(item)
            if len(unique) >= cfg.beam_size:
                break
        beam = unique

        rounds.append(
            {
                "round": round_index,
                "sigma": float(sigma),
                "evaluated_hypotheses": len(cache),
                "best_objective": (
                    float(beam[0]["objective"])
                    if beam else None
                ),
                "best_topology_score": (
                    float(beam[0]["topology"]["score"])
                    if beam else None
                ),
                "best_prior_radius": (
                    float(beam[0]["prior_radius"])
                    if beam else None
                ),
            }
        )

    ranked = sorted(
        cache.values(),
        key=lambda item: item["objective"],
        reverse=True,
    )[: cfg.final_candidate_count]

    return {
        "status": "ok",
        "candidate_count": len(cache),
        "state_scale": scale.tolist(),
        "rounds": rounds,
        "candidates": [
            {
                "state": item["state"].tolist(),
                "homography": item["homography"].tolist(),
                "prior_radius": item["prior_radius"],
                "objective": item["objective"],
                "topology_score": item["topology"]["score"],
                "visible_line_count": item["topology"]["visible_line_count"],
                "line_coverage": item["topology"]["line_coverage"],
                "intersection_count": item["topology"]["intersection_count"],
                "intersection_coverage": item["topology"]["intersection_coverage"],
                "top_lines": [
                    {
                        "semantic": line["semantic"],
                        "family": line["family"],
                        "score": line["score"],
                    }
                    for line in item["topology"]["line_scores"][:8]
                ],
            }
            for item in ranked
        ],
        "config": asdict(cfg),
    }


def recover_from_topology_search(
    model: dict[str, Any],
    prior_state: Any,
    frame_bgr: np.ndarray,
    pitch_model: dict[str, Any],
    *,
    query_frame: int | None = None,
    config: VisualTopologySearchConfig | None = None,
) -> dict[str, Any]:
    cfg = config or VisualTopologySearchConfig()
    search = search_projective_state_topology(
        model,
        prior_state,
        frame_bgr,
        pitch_model,
        query_frame=query_frame,
        config=cfg,
    )
    if search.get("status") != "ok":
        return {"status": search.get("status"), "search": search}

    height, width = frame_bgr.shape[:2]
    attempts = []

    for candidate in search.get("candidates", []):
        state = np.asarray(candidate["state"], dtype=np.float64)
        h = np.asarray(candidate["homography"], dtype=np.float64)

        auto = observe_automatic_pitch_lines(
            frame_bgr,
            h,
            pitch_model,
            width,
            height,
            config=AutoLineEvidenceConfig(
                search_half_width_px=cfg.observer_search_half_width_px,
                minimum_observer_confidence=cfg.observer_minimum_confidence,
                observer_max_angle_delta_deg=cfg.observer_max_angle_delta_deg,
            ),
        )

        attempt = {
            "candidate": candidate,
            "eligible": bool(auto.get("eligible")),
            "automatic_line_count": len(auto.get("constraints", [])),
            "automatic_semantics": auto.get("observations", []),
            "family_counts": auto.get("family_counts", {}),
            "recovery": None,
        }

        if not auto.get("eligible"):
            attempts.append(attempt)
            continue

        recovered = recover_camera_from_projective_state(
            model,
            state,
            [],
            auto["constraints"],
            [],
            prior_confidence="good",
        )

        attempt["recovery"] = {
            "status": recovered.get("status"),
            "confidence": recovered.get("confidence"),
            "state_shift_z_rms": recovered.get("state_shift_z_rms"),
            "fit_rms_sigma": recovered.get("fit_rms_sigma"),
        }

        if (
            recovered.get("status") == "ok"
            and isinstance(recovered.get("homography"), list)
        ):
            attempt["homography"] = recovered["homography"]

        attempts.append(attempt)

    accepted = [
        item
        for item in attempts
        if isinstance(item.get("homography"), list)
    ]

    if not accepted:
        return {
            "status": "insufficient_automatic_evidence",
            "confidence": "poor",
            "search": search,
            "attempts": attempts,
        }

    # Keep V2 selection tied primarily to semantic topology, then line evidence.
    accepted.sort(
        key=lambda item: (
            float(item["candidate"].get("topology_score") or 0.0),
            int(item.get("automatic_line_count") or 0),
            -float(item["candidate"].get("prior_radius") or 0.0),
        ),
        reverse=True,
    )
    best = accepted[0]

    return {
        "status": "ok",
        "confidence": best["recovery"].get("confidence"),
        "homography": best["homography"],
        "state_shift_z_rms": best["recovery"].get("state_shift_z_rms"),
        "fit_rms_sigma": best["recovery"].get("fit_rms_sigma"),
        "automatic_line_count": best["automatic_line_count"],
        "automatic_semantics": best["automatic_semantics"],
        "family_counts": best["family_counts"],
        "search_candidate": best["candidate"],
        "search": search,
        "attempts": attempts,
        "policy": {
            "mode": "projective_camera_state_visual_topology_v2",
            "uses_arcs_or_circles": False,
            "uses_target_human_geometry": False,
            "is_human_anchor": False,
            "may_train_camera_state": False,
        },
    }
