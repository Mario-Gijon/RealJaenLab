"""Visual state-space search for Projective Camera State.

Analysis/core algorithm only.

The temporal 5-D state is a prior, not the answer. This module searches nearby
Projective Camera State hypotheses using raw image paint evidence, then uses the
existing local semantic line observer for precise constraints and the validated
Projective-State recovery for the final H.

No HUMAN geometry from the target frame is consumed.
No review mutation lives here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import cv2
import numpy as np

from supervisor.field_camera_state_auto_lines import (
    AutoLineEvidenceConfig,
    observe_automatic_pitch_lines,
)
from supervisor.field_camera_state_projective import homography_from_state
from supervisor.field_camera_state_recovery import (
    recover_camera_from_projective_state,
)


EPS = 1e-9


@dataclass(frozen=True)
class VisualStateSearchConfig:
    random_seed: int = 20260826
    round_scales: tuple[float, ...] = (2.4, 1.2, 0.6, 0.3)
    population_per_round: int = 120
    beam_size: int = 4
    final_candidate_count: int = 8
    minimum_state_scale: float = 0.30
    maximum_prior_radius: float = 5.0

    samples_per_line_min: int = 12
    samples_per_line_max: int = 42
    target_sample_spacing_px: float = 12.0
    normal_offsets_px: tuple[float, ...] = (-6.0, -3.0, 0.0, 3.0, 6.0)
    side_sample_distance_px: float = 4.0
    minimum_visible_line_px: float = 24.0
    temporal_penalty: float = 0.025

    observer_search_half_width_px: float = 12.0
    observer_minimum_confidence: float = 0.50
    observer_max_angle_delta_deg: float = 12.0


def _as_state(value: Any, dimensions: int) -> np.ndarray | None:
    try:
        state = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    if state.shape != (dimensions,) or not np.isfinite(state).all():
        return None
    return state


def _decode_h(model: dict[str, Any], state: np.ndarray) -> np.ndarray | None:
    result = homography_from_state(model, state)
    if result.get("status") != "ok":
        return None
    try:
        h = np.asarray(result["homography"], dtype=np.float64)
    except (KeyError, TypeError, ValueError):
        return None
    if h.shape != (3, 3) or not np.isfinite(h).all():
        return None
    return h


def _clip_segment(
    start: np.ndarray,
    end: np.ndarray,
    width: int,
    height: int,
    margin: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    left = -margin
    top = -margin
    right = float(width) + margin
    bottom = float(height) + margin

    delta = end - start
    p = (-float(delta[0]), float(delta[0]), -float(delta[1]), float(delta[1]))
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

    return start + t0 * delta, start + t1 * delta


def _project_segment(
    world_start: dict[str, float],
    world_end: dict[str, float],
    h: np.ndarray,
    width: int,
    height: int,
    *,
    margin: float = 8.0,
) -> tuple[np.ndarray, np.ndarray] | None:
    world = np.asarray(
        [
            [float(world_start["x"]), float(world_start["y"])],
            [float(world_end["x"]), float(world_end["y"])],
        ],
        dtype=np.float64,
    )
    q = np.column_stack((world, np.ones(2))) @ h.T
    d = q[:, 2]

    if (
        not np.isfinite(q).all()
        or np.any(np.abs(d) < 1e-8)
        or math.copysign(1.0, float(d[0])) != math.copysign(1.0, float(d[1]))
    ):
        return None

    image = q[:, :2] / d[:, None]
    if not np.isfinite(image).all():
        return None

    return _clip_segment(image[0], image[1], width, height, margin)


def _bilinear_sample(image: np.ndarray, points: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return np.empty((0,), dtype=np.float32)
    return cv2.remap(
        image,
        points[:, 0].astype(np.float32).reshape(-1, 1),
        points[:, 1].astype(np.float32).reshape(-1, 1),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    ).reshape(-1)


def prepare_paint_image(frame_bgr: np.ndarray) -> dict[str, np.ndarray]:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1].astype(np.float32)
    return {"gray": gray, "saturation": saturation}


def _line_family(world_start: dict[str, float], world_end: dict[str, float]) -> str:
    dx = abs(float(world_end["x"]) - float(world_start["x"]))
    dy = abs(float(world_end["y"]) - float(world_start["y"]))
    if dx <= 1e-6 and dy > 1e-6:
        return "x_constant"
    if dy <= 1e-6 and dx > 1e-6:
        return "y_constant"
    return "other"


def _score_one_projected_line(
    paint: dict[str, np.ndarray],
    start: np.ndarray,
    end: np.ndarray,
    config: VisualStateSearchConfig,
) -> dict[str, float] | None:
    delta = end - start
    length = float(np.linalg.norm(delta))
    if length < config.minimum_visible_line_px:
        return None

    direction = delta / max(length, EPS)
    normal = np.asarray([-direction[1], direction[0]], dtype=np.float64)

    count = int(
        np.clip(
            round(length / max(config.target_sample_spacing_px, 1.0)),
            config.samples_per_line_min,
            config.samples_per_line_max,
        )
    )
    t = np.linspace(0.08, 0.92, count)[:, None]
    base = start[None, :] + delta[None, :] * t

    gray = paint["gray"]
    saturation = paint["saturation"]
    side = float(config.side_sample_distance_px)
    best = None

    for offset in config.normal_offsets_px:
        centre = base + normal[None, :] * float(offset)
        side_a = centre + normal[None, :] * side
        side_b = centre - normal[None, :] * side

        cg = _bilinear_sample(gray, centre)
        ag = _bilinear_sample(gray, side_a)
        bg = _bilinear_sample(gray, side_b)
        cs = _bilinear_sample(saturation, centre)
        ass = _bilinear_sample(saturation, side_a)
        bs = _bilinear_sample(saturation, side_b)

        contrast_samples = cg - 0.5 * (ag + bg)
        desat_samples = 0.5 * (ass + bs) - cs

        contrast = float(np.median(contrast_samples))
        desaturation = float(np.median(desat_samples))
        brightness = float(np.median(cg))
        paint_fraction = float(np.mean(contrast_samples > 2.0))

        contrast_score = float(np.clip((contrast - 0.5) / 16.0, 0.0, 1.0))
        desat_score = float(np.clip(desaturation / 45.0, 0.0, 1.0))
        brightness_score = float(np.clip((brightness - 70.0) / 130.0, 0.0, 1.0))
        fraction_score = float(np.clip(paint_fraction / 0.55, 0.0, 1.0))

        score = (
            0.42 * contrast_score
            + 0.22 * desat_score
            + 0.16 * brightness_score
            + 0.20 * fraction_score
        )
        score = max(0.0, score - 0.015 * abs(float(offset)))

        candidate = {
            "score": float(score),
            "contrast": contrast,
            "desaturation": desaturation,
            "brightness": brightness,
            "paint_fraction": paint_fraction,
            "best_normal_offset_px": float(offset),
            "visible_length_px": length,
        }
        if best is None or candidate["score"] > best["score"]:
            best = candidate

    return best


def score_homography_paint(
    paint: dict[str, np.ndarray],
    homography: np.ndarray,
    pitch_model: dict[str, Any],
    image_width: int,
    image_height: int,
    *,
    config: VisualStateSearchConfig | None = None,
) -> dict[str, Any]:
    cfg = config or VisualStateSearchConfig()
    line_scores = []

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

        scored = _score_one_projected_line(paint, projected[0], projected[1], cfg)
        if scored is None:
            continue

        line_scores.append(
            {
                "semantic": definition.get("id"),
                "family": _line_family(start_world, end_world),
                **scored,
            }
        )

    x_scores = sorted(
        [item["score"] for item in line_scores if item["family"] == "x_constant"],
        reverse=True,
    )
    y_scores = sorted(
        [item["score"] for item in line_scores if item["family"] == "y_constant"],
        reverse=True,
    )

    def family_score(values):
        return float(np.mean(values[:3])) if values else 0.0

    x_score = family_score(x_scores)
    y_score = family_score(y_scores)
    balanced = 0.50 * (x_score + y_score) + 0.50 * min(x_score, y_score)
    support = sum(1 for item in line_scores if item["score"] >= 0.18)
    support_bonus = min(0.10, 0.02 * support)

    return {
        "status": "ok",
        "score": float(balanced + support_bonus),
        "x_family_score": x_score,
        "y_family_score": y_score,
        "supporting_lines": support,
        "line_scores": sorted(line_scores, key=lambda item: item["score"], reverse=True),
    }


def _anchor_states(model: dict[str, Any]) -> dict[int, np.ndarray]:
    return {
        int(anchor["frame"]): np.asarray(anchor["state"], dtype=np.float64)
        for anchor in model.get("anchors", [])
        if isinstance(anchor, dict) and anchor.get("state") is not None
    }


def _state_scale(model: dict[str, Any], config: VisualStateSearchConfig) -> np.ndarray:
    values = np.asarray(list(_anchor_states(model).values()), dtype=np.float64)
    scale = np.std(values, axis=0) if len(values) else np.ones(model["dimensions"])
    return np.maximum(scale, float(config.minimum_state_scale))


def _prior_radius(state: np.ndarray, prior: np.ndarray, scale: np.ndarray) -> float:
    return float(np.sqrt(np.mean(((state - prior) / scale) ** 2)))


def _key(state: np.ndarray) -> tuple[float, ...]:
    return tuple(np.round(state, 6).tolist())


def search_projective_state_from_image(
    projective_state_model: dict[str, Any],
    temporal_prior_state: Any,
    frame_bgr: np.ndarray,
    pitch_model: dict[str, Any],
    *,
    query_frame: int | None = None,
    config: VisualStateSearchConfig | None = None,
) -> dict[str, Any]:
    cfg = config or VisualStateSearchConfig()
    if projective_state_model.get("status") != "ok":
        return {"status": "invalid_model"}

    dimensions = int(projective_state_model.get("dimensions") or 0)
    prior = _as_state(temporal_prior_state, dimensions)
    if prior is None:
        return {"status": "invalid_prior_state"}

    height, width = frame_bgr.shape[:2]
    paint = prepare_paint_image(frame_bgr)
    scale = _state_scale(projective_state_model, cfg)
    rng = np.random.default_rng(cfg.random_seed + int(query_frame or 0))
    cache = {}

    def evaluate(state):
        radius = _prior_radius(state, prior, scale)
        if radius > cfg.maximum_prior_radius:
            return None
        key = _key(state)
        if key in cache:
            return cache[key]
        h = _decode_h(projective_state_model, state)
        if h is None:
            return None
        paint_result = score_homography_paint(
            paint, h, pitch_model, width, height, config=cfg
        )
        result = {
            "state": state.copy(),
            "homography": h,
            "paint": paint_result,
            "prior_radius": radius,
            "objective": float(
                paint_result["score"] - cfg.temporal_penalty * radius
            ),
        }
        cache[key] = result
        return result

    initial = [prior.copy()]
    anchors = _anchor_states(projective_state_model)

    for state in anchors.values():
        if _prior_radius(state, prior, scale) <= cfg.maximum_prior_radius:
            initial.append(state.copy())

    if query_frame is not None and anchors:
        frames = sorted(anchors)
        left = [frame for frame in frames if frame < query_frame]
        right = [frame for frame in frames if frame > query_frame]
        if left and right:
            a = anchors[left[-1]]
            b = anchors[right[0]]
            for alpha in np.linspace(0.0, 1.0, 13):
                initial.append((1.0 - alpha) * a + alpha * b)

    beam = [result for state in initial if (result := evaluate(state)) is not None]
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
                state = (
                    center["state"]
                    + rng.normal(0.0, float(sigma), size=dimensions) * scale
                )
                result = evaluate(state)
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
                "best_objective": float(beam[0]["objective"]) if beam else None,
                "best_paint_score": float(beam[0]["paint"]["score"]) if beam else None,
                "best_prior_radius": float(beam[0]["prior_radius"]) if beam else None,
                "evaluated_hypotheses": len(cache),
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
                "objective": item["objective"],
                "paint_score": item["paint"]["score"],
                "prior_radius": item["prior_radius"],
                "x_family_score": item["paint"]["x_family_score"],
                "y_family_score": item["paint"]["y_family_score"],
                "supporting_lines": item["paint"]["supporting_lines"],
                "top_paint_lines": item["paint"]["line_scores"][:6],
            }
            for item in ranked
        ],
        "config": asdict(cfg),
    }


def recover_from_visual_state_search(
    projective_state_model: dict[str, Any],
    temporal_prior_state: Any,
    frame_bgr: np.ndarray,
    pitch_model: dict[str, Any],
    *,
    query_frame: int | None = None,
    config: VisualStateSearchConfig | None = None,
) -> dict[str, Any]:
    cfg = config or VisualStateSearchConfig()

    search = search_projective_state_from_image(
        projective_state_model,
        temporal_prior_state,
        frame_bgr,
        pitch_model,
        query_frame=query_frame,
        config=cfg,
    )
    if search.get("status") != "ok":
        return {"status": search.get("status", "search_failed"), "search": search}

    height, width = frame_bgr.shape[:2]
    paint = prepare_paint_image(frame_bgr)
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
            "automatic_line_count": len(auto.get("constraints", [])),
            "automatic_semantics": auto.get("observations", []),
            "family_counts": auto.get("family_counts", {}),
            "eligible": bool(auto.get("eligible")),
            "recovery": None,
            "post_paint_score": None,
            "selection_score": None,
        }

        if not auto.get("eligible"):
            attempts.append(attempt)
            continue

        recovered = recover_camera_from_projective_state(
            projective_state_model,
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
            "mode": recovered.get("recovery", {}).get("mode"),
        }

        if (
            recovered.get("status") != "ok"
            or not isinstance(recovered.get("homography"), list)
        ):
            attempts.append(attempt)
            continue

        recovered_h = np.asarray(recovered["homography"], dtype=np.float64)
        post = score_homography_paint(
            paint, recovered_h, pitch_model, width, height, config=cfg
        )

        confidences = [
            float(item.get("confidence") or 0.0)
            for item in auto.get("observations", [])
        ]
        mean_confidence = float(np.mean(confidences)) if confidences else 0.0
        line_count = len(auto.get("constraints", []))
        family = auto.get("family_counts", {})
        family_bonus = (
            0.12
            if family.get("x_constant", 0) and family.get("y_constant", 0)
            else 0.0
        )

        attempt["post_paint_score"] = float(post["score"])
        attempt["selection_score"] = float(
            post["score"]
            + 0.025 * min(line_count, 6)
            + 0.10 * mean_confidence
            + family_bonus
        )
        attempt["homography"] = recovered["homography"]
        attempt["state"] = recovered.get("state")
        attempts.append(attempt)

    accepted = [
        item
        for item in attempts
        if isinstance(item.get("homography"), list)
        and item.get("recovery", {}).get("status") == "ok"
    ]

    if not accepted:
        return {
            "status": "insufficient_automatic_evidence",
            "confidence": "poor",
            "search": search,
            "attempts": attempts,
            "homography": None,
        }

    accepted.sort(
        key=lambda item: float(item.get("selection_score") or -1e9),
        reverse=True,
    )
    best = accepted[0]

    return {
        "status": "ok",
        "confidence": best["recovery"].get("confidence"),
        "homography": best["homography"],
        "state": best.get("state"),
        "state_shift_z_rms": best["recovery"].get("state_shift_z_rms"),
        "fit_rms_sigma": best["recovery"].get("fit_rms_sigma"),
        "automatic_line_count": best["automatic_line_count"],
        "automatic_semantics": best["automatic_semantics"],
        "family_counts": best["family_counts"],
        "search_candidate": best["candidate"],
        "post_paint_score": best["post_paint_score"],
        "selection_score": best["selection_score"],
        "search": search,
        "attempts": attempts,
        "policy": {
            "source": "camera_state_visual_search",
            "purpose": "derived_camera",
            "is_human_anchor": False,
            "may_train_camera_state": False,
            "uses_target_human_geometry": False,
        },
    }
