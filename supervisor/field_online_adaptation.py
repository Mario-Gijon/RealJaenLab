"""Online FIELD adaptation memory for the RealJaen Supervisor.

This module does not scan the full image and does not retrain a large network
inside the match. It turns human FIELD corrections into a match-specific memory
that guides a later local visual observer:

    current H -> expected semantic geometry -> similar corrected camera states
              -> small local search region -> visual confirmation -> H update

Human corrections stay authoritative. Automatic observations are proposals.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
import math
from typing import Any, Iterable

import numpy as np

PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0
CENTER_CIRCLE_RADIUS = 9.15

LANDMARKS: dict[str, tuple[float, float]] = {
    "top_left_corner": (-52.5, -34.0),
    "top_right_corner": (52.5, -34.0),
    "bottom_right_corner": (52.5, 34.0),
    "bottom_left_corner": (-52.5, 34.0),
    "halfway_top": (0.0, -34.0),
    "halfway_bottom": (0.0, 34.0),
    "centre_mark": (0.0, 0.0),
    "left_penalty_top": (-36.0, -20.16),
    "left_penalty_bottom": (-36.0, 20.16),
    "right_penalty_top": (36.0, -20.16),
    "right_penalty_bottom": (36.0, 20.16),
    "left_goal_top": (-47.0, -9.16),
    "left_goal_bottom": (-47.0, 9.16),
    "right_goal_top": (47.0, -9.16),
    "right_goal_bottom": (47.0, 9.16),
    "left_penalty_spot": (-41.5, 0.0),
    "right_penalty_spot": (41.5, 0.0),
    "left_goal_post_top": (-52.5, -3.66),
    "left_goal_post_bottom": (-52.5, 3.66),
    "right_goal_post_top": (52.5, -3.66),
    "right_goal_post_bottom": (52.5, 3.66),
    "left_penalty_goal_line_top": (-52.5, -20.16),
    "left_penalty_goal_line_bottom": (-52.5, 20.16),
    "right_penalty_goal_line_top": (52.5, -20.16),
    "right_penalty_goal_line_bottom": (52.5, 20.16),
    "left_goal_area_goal_line_top": (-52.5, -9.16),
    "left_goal_area_goal_line_bottom": (-52.5, 9.16),
    "right_goal_area_goal_line_top": (52.5, -9.16),
    "right_goal_area_goal_line_bottom": (52.5, 9.16),
    "centre_circle_halfway_top": (0.0, -9.15),
    "centre_circle_halfway_bottom": (0.0, 9.15),
}

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

PENALTY_ARC_ANGLE = math.acos(5.5 / CENTER_CIRCLE_RADIUS)
ARCS: dict[str, dict[str, Any]] = {
    "centre_circle": {
        "center": (0.0, 0.0),
        "radius": CENTER_CIRCLE_RADIUS,
        "start_angle": 0.0,
        "end_angle": 2.0 * math.pi,
    },
    "left_penalty_arc": {
        "center": (-41.5, 0.0),
        "radius": CENTER_CIRCLE_RADIUS,
        "start_angle": -PENALTY_ARC_ANGLE,
        "end_angle": PENALTY_ARC_ANGLE,
    },
    "right_penalty_arc": {
        "center": (41.5, 0.0),
        "radius": CENTER_CIRCLE_RADIUS,
        "start_angle": math.pi - PENALTY_ARC_ANGLE,
        "end_angle": math.pi + PENALTY_ARC_ANGLE,
    },
}

_CAMERA_SIGNATURE_WORLD = np.asarray(
    [
        (-52.5, -34.0), (0.0, -34.0), (52.5, -34.0),
        (-52.5, 0.0), (0.0, 0.0), (52.5, 0.0),
        (-52.5, 34.0), (0.0, 34.0), (52.5, 34.0),
        (-36.0, -20.16), (-36.0, 20.16),
        (36.0, -20.16), (36.0, 20.16),
    ],
    dtype=np.float64,
)


@dataclass(frozen=True)
class OnlineAdaptationConfig:
    point_base_radius_px: float = 18.0
    point_max_radius_px: float = 70.0
    line_base_half_width_px: float = 14.0
    line_max_half_width_px: float = 60.0
    arc_base_margin_px: float = 18.0
    arc_max_margin_px: float = 65.0
    min_similarity_for_correction: float = 0.42
    strong_similarity: float = 0.78
    max_neighbors: int = 5
    correction_gain: float = 0.85
    unknown_view_extra_radius_px: float = 28.0
    max_point_correction_px: float = 55.0
    max_line_normal_correction_px: float = 45.0
    max_line_angle_correction_deg: float = 4.0
    observations_for_mature_semantic: int = 3


def _as_h(matrix: Any) -> np.ndarray | None:
    try:
        h = np.asarray(matrix, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if h.shape != (3, 3) or not np.isfinite(h).all():
        return None
    scale = float(h[2, 2])
    if abs(scale) < 1e-12:
        return None
    h = h / scale
    try:
        if abs(float(np.linalg.det(h))) < 1e-12:
            return None
    except np.linalg.LinAlgError:
        return None
    return h


def _project_points(world_xy: np.ndarray, h: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(world_xy, dtype=np.float64).reshape(-1, 2)
    hom = np.column_stack((pts, np.ones(len(pts), dtype=np.float64)))
    q = hom @ h.T
    denom = q[:, 2]
    valid = np.isfinite(q).all(axis=1) & np.isfinite(denom) & (np.abs(denom) > 1e-9)
    out = np.full((len(pts), 2), np.nan, dtype=np.float64)
    out[valid] = q[valid, :2] / denom[valid, None]
    valid &= np.isfinite(out).all(axis=1)
    return out, valid


def _project_point(world: tuple[float, float], h: np.ndarray) -> np.ndarray | None:
    projected, valid = _project_points(np.asarray([world], dtype=np.float64), h)
    return projected[0] if bool(valid[0]) else None


def _line_from_points(p1: np.ndarray, p2: np.ndarray) -> np.ndarray | None:
    hp1 = np.asarray([float(p1[0]), float(p1[1]), 1.0], dtype=np.float64)
    hp2 = np.asarray([float(p2[0]), float(p2[1]), 1.0], dtype=np.float64)
    line = np.cross(hp1, hp2)
    norm = math.hypot(float(line[0]), float(line[1]))
    if norm < 1e-9 or not np.isfinite(line).all():
        return None
    return line / norm


def _project_world_line(world_start, world_end, h):
    projected, valid = _project_points(np.asarray([world_start, world_end], dtype=np.float64), h)
    if not np.all(valid):
        return None
    line = _line_from_points(projected[0], projected[1])
    if line is None:
        return None
    return projected[0], projected[1], line


def _normalise_line_sign(line: np.ndarray, reference: np.ndarray | None = None) -> np.ndarray:
    line = np.asarray(line, dtype=np.float64).copy()
    norm = math.hypot(float(line[0]), float(line[1]))
    if norm > 1e-12:
        line /= norm
    if reference is not None and float(np.dot(line[:2], reference[:2])) < 0.0:
        line *= -1.0
    return line


def _line_angle(line: np.ndarray) -> float:
    return math.atan2(float(-line[0]), float(line[1]))


def _wrap_half_pi(angle: float) -> float:
    while angle > math.pi / 2.0:
        angle -= math.pi
    while angle < -math.pi / 2.0:
        angle += math.pi
    return angle


def _camera_signature(h: np.ndarray, image_width: int, image_height: int) -> dict[str, Any]:
    projected, valid = _project_points(_CAMERA_SIGNATURE_WORLD, h)
    normalised = np.full_like(projected, np.nan)
    normalised[valid, 0] = projected[valid, 0] / max(float(image_width), 1.0)
    normalised[valid, 1] = projected[valid, 1] / max(float(image_height), 1.0)

    scale_world = np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    scale_proj, scale_valid = _project_points(scale_world, h)
    zoom = None
    if np.all(scale_valid):
        sx = float(np.linalg.norm(scale_proj[1] - scale_proj[0]))
        sy = float(np.linalg.norm(scale_proj[2] - scale_proj[0]))
        if sx > 0.0 and sy > 0.0:
            zoom = math.sqrt(sx * sy)

    centre = _project_point((0.0, 0.0), h)
    centre_norm = None
    if centre is not None:
        centre_norm = [
            float(centre[0] / max(float(image_width), 1.0)),
            float(centre[1] / max(float(image_height), 1.0)),
        ]

    return {
        "projection": normalised.tolist(),
        "valid": valid.astype(bool).tolist(),
        "zoom_proxy": float(zoom) if zoom is not None else None,
        "centre_norm": centre_norm,
    }


def _signature_similarity(a: dict[str, Any], b: dict[str, Any]) -> float:
    ap = np.asarray(a["projection"], dtype=np.float64)
    bp = np.asarray(b["projection"], dtype=np.float64)
    av = np.asarray(a["valid"], dtype=bool)
    bv = np.asarray(b["valid"], dtype=bool)
    common = av & bv & np.isfinite(ap).all(axis=1) & np.isfinite(bp).all(axis=1)

    if int(np.count_nonzero(common)) < 3:
        ca = a.get("centre_norm")
        cb = b.get("centre_norm")
        if ca is None or cb is None:
            return 0.0
        centre_distance = float(np.linalg.norm(np.asarray(ca) - np.asarray(cb)))
        projection_similarity = math.exp(-centre_distance / 0.32)
    else:
        distance = np.linalg.norm(ap[common] - bp[common], axis=1)
        projection_similarity = math.exp(-float(np.median(distance)) / 0.20)

    za = a.get("zoom_proxy")
    zb = b.get("zoom_proxy")
    if za and zb and za > 0.0 and zb > 0.0:
        zoom_similarity = math.exp(-abs(math.log(za / zb)) / 0.55)
    else:
        zoom_similarity = 0.75

    union = int(np.count_nonzero(av | bv))
    common_fraction = float(np.count_nonzero(common)) / max(1.0, float(union))
    overlap_bonus = 0.65 + 0.35 * common_fraction
    return float(max(0.0, min(1.0, projection_similarity * zoom_similarity * overlap_bonus)))


def _constraint_type(constraint: dict[str, Any]) -> str | None:
    kind = constraint.get("type")
    if kind in {"point", "line", "arc"}:
        return str(kind)
    world = constraint.get("world", {})
    image = constraint.get("image", {})
    if "start" in world and "end" in world and "start" in image and "end" in image:
        return "line"
    if "center" in world and "radius" in world:
        return "arc"
    if "x" in world and "y" in world and "x" in image and "y" in image:
        return "point"
    return None


def _observed_line(constraint: dict[str, Any]) -> np.ndarray | None:
    try:
        start = np.asarray([float(constraint["image"]["start"]["x"]), float(constraint["image"]["start"]["y"])])
        end = np.asarray([float(constraint["image"]["end"]["x"]), float(constraint["image"]["end"]["y"])])
    except (KeyError, TypeError, ValueError):
        return None
    return _line_from_points(start, end) if np.linalg.norm(end - start) >= 6.0 else None


def _extract_correction_example(calibration, constraint, image_width, image_height):
    semantic = constraint.get("semantic") or constraint.get("landmark")
    kind = _constraint_type(constraint)
    if not semantic or kind not in {"point", "line", "arc"}:
        return None

    solved_h = _as_h(calibration.get("homography"))
    seed_h = _as_h(calibration.get("initialHomography") or calibration.get("initial_homography"))
    if solved_h is None:
        return None

    signature_h = seed_h if seed_h is not None else solved_h
    example = {
        "frame": int(calibration.get("frame", 0)),
        "semantic": str(semantic),
        "type": kind,
        "source": "human",
        "camera_signature": _camera_signature(signature_h, image_width, image_height),
        "has_seed_correction": seed_h is not None,
        "correction": None,
    }
    if seed_h is None:
        return example

    if kind == "point":
        try:
            world = (float(constraint["world"]["x"]), float(constraint["world"]["y"]))
            observed = np.asarray([float(constraint["image"]["x"]), float(constraint["image"]["y"])])
        except (KeyError, TypeError, ValueError):
            return example
        predicted = _project_point(world, seed_h)
        if predicted is not None:
            delta = observed - predicted
            example["correction"] = {
                "delta_x_px": float(delta[0]),
                "delta_y_px": float(delta[1]),
                "magnitude_px": float(np.linalg.norm(delta)),
            }

    elif kind == "line":
        try:
            ws = (float(constraint["world"]["start"]["x"]), float(constraint["world"]["start"]["y"]))
            we = (float(constraint["world"]["end"]["x"]), float(constraint["world"]["end"]["y"]))
        except (KeyError, TypeError, ValueError):
            return example
        predicted = _project_world_line(ws, we, seed_h)
        observed = _observed_line(constraint)
        if predicted is not None and observed is not None:
            pred_line = _normalise_line_sign(predicted[2])
            obs_line = _normalise_line_sign(observed, pred_line)
            angle_delta = _wrap_half_pi(_line_angle(obs_line) - _line_angle(pred_line))
            image_centre = np.asarray([image_width / 2.0, image_height / 2.0, 1.0])
            normal_delta = float(obs_line @ image_centre) - float(pred_line @ image_centre)
            example["correction"] = {
                "normal_offset_px": normal_delta,
                "angle_delta_deg": float(math.degrees(angle_delta)),
            }
    return example


def build_field_online_profile(
    calibrations: Iterable[dict[str, Any]],
    image_width: int,
    image_height: int,
    *,
    config: OnlineAdaptationConfig | None = None,
) -> dict[str, Any]:
    """Build match-specific memory from solved manual FIELD anchors."""
    cfg = config or OnlineAdaptationConfig()
    examples = []

    for calibration in calibrations:
        if calibration.get("source", "manual") != "manual":
            continue
        if _as_h(calibration.get("homography")) is None:
            continue
        constraints = calibration.get("constraints")
        if not isinstance(constraints, list):
            constraints = []
            for item in calibration.get("correspondences", []) or []:
                converted = dict(item)
                converted.setdefault("type", "point")
                converted.setdefault("semantic", item.get("landmark"))
                constraints.append(converted)
        for constraint in constraints:
            if constraint.get("source", "manual") != "manual":
                continue
            example = _extract_correction_example(calibration, constraint, image_width, image_height)
            if example is not None:
                examples.append(example)

    stats = {}
    for example in examples:
        semantic = example["semantic"]
        entry = stats.setdefault(semantic, {
            "semantic": semantic,
            "type": example["type"],
            "human_observations": 0,
            "seed_corrections": 0,
            "frames": [],
        })
        entry["human_observations"] += 1
        entry["frames"].append(int(example["frame"]))
        if example.get("correction") is not None:
            entry["seed_corrections"] += 1

    for entry in stats.values():
        entry["maturity"] = float(min(1.0, entry["human_observations"] / max(1, cfg.observations_for_mature_semantic)))
        entry["frames"] = sorted(set(entry["frames"]))

    return {
        "format_version": 1,
        "mode": "field_online_adaptation_v1",
        "image_width": int(image_width),
        "image_height": int(image_height),
        "examples": examples,
        "semantic_stats": stats,
        "config": asdict(cfg),
    }


def _nearest_examples(profile, semantic, current_signature, max_neighbors):
    candidates = []
    for example in profile.get("examples", []):
        if example.get("semantic") != semantic:
            continue
        similarity = _signature_similarity(current_signature, example["camera_signature"])
        candidates.append((similarity, example))
    candidates.sort(key=lambda pair: pair[0], reverse=True)
    return candidates[:max_neighbors]


def _weighted_point_correction(neighbors, config):
    values, weights, frames = [], [], []
    best_similarity = 0.0
    for similarity, example in neighbors:
        correction = example.get("correction")
        if correction is None or "delta_x_px" not in correction or similarity < config.min_similarity_for_correction:
            continue
        values.append([float(correction["delta_x_px"]), float(correction["delta_y_px"])])
        weights.append(similarity ** 3)
        frames.append(int(example["frame"]))
        best_similarity = max(best_similarity, similarity)
    if not values:
        return np.zeros(2), best_similarity, []
    correction = np.average(np.asarray(values), axis=0, weights=np.asarray(weights))
    magnitude = float(np.linalg.norm(correction))
    if magnitude > config.max_point_correction_px:
        correction *= config.max_point_correction_px / magnitude
    correction *= config.correction_gain * best_similarity
    return correction, best_similarity, frames


def _weighted_line_correction(neighbors, config):
    offsets, angles, weights, frames = [], [], [], []
    best_similarity = 0.0
    for similarity, example in neighbors:
        correction = example.get("correction")
        if correction is None or "normal_offset_px" not in correction or similarity < config.min_similarity_for_correction:
            continue
        offsets.append(float(correction["normal_offset_px"]))
        angles.append(float(correction["angle_delta_deg"]))
        weights.append(similarity ** 3)
        frames.append(int(example["frame"]))
        best_similarity = max(best_similarity, similarity)
    if not offsets:
        return 0.0, 0.0, best_similarity, []
    w = np.asarray(weights)
    offset = float(np.average(np.asarray(offsets), weights=w))
    angle = float(np.average(np.asarray(angles), weights=w))
    offset = max(-config.max_line_normal_correction_px, min(config.max_line_normal_correction_px, offset))
    angle = max(-config.max_line_angle_correction_deg, min(config.max_line_angle_correction_deg, angle))
    gain = config.correction_gain * best_similarity
    return offset * gain, angle * gain, best_similarity, frames


def _clip_roi(x1, y1, x2, y2, width, height):
    left = int(max(0, min(width - 1, math.floor(min(x1, x2)))))
    top = int(max(0, min(height - 1, math.floor(min(y1, y2)))))
    right = int(max(0, min(width, math.ceil(max(x1, x2)))))
    bottom = int(max(0, min(height, math.ceil(max(y1, y2)))))
    if right <= left or bottom <= top:
        return None
    return {"x": left, "y": top, "width": right - left, "height": bottom - top}


def _line_visible_segment(line, width, height):
    a, b, c = map(float, line)
    points = []
    if abs(b) > 1e-9:
        for x in (0.0, float(width)):
            y = -(a * x + c) / b
            if 0.0 <= y <= height:
                points.append(np.asarray([x, y]))
    if abs(a) > 1e-9:
        for y in (0.0, float(height)):
            x = -(b * y + c) / a
            if 0.0 <= x <= width:
                points.append(np.asarray([x, y]))
    unique = []
    for p in points:
        if not any(np.linalg.norm(p - q) < 1.0 for q in unique):
            unique.append(p)
    if len(unique) < 2:
        return None
    best = max(((p, q) for i, p in enumerate(unique) for q in unique[i + 1:]), key=lambda pq: float(np.linalg.norm(pq[0] - pq[1])))
    return best


def _rotate_line_about_image_center(line, angle_delta_deg, normal_offset_px, width, height):
    line = _normalise_line_sign(line)
    angle = _line_angle(line) + math.radians(angle_delta_deg)
    normal = np.asarray([-math.sin(angle), math.cos(angle)])
    centre = np.asarray([width / 2.0, height / 2.0])
    signed_distance = float(line[0] * centre[0] + line[1] * centre[1] + line[2])
    new_distance = signed_distance + normal_offset_px
    c = new_distance - float(normal @ centre)
    return np.asarray([normal[0], normal[1], c])


def _point_search_candidate(semantic, world, h, profile, signature, width, height, config):
    predicted = _project_point(world, h)
    if predicted is None:
        return None
    neighbors = _nearest_examples(profile, semantic, signature, config.max_neighbors)
    correction, similarity, frames = _weighted_point_correction(neighbors, config)
    expected = predicted + correction
    radius = config.point_base_radius_px + (1.0 - similarity) * config.unknown_view_extra_radius_px
    magnitudes = [float(e["correction"]["magnitude_px"]) for s, e in neighbors if e.get("correction") and "magnitude_px" in e["correction"]]
    if magnitudes:
        radius = max(radius, float(np.percentile(magnitudes, 80)) + 8.0)
    radius = min(config.point_max_radius_px, radius)
    roi = _clip_roi(expected[0] - radius, expected[1] - radius, expected[0] + radius, expected[1] + radius, width, height)
    if roi is None:
        return None
    return {
        "semantic": semantic,
        "type": "point",
        "predicted": {"x": float(predicted[0]), "y": float(predicted[1])},
        "expected": {"x": float(expected[0]), "y": float(expected[1])},
        "search": {"kind": "point_roi", "roi": roi, "radius_px": float(radius)},
        "adaptation": {
            "best_camera_similarity": float(similarity),
            "learned_delta_px": {"x": float(correction[0]), "y": float(correction[1])},
            "learned_from_frames": frames,
        },
    }


def _line_search_candidate(semantic, world_line, h, profile, signature, width, height, config):
    projected = _project_world_line(world_line[0], world_line[1], h)
    if projected is None:
        return None
    predicted_line = projected[2]
    neighbors = _nearest_examples(profile, semantic, signature, config.max_neighbors)
    normal_offset, angle_delta, similarity, frames = _weighted_line_correction(neighbors, config)
    expected_line = _rotate_line_about_image_center(predicted_line, angle_delta, normal_offset, width, height)
    segment = _line_visible_segment(expected_line, width, height)
    if segment is None:
        return None
    half_width = config.line_base_half_width_px + (1.0 - similarity) * config.unknown_view_extra_radius_px
    prior_offsets = [abs(float(e["correction"]["normal_offset_px"])) for s, e in neighbors if e.get("correction") and "normal_offset_px" in e["correction"]]
    if prior_offsets:
        half_width = max(half_width, float(np.percentile(prior_offsets, 80)) + 7.0)
    half_width = min(config.line_max_half_width_px, half_width)
    return {
        "semantic": semantic,
        "type": "line",
        "predicted": {"line": [float(v) for v in _normalise_line_sign(predicted_line)]},
        "expected": {
            "line": [float(v) for v in _normalise_line_sign(expected_line)],
            "visible_start": {"x": float(segment[0][0]), "y": float(segment[0][1])},
            "visible_end": {"x": float(segment[1][0]), "y": float(segment[1][1])},
        },
        "search": {"kind": "line_band", "half_width_px": float(half_width)},
        "adaptation": {
            "best_camera_similarity": float(similarity),
            "learned_normal_offset_px": float(normal_offset),
            "learned_angle_delta_deg": float(angle_delta),
            "learned_from_frames": frames,
        },
    }


def _sample_arc_world(spec, count=40):
    centre = np.asarray(spec["center"])
    angles = np.linspace(float(spec["start_angle"]), float(spec["end_angle"]), count)
    return centre[None, :] + float(spec["radius"]) * np.column_stack((np.cos(angles), np.sin(angles)))


def _arc_search_candidate(semantic, spec, h, profile, signature, width, height, config):
    projected, valid = _project_points(_sample_arc_world(spec), h)
    visible = projected[valid]
    if len(visible) < 5:
        return None
    x1, y1 = np.min(visible, axis=0)
    x2, y2 = np.max(visible, axis=0)
    neighbors = _nearest_examples(profile, semantic, signature, config.max_neighbors)
    similarity = neighbors[0][0] if neighbors else 0.0
    margin = config.arc_base_margin_px + (1.0 - similarity) * config.unknown_view_extra_radius_px
    margin = min(config.arc_max_margin_px, margin)
    roi = _clip_roi(x1 - margin, y1 - margin, x2 + margin, y2 + margin, width, height)
    if roi is None:
        return None
    return {
        "semantic": semantic,
        "type": "arc",
        "predicted": {"samples": [{"x": float(p[0]), "y": float(p[1])} for p in visible]},
        "expected": {"samples": [{"x": float(p[0]), "y": float(p[1])} for p in visible]},
        "search": {"kind": "arc_roi", "roi": roi, "margin_px": float(margin)},
        "adaptation": {
            "best_camera_similarity": float(similarity),
            "learned_from_frames": [int(e["frame"]) for s, e in neighbors if s >= config.min_similarity_for_correction],
        },
    }


def suggest_local_field_searches(
    profile: dict[str, Any],
    current_homography: list[list[float]] | np.ndarray,
    image_width: int,
    image_height: int,
    *,
    include_points: bool = True,
    include_lines: bool = True,
    include_arcs: bool = True,
    only_seen_semantics: bool = False,
    config: OnlineAdaptationConfig | None = None,
) -> dict[str, Any]:
    """Predict SMALL semantic search regions around the current FIELD H.

    This performs no image detection. A later visual observer consumes these
    regions and confirms/corrects only nearby candidate markings.
    """
    cfg = config or OnlineAdaptationConfig()
    h = _as_h(current_homography)
    if h is None:
        return {"status": "invalid_homography", "detail": "La homografía actual no es válida.", "searches": []}
    width, height = int(image_width), int(image_height)
    if width <= 0 or height <= 0:
        return {"status": "invalid_image_size", "detail": "La resolución del vídeo no es válida.", "searches": []}

    signature = _camera_signature(h, width, height)
    seen = set(profile.get("semantic_stats", {}).keys())
    searches = []

    if include_points:
        for semantic, world in LANDMARKS.items():
            if only_seen_semantics and semantic not in seen:
                continue
            candidate = _point_search_candidate(semantic, world, h, profile, signature, width, height, cfg)
            if candidate is not None:
                searches.append(candidate)
    if include_lines:
        for semantic, world_line in LINES.items():
            if only_seen_semantics and semantic not in seen:
                continue
            candidate = _line_search_candidate(semantic, world_line, h, profile, signature, width, height, cfg)
            if candidate is not None:
                searches.append(candidate)
    if include_arcs:
        for semantic, spec in ARCS.items():
            if only_seen_semantics and semantic not in seen:
                continue
            candidate = _arc_search_candidate(semantic, spec, h, profile, signature, width, height, cfg)
            if candidate is not None:
                searches.append(candidate)

    def score(item):
        stats = profile.get("semantic_stats", {}).get(item["semantic"], {})
        maturity = float(stats.get("maturity", 0.0))
        similarity = float(item.get("adaptation", {}).get("best_camera_similarity", 0.0))
        type_bonus = {"line": 0.12, "point": 0.06, "arc": 0.02}.get(item["type"], 0.0)
        return 0.55 * similarity + 0.33 * maturity + type_bonus

    searches.sort(key=score, reverse=True)
    return {
        "status": "ok",
        "mode": "local_search_guidance_v1",
        "camera_signature": signature,
        "searches": searches,
        "summary": {
            "total": len(searches),
            "lines": sum(1 for item in searches if item["type"] == "line"),
            "points": sum(1 for item in searches if item["type"] == "point"),
            "arcs": sum(1 for item in searches if item["type"] == "arc"),
            "known_semantics": len(seen),
        },
    }


def summarize_field_learning(profile: dict[str, Any]) -> dict[str, Any]:
    stats = profile.get("semantic_stats", {})
    semantics = []
    for semantic, item in stats.items():
        semantics.append({
            "semantic": semantic,
            "type": item.get("type"),
            "human_observations": int(item.get("human_observations", 0)),
            "seed_corrections": int(item.get("seed_corrections", 0)),
            "maturity": float(item.get("maturity", 0.0)),
            "frames": list(item.get("frames", [])),
        })
    semantics.sort(key=lambda item: (item["maturity"], item["human_observations"]), reverse=True)
    return {
        "mode": "field_online_adaptation_v1",
        "total_human_examples": len(profile.get("examples", [])),
        "known_semantics": len(stats),
        "semantics": semantics,
    }