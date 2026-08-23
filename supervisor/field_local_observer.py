"""Finite local visual observer for semantic football-pitch lines.

V2 fixes the two main false-positive paths from V1:
- semantic pitch markings are finite segments, not infinite image lines;
- evidence must remain inside/near the projected playing surface.

It consumes guidance upgraded by
``field_spatial_gating.apply_field_spatial_gating``.

Public API remains compatible:
    observe_field_lines(frame_bgr, guidance)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Iterable

import cv2
import numpy as np


@dataclass(frozen=True)
class LocalLineObserverConfig:
    working_scale: float = 0.65
    max_angle_delta_deg: float = 8.0
    min_segment_length_px: float = 10.0
    min_total_support_px: float = 28.0
    offset_cluster_width_px: float = 3.5
    side_sample_distance_px: float = 4.0
    min_local_contrast: float = 1.0
    min_paint_score: float = 0.16
    min_confidence: float = 0.50
    max_offset_fraction_of_band: float = 0.95
    min_segment_capsule_fraction: float = 0.58
    min_segment_pitch_fraction: float = 0.58
    crop_padding_px: float = 8.0
    max_debug_segments: int = 20


def _normalise_line(line: Any) -> np.ndarray | None:
    try:
        arr = np.asarray(line, dtype=np.float64).reshape(3)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(arr).all():
        return None
    norm = math.hypot(float(arr[0]), float(arr[1]))
    return arr / norm if norm > 1e-9 else None


def _line_from_segment(segment):
    x1, y1, x2, y2 = segment
    return _normalise_line(np.cross(
        np.asarray([x1, y1, 1.0]),
        np.asarray([x2, y2, 1.0]),
    ))


def _line_direction(line):
    d = np.asarray([-line[1], line[0]], dtype=np.float64)
    n = float(np.linalg.norm(d))
    return d / n if n > 1e-9 else np.asarray([1.0, 0.0])


def _angle_difference_deg(first, second):
    cosine = float(np.clip(abs(np.dot(_line_direction(first), _line_direction(second))), 0.0, 1.0))
    return float(math.degrees(math.acos(cosine)))


def _signed_distance(line, point):
    return float(line[0] * point[0] + line[1] * point[1] + line[2])


def _point(value):
    try:
        p = np.asarray([float(value["x"]), float(value["y"])], dtype=np.float64)
    except (KeyError, TypeError, ValueError):
        return None
    return p if np.isfinite(p).all() else None


def _line_native_to_work(line, scale):
    return _normalise_line([line[0], line[1], line[2] * scale])


def _line_work_to_native(line, scale):
    return _normalise_line([line[0], line[1], line[2] / scale]) if scale > 0 else None


def _point_segment_distance(points, start, end):
    delta = end - start
    length2 = float(np.dot(delta, delta))
    if length2 < 1e-9:
        return np.linalg.norm(points - start[None, :], axis=1)
    t = ((points - start[None, :]) @ delta) / length2
    t = np.clip(t, 0.0, 1.0)
    closest = start[None, :] + t[:, None] * delta[None, :]
    return np.linalg.norm(points - closest, axis=1)


def _sample_segment(segment, count=13):
    x1, y1, x2, y2 = segment
    start = np.asarray([x1, y1], dtype=np.float64)
    end = np.asarray([x2, y2], dtype=np.float64)
    t = np.linspace(0.0, 1.0, count)[:, None]
    return start[None, :] + (end - start)[None, :] * t


def _capsule_fraction(segment, search_start, search_end, half_width):
    distances = _point_segment_distance(
        _sample_segment(segment), search_start, search_end
    )
    return float(np.mean(distances <= half_width))


def _sample_image(image, points):
    if len(points) == 0:
        return np.empty((0,), dtype=np.float32)
    return cv2.remap(
        image,
        points[:, 0].astype(np.float32).reshape(-1, 1),
        points[:, 1].astype(np.float32).reshape(-1, 1),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    ).reshape(-1)


def _paint_score(gray, saturation, segment, line, config):
    x1, y1, x2, y2 = segment
    start = np.asarray([x1, y1], dtype=np.float64)
    end = np.asarray([x2, y2], dtype=np.float64)
    length = float(np.linalg.norm(end - start))
    if length < 3.0:
        return 0.0, {}

    count = max(10, min(64, int(length / 3.0)))
    t = np.linspace(0.08, 0.92, count)[:, None]
    centre = start[None, :] + (end - start)[None, :] * t
    normal = line[:2]
    off = float(config.side_sample_distance_px)

    cg = _sample_image(gray, centre)
    ag = _sample_image(gray, centre + normal[None, :] * off)
    bg = _sample_image(gray, centre - normal[None, :] * off)
    cs = _sample_image(saturation, centre)
    ass = _sample_image(saturation, centre + normal[None, :] * off)
    bs = _sample_image(saturation, centre - normal[None, :] * off)

    contrast = float(np.median(cg - 0.5 * (ag + bg)))
    desaturation = float(np.median(0.5 * (ass + bs) - cs))
    brightness = float(np.median(cg))

    contrast_score = float(np.clip((contrast - config.min_local_contrast) / 22.0, 0.0, 1.0))
    desat_score = float(np.clip(desaturation / 55.0, 0.0, 1.0))
    brightness_score = float(np.clip((brightness - 70.0) / 125.0, 0.0, 1.0))
    score = 0.50 * contrast_score + 0.30 * desat_score + 0.20 * brightness_score
    return score, {
        "contrast": contrast,
        "desaturation": desaturation,
        "brightness": brightness,
    }


def _pitch_mask(shape, pitch_region, scale):
    if not pitch_region:
        return None
    values = pitch_region.get("polygon")
    if not isinstance(values, list) or len(values) < 3:
        return None

    points = []
    for value in values:
        p = _point(value)
        if p is None:
            return None
        points.append(p * scale)

    polygon = np.rint(np.asarray(points)).astype(np.int32)
    height, width = shape
    polygon[:, 0] = np.clip(polygon[:, 0], -2 * width, 3 * width)
    polygon[:, 1] = np.clip(polygon[:, 1], -2 * height, 3 * height)

    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillConvexPoly(mask, polygon, 255)

    margin = int(max(0, round(float(pitch_region.get("margin_px", 0.0)) * scale)))
    if margin:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * margin + 1, 2 * margin + 1)
        )
        mask = cv2.dilate(mask, kernel)
    return mask


def _pitch_fraction(segment, mask):
    if mask is None:
        return 1.0
    pts = _sample_segment(segment)
    x = np.rint(pts[:, 0]).astype(int)
    y = np.rint(pts[:, 1]).astype(int)
    valid = (x >= 0) & (x < mask.shape[1]) & (y >= 0) & (y < mask.shape[0])
    inside = np.zeros(len(pts), dtype=bool)
    indices = np.where(valid)[0]
    if len(indices):
        inside[indices] = mask[y[indices], x[indices]] > 0
    return float(np.mean(inside))


def _crop_bounds(start, end, half_width, padding, width, height):
    margin = half_width + padding
    left = max(0, int(math.floor(min(start[0], end[0]) - margin)))
    right = min(width, int(math.ceil(max(start[0], end[0]) + margin)))
    top = max(0, int(math.floor(min(start[1], end[1]) - margin)))
    bottom = min(height, int(math.ceil(max(start[1], end[1]) + margin)))
    return (left, top, right, bottom) if right - left >= 4 and bottom - top >= 4 else None


def _cluster_by_offset(candidates, width):
    if not candidates:
        return []
    ordered = sorted(candidates, key=lambda item: item["offset_px"])
    clusters = [[ordered[0]]]
    for candidate in ordered[1:]:
        current = clusters[-1]
        centre = float(np.average(
            [item["offset_px"] for item in current],
            weights=[max(1.0, item["length_px"]) for item in current],
        ))
        if abs(candidate["offset_px"] - centre) <= width:
            current.append(candidate)
        else:
            clusters.append([candidate])
    return clusters


def _fit_cluster_line(cluster, expected):
    points, weights = [], []
    for item in cluster:
        x1, y1, x2, y2 = item["segment"]
        w = max(1.0, item["length_px"]) * (0.65 + 0.35 * max(0.0, item["paint_score"]))
        points.extend(([x1, y1], [x2, y2]))
        weights.extend((w, w))
    pts = np.asarray(points, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if len(pts) < 2:
        return None
    centroid = np.average(pts, axis=0, weights=w)
    centered = pts - centroid
    cov = (centered * w[:, None]).T @ centered / max(float(np.sum(w)), 1e-9)
    try:
        values, vectors = np.linalg.eigh(cov)
    except np.linalg.LinAlgError:
        return None
    direction = vectors[:, int(np.argmax(values))]
    p1, p2 = centroid - direction * 100.0, centroid + direction * 100.0
    line = _normalise_line(np.cross(
        np.asarray([p1[0], p1[1], 1.0]),
        np.asarray([p2[0], p2[1], 1.0]),
    ))
    if line is not None and float(np.dot(line[:2], expected[:2])) < 0.0:
        line *= -1.0
    return line


def _project_to_line(point, line):
    return point - _signed_distance(line, point) * line[:2]


def observe_line_search(frame_bgr, search, *, pitch_region=None, config=None):
    cfg = config or LocalLineObserverConfig()
    semantic = str(search.get("semantic", "unknown"))

    if search.get("type") != "line":
        return {"semantic": semantic, "type": search.get("type"), "status": "unsupported"}

    expected_native = _normalise_line(
        search.get("expected", {}).get("line")
        or search.get("predicted", {}).get("line")
    )
    finite = search.get("search", {}).get("finite_segment")
    if expected_native is None or not isinstance(finite, dict):
        return {
            "semantic": semantic,
            "type": "line",
            "status": "invalid_search",
            "detail": "Falta finite_segment; aplica field_spatial_gating.",
        }

    physical_start_native = _point(finite.get("physical_start", {}))
    physical_end_native = _point(finite.get("physical_end", {}))
    search_start_native = _point(finite.get("search_start", {}))
    search_end_native = _point(finite.get("search_end", {}))
    try:
        half_width_native = float(search["search"]["half_width_px"])
    except (KeyError, TypeError, ValueError):
        half_width_native = 0.0

    if any(p is None for p in (
        physical_start_native, physical_end_native,
        search_start_native, search_end_native,
    )) or half_width_native <= 1.0:
        return {"semantic": semantic, "type": "line", "status": "invalid_search"}

    native_h, native_w = frame_bgr.shape[:2]
    scale = float(np.clip(cfg.working_scale, 0.25, 1.0))
    work = cv2.resize(
        frame_bgr,
        (max(2, int(round(native_w * scale))), max(2, int(round(native_h * scale)))),
        interpolation=cv2.INTER_AREA,
    ) if scale < 0.999 else frame_bgr

    expected = _line_native_to_work(expected_native, scale)
    physical_start = physical_start_native * scale
    physical_end = physical_end_native * scale
    search_start = search_start_native * scale
    search_end = search_end_native * scale
    half_width = max(2.0, half_width_native * scale)

    bounds = _crop_bounds(
        search_start, search_end, half_width, cfg.crop_padding_px,
        work.shape[1], work.shape[0]
    )
    if bounds is None:
        return {
            "semantic": semantic, "type": "line", "status": "not_found",
            "confidence": 0.0, "detail": "Segmento semántico fuera de imagen."
        }

    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    saturation = cv2.cvtColor(work, cv2.COLOR_BGR2HSV)[:, :, 1]
    enhanced = cv2.createCLAHE(clipLimit=1.6, tileGridSize=(8, 8)).apply(gray)
    pitch_mask = _pitch_mask(
        enhanced.shape, pitch_region or search.get("pitch_region"), scale
    )

    left, top, right, bottom = bounds
    crop = enhanced[top:bottom, left:right]

    # Detect on the untouched crop. Hard masking would create artificial edges.
    lsd = cv2.createLineSegmentDetector(
        cv2.LSD_REFINE_STD, scale=0.8, sigma_scale=0.6, quant=2.0,
        ang_th=22.5, log_eps=0.0, density_th=0.65, n_bins=1024
    )
    detected = lsd.detect(crop)[0]
    if detected is None:
        return {
            "semantic": semantic, "type": "line", "status": "not_found",
            "confidence": 0.0, "detail": "Sin segmentos visuales locales."
        }

    candidates = []
    for raw in detected.reshape(-1, 4):
        x1, y1, x2, y2 = map(float, raw)
        segment = (x1 + left, y1 + top, x2 + left, y2 + top)
        length = math.hypot(segment[2] - segment[0], segment[3] - segment[1])
        if length < cfg.min_segment_length_px:
            continue

        line = _line_from_segment(segment)
        if line is None:
            continue
        if float(np.dot(line[:2], expected[:2])) < 0.0:
            line *= -1.0

        angle_delta = _angle_difference_deg(line, expected)
        if angle_delta > cfg.max_angle_delta_deg:
            continue

        midpoint = np.asarray([
            0.5 * (segment[0] + segment[2]),
            0.5 * (segment[1] + segment[3]),
        ])
        offset = _signed_distance(expected, midpoint)
        if abs(offset) > half_width * cfg.max_offset_fraction_of_band:
            continue

        capsule = _capsule_fraction(segment, search_start, search_end, half_width)
        if capsule < cfg.min_segment_capsule_fraction:
            continue

        pitch = _pitch_fraction(segment, pitch_mask)
        if pitch < cfg.min_segment_pitch_fraction:
            continue

        paint, appearance = _paint_score(gray, saturation, segment, line, cfg)
        angle_score = max(0.0, 1.0 - angle_delta / cfg.max_angle_delta_deg)
        offset_score = max(0.0, 1.0 - abs(offset) / half_width)
        geometry = 0.32 * angle_score + 0.28 * offset_score + 0.22 * capsule + 0.18 * pitch

        candidates.append({
            "segment": segment,
            "line": line,
            "length_px": float(length),
            "offset_px": float(offset),
            "paint_score": float(paint),
            "geometry_score": float(geometry),
            "capsule_fraction": float(capsule),
            "pitch_fraction": float(pitch),
            "appearance": appearance,
        })

    if not candidates:
        return {
            "semantic": semantic, "type": "line", "status": "not_found",
            "confidence": 0.0,
            "detail": "Ningún borde respetó segmento físico + campo + orientación."
        }

    scored = []
    for cluster in _cluster_by_offset(candidates, cfg.offset_cluster_width_px):
        support = float(sum(item["length_px"] for item in cluster))
        if support < cfg.min_total_support_px:
            continue
        fitted = _fit_cluster_line(cluster, expected)
        if fitted is None:
            continue
        weights = [max(1.0, item["length_px"]) for item in cluster]
        mean_paint = float(np.average([i["paint_score"] for i in cluster], weights=weights))
        mean_geom = float(np.average([i["geometry_score"] for i in cluster], weights=weights))
        mean_capsule = float(np.average([i["capsule_fraction"] for i in cluster], weights=weights))
        mean_pitch = float(np.average([i["pitch_fraction"] for i in cluster], weights=weights))
        support_score = float(np.clip(support / 120.0, 0.0, 1.0))
        paint_gate = 0.45 + 0.55 * mean_paint
        confidence = (
            0.34 * mean_geom + 0.20 * support_score + 0.16 * mean_capsule
            + 0.16 * mean_pitch + 0.14 * mean_paint
        ) * paint_gate
        scored.append({
            "line": fitted,
            "cluster": cluster,
            "support_px": support,
            "mean_paint": mean_paint,
            "mean_geometry": mean_geom,
            "mean_capsule": mean_capsule,
            "mean_pitch": mean_pitch,
            "confidence": float(np.clip(confidence, 0.0, 1.0)),
        })

    if not scored:
        return {
            "semantic": semantic, "type": "line", "status": "not_found",
            "confidence": 0.0, "detail": "Soporte visual insuficiente."
        }

    scored.sort(key=lambda item: item["confidence"], reverse=True)
    best = scored[0]
    observed_native = _line_work_to_native(best["line"], scale)
    if observed_native is None:
        return {"semantic": semantic, "type": "line", "status": "not_found", "confidence": 0.0}

    observed_start = _project_to_line(physical_start, best["line"]) / scale
    observed_end = _project_to_line(physical_end, best["line"]) / scale
    physical_mid_native = 0.5 * (physical_start_native + physical_end_native)
    normal_offset = _signed_distance(
        expected_native, _project_to_line(physical_mid_native, observed_native)
    )
    angle_offset = _angle_difference_deg(observed_native, expected_native)
    confidence = float(best["confidence"])

    if confidence < cfg.min_confidence or (
        best["mean_paint"] < cfg.min_paint_score
        and best["mean_geometry"] < 0.72
    ):
        return {
            "semantic": semantic, "type": "line", "status": "not_found",
            "confidence": confidence,
            "best_candidate": {
                "normal_offset_px": float(normal_offset),
                "angle_offset_deg": float(angle_offset),
                "paint_score": float(best["mean_paint"]),
                "pitch_fraction": float(best["mean_pitch"]),
                "capsule_fraction": float(best["mean_capsule"]),
            },
        }

    debug_segments = []
    for item in sorted(
        best["cluster"],
        key=lambda i: i["geometry_score"] + i["paint_score"],
        reverse=True,
    )[:cfg.max_debug_segments]:
        x1, y1, x2, y2 = item["segment"]
        debug_segments.append({
            "start": {"x": float(x1 / scale), "y": float(y1 / scale)},
            "end": {"x": float(x2 / scale), "y": float(y2 / scale)},
            "paint_score": item["paint_score"],
            "pitch_fraction": item["pitch_fraction"],
            "capsule_fraction": item["capsule_fraction"],
        })

    return {
        "semantic": semantic,
        "type": "line",
        "status": "found",
        "confidence": confidence,
        "expected_line": expected_native.tolist(),
        "observed_line": observed_native.tolist(),
        "observed_segment_start": {"x": float(observed_start[0]), "y": float(observed_start[1])},
        "observed_segment_end": {"x": float(observed_end[0]), "y": float(observed_end[1])},
        "normal_offset_px": float(normal_offset),
        "angle_offset_deg": float(angle_offset),
        "support_length_px": float(best["support_px"] / scale),
        "paint_score": float(best["mean_paint"]),
        "pitch_fraction": float(best["mean_pitch"]),
        "capsule_fraction": float(best["mean_capsule"]),
        "search_half_width_px": float(half_width_native),
        "debug": {
            "mode": "finite_segment_pitch_gated_v2",
            "candidate_segment_count": len(candidates),
            "cluster_count": len(scored),
            "support_segments": debug_segments,
        },
    }


def observe_field_lines(
    frame_bgr: np.ndarray,
    guidance: dict[str, Any] | Iterable[dict[str, Any]],
    *,
    config: LocalLineObserverConfig | None = None,
):
    cfg = config or LocalLineObserverConfig()
    if isinstance(guidance, dict):
        searches = guidance.get("searches", [])
        pitch_region = guidance.get("pitch_region")
    else:
        searches = list(guidance)
        pitch_region = None

    line_searches = [
        item for item in searches
        if isinstance(item, dict) and item.get("type") == "line"
    ]
    observations = [
        observe_line_search(
            frame_bgr, item, pitch_region=pitch_region, config=cfg
        )
        for item in line_searches
    ]
    found = [item for item in observations if item.get("status") == "found"]
    not_found = [item for item in observations if item.get("status") == "not_found"]

    return {
        "status": "ok",
        "mode": "field_local_line_observer_v2",
        "observations": observations,
        "summary": {
            "searched_lines": len(line_searches),
            "found_lines": len(found),
            "not_found_lines": len(not_found),
            "mean_found_confidence": (
                float(np.mean([item["confidence"] for item in found])) if found else 0.0
            ),
        },
        "config": asdict(cfg),
    }