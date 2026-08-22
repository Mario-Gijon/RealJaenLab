"""Local visual observer for semantic football-pitch lines.

This module is intentionally *not* a full-frame field detector.

It consumes the small, semantically labelled search regions produced by
``field_online_adaptation.suggest_local_field_searches`` and inspects only those
regions.  The current FIELD homography therefore decides *where* to look; this
observer only tries to answer:

    "Is the expected pitch marking really near here, and where exactly?"

That design is deliberate:
- it avoids scanning stands / advertising / railings;
- it keeps compute bounded;
- it can safely return ``not_found`` instead of inventing a distant line;
- later learned models (PnLCalib/TVCalib/custom) can replace or complement the
  local pixel evidence without changing the surrounding architecture.

V1 handles straight semantic lines only. Points and arcs remain for a later
observer.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Iterable

import cv2
import numpy as np


@dataclass(frozen=True)
class LocalLineObserverConfig:
    # Work at a smaller resolution for speed, then convert the result back to
    # native video coordinates.
    working_scale: float = 0.65

    # Hard semantic gating around the line predicted by FIELD adaptation.
    # The observer NEVER searches outside the supplied band.
    max_angle_delta_deg: float = 8.0
    extra_angle_search_deg: float = 1.5

    # LSD / support filtering in WORKING pixels.
    min_segment_length_px: float = 10.0
    min_total_support_px: float = 34.0
    min_segment_count: int = 1

    # Candidate line grouping by signed normal offset from the expected line.
    offset_cluster_width_px: float = 3.5

    # Appearance checks. These are intentionally soft because Veo footage can
    # contain shadows, saturation changes and compression.
    side_sample_distance_px: float = 4.0
    min_local_contrast: float = 1.5
    min_paint_score: float = 0.18

    # Final confidence threshold. Returning "not_found" is preferred over a
    # confident-looking wrong line.
    min_confidence: float = 0.48

    # A candidate may not move farther than this fraction of the search band
    # from the expected line.
    max_offset_fraction_of_band: float = 0.95

    # Debug payload size control.
    max_debug_segments: int = 24


def _normalise_line(line: Iterable[float]) -> np.ndarray | None:
    try:
        arr = np.asarray(list(line), dtype=np.float64).reshape(3)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(arr).all():
        return None
    norm = math.hypot(float(arr[0]), float(arr[1]))
    if norm < 1e-9:
        return None
    arr = arr / norm
    # Stable sign makes offsets easier to compare/read.
    if arr[2] < 0.0:
        arr *= -1.0
    return arr


def _line_direction(line: np.ndarray) -> np.ndarray:
    # For ax + by + c = 0, a unit tangent is (-b, a).
    direction = np.asarray([-line[1], line[0]], dtype=np.float64)
    norm = float(np.linalg.norm(direction))
    if norm < 1e-9:
        return np.asarray([1.0, 0.0], dtype=np.float64)
    return direction / norm


def _line_angle_rad(line: np.ndarray) -> float:
    d = _line_direction(line)
    return math.atan2(float(d[1]), float(d[0]))


def _angle_difference_deg(a: np.ndarray, b: np.ndarray) -> float:
    delta = _line_angle_rad(a) - _line_angle_rad(b)
    # Lines are unoriented, so angle is modulo pi.
    while delta > math.pi / 2.0:
        delta -= math.pi
    while delta < -math.pi / 2.0:
        delta += math.pi
    return abs(math.degrees(delta))


def _signed_distance(line: np.ndarray, point_xy: np.ndarray) -> float:
    return float(line[0] * point_xy[0] + line[1] * point_xy[1] + line[2])


def _transform_line_to_working(line_native: np.ndarray, scale: float) -> np.ndarray:
    # x_native = x_work / scale. Substitute into ax+by+c=0:
    # a*xw/scale + b*yw/scale + c = 0 -> a*xw+b*yw+c*scale=0.
    return _normalise_line(
        [line_native[0], line_native[1], line_native[2] * scale]
    )


def _transform_line_to_native(line_work: np.ndarray, scale: float) -> np.ndarray:
    if scale <= 0.0:
        return line_work.copy()
    return _normalise_line(
        [line_work[0], line_work[1], line_work[2] / scale]
    )


def _visible_line_segment(
    line: np.ndarray,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    a, b, c = map(float, line)
    candidates: list[np.ndarray] = []

    if abs(b) > 1e-9:
        y = -(c) / b
        if 0.0 <= y <= height - 1:
            candidates.append(np.asarray([0.0, y], dtype=np.float64))
        y = -(a * (width - 1) + c) / b
        if 0.0 <= y <= height - 1:
            candidates.append(
                np.asarray([float(width - 1), y], dtype=np.float64)
            )

    if abs(a) > 1e-9:
        x = -(c) / a
        if 0.0 <= x <= width - 1:
            candidates.append(np.asarray([x, 0.0], dtype=np.float64))
        x = -(b * (height - 1) + c) / a
        if 0.0 <= x <= width - 1:
            candidates.append(
                np.asarray([x, float(height - 1)], dtype=np.float64)
            )

    unique: list[np.ndarray] = []
    for point in candidates:
        if not any(float(np.linalg.norm(point - q)) < 1.0 for q in unique):
            unique.append(point)

    if len(unique) < 2:
        return None

    best: tuple[np.ndarray, np.ndarray] | None = None
    best_distance = -1.0
    for i in range(len(unique)):
        for j in range(i + 1, len(unique)):
            distance = float(np.linalg.norm(unique[i] - unique[j]))
            if distance > best_distance:
                best_distance = distance
                best = (unique[i], unique[j])
    return best


def _band_mask(
    shape: tuple[int, int],
    expected_line: np.ndarray,
    half_width_px: float,
) -> np.ndarray:
    height, width = shape
    yy, xx = np.mgrid[0:height, 0:width]
    distance = np.abs(
        expected_line[0] * xx
        + expected_line[1] * yy
        + expected_line[2]
    )
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[distance <= half_width_px] = 255
    return mask


def _segment_line(
    segment: tuple[float, float, float, float]
) -> np.ndarray | None:
    x1, y1, x2, y2 = segment
    p1 = np.asarray([x1, y1, 1.0], dtype=np.float64)
    p2 = np.asarray([x2, y2, 1.0], dtype=np.float64)
    line = np.cross(p1, p2)
    return _normalise_line(line)


def _sample_bilinear(
    image: np.ndarray,
    points_xy: np.ndarray,
) -> np.ndarray:
    if len(points_xy) == 0:
        return np.asarray([], dtype=np.float32)
    map_x = points_xy[:, 0].astype(np.float32).reshape(-1, 1)
    map_y = points_xy[:, 1].astype(np.float32).reshape(-1, 1)
    sampled = cv2.remap(
        image,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return sampled.reshape(-1)


def _segment_paint_score(
    gray: np.ndarray,
    saturation: np.ndarray,
    segment: tuple[float, float, float, float],
    line: np.ndarray,
    config: LocalLineObserverConfig,
) -> tuple[float, dict[str, float]]:
    x1, y1, x2, y2 = segment
    p1 = np.asarray([x1, y1], dtype=np.float64)
    p2 = np.asarray([x2, y2], dtype=np.float64)
    length = float(np.linalg.norm(p2 - p1))
    if length < 2.0:
        return 0.0, {
            "contrast": 0.0,
            "desaturation": 0.0,
            "centre_gray": 0.0,
        }

    sample_count = max(12, min(80, int(length / 3.0)))
    t = np.linspace(0.08, 0.92, sample_count)[:, None]
    centre = p1[None, :] + (p2 - p1)[None, :] * t

    normal = np.asarray([line[0], line[1]], dtype=np.float64)
    side_distance = float(config.side_sample_distance_px)

    side_a = centre + normal[None, :] * side_distance
    side_b = centre - normal[None, :] * side_distance

    centre_gray = _sample_bilinear(gray, centre)
    side_a_gray = _sample_bilinear(gray, side_a)
    side_b_gray = _sample_bilinear(gray, side_b)

    centre_sat = _sample_bilinear(saturation, centre)
    side_a_sat = _sample_bilinear(saturation, side_a)
    side_b_sat = _sample_bilinear(saturation, side_b)

    if len(centre_gray) == 0:
        return 0.0, {
            "contrast": 0.0,
            "desaturation": 0.0,
            "centre_gray": 0.0,
        }

    side_gray = 0.5 * (side_a_gray + side_b_gray)
    side_sat = 0.5 * (side_a_sat + side_b_sat)

    contrast = float(np.median(centre_gray - side_gray))
    desaturation = float(np.median(side_sat - centre_sat))
    brightness = float(np.median(centre_gray))

    # Paint tends to be locally brighter and/or less saturated than the turf.
    contrast_score = float(
        np.clip((contrast - config.min_local_contrast) / 24.0, 0.0, 1.0)
    )
    desat_score = float(np.clip(desaturation / 55.0, 0.0, 1.0))
    brightness_score = float(np.clip((brightness - 75.0) / 120.0, 0.0, 1.0))

    score = (
        0.48 * contrast_score
        + 0.32 * desat_score
        + 0.20 * brightness_score
    )

    return score, {
        "contrast": contrast,
        "desaturation": desaturation,
        "centre_gray": brightness,
    }


def _cluster_candidates(
    candidates: list[dict[str, Any]],
    cluster_width_px: float,
) -> list[list[dict[str, Any]]]:
    if not candidates:
        return []

    ordered = sorted(candidates, key=lambda item: item["offset_px"])
    clusters: list[list[dict[str, Any]]] = [[ordered[0]]]

    for candidate in ordered[1:]:
        current = clusters[-1]
        weighted_offset = np.average(
            [item["offset_px"] for item in current],
            weights=[max(1.0, item["length_px"]) for item in current],
        )
        if abs(candidate["offset_px"] - weighted_offset) <= cluster_width_px:
            current.append(candidate)
        else:
            clusters.append([candidate])

    return clusters


def _fit_cluster_line(
    cluster: list[dict[str, Any]],
    expected_line: np.ndarray,
) -> np.ndarray | None:
    points: list[np.ndarray] = []
    weights: list[float] = []

    for candidate in cluster:
        x1, y1, x2, y2 = candidate["segment"]
        length = max(1.0, float(candidate["length_px"]))
        paint = max(0.05, float(candidate["paint_score"]))
        weight = length * (0.55 + 0.45 * paint)
        points.extend(
            [
                np.asarray([x1, y1], dtype=np.float64),
                np.asarray([x2, y2], dtype=np.float64),
            ]
        )
        weights.extend([weight, weight])

    if len(points) < 2:
        return None

    pts = np.asarray(points, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    centroid = np.average(pts, axis=0, weights=w)

    centred = pts - centroid
    covariance = (centred * w[:, None]).T @ centred / max(float(np.sum(w)), 1e-9)

    try:
        eigvals, eigvecs = np.linalg.eigh(covariance)
    except np.linalg.LinAlgError:
        return None

    direction = eigvecs[:, int(np.argmax(eigvals))]
    if float(np.linalg.norm(direction)) < 1e-9:
        return None
    direction = direction / np.linalg.norm(direction)

    p1 = centroid - direction * 100.0
    p2 = centroid + direction * 100.0
    fitted = _normalise_line(np.cross(
        np.asarray([p1[0], p1[1], 1.0]),
        np.asarray([p2[0], p2[1], 1.0]),
    ))
    if fitted is None:
        return None

    # Same normal orientation as expected -> meaningful signed offset.
    if float(np.dot(fitted[:2], expected_line[:2])) < 0.0:
        fitted *= -1.0
    return fitted


def _refine_line_offset_by_paint(
    gray: np.ndarray,
    saturation: np.ndarray,
    line: np.ndarray,
    expected_line: np.ndarray,
    search_half_width: float,
    config: LocalLineObserverConfig,
) -> tuple[np.ndarray, float]:
    """Small 1-D normal refinement around the fitted candidate.

    LSD often returns one edge of a painted stripe. This searches a few nearby
    normal offsets and prefers the centre with strongest paint-like response.
    """

    segment = _visible_line_segment(line, gray.shape[1], gray.shape[0])
    if segment is None:
        return line, 0.0

    p1, p2 = segment
    tangent = _line_direction(line)
    length = float(np.linalg.norm(p2 - p1))
    if length < 12.0:
        return line, 0.0

    # Keep only a central portion to avoid image-edge artefacts.
    centre = 0.5 * (p1 + p2)
    half = min(length * 0.38, 180.0)
    base_p1 = centre - tangent * half
    base_p2 = centre + tangent * half

    best_line = line.copy()
    best_score = -1.0
    max_local_shift = min(4.0, max(1.0, search_half_width * 0.18))

    for delta in np.linspace(-max_local_shift, max_local_shift, 9):
        shifted = line.copy()
        shifted[2] -= float(delta)

        # Reject if this local centre correction would leave the semantic band.
        centre_distance = abs(_signed_distance(expected_line, centre))
        if centre_distance > search_half_width:
            continue

        visible = _visible_line_segment(shifted, gray.shape[1], gray.shape[0])
        if visible is None:
            continue

        sp1 = 0.5 * (visible[0] + visible[1]) - tangent * half
        sp2 = 0.5 * (visible[0] + visible[1]) + tangent * half
        segment_tuple = (float(sp1[0]), float(sp1[1]), float(sp2[0]), float(sp2[1]))
        score, _ = _segment_paint_score(
            gray,
            saturation,
            segment_tuple,
            shifted,
            config,
        )
        if score > best_score:
            best_score = score
            best_line = shifted

    return best_line, max(0.0, best_score)


def observe_line_search(
    frame_bgr: np.ndarray,
    search: dict[str, Any],
    *,
    config: LocalLineObserverConfig | None = None,
) -> dict[str, Any]:
    """Inspect one semantic line band and return a local observation."""

    cfg = config or LocalLineObserverConfig()

    semantic = str(search.get("semantic", "unknown"))
    if search.get("type") != "line":
        return {
            "semantic": semantic,
            "type": search.get("type"),
            "status": "unsupported",
            "detail": "V1 solo observa líneas rectas.",
        }

    expected_raw = (
        search.get("expected", {}).get("line")
        or search.get("predicted", {}).get("line")
    )
    expected_native = _normalise_line(expected_raw or [])
    if expected_native is None:
        return {
            "semantic": semantic,
            "type": "line",
            "status": "invalid_search",
            "detail": "La guía no contiene una línea esperada válida.",
        }

    try:
        half_width_native = float(search["search"]["half_width_px"])
    except (KeyError, TypeError, ValueError):
        return {
            "semantic": semantic,
            "type": "line",
            "status": "invalid_search",
            "detail": "La guía no contiene el ancho de búsqueda.",
        }

    if half_width_native <= 1.0:
        return {
            "semantic": semantic,
            "type": "line",
            "status": "invalid_search",
            "detail": "La banda de búsqueda es demasiado estrecha.",
        }

    if frame_bgr is None or frame_bgr.ndim != 3 or frame_bgr.shape[2] < 3:
        return {
            "semantic": semantic,
            "type": "line",
            "status": "invalid_frame",
            "detail": "El fotograma BGR no es válido.",
        }

    scale = float(np.clip(cfg.working_scale, 0.25, 1.0))
    native_height, native_width = frame_bgr.shape[:2]

    if scale < 0.999:
        work = cv2.resize(
            frame_bgr,
            (
                max(2, int(round(native_width * scale))),
                max(2, int(round(native_height * scale))),
            ),
            interpolation=cv2.INTER_AREA,
        )
    else:
        work = frame_bgr

    expected_work = _transform_line_to_working(expected_native, scale)
    if expected_work is None:
        return {
            "semantic": semantic,
            "type": "line",
            "status": "invalid_search",
            "detail": "No se pudo transformar la línea esperada.",
        }

    half_width_work = max(2.0, half_width_native * scale)

    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(work, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]

    # Mild local contrast normalisation improves line extraction in shadow/high
    # saturation without turning this into a full-frame expensive detector.
    clahe = cv2.createCLAHE(clipLimit=1.6, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    band = _band_mask(
        enhanced.shape,
        expected_work,
        half_width_work,
    )

    # Suppress pixels outside the semantic band before LSD.
    masked = cv2.bitwise_and(enhanced, enhanced, mask=band)

    lsd = cv2.createLineSegmentDetector(
        cv2.LSD_REFINE_STD,
        scale=0.8,
        sigma_scale=0.6,
        quant=2.0,
        ang_th=22.5,
        log_eps=0.0,
        density_th=0.65,
        n_bins=1024,
    )
    detected = lsd.detect(masked)[0]

    if detected is None:
        return {
            "semantic": semantic,
            "type": "line",
            "status": "not_found",
            "confidence": 0.0,
            "detail": "No se detectaron segmentos compatibles en la banda local.",
            "expected_line": expected_native.tolist(),
            "search_half_width_px": half_width_native,
        }

    candidates: list[dict[str, Any]] = []

    for raw in detected.reshape(-1, 4):
        x1, y1, x2, y2 = map(float, raw)
        length = math.hypot(x2 - x1, y2 - y1)
        if length < cfg.min_segment_length_px:
            continue

        segment = (x1, y1, x2, y2)
        line = _segment_line(segment)
        if line is None:
            continue

        # Sign-align normals before computing offsets.
        if float(np.dot(line[:2], expected_work[:2])) < 0.0:
            line *= -1.0

        angle_delta = _angle_difference_deg(line, expected_work)
        if angle_delta > (
            cfg.max_angle_delta_deg + cfg.extra_angle_search_deg
        ):
            continue

        midpoint = np.asarray(
            [(x1 + x2) * 0.5, (y1 + y2) * 0.5],
            dtype=np.float64,
        )
        offset = _signed_distance(expected_work, midpoint)
        if abs(offset) > (
            half_width_work * cfg.max_offset_fraction_of_band
        ):
            continue

        paint_score, appearance = _segment_paint_score(
            gray,
            saturation,
            segment,
            line,
            cfg,
        )

        # Do not make paint appearance a hard white threshold: field lines in
        # Veo footage may be shaded/blueish/compressed. Geometry still matters.
        geometry_score = (
            0.58
            * max(
                0.0,
                1.0
                - angle_delta
                / max(cfg.max_angle_delta_deg, 1e-6),
            )
            + 0.42
            * max(
                0.0,
                1.0 - abs(offset) / max(half_width_work, 1e-6),
            )
        )

        combined = (
            0.64 * geometry_score
            + 0.36 * paint_score
        )

        candidates.append(
            {
                "segment": segment,
                "line": line,
                "length_px": length,
                "offset_px": float(offset),
                "angle_delta_deg": float(angle_delta),
                "paint_score": float(paint_score),
                "geometry_score": float(geometry_score),
                "combined_score": float(combined),
                "appearance": appearance,
            }
        )

    if not candidates:
        return {
            "semantic": semantic,
            "type": "line",
            "status": "not_found",
            "confidence": 0.0,
            "detail": "Había bordes locales, pero ninguno respetaba la geometría esperada.",
            "expected_line": expected_native.tolist(),
            "search_half_width_px": half_width_native,
        }

    clusters = _cluster_candidates(
        candidates,
        cfg.offset_cluster_width_px,
    )

    scored_clusters: list[dict[str, Any]] = []
    for cluster in clusters:
        support = float(sum(item["length_px"] for item in cluster))
        if (
            support < cfg.min_total_support_px
            or len(cluster) < cfg.min_segment_count
        ):
            continue

        fitted = _fit_cluster_line(cluster, expected_work)
        if fitted is None:
            continue

        angle_delta = _angle_difference_deg(fitted, expected_work)
        visible = _visible_line_segment(
            fitted,
            work.shape[1],
            work.shape[0],
        )
        if visible is None:
            continue

        centre = 0.5 * (visible[0] + visible[1])
        offset = _signed_distance(expected_work, centre)

        mean_paint = float(
            np.average(
                [item["paint_score"] for item in cluster],
                weights=[max(1.0, item["length_px"]) for item in cluster],
            )
        )
        mean_geometry = float(
            np.average(
                [item["geometry_score"] for item in cluster],
                weights=[max(1.0, item["length_px"]) for item in cluster],
            )
        )

        support_score = float(
            np.clip(
                support / max(110.0, cfg.min_total_support_px),
                0.0,
                1.0,
            )
        )
        count_score = float(
            np.clip(len(cluster) / 4.0, 0.25, 1.0)
        )
        offset_score = float(
            np.clip(
                1.0 - abs(offset) / max(half_width_work, 1e-6),
                0.0,
                1.0,
            )
        )
        angle_score = float(
            np.clip(
                1.0 - angle_delta / max(cfg.max_angle_delta_deg, 1e-6),
                0.0,
                1.0,
            )
        )

        confidence = (
            0.24 * mean_geometry
            + 0.20 * mean_paint
            + 0.21 * support_score
            + 0.15 * count_score
            + 0.12 * offset_score
            + 0.08 * angle_score
        )

        scored_clusters.append(
            {
                "cluster": cluster,
                "line": fitted,
                "support_px": support,
                "segment_count": len(cluster),
                "mean_paint_score": mean_paint,
                "mean_geometry_score": mean_geometry,
                "normal_offset_px": float(offset),
                "angle_offset_deg": float(angle_delta),
                "confidence": float(confidence),
            }
        )

    if not scored_clusters:
        return {
            "semantic": semantic,
            "type": "line",
            "status": "not_found",
            "confidence": 0.0,
            "detail": "Los segmentos compatibles no aportaron soporte suficiente.",
            "expected_line": expected_native.tolist(),
            "search_half_width_px": half_width_native,
        }

    scored_clusters.sort(
        key=lambda item: item["confidence"],
        reverse=True,
    )
    best = scored_clusters[0]

    refined_work, refined_paint_score = _refine_line_offset_by_paint(
        gray,
        saturation,
        best["line"],
        expected_work,
        half_width_work,
        cfg,
    )

    observed_native = _transform_line_to_native(refined_work, scale)
    if observed_native is None:
        return {
            "semantic": semantic,
            "type": "line",
            "status": "not_found",
            "confidence": 0.0,
            "detail": "La línea candidata no pudo convertirse a coordenadas nativas.",
            "expected_line": expected_native.tolist(),
            "search_half_width_px": half_width_native,
        }

    # Native offset is measured at the image centre, where it is easiest to
    # interpret for debugging / solver integration.
    centre_native = np.asarray(
        [native_width / 2.0, native_height / 2.0],
        dtype=np.float64,
    )
    expected_distance = _signed_distance(
        expected_native,
        centre_native,
    )
    observed_distance = _signed_distance(
        observed_native,
        centre_native,
    )
    normal_offset_native = observed_distance - expected_distance
    angle_offset_native = _angle_difference_deg(
        observed_native,
        expected_native,
    )

    confidence = float(best["confidence"])
    if refined_paint_score > 0.0:
        confidence = float(
            np.clip(
                0.86 * confidence
                + 0.14 * refined_paint_score,
                0.0,
                1.0,
            )
        )

    # Paint is useful, but V1 should still be allowed to find shaded lines if
    # geometric support is strong. Only reject very weak candidates.
    if (
        confidence < cfg.min_confidence
        or (
            best["mean_paint_score"] < cfg.min_paint_score
            and best["mean_geometry_score"] < 0.62
        )
    ):
        return {
            "semantic": semantic,
            "type": "line",
            "status": "not_found",
            "confidence": confidence,
            "detail": "La evidencia local no fue suficientemente fiable.",
            "expected_line": expected_native.tolist(),
            "search_half_width_px": half_width_native,
            "best_candidate": {
                "support_length_px": float(best["support_px"] / scale),
                "segment_count": int(best["segment_count"]),
                "normal_offset_px": float(normal_offset_native),
                "angle_offset_deg": float(angle_offset_native),
                "paint_score": float(best["mean_paint_score"]),
            },
        }

    debug_segments = []
    for item in sorted(
        best["cluster"],
        key=lambda candidate: candidate["combined_score"],
        reverse=True,
    )[: cfg.max_debug_segments]:
        x1, y1, x2, y2 = item["segment"]
        debug_segments.append(
            {
                "start": {
                    "x": float(x1 / scale),
                    "y": float(y1 / scale),
                },
                "end": {
                    "x": float(x2 / scale),
                    "y": float(y2 / scale),
                },
                "score": float(item["combined_score"]),
            }
        )

    return {
        "semantic": semantic,
        "type": "line",
        "status": "found",
        "confidence": confidence,
        "expected_line": expected_native.tolist(),
        "observed_line": observed_native.tolist(),
        "normal_offset_px": float(normal_offset_native),
        "angle_offset_deg": float(angle_offset_native),
        "support_length_px": float(best["support_px"] / scale),
        "segment_count": int(best["segment_count"]),
        "paint_score": float(best["mean_paint_score"]),
        "search_half_width_px": float(half_width_native),
        "debug": {
            "working_scale": scale,
            "candidate_segment_count": len(candidates),
            "cluster_count": len(scored_clusters),
            "support_segments": debug_segments,
        },
    }


def observe_field_lines(
    frame_bgr: np.ndarray,
    guidance: dict[str, Any] | Iterable[dict[str, Any]],
    *,
    config: LocalLineObserverConfig | None = None,
) -> dict[str, Any]:
    """Observe every straight-line search from FIELD adaptation guidance.

    ``guidance`` may be the complete result returned by
    ``suggest_local_field_searches`` or its ``searches`` list.
    """

    cfg = config or LocalLineObserverConfig()

    if isinstance(guidance, dict):
        searches = guidance.get("searches", [])
    else:
        searches = list(guidance)

    line_searches = [
        item
        for item in searches
        if isinstance(item, dict) and item.get("type") == "line"
    ]

    observations = [
        observe_line_search(
            frame_bgr,
            item,
            config=cfg,
        )
        for item in line_searches
    ]

    found = [
        item for item in observations if item.get("status") == "found"
    ]
    not_found = [
        item for item in observations if item.get("status") == "not_found"
    ]

    mean_confidence = (
        float(np.mean([item["confidence"] for item in found]))
        if found
        else 0.0
    )

    return {
        "status": "ok",
        "mode": "field_local_line_observer_v1",
        "observations": observations,
        "summary": {
            "searched_lines": len(line_searches),
            "found_lines": len(found),
            "not_found_lines": len(not_found),
            "mean_found_confidence": mean_confidence,
        },
        "config": asdict(cfg),
    }