#!/usr/bin/env python3
"""Exploratory Camera State V1 diagnostics for RealJaenLab.

This script is intentionally analysis-only.

It reads the saved review produced by PARTIDO V1, selects only completed HUMAN
anchors with:

    source == "manual"
    purpose == "match_setup"

and studies camera-state structure through the image projection of the currently
confirmed operational Pitch Model.

Important:
- It does NOT modify the review.
- It does NOT estimate per-frame homographies.
- It does NOT run Camera Atlas.
- It does NOT run Camera State.
- It does NOT use propagation/refinement.
- It never compares raw homography coefficients directly.

The camera representation is the projection of a fixed set of physical pitch
probe points. This makes distances invariant to the arbitrary scale of H and
keeps the diagnostic tied to what the homography actually does in the image.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


EPS = 1e-10


def _as_h(value: Any) -> np.ndarray | None:
    try:
        h = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return None

    if h.shape != (3, 3) or not np.isfinite(h).all():
        return None

    scale = float(h[2, 2])
    if abs(scale) < EPS:
        return None

    h = h / scale

    try:
        determinant = float(np.linalg.det(h))
        condition = float(np.linalg.cond(h))
    except np.linalg.LinAlgError:
        return None

    if (
        not math.isfinite(determinant)
        or abs(determinant) < 1e-12
        or not math.isfinite(condition)
        or condition > 1e13
    ):
        return None

    return h


def _project_points(points_xy: np.ndarray, h: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    homogeneous = np.column_stack((points, np.ones(len(points), dtype=np.float64)))
    q = homogeneous @ h.T
    denominator = q[:, 2]

    valid = (
        np.isfinite(q).all(axis=1)
        & np.isfinite(denominator)
        & (np.abs(denominator) > EPS)
    )

    projected = np.full((len(points), 2), np.nan, dtype=np.float64)
    projected[valid] = q[valid, :2] / denominator[valid, None]
    valid &= np.isfinite(projected).all(axis=1)

    return projected, valid


def _pitch_signature_world(length_m: float, width_m: float) -> np.ndarray:
    """Build a dynamic physical signature for the confirmed Pitch Model."""

    half_l = float(length_m) / 2.0
    half_w = float(width_m) / 2.0

    penalty_front = max(0.0, half_l - 16.5)
    goal_area_front = max(0.0, half_l - 5.5)

    penalty_y = min(20.16, half_w)
    goal_area_y = min(9.16, half_w)
    centre_circle_y = min(9.15, half_w)

    return np.asarray(
        [
            # Outer boundary + halfway support.
            (-half_l, -half_w),
            (0.0, -half_w),
            (half_l, -half_w),
            (-half_l, 0.0),
            (0.0, 0.0),
            (half_l, 0.0),
            (-half_l, half_w),
            (0.0, half_w),
            (half_l, half_w),

            # Penalty-area structure.
            (-half_l, -penalty_y),
            (-penalty_front, -penalty_y),
            (-penalty_front, 0.0),
            (-penalty_front, penalty_y),
            (-half_l, penalty_y),

            (half_l, -penalty_y),
            (penalty_front, -penalty_y),
            (penalty_front, 0.0),
            (penalty_front, penalty_y),
            (half_l, penalty_y),

            # Goal-area structure.
            (-half_l, -goal_area_y),
            (-goal_area_front, -goal_area_y),
            (-goal_area_front, goal_area_y),
            (-half_l, goal_area_y),

            (half_l, -goal_area_y),
            (goal_area_front, -goal_area_y),
            (goal_area_front, goal_area_y),
            (half_l, goal_area_y),

            # Centre circle support.
            (0.0, -centre_circle_y),
            (0.0, centre_circle_y),
            (-centre_circle_y, 0.0),
            (centre_circle_y, 0.0),
        ],
        dtype=np.float64,
    )


def _projection_signature(
    h: np.ndarray,
    world_signature: np.ndarray,
    image_width: int,
    image_height: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    projected, valid = _project_points(world_signature, h)

    if not np.all(valid):
        return projected, valid, None

    normalised = projected.copy()
    normalised[:, 0] /= max(float(image_width), 1.0)
    normalised[:, 1] /= max(float(image_height), 1.0)

    # Avoid one projective pole completely dominating PCA while still keeping
    # off-screen geometry informative.
    normalised = np.clip(normalised, -5.0, 5.0)

    return projected, valid, normalised.reshape(-1)


def _camera_distance_px(
    first_projected: np.ndarray,
    second_projected: np.ndarray,
) -> dict[str, float]:
    distances = np.linalg.norm(first_projected - second_projected, axis=1)
    return {
        "median_px": float(np.median(distances)),
        "p90_px": float(np.percentile(distances, 90)),
        "max_px": float(np.max(distances)),
    }


def _rankdata(values: np.ndarray) -> np.ndarray:
    """Average ranks, SciPy-free."""

    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)

    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1

        average_rank = (start + end - 1) / 2.0
        ranks[order[start:end]] = average_rank
        start = end

    return ranks


def _spearman(first: list[float], second: list[float]) -> float | None:
    if len(first) < 3 or len(first) != len(second):
        return None

    a = _rankdata(np.asarray(first, dtype=np.float64))
    b = _rankdata(np.asarray(second, dtype=np.float64))

    a -= np.mean(a)
    b -= np.mean(b)

    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator < EPS:
        return None

    return float(np.dot(a, b) / denominator)


def _camera_diagnostics(
    h: np.ndarray,
    image_width: int,
    image_height: int,
) -> dict[str, float | list[float] | None]:
    """Simple interpretable proxies: centre projection and local image scale."""

    support = np.asarray(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float64,
    )

    projected, valid = _project_points(support, h)

    centre_norm = None
    zoom_proxy = None
    pitch_x_angle_deg = None

    if valid[0]:
        centre_norm = [
            float(projected[0, 0] / max(float(image_width), 1.0)),
            float(projected[0, 1] / max(float(image_height), 1.0)),
        ]

    if np.all(valid):
        dx = projected[1] - projected[0]
        dy = projected[2] - projected[0]
        sx = float(np.linalg.norm(dx))
        sy = float(np.linalg.norm(dy))

        if sx > 0.0 and sy > 0.0:
            zoom_proxy = float(math.sqrt(sx * sy))

        if float(np.linalg.norm(dx)) > EPS:
            pitch_x_angle_deg = float(math.degrees(math.atan2(dx[1], dx[0])))

    return {
        "centre_projection_norm": centre_norm,
        "zoom_proxy_px_per_m": zoom_proxy,
        "pitch_x_angle_deg": pitch_x_angle_deg,
    }


def _scanner_recommendation(match_setup: dict[str, Any], frame: int) -> dict[str, Any] | None:
    scan = match_setup.get("scan")
    if not isinstance(scan, dict):
        return None

    recommendations = scan.get("recommendations")
    if not isinstance(recommendations, list):
        return None

    for recommendation in recommendations:
        if not isinstance(recommendation, dict):
            continue
        try:
            candidate_frame = int(recommendation.get("frame"))
        except (TypeError, ValueError):
            continue
        if candidate_frame == frame:
            return recommendation

    return None


def _resolve_image_size(
    review: dict[str, Any],
    match_setup: dict[str, Any],
) -> tuple[int, int]:
    scan = match_setup.get("scan")
    if isinstance(scan, dict):
        metadata = scan.get("metadata")
        if isinstance(metadata, dict):
            width = metadata.get("width")
            height = metadata.get("height")
            if (
                isinstance(width, (int, float))
                and isinstance(height, (int, float))
                and int(width) > 0
                and int(height) > 0
            ):
                return int(width), int(height)

    video = review.get("video")
    if isinstance(video, dict):
        width = video.get("width")
        height = video.get("height")
        if (
            isinstance(width, (int, float))
            and isinstance(height, (int, float))
            and int(width) > 0
            and int(height) > 0
        ):
            return int(width), int(height)

    raise RuntimeError(
        "No se pudo determinar image_width/image_height desde match_setup.scan.metadata "
        "ni desde review.video."
    )


def _collect_anchors(
    review: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    field = review.get("field")
    if not isinstance(field, dict):
        raise RuntimeError("La review no contiene field.")

    pitch_model = field.get("pitch_model")
    if not isinstance(pitch_model, dict) or pitch_model.get("status") != "confirmed":
        raise RuntimeError(
            "El Pitch Model no está confirmado. Camera State V1 debe analizar "
            "el modelo operativo confirmado."
        )

    match_setup = field.get("match_setup")
    if not isinstance(match_setup, dict):
        raise RuntimeError("La review no contiene field.match_setup.")

    completed_raw = match_setup.get("completed_frames")
    if not isinstance(completed_raw, list):
        raise RuntimeError("field.match_setup.completed_frames no es válido.")

    completed_frames: set[int] = set()
    for value in completed_raw:
        try:
            completed_frames.add(int(value))
        except (TypeError, ValueError):
            continue

    calibrations = field.get("calibrations")
    if not isinstance(calibrations, list):
        calibrations = []

    anchors: list[dict[str, Any]] = []

    for calibration in calibrations:
        if not isinstance(calibration, dict):
            continue

        try:
            frame = int(calibration.get("frame"))
        except (TypeError, ValueError):
            continue

        if frame not in completed_frames:
            continue
        if calibration.get("source") != "manual":
            continue
        if calibration.get("purpose") != "match_setup":
            continue

        h = _as_h(calibration.get("homography"))
        if h is None:
            continue

        anchors.append(
            {
                "frame": frame,
                "homography": h,
                "calibration": calibration,
                "scanner": _scanner_recommendation(match_setup, frame),
            }
        )

    anchors.sort(key=lambda item: item["frame"])

    missing = sorted(completed_frames - {item["frame"] for item in anchors})
    if missing:
        raise RuntimeError(
            "Hay frames marcados como completed_frames sin una H HUMAN match_setup válida: "
            + ", ".join(map(str, missing))
        )

    if len(anchors) < 3:
        raise RuntimeError(
            f"Solo hay {len(anchors)} anchors HUMAN válidos. "
            "Para este diagnóstico exploratorio necesitamos al menos 3."
        )

    return match_setup, anchors


def _pca_analysis(
    features: np.ndarray,
    projected: np.ndarray,
    image_width: int,
    image_height: int,
) -> dict[str, Any]:
    del image_width, image_height  # kept in signature to make intent explicit

    x = np.asarray(features, dtype=np.float64)
    mean = np.mean(x, axis=0)
    centered = x - mean

    u, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    del u

    variance = singular_values ** 2
    total = float(np.sum(variance))

    if total <= EPS:
        ratios = np.zeros_like(variance)
    else:
        ratios = variance / total

    cumulative = np.cumsum(ratios)

    max_rank = min(len(singular_values), len(features) - 1)

    ranks: dict[str, Any] = {}

    for rank in range(1, min(3, max_rank) + 1):
        basis = vt[:rank]
        coordinates = centered @ basis.T
        reconstructed = mean + coordinates @ basis

        anchor_errors = []
        for index in range(len(features)):
            reconstructed_points = reconstructed[index].reshape(-1, 2).copy()
            # Convert normalized signature back to actual image coordinates.
            # This mirrors how features were built.
            # NOTE: clipping only matters for extreme off-screen projections.
            original_points = projected[index]
            reconstruction_px = reconstructed_points.copy()

            # These scale factors are supplied outside below by replacing the
            # normalized coordinates after this function. We therefore report
            # normalized Euclidean error here and convert later.
            anchor_errors.append(
                float(np.median(np.linalg.norm(
                    features[index].reshape(-1, 2)
                    - reconstructed_points,
                    axis=1,
                )))
            )

        ranks[str(rank)] = {
            "variance_explained": float(cumulative[rank - 1]),
            "median_normalised_probe_error": float(np.median(anchor_errors)),
            "max_normalised_probe_error": float(np.max(anchor_errors)),
        }

    coordinates_2d = None
    if max_rank >= 2:
        coordinates_2d = (centered @ vt[:2].T).tolist()
    elif max_rank >= 1:
        one = centered @ vt[:1].T
        coordinates_2d = np.column_stack((one[:, 0], np.zeros(len(one)))).tolist()

    return {
        "singular_values": singular_values[:max_rank].tolist(),
        "variance_ratio": ratios[:max_rank].tolist(),
        "cumulative_variance": cumulative[:max_rank].tolist(),
        "rank_diagnostics": ranks,
        "coordinates_2d": coordinates_2d,
        "mean": mean,
        "components": vt,
    }


def _normalised_probe_error_to_pixels(
    first: np.ndarray,
    second: np.ndarray,
    image_width: int,
    image_height: int,
) -> float:
    a = np.asarray(first, dtype=np.float64).reshape(-1, 2)
    b = np.asarray(second, dtype=np.float64).reshape(-1, 2)

    delta = a - b
    delta[:, 0] *= float(image_width)
    delta[:, 1] *= float(image_height)

    return float(np.median(np.linalg.norm(delta, axis=1)))


def _pca_pixel_reconstruction(
    features: np.ndarray,
    pca: dict[str, Any],
    image_width: int,
    image_height: int,
) -> dict[str, Any]:
    x = np.asarray(features, dtype=np.float64)
    mean = np.asarray(pca["mean"], dtype=np.float64)
    vt = np.asarray(pca["components"], dtype=np.float64)
    centered = x - mean

    max_rank = min(len(vt), len(features) - 1)
    out: dict[str, Any] = {}

    for rank in range(1, min(3, max_rank) + 1):
        basis = vt[:rank]
        coordinates = centered @ basis.T
        reconstructed = mean + coordinates @ basis

        errors = [
            _normalised_probe_error_to_pixels(
                x[index],
                reconstructed[index],
                image_width,
                image_height,
            )
            for index in range(len(x))
        ]

        out[str(rank)] = {
            "variance_explained": pca["rank_diagnostics"][str(rank)]["variance_explained"],
            "median_probe_error_px": float(np.median(errors)),
            "max_probe_error_px": float(np.max(errors)),
            "per_anchor_probe_error_px": errors,
        }

    return out


def _best_segment_reconstruction(
    target_index: int,
    features: np.ndarray,
    image_width: int,
    image_height: int,
) -> dict[str, Any] | None:
    """Best convex reconstruction of one anchor from a pair of other anchors."""

    x = np.asarray(features, dtype=np.float64)
    target = x[target_index]

    others = [index for index in range(len(x)) if index != target_index]
    best = None

    for a_pos, first_index in enumerate(others):
        for second_index in others[a_pos + 1:]:
            first = x[first_index]
            second = x[second_index]
            direction = second - first
            denominator = float(np.dot(direction, direction))

            if denominator < EPS:
                alpha = 0.0
            else:
                alpha = float(np.dot(target - first, direction) / denominator)
                alpha = float(np.clip(alpha, 0.0, 1.0))

            reconstruction = first + alpha * direction
            error_px = _normalised_probe_error_to_pixels(
                target,
                reconstruction,
                image_width,
                image_height,
            )

            candidate = {
                "first_index": first_index,
                "second_index": second_index,
                "alpha": alpha,
                "median_probe_error_px": error_px,
            }

            if best is None or candidate["median_probe_error_px"] < best["median_probe_error_px"]:
                best = candidate

    return best


def _interpret_low_dimension(cumulative: list[float]) -> str:
    one = cumulative[0] if len(cumulative) >= 1 else 0.0
    two = cumulative[1] if len(cumulative) >= 2 else one

    if one >= 0.90:
        return (
            "Señal fuerte de trayectoria casi 1D en estos anchors. "
            "Aun así, con tan pocas muestras debe considerarse exploratorio."
        )

    if two >= 0.95:
        return (
            "Señal fuerte de estructura de cámara de baja dimensión: "
            "2 componentes explican al menos el 95 % de la variación."
        )

    if two >= 0.85:
        return (
            "Señal sugerente de estructura 2D, pero todavía no concluyente. "
            "Conviene validar con más anchors/probes."
        )

    return (
        "Estos anchors no muestran todavía una estructura 1D/2D claramente dominante. "
        "No conviene imponer un Camera State de baja dimensión sin más evidencia."
    )


def analyze_review(review_path: Path) -> dict[str, Any]:
    review = json.loads(review_path.read_text(encoding="utf-8"))

    field = review["field"]
    pitch_model = field["pitch_model"]

    length_m = float(pitch_model["length_m"])
    width_m = float(pitch_model["width_m"])

    match_setup, anchors = _collect_anchors(review)
    image_width, image_height = _resolve_image_size(review, match_setup)

    world_signature = _pitch_signature_world(length_m, width_m)

    projected_signatures: list[np.ndarray] = []
    features: list[np.ndarray] = []

    anchor_reports: list[dict[str, Any]] = []

    for anchor in anchors:
        projected, valid, feature = _projection_signature(
            anchor["homography"],
            world_signature,
            image_width,
            image_height,
        )

        if feature is None or not np.all(valid):
            raise RuntimeError(
                f"El anchor {anchor['frame']} no proyecta de forma finita toda la firma física."
            )

        projected_signatures.append(projected)
        features.append(feature)

        scanner = anchor.get("scanner")
        scanner_geometry = scanner.get("geometry") if isinstance(scanner, dict) else None

        anchor_reports.append(
            {
                "frame": anchor["frame"],
                "camera": _camera_diagnostics(
                    anchor["homography"],
                    image_width,
                    image_height,
                ),
                "scanner": {
                    "quality_score": scanner.get("quality_score"),
                    "state_distance_from_seed": scanner.get("state_distance_from_seed"),
                    "line_count": scanner_geometry.get("line_count") if isinstance(scanner_geometry, dict) else None,
                    "line_family_count": scanner_geometry.get("line_family_count") if isinstance(scanner_geometry, dict) else None,
                    "intersection_count": scanner_geometry.get("intersection_count") if isinstance(scanner_geometry, dict) else None,
                    "sharpness": scanner_geometry.get("sharpness") if isinstance(scanner_geometry, dict) else None,
                } if isinstance(scanner, dict) else None,
            }
        )

    projected_array = np.asarray(projected_signatures, dtype=np.float64)
    feature_array = np.asarray(features, dtype=np.float64)

    count = len(anchors)
    pairwise_median = np.zeros((count, count), dtype=np.float64)
    pairwise_p90 = np.zeros((count, count), dtype=np.float64)

    pair_records: list[dict[str, Any]] = []
    time_gaps: list[float] = []
    camera_distances: list[float] = []

    for first_index in range(count):
        for second_index in range(first_index + 1, count):
            distance = _camera_distance_px(
                projected_array[first_index],
                projected_array[second_index],
            )

            pairwise_median[first_index, second_index] = distance["median_px"]
            pairwise_median[second_index, first_index] = distance["median_px"]
            pairwise_p90[first_index, second_index] = distance["p90_px"]
            pairwise_p90[second_index, first_index] = distance["p90_px"]

            frame_gap = abs(
                anchors[first_index]["frame"]
                - anchors[second_index]["frame"]
            )

            time_gaps.append(float(frame_gap))
            camera_distances.append(distance["median_px"])

            pair_records.append(
                {
                    "first_frame": anchors[first_index]["frame"],
                    "second_frame": anchors[second_index]["frame"],
                    "frame_gap": frame_gap,
                    **distance,
                }
            )

    nearest_neighbors: list[dict[str, Any]] = []

    for index, anchor in enumerate(anchors):
        candidates = [
            (pairwise_median[index, other], other)
            for other in range(count)
            if other != index
        ]
        candidates.sort(key=lambda item: item[0])
        distance_px, nearest_index = candidates[0]

        nearest_neighbors.append(
            {
                "frame": anchor["frame"],
                "nearest_frame": anchors[nearest_index]["frame"],
                "camera_distance_median_px": float(distance_px),
                "frame_gap": abs(anchor["frame"] - anchors[nearest_index]["frame"]),
            }
        )

    temporal_steps = []
    for index in range(count - 1):
        temporal_steps.append(
            {
                "first_frame": anchors[index]["frame"],
                "second_frame": anchors[index + 1]["frame"],
                "frame_gap": anchors[index + 1]["frame"] - anchors[index]["frame"],
                "camera_distance_median_px": float(pairwise_median[index, index + 1]),
                "camera_distance_p90_px": float(pairwise_p90[index, index + 1]),
            }
        )

    pca_raw = _pca_analysis(
        feature_array,
        projected_array,
        image_width,
        image_height,
    )

    pca_reconstruction = _pca_pixel_reconstruction(
        feature_array,
        pca_raw,
        image_width,
        image_height,
    )

    coordinates_2d = pca_raw["coordinates_2d"] or []
    for report, coordinates in zip(anchor_reports, coordinates_2d):
        report["pca_2d"] = {
            "x": float(coordinates[0]),
            "y": float(coordinates[1]),
        }

    local_reconstruction = []

    for index, anchor in enumerate(anchors):
        best = _best_segment_reconstruction(
            index,
            feature_array,
            image_width,
            image_height,
        )

        if best is None:
            continue

        local_reconstruction.append(
            {
                "frame": anchor["frame"],
                "between_frames": [
                    anchors[best["first_index"]]["frame"],
                    anchors[best["second_index"]]["frame"],
                ],
                "alpha": float(best["alpha"]),
                "median_probe_error_px": float(best["median_probe_error_px"]),
            }
        )

    temporal_camera_spearman = _spearman(time_gaps, camera_distances)

    pca_public = {
        "variance_ratio": pca_raw["variance_ratio"],
        "cumulative_variance": pca_raw["cumulative_variance"],
        "reconstruction": pca_reconstruction,
        "interpretation": _interpret_low_dimension(
            pca_raw["cumulative_variance"]
        ),
    }

    rejected_frames = match_setup.get("rejected_frames")
    if not isinstance(rejected_frames, list):
        rejected_frames = []

    return {
        "status": "ok",
        "review": str(review_path),
        "pitch_model": {
            "length_m": length_m,
            "width_m": width_m,
            "status": pitch_model.get("status"),
            "dimension_status": pitch_model.get("dimension_status"),
        },
        "image": {
            "width": image_width,
            "height": image_height,
        },
        "match_setup": {
            "strong_anchor_count": len(anchors),
            "strong_anchor_frames": [item["frame"] for item in anchors],
            "rejected_frames": rejected_frames,
        },
        "signature_probe_count": int(len(world_signature)),
        "anchors": anchor_reports,
        "pairwise_camera_distance_median_px": pairwise_median.tolist(),
        "pairwise_camera_distance_p90_px": pairwise_p90.tolist(),
        "pairs": pair_records,
        "nearest_neighbors": nearest_neighbors,
        "temporal_steps": temporal_steps,
        "time_camera_spearman": temporal_camera_spearman,
        "pca": pca_public,
        "local_pair_reconstruction": local_reconstruction,
        "notes": [
            "PCA is descriptive, not a trained Camera State model.",
            "With only a handful of anchors, dimensionality estimates are exploratory.",
            "Distances compare projected physical pitch probes in image pixels, not raw H coefficients.",
            "Rejected PARTIDO views are never treated as strong anchors.",
        ],
    }


def _fmt(value: Any, digits: int = 1) -> str:
    if value is None:
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(number):
        return "—"
    return f"{number:.{digits}f}"


def print_report(result: dict[str, Any]) -> None:
    frames = result["match_setup"]["strong_anchor_frames"]
    rejected = result["match_setup"]["rejected_frames"]
    pca = result["pca"]
    cumulative = pca["cumulative_variance"]

    print()
    print("=" * 72)
    print("CAMERA STATE V1 · DIAGNÓSTICO EXPLORATORIO")
    print("=" * 72)

    pitch = result["pitch_model"]
    print(
        f"Pitch operativo: {pitch['length_m']:.1f} × {pitch['width_m']:.1f} m "
        f"({pitch.get('dimension_status')})"
    )
    print(
        f"Anchors HUMAN fuertes: {len(frames)} · frames: {frames}"
    )
    print(f"Vistas PARTIDO rechazadas: {rejected}")
    print(
        f"Resolución: {result['image']['width']} × {result['image']['height']} · "
        f"probes físicos: {result['signature_probe_count']}"
    )

    print()
    print("ANCHORS")
    for anchor in result["anchors"]:
        camera = anchor["camera"]
        centre = camera.get("centre_projection_norm")
        centre_text = (
            f"({centre[0]:.3f}, {centre[1]:.3f})"
            if isinstance(centre, list) and len(centre) == 2
            else "—"
        )
        scanner = anchor.get("scanner") or {}
        coords = anchor.get("pca_2d") or {}
        print(
            f"  f {anchor['frame']:>5} · centro={centre_text} · "
            f"zoom≈{_fmt(camera.get('zoom_proxy_px_per_m'), 2)} px/m · "
            f"PCA=({_fmt(coords.get('x'), 3)}, {_fmt(coords.get('y'), 3)}) · "
            f"scanner q={_fmt(scanner.get('quality_score'), 3)}"
        )

    print()
    print("DISTANCIA CÁMARA ↔ CÁMARA · mediana de desplazamiento de probes")
    frames_text = " ".join(f"{frame:>7}" for frame in frames)
    print(f"          {frames_text}")
    matrix = result["pairwise_camera_distance_median_px"]
    for frame, row in zip(frames, matrix):
        print(f"  {frame:>5} " + " ".join(f"{float(value):7.1f}" for value in row))

    print()
    print("VECINO DE CÁMARA MÁS CERCANO")
    for item in result["nearest_neighbors"]:
        print(
            f"  f {item['frame']:>5} → f {item['nearest_frame']:>5} · "
            f"{item['camera_distance_median_px']:.1f} px · "
            f"Δt={item['frame_gap']} frames"
        )

    print()
    print("PASOS TEMPORALES ENTRE ANCHORS")
    for item in result["temporal_steps"]:
        print(
            f"  {item['first_frame']:>5} → {item['second_frame']:>5} · "
            f"{item['camera_distance_median_px']:.1f} px median · "
            f"P90 {item['camera_distance_p90_px']:.1f} px"
        )

    print()
    print("DIMENSIONALIDAD PCA")
    for rank in ("1", "2", "3"):
        info = pca["reconstruction"].get(rank)
        if not info:
            continue
        print(
            f"  {rank}D · varianza acumulada {info['variance_explained'] * 100:.1f}% · "
            f"error reconstrucción mediano {info['median_probe_error_px']:.1f} px · "
            f"máximo {info['max_probe_error_px']:.1f} px"
        )

    print()
    print("INTERPRETACIÓN PCA:")
    print(f"  {pca['interpretation']}")

    rho = result.get("time_camera_spearman")
    print()
    print(
        "CORRELACIÓN |Δframe| ↔ distancia de cámara (Spearman): "
        + (_fmt(rho, 3) if rho is not None else "—")
    )
    print(
        "  Cerca de +1: las vistas se separan con el tiempo. "
        "Cerca de 0: la cámara revisita estados parecidos en momentos lejanos."
    )

    print()
    print("RECONSTRUCCIÓN LOCAL POR DOS OTROS ANCHORS")
    for item in result["local_pair_reconstruction"]:
        a, b = item["between_frames"]
        print(
            f"  f {item['frame']:>5} ≈ segmento [{a}, {b}] · "
            f"α={item['alpha']:.2f} · error {item['median_probe_error_px']:.1f} px"
        )

    print()
    print("=" * 72)
    print("No se ha modificado la review ni se ha entrenado ningún Camera State.")
    print("=" * 72)
    print()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analiza la estructura de cámara de los anchors HUMAN de PARTIDO V1."
    )
    parser.add_argument(
        "review",
        nargs="?",
        default="supervisor/data/reviews/benchmark_easy_2min.json",
        help="Ruta a la review JSON.",
    )
    parser.add_argument(
        "--json",
        dest="json_output",
        default=None,
        help="Opcional: guarda el diagnóstico completo en este JSON.",
    )

    args = parser.parse_args()
    review_path = Path(args.review)

    if not review_path.exists():
        parser.error(f"No existe la review: {review_path}")

    try:
        result = analyze_review(review_path)
    except (RuntimeError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(f"ERROR: {error}") from error

    print_report(result)

    if args.json_output:
        output_path = Path(args.json_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"Diagnóstico JSON guardado en: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
