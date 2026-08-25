#!/usr/bin/env python3
"""Camera State V2 diagnostic for RealJaenLab.

Why V2 exists
-------------
V1 compared cameras by projecting probe points across the entire physical pitch.
That is unsafe for independent HUMAN homographies that are locally excellent
but have a projective pole inside an unseen part of the pitch. In that case,
off-screen extrapolation can explode to thousands of pixels even though the
visible calibration is good.

V2 therefore describes each camera only with quantities evaluated at the
IMAGE CENTRE and the image horizon:

- physical pitch coordinate seen at the image centre (x, y)
- local zoom proxy at the image centre (px / m)
- local perspective anisotropy
- ground-plane horizon signed distance from the image centre
- horizon angle

These are finite, interpretable camera observables and do not require projecting
the whole pitch.

This script is analysis-only:
- reads field.match_setup completed HUMAN anchors
- does not modify the review
- does not train or persist Camera State
- does not use propagation/refinement/legacy Atlas
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
        det = float(np.linalg.det(h))
    except np.linalg.LinAlgError:
        return None

    if not math.isfinite(det) or abs(det) < 1e-12:
        return None

    return h


def _project_one(point: tuple[float, float], h: np.ndarray) -> np.ndarray | None:
    q = h @ np.asarray([point[0], point[1], 1.0], dtype=np.float64)
    if not np.isfinite(q).all() or abs(float(q[2])) < EPS:
        return None
    result = q[:2] / q[2]
    return result if np.isfinite(result).all() else None


def _inverse_map_and_jacobian(
    h: np.ndarray,
    u: float,
    v: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Map image -> pitch and return analytic d(world)/d(image)."""

    try:
        g = np.linalg.inv(h)
    except np.linalg.LinAlgError:
        return None

    g = g / g[2, 2] if abs(float(g[2, 2])) > EPS else g

    a, b, c = g[0]
    d, e, f = g[1]
    p, q, r = g[2]

    denominator = p * u + q * v + r
    if not math.isfinite(float(denominator)) or abs(float(denominator)) < EPS:
        return None

    nx = a * u + b * v + c
    ny = d * u + e * v + f

    x = nx / denominator
    y = ny / denominator

    denominator_sq = denominator * denominator

    dx_du = (a * denominator - nx * p) / denominator_sq
    dx_dv = (b * denominator - nx * q) / denominator_sq
    dy_du = (d * denominator - ny * p) / denominator_sq
    dy_dv = (e * denominator - ny * q) / denominator_sq

    world = np.asarray([x, y], dtype=np.float64)
    jacobian = np.asarray(
        [
            [dx_du, dx_dv],
            [dy_du, dy_dv],
        ],
        dtype=np.float64,
    )

    if not np.isfinite(world).all() or not np.isfinite(jacobian).all():
        return None

    return world, jacobian


def _horizon(h: np.ndarray) -> np.ndarray | None:
    """Ground-plane horizon in image coordinates.

    The pitch-plane line at infinity [0,0,1] maps as H^-T l.
    """

    try:
        line = np.linalg.inv(h).T @ np.asarray([0.0, 0.0, 1.0])
    except np.linalg.LinAlgError:
        return None

    norm = float(np.linalg.norm(line[:2]))
    if not math.isfinite(norm) or norm < EPS:
        return None

    line = line / norm

    # Fix sign so pairwise signed distances are comparable.
    if line[1] < 0 or (abs(float(line[1])) < EPS and line[0] < 0):
        line = -line

    return line


def _angle_wrap_degrees(value: float) -> float:
    while value > 90.0:
        value -= 180.0
    while value <= -90.0:
        value += 180.0
    return value


def _angle_difference_degrees(first: float, second: float) -> float:
    return abs(_angle_wrap_degrees(first - second))


def _camera_observables(
    h: np.ndarray,
    image_width: int,
    image_height: int,
) -> dict[str, Any]:
    cx = float(image_width) / 2.0
    cy = float(image_height) / 2.0

    local = _inverse_map_and_jacobian(h, cx, cy)
    if local is None:
        raise RuntimeError("No se pudo invertir H en el centro de imagen.")

    centre_world, image_to_world_jacobian = local

    try:
        singular_values = np.linalg.svd(
            image_to_world_jacobian,
            compute_uv=False,
        )
    except np.linalg.LinAlgError as error:
        raise RuntimeError("No se pudo calcular la escala local de H.") from error

    if (
        len(singular_values) != 2
        or singular_values[1] <= EPS
        or not np.isfinite(singular_values).all()
    ):
        raise RuntimeError("Jacobiano local degenerado.")

    # |det J_image->world| has units m²/px².
    area_world_per_pixel_sq = abs(float(np.linalg.det(image_to_world_jacobian)))
    if area_world_per_pixel_sq <= EPS:
        raise RuntimeError("Escala local degenerada.")

    zoom_px_per_m = 1.0 / math.sqrt(area_world_per_pixel_sq)
    anisotropy = float(singular_values[0] / singular_values[1])

    horizon = _horizon(h)
    if horizon is None:
        raise RuntimeError("No se pudo obtener el horizonte del plano.")

    a, b, c = [float(value) for value in horizon]
    signed_horizon_distance_px = a * cx + b * cy + c

    # Direction vector of line ax + by + c = 0 is (b, -a).
    horizon_angle_deg = _angle_wrap_degrees(
        math.degrees(math.atan2(-a, b))
    )

    # Forward local field-x direction is useful as a separate sanity diagnostic.
    pitch_centre_image = _project_one(
        (float(centre_world[0]), float(centre_world[1])),
        h,
    )
    pitch_x_image = _project_one(
        (float(centre_world[0] + 1.0), float(centre_world[1])),
        h,
    )

    pitch_x_angle_deg = None
    if pitch_centre_image is not None and pitch_x_image is not None:
        delta = pitch_x_image - pitch_centre_image
        if float(np.linalg.norm(delta)) > EPS:
            pitch_x_angle_deg = _angle_wrap_degrees(
                math.degrees(math.atan2(float(delta[1]), float(delta[0])))
            )

    return {
        "centre_world_m": [
            float(centre_world[0]),
            float(centre_world[1]),
        ],
        "zoom_px_per_m": float(zoom_px_per_m),
        "anisotropy": anisotropy,
        "horizon_signed_distance_px": float(signed_horizon_distance_px),
        "horizon_signed_distance_norm": float(
            signed_horizon_distance_px / max(float(image_height), 1.0)
        ),
        "horizon_angle_deg": float(horizon_angle_deg),
        "pitch_x_angle_deg": pitch_x_angle_deg,
        "image_to_world_jacobian": image_to_world_jacobian.tolist(),
    }


def _feature_vector(
    observables: dict[str, Any],
    length_m: float,
    width_m: float,
) -> np.ndarray:
    centre_x, centre_y = observables["centre_world_m"]
    zoom = max(float(observables["zoom_px_per_m"]), EPS)
    anisotropy = max(float(observables["anisotropy"]), 1.0)

    return np.asarray(
        [
            centre_x / max(length_m, EPS),
            centre_y / max(width_m, EPS),
            math.log(zoom),
            math.log(anisotropy),
            float(observables["horizon_signed_distance_norm"]),
            math.radians(float(observables["horizon_angle_deg"])),
        ],
        dtype=np.float64,
    )


FEATURE_NAMES = [
    "centre_x / length",
    "centre_y / width",
    "log zoom",
    "log anisotropy",
    "horizon distance / height",
    "horizon angle rad",
]


def _rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)

    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + end - 1) / 2.0
        ranks[order[start:end]] = rank
        start = end

    return ranks


def _spearman(first: list[float], second: list[float]) -> float | None:
    if len(first) < 3 or len(first) != len(second):
        return None

    a = _rankdata(np.asarray(first))
    b = _rankdata(np.asarray(second))
    a -= np.mean(a)
    b -= np.mean(b)

    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator < EPS:
        return None

    return float(np.dot(a, b) / denominator)


def _standardise_fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0, ddof=0)
    std = np.where(std < 1e-8, 1.0, std)
    return mean, std


def _standardise_apply(
    x: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    return (x - mean) / std


def _pca_fit(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centered = z - np.mean(z, axis=0)
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    return singular_values, vt


def _global_pca(features: np.ndarray) -> dict[str, Any]:
    mean, std = _standardise_fit(features)
    z = _standardise_apply(features, mean, std)
    z_mean = np.mean(z, axis=0)
    centered = z - z_mean

    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)

    variance = singular_values ** 2
    total = float(np.sum(variance))
    ratio = variance / total if total > EPS else np.zeros_like(variance)
    cumulative = np.cumsum(ratio)

    max_rank = min(len(features) - 1, len(vt))
    coords_3 = np.zeros((len(features), 3), dtype=np.float64)

    if max_rank > 0:
        use = min(3, max_rank)
        coords_3[:, :use] = centered @ vt[:use].T

    return {
        "mean": mean,
        "std": std,
        "z_mean": z_mean,
        "components": vt,
        "variance_ratio": ratio[:max_rank],
        "cumulative_variance": cumulative[:max_rank],
        "coordinates_3d": coords_3,
    }


def _loo_pca(features: np.ndarray, rank: int) -> dict[str, Any]:
    errors: list[float] = []
    per_anchor: list[float] = []

    for holdout in range(len(features)):
        train_indices = [
            index for index in range(len(features))
            if index != holdout
        ]

        train = features[train_indices]
        test = features[holdout]

        mean, std = _standardise_fit(train)
        train_z = _standardise_apply(train, mean, std)
        test_z = _standardise_apply(test, mean, std)

        train_mean = np.mean(train_z, axis=0)
        centered_train = train_z - train_mean

        _, _, vt = np.linalg.svd(centered_train, full_matrices=False)

        available_rank = min(rank, len(train) - 1, len(vt))
        if available_rank <= 0:
            error = float(np.sqrt(np.mean((test_z - train_mean) ** 2)))
        else:
            basis = vt[:available_rank]
            delta = test_z - train_mean
            reconstruction = train_mean + (delta @ basis.T) @ basis
            error = float(np.sqrt(np.mean((test_z - reconstruction) ** 2)))

        errors.append(error)
        per_anchor.append(error)

    return {
        "rank": rank,
        "median_standardised_rmse": float(np.median(errors)),
        "max_standardised_rmse": float(np.max(errors)),
        "per_anchor_standardised_rmse": per_anchor,
    }


def _pairwise_state(
    anchors: list[dict[str, Any]],
    features: np.ndarray,
) -> tuple[list[list[float]], list[dict[str, Any]]]:
    mean, std = _standardise_fit(features)
    z = _standardise_apply(features, mean, std)

    count = len(anchors)
    matrix = np.zeros((count, count), dtype=np.float64)
    pairs: list[dict[str, Any]] = []

    for first in range(count):
        for second in range(first + 1, count):
            state_distance = float(
                np.linalg.norm(z[first] - z[second]) / math.sqrt(z.shape[1])
            )

            first_obs = anchors[first]["observables"]
            second_obs = anchors[second]["observables"]

            first_centre = np.asarray(first_obs["centre_world_m"])
            second_centre = np.asarray(second_obs["centre_world_m"])

            centre_distance = float(np.linalg.norm(first_centre - second_centre))

            first_zoom = float(first_obs["zoom_px_per_m"])
            second_zoom = float(second_obs["zoom_px_per_m"])
            zoom_ratio = max(first_zoom, second_zoom) / max(
                min(first_zoom, second_zoom),
                EPS,
            )

            horizon_distance_delta = abs(
                float(first_obs["horizon_signed_distance_px"])
                - float(second_obs["horizon_signed_distance_px"])
            )

            horizon_angle_delta = _angle_difference_degrees(
                float(first_obs["horizon_angle_deg"]),
                float(second_obs["horizon_angle_deg"]),
            )

            matrix[first, second] = state_distance
            matrix[second, first] = state_distance

            pairs.append(
                {
                    "first_frame": anchors[first]["frame"],
                    "second_frame": anchors[second]["frame"],
                    "frame_gap": abs(
                        anchors[first]["frame"] - anchors[second]["frame"]
                    ),
                    "state_distance": state_distance,
                    "centre_world_distance_m": centre_distance,
                    "zoom_ratio": zoom_ratio,
                    "horizon_distance_delta_px": horizon_distance_delta,
                    "horizon_angle_delta_deg": horizon_angle_delta,
                }
            )

    return matrix.tolist(), pairs


def _collect_review(
    review_path: Path,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    list[dict[str, Any]],
    int,
    int,
]:
    review = json.loads(review_path.read_text(encoding="utf-8"))

    field = review.get("field")
    if not isinstance(field, dict):
        raise RuntimeError("La review no contiene field.")

    pitch_model = field.get("pitch_model")
    if not isinstance(pitch_model, dict) or pitch_model.get("status") != "confirmed":
        raise RuntimeError("field.pitch_model no está confirmado.")

    match_setup = field.get("match_setup")
    if not isinstance(match_setup, dict):
        raise RuntimeError("La review no contiene field.match_setup.")

    completed = {
        int(value)
        for value in match_setup.get("completed_frames", [])
        if isinstance(value, (int, float)) or str(value).isdigit()
    }

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

        if frame not in completed:
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
            }
        )

    anchors.sort(key=lambda item: item["frame"])

    missing = sorted(completed - {item["frame"] for item in anchors})
    if missing:
        raise RuntimeError(
            "completed_frames sin H HUMAN match_setup válida: "
            + ", ".join(map(str, missing))
        )

    if len(anchors) < 4:
        raise RuntimeError(
            f"Solo hay {len(anchors)} anchors fuertes; V2 necesita al menos 4."
        )

    image_width = None
    image_height = None

    scan = match_setup.get("scan")
    if isinstance(scan, dict):
        metadata = scan.get("metadata")
        if isinstance(metadata, dict):
            image_width = metadata.get("width")
            image_height = metadata.get("height")

    if not image_width or not image_height:
        video = review.get("video")
        if isinstance(video, dict):
            image_width = video.get("width")
            image_height = video.get("height")

    if not image_width or not image_height:
        raise RuntimeError("No se pudo determinar la resolución del vídeo.")

    return review, pitch_model, anchors, int(image_width), int(image_height)


def analyze(review_path: Path) -> dict[str, Any]:
    (
        review,
        pitch_model,
        anchors,
        image_width,
        image_height,
    ) = _collect_review(review_path)

    length_m = float(pitch_model["length_m"])
    width_m = float(pitch_model["width_m"])

    feature_rows = []

    for anchor in anchors:
        observables = _camera_observables(
            anchor["homography"],
            image_width,
            image_height,
        )
        anchor["observables"] = observables
        feature_rows.append(
            _feature_vector(
                observables,
                length_m,
                width_m,
            )
        )

    features = np.asarray(feature_rows, dtype=np.float64)

    pca = _global_pca(features)

    for anchor, coordinates in zip(
        anchors,
        pca["coordinates_3d"],
    ):
        anchor["pca_coordinates"] = [
            float(coordinates[0]),
            float(coordinates[1]),
            float(coordinates[2]),
        ]

    pairwise, pairs = _pairwise_state(
        anchors,
        features,
    )

    frame_gaps = [float(pair["frame_gap"]) for pair in pairs]
    state_distances = [float(pair["state_distance"]) for pair in pairs]

    nearest = []
    matrix = np.asarray(pairwise, dtype=np.float64)

    for index, anchor in enumerate(anchors):
        candidates = [
            (matrix[index, other], other)
            for other in range(len(anchors))
            if other != index
        ]
        candidates.sort(key=lambda item: item[0])
        distance, other = candidates[0]
        nearest.append(
            {
                "frame": anchor["frame"],
                "nearest_frame": anchors[other]["frame"],
                "state_distance": float(distance),
                "frame_gap": abs(anchor["frame"] - anchors[other]["frame"]),
            }
        )

    loo = {
        str(rank): _loo_pca(features, rank)
        for rank in (1, 2, 3)
    }

    public_anchors = []
    for anchor in anchors:
        calibration = anchor["calibration"]
        constraints = calibration.get("constraints")
        if not isinstance(constraints, list):
            constraints = []

        public_anchors.append(
            {
                "frame": anchor["frame"],
                "observables": anchor["observables"],
                "pca_coordinates": anchor["pca_coordinates"],
                "constraint_counts": {
                    "point": sum(
                        1 for item in constraints
                        if isinstance(item, dict) and item.get("type") == "point"
                    ),
                    "line": sum(
                        1 for item in constraints
                        if isinstance(item, dict) and item.get("type") == "line"
                    ),
                    "arc": sum(
                        1 for item in constraints
                        if isinstance(item, dict) and item.get("type") == "arc"
                    ),
                },
                "reprojection_error": calibration.get(
                    "reprojectionError",
                    calibration.get("reprojection_error"),
                ),
            }
        )

    match_setup = review["field"]["match_setup"]

    return {
        "status": "ok",
        "review": str(review_path),
        "pitch_model": pitch_model,
        "image": {
            "width": image_width,
            "height": image_height,
        },
        "strong_anchor_frames": [
            anchor["frame"] for anchor in anchors
        ],
        "rejected_frames": match_setup.get(
            "rejected_frames",
            [],
        ),
        "feature_names": FEATURE_NAMES,
        "anchors": public_anchors,
        "pairwise_state_distance": pairwise,
        "pairs": pairs,
        "nearest_neighbors": nearest,
        "time_camera_spearman": _spearman(
            frame_gaps,
            state_distances,
        ),
        "pca": {
            "variance_ratio": [
                float(value)
                for value in pca["variance_ratio"]
            ],
            "cumulative_variance": [
                float(value)
                for value in pca["cumulative_variance"]
            ],
            "loo": loo,
        },
        "notes": [
            "No full-pitch projection probes are used.",
            "Camera observables are evaluated at the image centre plus the ground-plane horizon.",
            "PCA is exploratory and is not persisted as Camera State.",
            "LOO PCA fits mean/std/basis only on the training anchors for each holdout.",
        ],
    }


def _fmt(value: Any, digits: int = 2) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(value):
        return "—"
    return f"{value:.{digits}f}"


def print_report(result: dict[str, Any]) -> None:
    frames = result["strong_anchor_frames"]

    print()
    print("=" * 78)
    print("CAMERA STATE V2 · OBSERVABLES LOCALES ROBUSTOS")
    print("=" * 78)
    pitch = result["pitch_model"]
    print(
        f"Pitch operativo: {float(pitch['length_m']):.1f} × "
        f"{float(pitch['width_m']):.1f} m"
    )
    print(f"Anchors fuertes: {frames}")
    print(f"Rechazados: {result['rejected_frames']}")
    print(
        f"Resolución: {result['image']['width']} × "
        f"{result['image']['height']}"
    )

    print()
    print("OBSERVABLES POR ANCHOR")
    for anchor in result["anchors"]:
        obs = anchor["observables"]
        x, y = obs["centre_world_m"]
        pca = anchor["pca_coordinates"]
        counts = anchor["constraint_counts"]
        print(
            f"  f {anchor['frame']:>5} · "
            f"centro campo=({_fmt(x,1)}, {_fmt(y,1)}) m · "
            f"zoom={_fmt(obs['zoom_px_per_m'],2)} px/m · "
            f"anis={_fmt(obs['anisotropy'],2)} · "
            f"horiz dist={_fmt(obs['horizon_signed_distance_px'],1)} px · "
            f"horiz ang={_fmt(obs['horizon_angle_deg'],2)}° · "
            f"PCA=({_fmt(pca[0],2)}, {_fmt(pca[1],2)}, {_fmt(pca[2],2)}) · "
            f"HUMAN={counts['point']}p/{counts['line']}l/{counts['arc']}a"
        )

    print()
    print("DISTANCIA DE ESTADO ROBUSTA · z-score RMS")
    print("          " + " ".join(f"{frame:>7}" for frame in frames))
    for frame, row in zip(frames, result["pairwise_state_distance"]):
        print(
            f"  {frame:>5} "
            + " ".join(f"{float(value):7.2f}" for value in row)
        )

    print()
    print("VECINO ROBUSTO MÁS CERCANO")
    for item in result["nearest_neighbors"]:
        print(
            f"  f {item['frame']:>5} → f {item['nearest_frame']:>5} · "
            f"d={item['state_distance']:.2f} · "
            f"Δt={item['frame_gap']} frames"
        )

    print()
    print("DETALLE FÍSICO ENTRE PARES")
    for pair in sorted(
        result["pairs"],
        key=lambda item: item["state_distance"],
    ):
        print(
            f"  {pair['first_frame']:>5} ↔ {pair['second_frame']:>5} · "
            f"d={pair['state_distance']:.2f} · "
            f"centros={pair['centre_world_distance_m']:.1f} m · "
            f"zoom×{pair['zoom_ratio']:.2f} · "
            f"Δhoriz={pair['horizon_distance_delta_px']:.1f} px · "
            f"Δang={pair['horizon_angle_delta_deg']:.2f}°"
        )

    print()
    print("PCA IN-SAMPLE · solo descriptivo")
    cumulative = result["pca"]["cumulative_variance"]
    for index, value in enumerate(cumulative[:3], start=1):
        print(
            f"  {index}D → varianza acumulada {value * 100:.1f}%"
        )

    print()
    print("PCA LEAVE-ONE-OUT · prueba más importante")
    for rank in ("1", "2", "3"):
        info = result["pca"]["loo"][rank]
        errors = info["per_anchor_standardised_rmse"]
        details = ", ".join(
            f"f{frame}:{error:.2f}"
            for frame, error in zip(frames, errors)
        )
        print(
            f"  {rank}D · RMSE z mediana={info['median_standardised_rmse']:.2f} · "
            f"máx={info['max_standardised_rmse']:.2f} · {details}"
        )

    rho = result["time_camera_spearman"]
    print()
    print(
        "Spearman |Δframe| ↔ distancia estado robusta: "
        + (_fmt(rho, 3) if rho is not None else "—")
    )

    print()
    print("LECTURA")
    loo2 = result["pca"]["loo"]["2"]["median_standardised_rmse"]
    loo3 = result["pca"]["loo"]["3"]["median_standardised_rmse"]

    if loo3 < loo2 * 0.75:
        print(
            "  3D generaliza claramente mejor que 2D en leave-one-out. "
            "La hipótesis Camera State ≈ 3D gana fuerza."
        )
    elif loo2 <= loo3 * 1.10:
        print(
            "  2D no pierde claramente frente a 3D en leave-one-out. "
            "No hay motivo todavía para pagar una tercera dimensión."
        )
    else:
        print(
            "  2D y 3D están demasiado cerca/inestables para decidir con 5 anchors."
        )

    print(
        "  Este diagnóstico evita por completo la extrapolación del campo entero "
        "que hacía explotar V1."
    )

    print()
    print("=" * 78)
    print("No se ha modificado ninguna review ni ningún módulo core.")
    print("=" * 78)
    print()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "review",
        nargs="?",
        default="supervisor/data/reviews/benchmark_easy_2min.json",
    )
    parser.add_argument(
        "--json",
        dest="json_output",
        default=None,
    )

    args = parser.parse_args()
    review_path = Path(args.review)

    if not review_path.exists():
        parser.error(f"No existe: {review_path}")

    try:
        result = analyze(review_path)
    except (
        RuntimeError,
        KeyError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as error:
        raise SystemExit(f"ERROR: {error}") from error

    print_report(result)

    if args.json_output:
        output = Path(args.json_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(
                result,
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"JSON guardado en: {output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
