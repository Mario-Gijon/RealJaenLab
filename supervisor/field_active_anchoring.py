#!/usr/bin/env python3
"""Active Anchoring V2 for RealJaenLab.

Purpose
-------
Given the strong HUMAN anchors already collected by PARTIDO V1, find new raw
video frames that fill the largest current camera-state coverage gap.

V2 is rejection-aware: a HUMAN-rejected PARTIDO frame is treated as negative
evidence for independent projective observability in its immediate descriptor
neighbourhood. The algorithm no longer rewards similarity to that rejected
state. Instead it seeks a bridge across the gap while avoiding candidates that
are too similar to the rejected view.

This module is analysis-only. It does NOT:
- modify the review;
- create calibrations;
- estimate homographies for candidate frames;
- use PEOPLE;
- use propagation/refinement;
- train Camera State.

Design
------
1. Build robust local camera observables from each completed HUMAN match_setup H.
2. Find the largest gap between consecutive HUMAN anchors in temporal order.
3. Decode only the raw-video interval inside that gap.
4. Use the existing geometry-first FIELD scanner descriptor to search for frames
   near the descriptor midpoint between the two strong anchors.
5. If a HUMAN-rejected PARTIDO frame lies inside the gap, suppress the immediate
   descriptor neighbourhood of that rejected view. This does NOT mean the camera
   state is invalid; it means that neighbourhood has already produced a frame
   that was not independently observable enough for a trustworthy H.
6. Return a small diverse shortlist for HUMAN inspection.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import argparse
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from supervisor.field_view_scanner import (
    FieldViewScannerConfig,
    _descriptor,
    field_view_distance,
)


EPS = 1e-10


@dataclass(frozen=True)
class ActiveAnchoringConfig:
    # Raw-video search density inside the selected coverage gap.
    sample_step_frames: int = 2

    # Avoid proposing an endpoint anchor again.
    endpoint_exclusion_frames: int = 10

    # Avoid proposing essentially the exact rejected frame again, while still
    # allowing the same camera-state neighborhood to be searched.
    rejected_exact_exclusion_frames: int = 5

    # How many frames are returned to the HUMAN.
    maximum_recommendations: int = 5

    # Keep shortlist items reasonably distinct in time.
    minimum_temporal_separation_frames: int = 10

    # Keep shortlist items reasonably distinct in scanner descriptor space.
    minimum_descriptor_separation: float = 0.008

    # Reject very weak geometry outright.
    minimum_measurement_utility: float = 0.20

    # Rejection-aware guard in FIELD-scanner descriptor distance.
    # A candidate closer than this to a HUMAN-rejected view is suppressed.
    rejected_descriptor_guard_radius: float = 0.055

    # Ranking weights.
    target_similarity_weight: float = 0.46
    measurement_utility_weight: float = 0.30
    bridge_balance_weight: float = 0.12
    rejected_separation_weight: float = 0.12


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
    except np.linalg.LinAlgError:
        return None

    if not math.isfinite(determinant) or abs(determinant) < 1e-12:
        return None

    return h


def _inverse_map_and_jacobian(
    h: np.ndarray,
    u: float,
    v: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    try:
        g = np.linalg.inv(h)
    except np.linalg.LinAlgError:
        return None

    a, b, c = g[0]
    d, e, f = g[1]
    p, q, r = g[2]

    denominator = p * u + q * v + r
    if not math.isfinite(float(denominator)) or abs(float(denominator)) < EPS:
        return None

    nx = a * u + b * v + c
    ny = d * u + e * v + f

    world = np.asarray(
        [
            nx / denominator,
            ny / denominator,
        ],
        dtype=np.float64,
    )

    denominator_sq = denominator * denominator
    jacobian = np.asarray(
        [
            [
                (a * denominator - nx * p) / denominator_sq,
                (b * denominator - nx * q) / denominator_sq,
            ],
            [
                (d * denominator - ny * p) / denominator_sq,
                (e * denominator - ny * q) / denominator_sq,
            ],
        ],
        dtype=np.float64,
    )

    if not np.isfinite(world).all() or not np.isfinite(jacobian).all():
        return None

    return world, jacobian


def _horizon(h: np.ndarray) -> np.ndarray | None:
    try:
        line = np.linalg.inv(h).T @ np.asarray(
            [0.0, 0.0, 1.0],
            dtype=np.float64,
        )
    except np.linalg.LinAlgError:
        return None

    norm = float(np.linalg.norm(line[:2]))
    if not math.isfinite(norm) or norm < EPS:
        return None

    line = line / norm

    if line[1] < 0 or (
        abs(float(line[1])) < EPS
        and line[0] < 0
    ):
        line = -line

    return line


def _camera_feature(
    h: np.ndarray,
    *,
    image_width: int,
    image_height: int,
    length_m: float,
    width_m: float,
) -> np.ndarray | None:
    cx = float(image_width) / 2.0
    cy = float(image_height) / 2.0

    local = _inverse_map_and_jacobian(
        h,
        cx,
        cy,
    )
    if local is None:
        return None

    centre_world, image_to_world = local

    try:
        singular_values = np.linalg.svd(
            image_to_world,
            compute_uv=False,
        )
    except np.linalg.LinAlgError:
        return None

    if (
        len(singular_values) != 2
        or singular_values[1] <= EPS
        or not np.isfinite(singular_values).all()
    ):
        return None

    area_world_per_pixel_sq = abs(
        float(np.linalg.det(image_to_world))
    )
    if area_world_per_pixel_sq <= EPS:
        return None

    zoom_px_per_m = 1.0 / math.sqrt(
        area_world_per_pixel_sq
    )
    anisotropy = float(
        singular_values[0] / singular_values[1]
    )

    horizon = _horizon(h)
    if horizon is None:
        return None

    a, b, c = [
        float(value)
        for value in horizon
    ]

    horizon_distance_norm = (
        a * cx + b * cy + c
    ) / max(float(image_height), 1.0)

    horizon_angle = math.atan2(
        -a,
        b,
    )

    return np.asarray(
        [
            float(centre_world[0]) / max(length_m, EPS),
            float(centre_world[1]) / max(width_m, EPS),
            math.log(max(zoom_px_per_m, EPS)),
            math.log(max(anisotropy, 1.0)),
            float(horizon_distance_norm),
            float(horizon_angle),
        ],
        dtype=np.float64,
    )


def _standardised_camera_features(
    anchors: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    matrix = np.asarray(
        [anchor["camera_feature"] for anchor in anchors],
        dtype=np.float64,
    )

    mean = np.mean(
        matrix,
        axis=0,
    )
    std = np.std(
        matrix,
        axis=0,
        ddof=0,
    )
    std = np.where(
        std < 1e-8,
        1.0,
        std,
    )

    return (
        (matrix - mean) / std,
        mean,
        std,
    )


def _camera_state_distance(
    first: np.ndarray,
    second: np.ndarray,
) -> float:
    delta = np.asarray(first) - np.asarray(second)
    return float(
        np.linalg.norm(delta)
        / math.sqrt(max(delta.size, 1))
    )


def _measurement_utility(
    descriptor_result: dict[str, Any],
) -> float:
    geometry = descriptor_result.get("geometry")
    if not isinstance(geometry, dict):
        geometry = {}

    quality = float(
        np.clip(
            descriptor_result.get(
                "quality_score",
                0.0,
            ),
            0.0,
            1.0,
        )
    )

    line_families = min(
        1.0,
        max(
            0.0,
            float(
                geometry.get(
                    "line_family_count",
                    0,
                )
            )
            / 3.0,
        ),
    )

    intersections = min(
        1.0,
        max(
            0.0,
            float(
                geometry.get(
                    "intersection_count",
                    0,
                )
            )
            / 8.0,
        ),
    )

    line_count = min(
        1.0,
        max(
            0.0,
            float(
                geometry.get(
                    "line_count",
                    0,
                )
            )
            / 10.0,
        ),
    )

    sharpness = min(
        1.0,
        max(
            0.0,
            float(
                geometry.get(
                    "sharpness",
                    0.0,
                )
            ),
        ),
    )

    return float(
        0.50 * quality
        + 0.20 * line_families
        + 0.15 * intersections
        + 0.10 * line_count
        + 0.05 * sharpness
    )


def _normalised_midpoint_descriptor(
    first: np.ndarray,
    second: np.ndarray,
) -> np.ndarray:
    midpoint = (
        np.asarray(first, dtype=np.float64)
        + np.asarray(second, dtype=np.float64)
    )
    norm = float(
        np.linalg.norm(midpoint)
    )

    if norm < EPS:
        return np.asarray(
            first,
            dtype=np.float64,
        )

    return midpoint / norm


def _similarity_score(
    distance: float,
    scale: float,
) -> float:
    scale = max(
        float(scale),
        0.01,
    )

    return float(
        math.exp(
            -max(float(distance), 0.0)
            / scale
        )
    )


def _balance_score(
    left_distance: float,
    right_distance: float,
) -> tuple[float, float]:
    total = left_distance + right_distance

    if total <= EPS:
        return 1.0, 0.5

    alpha = left_distance / total
    balance = max(
        0.0,
        1.0 - 2.0 * abs(alpha - 0.5),
    )

    return float(balance), float(alpha)


def _read_frame(
    capture: cv2.VideoCapture,
    frame: int,
) -> np.ndarray | None:
    capture.set(
        cv2.CAP_PROP_POS_FRAMES,
        int(frame),
    )
    ok, image = capture.read()
    return image if ok else None


def _review_video_path(
    review: dict[str, Any],
    explicit_video: str | None,
) -> Path:
    if explicit_video:
        return Path(explicit_video)

    video = review.get("video")
    filename = (
        video.get("filename")
        if isinstance(video, dict)
        else None
    )

    if not filename:
        raise RuntimeError(
            "La review no contiene video.filename. "
            "Usa --video."
        )

    return Path("input") / str(filename)


def _image_size(
    review: dict[str, Any],
    match_setup: dict[str, Any],
    capture: cv2.VideoCapture,
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

    width = int(
        capture.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
        or 0
    )
    height = int(
        capture.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
        or 0
    )

    if width <= 0 or height <= 0:
        raise RuntimeError(
            "No se pudo determinar la resolución del vídeo."
        )

    return width, height


def _collect_anchors(
    review: dict[str, Any],
    *,
    image_width: int,
    image_height: int,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    list[dict[str, Any]],
]:
    field = review.get("field")
    if not isinstance(field, dict):
        raise RuntimeError(
            "La review no contiene field."
        )

    pitch_model = field.get("pitch_model")
    if (
        not isinstance(pitch_model, dict)
        or pitch_model.get("status") != "confirmed"
    ):
        raise RuntimeError(
            "field.pitch_model debe estar confirmado."
        )

    length_m = float(
        pitch_model["length_m"]
    )
    width_m = float(
        pitch_model["width_m"]
    )

    match_setup = field.get("match_setup")
    if not isinstance(match_setup, dict):
        raise RuntimeError(
            "La review no contiene field.match_setup."
        )

    completed_frames: set[int] = set()

    for value in match_setup.get(
        "completed_frames",
        [],
    ):
        try:
            completed_frames.add(
                int(value)
            )
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
            frame = int(
                calibration.get("frame")
            )
        except (TypeError, ValueError):
            continue

        if frame not in completed_frames:
            continue

        if calibration.get("source") != "manual":
            continue

        if calibration.get("purpose") != "match_setup":
            continue

        h = _as_h(
            calibration.get(
                "homography"
            )
        )
        if h is None:
            continue

        feature = _camera_feature(
            h,
            image_width=image_width,
            image_height=image_height,
            length_m=length_m,
            width_m=width_m,
        )
        if feature is None:
            continue

        anchors.append(
            {
                "frame": frame,
                "homography": h,
                "camera_feature": feature,
            }
        )

    anchors.sort(
        key=lambda item: item["frame"]
    )

    missing = sorted(
        completed_frames
        - {
            item["frame"]
            for item in anchors
        }
    )

    if missing:
        raise RuntimeError(
            "Hay completed_frames sin H HUMAN match_setup válida: "
            + ", ".join(
                map(
                    str,
                    missing,
                )
            )
        )

    if len(anchors) < 3:
        raise RuntimeError(
            "Se necesitan al menos 3 anchors HUMAN fuertes."
        )

    return (
        pitch_model,
        match_setup,
        anchors,
    )


def _largest_temporal_camera_gap(
    anchors: list[dict[str, Any]],
) -> dict[str, Any]:
    standardised, _, _ = (
        _standardised_camera_features(
            anchors
        )
    )

    gaps: list[dict[str, Any]] = []

    for index in range(
        len(anchors) - 1
    ):
        first = anchors[index]
        second = anchors[index + 1]

        distance = _camera_state_distance(
            standardised[index],
            standardised[index + 1],
        )

        gaps.append(
            {
                "left_index": index,
                "right_index": index + 1,
                "left_frame": first["frame"],
                "right_frame": second["frame"],
                "frame_gap": (
                    second["frame"]
                    - first["frame"]
                ),
                "robust_state_distance": distance,
            }
        )

    if not gaps:
        raise RuntimeError(
            "No hay huecos temporales entre anchors."
        )

    gaps.sort(
        key=lambda item: (
            -item["robust_state_distance"],
            -item["frame_gap"],
        )
    )

    return {
        "selected": gaps[0],
        "all": gaps,
    }


def _descriptor_for_frame(
    capture: cv2.VideoCapture,
    frame: int,
    scanner_config: FieldViewScannerConfig,
) -> dict[str, Any] | None:
    image = _read_frame(
        capture,
        frame,
    )
    if image is None:
        return None

    result = _descriptor(
        image,
        scanner_config,
    )

    result = dict(result)
    result["frame"] = int(frame)
    result["measurement_utility"] = (
        _measurement_utility(
            result
        )
    )

    return result


def _candidate_target_score(
    candidate_descriptor: np.ndarray,
    *,
    midpoint_descriptor: np.ndarray,
    rejected_descriptors: list[np.ndarray],
    gap_descriptor_distance: float,
    rejected_guard_radius: float,
) -> tuple[
    float,
    float,
    float | None,
    float,
]:
    """Score bridge similarity and rejection-aware separation.

    HUMAN rejection is not interpreted as "bad camera state". It is interpreted
    as "this immediate raw-view neighbourhood has already failed independent H
    observability once". Therefore V2 never rewards similarity to it.
    """

    midpoint_distance = field_view_distance(
        candidate_descriptor,
        midpoint_descriptor,
    )

    midpoint_score = _similarity_score(
        midpoint_distance,
        max(
            gap_descriptor_distance * 0.28,
            0.012,
        ),
    )

    rejected_distance = None
    rejected_separation_score = 1.0

    if rejected_descriptors:
        rejected_distance = min(
            field_view_distance(
                candidate_descriptor,
                descriptor,
            )
            for descriptor in rejected_descriptors
        )

        if rejected_distance < rejected_guard_radius:
            return (
                midpoint_score,
                midpoint_distance,
                rejected_distance,
                -1.0,
            )

        # 0 just outside the guard, saturating to 1 after roughly one more guard
        # radius. This is deliberately weak compared with midpoint similarity:
        # its job is to avoid repeatedly proposing the rejected neighbourhood.
        rejected_separation_score = float(
            np.clip(
                (rejected_distance - rejected_guard_radius)
                / max(rejected_guard_radius, 1e-9),
                0.0,
                1.0,
            )
        )

    return (
        midpoint_score,
        midpoint_distance,
        rejected_distance,
        rejected_separation_score,
    )

def select_active_anchor_candidates(
    review_path: str | Path,
    *,
    video_path: str | Path | None = None,
    config: ActiveAnchoringConfig | None = None,
    scanner_config: FieldViewScannerConfig | None = None,
) -> dict[str, Any]:
    cfg = config or ActiveAnchoringConfig()
    scan_cfg = scanner_config or FieldViewScannerConfig()

    path = Path(
        review_path
    )

    if not path.exists():
        return {
            "status": "review_not_found",
            "review": str(path),
        }

    try:
        review = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )
    except json.JSONDecodeError:
        return {
            "status": "invalid_review_json",
            "review": str(path),
        }

    resolved_video_path = _review_video_path(
        review,
        (
            str(video_path)
            if video_path is not None
            else None
        ),
    )

    if not resolved_video_path.exists():
        return {
            "status": "video_not_found",
            "video": str(
                resolved_video_path
            ),
        }

    capture = cv2.VideoCapture(
        str(
            resolved_video_path
        )
    )

    if not capture.isOpened():
        return {
            "status": "video_open_failed",
            "video": str(
                resolved_video_path
            ),
        }

    try:
        # Image size must be known before camera observables can be built.
        field = review.get("field")
        match_setup_raw = (
            field.get("match_setup")
            if isinstance(field, dict)
            else {}
        )
        if not isinstance(
            match_setup_raw,
            dict,
        ):
            match_setup_raw = {}

        image_width, image_height = _image_size(
            review,
            match_setup_raw,
            capture,
        )

        (
            pitch_model,
            match_setup,
            anchors,
        ) = _collect_anchors(
            review,
            image_width=image_width,
            image_height=image_height,
        )

        gap_analysis = _largest_temporal_camera_gap(
            anchors
        )
        gap = gap_analysis["selected"]

        left_frame = int(
            gap["left_frame"]
        )
        right_frame = int(
            gap["right_frame"]
        )

        left_descriptor_result = _descriptor_for_frame(
            capture,
            left_frame,
            scan_cfg,
        )
        right_descriptor_result = _descriptor_for_frame(
            capture,
            right_frame,
            scan_cfg,
        )

        if (
            left_descriptor_result is None
            or right_descriptor_result is None
        ):
            raise RuntimeError(
                "No se pudieron leer los frames extremos del hueco."
            )

        left_descriptor = np.asarray(
            left_descriptor_result[
                "descriptor"
            ],
            dtype=np.float64,
        )
        right_descriptor = np.asarray(
            right_descriptor_result[
                "descriptor"
            ],
            dtype=np.float64,
        )

        gap_descriptor_distance = field_view_distance(
            left_descriptor,
            right_descriptor,
        )

        midpoint_descriptor = (
            _normalised_midpoint_descriptor(
                left_descriptor,
                right_descriptor,
            )
        )

        rejected_frames: list[int] = []

        for value in match_setup.get(
            "rejected_frames",
            [],
        ):
            try:
                frame = int(
                    value
                )
            except (TypeError, ValueError):
                continue

            if left_frame < frame < right_frame:
                rejected_frames.append(
                    frame
                )

        rejected_descriptors: list[np.ndarray] = []
        rejected_utilities: list[float] = []

        for frame in rejected_frames:
            descriptor_result = _descriptor_for_frame(
                capture,
                frame,
                scan_cfg,
            )
            if descriptor_result is None:
                continue

            rejected_descriptors.append(
                np.asarray(
                    descriptor_result[
                        "descriptor"
                    ],
                    dtype=np.float64,
                )
            )
            rejected_utilities.append(
                float(
                    descriptor_result[
                        "measurement_utility"
                    ]
                )
            )

        baseline_rejected_utility = (
            max(
                rejected_utilities
            )
            if rejected_utilities
            else 0.0
        )

        existing_anchor_frames = {
            int(
                anchor["frame"]
            )
            for anchor in anchors
        }

        all_rejected_frames = {
            int(value)
            for value in match_setup.get(
                "rejected_frames",
                [],
            )
            if (
                isinstance(
                    value,
                    (int, float),
                )
                or str(value).isdigit()
            )
        }

        candidates: list[dict[str, Any]] = []

        search_start = (
            left_frame
            + cfg.endpoint_exclusion_frames
        )
        search_end = (
            right_frame
            - cfg.endpoint_exclusion_frames
        )

        step = max(
            1,
            int(
                cfg.sample_step_frames
            ),
        )

        for frame in range(
            search_start,
            search_end + 1,
            step,
        ):
            if frame in existing_anchor_frames:
                continue

            if any(
                abs(
                    frame - rejected
                )
                <= cfg.rejected_exact_exclusion_frames
                for rejected in all_rejected_frames
            ):
                continue

            descriptor_result = _descriptor_for_frame(
                capture,
                frame,
                scan_cfg,
            )
            if descriptor_result is None:
                continue

            utility = float(
                descriptor_result[
                    "measurement_utility"
                ]
            )

            if utility < cfg.minimum_measurement_utility:
                continue

            descriptor = np.asarray(
                descriptor_result[
                    "descriptor"
                ],
                dtype=np.float64,
            )

            d_left = field_view_distance(
                left_descriptor,
                descriptor,
            )
            d_right = field_view_distance(
                descriptor,
                right_descriptor,
            )

            balance, alpha = _balance_score(
                d_left,
                d_right,
            )

            (
                target_similarity,
                midpoint_distance,
                rejected_state_distance,
                rejected_separation_score,
            ) = _candidate_target_score(
                descriptor,
                midpoint_descriptor=midpoint_descriptor,
                rejected_descriptors=rejected_descriptors,
                gap_descriptor_distance=gap_descriptor_distance,
                rejected_guard_radius=cfg.rejected_descriptor_guard_radius,
            )

            # Hard suppress the immediate HUMAN-rejected descriptor neighbourhood.
            if rejected_separation_score < 0.0:
                continue

            score = (
                cfg.target_similarity_weight
                * target_similarity
                + cfg.measurement_utility_weight
                * utility
                + cfg.bridge_balance_weight
                * balance
                + cfg.rejected_separation_weight
                * rejected_separation_score
            )

            geometry = descriptor_result.get(
                "geometry",
                {},
            )

            candidates.append(
                {
                    "frame": int(
                        frame
                    ),
                    "score": float(
                        score
                    ),
                    "target_mode": "gap_midpoint_rejection_aware",
                    "target_similarity": float(
                        target_similarity
                    ),
                    "measurement_utility": utility,
                    "bridge_balance": float(
                        balance
                    ),
                    "descriptor_alpha": float(
                        alpha
                    ),
                    "distance_from_left": float(
                        d_left
                    ),
                    "distance_from_right": float(
                        d_right
                    ),
                    "distance_from_midpoint": float(
                        midpoint_distance
                    ),
                    "distance_from_rejected_state": (
                        float(
                            rejected_state_distance
                        )
                        if rejected_state_distance
                        is not None
                        else None
                    ),
                    "rejected_separation_score": float(
                        rejected_separation_score
                    ),
                    "quality_score": float(
                        descriptor_result.get(
                            "quality_score",
                            0.0,
                        )
                    ),
                    "geometry": {
                        "line_count": int(
                            geometry.get(
                                "line_count",
                                0,
                            )
                            or 0
                        ),
                        "line_family_count": int(
                            geometry.get(
                                "line_family_count",
                                0,
                            )
                            or 0
                        ),
                        "intersection_count": int(
                            geometry.get(
                                "intersection_count",
                                0,
                            )
                            or 0
                        ),
                        "green_fraction": float(
                            geometry.get(
                                "green_fraction",
                                0.0,
                            )
                            or 0.0
                        ),
                        "sharpness": float(
                            geometry.get(
                                "sharpness",
                                0.0,
                            )
                            or 0.0
                        ),
                    },
                    "_descriptor": descriptor,
                }
            )

        candidates.sort(
            key=lambda item: (
                -item["score"],
                -item["measurement_utility"],
                abs(
                    item["descriptor_alpha"]
                    - 0.5
                ),
                item["frame"],
            )
        )

        selected: list[dict[str, Any]] = []

        for candidate in candidates:
            if len(selected) >= cfg.maximum_recommendations:
                break

            if any(
                abs(
                    candidate["frame"]
                    - previous["frame"]
                )
                < cfg.minimum_temporal_separation_frames
                for previous in selected
            ):
                continue

            if any(
                field_view_distance(
                    candidate["_descriptor"],
                    previous["_descriptor"],
                )
                < cfg.minimum_descriptor_separation
                for previous in selected
            ):
                continue

            selected.append(
                candidate
            )

        public_selected = []

        for rank, candidate in enumerate(
            selected,
            start=1,
        ):
            public = {
                key: value
                for key, value in candidate.items()
                if not key.startswith("_")
            }
            public["rank"] = rank
            public_selected.append(
                public
            )

        all_gap_public = [
            {
                **item,
            }
            for item in gap_analysis["all"]
        ]

        return {
            "status": (
                "ok"
                if public_selected
                else "no_candidates"
            ),
            "video": resolved_video_path.name,
            "pitch_model": {
                "length_m": float(
                    pitch_model["length_m"]
                ),
                "width_m": float(
                    pitch_model["width_m"]
                ),
                "dimension_status": pitch_model.get(
                    "dimension_status"
                ),
            },
            "strong_anchor_frames": [
                int(
                    anchor["frame"]
                )
                for anchor in anchors
            ],
            "selected_gap": {
                **gap,
                "scanner_descriptor_distance": float(
                    gap_descriptor_distance
                ),
                "rejected_frames_inside": rejected_frames,
                "rejected_measurement_utility": (
                    float(
                        baseline_rejected_utility
                    )
                    if rejected_descriptors
                    else None
                ),
            },
            "all_temporal_gaps": all_gap_public,
            "recommendations": public_selected,
            "search": {
                "start_frame": search_start,
                "end_frame": search_end,
                "sample_step_frames": step,
                "evaluated_candidate_count": len(
                    candidates
                ),
            },
            "policy": {
                "uses_raw_video": True,
                "uses_people": False,
                "uses_propagation": False,
                "uses_refinement": False,
                "uses_legacy_camera_atlas": False,
                "uses_rejected_frame_as_bad_state": False,
                "rejected_frame_role": (
                    "negative observability evidence for immediate descriptor neighbourhood"
                ),
                "rejected_descriptor_guard_radius": cfg.rejected_descriptor_guard_radius,
                "scanner_identity": "geometry_first",
                "camera_gap_metric": "robust_local_observables",
            },
            "config": asdict(
                cfg
            ),
        }

    finally:
        capture.release()


def summarize_active_anchor_selection(
    result: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(
        result,
        dict,
    ):
        return {
            "status": "invalid_result",
        }

    gap = result.get(
        "selected_gap",
        {},
    )

    recommendations = result.get(
        "recommendations",
        [],
    )

    return {
        "status": result.get(
            "status"
        ),
        "strong_anchor_frames": result.get(
            "strong_anchor_frames",
            [],
        ),
        "gap": (
            [
                gap.get(
                    "left_frame"
                ),
                gap.get(
                    "right_frame"
                ),
            ]
            if isinstance(
                gap,
                dict,
            )
            else None
        ),
        "gap_state_distance": (
            gap.get(
                "robust_state_distance"
            )
            if isinstance(
                gap,
                dict,
            )
            else None
        ),
        "rejected_frames_inside": (
            gap.get(
                "rejected_frames_inside",
                [],
            )
            if isinstance(
                gap,
                dict,
            )
            else []
        ),
        "recommended_frames": [
            item.get(
                "frame"
            )
            for item in recommendations
            if isinstance(
                item,
                dict,
            )
        ],
    }


def _fmt(
    value: Any,
    digits: int = 3,
) -> str:
    try:
        number = float(
            value
        )
    except (TypeError, ValueError):
        return "—"

    if not math.isfinite(
        number
    ):
        return "—"

    return f"{number:.{digits}f}"


def _print_report(
    result: dict[str, Any],
) -> None:
    print()
    print(
        "=" * 78
    )
    print(
        "ACTIVE ANCHORING V2 · RELLENAR COBERTURA DE CÁMARA"
    )
    print(
        "=" * 78
    )

    print(
        "Status:",
        result.get(
            "status"
        ),
    )

    if result.get(
        "status"
    ) not in {
        "ok",
        "no_candidates",
    }:
        print(
            json.dumps(
                result,
                indent=2,
                ensure_ascii=False,
            )
        )
        return

    print(
        "Anchors HUMAN fuertes:",
        result.get(
            "strong_anchor_frames"
        ),
    )

    gap = result.get(
        "selected_gap",
        {},
    )

    print()
    print(
        "MAYOR HUECO DE COBERTURA"
    )
    print(
        f"  {gap.get('left_frame')} → {gap.get('right_frame')} · "
        f"distancia estado robusta={_fmt(gap.get('robust_state_distance'), 2)} · "
        f"Δt={gap.get('frame_gap')} frames"
    )
    print(
        "  Rechazados dentro:",
        gap.get(
            "rejected_frames_inside",
            [],
        ),
    )
    print(
        "  Distancia scanner entre extremos:",
        _fmt(
            gap.get(
                "scanner_descriptor_distance"
            ),
            4,
        ),
    )

    rejected_utility = gap.get(
        "rejected_measurement_utility"
    )
    if rejected_utility is not None:
        print(
            "  Utilidad geométrica del rechazado:",
            _fmt(
                rejected_utility,
                3,
            ),
        )

    print()
    print(
        "CANDIDATOS ACTIVOS"
    )

    recommendations = result.get(
        "recommendations",
        [],
    )

    if not recommendations:
        print(
            "  No se encontró ningún candidato suficientemente fuerte."
        )
    else:
        for item in recommendations:
            geometry = item.get(
                "geometry",
                {},
            )

            print(
                f"  #{item.get('rank')} · frame {item.get('frame')} · "
                f"score={_fmt(item.get('score'), 3)} · "
                f"target={item.get('target_mode')} · "
                f"alpha={_fmt(item.get('descriptor_alpha'), 2)}"
            )
            print(
                f"       utility={_fmt(item.get('measurement_utility'), 3)} · "
                f"q={_fmt(item.get('quality_score'), 3)} · "
                f"líneas={geometry.get('line_count')} · "
                f"familias={geometry.get('line_family_count')} · "
                f"intersecciones={geometry.get('intersection_count')}"
            )
            print(
                f"       d(left)={_fmt(item.get('distance_from_left'), 4)} · "
                f"d(right)={_fmt(item.get('distance_from_right'), 4)} · "
                f"d(mid)={_fmt(item.get('distance_from_midpoint'), 4)} · "
                f"d(rejected)={_fmt(item.get('distance_from_rejected_state'), 4)}"
            )
            print(
                f"       separación rechazo="
                f"{_fmt(item.get('rejected_separation_score'), 3)}"
            )

    print()
    print(
        "SIGUIENTE USO"
    )
    print(
        "  Inspecciona primero el candidato #1. "
        "No lo conviertas en anchor hasta confirmar visualmente "
        "que tiene geometría HUMAN suficiente."
    )
    print(
        "  Si #1 sigue siendo una vista subdeterminada, prueba #2 y #3."
    )

    print()
    print(
        "=" * 78
    )
    print(
        "No se ha modificado la review."
    )
    print(
        "=" * 78
    )
    print()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Busca nuevos anchors HUMAN que rellenen el mayor hueco "
            "actual de Camera State."
        )
    )
    parser.add_argument(
        "review",
        nargs="?",
        default=(
            "supervisor/data/reviews/"
            "benchmark_easy_2min.json"
        ),
    )
    parser.add_argument(
        "--video",
        default=None,
        help=(
            "Ruta al MP4. Si se omite usa "
            "input/<review.video.filename>."
        ),
    )
    parser.add_argument(
        "--step",
        type=int,
        default=2,
        help=(
            "Paso de búsqueda en frames dentro del hueco."
        ),
    )
    parser.add_argument(
        "--json",
        dest="json_output",
        default=None,
    )

    args = parser.parse_args()

    config = ActiveAnchoringConfig(
        sample_step_frames=max(
            1,
            int(
                args.step
            ),
        )
    )

    try:
        result = select_active_anchor_candidates(
            args.review,
            video_path=args.video,
            config=config,
        )
    except (
        RuntimeError,
        KeyError,
        TypeError,
        ValueError,
    ) as error:
        raise SystemExit(
            f"ERROR: {error}"
        ) from error

    _print_report(
        result
    )

    if args.json_output:
        output = Path(
            args.json_output
        )
        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        output.write_text(
            json.dumps(
                result,
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        print(
            "JSON guardado en:",
            output,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
