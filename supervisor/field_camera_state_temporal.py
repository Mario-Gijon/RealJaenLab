"""Temporal Camera State Observer V1 for RealJaen FIELD.

This module estimates position *inside an already learned HUMAN Camera State
trajectory* using raw-video dominant motion between temporal HUMAN anchors.

It intentionally does NOT reconstruct homographies.

Why this exists
---------------
The geometry-first FIELD scanner descriptor works well for:
- discovering representative views;
- active anchoring;
- coarse view-family similarity.

But it aliases some physically different cameras, so it is not precise enough
by itself to locate every raw frame in Camera State.

Temporal Observer V1 uses a different signal:

    HUMAN state at temporal boundary
        + raw KLT/RANSAC camera motion through the interval
        + HUMAN state at the other boundary
        -> state position inside the interval

The state-space itself is still the 3-D HUMAN model from field_camera_state.py.

Public API
----------
measure_temporal_motion_path(...)
estimate_bracketed_camera_state(...)
estimate_one_sided_camera_state(...)
temporal_camera_state_leave_one_out(...)
summarize_temporal_benchmark(...)

No review mutation, PEOPLE, propagation, refinement, legacy Atlas, or H
reconstruction lives here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np


EPS = 1e-10


@dataclass(frozen=True)
class TemporalCameraStateConfig:
    analysis_width: int = 640
    sample_step_frames: int = 2

    max_features: int = 450
    feature_quality_level: float = 0.01
    feature_min_distance: float = 7.0
    feature_block_size: int = 7

    lk_window: int = 21
    lk_max_level: int = 3

    ransac_reprojection_threshold_px: float = 2.5
    minimum_tracks: int = 12
    minimum_inlier_ratio: float = 0.25

    # Motion-feature scales for progress estimation.
    # Translation is already normalized by analysis image size.
    tx_scale_floor: float = 0.002
    ty_scale_floor: float = 0.002
    log_scale_floor: float = 0.0008
    rotation_floor_rad: float = math.radians(0.06)

    # Projection alpha is trusted only when the interval has meaningful net motion.
    minimum_interval_motion_norm: float = 0.35

    # If optical flow is weak, blend towards simple temporal alpha.
    good_valid_step_fraction: float = 0.85
    weak_valid_step_fraction: float = 0.55

    # Clamp only small numerical overshoot for bracketed estimates.
    bracket_alpha_min: float = -0.08
    bracket_alpha_max: float = 1.08


def _resize_gray(frame: np.ndarray, width: int) -> np.ndarray:
    height, original_width = frame.shape[:2]

    if original_width > width:
        scale = width / float(original_width)
        frame = cv2.resize(
            frame,
            (
                width,
                max(1, round(height * scale)),
            ),
            interpolation=cv2.INTER_AREA,
        )

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY,
    )

    return cv2.equalizeHist(gray)


def _estimate_affine_motion(
    previous_gray: np.ndarray,
    current_gray: np.ndarray,
    cfg: TemporalCameraStateConfig,
) -> dict[str, Any]:
    points = cv2.goodFeaturesToTrack(
        previous_gray,
        maxCorners=cfg.max_features,
        qualityLevel=cfg.feature_quality_level,
        minDistance=cfg.feature_min_distance,
        blockSize=cfg.feature_block_size,
    )

    if points is None or len(points) < cfg.minimum_tracks:
        return {
            "status": "insufficient_features",
            "track_count": 0 if points is None else int(len(points)),
        }

    next_points, status, _ = cv2.calcOpticalFlowPyrLK(
        previous_gray,
        current_gray,
        points,
        None,
        winSize=(cfg.lk_window, cfg.lk_window),
        maxLevel=cfg.lk_max_level,
    )

    if next_points is None or status is None:
        return {
            "status": "flow_failed",
            "track_count": int(len(points)),
        }

    good = status.ravel() > 0
    source = points.reshape(-1, 2)[good]
    target = next_points.reshape(-1, 2)[good]

    if len(source) < cfg.minimum_tracks:
        return {
            "status": "insufficient_tracks",
            "track_count": int(len(source)),
        }

    matrix, inliers = cv2.estimateAffinePartial2D(
        source,
        target,
        method=cv2.RANSAC,
        ransacReprojThreshold=cfg.ransac_reprojection_threshold_px,
    )

    if matrix is None:
        return {
            "status": "ransac_failed",
            "track_count": int(len(source)),
        }

    inlier_ratio = (
        float(np.mean(inliers.ravel() > 0))
        if inliers is not None
        else 0.0
    )

    a = float(matrix[0, 0])
    b = float(matrix[0, 1])
    c = float(matrix[1, 0])
    d = float(matrix[1, 1])

    scale_x = math.hypot(a, c)
    scale_y = math.hypot(b, d)
    scale = math.sqrt(max(scale_x * scale_y, EPS))
    rotation = math.atan2(c, a)

    homogeneous = np.asarray(
        [
            [a, b, float(matrix[0, 2])],
            [c, d, float(matrix[1, 2])],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    if not np.isfinite(homogeneous).all():
        return {
            "status": "invalid_affine",
            "track_count": int(len(source)),
        }

    return {
        "status": (
            "ok"
            if inlier_ratio >= cfg.minimum_inlier_ratio
            else "low_inlier_ratio"
        ),
        "matrix": homogeneous,
        "track_count": int(len(source)),
        "inlier_ratio": inlier_ratio,
        "scale": float(scale),
        "rotation_rad": float(rotation),
    }


def _frame_sequence(
    start_frame: int,
    end_frame: int,
    step: int,
) -> list[int]:
    start = int(start_frame)
    end = int(end_frame)

    if start == end:
        return [start]

    direction = 1 if end > start else -1
    signed_step = max(1, int(step)) * direction

    frames = list(
        range(
            start,
            end,
            signed_step,
        )
    )

    if not frames or frames[0] != start:
        frames.insert(0, start)

    if frames[-1] != end:
        frames.append(end)

    return frames


def _read_gray(
    capture: cv2.VideoCapture,
    frame_index: int,
    cfg: TemporalCameraStateConfig,
) -> np.ndarray | None:
    capture.set(
        cv2.CAP_PROP_POS_FRAMES,
        int(frame_index),
    )

    ok, frame = capture.read()
    if not ok:
        return None

    return _resize_gray(
        frame,
        cfg.analysis_width,
    )


def _compose_motion(
    cumulative: np.ndarray,
    step_matrix: np.ndarray,
) -> np.ndarray:
    # step maps previous-image coordinates -> current-image coordinates.
    # Therefore cumulative(start->current) = step @ cumulative(start->previous).
    return step_matrix @ cumulative


def _motion_feature_from_transform(
    transform: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    a = float(transform[0, 0])
    c = float(transform[1, 0])

    scale = math.sqrt(
        max(
            a * a + c * c,
            EPS,
        )
    )

    rotation = math.atan2(
        c,
        a,
    )

    return np.asarray(
        [
            float(transform[0, 2]) / max(float(width), 1.0),
            float(transform[1, 2]) / max(float(height), 1.0),
            math.log(max(scale, EPS)),
            rotation,
        ],
        dtype=np.float64,
    )


def measure_temporal_motion_path(
    video_path: str | Path,
    start_frame: int,
    end_frame: int,
    *,
    config: TemporalCameraStateConfig | None = None,
) -> dict[str, Any]:
    """Measure cumulative dominant raw-video motion from start to end."""

    cfg = config or TemporalCameraStateConfig()
    path = Path(video_path)

    if not path.exists():
        return {
            "status": "video_not_found",
            "video": str(path),
        }

    capture = cv2.VideoCapture(
        str(path)
    )

    if not capture.isOpened():
        return {
            "status": "video_open_failed",
            "video": str(path),
        }

    try:
        frames = _frame_sequence(
            int(start_frame),
            int(end_frame),
            cfg.sample_step_frames,
        )

        first_gray = _read_gray(
            capture,
            frames[0],
            cfg,
        )

        if first_gray is None:
            return {
                "status": "frame_read_failed",
                "frame": frames[0],
            }

        image_height, image_width = first_gray.shape
        cumulative = np.eye(
            3,
            dtype=np.float64,
        )

        snapshots = [
            {
                "frame": int(frames[0]),
                "transform": cumulative.copy(),
                "motion_feature": _motion_feature_from_transform(
                    cumulative,
                    image_width,
                    image_height,
                ),
            }
        ]

        valid_steps = 0
        weak_steps = 0
        failed_steps = 0
        inlier_ratios = []
        track_counts = []

        previous_gray = first_gray
        previous_frame = frames[0]

        for frame_index in frames[1:]:
            current_gray = _read_gray(
                capture,
                frame_index,
                cfg,
            )

            if current_gray is None:
                failed_steps += 1
                previous_frame = frame_index
                continue

            motion = _estimate_affine_motion(
                previous_gray,
                current_gray,
                cfg,
            )

            status = motion.get("status")

            if status in {"ok", "low_inlier_ratio"}:
                step_matrix = np.asarray(
                    motion["matrix"],
                    dtype=np.float64,
                )
                cumulative = _compose_motion(
                    cumulative,
                    step_matrix,
                )

                if status == "ok":
                    valid_steps += 1
                else:
                    weak_steps += 1

                inlier_ratios.append(
                    float(
                        motion.get(
                            "inlier_ratio",
                            0.0,
                        )
                    )
                )
                track_counts.append(
                    int(
                        motion.get(
                            "track_count",
                            0,
                        )
                    )
                )
            else:
                failed_steps += 1

            snapshots.append(
                {
                    "frame": int(frame_index),
                    "transform": cumulative.copy(),
                    "motion_feature": _motion_feature_from_transform(
                        cumulative,
                        image_width,
                        image_height,
                    ),
                    "step_status": status,
                    "from_frame": int(previous_frame),
                }
            )

            previous_gray = current_gray
            previous_frame = frame_index

        attempted_steps = max(
            len(frames) - 1,
            0,
        )
        successful_steps = (
            valid_steps
            + weak_steps
        )

        successful_fraction = (
            successful_steps
            / attempted_steps
            if attempted_steps > 0
            else 1.0
        )

        valid_fraction = (
            valid_steps
            / attempted_steps
            if attempted_steps > 0
            else 1.0
        )

        return {
            "status": (
                "ok"
                if successful_fraction
                >= cfg.weak_valid_step_fraction
                else "weak_motion_path"
            ),
            "video": path.name,
            "start_frame": int(start_frame),
            "end_frame": int(end_frame),
            "sample_step_frames": cfg.sample_step_frames,
            "analysis_width": image_width,
            "analysis_height": image_height,
            "attempted_steps": attempted_steps,
            "valid_steps": valid_steps,
            "weak_steps": weak_steps,
            "failed_steps": failed_steps,
            "successful_step_fraction": float(successful_fraction),
            "valid_step_fraction": float(valid_fraction),
            "median_inlier_ratio": (
                float(
                    np.median(
                        inlier_ratios
                    )
                )
                if inlier_ratios
                else None
            ),
            "median_track_count": (
                float(
                    np.median(
                        track_counts
                    )
                )
                if track_counts
                else None
            ),
            "snapshots": snapshots,
            "final_transform": cumulative,
            "final_motion_feature": _motion_feature_from_transform(
                cumulative,
                image_width,
                image_height,
            ),
            "config": asdict(cfg),
        }

    finally:
        capture.release()


def _robust_motion_scales(
    full_feature: np.ndarray,
    cfg: TemporalCameraStateConfig,
) -> np.ndarray:
    return np.asarray(
        [
            max(
                abs(float(full_feature[0])),
                cfg.tx_scale_floor,
            ),
            max(
                abs(float(full_feature[1])),
                cfg.ty_scale_floor,
            ),
            max(
                abs(float(full_feature[2])),
                cfg.log_scale_floor,
            ),
            max(
                abs(float(full_feature[3])),
                cfg.rotation_floor_rad,
            ),
        ],
        dtype=np.float64,
    )


def _motion_progress_alpha(
    query_feature: np.ndarray,
    full_feature: np.ndarray,
    cfg: TemporalCameraStateConfig,
) -> dict[str, Any]:
    scales = _robust_motion_scales(
        full_feature,
        cfg,
    )

    q = np.asarray(
        query_feature,
        dtype=np.float64,
    ) / scales
    full = np.asarray(
        full_feature,
        dtype=np.float64,
    ) / scales

    denominator = float(
        np.dot(
            full,
            full,
        )
    )

    if denominator < EPS:
        return {
            "status": "insufficient_net_motion",
            "alpha": None,
            "motion_norm": 0.0,
        }

    alpha = float(
        np.dot(
            q,
            full,
        )
        / denominator
    )

    residual = q - alpha * full
    residual_norm = float(
        np.linalg.norm(
            residual
        )
        / math.sqrt(
            len(
                residual
            )
        )
    )

    motion_norm = float(
        math.sqrt(
            denominator
            / len(
                full
            )
        )
    )

    return {
        "status": (
            "ok"
            if motion_norm
            >= cfg.minimum_interval_motion_norm
            else "weak_net_motion"
        ),
        "alpha": alpha,
        "motion_norm": motion_norm,
        "residual_norm": residual_norm,
        "scales": scales.tolist(),
    }


def _blend_motion_and_time_alpha(
    motion_alpha: float | None,
    time_alpha: float,
    *,
    valid_step_fraction: float,
    motion_status: str,
    cfg: TemporalCameraStateConfig,
) -> tuple[float, float]:
    if motion_alpha is None:
        return float(time_alpha), 0.0

    if (
        motion_status == "ok"
        and valid_step_fraction
        >= cfg.good_valid_step_fraction
    ):
        motion_weight = 0.85
    elif (
        valid_step_fraction
        >= cfg.weak_valid_step_fraction
    ):
        motion_weight = 0.60
    else:
        motion_weight = 0.25

    alpha = (
        motion_weight
        * float(motion_alpha)
        + (
            1.0
            - motion_weight
        )
        * float(time_alpha)
    )

    return (
        float(alpha),
        float(motion_weight),
    )


def _extract_state_map(
    camera_state_model: dict[str, Any],
) -> dict[int, np.ndarray]:
    return {
        int(anchor["frame"]): np.asarray(
            anchor["state"],
            dtype=np.float64,
        )
        for anchor in camera_state_model.get(
            "anchors",
            [],
        )
        if isinstance(
            anchor,
            dict,
        )
        and anchor.get(
            "state"
        )
        is not None
    }


def estimate_bracketed_camera_state(
    camera_state_model: dict[str, Any],
    video_path: str | Path,
    *,
    left_frame: int,
    query_frame: int,
    right_frame: int,
    config: TemporalCameraStateConfig | None = None,
) -> dict[str, Any]:
    """Estimate query state between two known HUMAN temporal boundaries."""

    cfg = config or TemporalCameraStateConfig()

    left = int(left_frame)
    query = int(query_frame)
    right = int(right_frame)

    if not (
        left
        < query
        < right
    ):
        return {
            "status": "invalid_bracket",
            "left_frame": left,
            "query_frame": query,
            "right_frame": right,
        }

    state_map = _extract_state_map(
        camera_state_model
    )

    if (
        left not in state_map
        or right not in state_map
    ):
        return {
            "status": "missing_boundary_state",
        }

    full_path = measure_temporal_motion_path(
        video_path,
        left,
        right,
        config=cfg,
    )
    query_path = measure_temporal_motion_path(
        video_path,
        left,
        query,
        config=cfg,
    )

    if (
        full_path.get("status")
        not in {
            "ok",
            "weak_motion_path",
        }
        or query_path.get("status")
        not in {
            "ok",
            "weak_motion_path",
        }
    ):
        return {
            "status": "motion_measurement_failed",
            "full_path_status": full_path.get(
                "status"
            ),
            "query_path_status": query_path.get(
                "status"
            ),
        }

    progress = _motion_progress_alpha(
        np.asarray(
            query_path[
                "final_motion_feature"
            ],
            dtype=np.float64,
        ),
        np.asarray(
            full_path[
                "final_motion_feature"
            ],
            dtype=np.float64,
        ),
        cfg,
    )

    alpha_time = (
        query - left
    ) / float(
        right - left
    )

    combined_valid_fraction = min(
        float(
            full_path.get(
                "valid_step_fraction",
                0.0,
            )
        ),
        float(
            query_path.get(
                "valid_step_fraction",
                0.0,
            )
        ),
    )

    alpha, motion_weight = (
        _blend_motion_and_time_alpha(
            progress.get(
                "alpha"
            ),
            alpha_time,
            valid_step_fraction=combined_valid_fraction,
            motion_status=str(
                progress.get(
                    "status"
                )
            ),
            cfg=cfg,
        )
    )

    alpha = float(
        np.clip(
            alpha,
            cfg.bracket_alpha_min,
            cfg.bracket_alpha_max,
        )
    )

    left_state = state_map[
        left
    ]
    right_state = state_map[
        right
    ]

    estimated_state = (
        (
            1.0
            - alpha
        )
        * left_state
        + alpha
        * right_state
    )

    time_state = (
        (
            1.0
            - alpha_time
        )
        * left_state
        + alpha_time
        * right_state
    )

    confidence = (
        "good"
        if (
            combined_valid_fraction
            >= cfg.good_valid_step_fraction
            and progress.get(
                "status"
            )
            == "ok"
            and progress.get(
                "residual_norm",
                999.0,
            )
            <= 0.80
        )
        else (
            "weak"
            if combined_valid_fraction
            >= cfg.weak_valid_step_fraction
            else "poor"
        )
    )

    return {
        "status": "ok",
        "method": "bracketed_temporal_motion",
        "left_frame": left,
        "query_frame": query,
        "right_frame": right,
        "state": estimated_state.tolist(),
        "time_only_state": time_state.tolist(),
        "alpha_motion_raw": progress.get(
            "alpha"
        ),
        "alpha_time": float(
            alpha_time
        ),
        "alpha_used": alpha,
        "motion_weight": motion_weight,
        "motion_progress_status": progress.get(
            "status"
        ),
        "motion_feature_residual": progress.get(
            "residual_norm"
        ),
        "motion_norm": progress.get(
            "motion_norm"
        ),
        "valid_step_fraction": combined_valid_fraction,
        "confidence": confidence,
        "full_path": {
            key: value
            for key, value in full_path.items()
            if key
            not in {
                "snapshots",
                "final_transform",
                "final_motion_feature",
                "config",
            }
        },
        "query_path": {
            key: value
            for key, value in query_path.items()
            if key
            not in {
                "snapshots",
                "final_transform",
                "final_motion_feature",
                "config",
            }
        },
        "policy": {
            "returns_homography": False,
            "uses_raw_temporal_motion": True,
            "uses_static_view_descriptor": False,
            "uses_time_only_as_fallback": True,
        },
    }


def estimate_one_sided_camera_state(
    camera_state_model: dict[str, Any],
    video_path: str | Path,
    *,
    query_frame: int,
    first_known_frame: int,
    second_known_frame: int,
    side: str,
    config: TemporalCameraStateConfig | None = None,
) -> dict[str, Any]:
    """One-sided endpoint extrapolation for LOO diagnostics.

    side="left":
        query < first_known < second_known

    side="right":
        first_known < second_known < query

    This exists so endpoint HUMAN anchors can participate in LOO. It should not
    be preferred for production inference when a temporal bracket exists.
    """

    cfg = config or TemporalCameraStateConfig()
    query = int(query_frame)
    first = int(first_known_frame)
    second = int(second_known_frame)

    if side not in {"left", "right"}:
        return {
            "status": "invalid_side",
        }

    if (
        side == "left"
        and not (
            query
            < first
            < second
        )
    ):
        return {
            "status": "invalid_order",
        }

    if (
        side == "right"
        and not (
            first
            < second
            < query
        )
    ):
        return {
            "status": "invalid_order",
        }

    state_map = _extract_state_map(
        camera_state_model
    )

    if (
        first not in state_map
        or second not in state_map
    ):
        return {
            "status": "missing_boundary_state",
        }

    if side == "left":
        # Measure query -> first and query -> second, then infer where first lies
        # on the query-to-second motion path.
        full_path = measure_temporal_motion_path(
            video_path,
            query,
            second,
            config=cfg,
        )
        first_path = measure_temporal_motion_path(
            video_path,
            query,
            first,
            config=cfg,
        )

        progress = _motion_progress_alpha(
            np.asarray(
                first_path[
                    "final_motion_feature"
                ]
            ),
            np.asarray(
                full_path[
                    "final_motion_feature"
                ]
            ),
            cfg,
        )

        alpha_time = (
            first - query
        ) / float(
            second - query
        )

        valid_fraction = min(
            float(
                full_path.get(
                    "valid_step_fraction",
                    0.0,
                )
            ),
            float(
                first_path.get(
                    "valid_step_fraction",
                    0.0,
                )
            ),
        )

        alpha_first, motion_weight = (
            _blend_motion_and_time_alpha(
                progress.get(
                    "alpha"
                ),
                alpha_time,
                valid_step_fraction=valid_fraction,
                motion_status=str(
                    progress.get(
                        "status"
                    )
                ),
                cfg=cfg,
            )
        )

        if abs(
            1.0
            - alpha_first
        ) < 0.08:
            return {
                "status": "unstable_extrapolation",
            }

        first_state = state_map[
            first
        ]
        second_state = state_map[
            second
        ]

        # first = (1-a)*query + a*second
        estimated_query = (
            first_state
            - alpha_first
            * second_state
        ) / (
            1.0
            - alpha_first
        )

        alpha_time_first = alpha_time
        time_query = (
            first_state
            - alpha_time_first
            * second_state
        ) / (
            1.0
            - alpha_time_first
        )

    else:
        # Measure first -> query and first -> second, then infer query beyond
        # second along the same local state trajectory.
        full_path = measure_temporal_motion_path(
            video_path,
            first,
            query,
            config=cfg,
        )
        second_path = measure_temporal_motion_path(
            video_path,
            first,
            second,
            config=cfg,
        )

        progress = _motion_progress_alpha(
            np.asarray(
                second_path[
                    "final_motion_feature"
                ]
            ),
            np.asarray(
                full_path[
                    "final_motion_feature"
                ]
            ),
            cfg,
        )

        alpha_time = (
            second - first
        ) / float(
            query - first
        )

        valid_fraction = min(
            float(
                full_path.get(
                    "valid_step_fraction",
                    0.0,
                )
            ),
            float(
                second_path.get(
                    "valid_step_fraction",
                    0.0,
                )
            ),
        )

        alpha_second, motion_weight = (
            _blend_motion_and_time_alpha(
                progress.get(
                    "alpha"
                ),
                alpha_time,
                valid_step_fraction=valid_fraction,
                motion_status=str(
                    progress.get(
                        "status"
                    )
                ),
                cfg=cfg,
            )
        )

        if abs(
            alpha_second
        ) < 0.08:
            return {
                "status": "unstable_extrapolation",
            }

        first_state = state_map[
            first
        ]
        second_state = state_map[
            second
        ]

        # second = (1-a)*first + a*query
        estimated_query = (
            second_state
            - (
                1.0
                - alpha_second
            )
            * first_state
        ) / alpha_second

        alpha_time_second = alpha_time
        time_query = (
            second_state
            - (
                1.0
                - alpha_time_second
            )
            * first_state
        ) / alpha_time_second

    confidence = (
        "weak"
        if valid_fraction
        >= cfg.weak_valid_step_fraction
        else "poor"
    )

    return {
        "status": "ok",
        "method": (
            f"one_sided_{side}_temporal_motion"
        ),
        "query_frame": query,
        "first_known_frame": first,
        "second_known_frame": second,
        "state": estimated_query.tolist(),
        "time_only_state": time_query.tolist(),
        "motion_weight": motion_weight,
        "valid_step_fraction": valid_fraction,
        "motion_progress_status": progress.get(
            "status"
        ),
        "motion_feature_residual": progress.get(
            "residual_norm"
        ),
        "confidence": confidence,
        "policy": {
            "endpoint_extrapolation": True,
            "prefer_temporal_bracket_when_available": True,
            "returns_homography": False,
        },
    }


def _state_rmse(
    predicted: np.ndarray,
    target: np.ndarray,
) -> float:
    delta = (
        np.asarray(
            predicted,
            dtype=np.float64,
        )
        - np.asarray(
            target,
            dtype=np.float64,
        )
    )

    return float(
        np.linalg.norm(
            delta
        )
        / math.sqrt(
            len(
                delta
            )
        )
    )


def temporal_camera_state_leave_one_out(
    camera_state_model: dict[str, Any],
    video_path: str | Path,
    *,
    config: TemporalCameraStateConfig | None = None,
) -> dict[str, Any]:
    """LOO benchmark using temporal HUMAN boundaries + raw video motion."""

    cfg = config or TemporalCameraStateConfig()

    if camera_state_model.get(
        "status"
    ) != "ok":
        return {
            "status": "invalid_camera_state_model",
        }

    frames = sorted(
        int(
            frame
        )
        for frame in camera_state_model.get(
            "anchor_frames",
            [],
        )
    )

    state_map = _extract_state_map(
        camera_state_model
    )

    if len(frames) < 5:
        return {
            "status": "not_enough_anchors",
            "anchor_count": len(
                frames
            ),
        }

    results = []

    for index, frame in enumerate(
        frames
    ):
        true_state = state_map[
            frame
        ]

        if (
            index > 0
            and index < len(
                frames
            )
            - 1
        ):
            estimate = (
                estimate_bracketed_camera_state(
                    camera_state_model,
                    video_path,
                    left_frame=frames[
                        index - 1
                    ],
                    query_frame=frame,
                    right_frame=frames[
                        index + 1
                    ],
                    config=cfg,
                )
            )
            benchmark_group = "bracketed"

        elif index == 0:
            estimate = (
                estimate_one_sided_camera_state(
                    camera_state_model,
                    video_path,
                    query_frame=frame,
                    first_known_frame=frames[
                        1
                    ],
                    second_known_frame=frames[
                        2
                    ],
                    side="left",
                    config=cfg,
                )
            )
            benchmark_group = "endpoint"

        else:
            estimate = (
                estimate_one_sided_camera_state(
                    camera_state_model,
                    video_path,
                    query_frame=frame,
                    first_known_frame=frames[
                        -3
                    ],
                    second_known_frame=frames[
                        -2
                    ],
                    side="right",
                    config=cfg,
                )
            )
            benchmark_group = "endpoint"

        if estimate.get(
            "status"
        ) != "ok":
            results.append(
                {
                    "frame": frame,
                    "status": estimate.get(
                        "status"
                    ),
                    "benchmark_group": benchmark_group,
                }
            )
            continue

        predicted = np.asarray(
            estimate[
                "state"
            ],
            dtype=np.float64,
        )

        time_predicted = np.asarray(
            estimate[
                "time_only_state"
            ],
            dtype=np.float64,
        )

        motion_error = _state_rmse(
            predicted,
            true_state,
        )
        time_error = _state_rmse(
            time_predicted,
            true_state,
        )

        results.append(
            {
                "frame": frame,
                "status": "ok",
                "benchmark_group": benchmark_group,
                "method": estimate.get(
                    "method"
                ),
                "motion_state_rmse": motion_error,
                "time_only_state_rmse": time_error,
                "improvement_vs_time": float(
                    time_error
                    - motion_error
                ),
                "confidence": estimate.get(
                    "confidence"
                ),
                "valid_step_fraction": estimate.get(
                    "valid_step_fraction"
                ),
                "motion_progress_status": estimate.get(
                    "motion_progress_status"
                ),
                "motion_feature_residual": estimate.get(
                    "motion_feature_residual"
                ),
                "alpha_motion_raw": estimate.get(
                    "alpha_motion_raw"
                ),
                "alpha_time": estimate.get(
                    "alpha_time"
                ),
                "alpha_used": estimate.get(
                    "alpha_used"
                ),
                "motion_weight": estimate.get(
                    "motion_weight"
                ),
            }
        )

    valid = [
        item
        for item in results
        if item.get(
            "status"
        )
        == "ok"
    ]

    bracketed = [
        item
        for item in valid
        if item.get(
            "benchmark_group"
        )
        == "bracketed"
    ]

    if not valid:
        return {
            "status": "no_valid_holdouts",
            "per_anchor": results,
        }

    def summarize_group(
        group: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        if not group:
            return None

        motion_errors = np.asarray(
            [
                item[
                    "motion_state_rmse"
                ]
                for item in group
            ],
            dtype=np.float64,
        )

        time_errors = np.asarray(
            [
                item[
                    "time_only_state_rmse"
                ]
                for item in group
            ],
            dtype=np.float64,
        )

        return {
            "count": len(
                group
            ),
            "motion_median_rmse": float(
                np.median(
                    motion_errors
                )
            ),
            "motion_p90_rmse": float(
                np.percentile(
                    motion_errors,
                    90,
                )
            ),
            "motion_max_rmse": float(
                np.max(
                    motion_errors
                )
            ),
            "time_median_rmse": float(
                np.median(
                    time_errors
                )
            ),
            "time_p90_rmse": float(
                np.percentile(
                    time_errors,
                    90,
                )
            ),
            "time_max_rmse": float(
                np.max(
                    time_errors
                )
            ),
            "median_improvement_vs_time": float(
                np.median(
                    time_errors
                    - motion_errors
                )
            ),
        }

    return {
        "status": "ok",
        "all": summarize_group(
            valid
        ),
        "bracketed": summarize_group(
            bracketed
        ),
        "per_anchor": results,
        "config": asdict(
            cfg
        ),
    }


def summarize_temporal_benchmark(
    result: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(
        result,
        dict,
    ):
        return {
            "status": "invalid_result",
        }

    return {
        "status": result.get(
            "status"
        ),
        "all": result.get(
            "all"
        ),
        "bracketed": result.get(
            "bracketed"
        ),
        "valid_holdouts": sum(
            1
            for item in result.get(
                "per_anchor",
                [],
            )
            if item.get(
                "status"
            )
            == "ok"
        ),
    }
