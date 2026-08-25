"""Camera State -> Homography Prior V1 for RealJaen FIELD.

This module decodes an already estimated 3-D Camera State into a plausible
homography prior.

It NEVER interpolates raw homography coefficients.

Instead, each HUMAN anchor H is inverted at a fixed grid of image-space probe
points located safely below the ground-plane horizon. For a query Camera State:

    query state
        -> nearest HUMAN states
        -> positive local weights
        -> interpolate image-probe world coordinates
        -> fit image->world homography with normalized DLT
        -> invert
        -> H prior

This keeps the interpolation tied to what each camera means geometrically in
the visible image and avoids the whole-pitch projective-pole explosion that
invalidated earlier camera-distance diagnostics.

V1 is still a PRIOR only. A low-confidence prior must never become an automatic
final field calibration without refinement/validation.

Public API
----------
estimate_homography_prior_from_state(...)
evaluate_prior_against_homography(...)
evaluate_prior_against_manual_points(...)
homography_prior_leave_one_out(...)
summarize_homography_prior_benchmark(...)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import numpy as np


EPS = 1e-10


@dataclass(frozen=True)
class CameraStateHPriorConfig:
    neighbor_count: int = 3
    bandwidth_multiplier: float = 1.25
    bandwidth_floor: float = 0.30

    good_support_ratio: float = 1.10
    weak_support_ratio: float = 1.80

    # Image probes are intentionally below the typical pitch horizon.
    probe_x_normalized: tuple[float, ...] = (0.10, 0.30, 0.50, 0.70, 0.90)
    probe_y_normalized: tuple[float, ...] = (0.38, 0.55, 0.72, 0.88)

    # Visible-world evaluation grid for LOO diagnostics.
    evaluation_x_samples: int = 15
    evaluation_y_samples: int = 11
    evaluation_image_margin_px: float = 12.0


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
        or condition > 1e14
    ):
        return None

    return h


def _project_points(
    points_xy: np.ndarray,
    h: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(
        points_xy,
        dtype=np.float64,
    ).reshape(-1, 2)

    homogeneous = np.column_stack(
        (
            points,
            np.ones(
                len(points),
                dtype=np.float64,
            ),
        )
    )

    q = homogeneous @ h.T
    denominator = q[:, 2]

    valid = (
        np.isfinite(q).all(axis=1)
        & np.isfinite(denominator)
        & (np.abs(denominator) > EPS)
    )

    projected = np.full(
        (len(points), 2),
        np.nan,
        dtype=np.float64,
    )

    projected[valid] = (
        q[valid, :2]
        / denominator[valid, None]
    )

    valid &= np.isfinite(projected).all(axis=1)

    return projected, valid


def _inverse_map_points(
    image_points: np.ndarray,
    h: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        inverse = np.linalg.inv(h)
    except np.linalg.LinAlgError:
        return (
            np.full(
                (len(image_points), 2),
                np.nan,
                dtype=np.float64,
            ),
            np.zeros(
                len(image_points),
                dtype=bool,
            ),
        )

    return _project_points(
        image_points,
        inverse,
    )


def _normalise_points_for_dlt(
    points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    points = np.asarray(
        points,
        dtype=np.float64,
    ).reshape(-1, 2)

    if (
        len(points) < 4
        or not np.isfinite(points).all()
    ):
        return None

    centroid = np.mean(
        points,
        axis=0,
    )

    shifted = points - centroid
    mean_distance = float(
        np.mean(
            np.linalg.norm(
                shifted,
                axis=1,
            )
        )
    )

    if mean_distance < EPS:
        return None

    scale = math.sqrt(2.0) / mean_distance

    transform = np.asarray(
        [
            [
                scale,
                0.0,
                -scale * centroid[0],
            ],
            [
                0.0,
                scale,
                -scale * centroid[1],
            ],
            [
                0.0,
                0.0,
                1.0,
            ],
        ],
        dtype=np.float64,
    )

    homogeneous = np.column_stack(
        (
            points,
            np.ones(
                len(points),
                dtype=np.float64,
            ),
        )
    )

    transformed = homogeneous @ transform.T

    return (
        transformed[:, :2],
        transform,
    )


def _solve_homography_dlt(
    source_points: np.ndarray,
    target_points: np.ndarray,
) -> np.ndarray | None:
    source_result = _normalise_points_for_dlt(
        source_points
    )
    target_result = _normalise_points_for_dlt(
        target_points
    )

    if (
        source_result is None
        or target_result is None
    ):
        return None

    source_norm, source_transform = (
        source_result
    )
    target_norm, target_transform = (
        target_result
    )

    rows = []

    for (
        (x, y),
        (u, v),
    ) in zip(
        source_norm,
        target_norm,
    ):
        rows.append(
            [
                -x,
                -y,
                -1.0,
                0.0,
                0.0,
                0.0,
                u * x,
                u * y,
                u,
            ]
        )
        rows.append(
            [
                0.0,
                0.0,
                0.0,
                -x,
                -y,
                -1.0,
                v * x,
                v * y,
                v,
            ]
        )

    try:
        _, _, vt = np.linalg.svd(
            np.asarray(
                rows,
                dtype=np.float64,
            )
        )

        normalized_h = vt[
            -1
        ].reshape(
            3,
            3,
        )

        h = (
            np.linalg.inv(
                target_transform
            )
            @ normalized_h
            @ source_transform
        )

    except np.linalg.LinAlgError:
        return None

    return _as_h(
        h
    )


def _image_probe_points(
    image_width: int,
    image_height: int,
    cfg: CameraStateHPriorConfig,
) -> np.ndarray:
    points = []

    for y_norm in cfg.probe_y_normalized:
        for x_norm in cfg.probe_x_normalized:
            points.append(
                (
                    float(x_norm)
                    * float(image_width),
                    float(y_norm)
                    * float(image_height),
                )
            )

    return np.asarray(
        points,
        dtype=np.float64,
    )


def _state_distance(
    first: np.ndarray,
    second: np.ndarray,
) -> float:
    delta = (
        np.asarray(
            first,
            dtype=np.float64,
        )
        - np.asarray(
            second,
            dtype=np.float64,
        )
    )

    return float(
        np.linalg.norm(
            delta
        )
        / math.sqrt(
            max(
                len(delta),
                1,
            )
        )
    )


def _collect_decoder_anchors(
    camera_state_model: dict[str, Any],
    calibrations: list[dict[str, Any]],
    *,
    excluded_frames: set[int] | None = None,
) -> list[dict[str, Any]]:
    excluded = {
        int(frame)
        for frame in (
            excluded_frames
            or set()
        )
    }

    state_map = {
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

    anchors = []

    for calibration in calibrations:
        if not isinstance(
            calibration,
            dict,
        ):
            continue

        try:
            frame = int(
                calibration.get(
                    "frame"
                )
            )
        except (
            TypeError,
            ValueError,
        ):
            continue

        if frame in excluded:
            continue

        if frame not in state_map:
            continue

        if calibration.get(
            "source"
        ) != "manual":
            continue

        if calibration.get(
            "purpose"
        ) != "match_setup":
            continue

        h = _as_h(
            calibration.get(
                "homography"
            )
        )
        if h is None:
            continue

        anchors.append(
            {
                "frame": frame,
                "state": state_map[
                    frame
                ],
                "homography": h,
            }
        )

    anchors.sort(
        key=lambda item: item[
            "frame"
        ]
    )

    return anchors


def estimate_homography_prior_from_state(
    camera_state_model: dict[str, Any],
    calibrations: list[dict[str, Any]],
    query_state: Any,
    *,
    excluded_frames: set[int] | None = None,
    config: CameraStateHPriorConfig | None = None,
) -> dict[str, Any]:
    """Decode one Camera State into a projective H prior."""

    cfg = (
        config
        or CameraStateHPriorConfig()
    )

    if camera_state_model.get(
        "status"
    ) != "ok":
        return {
            "status": "invalid_camera_state_model",
        }

    try:
        state = np.asarray(
            query_state,
            dtype=np.float64,
        ).reshape(-1)
    except (
        TypeError,
        ValueError,
    ):
        return {
            "status": "invalid_query_state",
        }

    dimensions = int(
        camera_state_model.get(
            "dimensions",
            0,
        )
        or 0
    )

    if (
        dimensions <= 0
        or state.shape
        != (
            dimensions,
        )
        or not np.isfinite(
            state
        ).all()
    ):
        return {
            "status": "invalid_query_state",
        }

    anchors = _collect_decoder_anchors(
        camera_state_model,
        calibrations,
        excluded_frames=excluded_frames,
    )

    if len(
        anchors
    ) < 2:
        return {
            "status": "not_enough_decoder_anchors",
            "anchor_count": len(
                anchors
            ),
        }

    distances = np.asarray(
        [
            _state_distance(
                state,
                anchor[
                    "state"
                ],
            )
            for anchor in anchors
        ],
        dtype=np.float64,
    )

    order = np.argsort(
        distances
    )

    neighbor_count = min(
        max(
            2,
            int(
                cfg.neighbor_count
            ),
        ),
        len(
            anchors
        ),
    )

    selected_indices = order[
        :neighbor_count
    ]

    coverage_scale = float(
        camera_state_model.get(
            "state_coverage_scale",
            0.7,
        )
        or 0.7
    )

    bandwidth = max(
        cfg.bandwidth_floor,
        coverage_scale
        * cfg.bandwidth_multiplier,
    )

    selected_distances = distances[
        selected_indices
    ]

    weights = np.exp(
        -(
            selected_distances
            / max(
                bandwidth,
                EPS,
            )
        )
        ** 2
    )

    weight_sum = float(
        np.sum(
            weights
        )
    )

    if weight_sum <= EPS:
        weights = np.ones_like(
            weights
        ) / len(
            weights
        )
    else:
        weights = (
            weights
            / weight_sum
        )

    width = int(
        camera_state_model[
            "image_width"
        ]
    )
    height = int(
        camera_state_model[
            "image_height"
        ]
    )

    image_probes = _image_probe_points(
        width,
        height,
        cfg,
    )

    world_maps = []

    for index in selected_indices:
        anchor = anchors[
            int(
                index
            )
        ]

        world_points, valid = (
            _inverse_map_points(
                image_probes,
                anchor[
                    "homography"
                ],
            )
        )

        if not np.all(
            valid
        ):
            return {
                "status": "anchor_probe_mapping_failed",
                "frame": int(
                    anchor[
                        "frame"
                    ]
                ),
            }

        world_maps.append(
            world_points
        )

    world_stack = np.asarray(
        world_maps,
        dtype=np.float64,
    )

    interpolated_world = np.sum(
        world_stack
        * weights[
            :,
            None,
            None,
        ],
        axis=0,
    )

    image_to_world = (
        _solve_homography_dlt(
            image_probes,
            interpolated_world,
        )
    )

    if image_to_world is None:
        return {
            "status": "image_to_world_fit_failed",
        }

    try:
        h_prior = np.linalg.inv(
            image_to_world
        )
    except np.linalg.LinAlgError:
        return {
            "status": "prior_inversion_failed",
        }

    h_prior = _as_h(
        h_prior
    )

    if h_prior is None:
        return {
            "status": "invalid_prior_homography",
        }

    reconstructed_world, valid = (
        _inverse_map_points(
            image_probes,
            h_prior,
        )
    )

    if not np.all(
        valid
    ):
        return {
            "status": "prior_probe_mapping_failed",
        }

    fit_errors_m = np.linalg.norm(
        reconstructed_world
        - interpolated_world,
        axis=1,
    )

    nearest_distance = float(
        selected_distances[
            0
        ]
    )

    support_ratio = (
        nearest_distance
        / max(
            coverage_scale,
            EPS,
        )
    )

    if support_ratio <= (
        cfg.good_support_ratio
    ):
        confidence = "good"
    elif support_ratio <= (
        cfg.weak_support_ratio
    ):
        confidence = "weak"
    else:
        confidence = "outside_coverage"

    neighbors = []

    for local_index, global_index in enumerate(
        selected_indices
    ):
        anchor = anchors[
            int(
                global_index
            )
        ]

        neighbors.append(
            {
                "frame": int(
                    anchor[
                        "frame"
                    ]
                ),
                "state_distance": float(
                    selected_distances[
                        local_index
                    ]
                ),
                "weight": float(
                    weights[
                        local_index
                    ]
                ),
            }
        )

    return {
        "status": "ok",
        "homography": h_prior.tolist(),
        "confidence": confidence,
        "nearest_state_distance": (
            nearest_distance
        ),
        "support_ratio": float(
            support_ratio
        ),
        "coverage_scale": float(
            coverage_scale
        ),
        "bandwidth": float(
            bandwidth
        ),
        "neighbors": neighbors,
        "probe_fit_median_m": float(
            np.median(
                fit_errors_m
            )
        ),
        "probe_fit_max_m": float(
            np.max(
                fit_errors_m
            )
        ),
        "policy": {
            "interpolates_raw_h_coefficients": False,
            "interpolates_image_probe_world_maps": True,
            "is_prior_only": True,
        },
        "config": asdict(
            cfg
        ),
    }


def _visible_world_evaluation_points(
    true_h: np.ndarray,
    *,
    image_width: int,
    image_height: int,
    pitch_length_m: float,
    pitch_width_m: float,
    cfg: CameraStateHPriorConfig,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    half_length = float(
        pitch_length_m
    ) / 2.0
    half_width = float(
        pitch_width_m
    ) / 2.0

    xs = np.linspace(
        -half_length,
        half_length,
        max(
            2,
            int(
                cfg.evaluation_x_samples
            ),
        ),
    )
    ys = np.linspace(
        -half_width,
        half_width,
        max(
            2,
            int(
                cfg.evaluation_y_samples
            ),
        ),
    )

    world = np.asarray(
        [
            (
                x,
                y,
            )
            for y in ys
            for x in xs
        ],
        dtype=np.float64,
    )

    image, valid = _project_points(
        world,
        true_h,
    )

    margin = float(
        cfg.evaluation_image_margin_px
    )

    inside = (
        valid
        & (
            image[
                :,
                0,
            ]
            >= margin
        )
        & (
            image[
                :,
                0,
            ]
            <= float(
                image_width
            )
            - margin
        )
        & (
            image[
                :,
                1,
            ]
            >= margin
        )
        & (
            image[
                :,
                1,
            ]
            <= float(
                image_height
            )
            - margin
        )
    )

    return (
        world[
            inside
        ],
        image[
            inside
        ],
    )


def evaluate_prior_against_homography(
    predicted_homography: Any,
    true_homography: Any,
    *,
    image_width: int,
    image_height: int,
    pitch_length_m: float,
    pitch_width_m: float,
    config: CameraStateHPriorConfig | None = None,
) -> dict[str, Any]:
    """Compare Hs only where the TRUE camera sees sampled pitch points."""

    cfg = (
        config
        or CameraStateHPriorConfig()
    )

    predicted = _as_h(
        predicted_homography
    )
    true_h = _as_h(
        true_homography
    )

    if (
        predicted is None
        or true_h is None
    ):
        return {
            "status": "invalid_homography",
        }

    world, true_image = (
        _visible_world_evaluation_points(
            true_h,
            image_width=image_width,
            image_height=image_height,
            pitch_length_m=pitch_length_m,
            pitch_width_m=pitch_width_m,
            cfg=cfg,
        )
    )

    if len(
        world
    ) < 4:
        return {
            "status": "not_enough_visible_evaluation_points",
            "count": len(
                world
            ),
        }

    predicted_image, valid = (
        _project_points(
            world,
            predicted,
        )
    )

    common = (
        valid
        & np.isfinite(
            predicted_image
        ).all(
            axis=1
        )
    )

    if int(
        np.count_nonzero(
            common
        )
    ) < 4:
        return {
            "status": "predicted_projection_failed",
        }

    errors_px = np.linalg.norm(
        predicted_image[
            common
        ]
        - true_image[
            common
        ],
        axis=1,
    )

    return {
        "status": "ok",
        "visible_sample_count": int(
            np.count_nonzero(
                common
            )
        ),
        "median_px": float(
            np.median(
                errors_px
            )
        ),
        "p90_px": float(
            np.percentile(
                errors_px,
                90,
            )
        ),
        "max_px": float(
            np.max(
                errors_px
            )
        ),
    }


def evaluate_prior_against_manual_points(
    predicted_homography: Any,
    calibration: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate a prior on saved HUMAN point constraints without fitting them."""

    h = _as_h(
        predicted_homography
    )

    if h is None:
        return {
            "status": "invalid_homography",
        }

    constraints = calibration.get(
        "constraints",
        [],
    )

    world = []
    image = []
    semantics = []

    if not isinstance(
        constraints,
        list,
    ):
        constraints = []

    for constraint in constraints:
        if (
            not isinstance(
                constraint,
                dict,
            )
            or constraint.get(
                "type"
            )
            != "point"
        ):
            continue

        world_value = constraint.get(
            "world"
        )
        image_value = constraint.get(
            "image"
        )

        if (
            not isinstance(
                world_value,
                dict,
            )
            or not isinstance(
                image_value,
                dict,
            )
        ):
            continue

        try:
            world.append(
                [
                    float(
                        world_value[
                            "x"
                        ]
                    ),
                    float(
                        world_value[
                            "y"
                        ]
                    ),
                ]
            )
            image.append(
                [
                    float(
                        image_value[
                            "x"
                        ]
                    ),
                    float(
                        image_value[
                            "y"
                        ]
                    ),
                ]
            )
        except (
            KeyError,
            TypeError,
            ValueError,
        ):
            continue

        semantics.append(
            constraint.get(
                "semantic"
            )
        )

    if not world:
        return {
            "status": "no_manual_points",
            "point_count": 0,
        }

    projected, valid = _project_points(
        np.asarray(
            world,
            dtype=np.float64,
        ),
        h,
    )

    observed = np.asarray(
        image,
        dtype=np.float64,
    )

    common = (
        valid
        & np.isfinite(
            projected
        ).all(
            axis=1
        )
    )

    if not np.any(
        common
    ):
        return {
            "status": "projection_failed",
            "point_count": len(
                world
            ),
        }

    errors = np.linalg.norm(
        projected[
            common
        ]
        - observed[
            common
        ],
        axis=1,
    )

    used_semantics = [
        semantics[
            index
        ]
        for index in np.flatnonzero(
            common
        )
    ]

    return {
        "status": "ok",
        "point_count": int(
            len(
                errors
            )
        ),
        "median_px": float(
            np.median(
                errors
            )
        ),
        "p90_px": float(
            np.percentile(
                errors,
                90,
            )
        ),
        "max_px": float(
            np.max(
                errors
            )
        ),
        "per_point": [
            {
                "semantic": semantic,
                "error_px": float(
                    error
                ),
            }
            for semantic, error in zip(
                used_semantics,
                errors,
            )
        ],
    }


def homography_prior_leave_one_out(
    camera_state_model: dict[str, Any],
    calibrations: list[dict[str, Any]],
    *,
    config: CameraStateHPriorConfig | None = None,
) -> dict[str, Any]:
    """Decoder-only LOO using each HUMAN anchor's known 3-D state.

    This intentionally isolates State -> H from the temporal observer.
    The held-out frame's H is excluded from the decoder anchors.
    """

    cfg = (
        config
        or CameraStateHPriorConfig()
    )

    if camera_state_model.get(
        "status"
    ) != "ok":
        return {
            "status": "invalid_camera_state_model",
        }

    calibration_map = {}

    for calibration in calibrations:
        if (
            not isinstance(
                calibration,
                dict,
            )
            or calibration.get(
                "source"
            )
            != "manual"
            or calibration.get(
                "purpose"
            )
            != "match_setup"
        ):
            continue

        try:
            frame = int(
                calibration.get(
                    "frame"
                )
            )
        except (
            TypeError,
            ValueError,
        ):
            continue

        if _as_h(
            calibration.get(
                "homography"
            )
        ) is not None:
            calibration_map[
                frame
            ] = calibration

    width = int(
        camera_state_model[
            "image_width"
        ]
    )
    height = int(
        camera_state_model[
            "image_height"
        ]
    )
    length_m = float(
        camera_state_model[
            "pitch_length_m"
        ]
    )
    pitch_width_m = float(
        camera_state_model[
            "pitch_width_m"
        ]
    )

    results = []

    for anchor in camera_state_model.get(
        "anchors",
        [],
    ):
        frame = int(
            anchor[
                "frame"
            ]
        )

        calibration = calibration_map.get(
            frame
        )
        if calibration is None:
            continue

        prior = (
            estimate_homography_prior_from_state(
                camera_state_model,
                calibrations,
                anchor[
                    "state"
                ],
                excluded_frames={
                    frame
                },
                config=cfg,
            )
        )

        if prior.get(
            "status"
        ) != "ok":
            results.append(
                {
                    "frame": frame,
                    "status": prior.get(
                        "status"
                    ),
                }
            )
            continue

        evaluation = (
            evaluate_prior_against_homography(
                prior[
                    "homography"
                ],
                calibration[
                    "homography"
                ],
                image_width=width,
                image_height=height,
                pitch_length_m=length_m,
                pitch_width_m=pitch_width_m,
                config=cfg,
            )
        )

        point_evaluation = (
            evaluate_prior_against_manual_points(
                prior[
                    "homography"
                ],
                calibration,
            )
        )

        results.append(
            {
                "frame": frame,
                "status": "ok",
                "confidence": prior[
                    "confidence"
                ],
                "support_ratio": prior[
                    "support_ratio"
                ],
                "neighbors": prior[
                    "neighbors"
                ],
                "visible_grid": evaluation,
                "manual_points": point_evaluation,
            }
        )

    valid = [
        item
        for item in results
        if (
            item.get(
                "status"
            )
            == "ok"
            and item.get(
                "visible_grid",
                {},
            ).get(
                "status"
            )
            == "ok"
        )
    ]

    if not valid:
        return {
            "status": "no_valid_holdouts",
            "per_anchor": results,
        }

    medians = np.asarray(
        [
            item[
                "visible_grid"
            ][
                "median_px"
            ]
            for item in valid
        ],
        dtype=np.float64,
    )
    p90s = np.asarray(
        [
            item[
                "visible_grid"
            ][
                "p90_px"
            ]
            for item in valid
        ],
        dtype=np.float64,
    )

    return {
        "status": "ok",
        "holdout_count": len(
            valid
        ),
        "visible_grid_median_of_medians_px": float(
            np.median(
                medians
            )
        ),
        "visible_grid_p90_of_medians_px": float(
            np.percentile(
                medians,
                90,
            )
        ),
        "visible_grid_max_median_px": float(
            np.max(
                medians
            )
        ),
        "visible_grid_median_p90_px": float(
            np.median(
                p90s
            )
        ),
        "per_anchor": results,
    }


def summarize_homography_prior_benchmark(
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
        "holdout_count": result.get(
            "holdout_count",
            0,
        ),
        "visible_grid_median_of_medians_px": (
            result.get(
                "visible_grid_median_of_medians_px"
            )
        ),
        "visible_grid_p90_of_medians_px": (
            result.get(
                "visible_grid_p90_of_medians_px"
            )
        ),
        "visible_grid_max_median_px": (
            result.get(
                "visible_grid_max_median_px"
            )
        ),
        "visible_grid_median_p90_px": (
            result.get(
                "visible_grid_median_p90_px"
            )
        ),
    }
