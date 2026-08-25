"""Camera State V1 core for RealJaen FIELD.

Camera State V1 is deliberately narrower than homography reconstruction.

It learns a 3-D state space from strong independent HUMAN match_setup anchors,
using robust local camera observables derived from each H:

    - physical pitch coordinate below the image centre (x, y)
    - local zoom
    - local projective anisotropy
    - ground-plane horizon distance
    - ground-plane horizon angle

It then estimates the state of a raw-video view from the existing
geometry-first FIELD scanner descriptor by local interpolation between HUMAN
anchors in descriptor space.

Important boundaries
--------------------
- No raw homography coefficients are compared directly.
- No whole-pitch projected signature is required, so projective poles in unseen
  regions do not explode the metric.
- No H is reconstructed for non-HUMAN frames in V1.
- No propagation/refinement/legacy Camera Atlas data are consumed.
- A state estimate may be returned with low confidence; callers must not turn a
  low-confidence state into a field projection automatically.

Public API
----------
build_camera_state_model(...)
fit_camera_state_view_index(...)
estimate_camera_state(...)
camera_state_leave_one_out(...)
summarize_camera_state_model(...)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import numpy as np

from supervisor.field_view_scanner import field_view_distance


EPS = 1e-10

STATE_FEATURE_NAMES = (
    "centre_x_over_length",
    "centre_y_over_width",
    "log_zoom_px_per_m",
    "log_anisotropy",
    "horizon_distance_over_image_height",
    "horizon_angle_rad",
)


@dataclass(frozen=True)
class CameraStateConfig:
    dimensions: int = 3
    interpolation_neighbors: int = 3
    descriptor_bandwidth_floor: float = 0.018
    descriptor_bandwidth_multiplier: float = 1.35

    # Confidence is based on descriptor support + neighbour agreement.
    good_support_ratio: float = 1.35
    weak_support_ratio: float = 2.40
    good_state_spread: float = 0.85
    weak_state_spread: float = 1.60

    # Numerical regularisation.
    minimum_feature_std: float = 1e-8
    minimum_component_std: float = 1e-8


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
    """Map image -> pitch and return d(world)/d(image) at (u,v)."""

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
        [nx / denominator, ny / denominator],
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


def _ground_horizon(h: np.ndarray) -> np.ndarray | None:
    try:
        line = np.linalg.inv(h).T @ np.asarray([0.0, 0.0, 1.0])
    except np.linalg.LinAlgError:
        return None

    norm = float(np.linalg.norm(line[:2]))
    if not math.isfinite(norm) or norm < EPS:
        return None

    line = line / norm

    # Stable line sign.
    if line[1] < 0 or (abs(float(line[1])) < EPS and line[0] < 0):
        line = -line

    return line


def _wrap_half_turn_radians(value: float) -> float:
    while value > math.pi / 2:
        value -= math.pi
    while value <= -math.pi / 2:
        value += math.pi
    return float(value)


def camera_observables_from_homography(
    homography: Any,
    *,
    image_width: int,
    image_height: int,
    pitch_length_m: float,
    pitch_width_m: float,
) -> dict[str, Any] | None:
    """Return robust local camera observables and the 6-D feature vector."""

    h = _as_h(homography)
    if h is None:
        return None

    width = int(image_width)
    height = int(image_height)
    length_m = float(pitch_length_m)
    pitch_width = float(pitch_width_m)

    if width <= 0 or height <= 0 or length_m <= 0 or pitch_width <= 0:
        return None

    cx = width / 2.0
    cy = height / 2.0

    local = _inverse_map_and_jacobian(h, cx, cy)
    if local is None:
        return None

    centre_world, image_to_world = local

    try:
        singular_values = np.linalg.svd(image_to_world, compute_uv=False)
    except np.linalg.LinAlgError:
        return None

    if (
        len(singular_values) != 2
        or not np.isfinite(singular_values).all()
        or singular_values[1] <= EPS
    ):
        return None

    area_world_per_pixel_sq = abs(float(np.linalg.det(image_to_world)))
    if area_world_per_pixel_sq <= EPS:
        return None

    zoom_px_per_m = 1.0 / math.sqrt(area_world_per_pixel_sq)
    anisotropy = float(singular_values[0] / singular_values[1])

    horizon = _ground_horizon(h)
    if horizon is None:
        return None

    line_a, line_b, line_c = [float(value) for value in horizon]
    horizon_distance_px = line_a * cx + line_b * cy + line_c
    horizon_angle_rad = _wrap_half_turn_radians(
        math.atan2(-line_a, line_b)
    )

    feature = np.asarray(
        [
            float(centre_world[0]) / length_m,
            float(centre_world[1]) / pitch_width,
            math.log(max(float(zoom_px_per_m), EPS)),
            math.log(max(float(anisotropy), 1.0)),
            float(horizon_distance_px) / height,
            horizon_angle_rad,
        ],
        dtype=np.float64,
    )

    if not np.isfinite(feature).all():
        return None

    return {
        "centre_world_m": [
            float(centre_world[0]),
            float(centre_world[1]),
        ],
        "zoom_px_per_m": float(zoom_px_per_m),
        "anisotropy": float(anisotropy),
        "horizon_signed_distance_px": float(horizon_distance_px),
        "horizon_angle_deg": float(math.degrees(horizon_angle_rad)),
        "feature": feature,
    }


def _collect_strong_match_anchors(
    calibrations: list[dict[str, Any]],
    completed_frames: list[int] | set[int],
    *,
    image_width: int,
    image_height: int,
    pitch_length_m: float,
    pitch_width_m: float,
) -> list[dict[str, Any]]:
    completed: set[int] = set()
    for value in completed_frames:
        try:
            completed.add(int(value))
        except (TypeError, ValueError):
            continue

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

        observables = camera_observables_from_homography(
            h,
            image_width=image_width,
            image_height=image_height,
            pitch_length_m=pitch_length_m,
            pitch_width_m=pitch_width_m,
        )
        if observables is None:
            continue

        anchors.append(
            {
                "frame": frame,
                "homography": h,
                "observables": observables,
                "feature": observables["feature"],
            }
        )

    anchors.sort(key=lambda item: item["frame"])
    return anchors


def build_camera_state_model(
    calibrations: list[dict[str, Any]],
    completed_frames: list[int] | set[int],
    *,
    image_width: int,
    image_height: int,
    pitch_length_m: float,
    pitch_width_m: float,
    config: CameraStateConfig | None = None,
) -> dict[str, Any]:
    """Fit the 3-D HUMAN Camera State coordinate system."""

    cfg = config or CameraStateConfig()

    anchors = _collect_strong_match_anchors(
        calibrations,
        completed_frames,
        image_width=image_width,
        image_height=image_height,
        pitch_length_m=pitch_length_m,
        pitch_width_m=pitch_width_m,
    )

    if len(anchors) < cfg.dimensions + 2:
        return {
            "status": "not_enough_anchors",
            "anchor_count": len(anchors),
            "required_anchor_count": cfg.dimensions + 2,
            "config": asdict(cfg),
        }

    features = np.asarray(
        [anchor["feature"] for anchor in anchors],
        dtype=np.float64,
    )

    feature_mean = np.mean(features, axis=0)
    feature_std = np.std(features, axis=0, ddof=0)
    feature_std = np.where(
        feature_std < cfg.minimum_feature_std,
        1.0,
        feature_std,
    )

    z_features = (features - feature_mean) / feature_std
    z_mean = np.mean(z_features, axis=0)
    centered = z_features - z_mean

    try:
        _, singular_values, vt = np.linalg.svd(
            centered,
            full_matrices=False,
        )
    except np.linalg.LinAlgError:
        return {
            "status": "pca_failed",
            "anchor_count": len(anchors),
            "config": asdict(cfg),
        }

    maximum_rank = min(len(anchors) - 1, len(vt))
    dimensions = min(max(1, int(cfg.dimensions)), maximum_rank)
    basis = vt[:dimensions]

    raw_states = centered @ basis.T
    state_mean = np.mean(raw_states, axis=0)
    state_std = np.std(raw_states, axis=0, ddof=0)
    state_std = np.where(
        state_std < cfg.minimum_component_std,
        1.0,
        state_std,
    )
    states = (raw_states - state_mean) / state_std

    variance = singular_values ** 2
    variance_total = float(np.sum(variance))
    variance_ratio = (
        variance / variance_total
        if variance_total > EPS
        else np.zeros_like(variance)
    )

    anchor_records = []
    for anchor, state, raw_state in zip(anchors, states, raw_states):
        anchor_records.append(
            {
                "frame": int(anchor["frame"]),
                "state": state.astype(float).tolist(),
                "raw_state": raw_state.astype(float).tolist(),
                "observables": {
                    key: value
                    for key, value in anchor["observables"].items()
                    if key != "feature"
                },
            }
        )

    # Camera-state coverage scale: nearest HUMAN state distance.
    state_distances = np.full(
        (len(states), len(states)),
        np.inf,
        dtype=np.float64,
    )
    for first in range(len(states)):
        for second in range(first + 1, len(states)):
            distance = float(
                np.linalg.norm(states[first] - states[second])
                / math.sqrt(dimensions)
            )
            state_distances[first, second] = distance
            state_distances[second, first] = distance

    nearest_state_distances = np.min(state_distances, axis=1)
    state_coverage_scale = float(np.median(nearest_state_distances))

    return {
        "status": "ok",
        "dimensions": dimensions,
        "feature_names": list(STATE_FEATURE_NAMES),
        "anchor_count": len(anchor_records),
        "anchor_frames": [item["frame"] for item in anchor_records],
        "anchors": anchor_records,
        "feature_mean": feature_mean.tolist(),
        "feature_std": feature_std.tolist(),
        "pca_mean": z_mean.tolist(),
        "pca_components": basis.tolist(),
        "state_mean": state_mean.tolist(),
        "state_std": state_std.tolist(),
        "variance_ratio": variance_ratio[:maximum_rank].tolist(),
        "cumulative_variance": np.cumsum(
            variance_ratio[:maximum_rank]
        ).tolist(),
        "state_coverage_scale": state_coverage_scale,
        "image_width": int(image_width),
        "image_height": int(image_height),
        "pitch_length_m": float(pitch_length_m),
        "pitch_width_m": float(pitch_width_m),
        "config": asdict(cfg),
    }


def state_from_homography(
    model: dict[str, Any],
    homography: Any,
) -> dict[str, Any]:
    """Project a known H into the learned HUMAN state space."""

    if model.get("status") != "ok":
        return {"status": "invalid_model"}

    observables = camera_observables_from_homography(
        homography,
        image_width=int(model["image_width"]),
        image_height=int(model["image_height"]),
        pitch_length_m=float(model["pitch_length_m"]),
        pitch_width_m=float(model["pitch_width_m"]),
    )
    if observables is None:
        return {"status": "invalid_homography"}

    feature = np.asarray(observables["feature"], dtype=np.float64)

    feature_mean = np.asarray(model["feature_mean"], dtype=np.float64)
    feature_std = np.asarray(model["feature_std"], dtype=np.float64)
    pca_mean = np.asarray(model["pca_mean"], dtype=np.float64)
    components = np.asarray(model["pca_components"], dtype=np.float64)
    state_mean = np.asarray(model["state_mean"], dtype=np.float64)
    state_std = np.asarray(model["state_std"], dtype=np.float64)

    z = (feature - feature_mean) / feature_std
    raw_state = (z - pca_mean) @ components.T
    state = (raw_state - state_mean) / state_std

    return {
        "status": "ok",
        "state": state.tolist(),
        "raw_state": raw_state.tolist(),
        "observables": {
            key: value
            for key, value in observables.items()
            if key != "feature"
        },
    }


def _descriptor_array(value: Any) -> np.ndarray | None:
    try:
        descriptor = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None

    if descriptor.size == 0 or not np.isfinite(descriptor).all():
        return None

    norm = float(np.linalg.norm(descriptor))
    if norm < EPS:
        return None

    return descriptor / norm


def fit_camera_state_view_index(
    model: dict[str, Any],
    anchor_view_descriptors: dict[int, Any],
) -> dict[str, Any]:
    """Attach one raw-view scanner descriptor to each HUMAN state anchor."""

    if model.get("status") != "ok":
        return {"status": "invalid_model"}

    records = []

    for anchor in model.get("anchors", []):
        frame = int(anchor["frame"])
        descriptor = _descriptor_array(
            anchor_view_descriptors.get(frame)
        )
        if descriptor is None:
            continue

        state = np.asarray(anchor["state"], dtype=np.float64)
        records.append(
            {
                "frame": frame,
                "descriptor": descriptor,
                "state": state,
            }
        )

    required = max(
        int(model.get("dimensions", 3)) + 1,
        4,
    )
    if len(records) < required:
        return {
            "status": "not_enough_descriptors",
            "descriptor_count": len(records),
            "required_descriptor_count": required,
        }

    descriptor_distances = np.full(
        (len(records), len(records)),
        np.inf,
        dtype=np.float64,
    )

    for first in range(len(records)):
        for second in range(first + 1, len(records)):
            distance = field_view_distance(
                records[first]["descriptor"],
                records[second]["descriptor"],
            )
            descriptor_distances[first, second] = distance
            descriptor_distances[second, first] = distance

    nearest_descriptor_distances = np.min(
        descriptor_distances,
        axis=1,
    )

    typical_nearest_descriptor_distance = float(
        np.median(nearest_descriptor_distances)
    )

    if not math.isfinite(typical_nearest_descriptor_distance):
        typical_nearest_descriptor_distance = 0.03

    return {
        "status": "ok",
        "anchor_count": len(records),
        "anchor_frames": [item["frame"] for item in records],
        "typical_nearest_descriptor_distance": (
            typical_nearest_descriptor_distance
        ),
        # NumPy values stay internal to Python callers. This object is not meant
        # for direct JSON persistence.
        "_records": records,
    }


def _confidence_label(
    support_ratio: float,
    state_spread: float,
    cfg: CameraStateConfig,
) -> str:
    if (
        support_ratio <= cfg.good_support_ratio
        and state_spread <= cfg.good_state_spread
    ):
        return "good"

    if (
        support_ratio <= cfg.weak_support_ratio
        and state_spread <= cfg.weak_state_spread
    ):
        return "weak"

    return "outside_coverage"


def estimate_camera_state(
    model: dict[str, Any],
    view_index: dict[str, Any],
    query_descriptor: Any,
    *,
    query_frame: int | None = None,
    config: CameraStateConfig | None = None,
) -> dict[str, Any]:
    """Estimate Camera State from a raw-view geometry-first descriptor."""

    cfg = config or CameraStateConfig()

    if model.get("status") != "ok":
        return {"status": "invalid_model"}
    if view_index.get("status") != "ok":
        return {"status": "invalid_view_index"}

    descriptor = _descriptor_array(query_descriptor)
    if descriptor is None:
        return {"status": "invalid_descriptor"}

    records = view_index.get("_records", [])
    if not records:
        return {"status": "empty_view_index"}

    distances = np.asarray(
        [
            field_view_distance(
                descriptor,
                record["descriptor"],
            )
            for record in records
        ],
        dtype=np.float64,
    )

    order = np.argsort(distances)
    neighbor_count = min(
        max(2, int(cfg.interpolation_neighbors)),
        len(records),
    )
    selected = order[:neighbor_count]

    typical = float(
        view_index.get(
            "typical_nearest_descriptor_distance",
            0.03,
        )
    )
    bandwidth = max(
        cfg.descriptor_bandwidth_floor,
        typical * cfg.descriptor_bandwidth_multiplier,
    )

    selected_distances = distances[selected]
    weights = np.exp(
        -selected_distances / max(bandwidth, EPS)
    )

    weight_sum = float(np.sum(weights))
    if weight_sum <= EPS:
        weights = np.ones_like(weights) / len(weights)
    else:
        weights = weights / weight_sum

    neighbor_states = np.asarray(
        [records[int(index)]["state"] for index in selected],
        dtype=np.float64,
    )

    estimated_state = np.sum(
        neighbor_states * weights[:, None],
        axis=0,
    )

    state_distances_to_estimate = np.linalg.norm(
        neighbor_states - estimated_state[None, :],
        axis=1,
    ) / math.sqrt(max(1, neighbor_states.shape[1]))

    state_spread = float(
        np.sqrt(
            np.sum(
                weights
                * state_distances_to_estimate ** 2
            )
        )
    )

    nearest_descriptor_distance = float(
        selected_distances[0]
    )
    support_ratio = nearest_descriptor_distance / max(
        typical,
        cfg.descriptor_bandwidth_floor,
    )

    confidence = _confidence_label(
        support_ratio,
        state_spread,
        cfg,
    )

    neighbors = []
    for local_index, global_index in enumerate(selected):
        record = records[int(global_index)]
        neighbors.append(
            {
                "frame": int(record["frame"]),
                "descriptor_distance": float(
                    selected_distances[local_index]
                ),
                "weight": float(weights[local_index]),
                "state_distance_from_estimate": float(
                    state_distances_to_estimate[local_index]
                ),
            }
        )

    return {
        "status": "ok",
        "query_frame": (
            int(query_frame)
            if query_frame is not None
            else None
        ),
        "state": estimated_state.tolist(),
        "confidence": confidence,
        "nearest_descriptor_distance": nearest_descriptor_distance,
        "support_ratio": float(support_ratio),
        "neighbor_state_spread": state_spread,
        "descriptor_bandwidth": float(bandwidth),
        "neighbors": neighbors,
        "policy": {
            "returns_homography": False,
            "low_confidence_state_must_not_drive_projection": True,
            "uses_raw_view_descriptor": True,
            "uses_time_as_state": False,
        },
    }


def camera_state_leave_one_out(
    model: dict[str, Any],
    anchor_view_descriptors: dict[int, Any],
    *,
    config: CameraStateConfig | None = None,
) -> dict[str, Any]:
    """LOO benchmark of RAW view descriptor -> known HUMAN 3-D state."""

    cfg = config or CameraStateConfig()

    if model.get("status") != "ok":
        return {"status": "invalid_model"}

    anchor_states = {
        int(item["frame"]): np.asarray(
            item["state"],
            dtype=np.float64,
        )
        for item in model.get("anchors", [])
    }

    frames = [
        frame
        for frame in model.get("anchor_frames", [])
        if _descriptor_array(
            anchor_view_descriptors.get(int(frame))
        )
        is not None
    ]

    if len(frames) < 4:
        return {
            "status": "not_enough_descriptors",
            "anchor_count": len(frames),
        }

    results = []

    for holdout_frame in frames:
        reduced_descriptors = {
            int(frame): anchor_view_descriptors[int(frame)]
            for frame in frames
            if int(frame) != int(holdout_frame)
        }

        view_index = fit_camera_state_view_index(
            model,
            reduced_descriptors,
        )
        if view_index.get("status") != "ok":
            continue

        estimate = estimate_camera_state(
            model,
            view_index,
            anchor_view_descriptors[int(holdout_frame)],
            query_frame=int(holdout_frame),
            config=cfg,
        )
        if estimate.get("status") != "ok":
            continue

        true_state = anchor_states[int(holdout_frame)]
        estimated_state = np.asarray(
            estimate["state"],
            dtype=np.float64,
        )

        error = float(
            np.linalg.norm(true_state - estimated_state)
            / math.sqrt(max(1, len(true_state)))
        )

        results.append(
            {
                "frame": int(holdout_frame),
                "state_rmse": error,
                "confidence": estimate["confidence"],
                "nearest_descriptor_distance": (
                    estimate["nearest_descriptor_distance"]
                ),
                "support_ratio": estimate["support_ratio"],
                "neighbor_state_spread": (
                    estimate["neighbor_state_spread"]
                ),
                "neighbors": estimate["neighbors"],
            }
        )

    if not results:
        return {"status": "no_valid_holdouts"}

    errors = np.asarray(
        [item["state_rmse"] for item in results],
        dtype=np.float64,
    )

    return {
        "status": "ok",
        "holdout_count": len(results),
        "median_state_rmse": float(np.median(errors)),
        "p90_state_rmse": float(np.percentile(errors, 90)),
        "max_state_rmse": float(np.max(errors)),
        "per_anchor": results,
    }


def summarize_camera_state_model(
    model: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(model, dict):
        return {"status": "invalid_model"}

    return {
        "status": model.get("status"),
        "dimensions": model.get("dimensions"),
        "anchor_count": model.get("anchor_count", 0),
        "anchor_frames": model.get("anchor_frames", []),
        "variance_ratio": model.get("variance_ratio", []),
        "cumulative_variance": model.get(
            "cumulative_variance",
            [],
        ),
        "state_coverage_scale": model.get(
            "state_coverage_scale"
        ),
    }
