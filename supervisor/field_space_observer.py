"""FIELD SPACE / OBSERVER V1 · FSO04.

Correct canonical-pitch coordinate system.

The RealJaen canonical pitch is CENTERED at (0, 0):

    X in [-L/2, +L/2]
    Y in [-W/2, +W/2]

This is the same convention used by supervisor/static/pitch.js and by every
existing calibration homography.

Earlier FSO prototypes incorrectly assumed [0,L] x [0,W]. That shifted the
entire pitch rectangle and made valid camera homographies appear to have no
visible-pitch intersection in many frames.

FSO04 keeps the robust two-projective-branch visibility model from FSO03, but
uses the correct canonical field geometry.

Coordinate outputs
------------------
Centered canonical coordinates are authoritative because H uses them:
    centered.x_m, centered.y_m

For UI/football-analysis convenience we also expose shifted pitch coordinates:
    pitch.x_m = centered.x_m + L/2   in [0,L]
    pitch.y_m = centered.y_m + W/2   in [0,W]

Homography convention:
    image ~= H @ [X_centered, Y_centered, 1]

`optical_axis_ground_point` is H^-1(image centre).
`visible_focus_field` is the centroid of the selected visible-pitch polygon.
Neither is the physical 3-D camera position.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


EPS = 1e-9


@dataclass(frozen=True)
class FieldSpaceObserverConfig:
    viewport_tolerance_px: float = 1.5
    pitch_tolerance_m: float = 1e-6

    support_x_fracs: tuple[float, ...] = (
        0.10, 0.25, 0.50, 0.75, 0.90
    )
    support_y_fracs: tuple[float, ...] = (
        0.50, 0.62, 0.74, 0.86, 0.94
    )

    emission_lower_support_weight: float = 5.0
    emission_image_area_weight: float = 2.0
    emission_bottom_centre_weight: float = 2.0
    emission_image_centre_weight: float = 0.75

    transition_centroid_m_weight: float = 0.08
    transition_fraction_weight: float = 1.5
    transition_empty_penalty: float = 8.0


def _normalise_h(value: Any) -> np.ndarray | None:
    try:
        h = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return None

    if h.shape != (3, 3) or not np.isfinite(h).all():
        return None

    if abs(float(h[2, 2])) > EPS:
        h = h / float(h[2, 2])
    else:
        norm = float(np.linalg.norm(h))
        if not np.isfinite(norm) or norm < EPS:
            return None
        h = h / norm

    try:
        det = float(np.linalg.det(h))
        cond = float(np.linalg.cond(h))
    except np.linalg.LinAlgError:
        return None

    if (
        not np.isfinite(det)
        or abs(det) < 1e-12
        or not np.isfinite(cond)
        or cond > 1e14
    ):
        return None

    return h


def _project_homogeneous(
    h: np.ndarray,
    points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    hp = np.column_stack((pts, np.ones(len(pts), dtype=np.float64)))
    q = hp @ h.T
    denominator = q[:, 2]

    valid = (
        np.isfinite(q).all(axis=1)
        & np.isfinite(denominator)
        & (np.abs(denominator) > EPS)
    )

    projected = np.full((len(pts), 2), np.nan, dtype=np.float64)
    projected[valid] = q[valid, :2] / denominator[valid, None]
    valid &= np.isfinite(projected).all(axis=1)

    return projected, denominator, valid


def field_to_image(
    homography: Any,
    field_points: Any,
) -> dict[str, Any]:
    """Project CENTERED canonical pitch points to image pixels."""

    h = _normalise_h(homography)
    if h is None:
        return {"status": "invalid_homography"}

    try:
        points = np.asarray(field_points, dtype=np.float64).reshape(-1, 2)
    except (TypeError, ValueError):
        return {"status": "invalid_points"}

    projected, denominator, valid = _project_homogeneous(h, points)

    return {
        "status": "ok",
        "points": projected.tolist(),
        "valid": valid.tolist(),
        "denominator": denominator.tolist(),
    }


def image_to_field(
    homography: Any,
    image_points: Any,
) -> dict[str, Any]:
    """Map image pixels to CENTERED canonical pitch coordinates."""

    h = _normalise_h(homography)
    if h is None:
        return {"status": "invalid_homography"}

    try:
        inverse = np.linalg.inv(h)
    except np.linalg.LinAlgError:
        return {"status": "inverse_failed"}

    try:
        points = np.asarray(image_points, dtype=np.float64).reshape(-1, 2)
    except (TypeError, ValueError):
        return {"status": "invalid_points"}

    projected, denominator, valid = _project_homogeneous(
        inverse,
        points,
    )

    return {
        "status": "ok",
        "points": projected.tolist(),
        "valid": valid.tolist(),
        "denominator": denominator.tolist(),
    }


def centered_to_pitch_coordinates(
    point: Any,
    *,
    pitch_length_m: float,
    pitch_width_m: float,
) -> dict[str, float]:
    p = np.asarray(point, dtype=np.float64).reshape(2)
    return {
        "x_m": float(p[0] + 0.5 * pitch_length_m),
        "y_m": float(p[1] + 0.5 * pitch_width_m),
    }


def pitch_to_centered_coordinates(
    point: Any,
    *,
    pitch_length_m: float,
    pitch_width_m: float,
) -> dict[str, float]:
    p = np.asarray(point, dtype=np.float64).reshape(2)
    return {
        "x_m": float(p[0] - 0.5 * pitch_length_m),
        "y_m": float(p[1] - 0.5 * pitch_width_m),
    }


def _inside_pitch_centered(
    point: np.ndarray,
    *,
    pitch_length_m: float,
    pitch_width_m: float,
    tolerance_m: float,
) -> bool:
    half_l = 0.5 * float(pitch_length_m)
    half_w = 0.5 * float(pitch_width_m)

    return bool(
        -half_l - tolerance_m
        <= float(point[0])
        <= half_l + tolerance_m
        and -half_w - tolerance_m
        <= float(point[1])
        <= half_w + tolerance_m
    )


def _line_value(line: np.ndarray, point: np.ndarray) -> float:
    return float(
        line[0] * point[0]
        + line[1] * point[1]
        + line[2]
    )


def _segment_boundary_intersection(
    start: np.ndarray,
    end: np.ndarray,
    start_value: float,
    end_value: float,
) -> np.ndarray:
    denom = start_value - end_value
    if abs(denom) < EPS:
        return 0.5 * (start + end)

    t = float(start_value / denom)
    t = min(max(t, 0.0), 1.0)
    return start + t * (end - start)


def _clip_polygon_halfplane(
    polygon: list[np.ndarray],
    line: np.ndarray,
) -> list[np.ndarray]:
    if not polygon:
        return []

    result: list[np.ndarray] = []

    previous = polygon[-1]
    previous_value = _line_value(line, previous)
    previous_inside = previous_value >= -EPS

    for current in polygon:
        current_value = _line_value(line, current)
        current_inside = current_value >= -EPS

        if current_inside:
            if not previous_inside:
                result.append(
                    _segment_boundary_intersection(
                        previous,
                        current,
                        previous_value,
                        current_value,
                    )
                )
            result.append(current)
        elif previous_inside:
            result.append(
                _segment_boundary_intersection(
                    previous,
                    current,
                    previous_value,
                    current_value,
                )
            )

        previous = current
        previous_value = current_value
        previous_inside = current_inside

    return result


def _polygon_area(points: np.ndarray) -> float:
    if len(points) < 3:
        return 0.0

    x = points[:, 0]
    y = points[:, 1]

    return float(
        0.5
        * abs(
            np.dot(x, np.roll(y, -1))
            - np.dot(y, np.roll(x, -1))
        )
    )


def _polygon_centroid(points: np.ndarray) -> np.ndarray | None:
    if len(points) < 3:
        return None

    x = points[:, 0]
    y = points[:, 1]
    cross = x * np.roll(y, -1) - np.roll(x, -1) * y
    area2 = float(np.sum(cross))

    if abs(area2) < EPS:
        return np.mean(points, axis=0)

    return np.asarray(
        [
            np.sum((x + np.roll(x, -1)) * cross) / (3.0 * area2),
            np.sum((y + np.roll(y, -1)) * cross) / (3.0 * area2),
        ],
        dtype=np.float64,
    )


def _viewport_halfplanes(
    h: np.ndarray,
    *,
    image_width: int,
    image_height: int,
    branch_sign: int,
) -> list[np.ndarray]:
    hu = np.asarray(h[0], dtype=np.float64)
    hv = np.asarray(h[1], dtype=np.float64)
    hw = np.asarray(h[2], dtype=np.float64)
    s = float(branch_sign)

    return [
        s * hw,
        s * hu,
        s * (float(image_width) * hw - hu),
        s * hv,
        s * (float(image_height) * hw - hv),
    ]


def _clip_segment(
    start: np.ndarray,
    end: np.ndarray,
    halfplanes: list[np.ndarray],
) -> tuple[float, float] | None:
    t_min = 0.0
    t_max = 1.0
    direction = end - start

    for line in halfplanes:
        slope = float(
            line[0] * direction[0]
            + line[1] * direction[1]
        )
        intercept = _line_value(line, start)

        if abs(slope) < EPS:
            if intercept < 0.0:
                return None
            continue

        boundary = -intercept / slope

        if slope > 0.0:
            t_min = max(t_min, boundary)
        else:
            t_max = min(t_max, boundary)

        if t_min > t_max:
            return None

    t_min = min(max(t_min, 0.0), 1.0)
    t_max = min(max(t_max, 0.0), 1.0)

    return (
        (float(t_min), float(t_max))
        if t_max >= t_min
        else None
    )


def _boundary_visibility(
    halfplanes: list[np.ndarray],
    *,
    pitch_length_m: float,
    pitch_width_m: float,
) -> dict[str, dict[str, Any]]:
    half_l = 0.5 * pitch_length_m
    half_w = 0.5 * pitch_width_m

    segments = {
        "left_goal_line": (
            np.asarray([-half_l, -half_w], dtype=np.float64),
            np.asarray([-half_l, half_w], dtype=np.float64),
            pitch_width_m,
        ),
        "right_goal_line": (
            np.asarray([half_l, -half_w], dtype=np.float64),
            np.asarray([half_l, half_w], dtype=np.float64),
            pitch_width_m,
        ),
        "top_touchline": (
            np.asarray([-half_l, -half_w], dtype=np.float64),
            np.asarray([half_l, -half_w], dtype=np.float64),
            pitch_length_m,
        ),
        "bottom_touchline": (
            np.asarray([-half_l, half_w], dtype=np.float64),
            np.asarray([half_l, half_w], dtype=np.float64),
            pitch_length_m,
        ),
    }

    result: dict[str, dict[str, Any]] = {}

    for name, (start, end, full_length) in segments.items():
        clipped = _clip_segment(start, end, halfplanes)

        if clipped is None:
            result[name] = {
                "visible": False,
                "visible_fraction": 0.0,
                "visible_length_m": 0.0,
                "segment_field_centered": None,
                "segment_pitch": None,
            }
            continue

        t0, t1 = clipped
        fraction = max(0.0, min(1.0, t1 - t0))
        p0 = start + t0 * (end - start)
        p1 = start + t1 * (end - start)

        result[name] = {
            "visible": fraction > 1e-6,
            "visible_fraction": float(fraction),
            "visible_length_m": float(fraction * full_length),
            "segment_field_centered": [p0.tolist(), p1.tolist()],
            "segment_pitch": [
                [
                    float(p0[0] + half_l),
                    float(p0[1] + half_w),
                ],
                [
                    float(p1[0] + half_l),
                    float(p1[1] + half_w),
                ],
            ],
        }

    return result


def _image_polygon_metrics(
    h: np.ndarray,
    polygon_field: np.ndarray,
    *,
    image_width: int,
    image_height: int,
) -> dict[str, Any]:
    if len(polygon_field) < 3:
        return {
            "polygon": [],
            "area_px2": 0.0,
            "fraction": 0.0,
            "centroid": None,
        }

    projected, _denominator, valid = _project_homogeneous(
        h,
        polygon_field,
    )

    if not valid.all():
        return {
            "polygon": [],
            "area_px2": 0.0,
            "fraction": 0.0,
            "centroid": None,
        }

    area = _polygon_area(projected)
    image_area = float(image_width * image_height)
    centroid = _polygon_centroid(projected)

    return {
        "polygon": projected.tolist(),
        "area_px2": float(area),
        "fraction": float(
            max(
                0.0,
                min(
                    1.0,
                    area / max(image_area, 1.0),
                ),
            )
        ),
        "centroid": (
            {
                "x_px": float(centroid[0]),
                "y_px": float(centroid[1]),
            }
            if centroid is not None
            else None
        ),
    }


def _single_image_probe_inside(
    h: np.ndarray,
    *,
    image_point: tuple[float, float],
    branch_sign: int,
    pitch_length_m: float,
    pitch_width_m: float,
    tolerance_m: float,
) -> bool:
    mapped = image_to_field(h, [image_point])

    if (
        mapped.get("status") != "ok"
        or mapped.get("valid") != [True]
    ):
        return False

    point = np.asarray(mapped["points"][0], dtype=np.float64)

    if not np.isfinite(point).all():
        return False

    if not _inside_pitch_centered(
        point,
        pitch_length_m=pitch_length_m,
        pitch_width_m=pitch_width_m,
        tolerance_m=tolerance_m,
    ):
        return False

    _projected, denominator, valid = _project_homogeneous(
        h,
        point.reshape(1, 2),
    )

    return bool(
        valid[0]
        and int(np.sign(float(denominator[0])))
        == int(branch_sign)
    )


def _ray_support(
    h: np.ndarray,
    *,
    branch_sign: int,
    image_width: int,
    image_height: int,
    pitch_length_m: float,
    pitch_width_m: float,
    config: FieldSpaceObserverConfig,
) -> dict[str, Any]:
    image_points = np.asarray(
        [
            [x * image_width, y * image_height]
            for y in config.support_y_fracs
            for x in config.support_x_fracs
        ],
        dtype=np.float64,
    )

    mapped = image_to_field(h, image_points)

    if mapped.get("status") != "ok":
        return {
            "inside_count": 0,
            "sample_count": len(image_points),
            "fraction": 0.0,
        }

    field_points = np.asarray(mapped["points"], dtype=np.float64)
    valid = np.asarray(mapped["valid"], dtype=bool)

    inside_count = 0

    for point, is_valid in zip(field_points, valid):
        if not is_valid or not np.isfinite(point).all():
            continue

        if not _inside_pitch_centered(
            point,
            pitch_length_m=pitch_length_m,
            pitch_width_m=pitch_width_m,
            tolerance_m=config.pitch_tolerance_m,
        ):
            continue

        _projected, denominator, forward_valid = (
            _project_homogeneous(
                h,
                point.reshape(1, 2),
            )
        )

        if not bool(forward_valid[0]):
            continue

        if int(np.sign(float(denominator[0]))) == int(branch_sign):
            inside_count += 1

    sample_count = len(image_points)

    return {
        "inside_count": int(inside_count),
        "sample_count": int(sample_count),
        "fraction": float(
            inside_count / max(sample_count, 1)
        ),
    }


def _candidate_for_branch(
    h: np.ndarray,
    *,
    branch_sign: int,
    image_width: int,
    image_height: int,
    pitch_length_m: float,
    pitch_width_m: float,
    config: FieldSpaceObserverConfig,
) -> dict[str, Any]:
    half_l = 0.5 * pitch_length_m
    half_w = 0.5 * pitch_width_m

    halfplanes = _viewport_halfplanes(
        h,
        image_width=image_width,
        image_height=image_height,
        branch_sign=branch_sign,
    )

    polygon: list[np.ndarray] = [
        np.asarray([-half_l, -half_w], dtype=np.float64),
        np.asarray([half_l, -half_w], dtype=np.float64),
        np.asarray([half_l, half_w], dtype=np.float64),
        np.asarray([-half_l, half_w], dtype=np.float64),
    ]

    for line in halfplanes:
        polygon = _clip_polygon_halfplane(polygon, line)
        if not polygon:
            break

    poly = (
        np.asarray(polygon, dtype=np.float64).reshape(-1, 2)
        if polygon
        else np.empty((0, 2), dtype=np.float64)
    )

    area = _polygon_area(poly)
    pitch_area = float(pitch_length_m * pitch_width_m)
    field_fraction = float(
        max(
            0.0,
            min(
                1.0,
                area / max(pitch_area, 1.0),
            ),
        )
    )

    centroid = _polygon_centroid(poly)

    bounds_centered = None
    bounds_pitch = None

    if len(poly):
        bounds_centered = {
            "min_x_m": float(np.min(poly[:, 0])),
            "max_x_m": float(np.max(poly[:, 0])),
            "min_y_m": float(np.min(poly[:, 1])),
            "max_y_m": float(np.max(poly[:, 1])),
        }
        bounds_pitch = {
            "min_x_m": float(np.min(poly[:, 0]) + half_l),
            "max_x_m": float(np.max(poly[:, 0]) + half_l),
            "min_y_m": float(np.min(poly[:, 1]) + half_w),
            "max_y_m": float(np.max(poly[:, 1]) + half_w),
        }

    image_metrics = _image_polygon_metrics(
        h,
        poly,
        image_width=image_width,
        image_height=image_height,
    )

    support = _ray_support(
        h,
        branch_sign=branch_sign,
        image_width=image_width,
        image_height=image_height,
        pitch_length_m=pitch_length_m,
        pitch_width_m=pitch_width_m,
        config=config,
    )

    image_centre_inside = _single_image_probe_inside(
        h,
        image_point=(
            0.5 * image_width,
            0.5 * image_height,
        ),
        branch_sign=branch_sign,
        pitch_length_m=pitch_length_m,
        pitch_width_m=pitch_width_m,
        tolerance_m=config.pitch_tolerance_m,
    )

    bottom_centre_inside = _single_image_probe_inside(
        h,
        image_point=(
            0.5 * image_width,
            0.88 * image_height,
        ),
        branch_sign=branch_sign,
        pitch_length_m=pitch_length_m,
        pitch_width_m=pitch_width_m,
        tolerance_m=config.pitch_tolerance_m,
    )

    boundaries = _boundary_visibility(
        halfplanes,
        pitch_length_m=pitch_length_m,
        pitch_width_m=pitch_width_m,
    )

    emission = (
        config.emission_lower_support_weight
        * float(support["fraction"])
        + config.emission_image_area_weight
        * float(image_metrics["fraction"])
        + config.emission_bottom_centre_weight
        * float(bottom_centre_inside)
        + config.emission_image_centre_weight
        * float(image_centre_inside)
    )

    if area <= EPS:
        emission -= config.transition_empty_penalty

    polygon_pitch = [
        [
            float(point[0] + half_l),
            float(point[1] + half_w),
        ]
        for point in poly
    ]

    centroid_centered = None
    centroid_pitch = None

    if centroid is not None:
        centroid_centered = {
            "x_m": float(centroid[0]),
            "y_m": float(centroid[1]),
        }
        centroid_pitch = {
            "x_m": float(centroid[0] + half_l),
            "y_m": float(centroid[1] + half_w),
        }

    return {
        "branch_sign": int(branch_sign),
        "visible_pitch_polygon_centered": poly.tolist(),
        "visible_pitch_polygon": polygon_pitch,
        "visible_pitch_area_m2": float(area),
        "visible_fraction": field_fraction,
        "visible_bounds_centered_m": bounds_centered,
        "visible_bounds_m": bounds_pitch,
        "visible_centroid_field_centered": centroid_centered,
        "visible_centroid_field": centroid_pitch,
        "boundary_visibility": boundaries,
        "image_footprint": image_metrics,
        "ray_support": support,
        "image_centre_inside_pitch": bool(image_centre_inside),
        "bottom_centre_inside_pitch": bool(bottom_centre_inside),
        "emission_score": float(emission),
    }


def analyse_field_space_candidates(
    homography: Any,
    *,
    image_width: int,
    image_height: int,
    pitch_length_m: float,
    pitch_width_m: float,
    config: FieldSpaceObserverConfig | None = None,
) -> dict[str, Any]:
    cfg = config or FieldSpaceObserverConfig()

    h = _normalise_h(homography)
    if h is None:
        return {"status": "invalid_homography"}

    if (
        image_width <= 0
        or image_height <= 0
        or pitch_length_m <= 0
        or pitch_width_m <= 0
    ):
        return {"status": "invalid_dimensions"}

    optical_map = image_to_field(
        h,
        [[0.5 * image_width, 0.5 * image_height]],
    )

    optical_centered = None
    optical_pitch = None

    if (
        optical_map.get("status") == "ok"
        and optical_map.get("valid") == [True]
    ):
        point = np.asarray(
            optical_map["points"][0],
            dtype=np.float64,
        )

        if np.isfinite(point).all():
            optical_centered = {
                "x_m": float(point[0]),
                "y_m": float(point[1]),
                "inside_pitch": _inside_pitch_centered(
                    point,
                    pitch_length_m=pitch_length_m,
                    pitch_width_m=pitch_width_m,
                    tolerance_m=cfg.pitch_tolerance_m,
                ),
            }
            optical_pitch = {
                "x_m": float(point[0] + 0.5 * pitch_length_m),
                "y_m": float(point[1] + 0.5 * pitch_width_m),
                "inside_pitch": optical_centered["inside_pitch"],
            }

    return {
        "status": "ok",
        "version": 1,
        "algorithm": "field_space_observer_v1",
        "implementation": "FSO04_centered_pitch_two_branch",
        "coordinate_system": {
            "canonical": "pitch_centered",
            "x_range_m": [
                -0.5 * pitch_length_m,
                0.5 * pitch_length_m,
            ],
            "y_range_m": [
                -0.5 * pitch_width_m,
                0.5 * pitch_width_m,
            ],
            "ui_pitch_range_x_m": [0.0, pitch_length_m],
            "ui_pitch_range_y_m": [0.0, pitch_width_m],
        },
        "optical_axis_ground_point_centered": optical_centered,
        "optical_axis_ground_point": optical_pitch,
        "candidates": {
            "+1": _candidate_for_branch(
                h,
                branch_sign=1,
                image_width=image_width,
                image_height=image_height,
                pitch_length_m=pitch_length_m,
                pitch_width_m=pitch_width_m,
                config=cfg,
            ),
            "-1": _candidate_for_branch(
                h,
                branch_sign=-1,
                image_width=image_width,
                image_height=image_height,
                pitch_length_m=pitch_length_m,
                pitch_width_m=pitch_width_m,
                config=cfg,
            ),
        },
        "config": asdict(cfg),
    }


def _candidate_centroid_centered(
    candidate: dict[str, Any],
) -> np.ndarray | None:
    value = candidate.get(
        "visible_centroid_field_centered"
    )

    if not isinstance(value, dict):
        return None

    try:
        point = np.asarray(
            [value["x_m"], value["y_m"]],
            dtype=np.float64,
        )
    except (KeyError, TypeError, ValueError):
        return None

    return point if np.isfinite(point).all() else None


def _transition_score(
    previous: dict[str, Any],
    current: dict[str, Any],
    *,
    config: FieldSpaceObserverConfig,
) -> float:
    score = 0.0

    p_centroid = _candidate_centroid_centered(previous)
    c_centroid = _candidate_centroid_centered(current)

    if p_centroid is None or c_centroid is None:
        score -= config.transition_empty_penalty
    else:
        distance = float(
            np.linalg.norm(c_centroid - p_centroid)
        )
        score -= (
            config.transition_centroid_m_weight
            * distance
        )

    p_fraction = float(
        previous.get("visible_fraction") or 0.0
    )
    c_fraction = float(
        current.get("visible_fraction") or 0.0
    )

    score -= (
        config.transition_fraction_weight
        * abs(c_fraction - p_fraction)
    )

    return float(score)


def _x_zone_from_pitch_x(
    x_m: float,
    pitch_length_m: float,
) -> str:
    ratio = float(x_m) / float(pitch_length_m)

    if ratio < 1.0 / 3.0:
        return "left_third"
    if ratio < 2.0 / 3.0:
        return "centre_third"
    return "right_third"


def _y_zone_from_pitch_y(
    y_m: float,
    pitch_width_m: float,
) -> str:
    ratio = float(y_m) / float(pitch_width_m)

    if ratio < 1.0 / 3.0:
        return "top_third"
    if ratio < 2.0 / 3.0:
        return "middle_third"
    return "bottom_third"


def _selected_record(
    frame_result: dict[str, Any],
    candidate: dict[str, Any],
    *,
    sequence_score: float,
    alternative_candidate: dict[str, Any],
) -> dict[str, Any]:
    focus_pitch = candidate.get(
        "visible_centroid_field"
    )

    semantic = {
        "focus_x_zone": None,
        "focus_y_zone": None,
        "left_goal_line_visible": bool(
            candidate["boundary_visibility"][
                "left_goal_line"
            ]["visible"]
        ),
        "right_goal_line_visible": bool(
            candidate["boundary_visibility"][
                "right_goal_line"
            ]["visible"]
        ),
        "top_touchline_visible": bool(
            candidate["boundary_visibility"][
                "top_touchline"
            ]["visible"]
        ),
        "bottom_touchline_visible": bool(
            candidate["boundary_visibility"][
                "bottom_touchline"
            ]["visible"]
        ),
    }

    if isinstance(focus_pitch, dict):
        semantic["focus_x_zone"] = _x_zone_from_pitch_x(
            float(focus_pitch["x_m"]),
            float(frame_result["pitch_length_m"]),
        )
        semantic["focus_y_zone"] = _y_zone_from_pitch_y(
            float(focus_pitch["y_m"]),
            float(frame_result["pitch_width_m"]),
        )

    return {
        "status": "ok",
        "branch_sign": int(candidate["branch_sign"]),
        "branch_sequence_score": float(sequence_score),
        "branch_emission_score": float(
            candidate["emission_score"]
        ),
        "alternative_branch_emission_score": float(
            alternative_candidate["emission_score"]
        ),
        "branch_emission_margin": float(
            candidate["emission_score"]
            - alternative_candidate["emission_score"]
        ),
        "coordinate_system": frame_result[
            "coordinate_system"
        ],
        "optical_axis_ground_point_centered": (
            frame_result.get(
                "optical_axis_ground_point_centered"
            )
        ),
        "optical_axis_ground_point": frame_result.get(
            "optical_axis_ground_point"
        ),
        "visible_focus_field_centered": candidate.get(
            "visible_centroid_field_centered"
        ),
        "visible_focus_field": focus_pitch,
        "visible_pitch_polygon_centered": candidate[
            "visible_pitch_polygon_centered"
        ],
        "visible_pitch_polygon": candidate[
            "visible_pitch_polygon"
        ],
        "visible_pitch_area_m2": candidate[
            "visible_pitch_area_m2"
        ],
        "visible_fraction": candidate[
            "visible_fraction"
        ],
        "visible_bounds_centered_m": candidate[
            "visible_bounds_centered_m"
        ],
        "visible_bounds_m": candidate[
            "visible_bounds_m"
        ],
        "boundary_visibility": candidate[
            "boundary_visibility"
        ],
        "image_footprint": candidate[
            "image_footprint"
        ],
        "ray_support": candidate["ray_support"],
        "semantic": semantic,
    }


def select_field_space_observer_sequence(
    homographies: list[Any],
    *,
    image_width: int,
    image_height: int,
    pitch_length_m: float,
    pitch_width_m: float,
    config: FieldSpaceObserverConfig | None = None,
) -> dict[str, Any]:
    """Select one physical projective branch per frame using Viterbi."""

    cfg = config or FieldSpaceObserverConfig()

    analysed: list[dict[str, Any]] = []

    for frame, homography in enumerate(homographies):
        result = analyse_field_space_candidates(
            homography,
            image_width=image_width,
            image_height=image_height,
            pitch_length_m=pitch_length_m,
            pitch_width_m=pitch_width_m,
            config=cfg,
        )

        if result.get("status") != "ok":
            return {
                "status": "candidate_analysis_failed",
                "frame": frame,
                "failure": result,
            }

        analysed.append(
            {
                **result,
                "frame": int(frame),
                "pitch_length_m": float(pitch_length_m),
                "pitch_width_m": float(pitch_width_m),
            }
        )

    if not analysed:
        return {
            "status": "ok",
            "records": [],
            "candidate_frames": [],
        }

    states = ("+1", "-1")
    n = len(analysed)

    dp = np.full(
        (n, 2),
        -np.inf,
        dtype=np.float64,
    )
    parent = np.full(
        (n, 2),
        -1,
        dtype=np.int8,
    )

    for state_index, state in enumerate(states):
        candidate = analysed[0]["candidates"][state]
        dp[0, state_index] = float(
            candidate["emission_score"]
        )

    for frame in range(1, n):
        for current_index, current_state in enumerate(states):
            current = analysed[frame]["candidates"][
                current_state
            ]
            emission = float(
                current["emission_score"]
            )

            best_score = -np.inf
            best_parent = -1

            for previous_index, previous_state in enumerate(states):
                previous = analysed[frame - 1]["candidates"][
                    previous_state
                ]

                score = (
                    dp[frame - 1, previous_index]
                    + _transition_score(
                        previous,
                        current,
                        config=cfg,
                    )
                    + emission
                )

                if score > best_score:
                    best_score = score
                    best_parent = previous_index

            dp[frame, current_index] = best_score
            parent[frame, current_index] = best_parent

    selected_indices = np.zeros(
        n,
        dtype=np.int8,
    )
    selected_indices[-1] = int(
        np.argmax(dp[-1])
    )

    for frame in range(n - 1, 0, -1):
        selected_indices[frame - 1] = parent[
            frame,
            selected_indices[frame],
        ]

    records = []

    for frame, state_index in enumerate(selected_indices):
        state = states[int(state_index)]
        alternative_state = states[
            1 - int(state_index)
        ]

        candidate = analysed[frame]["candidates"][
            state
        ]
        alternative = analysed[frame]["candidates"][
            alternative_state
        ]

        record = _selected_record(
            analysed[frame],
            candidate,
            sequence_score=float(
                dp[frame, int(state_index)]
            ),
            alternative_candidate=alternative,
        )
        record["frame"] = int(frame)
        records.append(record)

    switches = sum(
        1
        for left, right in zip(
            selected_indices[:-1],
            selected_indices[1:],
        )
        if int(left) != int(right)
    )

    return {
        "status": "ok",
        "version": 1,
        "algorithm": "field_space_observer_v1",
        "implementation": (
            "FSO04_centered_pitch_two_branch_temporal_viterbi"
        ),
        "record_count": len(records),
        "branch_switch_count": int(switches),
        "records": records,
        "config": asdict(cfg),
    }
