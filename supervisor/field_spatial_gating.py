"""Spatial gating for FIELD adaptation guidance.

Upgrades broad semantic guidance into finite search geometry. A football marking
is a finite segment, not an infinite image line. The module also adds the
projected 105x68 pitch polygon so later visual observation can reject stands,
advertising and railings.

It does not change online-adaptation math; it only constrains where visual
evidence is allowed to exist.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import math
from typing import Any

import numpy as np


PITCH_CORNERS = np.asarray(
    [(-52.5, -34.0), (52.5, -34.0), (52.5, 34.0), (-52.5, 34.0)],
    dtype=np.float64,
)

LINES = {
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


@dataclass(frozen=True)
class FieldSpatialGatingConfig:
    longitudinal_margin_ratio: float = 0.12
    min_longitudinal_margin_px: float = 10.0
    max_longitudinal_margin_px: float = 60.0
    pitch_region_margin_px: float = 24.0
    max_coordinate_extent_factor: float = 8.0


def _as_h(matrix: Any) -> np.ndarray | None:
    try:
        h = np.asarray(matrix, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if h.shape != (3, 3) or not np.isfinite(h).all() or abs(float(h[2, 2])) < 1e-12:
        return None
    h = h / h[2, 2]
    try:
        if abs(float(np.linalg.det(h))) < 1e-12:
            return None
    except np.linalg.LinAlgError:
        return None
    return h


def _project_points(world_xy: np.ndarray, h: np.ndarray):
    points = np.asarray(world_xy, dtype=np.float64).reshape(-1, 2)
    homogeneous = np.column_stack((points, np.ones(len(points))))
    q = homogeneous @ h.T
    d = q[:, 2]
    valid = np.isfinite(q).all(axis=1) & np.isfinite(d) & (np.abs(d) > 1e-9)
    out = np.full((len(points), 2), np.nan)
    out[valid] = q[valid, :2] / d[valid, None]
    valid &= np.isfinite(out).all(axis=1)
    return out, valid


def _normalise_line(line: Any) -> np.ndarray | None:
    try:
        arr = np.asarray(line, dtype=np.float64).reshape(3)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(arr).all():
        return None
    n = math.hypot(float(arr[0]), float(arr[1]))
    return arr / n if n > 1e-9 else None


def _project_onto_line(point: np.ndarray, line: np.ndarray) -> np.ndarray:
    d = float(line[0] * point[0] + line[1] * point[1] + line[2])
    return point - d * line[:2]


def _clip_extreme(point: np.ndarray, width: int, height: int, factor: float):
    return np.asarray([
        np.clip(point[0], -width * factor, width * factor),
        np.clip(point[1], -height * factor, height * factor),
    ], dtype=np.float64)


def _json_point(point: np.ndarray):
    return {"x": float(point[0]), "y": float(point[1])}


def _finite_geometry(semantic, item, h, width, height, cfg):
    world = LINES.get(semantic)
    if world is None:
        return None
    projected, valid = _project_points(np.asarray(world), h)
    if not np.all(valid):
        return None

    expected_line = _normalise_line(
        item.get("expected", {}).get("line")
        or item.get("predicted", {}).get("line")
    )
    if expected_line is None:
        return None

    start = _project_onto_line(projected[0], expected_line)
    end = _project_onto_line(projected[1], expected_line)
    delta = end - start
    length = float(np.linalg.norm(delta))
    if length < 2.0:
        return None

    tangent = delta / length
    margin = float(np.clip(
        length * cfg.longitudinal_margin_ratio,
        cfg.min_longitudinal_margin_px,
        cfg.max_longitudinal_margin_px,
    ))
    search_start = start - tangent * margin
    search_end = end + tangent * margin

    start = _clip_extreme(start, width, height, cfg.max_coordinate_extent_factor)
    end = _clip_extreme(end, width, height, cfg.max_coordinate_extent_factor)
    search_start = _clip_extreme(search_start, width, height, cfg.max_coordinate_extent_factor)
    search_end = _clip_extreme(search_end, width, height, cfg.max_coordinate_extent_factor)

    return {
        "semantic": semantic,
        "physical_start": _json_point(start),
        "physical_end": _json_point(end),
        "search_start": _json_point(search_start),
        "search_end": _json_point(search_end),
        "physical_length_px": length,
        "longitudinal_margin_px": margin,
    }


def apply_field_spatial_gating(
    guidance: dict[str, Any],
    current_homography: Any,
    image_width: int,
    image_height: int,
    *,
    config: FieldSpatialGatingConfig | None = None,
) -> dict[str, Any]:
    cfg = config or FieldSpatialGatingConfig()
    output = deepcopy(guidance)
    h = _as_h(current_homography)
    width, height = int(image_width), int(image_height)

    if h is None or width <= 0 or height <= 0:
        output["spatial_gating"] = {"status": "unavailable"}
        return output

    projected_pitch, valid_pitch = _project_points(PITCH_CORNERS, h)
    pitch_region = None
    if np.all(valid_pitch):
        pitch_region = {
            "polygon": [
                _json_point(_clip_extreme(p, width, height, cfg.max_coordinate_extent_factor))
                for p in projected_pitch
            ],
            "margin_px": float(cfg.pitch_region_margin_px),
        }
        output["pitch_region"] = pitch_region

    gated = 0
    for item in output.get("searches", []):
        if not isinstance(item, dict) or item.get("type") != "line":
            continue
        finite = _finite_geometry(
            str(item.get("semantic", "")), item, h, width, height, cfg
        )
        if finite is None:
            continue

        search_data = dict(item.get("search") or {})
        search_data["kind"] = "finite_line_capsule"
        search_data["finite_segment"] = finite
        item["search"] = search_data

        expected = dict(item.get("expected") or {})
        expected["visible_start"] = finite["physical_start"]
        expected["visible_end"] = finite["physical_end"]
        item["expected"] = expected

        if pitch_region is not None:
            item["pitch_region"] = pitch_region
        gated += 1

    output["spatial_gating"] = {
        "status": "ok",
        "mode": "finite_semantic_segment_v1",
        "gated_lines": gated,
        "pitch_region_available": pitch_region is not None,
        "config": asdict(cfg),
    }
    return output