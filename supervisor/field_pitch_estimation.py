"""Multi-frame physical pitch-dimension estimation from HUMAN constraints.

Each observation frame owns its own world-to-image homography.  The only
shared variables are the total physical pitch length and width.  Semantic
constraint IDs are deliberately resolved against every candidate pitch model;
stored ``constraint.world`` coordinates are never used for estimation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import copy
import math
from typing import Any, Iterable

import numpy as np

from supervisor.field_camera_calibration import solve_field_camera_calibration
from supervisor.field_pitch_model import build_pitch_model


SUCCESS_STATUSES = {"solved", "solved_with_warnings"}
LINE_HOLDOUT_SEMANTICS = {
    "bottom_touchline",
    "top_touchline",
    "halfway_line",
    "left_goal_line",
    "right_goal_line",
}


@dataclass(frozen=True)
class PitchEstimationConfig:
    """Configurable deterministic dimensions search bounds and steps."""

    length_min_m: float = 100.0
    length_max_m: float = 110.0
    width_min_m: float = 64.0
    width_max_m: float = 75.0
    coarse_step_m: float = 1.0
    fine_radius_m: float = 1.0
    fine_step_m: float = 0.1
    near_best_ratio: float = 0.05
    numerical_score_tolerance: float = 1e-6


def _grid(start: float, stop: float, step: float) -> list[float]:
    if not all(math.isfinite(value) for value in (start, stop, step)) or step <= 0.0 or stop < start:
        raise ValueError("Invalid pitch-dimension search bounds")
    count = int(math.floor((stop - start) / step + 1e-9))
    values = [round(start + index * step, 10) for index in range(count + 1)]
    if not values or values[-1] < stop - 1e-9:
        values.append(float(stop))
    return values


def _xy(value: Any) -> dict[str, float] | None:
    try:
        x, y = float(value["x"]), float(value["y"])
    except (KeyError, TypeError, ValueError):
        return None
    if not math.isfinite(x) or not math.isfinite(y):
        return None
    return {"x": x, "y": y}


def _line_image(value: Any) -> dict[str, dict[str, float]] | None:
    if not isinstance(value, dict):
        return None
    start, end = _xy(value.get("start")), _xy(value.get("end"))
    if start is None or end is None:
        return None
    if math.hypot(end["x"] - start["x"], end["y"] - start["y"]) < 1e-9:
        return None
    return {"start": start, "end": end}


def _arc_image(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict) or not isinstance(value.get("samples"), list):
        return None
    samples = [point for raw in value["samples"] if (point := _xy(raw)) is not None]
    return {"samples": samples, "geometry": copy.deepcopy(value.get("geometry"))} if len(samples) >= 3 else None


def _semantic_maps(model: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    return (
        {item["id"]: item for item in model["landmarks"]},
        {item["id"]: item for item in model["calibration_lines"]},
        {item["id"]: item for item in model["arcs"]},
    )


def _rebuild_constraints(
    constraints: Iterable[dict[str, Any]],
    model: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Rebuild semantic world geometry while preserving only human image data."""
    landmarks, lines, arcs = _semantic_maps(model)
    rebuilt: list[dict[str, Any]] = []
    excluded: list[dict[str, str]] = []

    for raw in constraints:
        if not isinstance(raw, dict):
            excluded.append({"semantic": "", "reason": "invalid_constraint"})
            continue
        kind, semantic = raw.get("type"), raw.get("semantic")
        if not isinstance(semantic, str):
            excluded.append({"semantic": str(semantic or ""), "reason": "missing_semantic"})
            continue

        if kind == "point":
            definition, image = landmarks.get(semantic), _xy(raw.get("image"))
        elif kind == "line":
            definition, image = lines.get(semantic), _line_image(raw.get("image"))
        elif kind == "arc":
            definition, image = arcs.get(semantic), _arc_image(raw.get("image"))
        else:
            excluded.append({"semantic": semantic, "reason": "unsupported_constraint_type"})
            continue

        if definition is None:
            excluded.append({"semantic": semantic, "reason": "unknown_semantic_for_pitch_model"})
            continue
        if image is None:
            excluded.append({"semantic": semantic, "reason": "invalid_image_geometry"})
            continue
        rebuilt.append({
            "type": kind,
            "semantic": semantic,
            "world": copy.deepcopy(definition["world"]),
            "image": image,
            "source": raw.get("source", "manual"),
        })

    return rebuilt, excluded


def _split_constraints(constraints: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    return (
        [item for item in constraints if item["type"] == "point"],
        [item for item in constraints if item["type"] == "line"],
        [item for item in constraints if item["type"] == "arc"],
    )


def _normalised_frame_score(result: dict[str, Any]) -> float | None:
    diagnostics = result.get("diagnostics", {})
    try:
        cost = float(diagnostics["score"])
        points, lines, arcs = int(diagnostics["points"]), int(diagnostics["lines"]), int(diagnostics["arcs"])
    except (KeyError, TypeError, ValueError):
        return None
    if not math.isfinite(cost) or cost < 0.0:
        return None
    # Solver cost is half the squared, uncertainty-normalised residual.  This
    # converts it to a per-frame RMS-like score without weighting a frame by
    # the number of annotations it happened to contain.
    support = max(1, points + 2 * lines + arcs)
    return math.sqrt(2.0 * cost / support)


def _solve_frame(observation: dict[str, Any], model: dict[str, Any]) -> dict[str, Any]:
    rebuilt, excluded = _rebuild_constraints(observation.get("constraints", []), model)
    points, lines, arcs = _split_constraints(rebuilt)
    result = solve_field_camera_calibration(
        points,
        lines,
        arcs,
        observation.get("image_width"),
        observation.get("image_height"),
    )
    score = _normalised_frame_score(result) if result.get("status") in SUCCESS_STATUSES else None
    return {
        "frame": observation.get("frame"),
        "constraints": rebuilt,
        "excluded": excluded,
        "solver": result,
        "normalised_score": score,
    }


def _candidate_result(observations: list[dict[str, Any]], length_m: float, width_m: float) -> dict[str, Any]:
    model = build_pitch_model(length_m, width_m)
    frame_results = [_solve_frame(observation, model) for observation in observations]
    scores = [item["normalised_score"] for item in frame_results if item["normalised_score"] is not None]
    return {
        "length_m": length_m,
        "width_m": width_m,
        "score": float(np.median(scores)) if scores else None,
        "usable_frames": len(scores),
        "frame_results": frame_results,
    }


def _surface_entry(candidate: dict[str, Any], stage: str) -> dict[str, Any]:
    return {
        "length_m": candidate["length_m"],
        "width_m": candidate["width_m"],
        "score": candidate["score"],
        "usable_frames": candidate["usable_frames"],
        "stage": stage,
    }


def _valid_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    finite = [item for item in candidates if item["score"] is not None and math.isfinite(item["score"])]
    return min(finite, key=lambda item: item["score"]) if finite else None


def _line_from_points(start: dict[str, float], end: dict[str, float]) -> np.ndarray | None:
    p0 = np.array([start["x"], start["y"], 1.0], dtype=np.float64)
    p1 = np.array([end["x"], end["y"], 1.0], dtype=np.float64)
    line = np.cross(p0, p1)
    scale = math.hypot(float(line[0]), float(line[1]))
    return line / scale if scale > 1e-12 and np.isfinite(line).all() else None


def _project_point(world: dict[str, float], homography: list[list[float]]) -> dict[str, float] | None:
    try:
        h = np.asarray(homography, dtype=np.float64)
        point = np.array([world["x"], world["y"], 1.0], dtype=np.float64)
        projected = h @ point
    except (KeyError, TypeError, ValueError):
        return None
    if h.shape != (3, 3) or not np.isfinite(projected).all() or abs(float(projected[2])) < 1e-12:
        return None
    return {"x": float(projected[0] / projected[2]), "y": float(projected[1] / projected[2])}


def _held_out_error(constraint: dict[str, Any], homography: list[list[float]]) -> float | None:
    if constraint["type"] == "point":
        predicted = _project_point(constraint["world"], homography)
        if predicted is None:
            return None
        return math.hypot(predicted["x"] - constraint["image"]["x"], predicted["y"] - constraint["image"]["y"])
    if constraint["type"] != "line":
        return None
    world_line = _line_from_points(constraint["world"]["start"], constraint["world"]["end"])
    if world_line is None:
        return None
    try:
        h = np.asarray(homography, dtype=np.float64)
        image_line = np.linalg.inv(h).T @ world_line
    except (TypeError, ValueError, np.linalg.LinAlgError):
        return None
    norm = math.hypot(float(image_line[0]), float(image_line[1]))
    if norm < 1e-12 or not np.isfinite(image_line).all():
        return None
    image_line /= norm
    errors = [
        abs(float(image_line[0] * endpoint["x"] + image_line[1] * endpoint["y"] + image_line[2]))
        for endpoint in (constraint["image"]["start"], constraint["image"]["end"])
    ]
    return float(np.mean(errors))


def _holdouts(observation: dict[str, Any], model: dict[str, Any]) -> list[dict[str, Any]]:
    rebuilt, _ = _rebuild_constraints(observation.get("constraints", []), model)
    results: list[dict[str, Any]] = []
    for index, held_out in enumerate(rebuilt):
        remaining = rebuilt[:index] + rebuilt[index + 1:]
        points, lines, arcs = _split_constraints(remaining)
        solved = solve_field_camera_calibration(points, lines, arcs, observation.get("image_width"), observation.get("image_height"))
        detail: dict[str, Any] = {
            "semantic": held_out["semantic"],
            "type": held_out["type"],
            "status": "not_solved",
            "error_px": None,
        }
        if solved.get("status") in SUCCESS_STATUSES:
            error = _held_out_error(held_out, solved["homography"])
            detail["status"] = "ok" if error is not None else "unsupported_geometry"
            detail["error_px"] = error
        results.append(detail)
    return results


def _frame_observability(observation: dict[str, Any], model: dict[str, Any]) -> dict[str, Any]:
    constraints, excluded = _rebuild_constraints(observation.get("constraints", []), model)
    world_points: list[tuple[float, float]] = []
    semantics: list[str] = []
    counts = {"points": 0, "lines": 0, "arcs": 0}
    for item in constraints:
        counts[f"{item['type']}s"] += 1
        semantics.append(item["semantic"])
        if item["type"] == "point":
            world_points.append((item["world"]["x"], item["world"]["y"]))
        elif item["type"] == "line":
            world_points.extend(((item["world"]["start"]["x"], item["world"]["start"]["y"]), (item["world"]["end"]["x"], item["world"]["end"]["y"])))
        else:
            center = item["world"]["center"]
            world_points.append((center["x"], center["y"]))
    if world_points:
        world = np.asarray(world_points, dtype=np.float64)
        span_x, span_y = float(np.ptp(world[:, 0])), float(np.ptp(world[:, 1]))
    else:
        span_x = span_y = 0.0
    return {
        "frame": observation.get("frame"),
        "semantic_point_count": counts["points"],
        "semantic_line_count": counts["lines"],
        "semantic_arc_count": counts["arcs"],
        "world_x_span_m": span_x,
        "world_y_span_m": span_y,
        "reaches_touchline": any("touchline" in semantic for semantic in semantics),
        "reaches_goal_line_structure": any(
            semantic in {"left_goal_line", "right_goal_line"}
            or "goal_line" in semantic
            or "goal_post" in semantic
            for semantic in semantics
        ),
        "includes_centre_or_halfway": any("centre" in semantic or "halfway" in semantic for semantic in semantics),
        "excluded_semantics": excluded,
    }


def _summarise_holdouts(frames: list[dict[str, Any]]) -> dict[str, Any]:
    by_semantic: dict[str, list[float]] = {}
    point_errors: list[float] = []
    line_errors: list[float] = []
    for frame in frames:
        for item in frame["semantic_holdouts"]:
            error = item.get("error_px")
            if not isinstance(error, (int, float)) or not math.isfinite(error):
                continue
            by_semantic.setdefault(item["semantic"], []).append(float(error))
            (point_errors if item["type"] == "point" else line_errors).append(float(error))

    semantic = {
        name: {"count": len(errors), "median_error_px": float(np.median(errors)), "max_error_px": float(np.max(errors))}
        for name, errors in sorted(by_semantic.items())
    }
    return {
        "point_median_error_px": float(np.median(point_errors)) if point_errors else None,
        "line_median_error_px": float(np.median(line_errors)) if line_errors else None,
        "semantics": semantic,
        "key_line_holdouts": {name: semantic[name] for name in sorted(LINE_HOLDOUT_SEMANTICS & set(semantic))},
    }


def estimate_pitch_dimensions(
    observations: list[dict[str, Any]],
    *,
    config: PitchEstimationConfig | None = None,
) -> dict[str, Any]:
    """Estimate global pitch dimensions from independent HUMAN camera solves."""
    cfg = config or PitchEstimationConfig()
    observations = [item for item in observations if isinstance(item, dict)]
    if not observations:
        return {"status": "no_observations", "detail": "No se recibieron observaciones manuales.", "frames": [], "search": asdict(cfg)}

    coarse: list[dict[str, Any]] = []
    surface: list[dict[str, Any]] = []
    for length_m in _grid(cfg.length_min_m, cfg.length_max_m, cfg.coarse_step_m):
        for width_m in _grid(cfg.width_min_m, cfg.width_max_m, cfg.coarse_step_m):
            candidate = _candidate_result(observations, length_m, width_m)
            coarse.append(candidate)
            surface.append(_surface_entry(candidate, "coarse"))
    coarse_best = _valid_candidate(coarse)
    if coarse_best is None:
        return {
            "status": "no_usable_frames",
            "detail": "Ningún fotograma pudo resolver una calibración geométrica estable.",
            "frames": [],
            "search": {**asdict(cfg), "score_surface": surface},
        }

    fine_lengths = _grid(max(cfg.length_min_m, coarse_best["length_m"] - cfg.fine_radius_m), min(cfg.length_max_m, coarse_best["length_m"] + cfg.fine_radius_m), cfg.fine_step_m)
    fine_widths = _grid(max(cfg.width_min_m, coarse_best["width_m"] - cfg.fine_radius_m), min(cfg.width_max_m, coarse_best["width_m"] + cfg.fine_radius_m), cfg.fine_step_m)
    fine: list[dict[str, Any]] = []
    for length_m in fine_lengths:
        for width_m in fine_widths:
            candidate = _candidate_result(observations, length_m, width_m)
            fine.append(candidate)
            surface.append(_surface_entry(candidate, "fine"))
    best = _valid_candidate(fine) or coarse_best
    assert best is not None
    best_model = build_pitch_model(best["length_m"], best["width_m"])

    threshold = best["score"] * (1.0 + cfg.near_best_ratio) + cfg.numerical_score_tolerance
    near_best = [item for item in fine + coarse if item["score"] is not None and item["score"] <= threshold]
    length_interval = [float(min(item["length_m"] for item in near_best)), float(max(item["length_m"] for item in near_best))]
    width_interval = [float(min(item["width_m"] for item in near_best)), float(max(item["width_m"] for item in near_best))]
    boundary_limited = {
        "length": bool(abs(best["length_m"] - cfg.length_min_m) < 1e-9 or abs(best["length_m"] - cfg.length_max_m) < 1e-9),
        "width": bool(abs(best["width_m"] - cfg.width_min_m) < 1e-9 or abs(best["width_m"] - cfg.width_max_m) < 1e-9),
    }

    observability = [_frame_observability(observation, best_model) for observation in observations]
    total_counts = {
        key: sum(item[f"semantic_{key[:-1]}_count"] for item in observability)
        for key in ("points", "lines", "arcs")
    }
    reaches_touchline = any(item["reaches_touchline"] for item in observability)
    reaches_goal = any(item["reaches_goal_line_structure"] for item in observability)
    includes_centre = any(item["includes_centre_or_halfway"] for item in observability)
    max_x_span = max((item["world_x_span_m"] for item in observability), default=0.0)
    max_y_span = max((item["world_y_span_m"] for item in observability), default=0.0)
    length_constrained = not boundary_limited["length"] and length_interval[1] - length_interval[0] <= 1.0 and max_x_span >= best_model["length_m"] * 0.15 and (reaches_goal or includes_centre)
    width_constrained = not boundary_limited["width"] and width_interval[1] - width_interval[0] <= 1.0 and max_y_span >= best_model["width_m"] * 0.15 and (reaches_touchline or includes_centre)

    frames: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for observation, frame_result in zip(observations, best["frame_results"]):
        solver = frame_result["solver"]
        if solver.get("status") not in SUCCESS_STATUSES:
            rejected.append({"frame": observation.get("frame"), "reason": solver.get("detail") or solver.get("status"), "excluded_semantics": frame_result["excluded"]})
            continue
        diagnostics = solver.get("diagnostics", {})
        frames.append({
            "frame": observation.get("frame"),
            "homography": solver["homography"],
            "point_error_px": diagnostics.get("point_error_px"),
            "line_error_px": diagnostics.get("line_error_px"),
            "solver_status": solver.get("status"),
            "solver_candidate": diagnostics.get("selected_candidate"),
            "homography_condition": diagnostics.get("homography_condition"),
            "constraint_spread": diagnostics.get("constraint_spread"),
            "normalised_score": frame_result["normalised_score"],
            "excluded_semantics": frame_result["excluded"],
            "semantic_holdouts": _holdouts(observation, best_model),
        })

    return {
        "status": "estimated" if frames else "no_usable_frames",
        "pitch_model": {"length_m": best["length_m"], "width_m": best["width_m"], "status": "estimated"},
        "confidence": {
            "length_interval_m": length_interval,
            "width_interval_m": width_interval,
            "boundary_limited": boundary_limited,
            "length_constrained": length_constrained,
            "width_constrained": width_constrained,
        },
        "score": {"best_normalised_score": best["score"], "near_best_candidate_count": len(near_best)},
        "frames": frames,
        "frames_rejected": rejected,
        "observability": {
            "frames_usable": len(frames),
            "semantic_point_count": total_counts["points"],
            "semantic_line_count": total_counts["lines"],
            "semantic_arc_count": total_counts["arcs"],
            "by_frame": observability,
            "reaches_touchline": reaches_touchline,
            "reaches_goal_line_structure": reaches_goal,
            "includes_centre_or_halfway": includes_centre,
        },
        "holdout_summary": _summarise_holdouts(frames),
        "search": {**asdict(cfg), "coarse_best": _surface_entry(coarse_best, "coarse"), "score_surface": surface},
    }


def summarize_pitch_estimation(result: dict[str, Any]) -> dict[str, Any]:
    """Compact JSON-safe summary for a future UI or analysis endpoint."""
    model = result.get("pitch_model", {})
    confidence = result.get("confidence", {})
    observability = result.get("observability", {})
    return {
        "status": result.get("status"),
        "length_m": model.get("length_m"),
        "width_m": model.get("width_m"),
        "length_interval_m": confidence.get("length_interval_m"),
        "width_interval_m": confidence.get("width_interval_m"),
        "boundary_limited": confidence.get("boundary_limited"),
        "length_constrained": confidence.get("length_constrained"),
        "width_constrained": confidence.get("width_constrained"),
        "best_normalised_score": result.get("score", {}).get("best_normalised_score"),
        "frames_usable": observability.get("frames_usable", 0),
        "frames_rejected": len(result.get("frames_rejected", [])),
        "holdout_summary": result.get("holdout_summary", {}),
    }
