#!/usr/bin/env python3
"""Ablation for Projective State prior + weak HUMAN geometry.

This is the validation step before promoting the method to production.

For each interior strong HUMAN anchor:
- hide its H from the 5-D Projective State model;
- choose the 5 HUMAN point constraints with MINIMUM world-x span;
- report how degenerate that subset actually is;
- compare three methods against the hidden full HUMAN H:

    A) temporal Projective State prior only
    B) independent existing H solver from the same weak points only
    C) temporal prior + weak points, refining only the 5-D state

Frame 115 is then reported with the same baselines.

Nothing is persisted.
"""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

import cv2
import numpy as np

from supervisor.field_camera_calibration import solve_field_camera_calibration
from supervisor.field_camera_state_projective import (
    ProjectiveCameraStateConfig,
    build_projective_camera_state_model,
    evaluate_homography_manual_points,
    evaluate_homography_visible_grid,
    homography_from_state,
)

REVIEW = Path("supervisor/data/reviews/benchmark_easy_2min.json")
PRIOR_SIGMA = 3.0
POINT_SIGMA_PX = 3.5
EPS = 1e-10


def calibration_for(calibrations, frame):
    for c in calibrations:
        if not isinstance(c, dict):
            continue
        try:
            same = int(c.get("frame")) == int(frame)
        except (TypeError, ValueError):
            continue
        if same and c.get("source") == "manual" and c.get("purpose") == "match_setup":
            return c
    return None


def point_constraints(calibration):
    return [
        c for c in calibration.get("constraints", [])
        if isinstance(c, dict) and c.get("type") == "point"
    ]


def world_xy(c):
    return float(c["world"]["x"]), float(c["world"]["y"])


def image_xy(c):
    return float(c["image"]["x"]), float(c["image"]["y"])


def minimum_x_span_subset(calibration, count=5):
    points = point_constraints(calibration)
    if len(points) <= count:
        return points

    best = None
    for combo in itertools.combinations(points, count):
        try:
            xs = np.asarray([world_xy(c)[0] for c in combo], float)
            ys = np.asarray([world_xy(c)[1] for c in combo], float)
        except Exception:
            continue

        x_span = float(np.ptp(xs))
        y_span = float(np.ptp(ys))
        centre_bias = float(abs(np.mean(xs)))

        # Primary goal: longitudinal degeneracy.
        # Secondary: keep useful transverse span, then prefer centre-ish subsets.
        score = (x_span, -y_span, centre_bias)

        if best is None or score < best[0]:
            best = (score, list(combo))

    return best[1] if best is not None else points[:count]


def subset_diagnostics(points):
    xs, ys, us, vs = [], [], [], []
    semantics = []

    for c in points:
        try:
            x, y = world_xy(c)
            u, v = image_xy(c)
        except Exception:
            continue
        xs.append(x); ys.append(y); us.append(u); vs.append(v)
        semantics.append(c.get("semantic"))

    if not xs:
        return {}

    return {
        "world_x_span_m": float(np.ptp(xs)),
        "world_y_span_m": float(np.ptp(ys)),
        "image_x_span_px": float(np.ptp(us)),
        "image_y_span_px": float(np.ptp(vs)),
        "semantics": semantics,
        "world_xs": [float(x) for x in xs],
    }


def state_map(model):
    return {
        int(a["frame"]): np.asarray(a["state"], float)
        for a in model["anchors"]
    }


def temporal_prior(model, query_frame):
    states = state_map(model)
    frames = sorted(states)
    left = [f for f in frames if f < query_frame]
    right = [f for f in frames if f > query_frame]

    if not left or not right:
        return None

    a = left[-1]
    b = right[0]
    alpha = (query_frame - a) / float(b - a)
    state = (1.0 - alpha) * states[a] + alpha * states[b]

    return {
        "left": a,
        "right": b,
        "alpha": float(alpha),
        "state": state,
    }


def decode_h(model, state):
    result = homography_from_state(model, state)
    if result.get("status") != "ok":
        return None
    return np.asarray(result["homography"], float)


def project(world_xy_values, h):
    pts = np.asarray(world_xy_values, float).reshape(-1, 2)
    q = np.c_[pts, np.ones(len(pts))] @ np.asarray(h, float).T
    den = q[:, 2]
    valid = np.isfinite(q).all(1) & np.isfinite(den) & (np.abs(den) > EPS)
    out = np.full((len(pts), 2), np.nan)
    out[valid] = q[valid, :2] / den[valid, None]
    valid &= np.isfinite(out).all(1)
    return out, valid


def parse_points(points):
    world = []
    image = []

    for c in points:
        try:
            world.append(world_xy(c))
            image.append(image_xy(c))
        except Exception:
            continue

    return np.asarray(world, float), np.asarray(image, float)


def state_scale(model):
    states = np.asarray([a["state"] for a in model["anchors"]], float)
    scale = np.std(states, axis=0)
    return np.where(scale < 1e-8, 1.0, scale)


def refine_state(model, prior, points, prior_sigma=PRIOR_SIGMA, max_iter=70):
    world, image = parse_points(points)
    if len(world) < 4:
        return None

    state = np.asarray(prior, float).copy()
    scale = state_scale(model)
    data_count = len(world) * 2
    dimensions = len(state)
    damping = 1e-3

    def evaluate(s):
        h = decode_h(model, s)
        if h is None:
            return None
        predicted, valid = project(world, h)
        if not np.all(valid):
            return None

        data = ((predicted - image) / POINT_SIGMA_PX).reshape(-1)
        prior_r = (s - prior) / (scale * max(float(prior_sigma), 1e-6))
        return np.r_[data, prior_r], h

    def weights(r):
        w = np.ones(len(r), float)
        for i in range(data_count):
            value = abs(float(r[i]))
            if value > 3.0:
                w[i] = 3.0 / max(value, EPS)
        return w

    current = evaluate(state)
    if current is None:
        return None

    r, _ = current
    w = weights(r)
    best_cost = 0.5 * float(np.sum(w * r * r))

    for _ in range(max_iter):
        current = evaluate(state)
        if current is None:
            break

        r, _ = current
        w = weights(r)

        J = np.zeros((len(r), dimensions), float)

        for j in range(dimensions):
            step = 1e-5 * max(1.0, abs(float(state[j])))
            trial = state.copy()
            trial[j] += step
            candidate = evaluate(trial)
            if candidate is not None:
                J[:, j] = (candidate[0] - r) / step

        sw = np.sqrt(w)
        JW = J * sw[:, None]
        rW = r * sw

        lhs = JW.T @ JW + damping * np.eye(dimensions)
        rhs = -(JW.T @ rW)

        try:
            delta = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            damping *= 10.0
            continue

        trial = state + delta
        candidate = evaluate(trial)

        if candidate is None:
            damping *= 5.0
            continue

        wr = weights(candidate[0])
        cost = 0.5 * float(np.sum(wr * candidate[0] * candidate[0]))

        if cost < best_cost:
            state = trial
            best_cost = cost
            damping = max(1e-8, damping * 0.35)

            if float(np.linalg.norm(delta)) < 1e-6:
                break
        else:
            damping = min(1e8, damping * 4.0)

    final = evaluate(state)
    if final is None:
        return None

    shift = (state - prior) / scale

    return {
        "state": state,
        "homography": final[1],
        "state_shift_z_rms": float(
            np.linalg.norm(shift) / math.sqrt(len(state))
        ),
    }


def grid_eval(predicted_h, true_h, width, height, length_m, pitch_width_m):
    if predicted_h is None:
        return None

    result = evaluate_homography_visible_grid(
        predicted_h,
        true_h,
        image_width=width,
        image_height=height,
        pitch_length_m=length_m,
        pitch_width_m=pitch_width_m,
    )

    return result if result.get("status") == "ok" else None


def independent_solver(points, width, height):
    result = solve_field_camera_calibration(
        points,
        [],
        [],
        width,
        height,
    )

    if result.get("status") not in {"solved", "solved_with_warnings"}:
        return {
            "status": result.get("status"),
            "homography": None,
        }

    return {
        "status": result.get("status"),
        "homography": np.asarray(result["homography"], float),
    }


def video_size(review, path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir {path}")

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        or review["video"].get("width")
        or 0
    )
    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        or review["video"].get("height")
        or 0
    )
    cap.release()

    return width, height


def fmt_grid(value):
    return "—" if value is None else f"{value['median_px']:.1f}px"


def main():
    review = json.loads(REVIEW.read_text(encoding="utf-8"))
    field = review["field"]
    pitch = field["pitch_model"]
    setup = field["match_setup"]
    calibrations = field.get("calibrations", [])

    completed = sorted(
        int(x) for x in setup.get("completed_frames", [])
    )

    video = Path("input") / review["video"]["filename"]
    width, height = video_size(review, video)

    length_m = float(pitch["length_m"])
    pitch_width_m = float(pitch["width_m"])

    print("\nABLATION · PROJECTIVE STATE PRIOR + LONGITUDINALLY WEAK POINTS")
    print("Strong anchors:", completed)
    print(f"Prior sigma fixed at {PRIOR_SIGMA}")

    records = []

    for holdout in completed[1:-1]:
        cal = calibration_for(calibrations, holdout)
        if cal is None:
            continue

        weak = minimum_x_span_subset(cal, 5)
        diag = subset_diagnostics(weak)

        training = [f for f in completed if f != holdout]

        model = build_projective_camera_state_model(
            calibrations,
            training,
            image_width=width,
            image_height=height,
            pitch_length_m=length_m,
            pitch_width_m=pitch_width_m,
            config=ProjectiveCameraStateConfig(dimensions=5),
        )

        if model.get("status") != "ok":
            continue

        prior_info = temporal_prior(model, holdout)
        if prior_info is None:
            continue

        prior_h = decode_h(model, prior_info["state"])
        prior_grid = grid_eval(
            prior_h,
            cal["homography"],
            width,
            height,
            length_m,
            pitch_width_m,
        )

        independent = independent_solver(
            weak,
            width,
            height,
        )
        independent_grid = grid_eval(
            independent["homography"],
            cal["homography"],
            width,
            height,
            length_m,
            pitch_width_m,
        )

        refined = refine_state(
            model,
            prior_info["state"],
            weak,
            PRIOR_SIGMA,
        )

        if refined is None:
            continue

        refined_grid = grid_eval(
            refined["homography"],
            cal["homography"],
            width,
            height,
            length_m,
            pitch_width_m,
        )

        if refined_grid is None:
            continue

        records.append({
            "frame": holdout,
            "diag": diag,
            "prior": prior_grid,
            "independent_status": independent["status"],
            "independent": independent_grid,
            "refined": refined_grid,
            "shift": refined["state_shift_z_rms"],
            "bracket": [prior_info["left"], prior_info["right"]],
        })

    print("\nHOLDOUTS")

    for r in records:
        d = r["diag"]
        print(
            f"f {r['frame']:>5} · "
            f"world span x={d.get('world_x_span_m', float('nan')):.2f}m "
            f"y={d.get('world_y_span_m', float('nan')):.2f}m · "
            f"prior={fmt_grid(r['prior'])} · "
            f"independent={fmt_grid(r['independent'])}"
            f"[{r['independent_status']}] · "
            f"prior+weak={fmt_grid(r['refined'])} · "
            f"shift-z={r['shift']:.2f} · "
            f"bracket={r['bracket']}"
        )
        print(
            "          xs:",
            [round(x, 2) for x in d.get("world_xs", [])],
        )
        print(
            "          semantics:",
            d.get("semantics", []),
        )

    if records:
        prior_values = np.asarray(
            [r["prior"]["median_px"] for r in records if r["prior"] is not None],
            float,
        )
        independent_values = np.asarray(
            [r["independent"]["median_px"] for r in records if r["independent"] is not None],
            float,
        )
        refined_values = np.asarray(
            [r["refined"]["median_px"] for r in records],
            float,
        )

        print("\nSUMMARY")
        print(
            "prior-only median:",
            f"{np.median(prior_values):.1f}px" if len(prior_values) else "—",
        )
        print(
            "independent weak-solver median:",
            f"{np.median(independent_values):.1f}px" if len(independent_values) else "—",
        )
        print(
            "prior + weak median:",
            f"{np.median(refined_values):.1f}px",
        )
        print(
            "prior + weak max median:",
            f"{np.max(refined_values):.1f}px",
        )

    print("\nFRAME 115")

    model = build_projective_camera_state_model(
        calibrations,
        completed,
        image_width=width,
        image_height=height,
        pitch_length_m=length_m,
        pitch_width_m=pitch_width_m,
        config=ProjectiveCameraStateConfig(dimensions=5),
    )

    cal115 = calibration_for(calibrations, 115)
    prior115 = temporal_prior(model, 115)

    if cal115 is None or prior115 is None:
        print("No se pudo preparar 115.")
        return

    points115 = point_constraints(cal115)
    diag115 = subset_diagnostics(points115)

    prior_h = decode_h(model, prior115["state"])
    prior_points = evaluate_homography_manual_points(
        prior_h,
        cal115,
    )

    independent115 = independent_solver(
        points115,
        width,
        height,
    )

    independent115_points = (
        evaluate_homography_manual_points(
            independent115["homography"],
            cal115,
        )
        if independent115["homography"] is not None
        else None
    )

    refined115 = refine_state(
        model,
        prior115["state"],
        points115,
        PRIOR_SIGMA,
    )

    refined_points = evaluate_homography_manual_points(
        refined115["homography"],
        cal115,
    )

    print(
        f"world span x={diag115['world_x_span_m']:.3f}m · "
        f"y={diag115['world_y_span_m']:.3f}m"
    )
    print("world xs:", [round(x, 3) for x in diag115["world_xs"]])
    print("semantics:", diag115["semantics"])
    print(
        f"prior points median={prior_points['median_px']:.1f}px"
    )
    print(
        "independent solver:",
        independent115["status"],
    )
    if independent115_points is not None and independent115_points.get("status") == "ok":
        print(
            f"independent fit median={independent115_points['median_px']:.2f}px"
        )
    print(
        f"prior+weak fit median={refined_points['median_px']:.2f}px · "
        f"P90={refined_points['p90_px']:.2f}px · "
        f"shift-z={refined115['state_shift_z_rms']:.3f}"
    )

    print("\nNo se ha modificado la review.")


if __name__ == "__main__":
    main()
