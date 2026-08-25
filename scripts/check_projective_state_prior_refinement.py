#!/usr/bin/env python3
"""Benchmark: temporal Projective State prior + weak HUMAN points.

Purpose
-------
The 5-D Projective Camera State can reconstruct held-out HUMAN homographies
well, but time interpolation alone still misses frame 115 by ~142 px.

This script tests the missing piece before we touch production code:

    temporal 5-D Projective State prior
    + weak center-heavy HUMAN points
    -> refined 5-D state
    -> H

The optimizer NEVER changes raw H coefficients. It only moves inside the learned
5-D Projective State.

LOO benchmark
-------------
For each interior strong HUMAN anchor:
- remove that anchor from the Projective State model;
- keep only its 5 point constraints closest to halfway (smallest |world x|),
  emulating the weak geometry of frame 115;
- interpolate a temporal state prior from the nearest remaining anchors;
- refine the 5-D state with those 5 weak points;
- compare the resulting H against the hidden full HUMAN H.

The script sweeps prior strengths and then applies the empirically best one to
frame 115.

Nothing is persisted.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np

from supervisor.field_camera_state_projective import (
    ProjectiveCameraStateConfig,
    build_projective_camera_state_model,
    evaluate_homography_manual_points,
    evaluate_homography_visible_grid,
    homography_from_state,
)


REVIEW = Path("supervisor/data/reviews/benchmark_easy_2min.json")
SIGMAS = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)
POINT_SIGMA_PX = 3.5
EPS = 1e-10


def project(world_xy, h):
    pts = np.asarray(world_xy, dtype=float).reshape(-1, 2)
    q = np.c_[pts, np.ones(len(pts))] @ np.asarray(h, dtype=float).T
    den = q[:, 2]
    valid = np.isfinite(q).all(1) & np.isfinite(den) & (np.abs(den) > EPS)
    out = np.full((len(pts), 2), np.nan)
    out[valid] = q[valid, :2] / den[valid, None]
    valid &= np.isfinite(out).all(1)
    return out, valid


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


def weak_center_points(calibration, count=5):
    ranked = []
    for c in point_constraints(calibration):
        try:
            x = float(c["world"]["x"])
        except (KeyError, TypeError, ValueError):
            continue
        ranked.append((abs(x), c))
    ranked.sort(key=lambda item: item[0])
    return [c for _, c in ranked[:count]]


def parse_points(constraints):
    world, image = [], []
    for c in constraints:
        try:
            world.append([float(c["world"]["x"]), float(c["world"]["y"])])
            image.append([float(c["image"]["x"]), float(c["image"]["y"])])
        except (KeyError, TypeError, ValueError):
            continue
    return np.asarray(world, float), np.asarray(image, float)


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
    a, b = left[-1], right[0]
    alpha = (query_frame - a) / float(b - a)
    state = (1.0 - alpha) * states[a] + alpha * states[b]
    return a, b, alpha, state


def decode_h(model, state):
    result = homography_from_state(model, state)
    if result.get("status") != "ok":
        return None
    return np.asarray(result["homography"], float)


def anchor_state_scale(model):
    states = np.asarray([a["state"] for a in model["anchors"]], float)
    scale = np.std(states, axis=0)
    return np.where(scale < 1e-8, 1.0, scale)


def residual(model, state, prior, state_scale, world, image, prior_sigma):
    h = decode_h(model, state)
    if h is None:
        return None
    predicted, valid = project(world, h)
    if not np.all(valid):
        return None
    data = ((predicted - image) / POINT_SIGMA_PX).reshape(-1)
    prior_r = (state - prior) / (state_scale * max(prior_sigma, 1e-6))
    r = np.r_[data, prior_r]
    return r, h


def huber_weights(residual_vector, data_count, delta=3.0):
    w = np.ones(len(residual_vector), float)
    for i in range(data_count):
        value = abs(float(residual_vector[i]))
        if value > delta:
            w[i] = delta / max(value, EPS)
    return w


def refine_state(model, prior, points, prior_sigma, max_iter=70):
    world, image = parse_points(points)
    if len(world) < 4:
        return None

    state = np.asarray(prior, float).copy()
    scale = anchor_state_scale(model)
    damping = 1e-3
    dimensions = len(state)
    data_count = len(world) * 2

    initial = residual(model, state, prior, scale, world, image, prior_sigma)
    if initial is None:
        return None

    r, _ = initial
    w = huber_weights(r, data_count)
    best_cost = 0.5 * float(np.sum(w * r * r))
    initial_cost = best_cost

    for _ in range(max_iter):
        evaluated = residual(model, state, prior, scale, world, image, prior_sigma)
        if evaluated is None:
            break
        r, _ = evaluated
        w = huber_weights(r, data_count)

        J = np.zeros((len(r), dimensions), float)
        for j in range(dimensions):
            step = 1e-5 * max(1.0, abs(float(state[j])))
            trial = state.copy()
            trial[j] += step
            ev = residual(model, trial, prior, scale, world, image, prior_sigma)
            if ev is not None:
                J[:, j] = (ev[0] - r) / step

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

        if not np.isfinite(delta).all():
            break

        trial = state + delta
        ev = residual(model, trial, prior, scale, world, image, prior_sigma)
        if ev is None:
            damping *= 5.0
            continue

        rw = huber_weights(ev[0], data_count)
        cost = 0.5 * float(np.sum(rw * ev[0] * ev[0]))

        if cost < best_cost:
            state = trial
            best_cost = cost
            damping = max(1e-8, damping * 0.35)
            if float(np.linalg.norm(delta)) < 1e-6:
                break
        else:
            damping = min(1e8, damping * 4.0)

    final = residual(model, state, prior, scale, world, image, prior_sigma)
    if final is None:
        return None

    h = final[1]
    shift_z = (state - prior) / scale
    data_r = final[0][:data_count]

    return {
        "state": state,
        "homography": h,
        "state_shift_z_rms": float(np.linalg.norm(shift_z) / math.sqrt(dimensions)),
        "fit_rms_sigma": float(np.linalg.norm(data_r) / math.sqrt(max(data_count, 1))),
        "initial_cost": initial_cost,
        "final_cost": best_cost,
    }


def video_size(review, path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir {path}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or review["video"].get("width") or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or review["video"].get("height") or 0)
    cap.release()
    return w, h


def main():
    review = json.loads(REVIEW.read_text(encoding="utf-8"))
    field = review["field"]
    pitch = field["pitch_model"]
    setup = field["match_setup"]
    calibrations = field.get("calibrations", [])
    completed = sorted(int(x) for x in setup.get("completed_frames", []))

    video = Path("input") / review["video"]["filename"]
    width, height = video_size(review, video)
    length_m = float(pitch["length_m"])
    pitch_width_m = float(pitch["width_m"])

    print("\nPROJECTIVE STATE PRIOR + 5 WEAK HUMAN POINTS · LOO")
    print("Strong anchors:", completed)

    summaries = {}

    for sigma in SIGMAS:
        records = []

        for holdout in completed[1:-1]:
            cal = calibration_for(calibrations, holdout)
            if cal is None:
                continue

            weak = weak_center_points(cal, 5)
            if len(weak) < 4:
                continue

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
            left, right, alpha, prior = prior_info

            prior_h = decode_h(model, prior)
            prior_grid = None
            if prior_h is not None:
                prior_grid = evaluate_homography_visible_grid(
                    prior_h,
                    cal["homography"],
                    image_width=width,
                    image_height=height,
                    pitch_length_m=length_m,
                    pitch_width_m=pitch_width_m,
                )

            refined = refine_state(model, prior, weak, sigma)
            if refined is None:
                continue

            grid = evaluate_homography_visible_grid(
                refined["homography"],
                cal["homography"],
                image_width=width,
                image_height=height,
                pitch_length_m=length_m,
                pitch_width_m=pitch_width_m,
            )
            if grid.get("status") != "ok":
                continue

            all_human = evaluate_homography_manual_points(
                refined["homography"],
                cal,
            )

            records.append({
                "frame": holdout,
                "bracket": [left, right],
                "prior_grid": prior_grid.get("median_px") if prior_grid and prior_grid.get("status") == "ok" else None,
                "grid_median": grid["median_px"],
                "grid_p90": grid["p90_px"],
                "human_median": all_human.get("median_px") if all_human.get("status") == "ok" else None,
                "shift_z": refined["state_shift_z_rms"],
                "fit_sigma": refined["fit_rms_sigma"],
            })

        if not records:
            continue

        med = np.asarray([r["grid_median"] for r in records], float)
        p90 = np.asarray([r["grid_p90"] for r in records], float)
        prior_med = [r["prior_grid"] for r in records if r["prior_grid"] is not None]

        summary = {
            "records": records,
            "median": float(np.median(med)),
            "p90_medians": float(np.percentile(med, 90)),
            "max_median": float(np.max(med)),
            "median_p90": float(np.median(p90)),
            "prior_median": float(np.median(prior_med)) if prior_med else None,
        }
        summaries[sigma] = summary

        print(
            f"sigma={sigma:>4.2f} · grid med={summary['median']:.1f}px · "
            f"P90 medians={summary['p90_medians']:.1f}px · "
            f"max med={summary['max_median']:.1f}px · "
            f"median P90={summary['median_p90']:.1f}px · "
            f"prior med={summary['prior_median']:.1f}px"
        )

    if not summaries:
        raise SystemExit("No hubo benchmarks válidos.")

    best_sigma, best = min(
        summaries.items(),
        key=lambda item: (item[1]["median"], item[1]["p90_medians"]),
    )

    print(f"\nMEJOR SIGMA: {best_sigma}")
    for r in best["records"]:
        print(
            f"  f {r['frame']:>5} · prior={r['prior_grid']:.1f}px → "
            f"refined={r['grid_median']:.1f}px (P90 {r['grid_p90']:.1f}px) · "
            f"all-HUMAN={r['human_median']:.1f}px · shift-z={r['shift_z']:.2f} · "
            f"fitσ={r['fit_sigma']:.2f} · bracket={r['bracket']}"
        )

    print("\nFRAME 115 · PRIOR 87→175 + SUS 5 PUNTOS HUMAN")

    full_model = build_projective_camera_state_model(
        calibrations,
        completed,
        image_width=width,
        image_height=height,
        pitch_length_m=length_m,
        pitch_width_m=pitch_width_m,
        config=ProjectiveCameraStateConfig(dimensions=5),
    )

    prior_info = temporal_prior(full_model, 115)
    cal115 = calibration_for(calibrations, 115)

    if prior_info is None or cal115 is None:
        print("No se pudo preparar 115.")
        return

    left, right, alpha, prior = prior_info
    points115 = point_constraints(cal115)

    prior_h = decode_h(full_model, prior)
    prior_eval = evaluate_homography_manual_points(prior_h, cal115)

    refined = refine_state(full_model, prior, points115, best_sigma)
    if refined is None:
        print("No se pudo refinar 115.")
        return

    fit = evaluate_homography_manual_points(refined["homography"], cal115)

    print(f"bracket={left}→115→{right} · alpha={alpha:.3f}")
    print(
        f"prior points: median={prior_eval['median_px']:.1f}px · "
        f"P90={prior_eval['p90_px']:.1f}px"
    )
    print(
        f"refined points: median={fit['median_px']:.2f}px · "
        f"P90={fit['p90_px']:.2f}px · max={fit['max_px']:.2f}px"
    )
    print(f"state shift z RMS={refined['state_shift_z_rms']:.3f}")
    print(f"fit RMS sigma={refined['fit_rms_sigma']:.3f}")
    print("refined state:", [round(float(x), 4) for x in refined["state"]])
    print("REFINED H:")
    print(json.dumps(refined["homography"].tolist(), indent=2))

    print(
        "\nIMPORTANTE: los 5 puntos del 115 son datos de ajuste, no validación global. "
        "La evidencia global viene del LOO débil sobre anchors fuertes."
    )
    print("No se ha modificado la review.")


if __name__ == "__main__":
    main()
