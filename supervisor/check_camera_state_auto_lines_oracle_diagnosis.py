#!/usr/bin/env python3
"""Diagnose why prior-guided automatic line recovery failed.

The hidden HUMAN H is used ONLY to center semantic search windows and for final
evaluation. The line observer still reads pixels, and recovery still starts from
the ordinary linear temporal 5-D Projective Camera State prior.

If this succeeds, localization from the temporal prior is the bottleneck.
If this also fails, the visual line observer itself is the bottleneck.
Nothing is persisted.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np

from supervisor.field_camera_state_auto_lines import (
    AutoLineEvidenceConfig,
    observe_automatic_pitch_lines,
)
from supervisor.field_camera_state_projective import (
    ProjectiveCameraStateConfig,
    build_projective_camera_state_model,
    evaluate_homography_visible_grid,
    homography_from_state,
)
from supervisor.field_camera_state_recovery import recover_camera_from_projective_state
from supervisor.field_pitch_model import build_pitch_model


REVIEW = Path("supervisor/data/reviews/benchmark_easy_2min.json")
WIDTHS = (12.0, 20.0, 30.0, 45.0)
CONFS = (0.50, 0.60, 0.70)


def calibration_for(calibrations, frame):
    for c in calibrations:
        if not isinstance(c, dict):
            continue
        try:
            same = int(c.get("frame")) == int(frame)
        except (TypeError, ValueError):
            continue
        if (
            same
            and c.get("source") == "manual"
            and c.get("purpose") == "match_setup"
            and isinstance(c.get("homography"), list)
        ):
            return c
    return None


def states(model):
    return {
        int(a["frame"]): np.asarray(a["state"], float)
        for a in model.get("anchors", [])
        if isinstance(a, dict) and a.get("state") is not None
    }


def prior_for(model, frame):
    state_by_frame = states(model)
    frames = sorted(state_by_frame)
    left = [f for f in frames if f < frame]
    right = [f for f in frames if f > frame]
    if not left or not right:
        return None
    a, b = left[-1], right[0]
    alpha = (frame - a) / float(b - a)
    return (a, b, alpha, (1.0 - alpha) * state_by_frame[a] + alpha * state_by_frame[b])


def decode(model, state):
    result = homography_from_state(model, state)
    if result.get("status") != "ok":
        return None
    return np.asarray(result["homography"], float)


def norm_line(value):
    try:
        line = np.asarray(value, float).reshape(3)
    except (TypeError, ValueError):
        return None
    n = math.hypot(float(line[0]), float(line[1]))
    return line / n if np.isfinite(line).all() and n > 1e-9 else None


def project(point, h):
    q = np.asarray([float(point["x"]), float(point["y"]), 1.0]) @ h.T
    if not np.isfinite(q).all() or abs(float(q[2])) < 1e-9:
        return None
    out = q[:2] / q[2]
    return out if np.isfinite(out).all() else None


def semantic_errors(auto, true_h, pitch_model):
    definitions = {
        item["id"]: item
        for item in pitch_model["calibration_lines"]
    }
    output = []
    for obs in auto.get("observations", []):
        definition = definitions.get(obs.get("semantic"))
        observed = norm_line(obs.get("observed_line"))
        if definition is None or observed is None:
            continue
        first = project(definition["world"]["start"], true_h)
        second = project(definition["world"]["end"], true_h)
        if first is None or second is None:
            continue
        truth = norm_line(np.cross(
            np.asarray([first[0], first[1], 1.0]),
            np.asarray([second[0], second[1], 1.0]),
        ))
        if truth is None:
            continue
        midpoint = 0.5 * (first + second)
        normal = abs(float(observed[0] * midpoint[0] + observed[1] * midpoint[1] + observed[2]))
        d1 = np.asarray([-truth[1], truth[0]])
        d2 = np.asarray([-observed[1], observed[0]])
        angle = math.degrees(math.acos(float(np.clip(abs(np.dot(d1, d2)), 0.0, 1.0))))
        output.append({
            "semantic": obs["semantic"],
            "confidence": float(obs.get("confidence") or 0.0),
            "normal_px": normal,
            "angle_deg": angle,
        })
    return output


def grid(pred, truth, width, height, length_m, pitch_width_m):
    if pred is None:
        return None
    result = evaluate_homography_visible_grid(
        pred,
        truth,
        image_width=width,
        image_height=height,
        pitch_length_m=length_m,
        pitch_width_m=pitch_width_m,
        config=ProjectiveCameraStateConfig(dimensions=5),
    )
    return result if result.get("status") == "ok" else None


def evaluate(
    frame,
    strong,
    calibrations,
    image,
    pitch_model,
    width,
    height,
    length_m,
    pitch_width_m,
    search_width,
    confidence,
):
    hidden = calibration_for(calibrations, frame)
    true_h = np.asarray(hidden["homography"], float)

    model = build_projective_camera_state_model(
        calibrations,
        [f for f in strong if f != frame],
        image_width=width,
        image_height=height,
        pitch_length_m=length_m,
        pitch_width_m=pitch_width_m,
        config=ProjectiveCameraStateConfig(dimensions=5),
    )
    if model.get("status") != "ok":
        return {"frame": frame, "status": model.get("status")}

    prior = prior_for(model, frame)
    if prior is None:
        return {"frame": frame, "status": "not_bracketed"}
    left, right, alpha, prior_state = prior
    prior_h = decode(model, prior_state)
    prior_grid = grid(prior_h, true_h, width, height, length_m, pitch_width_m)

    # Oracle H only centers search windows.
    auto = observe_automatic_pitch_lines(
        image,
        true_h,
        pitch_model,
        width,
        height,
        config=AutoLineEvidenceConfig(
            search_half_width_px=search_width,
            minimum_observer_confidence=confidence,
            observer_max_angle_delta_deg=12.0,
        ),
    )

    errors = semantic_errors(auto, true_h, pitch_model)
    record = {
        "frame": frame,
        "status": "insufficient",
        "eligible": bool(auto.get("eligible")),
        "line_count": len(auto.get("constraints", [])),
        "families": auto.get("family_counts", {}),
        "errors": errors,
        "prior_grid": prior_grid,
        "recovered_grid": None,
        "recovery": None,
        "lines": [
            (item.get("semantic"), float(item.get("confidence") or 0.0))
            for item in auto.get("observations", [])
        ],
        "bracket": [left, right],
        "alpha": alpha,
    }

    if not auto.get("eligible"):
        return record

    recovered = recover_camera_from_projective_state(
        model,
        prior_state,
        [],
        auto["constraints"],
        [],
        prior_confidence="good",
    )
    record["recovery"] = {
        "status": recovered.get("status"),
        "confidence": recovered.get("confidence"),
        "shift": recovered.get("state_shift_z_rms"),
        "fit": recovered.get("fit_rms_sigma"),
    }

    if recovered.get("status") != "ok" or not isinstance(recovered.get("homography"), list):
        record["status"] = "rejected"
        return record

    recovered_grid = grid(
        np.asarray(recovered["homography"], float),
        true_h,
        width,
        height,
        length_m,
        pitch_width_m,
    )
    if recovered_grid is None:
        record["status"] = "eval_failed"
        return record

    record["status"] = "recovered"
    record["recovered_grid"] = recovered_grid
    return record


def summarize(records):
    recovered = [r for r in records if r.get("status") == "recovered"]
    rec = np.asarray([r["recovered_grid"]["median_px"] for r in recovered], float)
    prior = np.asarray(
        [r["prior_grid"]["median_px"] for r in records if isinstance(r.get("prior_grid"), dict)],
        float,
    )
    sem = np.asarray(
        [e["normal_px"] for r in records for e in r.get("errors", [])],
        float,
    )
    ang = np.asarray(
        [e["angle_deg"] for r in records for e in r.get("errors", [])],
        float,
    )
    return {
        "records": records,
        "eligible": sum(bool(r.get("eligible")) for r in records),
        "recovered": len(recovered),
        "lines": sum(int(r.get("line_count", 0)) for r in records),
        "prior": float(np.median(prior)) if len(prior) else None,
        "rec": float(np.median(rec)) if len(rec) else None,
        "p90": float(np.percentile(rec, 90)) if len(rec) else None,
        "max": float(np.max(rec)) if len(rec) else None,
        "sem": float(np.median(sem)) if len(sem) else None,
        "ang": float(np.median(ang)) if len(ang) else None,
    }


def fmt(value, digits=1):
    return "—" if value is None else f"{value:.{digits}f}"


def main():
    review = json.loads(REVIEW.read_text(encoding="utf-8"))
    field = review["field"]
    calibrations = field.get("calibrations", [])
    strong = sorted({int(x) for x in field["match_setup"]["completed_frames"]})
    interior = strong[1:-1]
    length_m = float(field["pitch_model"]["length_m"])
    pitch_width_m = float(field["pitch_model"]["width_m"])
    pitch_model = build_pitch_model(length_m, pitch_width_m)

    video_path = Path("input") / review["video"]["filename"]
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"No se pudo abrir {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or review["video"].get("width") or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or review["video"].get("height") or 0)
    images = {}
    for frame in interior:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ok, image = cap.read()
        if ok:
            images[frame] = image
    cap.release()

    print("\nAUTO-LINES · ORACLE SEARCH DIAGNOSIS")
    print("Strong anchors:", strong)
    print("Interior holdouts:", interior)
    print(f"Pitch: {length_m:g} × {pitch_width_m:g} m")
    print("Hidden HUMAN H only centers the visual search; recovery still starts from temporal 5-D prior.\n")

    configs = []
    for search_width in WIDTHS:
        for confidence in CONFS:
            records = [
                evaluate(
                    frame, strong, calibrations, images[frame], pitch_model,
                    width, height, length_m, pitch_width_m,
                    search_width, confidence,
                )
                for frame in interior
                if frame in images
            ]
            summary = summarize(records)
            configs.append((search_width, confidence, summary))
            print(
                f"width={search_width:>4.0f}px · conf={confidence:.2f} · "
                f"eligible={summary['eligible']}/{len(records)} · "
                f"recovered={summary['recovered']}/{len(records)} · "
                f"lines={summary['lines']} · semantic={fmt(summary['sem'])}px · "
                f"angle={fmt(summary['ang'],2)}° · prior={fmt(summary['prior'])}px · "
                f"recovered={fmt(summary['rec'])}px · max={fmt(summary['max'])}px"
            )

    best = min(
        configs,
        key=lambda item: (
            -item[2]["recovered"],
            -item[2]["eligible"],
            item[2]["rec"] if item[2]["rec"] is not None else float("inf"),
            item[2]["sem"] if item[2]["sem"] is not None else float("inf"),
        ),
    )

    search_width, confidence, summary = best
    print("\nBEST ORACLE DIAGNOSTIC CONFIG")
    print(f"width={search_width:.0f}px · conf={confidence:.2f}")
    print(
        f"eligible={summary['eligible']}/{len(summary['records'])} · "
        f"recovered={summary['recovered']}/{len(summary['records'])} · "
        f"semantic={fmt(summary['sem'])}px · angle={fmt(summary['ang'],2)}° · "
        f"prior={fmt(summary['prior'])}px · recovered={fmt(summary['rec'])}px · "
        f"P90={fmt(summary['p90'])}px · max={fmt(summary['max'])}px"
    )

    for r in summary["records"]:
        fam = r.get("families", {})
        rec = r.get("recovery") or {}
        prior_error = r["prior_grid"]["median_px"] if isinstance(r.get("prior_grid"), dict) else None
        recovered_error = r["recovered_grid"]["median_px"] if isinstance(r.get("recovered_grid"), dict) else None
        sem_values = [e["normal_px"] for e in r.get("errors", [])]
        sem_median = float(np.median(sem_values)) if sem_values else None
        print(
            f"f {r['frame']:>4} · status={r['status']} · lines={r['line_count']} · "
            f"families=x{fam.get('x_constant',0)}/y{fam.get('y_constant',0)} · "
            f"semantic={fmt(sem_median)}px · prior={fmt(prior_error)}px · "
            f"recovered={fmt(recovered_error)}px · shift-z={fmt(rec.get('shift'),2)} · "
            f"fitσ={fmt(rec.get('fit'),2)}"
        )
        print("       lines:", ", ".join(f"{name}:{conf:.2f}" for name, conf in r.get("lines", [])) or "—")

    print("\nINTERPRETATION")
    print("Oracle works -> localization from the temporal prior is the bottleneck.")
    print("Oracle also fails -> improve the visual line observer before state search.")
    print("No se ha modificado la review.")


if __name__ == "__main__":
    main()
