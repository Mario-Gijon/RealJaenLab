#!/usr/bin/env python3
"""LOO sweep for direct reliability selection of planar-flow H estimates."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from supervisor.field_camera_state_planar_flow import (
    PlanarFlowConfig,
    homography_disagreement_px,
    propagate_pitch_homography,
)
from supervisor.field_camera_state_planar_selection import (
    PlanarSelectionConfig,
    select_planar_flow_camera,
)
from supervisor.field_camera_state_projective import (
    ProjectiveCameraStateConfig,
    evaluate_homography_manual_points,
    evaluate_homography_visible_grid,
)


REVIEW = Path("supervisor/data/reviews/benchmark_easy_2min.json")

TAUS = (20.0, 25.0, 30.0, 40.0, 50.0, 60.0)
INLIER_POWERS = (2.0, 4.0, 6.0, 8.0)
REPROJECTION_SCALES = (1.0, 1.5, 2.0)


def calibration_for(calibrations, frame, require_h=False):
    for calibration in calibrations:
        if not isinstance(calibration, dict):
            continue
        try:
            same = int(calibration.get("frame")) == int(frame)
        except (TypeError, ValueError):
            continue
        if (
            same
            and calibration.get("source") == "manual"
            and calibration.get("purpose") == "match_setup"
            and (
                not require_h
                or isinstance(calibration.get("homography"), list)
            )
        ):
            return calibration
    return None


def grid(predicted, truth, width, height, pitch_length, pitch_width):
    result = evaluate_homography_visible_grid(
        predicted,
        truth,
        image_width=width,
        image_height=height,
        pitch_length_m=pitch_length,
        pitch_width_m=pitch_width,
        config=ProjectiveCameraStateConfig(dimensions=5),
    )
    return result if result.get("status") == "ok" else None


def fmt(value, digits=1):
    return "—" if value is None else f"{float(value):.{digits}f}"


def main():
    review = json.loads(REVIEW.read_text(encoding="utf-8"))
    field = review["field"]
    calibrations = field.get("calibrations", [])
    strong = sorted({int(frame) for frame in field["match_setup"]["completed_frames"]})
    interior = strong[1:-1]

    pitch_length = float(field["pitch_model"]["length_m"])
    pitch_width = float(field["pitch_model"]["width_m"])
    video_path = Path("input") / review["video"]["filename"]

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise SystemExit(f"No se pudo abrir {video_path}")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or review["video"].get("width") or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or review["video"].get("height") or 0)
    capture.release()

    flow_cfg = PlanarFlowConfig()

    print()
    print("GROUND-PLANE FLOW · DIRECT RELIABILITY SELECTION · LOO")
    print("Strong anchors:", strong)
    print("Interior holdouts:", interior)
    print(f"Pitch: {pitch_length:g} × {pitch_width:g} m")
    print()

    # Compute expensive flow paths once.
    holdouts = []

    for frame in interior:
        hidden = calibration_for(calibrations, frame, require_h=True)
        left_frame = max(anchor for anchor in strong if anchor < frame)
        right_frame = min(anchor for anchor in strong if anchor > frame)

        left_cal = calibration_for(calibrations, left_frame, require_h=True)
        right_cal = calibration_for(calibrations, right_frame, require_h=True)

        left_path = propagate_pitch_homography(
            video_path,
            left_frame,
            frame,
            left_cal["homography"],
            pitch_length_m=pitch_length,
            pitch_width_m=pitch_width,
            config=flow_cfg,
        )

        right_path = propagate_pitch_homography(
            video_path,
            right_frame,
            frame,
            right_cal["homography"],
            pitch_length_m=pitch_length,
            pitch_width_m=pitch_width,
            config=flow_cfg,
        )

        disagreement = (
            homography_disagreement_px(
                left_path["homography"],
                right_path["homography"],
                pitch_length_m=pitch_length,
                pitch_width_m=pitch_width,
                image_width=width,
                image_height=height,
            )
            if left_path.get("status") == "ok"
            and right_path.get("status") == "ok"
            else None
        )

        holdouts.append(
            {
                "frame": frame,
                "truth_h": np.asarray(hidden["homography"], dtype=np.float64),
                "left_path": left_path,
                "right_path": right_path,
                "disagreement_px": (
                    disagreement["median_px"]
                    if isinstance(disagreement, dict)
                    and disagreement.get("status") == "ok"
                    else None
                ),
            }
        )

    configurations = []

    for tau in TAUS:
        for power in INLIER_POWERS:
            for reprojection_scale in REPROJECTION_SCALES:
                cfg = PlanarSelectionConfig(
                    step_decay_tau=tau,
                    inlier_ratio_power=power,
                    reprojection_decay_px=reprojection_scale,
                )

                records = []

                for item in holdouts:
                    selected = select_planar_flow_camera(
                        item["left_path"],
                        item["right_path"],
                        disagreement_px=item["disagreement_px"],
                        config=cfg,
                    )

                    evaluation = None

                    if (
                        selected.get("status") == "ok"
                        and isinstance(selected.get("homography"), list)
                    ):
                        evaluation = grid(
                            selected["homography"],
                            item["truth_h"],
                            width,
                            height,
                            pitch_length,
                            pitch_width,
                        )

                    records.append(
                        {
                            "frame": item["frame"],
                            "selected": selected,
                            "evaluation": evaluation,
                        }
                    )

                valid = [
                    record
                    for record in records
                    if isinstance(record.get("evaluation"), dict)
                ]

                values = np.asarray(
                    [
                        record["evaluation"]["median_px"]
                        for record in valid
                    ],
                    dtype=np.float64,
                )

                summary = {
                    "records": records,
                    "valid_count": len(valid),
                    "median_px": float(np.median(values)) if len(values) else None,
                    "p90_px": float(np.percentile(values, 90)) if len(values) else None,
                    "max_px": float(np.max(values)) if len(values) else None,
                    "under_50": int(np.count_nonzero(values < 50.0)),
                    "under_100": int(np.count_nonzero(values < 100.0)),
                }

                configurations.append(
                    {
                        "tau": tau,
                        "power": power,
                        "reprojection_scale": reprojection_scale,
                        "summary": summary,
                    }
                )

    configurations.sort(
        key=lambda item: (
            -item["summary"]["valid_count"],
            -item["summary"]["under_50"],
            -item["summary"]["under_100"],
            item["summary"]["median_px"] if item["summary"]["median_px"] is not None else float("inf"),
            item["summary"]["max_px"] if item["summary"]["max_px"] is not None else float("inf"),
        )
    )

    print("TOP SELECTION CONFIGS")
    for configuration in configurations[:12]:
        summary = configuration["summary"]
        print(
            f"tau={configuration['tau']:.0f} · "
            f"inlier^={configuration['power']:.0f} · "
            f"reproj_scale={configuration['reprojection_scale']:.1f}px · "
            f"valid={summary['valid_count']}/{len(interior)} · "
            f"<50={summary['under_50']} · <100={summary['under_100']} · "
            f"median={fmt(summary['median_px'])}px · "
            f"P90={fmt(summary['p90_px'])}px · max={fmt(summary['max_px'])}px"
        )

    best = configurations[0]
    summary = best["summary"]

    print()
    print("BEST DIRECT SELECTION CONFIG")
    print(f"step_decay_tau={best['tau']:.0f}")
    print(f"inlier_ratio_power={best['power']:.0f}")
    print(f"reprojection_decay_px={best['reprojection_scale']:.1f}")
    print(
        f"median={fmt(summary['median_px'])}px · "
        f"P90={fmt(summary['p90_px'])}px · max={fmt(summary['max_px'])}px · "
        f"<50={summary['under_50']}/{len(interior)} · "
        f"<100={summary['under_100']}/{len(interior)}"
    )
    print()

    for record in summary["records"]:
        selected = record["selected"]
        evaluation = record["evaluation"]
        print(
            f"f {record['frame']:>4} · "
            f"side={selected.get('selected_side')} · "
            f"confidence={selected.get('confidence')} · "
            f"error={fmt(evaluation.get('median_px') if evaluation else None)}px · "
            f"ratio={fmt(selected.get('reliability_ratio'),2)} · "
            f"disagreement={fmt(selected.get('disagreement_px'))}px"
        )
        print(
            "       candidates:",
            [
                {
                    "side": item.get("side"),
                    "rel": round(float(item.get("reliability") or 0.0), 4),
                    "steps": item.get("step_count"),
                    "inlier": (
                        round(float(item.get("median_inlier_ratio")), 3)
                        if item.get("median_inlier_ratio") is not None
                        else None
                    ),
                    "reproj": (
                        round(float(item.get("median_reprojection_px")), 3)
                        if item.get("median_reprojection_px") is not None
                        else None
                    ),
                }
                for item in selected.get("candidates", [])
            ],
        )

    print()
    print("FRAME 115 · BEST DIRECT SELECTION CONFIG")

    frame = 115
    left_frame = max(anchor for anchor in strong if anchor < frame)
    right_frame = min(anchor for anchor in strong if anchor > frame)
    left_cal = calibration_for(calibrations, left_frame, require_h=True)
    right_cal = calibration_for(calibrations, right_frame, require_h=True)
    target_cal = calibration_for(calibrations, frame, require_h=False)

    left_path = propagate_pitch_homography(
        video_path,
        left_frame,
        frame,
        left_cal["homography"],
        pitch_length_m=pitch_length,
        pitch_width_m=pitch_width,
        config=flow_cfg,
    )
    right_path = propagate_pitch_homography(
        video_path,
        right_frame,
        frame,
        right_cal["homography"],
        pitch_length_m=pitch_length,
        pitch_width_m=pitch_width,
        config=flow_cfg,
    )

    disagreement = (
        homography_disagreement_px(
            left_path["homography"],
            right_path["homography"],
            pitch_length_m=pitch_length,
            pitch_width_m=pitch_width,
            image_width=width,
            image_height=height,
        )
        if left_path.get("status") == "ok"
        and right_path.get("status") == "ok"
        else None
    )

    cfg = PlanarSelectionConfig(
        step_decay_tau=best["tau"],
        inlier_ratio_power=best["power"],
        reprojection_decay_px=best["reprojection_scale"],
    )

    selected = select_planar_flow_camera(
        left_path,
        right_path,
        disagreement_px=(
            disagreement["median_px"]
            if isinstance(disagreement, dict)
            and disagreement.get("status") == "ok"
            else None
        ),
        config=cfg,
    )

    print(f"bracket={left_frame}→115→{right_frame}")
    print(
        f"selected={selected.get('selected_side')} · "
        f"confidence={selected.get('confidence')} · "
        f"ratio={fmt(selected.get('reliability_ratio'),2)} · "
        f"disagreement={fmt(selected.get('disagreement_px'))}px"
    )

    if (
        selected.get("status") != "ok"
        or not isinstance(selected.get("homography"), list)
    ):
        print("No CAMERA.")
        print("No HUMAN fallback.")
        print("No se ha modificado la review.")
        return

    evaluation = evaluate_homography_manual_points(
        selected["homography"],
        target_cal,
    )

    if evaluation.get("status") == "ok":
        print(
            f"HUMAN points evaluation median={evaluation['median_px']:.1f}px · "
            f"P90={evaluation['p90_px']:.1f}px · max={evaluation['max_px']:.1f}px"
        )

    print("H:")
    print(json.dumps(selected["homography"], indent=2))
    print()
    print("115 HUMAN used ONLY for evaluation.")
    print("No se ha modificado la review.")


if __name__ == "__main__":
    main()
