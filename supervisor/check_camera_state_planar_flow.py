#!/usr/bin/env python3
"""LOO benchmark for ground-plane projective optical-flow propagation.

This tests a different use of optical flow from the old Temporal Observer V1.

Old experiment:
    dominant AFFINE motion -> scalar progress alpha -> Camera State

This experiment:
    field-masked KLT correspondences
    -> full image-to-image HOMOGRAPHY of the ground plane
    -> compose with a neighbouring strong HUMAN pitch H

The hidden target H is used only for final evaluation.
"""

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
from supervisor.field_camera_state_projective import (
    ProjectiveCameraStateConfig,
    evaluate_homography_manual_points,
    evaluate_homography_visible_grid,
)


REVIEW = Path("supervisor/data/reviews/benchmark_easy_2min.json")


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


def visible_grid(
    predicted,
    truth,
    width,
    height,
    pitch_length,
    pitch_width,
):
    if predicted is None:
        return None
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


def path_summary(path):
    if not isinstance(path, dict):
        return "—"
    return (
        f"{path.get('status')} · "
        f"steps={path.get('step_count','—')} · "
        f"inlier={fmt(path.get('median_inlier_ratio'),2)} · "
        f"reproj={fmt(path.get('median_reprojection_px'),2)}px"
    )


def main():
    review = json.loads(REVIEW.read_text(encoding="utf-8"))
    field = review["field"]
    calibrations = field.get("calibrations", [])
    strong = sorted({int(x) for x in field["match_setup"]["completed_frames"]})
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

    cfg = PlanarFlowConfig()

    print()
    print("GROUND-PLANE PROJECTIVE FLOW · LOO")
    print("Strong anchors:", strong)
    print("Interior holdouts:", interior)
    print(f"Pitch: {pitch_length:g} × {pitch_width:g} m")
    print(
        f"analysis_width={cfg.analysis_width} · "
        f"step={cfg.sample_step_frames} frames · "
        f"min inliers={cfg.minimum_inliers} · "
        f"min ratio={cfg.minimum_inlier_ratio:.2f}"
    )
    print("Circles/arcs: irrelevant / not used")
    print()

    records = []

    for frame in interior:
        hidden = calibration_for(calibrations, frame, require_h=True)
        if hidden is None:
            continue

        left_frames = [anchor for anchor in strong if anchor < frame]
        right_frames = [anchor for anchor in strong if anchor > frame]
        if not left_frames or not right_frames:
            continue

        left_frame = left_frames[-1]
        right_frame = right_frames[0]
        left_cal = calibration_for(calibrations, left_frame, require_h=True)
        right_cal = calibration_for(calibrations, right_frame, require_h=True)

        truth_h = np.asarray(hidden["homography"], dtype=np.float64)

        left_path = propagate_pitch_homography(
            video_path,
            left_frame,
            frame,
            left_cal["homography"],
            pitch_length_m=pitch_length,
            pitch_width_m=pitch_width,
            config=cfg,
        )
        right_path = propagate_pitch_homography(
            video_path,
            right_frame,
            frame,
            right_cal["homography"],
            pitch_length_m=pitch_length,
            pitch_width_m=pitch_width,
            config=cfg,
        )

        left_grid = (
            visible_grid(
                np.asarray(left_path["homography"], dtype=np.float64),
                truth_h,
                width,
                height,
                pitch_length,
                pitch_width,
            )
            if left_path.get("status") == "ok"
            else None
        )
        right_grid = (
            visible_grid(
                np.asarray(right_path["homography"], dtype=np.float64),
                truth_h,
                width,
                height,
                pitch_length,
                pitch_width,
            )
            if right_path.get("status") == "ok"
            else None
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

        left_distance = frame - left_frame
        right_distance = right_frame - frame
        selected_side = "left" if left_distance <= right_distance else "right"
        selected_path = left_path if selected_side == "left" else right_path
        selected_grid = left_grid if selected_side == "left" else right_grid

        # If nearest path fails but the other succeeds, use the successful path.
        if selected_path.get("status") != "ok":
            other_path = right_path if selected_side == "left" else left_path
            other_grid = right_grid if selected_side == "left" else left_grid
            if other_path.get("status") == "ok":
                selected_side = "right" if selected_side == "left" else "left"
                selected_path = other_path
                selected_grid = other_grid

        record = {
            "frame": frame,
            "left_frame": left_frame,
            "right_frame": right_frame,
            "left_path": left_path,
            "right_path": right_path,
            "left_grid": left_grid,
            "right_grid": right_grid,
            "disagreement": disagreement,
            "selected_side": selected_side,
            "selected_grid": selected_grid,
        }
        records.append(record)

        print(
            f"f {frame:>4} · bracket={left_frame}→{frame}→{right_frame} · "
            f"nearest={selected_side}"
        )
        print(
            f"       LEFT  {path_summary(left_path)} · "
            f"grid={fmt(left_grid.get('median_px') if left_grid else None)}px"
        )
        print(
            f"       RIGHT {path_summary(right_path)} · "
            f"grid={fmt(right_grid.get('median_px') if right_grid else None)}px"
        )
        print(
            f"       L↔R disagreement="
            f"{fmt(disagreement.get('median_px') if isinstance(disagreement,dict) and disagreement.get('status')=='ok' else None)}px · "
            f"selected grid={fmt(selected_grid.get('median_px') if selected_grid else None)}px"
        )

    selected = np.asarray(
        [
            record["selected_grid"]["median_px"]
            for record in records
            if isinstance(record.get("selected_grid"), dict)
        ],
        dtype=np.float64,
    )
    left_values = np.asarray(
        [
            record["left_grid"]["median_px"]
            for record in records
            if isinstance(record.get("left_grid"), dict)
        ],
        dtype=np.float64,
    )
    right_values = np.asarray(
        [
            record["right_grid"]["median_px"]
            for record in records
            if isinstance(record.get("right_grid"), dict)
        ],
        dtype=np.float64,
    )
    agreements = np.asarray(
        [
            record["disagreement"]["median_px"]
            for record in records
            if isinstance(record.get("disagreement"), dict)
            and record["disagreement"].get("status") == "ok"
        ],
        dtype=np.float64,
    )

    print()
    print("LOO SUMMARY")
    print(
        f"left successful={len(left_values)}/{len(interior)} · "
        f"median grid={fmt(np.median(left_values) if len(left_values) else None)}px"
    )
    print(
        f"right successful={len(right_values)}/{len(interior)} · "
        f"median grid={fmt(np.median(right_values) if len(right_values) else None)}px"
    )
    print(
        f"nearest/successful selected={len(selected)}/{len(interior)} · "
        f"median={fmt(np.median(selected) if len(selected) else None)}px · "
        f"P90={fmt(np.percentile(selected,90) if len(selected) else None)}px · "
        f"max={fmt(np.max(selected) if len(selected) else None)}px"
    )
    print(
        f"bidirectional disagreement median="
        f"{fmt(np.median(agreements) if len(agreements) else None)}px"
    )

    print()
    print("FRAME 115 · GROUND-PLANE FLOW")

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
        config=cfg,
    )
    right_path = propagate_pitch_homography(
        video_path,
        right_frame,
        frame,
        right_cal["homography"],
        pitch_length_m=pitch_length,
        pitch_width_m=pitch_width,
        config=cfg,
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

    print(f"bracket={left_frame}→115→{right_frame}")
    print("LEFT :", path_summary(left_path))
    print("RIGHT:", path_summary(right_path))
    print(
        "L↔R disagreement:",
        (
            f"median={disagreement['median_px']:.1f}px · "
            f"P90={disagreement['p90_px']:.1f}px"
            if isinstance(disagreement, dict)
            and disagreement.get("status") == "ok"
            else "—"
        ),
    )

    candidates = []
    if left_path.get("status") == "ok":
        candidates.append(("left", left_path))
    if right_path.get("status") == "ok":
        candidates.append(("right", right_path))

    if not candidates:
        print("No CAMERA: both planar-flow paths failed.")
        print("No HUMAN fallback.")
        print("No se ha modificado la review.")
        return

    candidates.sort(
        key=lambda item: (
            abs(frame - int(item[1]["start_frame"])),
            -float(item[1].get("median_inlier_ratio") or 0.0),
        )
    )
    selected_side, selected_path = candidates[0]

    print("selected:", selected_side)

    evaluation = evaluate_homography_manual_points(
        selected_path["homography"],
        target_cal,
    )
    if evaluation.get("status") == "ok":
        print(
            f"HUMAN points evaluation median={evaluation['median_px']:.1f}px · "
            f"P90={evaluation['p90_px']:.1f}px · "
            f"max={evaluation['max_px']:.1f}px"
        )
    print("H:")
    print(json.dumps(selected_path["homography"], indent=2))
    print()
    print("115 HUMAN used ONLY for evaluation.")
    print("No se ha modificado la review.")


if __name__ == "__main__":
    main()
