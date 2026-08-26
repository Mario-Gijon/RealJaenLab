#!/usr/bin/env python3
"""Dense validation of the frozen FIELD camera reconstruction V1.

Analysis-only. Does not modify the review.

For each dense sample between adjacent strong HUMAN anchors:
- propagate ground-plane projective flow from left and right anchors;
- endpoint-close both paths using the frozen exact 8-D projective-jet bridge;
- fuse corrected paths when disagreement <= 90 px, otherwise select one;
- measure corrected L/R disagreement, final-H temporal continuity, viewport
  footprint continuity, and flow quality.

No target-frame HUMAN geometry is read. No auto-line refinement is used.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

from supervisor.field_camera_state_planar_bridge import (
    PlanarBridgeConfig,
    bridge_correct_homography,
    corrected_path_result,
)
from supervisor.field_camera_state_planar_bridge_fusion import (
    PlanarBridgeFusionConfig,
    fuse_corrected_planar_paths,
)
from supervisor.field_camera_state_planar_flow import (
    PlanarFlowConfig,
    homography_disagreement_px,
    propagate_pitch_homography,
)
from supervisor.field_camera_state_planar_selection import PlanarSelectionConfig


REVIEW = Path("supervisor/data/reviews/benchmark_easy_2min.json")

BRIDGE_CONFIG = PlanarBridgeConfig(correction_exponent=2.0)
SELECTION_CONFIG = PlanarSelectionConfig(
    step_decay_tau=20.0,
    inlier_ratio_power=6.0,
    reprojection_decay_px=1.0,
)
FUSION_CONFIG = PlanarBridgeFusionConfig(
    maximum_fusion_disagreement_px=90.0,
    reliability_power=1.0,
)


def calibration_for(calibrations, frame):
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
            and isinstance(calibration.get("homography"), list)
        ):
            return calibration
    return None


def fmt(value, digits=1):
    return "—" if value is None else f"{float(value):.{digits}f}"


def finite(value):
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def median(values):
    values = np.asarray([float(v) for v in values if finite(v)], dtype=np.float64)
    return float(np.median(values)) if len(values) else None


def percentile(values, q):
    values = np.asarray([float(v) for v in values if finite(v)], dtype=np.float64)
    return float(np.percentile(values, q)) if len(values) else None


def maximum(values):
    values = [float(v) for v in values if finite(v)]
    return max(values) if values else None


def robust_threshold(values, minimum=None, mad_multiplier=5.0):
    values = np.asarray([float(v) for v in values if finite(v)], dtype=np.float64)
    if len(values) == 0:
        return minimum
    med = float(np.median(values))
    mad = float(np.median(np.abs(values - med)))
    threshold = med + mad_multiplier * 1.4826 * mad
    return max(float(minimum), threshold) if minimum is not None else threshold


def _normalise_h(value):
    try:
        h = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if h.shape != (3, 3) or not np.isfinite(h).all() or abs(float(h[2, 2])) < 1e-10:
        return None
    return h / float(h[2, 2])


def viewport_pitch_footprint(homography, image_width, image_height):
    h = _normalise_h(homography)
    if h is None:
        return {"status": "invalid_homography"}
    try:
        inv = np.linalg.inv(h)
    except np.linalg.LinAlgError:
        return {"status": "inverse_failed"}

    corners = np.asarray(
        [
            [0.0, 0.0],
            [float(image_width - 1), 0.0],
            [float(image_width - 1), float(image_height - 1)],
            [0.0, float(image_height - 1)],
        ],
        dtype=np.float64,
    )
    q = np.column_stack((corners, np.ones(4))) @ inv.T
    d = q[:, 2]
    if not np.isfinite(q).all() or np.any(np.abs(d) < 1e-9):
        return {"status": "projection_failed"}

    world = q[:, :2] / d[:, None]
    if not np.isfinite(world).all():
        return {"status": "projection_failed"}

    x = world[:, 0]
    y = world[:, 1]
    area = 0.5 * abs(float(np.sum(x * np.roll(y, -1) - y * np.roll(x, -1))))

    return {
        "status": "ok",
        "corners": world,
        "centroid": np.mean(world, axis=0),
        "area_m2": area,
    }


def footprint_step(previous, current):
    if (
        not isinstance(previous, dict)
        or previous.get("status") != "ok"
        or not isinstance(current, dict)
        or current.get("status") != "ok"
    ):
        return {"status": "invalid_footprint"}

    a = np.asarray(previous["corners"], dtype=np.float64)
    b = np.asarray(current["corners"], dtype=np.float64)
    shifts = np.linalg.norm(b - a, axis=1)
    centroid_shift = float(
        np.linalg.norm(
            np.asarray(current["centroid"], dtype=np.float64)
            - np.asarray(previous["centroid"], dtype=np.float64)
        )
    )
    area_log_ratio = abs(
        math.log(
            max(float(current["area_m2"]), 1e-9)
            / max(float(previous["area_m2"]), 1e-9)
        )
    )
    return {
        "status": "ok",
        "corner_shift_median_m": float(np.median(shifts)),
        "centroid_shift_m": centroid_shift,
        "area_log_ratio": float(area_log_ratio),
    }


def path_metric(path, key):
    if not isinstance(path, dict) or path.get("status") != "ok":
        return None
    value = path.get(key)
    return float(value) if finite(value) else None


def interval_samples(left_frame, right_frame, sample_step):
    step = max(1, int(sample_step))
    frames = list(range(left_frame + step, right_frame, step))
    if right_frame - left_frame > 1:
        last = right_frame - 1
        if not frames or frames[-1] != last:
            frames.append(last)
    return sorted(set(frames))


def build_target(
    *,
    frame,
    left_frame,
    right_frame,
    left_calibration,
    right_calibration,
    left_closure,
    right_closure,
    video_path,
    flow_config,
    width,
    height,
    pitch_length,
    pitch_width,
):
    left_target = propagate_pitch_homography(
        video_path,
        left_frame,
        frame,
        left_calibration["homography"],
        pitch_length_m=pitch_length,
        pitch_width_m=pitch_width,
        config=flow_config,
    )
    right_target = propagate_pitch_homography(
        video_path,
        right_frame,
        frame,
        right_calibration["homography"],
        pitch_length_m=pitch_length,
        pitch_width_m=pitch_width,
        config=flow_config,
    )

    if left_target.get("status") != "ok" or right_target.get("status") != "ok":
        return {
            "status": "target_flow_failed",
            "left_target": left_target,
            "right_target": right_target,
        }

    alpha = (frame - left_frame) / float(right_frame - left_frame)

    left_corr = bridge_correct_homography(
        left_target,
        left_closure,
        right_calibration["homography"],
        progress_to_endpoint=alpha,
        image_width=width,
        image_height=height,
        pitch_length_m=pitch_length,
        pitch_width_m=pitch_width,
        config=BRIDGE_CONFIG,
    )
    right_corr = bridge_correct_homography(
        right_target,
        right_closure,
        left_calibration["homography"],
        progress_to_endpoint=1.0 - alpha,
        image_width=width,
        image_height=height,
        pitch_length_m=pitch_length,
        pitch_width_m=pitch_width,
        config=BRIDGE_CONFIG,
    )

    if left_corr.get("status") != "ok" or right_corr.get("status") != "ok":
        return {"status": "bridge_failed"}

    left_corrected = corrected_path_result(left_target, left_corr)
    right_corrected = corrected_path_result(right_target, right_corr)

    disagreement = homography_disagreement_px(
        left_corrected["homography"],
        right_corrected["homography"],
        pitch_length_m=pitch_length,
        pitch_width_m=pitch_width,
        image_width=width,
        image_height=height,
    )
    disagreement_px = (
        disagreement["median_px"] if disagreement.get("status") == "ok" else None
    )

    final = fuse_corrected_planar_paths(
        left_corrected,
        right_corrected,
        disagreement_px=disagreement_px,
        image_width=width,
        image_height=height,
        pitch_length_m=pitch_length,
        pitch_width_m=pitch_width,
        selection_config=SELECTION_CONFIG,
        fusion_config=FUSION_CONFIG,
    )

    if final.get("status") != "ok" or not isinstance(final.get("homography"), list):
        return {"status": "final_failed"}

    return {
        "status": "ok",
        "alpha": alpha,
        "left_target": left_target,
        "right_target": right_target,
        "disagreement_px": disagreement_px,
        "final": final,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-step", type=int, default=5)
    parser.add_argument("--flow-step", type=int, default=2)
    args = parser.parse_args()

    review = json.loads(REVIEW.read_text(encoding="utf-8"))
    field = review["field"]
    calibrations = field.get("calibrations", [])
    strong = sorted({int(x) for x in field["match_setup"]["completed_frames"]})
    pitch_length = float(field["pitch_model"]["length_m"])
    pitch_width = float(field["pitch_model"]["width_m"])
    video_path = Path("input") / review["video"]["filename"]

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"No se pudo abrir {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or review["video"].get("width") or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or review["video"].get("height") or 0)
    cap.release()

    flow_config = PlanarFlowConfig(sample_step_frames=max(1, int(args.flow_step)))

    print()
    print("FIELD CAMERA V1 · DENSE TEMPORAL VALIDATION")
    print("Strong HUMAN anchors:", strong)
    print(f"Pitch: {pitch_length:g} × {pitch_width:g} m")
    print(
        f"dense sample step={args.sample_step} frames · "
        f"flow step={flow_config.sample_step_frames} frames"
    )
    print("Frozen: bridge exponent=2.0 · fusion gate=90px · reliability power=1.0")
    print("Target-frame HUMAN geometry used: NO")
    print("Auto-line refinement used: NO")
    print()

    records = []
    previous_h = None
    previous_footprint = None
    previous_frame = None

    for left_frame, right_frame in zip(strong[:-1], strong[1:]):
        left_cal = calibration_for(calibrations, left_frame)
        right_cal = calibration_for(calibrations, right_frame)
        if left_cal is None or right_cal is None:
            continue

        print(f"INTERVAL {left_frame}→{right_frame} ({right_frame-left_frame} frames)")

        left_closure = propagate_pitch_homography(
            video_path,
            left_frame,
            right_frame,
            left_cal["homography"],
            pitch_length_m=pitch_length,
            pitch_width_m=pitch_width,
            config=flow_config,
        )
        right_closure = propagate_pitch_homography(
            video_path,
            right_frame,
            left_frame,
            right_cal["homography"],
            pitch_length_m=pitch_length,
            pitch_width_m=pitch_width,
            config=flow_config,
        )

        if left_closure.get("status") != "ok" or right_closure.get("status") != "ok":
            print("  closure failed; interval skipped")
            print()
            continue

        left_close_error = homography_disagreement_px(
            left_closure["homography"],
            right_cal["homography"],
            pitch_length_m=pitch_length,
            pitch_width_m=pitch_width,
            image_width=width,
            image_height=height,
        )
        right_close_error = homography_disagreement_px(
            right_closure["homography"],
            left_cal["homography"],
            pitch_length_m=pitch_length,
            pitch_width_m=pitch_width,
            image_width=width,
            image_height=height,
        )
        print(
            f"  closure L→R={fmt(left_close_error.get('median_px'))}px · "
            f"R→L={fmt(right_close_error.get('median_px'))}px"
        )

        if previous_frame != left_frame:
            anchor_h = left_cal["homography"]
            fp = viewport_pitch_footprint(anchor_h, width, height)
            continuity = (
                homography_disagreement_px(
                    previous_h,
                    anchor_h,
                    pitch_length_m=pitch_length,
                    pitch_width_m=pitch_width,
                    image_width=width,
                    image_height=height,
                )
                if previous_h is not None
                else None
            )
            fp_step = footprint_step(previous_footprint, fp) if previous_footprint else None
            records.append(
                {
                    "frame": left_frame,
                    "kind": "anchor",
                    "mode": "human_anchor",
                    "status": "ok",
                    "disagreement_px": 0.0,
                    "continuity_px": (
                        continuity.get("median_px")
                        if isinstance(continuity, dict)
                        and continuity.get("status") == "ok"
                        else None
                    ),
                    "footprint_step": fp_step,
                }
            )
            previous_h = np.asarray(anchor_h, dtype=np.float64)
            previous_footprint = fp
            previous_frame = left_frame

        interval_records = []

        for frame in interval_samples(left_frame, right_frame, args.sample_step):
            result = build_target(
                frame=frame,
                left_frame=left_frame,
                right_frame=right_frame,
                left_calibration=left_cal,
                right_calibration=right_cal,
                left_closure=left_closure,
                right_closure=right_closure,
                video_path=video_path,
                flow_config=flow_config,
                width=width,
                height=height,
                pitch_length=pitch_length,
                pitch_width=pitch_width,
            )

            if result.get("status") != "ok":
                print(f"  f {frame:>4} · FAILED · {result.get('status')}")
                record = {
                    "frame": frame,
                    "kind": "derived",
                    "status": result.get("status"),
                }
                records.append(record)
                interval_records.append(record)
                continue

            final = result["final"]
            h = np.asarray(final["homography"], dtype=np.float64)
            continuity = (
                homography_disagreement_px(
                    previous_h,
                    h,
                    pitch_length_m=pitch_length,
                    pitch_width_m=pitch_width,
                    image_width=width,
                    image_height=height,
                )
                if previous_h is not None
                else None
            )
            continuity_px = (
                continuity["median_px"]
                if isinstance(continuity, dict) and continuity.get("status") == "ok"
                else None
            )

            fp = viewport_pitch_footprint(h, width, height)
            fp_step = footprint_step(previous_footprint, fp) if previous_footprint else None

            record = {
                "frame": frame,
                "kind": "derived",
                "status": "ok",
                "mode": final.get("mode"),
                "confidence": final.get("selection", {}).get("confidence"),
                "selected_side": final.get("selection", {}).get("selected_side"),
                "disagreement_px": result.get("disagreement_px"),
                "continuity_px": continuity_px,
                "footprint_step": fp_step,
                "left_inlier_ratio": path_metric(result["left_target"], "median_inlier_ratio"),
                "right_inlier_ratio": path_metric(result["right_target"], "median_inlier_ratio"),
                "left_reprojection_px": path_metric(result["left_target"], "median_reprojection_px"),
                "right_reprojection_px": path_metric(result["right_target"], "median_reprojection_px"),
            }
            records.append(record)
            interval_records.append(record)

            previous_h = h
            previous_footprint = fp
            previous_frame = frame

        anchor_h = right_cal["homography"]
        fp = viewport_pitch_footprint(anchor_h, width, height)
        continuity = homography_disagreement_px(
            previous_h,
            anchor_h,
            pitch_length_m=pitch_length,
            pitch_width_m=pitch_width,
            image_width=width,
            image_height=height,
        )
        fp_step = footprint_step(previous_footprint, fp)

        anchor_record = {
            "frame": right_frame,
            "kind": "anchor",
            "status": "ok",
            "mode": "human_anchor",
            "disagreement_px": 0.0,
            "continuity_px": (
                continuity["median_px"] if continuity.get("status") == "ok" else None
            ),
            "footprint_step": fp_step,
        }
        records.append(anchor_record)
        interval_records.append(anchor_record)
        previous_h = np.asarray(anchor_h, dtype=np.float64)
        previous_footprint = fp
        previous_frame = right_frame

        derived_interval = [
            r for r in interval_records if r.get("kind") == "derived" and r.get("status") == "ok"
        ]
        print(
            f"  dense={len(derived_interval)} · "
            f"fused={sum(r.get('mode')=='fused_8d_jet' for r in derived_interval)} · "
            f"selected-only={sum(r.get('mode')=='selected_only' for r in derived_interval)} · "
            f"failures={sum(r.get('kind')=='derived' and r.get('status')!='ok' for r in interval_records)}"
        )
        print(
            f"  disagreement med={fmt(median([r.get('disagreement_px') for r in derived_interval]))}px · "
            f"P90={fmt(percentile([r.get('disagreement_px') for r in derived_interval],90))}px"
        )
        print(
            f"  continuity med={fmt(median([r.get('continuity_px') for r in interval_records]))}px · "
            f"P90={fmt(percentile([r.get('continuity_px') for r in interval_records],90))}px"
        )
        print()

    derived = [r for r in records if r.get("kind") == "derived" and r.get("status") == "ok"]
    failures = [r for r in records if r.get("kind") == "derived" and r.get("status") != "ok"]

    disagreements = [r.get("disagreement_px") for r in derived]
    continuities = [r.get("continuity_px") for r in records if r.get("status") == "ok"]
    centroid_steps = [
        r["footprint_step"]["centroid_shift_m"]
        for r in records
        if isinstance(r.get("footprint_step"), dict)
        and r["footprint_step"].get("status") == "ok"
    ]
    corner_steps = [
        r["footprint_step"]["corner_shift_median_m"]
        for r in records
        if isinstance(r.get("footprint_step"), dict)
        and r["footprint_step"].get("status") == "ok"
    ]
    area_changes = [
        r["footprint_step"]["area_log_ratio"]
        for r in records
        if isinstance(r.get("footprint_step"), dict)
        and r["footprint_step"].get("status") == "ok"
    ]
    inliers = [
        v
        for r in derived
        for v in (r.get("left_inlier_ratio"), r.get("right_inlier_ratio"))
        if finite(v)
    ]
    reproj = [
        v
        for r in derived
        for v in (r.get("left_reprojection_px"), r.get("right_reprojection_px"))
        if finite(v)
    ]

    disagreement_threshold = robust_threshold(disagreements, minimum=90.0, mad_multiplier=4.0)
    continuity_threshold = robust_threshold(continuities, minimum=100.0, mad_multiplier=5.0)
    centroid_threshold = robust_threshold(centroid_steps, mad_multiplier=5.0)
    area_threshold = robust_threshold(area_changes, mad_multiplier=5.0)

    review_candidates = []
    for r in derived:
        reasons = []
        if finite(r.get("disagreement_px")) and r["disagreement_px"] > disagreement_threshold:
            reasons.append("high_corrected_disagreement")
        if finite(r.get("continuity_px")) and r["continuity_px"] > continuity_threshold:
            reasons.append("temporal_h_jump")
        fp_step = r.get("footprint_step")
        if isinstance(fp_step, dict) and fp_step.get("status") == "ok":
            if centroid_threshold is not None and fp_step["centroid_shift_m"] > centroid_threshold:
                reasons.append("footprint_centroid_jump")
            if area_threshold is not None and fp_step["area_log_ratio"] > area_threshold:
                reasons.append("footprint_scale_jump")
        if r.get("confidence") == "poor":
            reasons.append("poor_selection_confidence")
        if reasons:
            review_candidates.append((r, reasons))

    print("GLOBAL DENSE SUMMARY")
    print(
        f"derived samples={len(derived)} · failures={len(failures)} · "
        f"fused={sum(r.get('mode')=='fused_8d_jet' for r in derived)} · "
        f"selected-only={sum(r.get('mode')=='selected_only' for r in derived)}"
    )
    print(
        f"corrected disagreement median={fmt(median(disagreements))}px · "
        f"P90={fmt(percentile(disagreements,90))}px · max={fmt(maximum(disagreements))}px"
    )
    print(
        f"final-H continuity median={fmt(median(continuities))}px · "
        f"P90={fmt(percentile(continuities,90))}px · max={fmt(maximum(continuities))}px"
    )
    print(
        f"footprint centroid step median={fmt(median(centroid_steps),2)}m · "
        f"P90={fmt(percentile(centroid_steps,90),2)}m · max={fmt(maximum(centroid_steps),2)}m"
    )
    print(
        f"footprint corner step median={fmt(median(corner_steps),2)}m · "
        f"P90={fmt(percentile(corner_steps,90),2)}m · max={fmt(maximum(corner_steps),2)}m"
    )
    print(
        f"footprint |log area ratio| median={fmt(median(area_changes),3)} · "
        f"P90={fmt(percentile(area_changes,90),3)} · max={fmt(maximum(area_changes),3)}"
    )
    print(
        f"flow inlier ratio median={fmt(median(inliers),3)} · "
        f"P10={fmt(percentile(inliers,10),3)}"
    )
    print(
        f"flow reprojection median={fmt(median(reproj),3)}px · "
        f"P90={fmt(percentile(reproj,90),3)}px · max={fmt(maximum(reproj),3)}px"
    )

    print()
    print("ROBUST REVIEW THRESHOLDS")
    print(f"corrected disagreement > {fmt(disagreement_threshold)}px")
    print(f"final-H continuity > {fmt(continuity_threshold)}px")
    print(f"footprint centroid step > {fmt(centroid_threshold,2)}m")
    print(f"footprint |log area ratio| > {fmt(area_threshold,3)}")

    print()
    print("REVIEW CANDIDATES")
    if not review_candidates:
        print("none")
    else:
        for r, reasons in review_candidates:
            fp_step = r.get("footprint_step") or {}
            print(
                f"f {r['frame']:>4} · mode={r.get('mode')} · "
                f"reasons={','.join(reasons)} · "
                f"disagreement={fmt(r.get('disagreement_px'))}px · "
                f"continuity={fmt(r.get('continuity_px'))}px · "
                f"centroid-step={fmt(fp_step.get('centroid_shift_m'),2)}m"
            )

    print()
    print("No target-frame HUMAN geometry was used.")
    print("No automatic semantic-line refinement was used.")
    print("No se ha modificado la review.")


if __name__ == "__main__":
    main()
