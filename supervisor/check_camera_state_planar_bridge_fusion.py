#!/usr/bin/env python3
"""LOO benchmark: fuse endpoint-closed planar-flow cameras in exact 8-D jet."""

from __future__ import annotations

import json
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
BRIDGE_EXPONENT = 2.0
FUSION_THRESHOLDS = (30.0, 45.0, 60.0, 75.0, 90.0, 110.0)
RELIABILITY_POWERS = (0.0, 0.5, 1.0, 2.0, 4.0)

SELECTION_CONFIG = PlanarSelectionConfig(
    step_decay_tau=20.0,
    inlier_ratio_power=6.0,
    reprojection_decay_px=1.0,
)


def calibration_for(calibrations, frame, require_h=False):
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
            and (not require_h or isinstance(c.get("homography"), list))
        ):
            return c
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


def build_pair(
    *,
    frame,
    left_frame,
    right_frame,
    left_cal,
    right_cal,
    video_path,
    flow_config,
    bridge_config,
    width,
    height,
    pitch_length,
    pitch_width,
):
    left_target = propagate_pitch_homography(
        video_path,
        left_frame,
        frame,
        left_cal["homography"],
        pitch_length_m=pitch_length,
        pitch_width_m=pitch_width,
        config=flow_config,
    )
    right_target = propagate_pitch_homography(
        video_path,
        right_frame,
        frame,
        right_cal["homography"],
        pitch_length_m=pitch_length,
        pitch_width_m=pitch_width,
        config=flow_config,
    )
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

    alpha = (frame - left_frame) / float(right_frame - left_frame)

    left_corr = bridge_correct_homography(
        left_target,
        left_closure,
        right_cal["homography"],
        progress_to_endpoint=alpha,
        image_width=width,
        image_height=height,
        pitch_length_m=pitch_length,
        pitch_width_m=pitch_width,
        config=bridge_config,
    )
    right_corr = bridge_correct_homography(
        right_target,
        right_closure,
        left_cal["homography"],
        progress_to_endpoint=1.0 - alpha,
        image_width=width,
        image_height=height,
        pitch_length_m=pitch_length,
        pitch_width_m=pitch_width,
        config=bridge_config,
    )

    left = corrected_path_result(left_target, left_corr)
    right = corrected_path_result(right_target, right_corr)

    disagreement = homography_disagreement_px(
        left["homography"],
        right["homography"],
        pitch_length_m=pitch_length,
        pitch_width_m=pitch_width,
        image_width=width,
        image_height=height,
    )

    return {
        "alpha": alpha,
        "left": left,
        "right": right,
        "disagreement": disagreement,
    }


def main():
    review = json.loads(REVIEW.read_text(encoding="utf-8"))
    field = review["field"]
    calibrations = field.get("calibrations", [])
    strong = sorted({int(x) for x in field["match_setup"]["completed_frames"]})
    interior = strong[1:-1]

    pitch_length = float(field["pitch_model"]["length_m"])
    pitch_width = float(field["pitch_model"]["width_m"])
    video_path = Path("input") / review["video"]["filename"]

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"No se pudo abrir {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or review["video"].get("width") or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or review["video"].get("height") or 0)
    cap.release()

    flow_cfg = PlanarFlowConfig()
    bridge_cfg = PlanarBridgeConfig(correction_exponent=BRIDGE_EXPONENT)

    print()
    print("ENDPOINT-CLOSED PLANAR FLOW · BIDIRECTIONAL 8D-JET FUSION · LOO")
    print("Strong anchors:", strong)
    print("Interior holdouts:", interior)
    print(f"Pitch: {pitch_length:g} × {pitch_width:g} m")
    print(f"Bridge exponent: {BRIDGE_EXPONENT:.2f}")
    print("Raw-H coefficient averaging: NO")
    print("5-D PCA fusion: NO")
    print()

    prepared = []

    print("Preparing corrected pairs...")
    for frame in interior:
        hidden = calibration_for(calibrations, frame, require_h=True)
        left_frame = max(a for a in strong if a < frame)
        right_frame = min(a for a in strong if a > frame)
        left_cal = calibration_for(calibrations, left_frame, require_h=True)
        right_cal = calibration_for(calibrations, right_frame, require_h=True)

        pair = build_pair(
            frame=frame,
            left_frame=left_frame,
            right_frame=right_frame,
            left_cal=left_cal,
            right_cal=right_cal,
            video_path=video_path,
            flow_config=flow_cfg,
            bridge_config=bridge_cfg,
            width=width,
            height=height,
            pitch_length=pitch_length,
            pitch_width=pitch_width,
        )

        disagreement_px = (
            pair["disagreement"]["median_px"]
            if pair["disagreement"].get("status") == "ok"
            else None
        )

        bridge_selected = select_planar_flow_camera(
            pair["left"],
            pair["right"],
            disagreement_px=disagreement_px,
            config=SELECTION_CONFIG,
        )

        bridge_grid = grid(
            bridge_selected["homography"],
            hidden["homography"],
            width,
            height,
            pitch_length,
            pitch_width,
        )
        left_grid = grid(
            pair["left"]["homography"],
            hidden["homography"],
            width,
            height,
            pitch_length,
            pitch_width,
        )
        right_grid = grid(
            pair["right"]["homography"],
            hidden["homography"],
            width,
            height,
            pitch_length,
            pitch_width,
        )

        prepared.append(
            {
                "frame": frame,
                "hidden": hidden,
                "pair": pair,
                "bridge_selected": bridge_selected,
                "bridge_grid": bridge_grid,
                "left_grid": left_grid,
                "right_grid": right_grid,
            }
        )

        print(
            f"f {frame:>4} · bridge={bridge_grid['median_px']:.1f}px/"
            f"{bridge_selected.get('selected_side')} · "
            f"left={left_grid['median_px']:.1f}px · "
            f"right={right_grid['median_px']:.1f}px · "
            f"disagreement={fmt(disagreement_px)}px"
        )

    configs = []

    print()
    print("FUSION SWEEP")

    for threshold in FUSION_THRESHOLDS:
        for power in RELIABILITY_POWERS:
            fuse_cfg = PlanarBridgeFusionConfig(
                maximum_fusion_disagreement_px=threshold,
                reliability_power=power,
            )
            records = []

            for item in prepared:
                disagreement_px = (
                    item["pair"]["disagreement"]["median_px"]
                    if item["pair"]["disagreement"].get("status") == "ok"
                    else None
                )

                fused = fuse_corrected_planar_paths(
                    item["pair"]["left"],
                    item["pair"]["right"],
                    disagreement_px=disagreement_px,
                    image_width=width,
                    image_height=height,
                    pitch_length_m=pitch_length,
                    pitch_width_m=pitch_width,
                    selection_config=SELECTION_CONFIG,
                    fusion_config=fuse_cfg,
                )

                evaluation = (
                    grid(
                        fused["homography"],
                        item["hidden"]["homography"],
                        width,
                        height,
                        pitch_length,
                        pitch_width,
                    )
                    if fused.get("status") == "ok"
                    and isinstance(fused.get("homography"), list)
                    else None
                )

                records.append(
                    {
                        "frame": item["frame"],
                        "fused": fused,
                        "evaluation": evaluation,
                        "bridge_grid": item["bridge_grid"],
                    }
                )

            values = np.asarray(
                [
                    r["evaluation"]["median_px"]
                    for r in records
                    if isinstance(r.get("evaluation"), dict)
                ],
                dtype=np.float64,
            )
            bridge_values = np.asarray(
                [
                    r["bridge_grid"]["median_px"]
                    for r in records
                    if isinstance(r.get("bridge_grid"), dict)
                ],
                dtype=np.float64,
            )

            improved = sum(
                1
                for r in records
                if isinstance(r.get("evaluation"), dict)
                and r["evaluation"]["median_px"] < r["bridge_grid"]["median_px"]
            )

            summary = {
                "records": records,
                "valid_count": len(values),
                "improved_count": improved,
                "bridge_median_px": float(np.median(bridge_values)),
                "median_px": float(np.median(values)) if len(values) else None,
                "p90_px": float(np.percentile(values, 90)) if len(values) else None,
                "max_px": float(np.max(values)) if len(values) else None,
                "under_30": int(np.count_nonzero(values < 30.0)),
                "under_50": int(np.count_nonzero(values < 50.0)),
                "fused_count": sum(
                    1 for r in records if r["fused"].get("mode") == "fused_8d_jet"
                ),
            }

            configs.append(
                {
                    "threshold": threshold,
                    "power": power,
                    "summary": summary,
                }
            )

            print(
                f"gate={threshold:>5.0f}px · rel^={power:.1f} · "
                f"fused={summary['fused_count']}/{len(interior)} · "
                f"improved={summary['improved_count']} · "
                f"bridge med={summary['bridge_median_px']:.1f}px · "
                f"fusion med={fmt(summary['median_px'])}px · "
                f"P90={fmt(summary['p90_px'])}px · "
                f"max={fmt(summary['max_px'])}px · "
                f"<30={summary['under_30']} · <50={summary['under_50']}"
            )

    configs.sort(
        key=lambda item: (
            -item["summary"]["valid_count"],
            item["summary"]["max_px"] if item["summary"]["max_px"] is not None else float("inf"),
            item["summary"]["p90_px"] if item["summary"]["p90_px"] is not None else float("inf"),
            item["summary"]["median_px"] if item["summary"]["median_px"] is not None else float("inf"),
            -item["summary"]["under_30"],
            -item["summary"]["improved_count"],
        )
    )

    best = configs[0]
    summary = best["summary"]

    print()
    print("BEST SAFE FUSION CONFIG")
    print(f"maximum_fusion_disagreement_px={best['threshold']:.0f}")
    print(f"reliability_power={best['power']:.1f}")
    print(
        f"bridge median={summary['bridge_median_px']:.1f}px · "
        f"fusion median={summary['median_px']:.1f}px · "
        f"P90={summary['p90_px']:.1f}px · max={summary['max_px']:.1f}px · "
        f"improved={summary['improved_count']}/{len(interior)} · "
        f"<30={summary['under_30']}/{len(interior)} · "
        f"<50={summary['under_50']}/{len(interior)}"
    )
    print()

    for r in summary["records"]:
        print(
            f"f {r['frame']:>4} · mode={r['fused'].get('mode')} · "
            f"bridge={r['bridge_grid']['median_px']:.1f}px · "
            f"final={r['evaluation']['median_px']:.1f}px · "
            f"weights={r['fused'].get('weights')} · "
            f"disagreement={fmt(r['fused'].get('disagreement_px'))}px"
        )

    print()
    print("FRAME 115 · BEST LOO FUSION CONFIG")

    frame = 115
    left_frame = max(a for a in strong if a < frame)
    right_frame = min(a for a in strong if a > frame)
    left_cal = calibration_for(calibrations, left_frame, require_h=True)
    right_cal = calibration_for(calibrations, right_frame, require_h=True)
    target_cal = calibration_for(calibrations, frame, require_h=False)

    pair = build_pair(
        frame=frame,
        left_frame=left_frame,
        right_frame=right_frame,
        left_cal=left_cal,
        right_cal=right_cal,
        video_path=video_path,
        flow_config=flow_cfg,
        bridge_config=bridge_cfg,
        width=width,
        height=height,
        pitch_length=pitch_length,
        pitch_width=pitch_width,
    )

    disagreement_px = (
        pair["disagreement"]["median_px"]
        if pair["disagreement"].get("status") == "ok"
        else None
    )

    bridge_selected = select_planar_flow_camera(
        pair["left"],
        pair["right"],
        disagreement_px=disagreement_px,
        config=SELECTION_CONFIG,
    )
    bridge_eval = evaluate_homography_manual_points(
        bridge_selected["homography"],
        target_cal,
    )

    fused = fuse_corrected_planar_paths(
        pair["left"],
        pair["right"],
        disagreement_px=disagreement_px,
        image_width=width,
        image_height=height,
        pitch_length_m=pitch_length,
        pitch_width_m=pitch_width,
        selection_config=SELECTION_CONFIG,
        fusion_config=PlanarBridgeFusionConfig(
            maximum_fusion_disagreement_px=best["threshold"],
            reliability_power=best["power"],
        ),
    )
    fused_eval = evaluate_homography_manual_points(
        fused["homography"],
        target_cal,
    )

    print(
        f"bracket={left_frame}→115→{right_frame} · "
        f"corrected disagreement={fmt(disagreement_px)}px"
    )
    if bridge_eval.get("status") == "ok":
        print(
            f"bridge selected HUMAN median={bridge_eval['median_px']:.1f}px · "
            f"P90={bridge_eval['p90_px']:.1f}px · max={bridge_eval['max_px']:.1f}px"
        )

    print(
        f"fusion mode={fused.get('mode')} · "
        f"weights={fused.get('weights')}"
    )
    if fused_eval.get("status") == "ok":
        print(
            f"fusion HUMAN median={fused_eval['median_px']:.1f}px · "
            f"P90={fused_eval['p90_px']:.1f}px · max={fused_eval['max_px']:.1f}px"
        )

    print("Final H:")
    print(json.dumps(fused["homography"], indent=2))
    print()
    print("115 HUMAN used ONLY for evaluation.")
    print("No se ha modificado la review.")


if __name__ == "__main__":
    main()
