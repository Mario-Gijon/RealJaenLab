#!/usr/bin/env python3
"""LOO sweep for bidirectional ground-plane-flow fusion in 5-D state."""

from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np

from supervisor.field_camera_state_planar_flow import (
    PlanarFlowConfig,
    homography_disagreement_px,
    propagate_pitch_homography,
)
from supervisor.field_camera_state_planar_fusion import (
    PlanarFusionConfig,
    fuse_bidirectional_planar_flow,
)
from supervisor.field_camera_state_projective import (
    ProjectiveCameraStateConfig,
    build_projective_camera_state_model,
    evaluate_homography_manual_points,
    evaluate_homography_visible_grid,
)


REVIEW = Path("supervisor/data/reviews/benchmark_easy_2min.json")

TAUS = (25.0, 40.0, 60.0)
INLIER_POWERS = (2.0, 4.0, 6.0)
PRIOR_WEIGHTS = (0.0, 0.20)
DISAGREEMENT_THRESHOLDS = (80.0, 140.0, 200.0, 1e9)


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


def state_map(model):
    return {
        int(anchor["frame"]): np.asarray(
            anchor["state"],
            dtype=np.float64,
        )
        for anchor in model.get("anchors", [])
        if isinstance(anchor, dict)
        and anchor.get("state") is not None
    }


def temporal_prior(model, frame):
    states = state_map(model)
    frames = sorted(states)

    left = [value for value in frames if value < frame]
    right = [value for value in frames if value > frame]

    if not left or not right:
        return None

    a = left[-1]
    b = right[0]
    alpha = (frame - a) / float(b - a)

    return (
        a,
        b,
        float(alpha),
        (1.0 - alpha) * states[a] + alpha * states[b],
    )


def visible_grid(
    predicted,
    truth,
    width,
    height,
    pitch_length,
    pitch_width,
):
    result = evaluate_homography_visible_grid(
        predicted,
        truth,
        image_width=width,
        image_height=height,
        pitch_length_m=pitch_length,
        pitch_width_m=pitch_width,
        config=ProjectiveCameraStateConfig(
            dimensions=5
        ),
    )

    return (
        result
        if result.get("status") == "ok"
        else None
    )


def fmt(value, digits=1):
    return (
        "—"
        if value is None
        else f"{float(value):.{digits}f}"
    )


def main():
    review = json.loads(
        REVIEW.read_text(
            encoding="utf-8"
        )
    )

    field = review["field"]
    calibrations = field.get(
        "calibrations",
        [],
    )

    strong = sorted(
        {
            int(frame)
            for frame in field[
                "match_setup"
            ].get(
                "completed_frames",
                [],
            )
        }
    )

    interior = strong[1:-1]

    pitch_length = float(
        field["pitch_model"]["length_m"]
    )
    pitch_width = float(
        field["pitch_model"]["width_m"]
    )

    video_path = (
        Path("input")
        / review["video"]["filename"]
    )

    capture = cv2.VideoCapture(
        str(video_path)
    )
    if not capture.isOpened():
        raise SystemExit(
            f"No se pudo abrir {video_path}"
        )

    width = int(
        capture.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
        or review["video"].get("width")
        or 0
    )
    height = int(
        capture.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
        or review["video"].get("height")
        or 0
    )
    capture.release()

    flow_cfg = PlanarFlowConfig()

    print()
    print(
        "GROUND-PLANE FLOW · PROJECTIVE STATE FUSION · LOO"
    )
    print(
        "Strong anchors:",
        strong,
    )
    print(
        "Interior holdouts:",
        interior,
    )
    print(
        f"Pitch: {pitch_length:g} × {pitch_width:g} m"
    )
    print()

    # Expensive flow paths are computed once.
    holdouts = []

    for frame in interior:
        hidden = calibration_for(
            calibrations,
            frame,
            require_h=True,
        )

        left_frame = max(
            anchor
            for anchor in strong
            if anchor < frame
        )
        right_frame = min(
            anchor
            for anchor in strong
            if anchor > frame
        )

        left_cal = calibration_for(
            calibrations,
            left_frame,
            require_h=True,
        )
        right_cal = calibration_for(
            calibrations,
            right_frame,
            require_h=True,
        )

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

        model = build_projective_camera_state_model(
            calibrations,
            [
                anchor
                for anchor in strong
                if anchor != frame
            ],
            image_width=width,
            image_height=height,
            pitch_length_m=pitch_length,
            pitch_width_m=pitch_width,
            config=ProjectiveCameraStateConfig(
                dimensions=5
            ),
        )

        temporal = temporal_prior(
            model,
            frame,
        )

        if (
            hidden is None
            or model.get("status") != "ok"
            or temporal is None
        ):
            continue

        _, _, _, prior_state = temporal

        holdouts.append(
            {
                "frame": frame,
                "truth_h": np.asarray(
                    hidden["homography"],
                    dtype=np.float64,
                ),
                "model": model,
                "prior_state": prior_state,
                "left_path": left_path,
                "right_path": right_path,
                "disagreement_px": (
                    disagreement.get(
                        "median_px"
                    )
                    if isinstance(
                        disagreement,
                        dict,
                    )
                    and disagreement.get(
                        "status"
                    ) == "ok"
                    else None
                ),
            }
        )

    configurations = []

    for tau in TAUS:
        for inlier_power in INLIER_POWERS:
            for prior_weight in PRIOR_WEIGHTS:
                for disagreement_threshold in DISAGREEMENT_THRESHOLDS:
                    cfg = PlanarFusionConfig(
                        step_decay_tau=tau,
                        inlier_ratio_power=inlier_power,
                        reprojection_decay_px=1.5,
                        temporal_prior_weight=prior_weight,
                        hard_select_disagreement_px=disagreement_threshold,
                        maximum_state_radius_from_prior=3.2,
                    )

                    records = []

                    for item in holdouts:
                        fused = fuse_bidirectional_planar_flow(
                            item["model"],
                            item["prior_state"],
                            item["left_path"],
                            item["right_path"],
                            disagreement_px=item[
                                "disagreement_px"
                            ],
                            config=cfg,
                        )

                        evaluation = None

                        if (
                            fused.get("status")
                            == "ok"
                            and isinstance(
                                fused.get(
                                    "homography"
                                ),
                                list,
                            )
                        ):
                            evaluation = visible_grid(
                                fused["homography"],
                                item["truth_h"],
                                width,
                                height,
                                pitch_length,
                                pitch_width,
                            )

                        records.append(
                            {
                                "frame": item["frame"],
                                "fused": fused,
                                "evaluation": evaluation,
                            }
                        )

                    valid = [
                        record
                        for record in records
                        if isinstance(
                            record.get(
                                "evaluation"
                            ),
                            dict,
                        )
                    ]

                    values = np.asarray(
                        [
                            record[
                                "evaluation"
                            ][
                                "median_px"
                            ]
                            for record in valid
                        ],
                        dtype=np.float64,
                    )

                    summary = {
                        "records": records,
                        "valid_count": len(valid),
                        "median_px": (
                            float(
                                np.median(
                                    values
                                )
                            )
                            if len(values)
                            else None
                        ),
                        "p90_px": (
                            float(
                                np.percentile(
                                    values,
                                    90,
                                )
                            )
                            if len(values)
                            else None
                        ),
                        "max_px": (
                            float(
                                np.max(
                                    values
                                )
                            )
                            if len(values)
                            else None
                        ),
                        "under_50": int(
                            np.count_nonzero(
                                values < 50.0
                            )
                        ),
                        "under_100": int(
                            np.count_nonzero(
                                values < 100.0
                            )
                        ),
                    }

                    configurations.append(
                        {
                            "tau": tau,
                            "inlier_power": inlier_power,
                            "prior_weight": prior_weight,
                            "disagreement_threshold": disagreement_threshold,
                            "summary": summary,
                        }
                    )

    configurations.sort(
        key=lambda item: (
            -item["summary"]["valid_count"],
            -item["summary"]["under_50"],
            -item["summary"]["under_100"],
            (
                item["summary"]["median_px"]
                if item["summary"]["median_px"]
                is not None
                else float("inf")
            ),
            (
                item["summary"]["max_px"]
                if item["summary"]["max_px"]
                is not None
                else float("inf")
            ),
        )
    )

    print(
        "TOP FUSION CONFIGS"
    )

    for configuration in configurations[:10]:
        summary = configuration["summary"]

        print(
            f"tau={configuration['tau']:.0f} · "
            f"inlier^={configuration['inlier_power']:.0f} · "
            f"prior_w={configuration['prior_weight']:.2f} · "
            f"hard_select={fmt(configuration['disagreement_threshold'],0)}px · "
            f"valid={summary['valid_count']}/{len(interior)} · "
            f"<50={summary['under_50']} · "
            f"<100={summary['under_100']} · "
            f"median={fmt(summary['median_px'])}px · "
            f"P90={fmt(summary['p90_px'])}px · "
            f"max={fmt(summary['max_px'])}px"
        )

    best = configurations[0]
    summary = best["summary"]

    print()
    print(
        "BEST FUSION CONFIG"
    )
    print(
        f"step_decay_tau={best['tau']:.0f}"
    )
    print(
        f"inlier_ratio_power={best['inlier_power']:.0f}"
    )
    print(
        f"temporal_prior_weight={best['prior_weight']:.2f}"
    )
    print(
        f"hard_select_disagreement_px="
        f"{fmt(best['disagreement_threshold'],0)}"
    )
    print(
        f"median={fmt(summary['median_px'])}px · "
        f"P90={fmt(summary['p90_px'])}px · "
        f"max={fmt(summary['max_px'])}px · "
        f"<50={summary['under_50']}/{len(interior)} · "
        f"<100={summary['under_100']}/{len(interior)}"
    )
    print()

    for record in summary["records"]:
        fused = record["fused"]
        evaluation = record["evaluation"]

        print(
            f"f {record['frame']:>4} · "
            f"mode={fused.get('mode')} · "
            f"error={fmt(evaluation.get('median_px') if evaluation else None)}px · "
            f"radius={fmt(fused.get('radius_from_prior'),2)} · "
            f"weights={fused.get('weights')} · "
            f"disagreement={fmt(fused.get('disagreement_px'))}px"
        )

        print(
            "       candidates:",
            [
                {
                    "side": item.get(
                        "side"
                    ),
                    "rel": round(
                        float(
                            item.get(
                                "reliability",
                                0.0,
                            )
                        ),
                        4,
                    ),
                    "steps": item.get(
                        "step_count"
                    ),
                    "inlier": (
                        round(
                            float(
                                item.get(
                                    "median_inlier_ratio"
                                )
                            ),
                            3,
                        )
                        if item.get(
                            "median_inlier_ratio"
                        )
                        is not None
                        else None
                    ),
                }
                for item in fused.get(
                    "candidates",
                    [],
                )
            ],
        )

    print()
    print(
        "FRAME 115 · BEST FUSION CONFIG"
    )

    frame = 115
    left_frame = max(
        anchor
        for anchor in strong
        if anchor < frame
    )
    right_frame = min(
        anchor
        for anchor in strong
        if anchor > frame
    )

    left_cal = calibration_for(
        calibrations,
        left_frame,
        require_h=True,
    )
    right_cal = calibration_for(
        calibrations,
        right_frame,
        require_h=True,
    )
    target_cal = calibration_for(
        calibrations,
        frame,
        require_h=False,
    )

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

    model = build_projective_camera_state_model(
        calibrations,
        strong,
        image_width=width,
        image_height=height,
        pitch_length_m=pitch_length,
        pitch_width_m=pitch_width,
        config=ProjectiveCameraStateConfig(
            dimensions=5
        ),
    )

    temporal = temporal_prior(
        model,
        frame,
    )

    if (
        model.get("status") != "ok"
        or temporal is None
    ):
        print(
            "No se pudo preparar 115."
        )
        return

    _, _, alpha, prior_state = temporal

    cfg = PlanarFusionConfig(
        step_decay_tau=best["tau"],
        inlier_ratio_power=best[
            "inlier_power"
        ],
        reprojection_decay_px=1.5,
        temporal_prior_weight=best[
            "prior_weight"
        ],
        hard_select_disagreement_px=best[
            "disagreement_threshold"
        ],
        maximum_state_radius_from_prior=3.2,
    )

    fused = fuse_bidirectional_planar_flow(
        model,
        prior_state,
        left_path,
        right_path,
        disagreement_px=(
            disagreement.get(
                "median_px"
            )
            if isinstance(
                disagreement,
                dict,
            )
            and disagreement.get(
                "status"
            ) == "ok"
            else None
        ),
        config=cfg,
    )

    print(
        f"bracket={left_frame}→115→{right_frame} · "
        f"alpha={alpha:.3f}"
    )
    print(
        f"mode={fused.get('mode')} · "
        f"weights={fused.get('weights')} · "
        f"radius={fmt(fused.get('radius_from_prior'),2)} · "
        f"disagreement={fmt(fused.get('disagreement_px'))}px"
    )

    if (
        fused.get("status")
        != "ok"
        or not isinstance(
            fused.get(
                "homography"
            ),
            list,
        )
    ):
        print(
            "No CAMERA: fusion rejected."
        )
        print(
            "No HUMAN fallback."
        )
        print(
            "No se ha modificado la review."
        )
        return

    evaluation = evaluate_homography_manual_points(
        fused["homography"],
        target_cal,
    )

    if evaluation.get(
        "status"
    ) == "ok":
        print(
            f"HUMAN point evaluation "
            f"median={evaluation['median_px']:.1f}px · "
            f"P90={evaluation['p90_px']:.1f}px · "
            f"max={evaluation['max_px']:.1f}px"
        )

    print(
        "H:"
    )
    print(
        json.dumps(
            fused["homography"],
            indent=2,
        )
    )
    print()
    print(
        "115 HUMAN used ONLY for evaluation."
    )
    print(
        "No se ha modificado la review."
    )


if __name__ == "__main__":
    main()
