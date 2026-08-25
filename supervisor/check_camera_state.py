#!/usr/bin/env python3
"""Real-data Camera State V1 check for the current PARTIDO review.

This check intentionally uses the current saved review + raw video.
It does not mutate anything.

It reports:
1. the fitted 3-D HUMAN state model;
2. LOO accuracy of RAW scanner descriptor -> HUMAN state;
3. the estimated state/confidence for rejected frame 115.

Run:
    .venv/bin/python -m supervisor.check_camera_state
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from supervisor.field_camera_state import (
    build_camera_state_model,
    camera_state_leave_one_out,
    estimate_camera_state,
    fit_camera_state_view_index,
    summarize_camera_state_model,
)
from supervisor.field_view_scanner import (
    FieldViewScannerConfig,
    _descriptor,
)


REVIEW_PATH = Path(
    "supervisor/data/reviews/benchmark_easy_2min.json"
)


def _descriptor_for_frame(
    capture: cv2.VideoCapture,
    frame: int,
    config: FieldViewScannerConfig,
):
    capture.set(
        cv2.CAP_PROP_POS_FRAMES,
        int(frame),
    )
    ok, image = capture.read()
    if not ok:
        raise RuntimeError(
            f"No se pudo leer frame {frame}."
        )

    result = _descriptor(
        image,
        config,
    )

    return np.asarray(
        result["descriptor"],
        dtype=np.float64,
    ), result


def main() -> int:
    review = json.loads(
        REVIEW_PATH.read_text(
            encoding="utf-8"
        )
    )

    field = review["field"]
    pitch = field["pitch_model"]
    match_setup = field["match_setup"]

    video_name = review["video"]["filename"]
    video_path = Path("input") / video_name

    if not video_path.exists():
        raise SystemExit(
            f"No existe vídeo: {video_path}"
        )

    capture = cv2.VideoCapture(
        str(video_path)
    )
    if not capture.isOpened():
        raise SystemExit(
            f"No se pudo abrir: {video_path}"
        )

    try:
        image_width = int(
            capture.get(
                cv2.CAP_PROP_FRAME_WIDTH
            )
            or review["video"].get("width")
            or 0
        )
        image_height = int(
            capture.get(
                cv2.CAP_PROP_FRAME_HEIGHT
            )
            or review["video"].get("height")
            or 0
        )

        completed = [
            int(value)
            for value in match_setup.get(
                "completed_frames",
                [],
            )
        ]

        model = build_camera_state_model(
            field.get(
                "calibrations",
                [],
            ),
            completed,
            image_width=image_width,
            image_height=image_height,
            pitch_length_m=float(
                pitch["length_m"]
            ),
            pitch_width_m=float(
                pitch["width_m"]
            ),
        )

        print()
        print("=" * 78)
        print(
            "CAMERA STATE V1 · MODELO HUMAN 3D"
        )
        print("=" * 78)
        print(
            json.dumps(
                summarize_camera_state_model(
                    model
                ),
                indent=2,
                ensure_ascii=False,
            )
        )

        if model.get("status") != "ok":
            raise SystemExit(
                "No se pudo construir Camera State."
            )

        scanner_config = (
            FieldViewScannerConfig()
        )

        descriptors = {}
        descriptor_meta = {}

        for frame in model[
            "anchor_frames"
        ]:
            descriptor, meta = (
                _descriptor_for_frame(
                    capture,
                    int(frame),
                    scanner_config,
                )
            )
            descriptors[
                int(frame)
            ] = descriptor
            descriptor_meta[
                int(frame)
            ] = meta

        loo = camera_state_leave_one_out(
            model,
            descriptors,
        )

        print()
        print("=" * 78)
        print(
            "RAW VIEW → CAMERA STATE · LEAVE-ONE-OUT"
        )
        print("=" * 78)

        if loo.get("status") != "ok":
            print(
                json.dumps(
                    loo,
                    indent=2,
                    ensure_ascii=False,
                )
            )
        else:
            print(
                f"mediana RMSE estado: "
                f"{loo['median_state_rmse']:.3f}"
            )
            print(
                f"P90 RMSE estado: "
                f"{loo['p90_state_rmse']:.3f}"
            )
            print(
                f"máximo RMSE estado: "
                f"{loo['max_state_rmse']:.3f}"
            )
            print()

            for item in loo[
                "per_anchor"
            ]:
                neighbours = ", ".join(
                    f"{n['frame']}"
                    f"(d={n['descriptor_distance']:.3f},"
                    f"w={n['weight']:.2f})"
                    for n in item[
                        "neighbors"
                    ]
                )

                print(
                    f"f {item['frame']:>5} · "
                    f"error={item['state_rmse']:.3f} · "
                    f"conf={item['confidence']} · "
                    f"support={item['support_ratio']:.2f} · "
                    f"spread={item['neighbor_state_spread']:.2f} · "
                    f"vecinos: {neighbours}"
                )

        view_index = (
            fit_camera_state_view_index(
                model,
                descriptors,
            )
        )

        rejected_frames = [
            int(value)
            for value in match_setup.get(
                "rejected_frames",
                [],
            )
        ]

        print()
        print("=" * 78)
        print(
            "ESTADOS DE VISTAS HUMAN RECHAZADAS"
        )
        print("=" * 78)

        for frame in rejected_frames:
            descriptor, meta = (
                _descriptor_for_frame(
                    capture,
                    frame,
                    scanner_config,
                )
            )

            estimate = (
                estimate_camera_state(
                    model,
                    view_index,
                    descriptor,
                    query_frame=frame,
                )
            )

            print(
                f"frame {frame}:"
            )
            print(
                f"  scanner quality="
                f"{float(meta['quality_score']):.3f}"
            )

            if estimate.get(
                "status"
            ) != "ok":
                print(
                    "  ",
                    estimate,
                )
                continue

            print(
                "  state:",
                [
                    round(
                        float(value),
                        3,
                    )
                    for value in estimate[
                        "state"
                    ]
                ],
            )
            print(
                f"  confidence="
                f"{estimate['confidence']}"
            )
            print(
                f"  support_ratio="
                f"{estimate['support_ratio']:.2f}"
            )
            print(
                f"  neighbour_spread="
                f"{estimate['neighbor_state_spread']:.2f}"
            )
            print(
                "  vecinos:",
            )
            for neighbor in estimate[
                "neighbors"
            ]:
                print(
                    f"    f {neighbor['frame']} · "
                    f"d={neighbor['descriptor_distance']:.4f} · "
                    f"w={neighbor['weight']:.3f} · "
                    f"d_state={neighbor['state_distance_from_estimate']:.3f}"
                )

        print()
        print("=" * 78)
        print(
            "No se ha reconstruido ninguna H "
            "ni modificado la review."
        )
        print("=" * 78)
        print()

    finally:
        capture.release()

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
