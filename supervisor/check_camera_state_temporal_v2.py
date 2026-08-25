#!/usr/bin/env python3
"""Real-data check for Temporal Camera State Observer V2.

Compares, on the five interior HUMAN anchors:

    local Hermite state trajectory
    vs linear interpolation in time
    vs Temporal Observer V1 motion-alpha

Then estimates frame 115 between HUMAN anchors 87 and 175.

No H is reconstructed and nothing is persisted.

Run:
    .venv/bin/python -m supervisor.check_camera_state_temporal_v2
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2

from supervisor.field_camera_state import (
    build_camera_state_model,
)
from supervisor.field_camera_state_temporal_v2 import (
    estimate_temporal_camera_state_v2,
    temporal_camera_state_v2_leave_one_out,
)


REVIEW_PATH = Path(
    "supervisor/data/reviews/benchmark_easy_2min.json"
)


def main() -> int:
    review = json.loads(
        REVIEW_PATH.read_text(
            encoding="utf-8"
        )
    )

    field = review[
        "field"
    ]
    pitch = field[
        "pitch_model"
    ]
    match_setup = field[
        "match_setup"
    ]

    video_path = (
        Path(
            "input"
        )
        / review[
            "video"
        ][
            "filename"
        ]
    )

    if not video_path.exists():
        raise SystemExit(
            f"No existe vídeo: {video_path}"
        )

    capture = cv2.VideoCapture(
        str(
            video_path
        )
    )

    if not capture.isOpened():
        raise SystemExit(
            f"No se pudo abrir vídeo: {video_path}"
        )

    try:
        width = int(
            capture.get(
                cv2.CAP_PROP_FRAME_WIDTH
            )
            or review[
                "video"
            ].get(
                "width"
            )
            or 0
        )
        height = int(
            capture.get(
                cv2.CAP_PROP_FRAME_HEIGHT
            )
            or review[
                "video"
            ].get(
                "height"
            )
            or 0
        )
    finally:
        capture.release()

    completed = [
        int(
            value
        )
        for value in match_setup.get(
            "completed_frames",
            [],
        )
    ]

    model = (
        build_camera_state_model(
            field.get(
                "calibrations",
                [],
            ),
            completed,
            image_width=width,
            image_height=height,
            pitch_length_m=float(
                pitch[
                    "length_m"
                ]
            ),
            pitch_width_m=float(
                pitch[
                    "width_m"
                ]
            ),
        )
    )

    if model.get(
        "status"
    ) != "ok":
        print(
            json.dumps(
                model,
                indent=2,
                ensure_ascii=False,
            )
        )
        raise SystemExit(
            "No se pudo construir Camera State."
        )

    print()
    print(
        "=" * 84
    )
    print(
        "TEMPORAL CAMERA STATE OBSERVER V2 · "
        "HERMITE + FLOW COMO CONFIANZA"
    )
    print(
        "=" * 84
    )
    print(
        "Anchors:",
        model[
            "anchor_frames"
        ],
    )

    benchmark = (
        temporal_camera_state_v2_leave_one_out(
            model,
            video_path,
        )
    )

    print()
    print(
        "RESUMEN LOO INTERIOR"
    )
    print(
        json.dumps(
            benchmark.get(
                "summary",
                benchmark,
            ),
            indent=2,
            ensure_ascii=False,
        )
    )

    print()
    print(
        "HOLDOUTS"
    )

    for item in benchmark.get(
        "per_anchor",
        [],
    ):
        if item.get(
            "status"
        ) != "ok":
            print(
                f"  f {item.get('frame')} · "
                f"ERROR={item.get('status')}"
            )
            continue

        v1 = item.get(
            "v1_motion_state_rmse"
        )
        v1_text = (
            f"{v1:.3f}"
            if v1 is not None
            else "—"
        )

        residual = item.get(
            "composition_residual"
        )
        residual_text = (
            f"{residual:.3f}"
            if residual is not None
            else "—"
        )

        print(
            f"  f {item['frame']:>5} · "
            f"Hermite={item['hermite_state_rmse']:.3f} · "
            f"linear={item['linear_time_state_rmse']:.3f} · "
            f"V1-motion={v1_text} · "
            f"conf={item['confidence']} · "
            f"flow-valid={item['valid_step_fraction']:.2f} · "
            f"compose-res={residual_text}"
        )

    print()
    print(
        "=" * 84
    )
    print(
        "FRAME 115 · ESTADO SIN H HUMAN"
    )
    print(
        "=" * 84
    )

    frame_115 = (
        estimate_temporal_camera_state_v2(
            model,
            video_path,
            query_frame=115,
        )
    )

    if frame_115.get(
        "status"
    ) != "ok":
        print(
            json.dumps(
                frame_115,
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0

    print(
        f"bracket: "
        f"{frame_115['left_frame']} → "
        f"115 → "
        f"{frame_115['right_frame']}"
    )
    print(
        f"outer support: "
        f"{frame_115['left_outer_frame']} / "
        f"{frame_115['right_outer_frame']}"
    )
    print(
        f"alpha_time="
        f"{frame_115['alpha_time']:.3f}"
    )
    print(
        "Hermite state:",
        [
            round(
                float(
                    value
                ),
                3,
            )
            for value in frame_115[
                "state"
            ]
        ],
    )
    print(
        "Linear state:",
        [
            round(
                float(
                    value
                ),
                3,
            )
            for value in frame_115[
                "linear_time_state"
            ]
        ],
    )
    print(
        f"confidence="
        f"{frame_115['confidence']}"
    )

    motion = frame_115[
        "motion_continuity"
    ]
    print(
        f"flow valid fraction="
        f"{motion.get('valid_step_fraction'):.3f}"
    )
    residual = motion.get(
        "composition_residual"
    )
    print(
        "flow composition residual=",
        (
            f"{residual:.4f}"
            if residual is not None
            else "—"
        ),
    )

    print()
    print(
        "=" * 84
    )
    print(
        "No se ha reconstruido ninguna H "
        "ni modificado la review."
    )
    print(
        "=" * 84
    )
    print()

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
