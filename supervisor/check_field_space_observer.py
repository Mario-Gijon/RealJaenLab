#!/usr/bin/env python3
"""FIELD SPACE / OBSERVER V1 benchmark for FSO04."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from supervisor.field_space_observer import (
    FieldSpaceObserverConfig,
    field_to_image,
    select_field_space_observer_sequence,
)


DEFAULT_VIDEO = Path("input/benchmark_normal_2min.mp4")
DEFAULT_REVIEW = Path(
    "supervisor/data/reviews/benchmark_normal_2min.json"
)
DEFAULT_DENSE = Path(
    "supervisor/data/reviews/"
    "benchmark_normal_2min_camera_atlas_dense_v1.json"
)
DEFAULT_OUT = Path(
    "supervisor/data/reviews/"
    "benchmark_normal_2min_field_space_observer_v1.json"
)


def clean(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {
            str(k): clean(v)
            for k, v in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    return value


def timestamp(frame: int, fps: float) -> str:
    seconds = float(frame) / fps if fps > 0 else 0.0
    minute = int(seconds // 60)
    rest = seconds - 60 * minute
    return f"{minute:02d}:{rest:05.2f}"


def fmt(value: Any, digits: int = 2) -> str:
    return (
        "—"
        if value is None
        else f"{float(value):.{digits}f}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video",
        type=Path,
        default=DEFAULT_VIDEO,
    )
    parser.add_argument(
        "--review",
        type=Path,
        default=DEFAULT_REVIEW,
    )
    parser.add_argument(
        "--dense",
        type=Path,
        default=DEFAULT_DENSE,
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=DEFAULT_OUT,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    review = json.loads(
        args.review.read_text(encoding="utf-8")
    )
    dense = json.loads(
        args.dense.read_text(encoding="utf-8")
    )

    pitch_model = review.get(
        "field",
        {},
    ).get(
        "pitch_model",
        {},
    )

    if pitch_model.get("status") != "confirmed":
        raise SystemExit("pitch_model no confirmado.")

    pitch_length = float(
        pitch_model["length_m"]
    )
    pitch_width = float(
        pitch_model["width_m"]
    )

    capture = cv2.VideoCapture(
        str(args.video)
    )

    if not capture.isOpened():
        raise SystemExit(
            f"No se pudo abrir {args.video}"
        )

    image_width = int(
        capture.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
        or 0
    )
    image_height = int(
        capture.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
        or 0
    )
    fps = float(
        capture.get(cv2.CAP_PROP_FPS)
        or 0.0
    )
    frame_count = int(
        capture.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
        or 0
    )
    capture.release()

    dense_records = dense.get("records")

    if not isinstance(dense_records, list):
        raise SystemExit(
            "Dense Atlas no contiene records."
        )

    dense_by_frame = {
        int(record["frame"]): record
        for record in dense_records
        if (
            isinstance(record, dict)
            and isinstance(
                record.get("homography"),
                list,
            )
        )
    }

    missing_dense = [
        frame
        for frame in range(frame_count)
        if frame not in dense_by_frame
    ]

    if missing_dense:
        raise SystemExit(
            "Dense Atlas incompleto: "
            f"{len(missing_dense)} frames sin H."
        )

    homographies = [
        dense_by_frame[frame]["homography"]
        for frame in range(frame_count)
    ]

    cfg = FieldSpaceObserverConfig()

    print()
    print(
        "FIELD SPACE / OBSERVER V1 · FSO04"
    )
    print()
    print("Video:", args.video)
    print(
        f"image={image_width}×{image_height} · "
        f"fps={fps:g} · frames={frame_count}"
    )
    print(
        f"pitch={pitch_length:g}×"
        f"{pitch_width:g} m"
    )
    print(
        "canonical pitch="
        f"X=[{-pitch_length/2:.2f},"
        f"{pitch_length/2:.2f}] · "
        f"Y=[{-pitch_width/2:.2f},"
        f"{pitch_width/2:.2f}]"
    )
    print(
        f"dense H records={len(homographies)}"
    )
    print()

    sequence = (
        select_field_space_observer_sequence(
            homographies,
            image_width=image_width,
            image_height=image_height,
            pitch_length_m=pitch_length,
            pitch_width_m=pitch_width,
            config=cfg,
        )
    )

    if sequence.get("status") != "ok":
        print(
            json.dumps(
                clean(sequence),
                indent=2,
                ensure_ascii=False,
            )
        )
        raise SystemExit(1)

    records = sequence["records"]
    records_by_frame = {
        int(record["frame"]): record
        for record in records
    }

    invariant_failures = []

    for record in records:
        frame = int(record["frame"])
        polygon = (
            record.get(
                "visible_pitch_polygon_centered"
            )
            or []
        )

        if not polygon:
            continue

        projected = field_to_image(
            dense_by_frame[frame][
                "homography"
            ],
            polygon,
        )

        if projected.get("status") != "ok":
            invariant_failures.append(
                {
                    "frame": frame,
                    "reason": (
                        "polygon_projection_failed"
                    ),
                }
            )
            continue

        tolerance = (
            cfg.viewport_tolerance_px
        )

        for point, valid in zip(
            projected["points"],
            projected["valid"],
        ):
            if not valid:
                invariant_failures.append(
                    {
                        "frame": frame,
                        "reason": (
                            "invalid_polygon_vertex"
                        ),
                    }
                )
                break

            x, y = map(float, point)

            if (
                x < -tolerance
                or x > image_width + tolerance
                or y < -tolerance
                or y > image_height + tolerance
            ):
                invariant_failures.append(
                    {
                        "frame": frame,
                        "reason": (
                            "polygon_vertex_"
                            "outside_viewport"
                        ),
                        "image_point": [x, y],
                    }
                )
                break

    fractions = np.asarray(
        [
            float(record["visible_fraction"])
            for record in records
        ],
        dtype=np.float64,
    )

    areas = np.asarray(
        [
            float(
                record[
                    "visible_pitch_area_m2"
                ]
            )
            for record in records
        ],
        dtype=np.float64,
    )

    focus_points = []

    for record in records:
        focus = record.get(
            "visible_focus_field_centered"
        )

        if not isinstance(focus, dict):
            focus_points.append(None)
            continue

        focus_points.append(
            np.asarray(
                [
                    focus["x_m"],
                    focus["y_m"],
                ],
                dtype=np.float64,
            )
        )

    focus_jumps = np.asarray(
        [
            float(
                np.linalg.norm(
                    right - left
                )
            )
            for left, right in zip(
                focus_points[:-1],
                focus_points[1:],
            )
            if (
                left is not None
                and right is not None
            )
        ],
        dtype=np.float64,
    )

    zero_area_frames = [
        int(record["frame"])
        for record in records
        if float(
            record["visible_pitch_area_m2"]
        ) <= 1e-6
    ]

    branch_counts = {
        "+1": sum(
            1
            for record in records
            if int(record["branch_sign"]) == 1
        ),
        "-1": sum(
            1
            for record in records
            if int(record["branch_sign"]) == -1
        ),
    }

    boundary_names = (
        "left_goal_line",
        "right_goal_line",
        "top_touchline",
        "bottom_touchline",
    )

    boundary_counts = {
        name: sum(
            1
            for record in records
            if record[
                "boundary_visibility"
            ][name]["visible"]
        )
        for name in boundary_names
    }

    zone_counts: dict[str, int] = {}

    for record in records:
        zone = str(
            record.get(
                "semantic",
                {},
            ).get(
                "focus_x_zone"
            )
            or "unknown"
        )
        zone_counts[zone] = (
            zone_counts.get(zone, 0) + 1
        )

    print("SUMMARY")
    print(
        f"valid={len(records)}/{frame_count} · "
        f"invariant failures="
        f"{len(invariant_failures)} · "
        f"zero-area frames="
        f"{len(zero_area_frames)}"
    )
    print(
        f"branch switches="
        f"{sequence['branch_switch_count']} · "
        f"+1 frames={branch_counts['+1']} · "
        f"-1 frames={branch_counts['-1']}"
    )

    if len(fractions):
        print(
            "visible pitch fraction "
            f"min={100*np.min(fractions):.1f}% · "
            f"median="
            f"{100*np.median(fractions):.1f}% · "
            f"P90="
            f"{100*np.percentile(fractions,90):.1f}% · "
            f"max={100*np.max(fractions):.1f}%"
        )

    if len(areas):
        print(
            "visible pitch area "
            f"min={np.min(areas):.1f}m² · "
            f"median={np.median(areas):.1f}m² · "
            f"max={np.max(areas):.1f}m²"
        )

    if len(focus_jumps):
        print(
            "visible-focus frame motion "
            f"median="
            f"{np.median(focus_jumps):.3f}m · "
            f"P90="
            f"{np.percentile(focus_jumps,90):.3f}m · "
            f"max={np.max(focus_jumps):.3f}m"
        )

    print()
    print("FOCUS X ZONES")

    for zone, count in sorted(
        zone_counts.items()
    ):
        print(
            f"{zone:<14} "
            f"{count:>4}/{len(records)} "
            f"({100.0*count/max(len(records),1):5.1f}%)"
        )

    print()
    print("BOUNDARY VISIBILITY")

    for name in boundary_names:
        count = boundary_counts[name]
        print(
            f"{name:<18} "
            f"{count:>4}/{len(records)} "
            f"({100.0*count/max(len(records),1):5.1f}%)"
        )

    sample_frames = sorted(
        {
            0,
            30,
            92,
            899,
            1798,
            2526,
            2697,
            max(0, frame_count - 1),
        }
    )

    print()
    print("SAMPLE OBSERVER STATES")

    for frame in sample_frames:
        record = records_by_frame[frame]

        focus = (
            record.get(
                "visible_focus_field"
            )
            or {}
        )
        focus_centered = (
            record.get(
                "visible_focus_field_centered"
            )
            or {}
        )
        optical = (
            record.get(
                "optical_axis_ground_point"
            )
            or {}
        )
        bounds = (
            record.get(
                "visible_bounds_m"
            )
            or {}
        )
        semantic = (
            record.get("semantic")
            or {}
        )
        boundaries = record[
            "boundary_visibility"
        ]

        print(
            f"f={frame:>5} "
            f"t={timestamp(frame,fps)} · "
            f"branch={record['branch_sign']:+d} · "
            f"focus_pitch=("
            f"{fmt(focus.get('x_m'),1)},"
            f"{fmt(focus.get('y_m'),1)})m · "
            f"focus_center=("
            f"{fmt(focus_centered.get('x_m'),1)},"
            f"{fmt(focus_centered.get('y_m'),1)})m · "
            f"optical_pitch=("
            f"{fmt(optical.get('x_m'),1)},"
            f"{fmt(optical.get('y_m'),1)})m · "
            f"zone="
            f"{semantic.get('focus_x_zone','—')} · "
            f"visible="
            f"{100.0*record['visible_fraction']:.1f}% · "
            f"Xpitch=["
            f"{fmt(bounds.get('min_x_m'),1)},"
            f"{fmt(bounds.get('max_x_m'),1)}] · "
            f"goals="
            f"{'L' if boundaries['left_goal_line']['visible'] else '-'}"
            f"{'R' if boundaries['right_goal_line']['visible'] else '-'} · "
            f"support="
            f"{record['ray_support']['inside_count']}/"
            f"{record['ray_support']['sample_count']} · "
            f"margin="
            f"{record['branch_emission_margin']:+.2f}"
        )

    if zero_area_frames:
        print()
        print(
            "ZERO AREA FRAMES (first 30):",
            zero_area_frames[:30],
        )

    payload = {
        "status": "ok",
        "version": 1,
        "algorithm": (
            "field_space_observer_v1"
        ),
        "implementation": (
            "FSO04_centered_pitch_"
            "two_branch_temporal_viterbi"
        ),
        "video": {
            "path": str(args.video),
            "width": image_width,
            "height": image_height,
            "fps": fps,
            "frame_count": frame_count,
        },
        "pitch": {
            "length_m": pitch_length,
            "width_m": pitch_width,
            "canonical_origin": "centre",
            "canonical_x_range_m": [
                -0.5 * pitch_length,
                0.5 * pitch_length,
            ],
            "canonical_y_range_m": [
                -0.5 * pitch_width,
                0.5 * pitch_width,
            ],
        },
        "summary": {
            "valid_frame_count": len(records),
            "invariant_failure_count": (
                len(invariant_failures)
            ),
            "zero_area_frame_count": (
                len(zero_area_frames)
            ),
            "branch_switch_count": int(
                sequence[
                    "branch_switch_count"
                ]
            ),
            "branch_counts": branch_counts,
            "boundary_visible_frame_counts": (
                boundary_counts
            ),
            "focus_x_zone_counts": zone_counts,
        },
        "records": [
            {
                **record,
                "camera_source": (
                    dense_by_frame[
                        int(record["frame"])
                    ].get("source")
                ),
                "camera_method": (
                    dense_by_frame[
                        int(record["frame"])
                    ].get("method")
                ),
            }
            for record in records
        ],
        "invariant_failures": (
            invariant_failures
        ),
        "zero_area_frames": (
            zero_area_frames
        ),
    }

    args.json_out.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    args.json_out.write_text(
        json.dumps(
            clean(payload),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print("JSON:", args.json_out)
    print()
    print(
        "FSO04 now uses the SAME centered canonical "
        "pitch coordinate system as calibration and PREVIEW."
    )


if __name__ == "__main__":
    main()
