#!/usr/bin/env python3
"""FIELD SPACE / OBSERVER V1 · 2-D visual diagnostic.

Reads the FSO04 dense observer JSON and renders selected frames on a metric
105 x W football pitch.

This is deliberately OFFLINE diagnostic code:
- it does not modify CAMERA,
- it does not modify observer geometry,
- it does not touch the web UI,
- it does not use PEOPLE.

It answers one visual question:

    "Does the visible-pitch footprint/focus reported by Observer match the
     camera view we know from PREVIEW?"

Outputs
-------
One PNG per selected frame plus one contact sheet.

Default input:
    supervisor/data/reviews/
    benchmark_normal_2min_field_space_observer_v1.json

Default output directory:
    supervisor/data/reviews/
    benchmark_normal_2min_field_space_observer_diag_v1/

Coordinate convention
---------------------
The Observer JSON exposes UI/football coordinates:
    X = 0 .. pitch_length_m
    Y = 0 .. pitch_width_m

Those are used here directly.

The canonical homography remains centered internally in CAMERA:
    Xc = -L/2 .. +L/2
    Yc = -W/2 .. +W/2
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Circle, Rectangle


DEFAULT_OBSERVER = Path(
    "supervisor/data/reviews/"
    "benchmark_normal_2min_field_space_observer_v1.json"
)
DEFAULT_OUT_DIR = Path(
    "supervisor/data/reviews/"
    "benchmark_normal_2min_field_space_observer_diag_v1"
)
DEFAULT_FRAMES = (0, 30, 92, 899, 1798, 2526, 2697, 3595)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--observer",
        type=Path,
        default=DEFAULT_OBSERVER,
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
    )
    parser.add_argument(
        "--frames",
        type=int,
        nargs="*",
        default=list(DEFAULT_FRAMES),
        help="Frames to render.",
    )
    return parser.parse_args()


def timestamp(frame: int, fps: float) -> str:
    seconds = float(frame) / fps if fps > 0 else 0.0
    minute = int(seconds // 60)
    rest = seconds - minute * 60
    return f"{minute:02d}:{rest:05.2f}"


def fmt(value: Any, digits: int = 1) -> str:
    if value is None:
        return "—"
    return f"{float(value):.{digits}f}"


def draw_pitch_geometry(
    ax,
    *,
    length_m: float,
    width_m: float,
) -> None:
    """Draw standard pitch geometry using the project's pitch dimensions."""

    penalty_area_depth = 16.5
    goal_area_depth = 5.5
    goal_half_width = 3.66
    centre_circle_radius = 9.15
    penalty_spot_distance = 11.0

    penalty_y0 = width_m / 2.0 - (goal_half_width + penalty_area_depth)
    penalty_y1 = width_m / 2.0 + (goal_half_width + penalty_area_depth)
    goal_y0 = width_m / 2.0 - (goal_half_width + goal_area_depth)
    goal_y1 = width_m / 2.0 + (goal_half_width + goal_area_depth)

    # Outer boundary.
    ax.add_patch(
        Rectangle(
            (0.0, 0.0),
            length_m,
            width_m,
            fill=False,
            linewidth=1.5,
        )
    )

    # Halfway line.
    ax.plot(
        [length_m / 2.0, length_m / 2.0],
        [0.0, width_m],
        linewidth=1.0,
    )

    # Centre circle and centre mark.
    ax.add_patch(
        Circle(
            (length_m / 2.0, width_m / 2.0),
            centre_circle_radius,
            fill=False,
            linewidth=1.0,
        )
    )
    ax.scatter(
        [length_m / 2.0],
        [width_m / 2.0],
        s=12,
    )

    # Penalty areas.
    ax.add_patch(
        Rectangle(
            (0.0, penalty_y0),
            penalty_area_depth,
            penalty_y1 - penalty_y0,
            fill=False,
            linewidth=1.0,
        )
    )
    ax.add_patch(
        Rectangle(
            (length_m - penalty_area_depth, penalty_y0),
            penalty_area_depth,
            penalty_y1 - penalty_y0,
            fill=False,
            linewidth=1.0,
        )
    )

    # Goal areas.
    ax.add_patch(
        Rectangle(
            (0.0, goal_y0),
            goal_area_depth,
            goal_y1 - goal_y0,
            fill=False,
            linewidth=1.0,
        )
    )
    ax.add_patch(
        Rectangle(
            (length_m - goal_area_depth, goal_y0),
            goal_area_depth,
            goal_y1 - goal_y0,
            fill=False,
            linewidth=1.0,
        )
    )

    # Penalty spots.
    ax.scatter(
        [penalty_spot_distance, length_m - penalty_spot_distance],
        [width_m / 2.0, width_m / 2.0],
        s=10,
    )

    # Penalty arcs. Keep them approximate/diagnostic only.
    arc_radius = centre_circle_radius * 2.0
    ax.add_patch(
        Arc(
            (penalty_spot_distance, width_m / 2.0),
            arc_radius,
            arc_radius,
            angle=0.0,
            theta1=-53.0,
            theta2=53.0,
            linewidth=1.0,
        )
    )
    ax.add_patch(
        Arc(
            (length_m - penalty_spot_distance, width_m / 2.0),
            arc_radius,
            arc_radius,
            angle=0.0,
            theta1=127.0,
            theta2=233.0,
            linewidth=1.0,
        )
    )

    ax.set_xlim(-2.0, length_m + 2.0)
    ax.set_ylim(width_m + 2.0, -2.0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X · longitud del campo (m)")
    ax.set_ylabel("Y · anchura del campo (m)")
    ax.grid(False)


def draw_observer_record(
    ax,
    record: dict[str, Any],
    *,
    length_m: float,
    width_m: float,
    fps: float,
) -> None:
    draw_pitch_geometry(
        ax,
        length_m=length_m,
        width_m=width_m,
    )

    polygon = record.get("visible_pitch_polygon") or []
    if len(polygon) >= 3:
        xs = [float(point[0]) for point in polygon]
        ys = [float(point[1]) for point in polygon]
        ax.fill(
            xs,
            ys,
            alpha=0.22,
            label="Zona visible",
        )
        ax.plot(
            xs + [xs[0]],
            ys + [ys[0]],
            linewidth=1.8,
        )

    focus = record.get("visible_focus_field")
    if isinstance(focus, dict):
        ax.scatter(
            [float(focus["x_m"])],
            [float(focus["y_m"])],
            s=80,
            marker="o",
            label="Foco visible",
            zorder=5,
        )

    optical = record.get("optical_axis_ground_point")
    if isinstance(optical, dict):
        optical_x = float(optical["x_m"])
        optical_y = float(optical["y_m"])
        ax.scatter(
            [optical_x],
            [optical_y],
            s=55,
            marker="x",
            label="Eje óptico",
            zorder=5,
        )

    frame = int(record["frame"])
    semantic = record.get("semantic") or {}
    bounds = record.get("visible_bounds_m") or {}
    boundaries = record.get("boundary_visibility") or {}

    goals = (
        ("L" if boundaries.get("left_goal_line", {}).get("visible") else "-")
        + ("R" if boundaries.get("right_goal_line", {}).get("visible") else "-")
    )

    ax.set_title(
        f"Frame {frame} · {timestamp(frame, fps)}\n"
        f"visible={100.0 * float(record['visible_fraction']):.1f}% · "
        f"foco={semantic.get('focus_x_zone', '—')} · "
        f"X=[{fmt(bounds.get('min_x_m'))},{fmt(bounds.get('max_x_m'))}] m · "
        f"porterías={goals}"
    )

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.10),
        ncol=3,
        frameon=False,
    )


def save_single(
    record: dict[str, Any],
    *,
    length_m: float,
    width_m: float,
    fps: float,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 8))
    draw_observer_record(
        ax,
        record,
        length_m=length_m,
        width_m=width_m,
        fps=fps,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_contact_sheet(
    records: list[dict[str, Any]],
    *,
    length_m: float,
    width_m: float,
    fps: float,
    out_path: Path,
) -> None:
    if not records:
        return

    columns = 2
    rows = math.ceil(len(records) / columns)

    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(15, 5.6 * rows),
        squeeze=False,
    )

    flat_axes = [ax for row in axes for ax in row]

    for ax, record in zip(flat_axes, records):
        draw_observer_record(
            ax,
            record,
            length_m=length_m,
            width_m=width_m,
            fps=fps,
        )

    for ax in flat_axes[len(records):]:
        ax.axis("off")

    fig.suptitle(
        "FIELD SPACE / OBSERVER V1 · diagnóstico visual",
        fontsize=16,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()

    payload = json.loads(
        args.observer.read_text(encoding="utf-8")
    )

    if payload.get("status") != "ok":
        raise SystemExit("Observer JSON no está en estado ok.")

    implementation = str(payload.get("implementation") or "")
    if "FSO04" not in implementation:
        print(
            "WARNING: este diagnóstico fue diseñado para FSO04. "
            f"implementation={implementation!r}"
        )

    pitch = payload.get("pitch") or {}
    video = payload.get("video") or {}
    records = payload.get("records")

    if not isinstance(records, list):
        raise SystemExit("Observer JSON no contiene records.")

    length_m = float(pitch["length_m"])
    width_m = float(pitch["width_m"])
    fps = float(video.get("fps") or 0.0)

    records_by_frame = {
        int(record["frame"]): record
        for record in records
        if isinstance(record, dict)
    }

    selected: list[dict[str, Any]] = []

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print()
    print("FIELD SPACE / OBSERVER · 2-D VISUAL DIAGNOSTIC")
    print()
    print("Observer:", args.observer)
    print("Output:", args.out_dir)
    print(
        f"pitch={length_m:g}×{width_m:g}m · "
        f"frames requested={len(args.frames)}"
    )
    print()

    for frame in args.frames:
        record = records_by_frame.get(int(frame))

        if record is None:
            print(f"f={frame:>5} · unavailable")
            continue

        selected.append(record)

        filename = (
            f"observer_frame_{int(frame):05d}_"
            f"{timestamp(int(frame), fps).replace(':', '-')}.png"
        )
        out_path = args.out_dir / filename

        save_single(
            record,
            length_m=length_m,
            width_m=width_m,
            fps=fps,
            out_path=out_path,
        )

        focus = record.get("visible_focus_field") or {}
        bounds = record.get("visible_bounds_m") or {}
        semantic = record.get("semantic") or {}

        print(
            f"f={frame:>5} · "
            f"focus=({fmt(focus.get('x_m'))},"
            f"{fmt(focus.get('y_m'))})m · "
            f"zone={semantic.get('focus_x_zone','—')} · "
            f"visible={100.0*float(record['visible_fraction']):.1f}% · "
            f"X=[{fmt(bounds.get('min_x_m'))},"
            f"{fmt(bounds.get('max_x_m'))}] · "
            f"PNG={out_path}"
        )

    contact_sheet = args.out_dir / "observer_contact_sheet.png"
    save_contact_sheet(
        selected,
        length_m=length_m,
        width_m=width_m,
        fps=fps,
        out_path=contact_sheet,
    )

    print()
    print("CONTACT SHEET:", contact_sheet)
    print()
    print(
        "Visual rule: the shaded polygon must match the part of the real "
        "pitch seen in PREVIEW; the circular marker is the finite visible "
        "pitch centroid used as observer focus."
    )


if __name__ == "__main__":
    main()
