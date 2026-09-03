"""Run CAMERA Anchor Planner V1 on the 5-minute Córdoba clip.

Read-only diagnostic:
- does not read HUMAN calibrations;
- does not edit reviews;
- does not run Dense;
- does not modify CAMERA artifacts.

Usage from repository root:
    python -m supervisor.check_camera_anchor_planner_v1
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time

from supervisor.field_camera_anchor_planner_v1 import (
    CameraAnchorPlannerConfig,
    analyse_camera_anchor_planner,
)


ROOT = Path(__file__).resolve().parents[1]

CORDOBA_VIDEO = (
    ROOT
    / "supervisor"
    / "data"
    / "library"
    / "videos"
    / "managed_upload_559cfa321d00455d9ef8b6465dc23594.mp4"
)

CORDOBA_TRAJECTORY = (
    ROOT
    / "supervisor"
    / "data"
    / "reviews"
    / "asset_7fbc47b7b83647bc8fd52e6a676c9425_camera_trajectory_v1.json"
)

OUTPUT = (
    ROOT
    / "supervisor"
    / "data"
    / "reviews"
    / "asset_7fbc47b7b83647bc8fd52e6a676c9425_camera_anchor_planner_v1.json"
)


def _print_plan(label, plan):
    print()
    print(label)
    print("-" * len(label))
    print(
        "anchors:",
        plan.get("anchor_count"),
    )
    print(
        "coverage:",
        f"{100 * float(plan.get('coverage_fraction') or 0):.1f}%",
    )
    print(
        "states:",
        f"{plan.get('covered_state_count')}"
        f"/{plan.get('total_state_count')}",
    )
    print(
        "support histogram:",
        plan.get("support_histogram"),
    )

    anchors = plan.get("anchors", [])
    print("frames:", anchors)

    print()
    print("Top selections:")

    for item in plan.get("recommendations", [])[:20]:
        print(
            f"  #{int(item['rank']):>2}"
            f" · frame {int(item['frame']):>5}"
            f" · gain={int(item['gain_count']):>3}"
            f" · quality={float(item['quality_score']):.3f}"
        )

    uncovered = plan.get("uncovered_frames", [])
    if uncovered:
        print()
        print(
            "UNcovered:",
            uncovered[:50],
            "..." if len(uncovered) > 50 else "",
        )


def main():
    print("=" * 90)
    print("CAMERA ANCHOR PLANNER V1 · CÓRDOBA")
    print("=" * 90)
    print()
    print("VIDEO:")
    print(CORDOBA_VIDEO)
    print()
    print("TRAJECTORY:")
    print(CORDOBA_TRAJECTORY)
    print()

    if not CORDOBA_VIDEO.exists():
        print("ERROR: no existe el vídeo.")
        return 2

    if not CORDOBA_TRAJECTORY.exists():
        print(
            "ERROR: no existe Camera Trajectory V1."
        )
        return 2

    print(
        "IMPORTANTE: el planner NO recibe los HUMAN "
        "0/580/3450/3805/5101/7200."
    )
    print(
        "Los anchors existentes quedan ocultos "
        "durante este experimento."
    )
    print()

    cfg = CameraAnchorPlannerConfig(
        analysis_width=960,
        geometry_neighbors_per_state=8,
        temporal_neighbors_each_side=2,
        minimum_calibratability=0.16,
    )

    started = time.perf_counter()

    result = analyse_camera_anchor_planner(
        CORDOBA_VIDEO,
        CORDOBA_TRAJECTORY,
        config=cfg,
    )

    elapsed = time.perf_counter() - started

    print()
    print("=" * 90)
    print("RESULTADO")
    print("=" * 90)
    print("status:", result.get("status"))
    print("tiempo:", f"{elapsed:.1f}s")

    if result.get("status") != "ok":
        print(json.dumps(result, indent=2))
        return 1

    print(
        "trajectory keyframes:",
        result["trajectory_keyframe_count"],
    )
    print(
        "viable states:",
        result["viable_state_count"],
    )
    print(
        "candidate pairs:",
        result["candidate_pair_count"],
    )
    print(
        "verified direct edges:",
        result["verified_edge_count"],
    )

    if result["candidate_pair_count"]:
        ratio = (
            result["verified_edge_count"]
            / result["candidate_pair_count"]
        )
        print(
            "verified edge ratio:",
            f"{100 * ratio:.1f}%",
        )

    _print_plan(
        "PLAN 1× · soporte directo mínimo",
        result["plan_1x"],
    )

    _print_plan(
        "PLAN 2× · redundancia directa",
        result["plan_2x"],
    )

    print()
    print("ESTADOS MÁS DÉBILES DEL GRAFO")
    print("--------------------------------")

    for item in result.get("weak_states", [])[:20]:
        print(
            f"frame {int(item['frame']):>5}"
            f" · neighbors={int(item['direct_neighbor_count']):>2}"
            f" · quality={float(item['quality_score']):.3f}"
        )

    # Diagnostic artifact only. It contains no HUMAN calibration data.
    OUTPUT.write_text(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print("Diagnóstico guardado en:")
    print(OUTPUT)
    print()
    print(
        "No se modificó review, HUMAN, "
        "Relocalization, Dense ni FieldWorkflow."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
