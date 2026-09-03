"""Check CAMERA Anchor Planner V2 on the existing Córdoba V1 graph.

Fast/read-only diagnostic:
- reuses the V1 JSON;
- does NOT rerun SIFT/RANSAC;
- does NOT read HUMAN calibrations;
- does NOT edit review/CAMERA/Dense;
- discovers camera-state regions at multiple graph resolutions.

Usage:
    python -m supervisor.check_camera_anchor_planner_v2
"""

from __future__ import annotations

import json
from pathlib import Path
import time

from supervisor.field_camera_anchor_planner_v2 import (
    CameraAnchorPlannerV2Config,
    analyse_camera_anchor_planner_v2,
)


ROOT = Path(__file__).resolve().parents[1]

ASSET = "asset_7fbc47b7b83647bc8fd52e6a676c9425"

V1_RESULT = (
    ROOT
    / "supervisor"
    / "data"
    / "reviews"
    / f"{ASSET}_camera_anchor_planner_v1.json"
)

OUTPUT = (
    ROOT
    / "supervisor"
    / "data"
    / "reviews"
    / f"{ASSET}_camera_anchor_planner_v2.json"
)


def _nearest_state_frame(regions, target):
    candidates = []
    for region in regions:
        for frame in region.get("frames", []):
            candidates.append(
                (
                    abs(int(frame) - int(target)),
                    int(frame),
                    region,
                )
            )

    if not candidates:
        return None

    _, frame, region = min(
        candidates,
        key=lambda item: (
            item[0],
            item[1],
        ),
    )

    return {
        "frame": frame,
        "region": region,
    }


def main():
    print("=" * 92)
    print("CAMERA ANCHOR PLANNER V2 · CAMERA REGIONS · CÓRDOBA")
    print("=" * 92)
    print()
    print("INPUT V1:")
    print(V1_RESULT)
    print()
    print(
        "V2 NO conoce HUMAN, Dense ni Relocalization. "
        "Solo reutiliza el grafo visual sparse de V1."
    )
    print()

    if not V1_RESULT.exists():
        print(
            "ERROR: falta el resultado de V1. "
            "Ejecuta primero check_camera_anchor_planner_v1."
        )
        return 2

    cfg = CameraAnchorPlannerV2Config(
        minimum_regions=2,
        maximum_regions=16,
        local_neighbors_for_scale=3,
        local_scale_percentile=75.0,
        modularity_plateau_tolerance=0.012,
    )

    started = time.perf_counter()

    result = analyse_camera_anchor_planner_v2(
        V1_RESULT,
        config=cfg,
    )

    elapsed = time.perf_counter() - started

    print("status:", result.get("status"))
    print("tiempo:", f"{elapsed:.3f}s")

    if result.get("status") != "ok":
        print(json.dumps(result, indent=2))
        return 1

    print()
    print("GRAPH")
    print("-----")
    print("states:", result["state_count"])
    print(
        "verified edges:",
        result["verified_edge_count"],
    )
    print(
        "isolated frames:",
        result["isolated_frames"],
    )
    print(
        "learned local scale median:",
        f"{result['learned_graph_scales']['median_px']:.1f}px",
    )
    print(
        "learned local scale p90:",
        f"{result['learned_graph_scales']['p90_px']:.1f}px",
    )

    print()
    print("REGION RESOLUTION SWEEP")
    print("-----------------------")
    print(
        f"{'K':>3} "
        f"{'MODULARITY':>11} "
        f"{'N-CUT':>9} "
        f"{'REGION SIZES':<35} "
        f"REPRESENTATIVES"
    )
    print("-" * 115)

    for partition in result["partition_sweep"]:
        sizes = ",".join(
            str(int(value))
            for value in partition["region_sizes"]
        )
        representatives = ",".join(
            str(int(value))
            for value in partition[
                "representative_frames"
            ]
        )

        marker = (
            "  <=="
            if int(partition["region_count"])
            == int(result["recommended_region_count"])
            else ""
        )

        print(
            f"{int(partition['region_count']):>3} "
            f"{float(partition['modularity']):>11.4f} "
            f"{float(partition['normalized_cut']):>9.4f} "
            f"{sizes:<35.35} "
            f"{representatives}{marker}"
        )

    print()
    print("TOP EIGENGAPS · diagnostic only")
    print("--------------------------------")

    for item in result["top_eigengaps"][:8]:
        print(
            f"after K={int(item['after_components']):>2}"
            f" · gap={float(item['gap']):.5f}"
        )

    print()
    print("=" * 92)
    print(
        "RECOMMENDED REGION PLAN "
        f"· K={result['recommended_region_count']}"
    )
    print("=" * 92)
    print(
        "modularity:",
        f"{result['recommended_modularity']:.4f}",
    )
    print(
        "normalized cut:",
        f"{result['recommended_normalized_cut']:.4f}",
    )
    print(
        "representatives:",
        result["recommended_representative_frames"],
    )

    print()
    print("REGIONS")
    print("-------")

    for region in result["recommended_regions"]:
        representative = region["representative"]

        print(
            f"R{int(region['temporal_rank']):02}"
            f" · states={int(region['state_count']):>3}"
            f" · frames={int(region['frame_min'])}"
            f"–{int(region['frame_max'])}"
            f" · cohesion={100 * float(region['cohesion_fraction']):5.1f}%"
            f" · HUMAN candidate={int(representative['frame']):>5}"
            f" · score={float(representative['score']):.3f}"
            f" · quality={float(representative['quality_score']):.3f}"
            f" · centrality={float(representative['centrality']):.3f}"
        )

        alternates = ", ".join(
            (
                f"{int(item['frame'])}"
                f"(s={float(item['score']):.2f})"
            )
            for item in region.get("alternates", [])[:5]
        )
        print(
            f"      alternates: {alternates}"
        )

    # Post-hoc probes ONLY for us to interpret the blind plan.
    # They do not affect clustering, K, representatives or scores.
    print()
    print("POST-HOC PROBES · never used by planner")
    print("-----------------------------------------")

    for target in (0, 32, 580, 3450, 3805, 5101, 5980, 7200):
        located = _nearest_state_frame(
            result["recommended_regions"],
            target,
        )
        if located is None:
            continue

        region = located["region"]
        representative = region["representative"]

        print(
            f"probe {target:>5}"
            f" -> nearest state {int(located['frame']):>5}"
            f" -> R{int(region['temporal_rank']):02}"
            f" -> representative "
            f"{int(representative['frame'])}"
        )

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
    print("Guardado:")
    print(OUTPUT)
    print()
    print(
        "No se modificó review, HUMAN, CAMERA, "
        "Relocalization ni Dense."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
