#!/usr/bin/env python3
"""Run Active Anchor Selector V1 on benchmark_normal_2min transfer failures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from supervisor.field_camera_active_anchor import (
    ActiveAnchorConfig,
    select_next_active_anchor,
)


DEFAULT_VIDEO = Path("input/benchmark_normal_2min.mp4")
DEFAULT_TRANSFER = Path(
    "supervisor/data/reviews/"
    "benchmark_normal_2min_camera_anchor_transfer_v1.json"
)
DEFAULT_OUT = Path(
    "supervisor/data/reviews/"
    "benchmark_normal_2min_active_anchor_v1.json"
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
            str(key): clean(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    return value


def timestamp(frame: int, fps: float) -> str:
    seconds = float(frame) / fps if fps > 0 else 0.0
    minute = int(seconds // 60)
    rest = seconds - minute * 60
    return f"{minute:02d}:{rest:05.2f}"


def fmt(value: Any, digits: int = 2) -> str:
    return "—" if value is None else f"{float(value):.{digits}f}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument(
        "--transfer",
        type=Path,
        default=DEFAULT_TRANSFER,
    )
    parser.add_argument("--json-out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    transfer = json.loads(
        args.transfer.read_text(encoding="utf-8")
    )

    unresolved = [
        int(result["frame"])
        for result in transfer.get("results", [])
        if isinstance(result, dict)
        and result.get("status") != "ok"
    ]

    print()
    print("ACTIVE ANCHOR SELECTOR V1")
    print("Target:", args.video)
    print(
        f"Current source HUMAN anchors="
        f"{transfer.get('anchor_count','—')}"
    )
    print(
        f"Current target coverage="
        f"{transfer.get('covered_target_count','—')}/"
        f"{transfer.get('target_count','—')} "
        f"({100.0*float(transfer.get('coverage_fraction') or 0):.1f}%)"
    )
    print("Unresolved frames:", unresolved)
    print()

    result = select_next_active_anchor(
        args.video,
        unresolved,
        config=ActiveAnchorConfig(),
    )

    if result.get("status") == "complete":
        print("No unresolved frames. No HUMAN needed.")
        return
    if result.get("status") != "ok":
        print(
            json.dumps(
                clean(result),
                indent=2,
                ensure_ascii=False,
            )
        )
        raise SystemExit(1)

    fps = float(result["video"]["fps"])

    print("RANKED HUMAN CANDIDATES")
    for recommendation in result["recommendations"]:
        covered = recommendation["expected_unresolved_covered"]
        print(
            f"#{recommendation['rank']} "
            f"f={recommendation['frame']:>5} "
            f"t={timestamp(recommendation['frame'],fps)} · "
            f"expected gain={recommendation['expected_gain_count']}/"
            f"{result['unresolved_count']} · "
            f"covers={covered} · "
            f"median pair inliers="
            f"{recommendation['median_pair_inliers']:.0f} · "
            f"ratio={recommendation['median_pair_inlier_ratio']:.2f} · "
            f"reproj="
            f"{fmt(recommendation['median_pair_reprojection_px'])}px"
        )

    print()
    print("NEXT HUMAN")
    best = result["recommendations"][0]
    print(
        f"Calibrate ONLY frame {best['frame']} "
        f"({timestamp(best['frame'],fps)})."
    )
    print(
        f"Expected direct unresolved coverage: "
        f"{best['expected_gain_count']}/"
        f"{result['unresolved_count']} "
        f"→ frames {best['expected_unresolved_covered']}"
    )
    print(
        "After adding that HUMAN, RECOMPUTE transfer before choosing another. "
        "Do not blindly follow the full plan."
    )

    print()
    print("GREEDY PLAN ESTIMATE ONLY")
    for index, step in enumerate(
        result["greedy_plan_estimate"],
        start=1,
    ):
        print(
            f"{index}. f={step['frame']} "
            f"t={timestamp(step['frame'],fps)} · "
            f"gain={step['gain_count']} · "
            f"covers={step['covers']}"
        )

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(
        json.dumps(
            clean(result),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print()
    print("JSON:", args.json_out)


if __name__ == "__main__":
    main()
