#!/usr/bin/env python3

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch


DEFAULT_CHECKPOINT = Path(
    "experiments/match_person_v1_rfdetr_2xl_v2/checkpoint_best_total.pth"
)

BENCHMARKS = {
    "easy": Path("input/benchmark_easy_30s.mp4"),
    "normal": Path("input/benchmark_normal_30s.mp4"),
    "hard": Path("input/benchmark_hard_30s.mp4"),
}

DEFAULT_OUTPUT_ROOT = Path("outputs/rfdetr_match_person_v1")
DEFAULT_THRESHOLD = 0.35
DEFAULT_RESOLUTION = 880


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the fine-tuned RF-DETR 2XL match_person_v1 checkpoint "
            "on the three fixed 30-second Real Jaen benchmark clips."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help=f"Fine-tuned checkpoint (default: {DEFAULT_CHECKPOINT}).",
    )
    parser.add_argument(
        "--benchmark",
        choices=tuple(BENCHMARKS.keys()),
        help="Run only one benchmark clip.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run easy, normal and hard at the selected resolution.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=DEFAULT_RESOLUTION,
        help="Square RF-DETR inference resolution (default: 880).",
    )
    parser.add_argument(
        "--both-resolutions",
        action="store_true",
        help="Run all selected clips at both 880 and 1280.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"Confidence threshold (default: {DEFAULT_THRESHOLD}).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Output root (default: {DEFAULT_OUTPUT_ROOT}).",
    )
    parser.add_argument(
        "--no-confidence",
        action="store_true",
        help="Draw only boxes/label, without confidence score.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.checkpoint.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {args.checkpoint}"
        )

    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be between 0 and 1.")

    if args.resolution <= 0:
        raise ValueError("--resolution must be greater than 0.")

    if not args.all and args.benchmark is None:
        raise ValueError(
            "Choose --all or --benchmark {easy,normal,hard}."
        )

    selected = (
        BENCHMARKS
        if args.all
        else {args.benchmark: BENCHMARKS[args.benchmark]}
    )

    missing = [
        path
        for path in selected.values()
        if not path.is_file()
    ]

    if missing:
        raise FileNotFoundError(
            "Missing benchmark clip(s):\n"
            + "\n".join(f"  - {path}" for path in missing)
        )


def load_model(checkpoint: Path, resolution: int):
    try:
        from rfdetr import RFDETR2XLarge
    except ImportError as exc:
        raise RuntimeError(
            "RFDETR2XLarge could not be imported. "
            "Activate the environment containing rfdetr-plus."
        ) from exc

    print()
    print("Loading fine-tuned RF-DETR 2XL...")
    print(f"Checkpoint: {checkpoint}")
    print("Classes: 1 (match_person)")
    print(f"Inference resolution: {resolution}x{resolution}")

    model = RFDETR2XLarge(
        pretrain_weights=str(checkpoint),
        num_classes=1,
        resolution=resolution,
    )

    return model


def draw_detection(
    frame: np.ndarray,
    xyxy,
    confidence: float,
    show_confidence: bool,
) -> None:
    x1, y1, x2, y2 = [
        int(round(float(v))) for v in xyxy
    ]

    height, width = frame.shape[:2]

    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(0, min(x2, width - 1))
    y2 = max(0, min(y2, height - 1))

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2,
    )

    if show_confidence:
        label = f"match_person {confidence:.2f}"
    else:
        label = "match_person"

    (text_w, text_h), baseline = cv2.getTextSize(
        label,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        1,
    )

    text_y = max(text_h + 6, y1 - 5)

    cv2.rectangle(
        frame,
        (x1, text_y - text_h - 5),
        (min(width - 1, x1 + text_w + 6), text_y + baseline),
        (0, 255, 0),
        -1,
    )

    cv2.putText(
        frame,
        label,
        (x1 + 3, text_y - 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )


def create_writer(
    output_path: Path,
    fps: float,
    width: int,
    height: int,
):
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (width, height),
    )

    if not writer.isOpened():
        raise RuntimeError(
            f"Could not open output writer: {output_path}"
        )

    return writer


def run_video(
    model,
    benchmark_name: str,
    input_path: Path,
    output_path: Path,
    resolution: int,
    threshold: float,
    show_confidence: bool,
) -> dict:
    cap = cv2.VideoCapture(str(input_path))

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open input video: {input_path}"
        )

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0:
        fps = 30.0

    writer = create_writer(
        output_path,
        fps,
        width,
        height,
    )

    print()
    print("=" * 72)
    print(f"Benchmark: {benchmark_name}")
    print(f"Input: {input_path}")
    print(
        f"Video: {width}x{height} @ {fps:.3f} FPS "
        f"| frames={total_frames}"
    )
    print(
        f"Model resolution: {resolution} "
        f"| threshold={threshold:.2f}"
    )
    print(f"Output: {output_path}")
    print("=" * 72)

    frame_index = 0
    total_detections = 0
    frames_with_detections = 0
    start = time.perf_counter()

    try:
        while True:
            ok, frame_bgr = cap.read()

            if not ok:
                break

            frame_rgb = cv2.cvtColor(
                frame_bgr,
                cv2.COLOR_BGR2RGB,
            )

            detections = model.predict(
                frame_rgb,
                threshold=threshold,
                shape=(resolution, resolution),
            )

            count = len(detections)
            total_detections += count

            if count > 0:
                frames_with_detections += 1

            confidences = (
                detections.confidence
                if detections.confidence is not None
                else np.ones(count, dtype=np.float32)
            )

            for xyxy, confidence in zip(
                detections.xyxy,
                confidences,
            ):
                draw_detection(
                    frame_bgr,
                    xyxy,
                    float(confidence),
                    show_confidence=show_confidence,
                )

            cv2.putText(
                frame_bgr,
                f"match_person: {count}",
                (20, 38),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            writer.write(frame_bgr)
            frame_index += 1

            if (
                frame_index == 1
                or frame_index % 100 == 0
                or frame_index == total_frames
            ):
                elapsed = time.perf_counter() - start
                speed = (
                    frame_index / elapsed
                    if elapsed > 0
                    else 0.0
                )

                print(
                    f"\r  {frame_index}/{total_frames} "
                    f"| {speed:.2f} FPS "
                    f"| current={count:2d} detections",
                    end="",
                    flush=True,
                )

    finally:
        cap.release()
        writer.release()

    elapsed = time.perf_counter() - start
    processing_fps = (
        frame_index / elapsed
        if elapsed > 0
        else 0.0
    )
    avg_detections = (
        total_detections / frame_index
        if frame_index
        else 0.0
    )
    coverage = (
        frames_with_detections / frame_index
        if frame_index
        else 0.0
    )

    print()
    print(
        f"Finished {benchmark_name}: "
        f"{frame_index} frames in {elapsed:.2f}s "
        f"({processing_fps:.2f} FPS)"
    )
    print(
        f"Average match_person/frame: "
        f"{avg_detections:.2f}"
    )
    print(
        f"Frames with >=1 detection: "
        f"{coverage * 100:.2f}%"
    )

    return {
        "benchmark": benchmark_name,
        "resolution": resolution,
        "threshold": threshold,
        "frames": frame_index,
        "seconds": elapsed,
        "processing_fps": processing_fps,
        "total_detections": total_detections,
        "avg_detections_per_frame": avg_detections,
        "frames_with_detections_pct": coverage * 100.0,
        "output": str(output_path),
    }


def main() -> None:
    args = parse_args()
    validate_args(args)

    print("=== RF-DETR 2XL MATCH_PERSON_V1 BENCHMARK ===")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    selected = (
        BENCHMARKS
        if args.all
        else {args.benchmark: BENCHMARKS[args.benchmark]}
    )

    resolutions = (
        [880, 1280]
        if args.both_resolutions
        else [args.resolution]
    )

    all_results = []

    for resolution in resolutions:
        model = load_model(
            args.checkpoint,
            resolution,
        )

        for name, input_path in selected.items():
            output_dir = (
                args.output_root
                / f"{resolution}"
            )

            output_path = (
                output_dir
                / (
                    f"rfdetr_match_person_v1_"
                    f"{resolution}_{name}.mp4"
                )
            )

            result = run_video(
                model=model,
                benchmark_name=name,
                input_path=input_path,
                output_path=output_path,
                resolution=resolution,
                threshold=args.threshold,
                show_confidence=not args.no_confidence,
            )

            all_results.append(result)

        del model

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print()
    print("=== SUMMARY ===")

    for result in all_results:
        print(
            f"{result['benchmark']:>6} "
            f"@ {result['resolution']:4d}: "
            f"{result['processing_fps']:.2f} FPS, "
            f"{result['avg_detections_per_frame']:.2f} "
            f"match_person/frame"
        )

    print()
    print("Done.")


if __name__ == "__main__":
    main()
