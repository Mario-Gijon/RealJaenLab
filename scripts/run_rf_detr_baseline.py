#!/usr/bin/env python3

from __future__ import annotations

import argparse
import gc
import time
from collections import Counter
from pathlib import Path

import cv2
import torch
from rfdetr import RFDETRLarge


MODELS = {
    "v14_large": Path("weights/models/football_detector_exp004_v14_large.pth"),
    "v20_large": Path("weights/models/football_detector_exp003_v20_large.pth"),
}

BENCHMARKS = {
    "easy": Path("input/benchmark_easy_30s.mp4"),
    "normal": Path("input/benchmark_normal_30s.mp4"),
    "hard": Path("input/benchmark_hard_30s.mp4"),
}

OUTPUT_ROOT = Path("outputs/rf_detr_baseline")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run RF-DETR Large baselines on RealJaenLab benchmark videos."
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run v14 Large and v20 Large on easy, normal and hard benchmarks.",
    )
    parser.add_argument(
        "--model",
        choices=MODELS.keys(),
        help="Model to run when --all is not used.",
    )
    parser.add_argument(
        "--input",
        help="Benchmark name (easy, normal, hard) or path to a video.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.35,
        help="Detection confidence threshold (default: 0.35).",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device, for example cuda, cuda:0 or cpu.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output videos if they already exist.",
    )

    args = parser.parse_args()

    if args.all and (args.model or args.input):
        parser.error("--all cannot be combined with --model or --input.")

    if not args.all and (not args.model or not args.input):
        parser.error("Use --all, or provide both --model and --input.")

    if not 0.0 <= args.threshold <= 1.0:
        parser.error("--threshold must be between 0 and 1.")

    return args


def resolve_input(value: str) -> tuple[str, Path]:
    if value in BENCHMARKS:
        return value, BENCHMARKS[value]

    path = Path(value)
    return path.stem, path


def get_class_names(model: RFDETRLarge) -> dict[int, str]:
    names = getattr(model, "class_names", {}) or {}

    if isinstance(names, dict):
        result = {}
        for key, value in names.items():
            try:
                result[int(key)] = str(value)
            except (TypeError, ValueError):
                pass
        return result

    if isinstance(names, (list, tuple)):
        return {index: str(value) for index, value in enumerate(names)}

    return {}


def draw_detections(frame, detections, class_names):
    counts = Counter()

    boxes = getattr(detections, "xyxy", None)
    confidences = getattr(detections, "confidence", None)
    class_ids = getattr(detections, "class_id", None)

    if boxes is None or len(boxes) == 0:
        return frame, counts

    if confidences is None:
        confidences = [0.0] * len(boxes)

    if class_ids is None:
        class_ids = [-1] * len(boxes)

    height, width = frame.shape[:2]
    thickness = max(1, round(min(width, height) / 700))
    font_scale = max(0.4, min(width, height) / 1800)

    palette = (
        (255, 120, 60),
        (60, 200, 255),
        (80, 220, 100),
        (220, 80, 220),
        (80, 120, 255),
        (220, 220, 80),
        (180, 100, 255),
        (255, 180, 80),
    )

    for box, confidence, raw_class_id in zip(boxes, confidences, class_ids):
        class_id = int(raw_class_id)
        class_name = class_names.get(class_id, f"class_{class_id}")
        counts[class_name] += 1

        x1, y1, x2, y2 = map(int, box)
        x1 = max(0, min(x1, width - 1))
        x2 = max(0, min(x2, width - 1))
        y1 = max(0, min(y1, height - 1))
        y2 = max(0, min(y2, height - 1))

        color = palette[class_id % len(palette)]

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            thickness,
        )

        label = f"{class_name} {float(confidence):.2f}"

        (text_width, text_height), baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            thickness,
        )

        text_y = max(text_height + baseline + 4, y1)

        cv2.rectangle(
            frame,
            (x1, text_y - text_height - baseline - 4),
            (min(width - 1, x1 + text_width + 4), text_y),
            color,
            -1,
        )

        cv2.putText(
            frame,
            label,
            (x1 + 2, text_y - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            thickness,
            cv2.LINE_AA,
        )

    return frame, counts


def load_model(model_name: str, device: str) -> RFDETRLarge:
    weights = MODELS[model_name]

    if not weights.is_file():
        raise FileNotFoundError(f"Model weights not found: {weights}")

    print(f"\nLoading model: {model_name}")
    print(f"Weights: {weights}")
    print(f"Device: {device}")

    model = RFDETRLarge(
        pretrain_weights=str(weights),
        device=device,
    )

    print(f"Classes: {get_class_names(model)}")

    return model


def run_video(
    model,
    model_name: str,
    benchmark_name: str,
    input_path: Path,
    threshold: float,
    device: str,
    overwrite: bool,
) -> None:
    if not input_path.is_file():
        raise FileNotFoundError(f"Input video not found: {input_path}")

    output_dir = OUTPUT_ROOT / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"rf_detr_{model_name}_{benchmark_name}.mp4"

    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"{output_path} already exists. Use --overwrite to replace it."
        )

    capture = cv2.VideoCapture(str(input_path))

    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {input_path}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0 or width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError(f"Invalid video metadata: {input_path}")

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not create output video: {output_path}")

    class_names = get_class_names(model)

    processed_frames = 0
    total_detections = 0
    class_counts = Counter()
    inference_time = 0.0

    started = time.perf_counter()
    progress_interval = max(1, round(fps * 10))

    print(f"\nRunning {model_name} / {benchmark_name}")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Video: {width}x{height} @ {fps:.3f} FPS")
    print(f"Frames: {total_frames}")
    print(f"Threshold: {threshold:.2f}")

    try:
        while True:
            ok, frame = capture.read()

            if not ok:
                break

            # RF-DETR expects RGB images. OpenCV reads BGR.
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if device.startswith("cuda"):
                torch.cuda.synchronize()

            inference_started = time.perf_counter()
            detections = model.predict(rgb, threshold=threshold)

            if device.startswith("cuda"):
                torch.cuda.synchronize()

            inference_time += time.perf_counter() - inference_started

            annotated_frame, frame_counts = draw_detections(
                frame,
                detections,
                class_names,
            )

            total_detections += sum(frame_counts.values())
            class_counts.update(frame_counts)

            writer.write(annotated_frame)
            processed_frames += 1

            if (
                processed_frames % progress_interval == 0
                or processed_frames == total_frames
            ):
                progress = (
                    processed_frames / total_frames * 100
                    if total_frames
                    else 0.0
                )

                print(
                    f"\rProgress: {processed_frames}/{total_frames} "
                    f"({progress:5.1f}%)",
                    end="",
                    flush=True,
                )

    finally:
        capture.release()
        writer.release()

    elapsed = time.perf_counter() - started

    inference_fps = (
        processed_frames / inference_time
        if inference_time
        else 0.0
    )

    end_to_end_fps = (
        processed_frames / elapsed
        if elapsed
        else 0.0
    )

    average_detections = (
        total_detections / processed_frames
        if processed_frames
        else 0.0
    )

    print("\n")
    print("Summary")
    print(f"  Model: {model_name}")
    print(f"  Benchmark: {benchmark_name}")
    print(f"  Processed frames: {processed_frames}")
    print(f"  Processing time: {elapsed:.2f} s")
    print(f"  Inference FPS: {inference_fps:.2f}")
    print(f"  End-to-end FPS: {end_to_end_fps:.2f}")
    print(f"  Average detections/frame: {average_detections:.2f}")
    print(f"  Total detections: {total_detections}")
    print(f"  Detections by class: {dict(class_counts)}")
    print(f"  Output: {output_path}")


def release_model(model) -> None:
    del model
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_all(args: argparse.Namespace) -> None:
    for model_name in ("v14_large", "v20_large"):
        model = load_model(model_name, args.device)

        try:
            for benchmark_name in ("easy", "normal", "hard"):
                run_video(
                    model=model,
                    model_name=model_name,
                    benchmark_name=benchmark_name,
                    input_path=BENCHMARKS[benchmark_name],
                    threshold=args.threshold,
                    device=args.device,
                    overwrite=args.overwrite,
                )

        finally:
            release_model(model)


def run_single(args: argparse.Namespace) -> None:
    benchmark_name, input_path = resolve_input(args.input)

    model = load_model(args.model, args.device)

    try:
        run_video(
            model=model,
            model_name=args.model,
            benchmark_name=benchmark_name,
            input_path=input_path,
            threshold=args.threshold,
            device=args.device,
            overwrite=args.overwrite,
        )

    finally:
        release_model(model)


def main() -> None:
    args = parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA device requested ({args.device}), but CUDA is not available."
        )

    if args.all:
        run_all(args)
    else:
        run_single(args)


if __name__ == "__main__":
    main()