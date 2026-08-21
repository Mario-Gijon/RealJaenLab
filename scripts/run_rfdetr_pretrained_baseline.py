#!/usr/bin/env python3

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import torch
from rfdetr.assets.coco_classes import COCO_CLASSES


BENCHMARKS = {
    "easy": Path("input/benchmark_easy_30s.mp4"),
    "normal": Path("input/benchmark_normal_30s.mp4"),
    "hard": Path("input/benchmark_hard_30s.mp4"),
}

OUTPUT_ROOT = Path("outputs/rfdetr_pretrained_baseline")
PERSON_CLASS_ID = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run pretrained COCO RF-DETR models on the RealJaenLab "
            "30-second person-detection benchmarks."
        )
    )

    parser.add_argument(
        "--model",
        choices=("2xl", "large"),
        default="2xl",
        help="RF-DETR pretrained model to use (default: 2xl).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run easy, normal and hard 30-second benchmarks.",
    )
    parser.add_argument(
        "--input",
        help="Benchmark name (easy, normal, hard) or path to a video.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.30,
        help="Person confidence threshold (default: 0.30).",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device, e.g. cuda, cuda:0 or cpu.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help=(
            "Optimize model for FP16 inference. Leave disabled for the "
            "first quality baseline; useful if 2XL is too slow or runs out of VRAM."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output videos if they already exist.",
    )

    args = parser.parse_args()

    if args.all and args.input:
        parser.error("--all cannot be combined with --input.")

    if not args.all and not args.input:
        parser.error("Use --all or provide --input.")

    if not 0.0 <= args.threshold <= 1.0:
        parser.error("--threshold must be between 0 and 1.")

    return args


def resolve_input(value: str) -> tuple[str, Path]:
    if value in BENCHMARKS:
        return value, BENCHMARKS[value]

    path = Path(value)
    return path.stem, path


def load_model(model_key: str, device: str, fp16: bool):
    if model_key == "large":
        from rfdetr import RFDETRLarge

        print("\nLoading pretrained RF-DETR Large (COCO)")
        model = RFDETRLarge(device=device)
        model_label = "rfdetr_large_pretrained"

    else:
        try:
            from rfdetr_plus import RFDETR2XLarge
        except ImportError as exc:
            raise RuntimeError(
                "RF-DETR 2XL requires the rfdetr-plus package.\n"
                "Install it with:\n"
                "  pip install -U rfdetr-plus\n"
                "Then run this script again."
            ) from exc

        print("\nLoading pretrained RF-DETR 2XLarge (COCO)")
        print(
            "This model requires acceptance of Roboflow's "
            "Platform Model License."
        )

        model = RFDETR2XLarge(
            accept_platform_model_license=True,
            device=device,
        )
        model_label = "rfdetr_2xlarge_pretrained"

    print(f"Device: {device}")
    print(f"Person COCO class id: {PERSON_CLASS_ID}")
    print(f"Person label: {COCO_CLASSES[PERSON_CLASS_ID]}")

    if fp16:
        print("Optimizing for FP16 inference...")
        model.optimize_for_inference(dtype=torch.float16)
        print("Precision: FP16 optimized")
    else:
        print("Precision: default / non-optimized baseline")

    return model, model_label


def draw_person_detections(frame, boxes, confidences):
    height, width = frame.shape[:2]

    thickness = max(1, round(min(width, height) / 700))
    font_scale = max(0.4, min(width, height) / 1800)
    color = (80, 220, 100)

    count = 0

    for box, confidence in zip(boxes, confidences):
        count += 1

        x1, y1, x2, y2 = map(int, box)

        x1 = max(0, min(x1, width - 1))
        x2 = max(0, min(x2, width - 1))
        y1 = max(0, min(y1, height - 1))
        y2 = max(0, min(y2, height - 1))

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            thickness,
        )

        label = f"person {float(confidence):.2f}"

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

    return frame, count


def run_video(
    model,
    model_label: str,
    benchmark_name: str,
    input_path: Path,
    threshold: float,
    device: str,
    overwrite: bool,
) -> None:
    if not input_path.is_file():
        raise FileNotFoundError(f"Input video not found: {input_path}")

    output_dir = OUTPUT_ROOT / model_label
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{model_label}_{benchmark_name}.mp4"

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

    processed_frames = 0
    total_person_detections = 0
    inference_time = 0.0

    started = time.perf_counter()
    progress_interval = max(1, round(fps * 5))

    print(f"\nRunning {model_label} / {benchmark_name}")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Video: {width}x{height} @ {fps:.3f} FPS")
    print(f"Frames: {total_frames}")
    print(f"Person threshold: {threshold:.2f}")

    try:
        while True:
            ok, frame = capture.read()

            if not ok:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if device.startswith("cuda"):
                torch.cuda.synchronize()

            inference_started = time.perf_counter()

            detections = model.predict(
                rgb,
                threshold=threshold,
            )

            if device.startswith("cuda"):
                torch.cuda.synchronize()

            inference_time += time.perf_counter() - inference_started

            class_ids = detections.class_id
            person_mask = class_ids == PERSON_CLASS_ID

            person_boxes = detections.xyxy[person_mask]
            person_confidences = detections.confidence[person_mask]

            annotated_frame, count = draw_person_detections(
                frame,
                person_boxes,
                person_confidences,
            )

            total_person_detections += count

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
        total_person_detections / processed_frames
        if processed_frames
        else 0.0
    )

    print("\n")
    print("Summary")
    print(f"  Model: {model_label}")
    print(f"  Benchmark: {benchmark_name}")
    print(f"  Processed frames: {processed_frames}")
    print(f"  Processing time: {elapsed:.2f} s")
    print(f"  Inference FPS: {inference_fps:.2f}")
    print(f"  End-to-end FPS: {end_to_end_fps:.2f}")
    print(
        "  Average person detections/frame: "
        f"{average_detections:.2f}"
    )
    print(f"  Total person detections: {total_person_detections}")
    print(f"  Output: {output_path}")


def main() -> None:
    args = parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA device requested ({args.device}), "
            "but CUDA is not available."
        )

    model, model_label = load_model(
        model_key=args.model,
        device=args.device,
        fp16=args.fp16,
    )

    if args.all:
        jobs = [
            (name, BENCHMARKS[name])
            for name in ("easy", "normal", "hard")
        ]
    else:
        jobs = [resolve_input(args.input)]

    for benchmark_name, input_path in jobs:
        run_video(
            model=model,
            model_label=model_label,
            benchmark_name=benchmark_name,
            input_path=input_path,
            threshold=args.threshold,
            device=args.device,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()