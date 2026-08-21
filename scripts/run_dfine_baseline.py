#!/usr/bin/env python3

from __future__ import annotations

import argparse
import time
from collections import Counter
from pathlib import Path

import cv2
import torch
from transformers import AutoImageProcessor, DFineForObjectDetection


MODEL_ID = "ustc-community/dfine-xlarge-obj2coco"
MODEL_NAME = "dfine_xlarge_obj2coco"

BENCHMARKS = {
    "easy": Path("input/benchmark_easy_30s.mp4"),
    "normal": Path("input/benchmark_normal_30s.mp4"),
    "hard": Path("input/benchmark_hard_30s.mp4"),
}

OUTPUT_ROOT = Path("outputs/dfine_baseline") / MODEL_NAME
CACHE_DIR = Path("weights/huggingface")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run pretrained D-FINE-X Objects365->COCO on the "
            "RealJaenLab 30-second player-detection benchmarks."
        )
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
        help="Detection confidence threshold (default: 0.30).",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device, e.g. cuda, cuda:0 or cpu.",
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


def find_person_class_id(model: DFineForObjectDetection) -> int:
    id2label = model.config.id2label

    for raw_id, raw_label in id2label.items():
        if str(raw_label).strip().lower() == "person":
            return int(raw_id)

    available = ", ".join(
        f"{class_id}:{label}" for class_id, label in id2label.items()
    )
    raise RuntimeError(
        "Could not find the COCO 'person' class in the model labels. "
        f"Available labels: {available}"
    )


def load_model(device: str):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading model: {MODEL_ID}")
    print(f"Device: {device}")
    print(f"Cache: {CACHE_DIR}")

    processor = AutoImageProcessor.from_pretrained(
        MODEL_ID,
        cache_dir=str(CACHE_DIR),
    )

    model = DFineForObjectDetection.from_pretrained(
        MODEL_ID,
        cache_dir=str(CACHE_DIR),
    )

    model.to(device)
    model.eval()

    person_class_id = find_person_class_id(model)

    print(f"Person class id: {person_class_id}")
    print(
        "Precision: FP32 "
        "(intentional for the first quality baseline)"
    )

    return processor, model, person_class_id


def draw_person_detections(
    frame,
    boxes,
    scores,
) -> tuple[object, int]:
    height, width = frame.shape[:2]

    thickness = max(1, round(min(width, height) / 700))
    font_scale = max(0.4, min(width, height) / 1800)

    # BGR. Fixed color because this baseline only renders one class.
    color = (80, 220, 100)

    detection_count = 0

    for box, score in zip(boxes, scores):
        detection_count += 1

        x1, y1, x2, y2 = map(int, box.tolist())

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

        label = f"person {float(score):.2f}"

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

    return frame, detection_count


def run_video(
    processor,
    model,
    person_class_id: int,
    benchmark_name: str,
    input_path: Path,
    threshold: float,
    device: str,
    overwrite: bool,
) -> None:
    if not input_path.is_file():
        raise FileNotFoundError(f"Input video not found: {input_path}")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    output_path = OUTPUT_ROOT / f"{MODEL_NAME}_{benchmark_name}.mp4"

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

    print(f"\nRunning {MODEL_NAME} / {benchmark_name}")
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

            # OpenCV -> RGB. We deliberately do no enhancement here.
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            inputs = processor(
                images=rgb,
                return_tensors="pt",
            )
            inputs = {
                key: value.to(device)
                for key, value in inputs.items()
            }

            if device.startswith("cuda"):
                torch.cuda.synchronize()

            inference_started = time.perf_counter()

            with torch.inference_mode():
                outputs = model(**inputs)

            if device.startswith("cuda"):
                torch.cuda.synchronize()

            inference_time += time.perf_counter() - inference_started

            # Map the internally-resized predictions back to the
            # original 1920x1080 video coordinates.
            target_sizes = torch.tensor(
                [[height, width]],
                device=device,
            )

            results = processor.post_process_object_detection(
                outputs,
                target_sizes=target_sizes,
                threshold=threshold,
            )[0]

            labels = results["labels"]
            person_mask = labels == person_class_id

            person_boxes = results["boxes"][person_mask].detach().cpu()
            person_scores = results["scores"][person_mask].detach().cpu()

            annotated_frame, count = draw_person_detections(
                frame,
                person_boxes,
                person_scores,
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
    print(f"  Model: {MODEL_NAME}")
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

    processor, model, person_class_id = load_model(args.device)

    if args.all:
        jobs = [
            (name, BENCHMARKS[name])
            for name in ("easy", "normal", "hard")
        ]
    else:
        jobs = [resolve_input(args.input)]

    for benchmark_name, input_path in jobs:
        run_video(
            processor=processor,
            model=model,
            person_class_id=person_class_id,
            benchmark_name=benchmark_name,
            input_path=input_path,
            threshold=args.threshold,
            device=args.device,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()