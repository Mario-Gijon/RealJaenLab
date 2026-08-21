#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import torch
from transformers import AutoImageProcessor, DFineForObjectDetection

MODEL_ID = "ustc-community/dfine-xlarge-obj2coco"
INPUT = Path("input/benchmark_normal_30s.mp4")
OUTPUT_ROOT = Path("outputs/resolution_tests/dfine_xlarge")
CACHE_DIR = Path("weights/huggingface")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--threshold", type=float, default=0.30)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def person_id(model):
    for k, v in model.config.id2label.items():
        if str(v).lower() == "person":
            return int(k)
    raise RuntimeError("person class not found")


def draw(frame, boxes, scores):
    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = map(int, box.tolist())
        cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 220, 100), 2)
        cv2.putText(
            frame,
            f"person {float(score):.2f}",
            (x1, max(18, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (80, 220, 100),
            2,
            cv2.LINE_AA,
        )


def main():
    args = parse_args()

    if not INPUT.is_file():
        raise FileNotFoundError(INPUT)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    output = OUTPUT_ROOT / f"dfine_xlarge_normal_imgsz{args.imgsz}.mp4"
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"{output} exists; use --overwrite")

    print(f"Loading {MODEL_ID}")
    processor = AutoImageProcessor.from_pretrained(MODEL_ID, cache_dir=str(CACHE_DIR))
    model = DFineForObjectDetection.from_pretrained(MODEL_ID, cache_dir=str(CACHE_DIR))
    model.to(args.device).eval()

    pid = person_id(model)

    print(f"Original processor size: {getattr(processor, 'size', None)}")
    print(f"Requested test size: {args.imgsz}x{args.imgsz}")
    print(f"Model eval_size: {getattr(model.config, 'eval_size', None)}")
    print(f"Model anchor_image_size: {getattr(model.config, 'anchor_image_size', None)}")

    cap = cv2.VideoCapture(str(INPUT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    frames = 0
    detections_total = 0
    t0 = time.perf_counter()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        inputs = processor(
            images=rgb,
            return_tensors="pt",
            size={"height": args.imgsz, "width": args.imgsz},
        )
        inputs = {k: v.to(args.device) for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = model(**inputs)

        result = processor.post_process_object_detection(
            outputs,
            threshold=args.threshold,
            target_sizes=torch.tensor([[height, width]], device=args.device),
        )[0]

        mask = result["labels"] == pid
        boxes = result["boxes"][mask].detach().cpu()
        scores = result["scores"][mask].detach().cpu()

        detections_total += len(boxes)
        draw(frame, boxes, scores)
        writer.write(frame)

        frames += 1
        if frames % 150 == 0 or frames == total:
            print(f"\r{frames}/{total}", end="", flush=True)

    cap.release()
    writer.release()
    elapsed = time.perf_counter() - t0

    print()
    print(f"Output: {output}")
    print(f"Frames: {frames}")
    print(f"Time: {elapsed:.2f}s")
    print(f"Average persons/frame: {detections_total / frames:.2f}" if frames else "No frames")


if __name__ == "__main__":
    main()
