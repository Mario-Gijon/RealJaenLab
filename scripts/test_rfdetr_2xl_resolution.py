#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import torch
from rfdetr.assets.coco_classes import COCO_CLASSES
from rfdetr_plus import RFDETR2XLarge

INPUT = Path("input/benchmark_normal_30s.mp4")
OUTPUT_ROOT = Path("outputs/resolution_tests/rfdetr_2xlarge")
PERSON_CLASS_ID = 1


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--resolution", type=int, default=1280)
    p.add_argument("--threshold", type=float, default=0.30)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def draw(frame, boxes, scores):
    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = map(int, box)
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

    if args.resolution <= 0 or args.resolution % 32 != 0:
        raise ValueError("For this test, --resolution must be a positive multiple of 32.")

    if not INPUT.is_file():
        raise FileNotFoundError(INPUT)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    output = OUTPUT_ROOT / f"rfdetr_2xlarge_normal_res{args.resolution}.mp4"

    if output.exists() and not args.overwrite:
        raise FileExistsError(f"{output} exists; use --overwrite")

    print(f"Loading RF-DETR 2XL at requested resolution {args.resolution}x{args.resolution}")
    model = RFDETR2XLarge(
        accept_platform_model_license=True,
        device=args.device,
        resolution=args.resolution,
    )

    print(f"Person class: {COCO_CLASSES[PERSON_CLASS_ID]}")

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

        detections = model.predict(rgb, threshold=args.threshold)
        mask = detections.class_id == PERSON_CLASS_ID
        boxes = detections.xyxy[mask]
        scores = detections.confidence[mask]

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
