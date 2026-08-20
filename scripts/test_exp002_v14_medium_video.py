#!/usr/bin/env python3

import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from rfdetr import RFDETRMedium


DEFAULT_VIDEO = Path("input/match.mp4")
DEFAULT_WEIGHTS = Path("weights/models/football_detector_exp002_v14_medium.pth")

CLASS_NAMES = {
    0: "ball",
    1: "linesman",
    2: "referee",
    3: "team_a",
    4: "team_b",
}

# OpenCV usa BGR.
COLORS = {
    "ball": (0, 255, 0),          # verde
    "linesman": (255, 255, 0),    # cian
    "referee": (0, 255, 255),     # amarillo
    "team_a": (255, 0, 0),        # azul
    "team_b": (0, 0, 255),        # rojo
}


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--video",
        type=Path,
        default=DEFAULT_VIDEO,
    )

    parser.add_argument(
        "--weights",
        type=Path,
        default=DEFAULT_WEIGHTS,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/exp002_v14_medium_jaen.mp4"),
    )

    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("outputs/exp002_v14_medium_jaen_detections.csv"),
    )

    parser.add_argument(
        "--infer-fps",
        type=float,
        default=10.0,
        help="Frames por segundo sobre los que ejecutar RF-DETR.",
    )

    parser.add_argument(
        "--predict-threshold",
        type=float,
        default=0.15,
        help="Threshold mínimo solicitado a RF-DETR.",
    )

    parser.add_argument(
        "--display-threshold",
        type=float,
        default=0.30,
        help="Threshold mínimo para dibujar una detección.",
    )

    parser.add_argument(
        "--start",
        type=float,
        default=0.0,
        help="Segundo de inicio.",
    )

    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Duración a procesar. Omitir para llegar hasta el final.",
    )

    return parser.parse_args()


def draw_detection(
    frame,
    xyxy,
    class_name,
    confidence,
):
    x1, y1, x2, y2 = map(int, xyxy)

    h, w = frame.shape[:2]

    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w - 1))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h - 1))

    color = COLORS.get(class_name, (255, 255, 255))

    # Para el balón hacemos una marca extra para verlo bien.
    if class_name == "ball":
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        cv2.circle(
            frame,
            (cx, cy),
            10,
            color,
            2,
        )

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        color,
        2,
    )

    label = f"{class_name} {confidence:.2f}"

    (tw, th), _ = cv2.getTextSize(
        label,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        1,
    )

    text_y = max(th + 4, y1)

    cv2.rectangle(
        frame,
        (x1, text_y - th - 5),
        (x1 + tw + 5, text_y + 2),
        color,
        -1,
    )

    cv2.putText(
        frame,
        label,
        (x1 + 2, text_y - 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )


def main():
    args = parse_args()

    if not args.video.exists():
        raise FileNotFoundError(args.video)

    if not args.weights.exists():
        raise FileNotFoundError(args.weights)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA no está disponible.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.csv.parent.mkdir(parents=True, exist_ok=True)

    print("=== EXP_002 V14 Medium Real Jaén domain test ===")
    print("Video:  ", args.video.resolve())
    print("Weights:", args.weights.resolve())
    print("GPU:    ", torch.cuda.get_device_name(0))
    print()

    print("Loading RF-DETR Medium...")

    model = RFDETRMedium(
        num_classes=5,
        pretrain_weights=str(args.weights),
    )

    print("Model loaded.")
    print()

    cap = cv2.VideoCapture(str(args.video))

    if not cap.isOpened():
        raise RuntimeError(f"No puedo abrir {args.video}")

    source_fps = cap.get(cv2.CAP_PROP_FPS)
    source_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    source_duration = source_frames / source_fps

    start = max(0.0, args.start)

    if args.duration is None:
        end = source_duration
    else:
        end = min(
            source_duration,
            start + args.duration,
        )

    start_frame = int(round(start * source_fps))
    end_frame = int(round(end * source_fps))

    stride = max(
        1,
        int(round(source_fps / args.infer_fps)),
    )

    actual_output_fps = source_fps / stride

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        str(args.output),
        fourcc,
        actual_output_fps,
        (width, height),
    )

    if not writer.isOpened():
        raise RuntimeError(
            f"No puedo crear {args.output}"
        )

    frames_to_process = (
        (end_frame - start_frame + stride - 1)
        // stride
    )

    print(f"Source FPS:     {source_fps:.3f}")
    print(f"Inference FPS:  {actual_output_fps:.3f}")
    print(f"Resolution:     {width}x{height}")
    print(f"Range:          {start:.2f}s -> {end:.2f}s")
    print(f"Frames infer:   {frames_to_process}")
    print(f"Predict thresh: {args.predict_threshold}")
    print(f"Display thresh: {args.display_threshold}")
    print()

    csv_file = args.csv.open(
        "w",
        newline="",
        encoding="utf-8",
    )

    csv_writer = csv.writer(csv_file)

    csv_writer.writerow([
        "frame",
        "time_seconds",
        "class_id",
        "class_name",
        "confidence",
        "x1",
        "y1",
        "x2",
        "y2",
    ])

    processed = 0
    detection_count = {
        "ball": 0,
        "linesman": 0,
        "referee": 0,
        "team_a": 0,
        "team_b": 0,
    }

    t0 = time.time()

    progress = tqdm(
        total=frames_to_process,
        desc="Inference",
    )

    frame_idx = start_frame

    try:
        while frame_idx < end_frame:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

            ok, frame = cap.read()

            if not ok:
                break

            time_seconds = frame_idx / source_fps

            # OpenCV BGR -> PIL RGB
            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB,
            )

            image = Image.fromarray(rgb)

            detections = model.predict(
                image,
                threshold=args.predict_threshold,
                include_source_image=False,
            )

            xyxy = np.asarray(detections.xyxy)

            if detections.confidence is None:
                confidences = np.ones(len(xyxy))
            else:
                confidences = np.asarray(
                    detections.confidence
                )

            if detections.class_id is None:
                class_ids = np.full(
                    len(xyxy),
                    -1,
                    dtype=int,
                )
            else:
                class_ids = np.asarray(
                    detections.class_id
                ).astype(int)

            class_names_from_model = (
                detections.data.get("class_name")
                if detections.data is not None
                else None
            )

            for i in range(len(xyxy)):
                class_id = int(class_ids[i])
                confidence = float(confidences[i])

                if class_names_from_model is not None:
                    class_name = str(
                        class_names_from_model[i]
                    )
                else:
                    class_name = CLASS_NAMES.get(
                        class_id,
                        f"class_{class_id}",
                    )

                x1, y1, x2, y2 = map(
                    float,
                    xyxy[i],
                )

                # Guardamos TODO lo que supere predict-threshold.
                csv_writer.writerow([
                    frame_idx,
                    f"{time_seconds:.3f}",
                    class_id,
                    class_name,
                    f"{confidence:.6f}",
                    f"{x1:.2f}",
                    f"{y1:.2f}",
                    f"{x2:.2f}",
                    f"{y2:.2f}",
                ])

                if confidence < args.display_threshold:
                    continue

                if class_name in detection_count:
                    detection_count[class_name] += 1

                draw_detection(
                    frame,
                    xyxy[i],
                    class_name,
                    confidence,
                )

            # Información del frame.
            cv2.putText(
                frame,
                f"EXP_002 V14 Medium | t={time_seconds:.2f}s | frame={frame_idx}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            writer.write(frame)

            processed += 1
            progress.update(1)

            frame_idx += stride

    finally:
        progress.close()
        csv_file.close()
        writer.release()
        cap.release()

    elapsed = time.time() - t0

    print()
    print("=== DONE ===")
    print(f"Processed frames: {processed}")
    print(f"Elapsed:          {elapsed:.1f}s")

    if elapsed > 0:
        print(
            f"Inference speed:  "
            f"{processed / elapsed:.2f} frames/s"
        )

    print()
    print("Displayed detections:")
    for name, count in detection_count.items():
        print(f"  {name:12s}: {count}")

    print()
    print("Video:", args.output.resolve())
    print("CSV:  ", args.csv.resolve())


if __name__ == "__main__":
    main()
