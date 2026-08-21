#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import cv2
import numpy as np


DATASET_ROOT = Path("datasets/processed/match_person_v1")
OUTPUT_ROOT = Path("datasets/audits/match_person_v1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate annotated contact sheets for match_person_v1."
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=24,
        help="Number of random images per split (default: 24).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    return parser.parse_args()


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def draw_box(image, bbox, label="match_person"):
    x, y, w, h = map(float, bbox)
    x1 = int(round(x))
    y1 = int(round(y))
    x2 = int(round(x + w))
    y2 = int(round(y + h))

    ih, iw = image.shape[:2]

    x1 = max(0, min(x1, iw - 1))
    y1 = max(0, min(y1, ih - 1))
    x2 = max(0, min(x2, iw - 1))
    y2 = max(0, min(y2, ih - 1))

    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2,
    )

    cv2.putText(
        image,
        label,
        (x1, max(18, y1 - 5)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 255, 0),
        1,
        cv2.LINE_AA,
    )


def resize_tile(image, target_w=480, target_h=270):
    h, w = image.shape[:2]

    scale = min(target_w / w, target_h / h)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))

    resized = cv2.resize(
        image,
        (nw, nh),
        interpolation=cv2.INTER_AREA,
    )

    canvas = np.zeros(
        (target_h, target_w, 3),
        dtype=np.uint8,
    )

    x = (target_w - nw) // 2
    y = (target_h - nh) // 2

    canvas[y:y + nh, x:x + nw] = resized

    return canvas


def build_sheet(split: str, samples_count: int, seed: int):
    split_dir = DATASET_ROOT / split
    annotation_path = split_dir / "_annotations.coco.json"

    if not annotation_path.is_file():
        raise FileNotFoundError(annotation_path)

    data = load_json(annotation_path)

    images = {
        int(img["id"]): img
        for img in data.get("images", [])
    }

    annotations_by_image = {}

    for ann in data.get("annotations", []):
        annotations_by_image.setdefault(
            int(ann["image_id"]),
            [],
        ).append(ann)

    image_ids = list(images.keys())

    rng = random.Random(seed + hash(split) % 10000)
    rng.shuffle(image_ids)

    selected_ids = image_ids[: min(samples_count, len(image_ids))]

    cols = 3
    rows = math.ceil(len(selected_ids) / cols)

    tile_w = 480
    image_h = 270
    footer_h = 38
    tile_h = image_h + footer_h
    header_h = 42

    sheet = np.zeros(
        (header_h + rows * tile_h, cols * tile_w, 3),
        dtype=np.uint8,
    )

    cv2.putText(
        sheet,
        f"match_person_v1 - {split}",
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    rendered = 0
    missing = 0

    for index, image_id in enumerate(selected_ids):
        meta = images[image_id]
        image_path = split_dir / meta["file_name"]

        image = cv2.imread(str(image_path))

        if image is None:
            missing += 1
            continue

        anns = annotations_by_image.get(image_id, [])

        for ann in anns:
            draw_box(
                image,
                ann["bbox"],
            )

        tile_image = resize_tile(image)

        footer = np.zeros(
            (footer_h, tile_w, 3),
            dtype=np.uint8,
        )

        caption = (
            f'{meta["file_name"]} | '
            f'{len(anns)} match_person'
        )

        cv2.putText(
            footer,
            caption[:85],
            (6, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        tile = np.vstack([tile_image, footer])

        row = index // cols
        col = index % cols

        y0 = header_h + row * tile_h
        x0 = col * tile_w

        sheet[
            y0:y0 + tile_h,
            x0:x0 + tile_w,
        ] = tile

        rendered += 1

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = OUTPUT_ROOT / f"{split}_contact_sheet.jpg"

    cv2.imwrite(
        str(output_path),
        sheet,
    )

    print(
        f"{split}: rendered={rendered}, "
        f"missing={missing}, "
        f"output={output_path}"
    )


def main():
    args = parse_args()

    if args.samples <= 0:
        raise ValueError("--samples must be greater than 0.")

    if not DATASET_ROOT.is_dir():
        raise FileNotFoundError(DATASET_ROOT)

    for split in ("train", "valid", "test"):
        build_sheet(
            split,
            samples_count=args.samples,
            seed=args.seed,
        )

    print("Done. Dataset files were not modified.")


if __name__ == "__main__":
    main()
