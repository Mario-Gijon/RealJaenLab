#!/usr/bin/env python3

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


COLORS = {
    "ball": (0, 255, 0),
    "linesman": (255, 255, 0),
    "person": (255, 0, 255),
    "referee": (0, 255, 255),
    "team_a": (255, 0, 0),
    "team_b": (0, 0, 255),
}


def draw_box(img, bbox, label, thickness=2):
    x, y, w, h = bbox

    x1 = int(round(x))
    y1 = int(round(y))
    x2 = int(round(x + w))
    y2 = int(round(y + h))

    color = COLORS.get(label, (255, 255, 255))

    cv2.rectangle(
        img,
        (x1, y1),
        (x2, y2),
        color,
        thickness,
    )

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55

    (tw, th), _ = cv2.getTextSize(
        label,
        font,
        scale,
        1,
    )

    ty = max(y1, th + 6)

    cv2.rectangle(
        img,
        (x1, ty - th - 6),
        (x1 + tw + 6, ty + 2),
        color,
        -1,
    )

    cv2.putText(
        img,
        label,
        (x1 + 3, ty - 2),
        font,
        scale,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )


def contact_sheet(images, output, cols=4, cell_width=500):
    if not images:
        return

    thumbs = []

    for img in images:
        h, w = img.shape[:2]

        scale = cell_width / w
        nh = max(1, int(h * scale))

        resized = cv2.resize(
            img,
            (cell_width, nh),
            interpolation=cv2.INTER_AREA,
        )

        thumbs.append(resized)

    cell_height = max(i.shape[0] for i in thumbs)

    rows = (len(thumbs) + cols - 1) // cols

    sheet = np.zeros(
        (
            rows * cell_height,
            cols * cell_width,
            3,
        ),
        dtype=np.uint8,
    )

    sheet[:] = 235

    for i, img in enumerate(thumbs):
        row = i // cols
        col = i % cols

        h, w = img.shape[:2]

        y = row * cell_height
        x = col * cell_width

        sheet[y:y+h, x:x+w] = img

    cv2.imwrite(str(output), sheet)


def ball_zoom(
    img,
    bbox,
    context_factor=15,
    minimum_crop=120,
    output_size=400,
):
    x, y, w, h = bbox

    cx = x + w / 2
    cy = y + h / 2

    crop_size = max(
        minimum_crop,
        int(max(w, h) * context_factor),
    )

    x1 = int(cx - crop_size / 2)
    y1 = int(cy - crop_size / 2)
    x2 = x1 + crop_size
    y2 = y1 + crop_size

    ih, iw = img.shape[:2]

    x1c = max(0, x1)
    y1c = max(0, y1)
    x2c = min(iw, x2)
    y2c = min(ih, y2)

    crop = img[y1c:y2c, x1c:x2c].copy()

    if crop.size == 0:
        return None

    # Coordenadas bbox dentro del crop.
    bx1 = x - x1c
    by1 = y - y1c
    bx2 = x + w - x1c
    by2 = y + h - y1c

    ch, cw = crop.shape[:2]

    sx = output_size / cw
    sy = output_size / ch

    zoom = cv2.resize(
        crop,
        (output_size, output_size),
        interpolation=cv2.INTER_CUBIC,
    )

    p1 = (
        int(bx1 * sx),
        int(by1 * sy),
    )

    p2 = (
        int(bx2 * sx),
        int(by2 * sy),
    )

    cv2.rectangle(
        zoom,
        p1,
        p2,
        COLORS["ball"],
        3,
    )

    return zoom


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "dataset",
        type=Path,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "outputs/dataset_inspection/"
            "veo_v14_semantic"
        ),
    )

    parser.add_argument(
        "--ball-samples",
        type=int,
        default=60,
    )

    parser.add_argument(
        "--staff-audit-frames",
        type=int,
        default=48,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    random.seed(args.seed)

    args.output.mkdir(
        parents=True,
        exist_ok=True,
    )

    ball_records = []
    person_records = []
    frame_records = []

    class_counts = defaultdict(int)
    split_counts = defaultdict(int)

    for split in ["train", "valid", "test"]:

        split_dir = args.dataset / split

        json_path = (
            split_dir /
            "_annotations.coco.json"
        )

        if not json_path.exists():
            continue

        data = json.loads(
            json_path.read_text()
        )

        categories = {
            c["id"]: c["name"]
            for c in data["categories"]
        }

        images = {
            i["id"]: i
            for i in data["images"]
        }

        anns_by_image = defaultdict(list)

        for ann in data["annotations"]:

            name = categories[
                ann["category_id"]
            ]

            class_counts[name] += 1
            split_counts[split] += 1

            image_info = images[
                ann["image_id"]
            ]

            anns_by_image[
                ann["image_id"]
            ].append(ann)

            record = (
                split,
                image_info,
                ann,
                categories,
            )

            if name == "ball":
                ball_records.append(record)

            if name == "person":
                person_records.append(record)

        for image_id, image_info in images.items():

            frame_records.append(
                (
                    split,
                    image_info,
                    anns_by_image.get(
                        image_id,
                        [],
                    ),
                    categories,
                )
            )

    #
    # RESUMEN
    #

    print("=== VEO V14 SEMANTIC AUDIT ===")
    print()

    print("Classes:")

    for name, count in sorted(
        class_counts.items()
    ):
        print(
            f"  {name:12s} {count:6d}"
        )

    print()
    print(
        f"Total ball annotations: "
        f"{len(ball_records)}"
    )

    print(
        f"Total person annotations: "
        f"{len(person_records)}"
    )

    #
    # BALL ZOOMS
    #

    selected_balls = random.sample(
        ball_records,
        min(
            args.ball_samples,
            len(ball_records),
        ),
    )

    ball_images = []

    ball_dir = args.output / "ball_zooms"
    ball_dir.mkdir(exist_ok=True)

    for idx, (
        split,
        image_info,
        ann,
        categories,
    ) in enumerate(selected_balls):

        path = (
            args.dataset /
            split /
            image_info["file_name"]
        )

        img = cv2.imread(str(path))

        if img is None:
            continue

        zoom = ball_zoom(
            img,
            ann["bbox"],
        )

        if zoom is None:
            continue

        x, y, w, h = ann["bbox"]

        cv2.putText(
            zoom,
            f"{split} | bbox {w:.1f}x{h:.1f}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imwrite(
            str(
                ball_dir /
                f"{idx:03d}.jpg"
            ),
            zoom,
        )

        ball_images.append(zoom)

    contact_sheet(
        ball_images,
        args.output /
        "contact_ball_zooms.jpg",
        cols=6,
        cell_width=300,
    )

    #
    # STAFF / OTHER PERSON AUDIT
    #
    # Mostramos frames completos con todas las
    # anotaciones. Cualquier entrenador, utillero,
    # staff, etc. SIN caja será fácil de observar.
    #

    selected_frames = random.sample(
        frame_records,
        min(
            args.staff_audit_frames,
            len(frame_records),
        ),
    )

    staff_images = []

    staff_dir = (
        args.output /
        "staff_audit_frames"
    )

    staff_dir.mkdir(exist_ok=True)

    for idx, (
        split,
        image_info,
        anns,
        categories,
    ) in enumerate(selected_frames):

        path = (
            args.dataset /
            split /
            image_info["file_name"]
        )

        img = cv2.imread(str(path))

        if img is None:
            continue

        counts = defaultdict(int)

        for ann in anns:

            name = categories[
                ann["category_id"]
            ]

            counts[name] += 1

            draw_box(
                img,
                ann["bbox"],
                name,
                thickness=2,
            )

        summary = (
            f"{split} | "
            f"A:{counts['team_a']} "
            f"B:{counts['team_b']} "
            f"R:{counts['referee']} "
            f"L:{counts['linesman']} "
            f"P:{counts['person']} "
            f"Ball:{counts['ball']}"
        )

        cv2.rectangle(
            img,
            (0, 0),
            (900, 40),
            (0, 0, 0),
            -1,
        )

        cv2.putText(
            img,
            summary,
            (10, 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imwrite(
            str(
                staff_dir /
                f"{idx:03d}_{split}.jpg"
            ),
            img,
        )

        staff_images.append(img)

    contact_sheet(
        staff_images,
        args.output /
        "contact_staff_audit.jpg",
        cols=3,
        cell_width=640,
    )

    #
    # PERSON EXACTO
    #

    person_dir = (
        args.output /
        "person_samples"
    )

    person_dir.mkdir(exist_ok=True)

    person_images = []

    for idx, (
        split,
        image_info,
        ann,
        categories,
    ) in enumerate(person_records):

        path = (
            args.dataset /
            split /
            image_info["file_name"]
        )

        img = cv2.imread(str(path))

        if img is None:
            continue

        draw_box(
            img,
            ann["bbox"],
            "person",
            thickness=5,
        )

        cv2.putText(
            img,
            f"{split} | {image_info['file_name']}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imwrite(
            str(
                person_dir /
                f"{idx:03d}.jpg"
            ),
            img,
        )

        person_images.append(img)

    contact_sheet(
        person_images,
        args.output /
        "contact_person_exact.jpg",
        cols=1,
        cell_width=1000,
    )

    #
    # TXT
    #

    report_path = (
        args.output /
        "report.txt"
    )

    with report_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "Veo Football Detection v14 "
            "semantic audit\n\n"
        )

        f.write("Classes\n")

        for name, count in sorted(
            class_counts.items()
        ):
            f.write(
                f"{name}: {count}\n"
            )

        f.write(
            f"\nBall annotations: "
            f"{len(ball_records)}\n"
        )

        f.write(
            f"Person annotations: "
            f"{len(person_records)}\n"
        )

    print()
    print("Generated:")
    print(
        args.output.resolve()
    )

    print()
    print(
        "  contact_ball_zooms.jpg"
    )

    print(
        "  contact_staff_audit.jpg"
    )

    print(
        "  contact_person_exact.jpg"
    )

    print(
        "  report.txt"
    )


if __name__ == "__main__":
    main()
