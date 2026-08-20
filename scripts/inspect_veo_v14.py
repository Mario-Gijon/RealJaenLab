#!/usr/bin/env python3

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import cv2


COLORS = {
    "ball": (0, 255, 0),
    "team_a": (255, 0, 0),
    "team_b": (0, 0, 255),
    "referee": (0, 255, 255),
    "linesman": (255, 255, 0),
    "person": (255, 0, 255),
}


def draw_box(img, bbox, label, color, thickness=2):
    x, y, w, h = bbox

    x1 = int(round(x))
    y1 = int(round(y))
    x2 = int(round(x + w))
    y2 = int(round(y + h))

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

    text_y = max(th + 4, y1)

    cv2.rectangle(
        img,
        (x1, text_y - th - 5),
        (x1 + tw + 5, text_y + 2),
        color,
        -1,
    )

    cv2.putText(
        img,
        label,
        (x1 + 2, text_y - 2),
        font,
        scale,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )


def save_contact_sheet(images, output, cols=3, thumb_width=640):
    if not images:
        return

    thumbs = []

    for img in images:
        h, w = img.shape[:2]

        scale = thumb_width / w

        thumb = cv2.resize(
            img,
            (
                thumb_width,
                int(h * scale),
            ),
        )

        thumbs.append(thumb)

    thumb_h = max(x.shape[0] for x in thumbs)

    rows = (len(thumbs) + cols - 1) // cols

    sheet = 255 * (
        __import__("numpy")
        .ones(
            (
                rows * thumb_h,
                cols * thumb_width,
                3,
            ),
            dtype="uint8",
        )
    )

    for i, img in enumerate(thumbs):
        r = i // cols
        c = i % cols

        h, w = img.shape[:2]

        sheet[
            r * thumb_h:r * thumb_h + h,
            c * thumb_width:c * thumb_width + w,
        ] = img

    cv2.imwrite(str(output), sheet)


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
            "outputs/dataset_inspection/veo_v14"
        ),
    )

    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=12,
    )

    parser.add_argument(
        "--full-frame-samples",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    random.seed(args.seed)

    output = args.output
    output.mkdir(parents=True, exist_ok=True)

    class_samples_dir = output / "classes"
    problems_dir = output / "problems"
    full_frames_dir = output / "full_frames"

    class_samples_dir.mkdir(exist_ok=True)
    problems_dir.mkdir(exist_ok=True)
    full_frames_dir.mkdir(exist_ok=True)

    class_records = defaultdict(list)
    all_records = []
    problems = []

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
            anns_by_image[
                ann["image_id"]
            ].append(ann)

            name = categories[
                ann["category_id"]
            ]

            class_records[name].append(
                (
                    split,
                    images[ann["image_id"]],
                    ann,
                    categories,
                )
            )

        for image_id, image_info in images.items():
            all_records.append(
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

        # Detectar bboxes fuera de imagen.
        for ann in data["annotations"]:
            image_info = images[
                ann["image_id"]
            ]

            width = image_info["width"]
            height = image_info["height"]

            x, y, w, h = ann["bbox"]

            if (
                x < 0
                or y < 0
                or x + w > width
                or y + h > height
            ):
                problems.append(
                    (
                        split,
                        image_info,
                        ann,
                        categories,
                    )
                )

    print("=== CLASSES ===")

    for cls, records in sorted(
        class_records.items()
    ):
        print(
            f"{cls:15s}: "
            f"{len(records)} annotations"
        )

    print()
    print(
        f"Out-of-bounds boxes: "
        f"{len(problems)}"
    )

    #
    # 1. Muestras por clase
    #
    for cls, records in sorted(
        class_records.items()
    ):
        cls_dir = class_samples_dir / cls
        cls_dir.mkdir(exist_ok=True)

        selected = random.sample(
            records,
            min(
                args.samples_per_class,
                len(records),
            ),
        )

        contact_images = []

        for index, (
            split,
            image_info,
            target_ann,
            categories,
        ) in enumerate(selected):

            image_path = (
                args.dataset /
                split /
                image_info["file_name"]
            )

            img = cv2.imread(
                str(image_path)
            )

            if img is None:
                print(
                    f"WARNING cannot read "
                    f"{image_path}"
                )
                continue

            name = categories[
                target_ann["category_id"]
            ]

            color = COLORS.get(
                name,
                (255, 255, 255),
            )

            draw_box(
                img,
                target_ann["bbox"],
                name,
                color,
                thickness=4,
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

            save_path = (
                cls_dir /
                f"{index:02d}_{split}.jpg"
            )

            cv2.imwrite(
                str(save_path),
                img,
            )

            contact_images.append(img)

        save_contact_sheet(
            contact_images,
            output /
            f"contact_{cls}.jpg",
        )

    #
    # 2. Frames completos con TODAS
    #    las anotaciones.
    #
    selected_frames = random.sample(
        all_records,
        min(
            args.full_frame_samples,
            len(all_records),
        ),
    )

    full_contact = []

    for index, (
        split,
        image_info,
        annotations,
        categories,
    ) in enumerate(selected_frames):

        image_path = (
            args.dataset /
            split /
            image_info["file_name"]
        )

        img = cv2.imread(
            str(image_path)
        )

        if img is None:
            continue

        counts = defaultdict(int)

        for ann in annotations:
            name = categories[
                ann["category_id"]
            ]

            counts[name] += 1

            draw_box(
                img,
                ann["bbox"],
                name,
                COLORS.get(
                    name,
                    (255, 255, 255),
                ),
            )

        summary = " | ".join(
            f"{k}:{v}"
            for k, v in sorted(
                counts.items()
            )
        )

        cv2.putText(
            img,
            f"{split} | {summary}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        save_path = (
            full_frames_dir /
            f"{index:02d}_{split}.jpg"
        )

        cv2.imwrite(
            str(save_path),
            img,
        )

        full_contact.append(img)

    save_contact_sheet(
        full_contact,
        output / "contact_full_frames.jpg",
    )

    #
    # 3. Bounding boxes problemáticos.
    #
    problem_contact = []

    for index, (
        split,
        image_info,
        ann,
        categories,
    ) in enumerate(problems):

        image_path = (
            args.dataset /
            split /
            image_info["file_name"]
        )

        img = cv2.imread(
            str(image_path)
        )

        if img is None:
            continue

        name = categories[
            ann["category_id"]
        ]

        draw_box(
            img,
            ann["bbox"],
            f"OUT_OF_BOUNDS {name}",
            (0, 0, 255),
            thickness=5,
        )

        save_path = (
            problems_dir /
            f"{index:03d}_{split}_{name}.jpg"
        )

        cv2.imwrite(
            str(save_path),
            img,
        )

        problem_contact.append(img)

    save_contact_sheet(
        problem_contact,
        output /
        "contact_out_of_bounds.jpg",
    )

    print()
    print("=== OUTPUT ===")
    print(output.resolve())

    print()
    print("Main files:")
    print(
        "  contact_full_frames.jpg"
    )

    for cls in sorted(
        class_records
    ):
        print(
            f"  contact_{cls}.jpg"
        )

    if problems:
        print(
            "  contact_out_of_bounds.jpg"
        )


if __name__ == "__main__":
    main()
