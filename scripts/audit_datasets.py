#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

VEO_ROOT = Path("datasets/raw/football_images_veo")
SOCCERNET_ROOT = Path("datasets/raw/soccernet_gamestate")
AUDIT_ROOT = Path("datasets/audits")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit the Veo COCO dataset and SoccerNet GameState dataset."
    )
    parser.add_argument("--samples", type=int, default=36)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dataset",
        choices=("all", "veo", "soccernet"),
        default="all",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def bbox_valid(bbox: Any, width: int | None = None, height: int | None = None) -> bool:
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return False
    try:
        x, y, w, h = map(float, bbox)
    except (TypeError, ValueError):
        return False
    if not all(math.isfinite(v) for v in (x, y, w, h)):
        return False
    if w <= 0 or h <= 0 or x < 0 or y < 0:
        return False
    if width and x + w > width + 1:
        return False
    if height and y + h > height + 1:
        return False
    return True


def bbox_area_ratio(bbox, width: int, height: int):
    if width <= 0 or height <= 0:
        return None
    _, _, w, h = map(float, bbox)
    return (w * h) / float(width * height)


def size_bucket(area_ratio):
    if area_ratio is None:
        return "unknown"
    if area_ratio < 0.001:
        return "tiny"
    if area_ratio < 0.005:
        return "small"
    if area_ratio < 0.02:
        return "medium"
    return "large"


def draw_box(image: np.ndarray, bbox, label: str) -> None:
    x, y, w, h = map(float, bbox)
    x1, y1 = int(round(x)), int(round(y))
    x2, y2 = int(round(x + w)), int(round(y + h))
    ih, iw = image.shape[:2]
    x1 = max(0, min(x1, iw - 1))
    y1 = max(0, min(y1, ih - 1))
    x2 = max(0, min(x2, iw - 1))
    y2 = max(0, min(y2, ih - 1))
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(
        image,
        label[:48],
        (x1, max(18, y1 - 5)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (0, 255, 0),
        1,
        cv2.LINE_AA,
    )


def resize_tile(image: np.ndarray, target_w=480, target_h=270):
    h, w = image.shape[:2]
    scale = min(target_w / w, target_h / h)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    x = (target_w - nw) // 2
    y = (target_h - nh) // 2
    canvas[y:y + nh, x:x + nw] = resized
    return canvas


def make_contact_sheet(samples, output_path: Path, title: str) -> None:
    if not samples:
        print(f"[WARN] No samples available for contact sheet: {title}")
        return

    cols = 3
    rows = math.ceil(len(samples) / cols)
    tile_w, tile_h, header_h = 480, 300, 35
    sheet = np.zeros((header_h + rows * tile_h, cols * tile_w, 3), dtype=np.uint8)

    cv2.putText(
        sheet, title, (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
        (255, 255, 255), 2, cv2.LINE_AA
    )

    for idx, sample in enumerate(samples):
        image_path = sample["path"]
        image = cv2.imread(str(image_path))
        if image is None:
            continue

        for ann in sample.get("annotations", []):
            if bbox_valid(ann.get("bbox")):
                draw_box(image, ann["bbox"], ann["label"])

        tile = resize_tile(image)
        footer = np.zeros((30, tile_w, 3), dtype=np.uint8)
        cv2.putText(
            footer,
            sample.get("caption", image_path.name)[:75],
            (6, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        tile = np.vstack([tile, footer])

        row = idx // cols
        col = idx % cols
        y0 = header_h + row * tile_h
        x0 = col * tile_w
        sheet[y0:y0 + tile_h, x0:x0 + tile_w] = tile

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), sheet)
    print(f"Contact sheet: {output_path}")


def audit_coco_split(dataset_root: Path, split: str):
    split_dir = dataset_root / split
    annotation_path = split_dir / "_annotations.coco.json"
    data = load_json(annotation_path)

    categories = {
        safe_int(c.get("id")): str(c.get("name", c.get("id")))
        for c in data.get("categories", [])
    }
    images = {safe_int(img.get("id")): img for img in data.get("images", [])}
    anns_by_image = defaultdict(list)

    class_counts = Counter()
    size_counts = Counter()
    invalid_boxes = 0
    missing_images = 0
    empty_images = 0
    resolutions = Counter()

    for ann in data.get("annotations", []):
        image_id = safe_int(ann.get("image_id"))
        category_id = safe_int(ann.get("category_id"))
        image_meta = images.get(image_id, {})
        width = safe_int(image_meta.get("width"))
        height = safe_int(image_meta.get("height"))
        bbox = ann.get("bbox")
        label = categories.get(category_id, f"class_{category_id}")

        class_counts[label] += 1
        if not bbox_valid(bbox, width or None, height or None):
            invalid_boxes += 1
        elif width and height:
            size_counts[size_bucket(bbox_area_ratio(bbox, width, height))] += 1

        anns_by_image[image_id].append({"bbox": bbox, "label": label})

    samples = []

    for image_id, image_meta in images.items():
        file_name = str(image_meta.get("file_name", ""))
        image_path = split_dir / file_name
        width = safe_int(image_meta.get("width"))
        height = safe_int(image_meta.get("height"))

        if width and height:
            resolutions[f"{width}x{height}"] += 1
        if not image_path.is_file():
            missing_images += 1

        anns = anns_by_image.get(image_id, [])
        if not anns:
            empty_images += 1

        if image_path.is_file() and anns:
            samples.append({
                "path": image_path,
                "annotations": anns,
                "caption": f"{split}/{file_name}",
            })

    return {
        "split": split,
        "images": len(images),
        "annotations": len(data.get("annotations", [])),
        "categories": categories,
        "class_counts": dict(class_counts),
        "size_counts": dict(size_counts),
        "invalid_boxes": invalid_boxes,
        "missing_images": missing_images,
        "empty_images": empty_images,
        "resolutions": dict(resolutions),
        "samples": samples,
    }


def audit_veo(samples_count: int, seed: int):
    print("\n=== VEO / FOOTBALL_IMAGES AUDIT ===")
    split_results = []
    all_samples = []

    for split in ("train", "valid", "test"):
        result = audit_coco_split(VEO_ROOT, split)
        all_samples.extend(result.pop("samples"))
        split_results.append(result)
        print(
            f"{split:>5}: {result['images']} images, "
            f"{result['annotations']} boxes, "
            f"{result['invalid_boxes']} invalid boxes, "
            f"{result['missing_images']} missing images"
        )

    total_classes = Counter()
    total_sizes = Counter()

    for result in split_results:
        total_classes.update(result["class_counts"])
        total_sizes.update(result["size_counts"])

    print(f"Classes: {dict(total_classes)}")
    print(f"Object-size buckets: {dict(total_sizes)}")

    rng = random.Random(seed)
    rng.shuffle(all_samples)
    make_contact_sheet(
        all_samples[:samples_count],
        AUDIT_ROOT / "football_images_veo_contact_sheet.jpg",
        "Football_images Veo - random annotated samples",
    )

    return {
        "dataset": "football_images_veo",
        "splits": split_results,
        "class_counts": dict(total_classes),
        "size_counts": dict(total_sizes),
    }


def locate_soccernet_image(clip_dir: Path, image_meta: dict, file_cache: dict[str, Path]):
    file_name = image_meta.get("file_name")
    if file_name:
        direct = clip_dir / str(file_name)
        if direct.is_file():
            return direct
        basename = Path(str(file_name)).name
        if basename in file_cache:
            return file_cache[basename]

    image_id = image_meta.get("image_id")
    if image_id is not None:
        iid = safe_int(image_id)
        for candidate in (
            f"{image_id}.jpg",
            f"{image_id}.jpeg",
            f"{image_id}.png",
            f"{iid:06d}.jpg",
            f"{iid:06d}.png",
        ):
            if candidate in file_cache:
                return file_cache[candidate]
    return None


def audit_soccernet_clip(label_path: Path, split: str):
    clip_dir = label_path.parent
    data = load_json(label_path)

    info = data.get("info", {})
    version = info.get("version")

    categories = {
        safe_int(c.get("id")): str(c.get("name", c.get("id")))
        for c in data.get("categories", [])
    }

    images_raw = data.get("images", [])
    annotations_raw = data.get("annotations", [])
    images = {
        str(img.get("image_id")): img
        for img in images_raw
        if img.get("image_id") is not None
    }
    anns_by_image = defaultdict(list)

    role_counts = Counter()
    team_counts = Counter()
    category_counts = Counter()
    size_counts = Counter()
    invalid_boxes = 0

    for ann in annotations_raw:
        image_id = str(ann.get("image_id"))
        category_id = safe_int(ann.get("category_id"))
        image_meta = images.get(image_id, {})
        width = safe_int(image_meta.get("width"))
        height = safe_int(image_meta.get("height"))

        attrs = ann.get("attributes") or {}
        role = attrs.get("role")
        team = attrs.get("team")

        category_label = categories.get(category_id, f"class_{category_id}")
        category_counts[category_label] += 1

        if role is not None:
            role_counts[str(role)] += 1
        if team is not None:
            team_counts[str(team)] += 1

        bbox_image = ann.get("bbox_image")

        # SoccerNet stores image boxes as a dictionary:
        # {"x": ..., "y": ..., "w": ..., "h": ...}.
        # Pitch annotations do not represent a normal object bounding box,
        # so they are counted by category but intentionally skipped here.
        bbox = None
        if isinstance(bbox_image, dict):
            required = ("x", "y", "w", "h")
            if all(key in bbox_image for key in required):
                bbox = [
                    bbox_image["x"],
                    bbox_image["y"],
                    bbox_image["w"],
                    bbox_image["h"],
                ]

        if bbox is not None:
            if not bbox_valid(bbox, width or None, height or None):
                invalid_boxes += 1
            elif width and height:
                size_counts[size_bucket(bbox_area_ratio(bbox, width, height))] += 1

            display_label = str(role) if role is not None else category_label
            anns_by_image[image_id].append({
                "bbox": bbox,
                "label": display_label,
            })

    image_files = [
        p for p in clip_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    file_cache = {p.name: p for p in image_files}

    samples = []
    missing_referenced_images = 0

    for image_id, image_meta in images.items():
        anns = anns_by_image.get(str(image_id), [])
        if not anns:
            continue
        image_path = locate_soccernet_image(clip_dir, image_meta, file_cache)
        if image_path is None:
            missing_referenced_images += 1
            continue
        samples.append({
            "path": image_path,
            "annotations": anns,
            "caption": f"{split}/{clip_dir.name}/{image_path.name}",
        })

    return {
        "clip": clip_dir.name,
        "version": version,
        "images": len(images_raw),
        "annotations": len(annotations_raw),
        "image_files": len(image_files),
        "roles": dict(role_counts),
        "teams": dict(team_counts),
        "categories": dict(category_counts),
        "size_counts": dict(size_counts),
        "invalid_boxes": invalid_boxes,
        "missing_referenced_images": missing_referenced_images,
        "top_level_keys": sorted(data.keys()),
        "samples": samples,
    }


def audit_soccernet(samples_count: int, seed: int):
    print("\n=== SOCCERNET GAMESTATE AUDIT ===")

    all_clip_results = []
    all_samples = []
    total_roles = Counter()
    total_teams = Counter()
    total_categories = Counter()
    total_sizes = Counter()
    versions = Counter()
    split_summary = {}

    for split in ("train", "valid", "test"):
        split_dir = SOCCERNET_ROOT / split
        label_paths = sorted(split_dir.glob("SNGS-*/Labels-GameState.json"))

        split_images = 0
        split_annotations = 0
        split_image_files = 0
        split_invalid = 0

        for idx, label_path in enumerate(label_paths, start=1):
            result = audit_soccernet_clip(label_path, split)
            all_samples.extend(result.pop("samples"))
            all_clip_results.append({"split": split, **result})

            split_images += result["images"]
            split_annotations += result["annotations"]
            split_image_files += result["image_files"]
            split_invalid += result["invalid_boxes"]

            total_roles.update(result["roles"])
            total_teams.update(result["teams"])
            total_categories.update(result["categories"])
            total_sizes.update(result["size_counts"])
            versions[str(result["version"])] += 1

            if idx % 10 == 0 or idx == len(label_paths):
                print(f"\r{split}: audited {idx}/{len(label_paths)} clips", end="", flush=True)

        print()
        split_summary[split] = {
            "clips": len(label_paths),
            "metadata_images": split_images,
            "annotations": split_annotations,
            "image_files_found": split_image_files,
            "invalid_boxes": split_invalid,
        }

        print(
            f"{split:>5}: {len(label_paths)} clips, "
            f"{split_images} metadata images, "
            f"{split_annotations} annotations, "
            f"{split_image_files} image files"
        )

    print(f"Dataset versions: {dict(versions)}")
    print(f"Roles: {dict(total_roles)}")
    print(f"Teams: {dict(total_teams)}")
    print(f"Categories: {dict(total_categories)}")
    print(f"Object-size buckets: {dict(total_sizes)}")

    rng = random.Random(seed)
    rng.shuffle(all_samples)
    make_contact_sheet(
        all_samples[:samples_count],
        AUDIT_ROOT / "soccernet_gamestate_contact_sheet.jpg",
        "SoccerNet GameState - random annotated samples",
    )

    return {
        "dataset": "soccernet_gamestate",
        "splits": split_summary,
        "versions": dict(versions),
        "role_counts": dict(total_roles),
        "team_counts": dict(total_teams),
        "category_counts": dict(total_categories),
        "size_counts": dict(total_sizes),
        "clips": all_clip_results,
    }


def main():
    args = parse_args()
    if args.samples <= 0:
        raise ValueError("--samples must be > 0")

    AUDIT_ROOT.mkdir(parents=True, exist_ok=True)
    report = {
        "seed": args.seed,
        "samples_per_contact_sheet": args.samples,
    }

    if args.dataset in ("all", "veo"):
        report["football_images_veo"] = audit_veo(args.samples, args.seed)

    if args.dataset in ("all", "soccernet"):
        report["soccernet_gamestate"] = audit_soccernet(args.samples, args.seed)

    report_path = AUDIT_ROOT / "dataset_audit_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\nAudit report: {report_path}")
    print("Done. No dataset files were modified.")


if __name__ == "__main__":
    main()
