#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

SPLITS = ("train", "valid", "test")
EXPECTED_CLASSES = {"ball", "player", "referee", "goalkeeper"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "dataset",
        type=Path,
        nargs="?",
        default=Path("datasets/football_players_v20"),
    )
    return p.parse_args()


def source_key(filename: str) -> str:
    name = Path(filename).name
    if ".rf." in name:
        return name.split(".rf.", 1)[0]
    return Path(name).stem


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def median(values):
    if not values:
        return None
    v = sorted(values)
    n = len(v)
    m = n // 2
    return v[m] if n % 2 else (v[m - 1] + v[m]) / 2


def audit_split(root: Path, split: str):
    split_dir = root / split
    ann_path = split_dir / "_annotations.coco.json"
    errors, warnings = [], []

    if not split_dir.is_dir():
        return None, [f"{split}: missing directory"], warnings
    if not ann_path.is_file():
        return None, [f"{split}: missing _annotations.coco.json"], warnings

    try:
        coco = json.loads(ann_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return None, [f"{split}: cannot parse COCO JSON: {exc}"], warnings

    images = coco.get("images", [])
    annotations = coco.get("annotations", [])
    categories = coco.get("categories", [])

    if not isinstance(images, list):
        errors.append(f"{split}: images is not a list")
        images = []
    if not isinstance(annotations, list):
        errors.append(f"{split}: annotations is not a list")
        annotations = []
    if not isinstance(categories, list):
        errors.append(f"{split}: categories is not a list")
        categories = []

    category_by_id = {}
    for c in categories:
        try:
            cid = int(c["id"])
            name = str(c["name"]).strip()
        except Exception:
            errors.append(f"{split}: malformed category {c!r}")
            continue
        if cid in category_by_id:
            errors.append(f"{split}: duplicate category id {cid}")
        category_by_id[cid] = name

    image_by_id = {}
    filename_to_id = {}
    for img in images:
        try:
            iid = int(img["id"])
            fn = str(img["file_name"])
            width = int(img["width"])
            height = int(img["height"])
        except Exception:
            errors.append(f"{split}: malformed image entry {img!r}")
            continue

        if iid in image_by_id:
            errors.append(f"{split}: duplicate image id {iid}")
        if fn in filename_to_id:
            errors.append(f"{split}: duplicate filename {fn}")
        if width <= 0 or height <= 0:
            errors.append(f"{split}: invalid dimensions for {fn}: {width}x{height}")

        image_by_id[iid] = {
            "file_name": fn,
            "width": width,
            "height": height,
        }
        filename_to_id[fn] = iid

    actual_files = {
        p.name: p
        for p in split_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    }

    referenced_files = {img["file_name"] for img in image_by_id.values()}
    missing = referenced_files - set(actual_files)
    extra = set(actual_files) - referenced_files

    if missing:
        errors.append(f"{split}: {len(missing)} COCO image(s) missing on disk")
    if extra:
        warnings.append(f"{split}: {len(extra)} image file(s) not referenced by COCO")

    ann_ids = set()
    ann_per_image = Counter()
    ann_per_class = Counter()
    bbox_areas = defaultdict(list)
    bad_image_ref = bad_category = invalid_bbox = out_of_bounds = 0

    for ann in annotations:
        try:
            aid = int(ann["id"])
            iid = int(ann["image_id"])
            cid = int(ann["category_id"])
            x, y, w, h = map(float, ann["bbox"])
        except Exception:
            errors.append(f"{split}: malformed annotation {ann!r}")
            continue

        if aid in ann_ids:
            errors.append(f"{split}: duplicate annotation id {aid}")
        ann_ids.add(aid)

        if iid not in image_by_id:
            bad_image_ref += 1
            continue
        if cid not in category_by_id:
            bad_category += 1
            continue

        ann_per_image[iid] += 1
        cls = category_by_id[cid]
        ann_per_class[cls] += 1

        if x < 0 or y < 0 or w <= 0 or h <= 0:
            invalid_bbox += 1
            continue

        img = image_by_id[iid]
        width, height = img["width"], img["height"]
        if x + w > width + 1e-3 or y + h > height + 1e-3:
            out_of_bounds += 1

        bbox_areas[cls].append((w * h) / max(1.0, width * height))

    if bad_image_ref:
        errors.append(f"{split}: {bad_image_ref} annotation(s) reference unknown image ids")
    if bad_category:
        errors.append(f"{split}: {bad_category} annotation(s) reference unknown category ids")
    if invalid_bbox:
        errors.append(f"{split}: {invalid_bbox} invalid bbox(es)")
    if out_of_bounds:
        warnings.append(f"{split}: {out_of_bounds} bbox(es) extend outside image bounds")

    empty_images = [
        image_by_id[iid]["file_name"]
        for iid in image_by_id
        if ann_per_image[iid] == 0
    ]
    if empty_images:
        warnings.append(f"{split}: {len(empty_images)} image(s) without annotations")

    dims = Counter((x["width"], x["height"]) for x in image_by_id.values())

    return {
        "images": len(image_by_id),
        "image_files": len(actual_files),
        "annotations": len(annotations),
        "categories": category_by_id,
        "class_counts": ann_per_class,
        "bbox_areas": bbox_areas,
        "empty_images": len(empty_images),
        "dimensions": dims,
        "files": actual_files,
        "source_keys": {source_key(fn): fn for fn in actual_files},
    }, errors, warnings


def main():
    args = parse_args()
    root = args.dataset.expanduser().resolve()

    print("=== RealJaenLab dataset audit ===")
    print(f"dataset: {root}")
    print()

    errors, warnings = [], []
    data = {}

    if not root.is_dir():
        print(f"ERROR: dataset directory does not exist: {root}")
        return 1

    for split in SPLITS:
        d, e, w = audit_split(root, split)
        errors.extend(e)
        warnings.extend(w)
        if d is not None:
            data[split] = d

    category_sets = {
        split: set(d["categories"].values())
        for split, d in data.items()
    }
    if category_sets:
        first_split = next(iter(category_sets))
        first = category_sets[first_split]
        for split, names in category_sets.items():
            if names != first:
                errors.append(
                    f"category mismatch: {first_split}={sorted(first)} vs {split}={sorted(names)}"
                )
        if first != EXPECTED_CLASSES:
            missing = EXPECTED_CLASSES - first
            extra = first - EXPECTED_CLASSES
            if missing:
                errors.append(f"missing expected class(es): {sorted(missing)}")
            if extra:
                warnings.append(f"unexpected extra class(es): {sorted(extra)}")

    hashes = defaultdict(list)
    source_locations = defaultdict(list)

    for split, d in data.items():
        for fn, path in d["files"].items():
            hashes[sha256(path)].append((split, fn))
            source_locations[source_key(fn)].append((split, fn))

    exact_cross_split_duplicates = [
        locs for locs in hashes.values()
        if len({s for s, _ in locs}) > 1
    ]
    source_overlap = [
        (key, locs)
        for key, locs in source_locations.items()
        if len({s for s, _ in locs}) > 1
    ]

    if exact_cross_split_duplicates:
        warnings.append(
            f"{len(exact_cross_split_duplicates)} exact duplicate image group(s) occur across splits"
        )
    if source_overlap:
        warnings.append(
            f"{len(source_overlap)} source key(s) occur across multiple splits "
            "(possible source-level leakage)"
        )

    total_images = 0
    total_annotations = 0

    for split in SPLITS:
        d = data.get(split)
        if d is None:
            continue

        total_images += d["images"]
        total_annotations += d["annotations"]

        print(f"[{split}]")
        print(f"  images COCO:   {d['images']}")
        print(f"  image files:   {d['image_files']}")
        print(f"  annotations:   {d['annotations']}")
        print(
            "  dimensions:    "
            + ", ".join(
                f"{w}x{h}={n}"
                for (w, h), n in d["dimensions"].most_common()
            )
        )
        print("  classes:")
        for cls, count in sorted(d["class_counts"].items()):
            med = median(d["bbox_areas"].get(cls, []))
            med_text = "n/a" if med is None else f"{med * 100:.3f}% image"
            print(f"    {cls:<12} {count:>5} boxes | median area={med_text}")
        print(f"  empty images:  {d['empty_images']}")
        print()

    print(f"TOTAL images:      {total_images}")
    print(f"TOTAL annotations: {total_annotations}")

    if exact_cross_split_duplicates:
        print("\nCROSS-SPLIT EXACT DUPLICATES (first 10)")
        for group in exact_cross_split_duplicates[:10]:
            print("  " + " | ".join(f"{s}/{f}" for s, f in group))

    if source_overlap:
        print("\nCROSS-SPLIT SOURCE-KEY OVERLAP (first 20)")
        for key, locs in source_overlap[:20]:
            print(f"  {key}: " + " | ".join(f"{s}/{f}" for s, f in locs))

    print(f"\nErrors:   {len(errors)}")
    print(f"Warnings: {len(warnings)}")

    if errors:
        print("\nERRORS")
        for e in errors:
            print(f"- {e}")

    if warnings:
        print("\nWARNINGS")
        for w in warnings:
            print(f"- {w}")

    if errors:
        print("\nRESULT: FAIL")
        return 1

    print("\nRESULT: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())