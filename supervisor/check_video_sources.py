"""Focused regression checks for safe Supervisor V2 video-source resolution."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from supervisor.media_library import MediaLibrary
from supervisor.video_sources import (
    VideoSourceNotFoundError,
    VideoSourceUnsupportedError,
    resolve_video_source,
)


def _metadata(_path: Path) -> dict[str, int | float]:
    return {"width": 1920, "height": 1080, "duration": 10.0}


def _update_temporary_asset_metadata(catalog_path: Path, asset_id: str) -> None:
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    asset = next(item for item in catalog["assets"] if item["id"] == asset_id)
    asset["original_filename"] = "renamed-source.mp4"
    catalog_path.write_text(json.dumps(catalog), encoding="utf-8")


def main() -> None:
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        input_dir = root / "input"
        managed_dir = root / "supervisor" / "data" / "library" / "videos"
        review_dir = root / "supervisor" / "data" / "reviews"
        input_dir.mkdir(parents=True)
        managed_dir.mkdir(parents=True)

        legacy_path = input_dir / "benchmark_normal_2min.mp4"
        legacy_path.write_bytes(b"legacy")
        managed_path = managed_dir / "managed_upload_123.mp4"
        managed_path.write_bytes(b"managed")

        library = MediaLibrary(
            catalog_path=root / "supervisor" / "data" / "library" / "catalog_v1.json",
            project_root=root,
            input_dir=input_dir,
            managed_videos_dir=managed_dir,
            metadata_reader=_metadata,
        )
        managed = library.register_managed_original({
            "server_filename": managed_path.name,
            "display_name": "Primer original",
            "original_filename": "camera-original.mp4",
            "kind": "full_match",
            "technical": _metadata(managed_path),
        })

        legacy = resolve_video_source(legacy_path.name, library=library, input_dir=input_dir)
        assert legacy.source_type == "legacy_input"
        assert legacy.path == legacy_path.resolve()
        assert legacy.path.parent == input_dir.resolve()
        assert legacy.review_path(review_dir) == review_dir / "benchmark_normal_2min.json"
        assert legacy.result_basename("_camera_atlas_dense_v1.json") == "benchmark_normal_2min_camera_atlas_dense_v1.json"

        managed_source = resolve_video_source(managed["id"], library=library, input_dir=input_dir)
        assert managed_source.source_type == "managed_original"
        assert managed_source.asset_id == managed["id"]
        assert managed_source.path == managed_path.resolve()
        assert managed_source.path.parent == managed_dir.resolve()
        assert managed_source.review_path(review_dir) == review_dir / f"{managed['id']}.json"
        assert managed_source.result_basename("_camera_atlas_dense_v1.json") == (
            f"{managed['id']}_camera_atlas_dense_v1.json"
        )

        original_review_key = managed_source.review_key
        library.update_asset(managed["id"], {"display_name": "Nombre visible cambiado"})
        assert resolve_video_source(managed["id"], library=library, input_dir=input_dir).review_key == original_review_key
        _update_temporary_asset_metadata(library.catalog_path, managed["id"])
        assert resolve_video_source(managed["id"], library=library, input_dir=input_dir).review_key == original_review_key

        segment = library.create_segment(managed["id"], {
            "display_name": "Segmento",
            "segment_start_seconds": 0,
            "segment_end_seconds": 2,
        })
        for invalid_reference in ("../benchmark_normal_2min.mp4", "/tmp/anything.mp4", "missing_asset"):
            try:
                resolve_video_source(invalid_reference, library=library, input_dir=input_dir)
            except VideoSourceNotFoundError:
                pass
            else:
                raise AssertionError(f"unsafe or missing reference was accepted: {invalid_reference}")
        try:
            resolve_video_source(segment["id"], library=library, input_dir=input_dir)
        except VideoSourceUnsupportedError:
            pass
        else:
            raise AssertionError("segment_derived was accepted as a workspace source")

        managed_review = managed_source.review_path(review_dir)
        managed_review.parent.mkdir(parents=True, exist_ok=True)
        managed_review.write_text(json.dumps({"field": {"calibrations": []}}), encoding="utf-8")
        assert json.loads(managed_review.read_text(encoding="utf-8"))["field"]["calibrations"] == []

    print("check_video_sources: ok")


if __name__ == "__main__":
    main()
