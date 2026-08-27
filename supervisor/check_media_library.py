"""Offline checks for the Supervisor V2 JSON media library."""

from __future__ import annotations

from pathlib import Path
import sys
from tempfile import TemporaryDirectory

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from supervisor.media_library import (
    MediaLibrary,
    MediaLibraryConflictError,
    MediaLibraryError,
    MediaLibraryNotFoundError,
)


def expect_error(error_type: type[Exception], callback) -> None:
    try:
        callback()
    except error_type:
        return
    raise AssertionError(f"Expected {error_type.__name__}")


def main() -> None:
    with TemporaryDirectory(prefix="supervisor-media-library-") as temporary:
        root = Path(temporary)
        input_dir = root / "input"
        input_dir.mkdir()
        (input_dir / "match.mp4").touch()
        metadata = lambda _path: {
            "duration": 120.0,
            "width": 1920,
            "height": 1080,
            "fps": 25.0,
            "frame_count": 3000,
        }
        catalog_path = root / "supervisor" / "data" / "library" / "catalog_v1.json"
        library = MediaLibrary(catalog_path, root, input_dir, metadata)

        empty = library.list()
        assert empty["venues"] == [] and empty["matches"] == [] and empty["assets"] == []
        assert catalog_path.exists()

        venue = library.create_venue({"name": "La Victoria", "pitch_length_m": 105, "pitch_width_m": 68})
        venue = library.update_venue(venue["id"], {"pitch_width_m": 70.8})
        assert venue["pitch_width_m"] == 70.8

        match = library.create_match({"title": "Real Jaén - Cacereño", "date": "2026-08-27", "venue_id": venue["id"]})
        match = library.update_match(match["id"], {"title": "Real Jaén - Cacereño (amistoso)"})
        assert match["title"].endswith("(amistoso)")

        asset = library.register_legacy_input({
            "input_filename": "match.mp4",
            "display_name": "Partido completo",
            "kind": "full_match",
            "match_id": match["id"],
        })
        assert asset["relative_path"] == "input/match.mp4"
        assert asset["venue_id"] is None
        expect_error(MediaLibraryConflictError, lambda: library.register_legacy_input({
            "input_filename": "match.mp4", "display_name": "Duplicado", "kind": "full_match",
        }))

        segment = library.create_segment(asset["id"], {
            "display_name": "Primera parte", "kind": "first_half",
            "segment_start_seconds": 0, "segment_end_seconds": 60,
        })
        assert segment["source_asset_id"] == asset["id"]
        expect_error(MediaLibraryError, lambda: library.create_segment(asset["id"], {
            "display_name": "Rango inválido", "segment_start_seconds": 60, "segment_end_seconds": 60,
        }))
        expect_error(MediaLibraryError, lambda: library.create_segment(asset["id"], {
            "display_name": "Fuera de duración", "segment_start_seconds": 0, "segment_end_seconds": 121,
        }))
        expect_error(MediaLibraryNotFoundError, lambda: library.update_venue("venue_missing", {"name": "Nada"}))
        expect_error(MediaLibraryNotFoundError, lambda: library.create_segment("asset_missing", {
            "display_name": "Nada", "segment_start_seconds": 0, "segment_end_seconds": 1,
        }))

        reloaded = MediaLibrary(catalog_path, root, input_dir, metadata).list()
        assert len(reloaded["venues"]) == 1
        assert len(reloaded["matches"]) == 1
        assert len(reloaded["assets"]) == 2
        assert reloaded["legacy_unregistered"] == []
    print("check_media_library: ok")


if __name__ == "__main__":
    main()
