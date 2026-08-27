"""Offline checks for lossless resumable Supervisor managed-original uploads."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from supervisor.media_library import MediaLibrary, MediaLibraryConflictError, MediaLibraryError
from supervisor.media_uploads import (
    MAX_CHUNK_SIZE_BYTES,
    MediaUploadConflictError,
    MediaUploadError,
    MediaUploadManager,
)


async def body(payload: bytes):
    for index in range(0, len(payload), 2):
        yield payload[index:index + 2]


def expect_error(error_type: type[Exception], callback) -> None:
    try:
        callback()
    except error_type:
        return
    raise AssertionError(f"Expected {error_type.__name__}")


def append(manager: MediaUploadManager, upload_id: str, start: int, end: int, total: int, payload: bytes) -> dict:
    return asyncio.run(manager.append_chunk(upload_id, start, end, total, body(payload)))


def create(manager: MediaUploadManager, total_size: int = 10, filename: str = "camera.mp4") -> dict:
    return manager.create({
        "original_filename": filename,
        "total_size_bytes": total_size,
        "display_name": "Cámara principal",
        "kind": "full_match",
    })


def main() -> None:
    with TemporaryDirectory(prefix="supervisor-media-uploads-") as temporary:
        root = Path(temporary)
        input_dir = root / "input"
        input_dir.mkdir()
        library_root = root / "supervisor" / "data" / "library"
        metadata = lambda _path: {
            "duration": 12.5, "width": 1920, "height": 1080,
            "fps": 25.0, "frame_count": 313,
        }
        library = MediaLibrary(
            library_root / "catalog_v1.json", root, input_dir, metadata,
            managed_videos_dir=library_root / "videos",
        )
        manager = MediaUploadManager(library_root, library)

        session = create(manager)
        upload_id = session["upload_id"]
        assert manager.get(upload_id)["next_offset"] == 0
        expect_error(MediaUploadConflictError, lambda: append(manager, upload_id, 1, 2, 10, b"ab"))
        expect_error(MediaUploadConflictError, lambda: append(manager, upload_id, 0, 1, 11, b"ab"))
        expect_error(MediaUploadError, lambda: append(manager, upload_id, 0, 2, 10, b"ab"))

        append(manager, upload_id, 0, 3, 10, b"abcd")
        assert manager.get(upload_id)["next_offset"] == 4
        expect_error(MediaUploadConflictError, lambda: append(manager, upload_id, 0, 1, 10, b"ab"))
        append(manager, upload_id, 4, 5, 10, b"ef")
        reloaded_manager = MediaUploadManager(library_root, library)
        assert reloaded_manager.get(upload_id)["next_offset"] == 6
        append(reloaded_manager, upload_id, 6, 9, 10, b"ghij")
        expect_error(MediaUploadError, lambda: reloaded_manager.complete(upload_id + "0"))
        asset = reloaded_manager.complete(upload_id)
        assert asset["storage_type"] == "managed_original"
        managed_path = library.resolve_asset_path(asset["id"])
        assert managed_path.parent == library.managed_videos_dir
        assert managed_path.read_bytes() == b"abcdefghij"
        assert reloaded_manager.complete(upload_id)["id"] == asset["id"]
        expect_error(MediaLibraryConflictError, lambda: library.register_managed_original({
            "server_filename": managed_path.name,
            "display_name": "Duplicado", "kind": "full_match", "technical": metadata(managed_path),
        }))
        expect_error(MediaUploadConflictError, lambda: reloaded_manager.abort(upload_id))

        oversized = create(reloaded_manager, MAX_CHUNK_SIZE_BYTES + 1)
        expect_error(MediaUploadError, lambda: append(
            reloaded_manager, oversized["upload_id"], 0, MAX_CHUNK_SIZE_BYTES, MAX_CHUNK_SIZE_BYTES + 1, b"x"
        ))
        incomplete = create(reloaded_manager, 4)
        expect_error(MediaUploadConflictError, lambda: reloaded_manager.complete(incomplete["upload_id"]))
        incomplete_dir = reloaded_manager._session_dir(incomplete["upload_id"])
        reloaded_manager.abort(incomplete["upload_id"])
        assert not incomplete_dir.exists()
        expect_error(MediaUploadError, lambda: create(reloaded_manager, filename="../unsafe.mp4"))
        expect_error(MediaUploadError, lambda: create(reloaded_manager, filename="camera.mov"))
        expect_error(MediaLibraryError, lambda: library.register_managed_original({
            "server_filename": "../escape.mp4", "display_name": "Escape", "kind": "other", "technical": {},
        }))

        recovery = create(reloaded_manager, 4, "recovery.mp4")
        recovery_id = recovery["upload_id"]
        append(reloaded_manager, recovery_id, 0, 3, 4, b"data")
        recovery_session = reloaded_manager._load(recovery_id)
        recovery_session["status"] = "completing"
        recovery_session["managed_filename"] = f"managed_{recovery_id}.mp4"
        recovery_final = reloaded_manager.videos_dir / recovery_session["managed_filename"]
        recovery_final.parent.mkdir(parents=True, exist_ok=True)
        os.replace(reloaded_manager._session_dir(recovery_id) / "payload.part", recovery_final)
        reloaded_manager._save(recovery_session)
        recovered_asset = reloaded_manager.complete(recovery_id)
        assert recovered_asset["id"] == reloaded_manager.complete(recovery_id)["id"]
        assert recovery_final.read_bytes() == b"data"
    print("check_media_uploads: ok")


if __name__ == "__main__":
    main()
