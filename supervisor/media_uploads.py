"""Resumable, lossless HTTP upload sessions for Supervisor managed originals.

Locks are process-local. They protect concurrent requests for one upload in a
single FastAPI process; multi-process or multi-server coordination is deferred.
"""

from __future__ import annotations

from collections.abc import AsyncIterable
from copy import deepcopy
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import shutil
from tempfile import NamedTemporaryFile
from threading import Lock
from typing import Any
from uuid import uuid4

from supervisor.media_library import (
    MediaLibrary,
    MediaLibraryConflictError,
    MediaLibraryError,
    MediaLibraryNotFoundError,
)


MAX_CHUNK_SIZE_BYTES = 64 * 1024 * 1024


class MediaUploadError(ValueError):
    """Base client-visible upload error."""


class MediaUploadNotFoundError(MediaUploadError):
    """Upload session does not exist."""


class MediaUploadConflictError(MediaUploadError):
    """Upload operation conflicts with the persisted state."""


def _timestamp() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _safe_filename(value: Any) -> tuple[str, str]:
    if not isinstance(value, str) or not value or value != Path(value).name:
        raise MediaUploadError("original_filename must be a safe filename")
    suffix = Path(value).suffix.lower()
    if suffix != ".mp4":
        raise MediaUploadError("Only .mp4 uploads are supported")
    if any(ord(character) < 32 for character in value):
        raise MediaUploadError("original_filename contains unsafe characters")
    return value, suffix


def _positive_size(value: Any) -> int:
    try:
        size = int(value)
    except (TypeError, ValueError) as error:
        raise MediaUploadError("total_size_bytes must be a positive integer") from error
    if size <= 0:
        raise MediaUploadError("total_size_bytes must be a positive integer")
    return size


def parse_content_range(value: str | None) -> tuple[int, int, int]:
    if not isinstance(value, str) or not value.startswith("bytes ") or "/" not in value:
        raise MediaUploadError("Content-Range must use bytes START-END/TOTAL")
    try:
        byte_range, total_text = value[6:].split("/", 1)
        start_text, end_text = byte_range.split("-", 1)
        start, end, total = int(start_text), int(end_text), int(total_text)
    except (TypeError, ValueError) as error:
        raise MediaUploadError("Content-Range must use bytes START-END/TOTAL") from error
    if start < 0 or end < start or total <= 0:
        raise MediaUploadError("Content-Range values are invalid")
    return start, end, total


class MediaUploadManager:
    """Persisted upload sessions backed by temporary per-chunk files."""

    def __init__(self, library_root: Path, library: MediaLibrary):
        self.library_root = Path(library_root).resolve()
        self.uploads_dir = self.library_root / "uploads"
        self.videos_dir = self.library_root / "videos"
        self.library = library
        self._locks: dict[str, Lock] = {}
        self._locks_guard = Lock()

    def _lock_for(self, upload_id: str) -> Lock:
        with self._locks_guard:
            return self._locks.setdefault(upload_id, Lock())

    @staticmethod
    def _upload_id(value: str) -> str:
        if not isinstance(value, str) or not value.startswith("upload_") or not value[7:].isalnum():
            raise MediaUploadNotFoundError("Upload session not found")
        return value

    def _session_dir(self, upload_id: str) -> Path:
        return self.uploads_dir / self._upload_id(upload_id)

    def _session_path(self, upload_id: str) -> Path:
        return self._session_dir(upload_id) / "session.json"

    def _load(self, upload_id: str) -> dict[str, Any]:
        path = self._session_path(upload_id)
        if not path.exists():
            raise MediaUploadNotFoundError("Upload session not found")
        try:
            session = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise MediaUploadError("Upload session is invalid") from error
        if not isinstance(session, dict) or session.get("upload_id") != upload_id:
            raise MediaUploadError("Upload session is invalid")
        return session

    @staticmethod
    def _write_json(path: Path, value: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as temporary:
            json.dump(value, temporary, indent=2, sort_keys=True)
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_path = Path(temporary.name)
        os.replace(temporary_path, path)

    def _save(self, session: dict[str, Any]) -> None:
        session["updated_at"] = _timestamp()
        self._write_json(self._session_path(session["upload_id"]), session)

    @staticmethod
    def _public(session: dict[str, Any]) -> dict[str, Any]:
        keys = (
            "upload_id", "status", "next_offset", "current_size_bytes",
            "total_size_bytes", "original_filename", "display_name", "kind",
            "match_id", "venue_id", "completed_asset_id", "created_at",
            "updated_at", "error_message",
        )
        return {key: deepcopy(session.get(key)) for key in keys if key in session}

    def create(self, data: dict[str, Any]) -> dict[str, Any]:
        original_filename, extension = _safe_filename(data.get("original_filename"))
        total_size = _positive_size(data.get("total_size_bytes"))
        # Validate catalog relationships before creating any upload state.
        catalog = self.library.load()
        match_id, venue_id = self.library._validate_match_and_venue(
            catalog, data.get("match_id"), data.get("venue_id")
        )
        display_name = data.get("display_name")
        kind = data.get("kind")
        if not isinstance(display_name, str) or not display_name.strip():
            raise MediaUploadError("display_name is required")
        self.library._validate_kind(kind)
        upload_id = f"upload_{uuid4().hex}"
        now = _timestamp()
        session = {
            "upload_id": upload_id,
            "original_filename": original_filename,
            "extension": extension,
            "total_size_bytes": total_size,
            "current_size_bytes": 0,
            "next_offset": 0,
            "display_name": display_name.strip(),
            "kind": kind,
            "match_id": match_id,
            "venue_id": venue_id,
            "status": "uploading",
            "completed_asset_id": None,
            "created_at": now,
            "updated_at": now,
        }
        directory = self._session_dir(upload_id)
        directory.mkdir(parents=True, exist_ok=False)
        (directory / "chunks").mkdir()
        self._write_json(directory / "session.json", session)
        return self._public(session)

    def get(self, upload_id: str) -> dict[str, Any]:
        return self._public(self._load(upload_id))

    def list(self) -> list[dict[str, Any]]:
        if not self.uploads_dir.exists():
            return []
        sessions = []
        for directory in sorted(self.uploads_dir.iterdir()):
            if not directory.is_dir():
                continue
            try:
                sessions.append(self.get(directory.name))
            except MediaUploadError:
                continue
        return sessions

    def _validate_chunk_range(self, session: dict[str, Any], start: int, end: int, total: int) -> int:
        if session.get("status") != "uploading":
            raise MediaUploadConflictError("Upload is not accepting chunks")
        if total != session["total_size_bytes"]:
            raise MediaUploadConflictError("Content-Range total does not match the upload")
        if start != session["next_offset"]:
            raise MediaUploadConflictError("Content-Range start does not match the next upload offset")
        expected = end - start + 1
        if expected > MAX_CHUNK_SIZE_BYTES:
            raise MediaUploadError("Chunk exceeds the maximum allowed size")
        if end >= total:
            raise MediaUploadError("Content-Range exceeds the upload size")
        return expected

    async def append_chunk(
        self,
        upload_id: str,
        start: int,
        end: int,
        total: int,
        body: AsyncIterable[bytes],
    ) -> dict[str, Any]:
        # Early validation gives a client fast feedback; commit repeats it under
        # the session lock so simultaneous requests cannot corrupt the payload.
        session = self._load(upload_id)
        expected = self._validate_chunk_range(session, start, end, total)
        chunk_path = self._session_dir(upload_id) / "chunks" / f"{uuid4().hex}.part"
        received = 0
        try:
            with chunk_path.open("xb") as temporary:
                async for block in body:
                    if not block:
                        continue
                    received += len(block)
                    if received > expected or received > MAX_CHUNK_SIZE_BYTES:
                        raise MediaUploadError("Chunk byte count exceeds Content-Range")
                    temporary.write(block)
                temporary.flush()
                os.fsync(temporary.fileno())
            if received != expected:
                raise MediaUploadError("Chunk byte count does not match Content-Range")
            return self._commit_chunk(upload_id, start, end, total, chunk_path, received)
        finally:
            if chunk_path.exists():
                chunk_path.unlink()

    def _commit_chunk(
        self,
        upload_id: str,
        start: int,
        end: int,
        total: int,
        chunk_path: Path,
        received: int,
    ) -> dict[str, Any]:
        with self._lock_for(upload_id):
            session = self._load(upload_id)
            expected = self._validate_chunk_range(session, start, end, total)
            if received != expected or chunk_path.stat().st_size != expected:
                raise MediaUploadError("Chunk byte count does not match Content-Range")
            payload_path = self._session_dir(upload_id) / "payload.part"
            committed_size = int(session["current_size_bytes"])
            if payload_path.exists() and payload_path.stat().st_size != committed_size:
                # A previous process may have crashed after appending but before
                # atomically advancing the session. The session remains truth.
                with payload_path.open("r+b") as payload:
                    payload.truncate(committed_size)
                    payload.flush()
                    os.fsync(payload.fileno())
            with payload_path.open("ab") as payload, chunk_path.open("rb") as chunk:
                shutil.copyfileobj(chunk, payload, length=1024 * 1024)
                payload.flush()
                os.fsync(payload.fileno())
            if payload_path.stat().st_size != committed_size + expected:
                raise MediaUploadError("Chunk could not be committed safely")
            session["current_size_bytes"] = committed_size + expected
            session["next_offset"] = session["current_size_bytes"]
            self._save(session)
            return self._public(session)

    def _managed_filename(self, session: dict[str, Any]) -> str:
        return session.get("managed_filename") or f"managed_{session['upload_id']}{session['extension']}"

    def _recover_or_register_completed_asset(self, session: dict[str, Any], final_path: Path) -> dict[str, Any]:
        relative_path = final_path.resolve().relative_to(self.library.project_root).as_posix()
        existing = self.library.asset_by_relative_path(relative_path)
        if existing is not None:
            return existing
        metadata = self.library.metadata_reader(final_path)
        if not isinstance(metadata, dict) or not metadata.get("width") or not metadata.get("height") or metadata.get("duration") is None:
            session["status"] = "error"
            session["error_message"] = "Uploaded file could not be read as a video"
            self._save(session)
            raise MediaUploadError(session["error_message"])
        technical = {**metadata, "file_size_bytes": final_path.stat().st_size}
        return self.library.register_managed_original({
            "server_filename": final_path.name,
            "display_name": session["display_name"],
            "kind": session["kind"],
            "match_id": session.get("match_id"),
            "venue_id": session.get("venue_id"),
            "technical": technical,
            "original_filename": session["original_filename"],
        })

    def complete(self, upload_id: str) -> dict[str, Any]:
        with self._lock_for(upload_id):
            session = self._load(upload_id)
            if session.get("completed_asset_id"):
                try:
                    return self.library._find(self.library.load()["assets"], session["completed_asset_id"], "Video asset")
                except MediaLibraryNotFoundError:
                    pass
            if session["current_size_bytes"] != session["total_size_bytes"]:
                raise MediaUploadConflictError("Upload is incomplete")
            payload_path = self._session_dir(upload_id) / "payload.part"
            self.videos_dir.mkdir(parents=True, exist_ok=True)
            managed_filename = self._managed_filename(session)
            final_path = (self.videos_dir / managed_filename).resolve()
            if final_path.parent != self.videos_dir:
                raise MediaUploadError("Managed video path is invalid")
            session["status"] = "completing"
            session["managed_filename"] = managed_filename
            session["final_relative_path"] = final_path.relative_to(self.library.project_root).as_posix()
            self._save(session)
            if not final_path.exists():
                if not payload_path.is_file() or payload_path.stat().st_size != session["total_size_bytes"]:
                    raise MediaUploadConflictError("Uploaded payload size does not match the session")
                os.replace(payload_path, final_path)
            asset = self._recover_or_register_completed_asset(session, final_path)
            session["status"] = "completed"
            session["completed_asset_id"] = asset["id"]
            session.pop("error_message", None)
            self._save(session)
            return asset

    def abort(self, upload_id: str) -> None:
        with self._lock_for(upload_id):
            session = self._load(upload_id)
            if session.get("status") == "completed" or session.get("completed_asset_id"):
                raise MediaUploadConflictError("Completed uploads cannot be aborted")
            if session.get("managed_filename"):
                raise MediaUploadConflictError("Upload completion has already started; uploaded media is preserved")
            shutil.rmtree(self._session_dir(upload_id))
