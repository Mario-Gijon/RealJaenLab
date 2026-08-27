"""Small persistent media catalog for the local Supervisor V2 library."""

from __future__ import annotations

from copy import deepcopy
from datetime import UTC, datetime
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Callable
from uuid import uuid4


CATALOG_VERSION = 1
ASSET_KINDS = {"full_match", "first_half", "second_half", "clip", "prematch", "other"}
STORAGE_TYPES = {"legacy_input", "managed_original", "segment_derived"}


class MediaLibraryError(ValueError):
    """Base error for client-invalid media-library operations."""


class MediaLibraryNotFoundError(MediaLibraryError):
    """Requested catalog item does not exist."""


class MediaLibraryConflictError(MediaLibraryError):
    """Requested operation would duplicate a catalog physical asset."""


def _timestamp() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _identifier(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex}"


def _optional_positive(value: Any, field: str) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise MediaLibraryError(f"{field} must be a positive number") from error
    if number <= 0:
        raise MediaLibraryError(f"{field} must be a positive number")
    return number


def _required_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MediaLibraryError(f"{field} is required")
    return value.strip()


def _optional_text(value: Any, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise MediaLibraryError(f"{field} must be text")
    return value.strip() or None


class MediaLibrary:
    """JSON-backed Venue, Match, and VideoAsset catalog operations."""

    def __init__(
        self,
        catalog_path: Path,
        project_root: Path,
        input_dir: Path,
        metadata_reader: Callable[[Path], dict[str, Any]] | None = None,
        managed_videos_dir: Path | None = None,
    ):
        self.catalog_path = Path(catalog_path)
        self.project_root = Path(project_root).resolve()
        self.input_dir = Path(input_dir).resolve()
        self.metadata_reader = metadata_reader or (lambda _path: {})
        self.managed_videos_dir = Path(managed_videos_dir or self.catalog_path.parent / "videos").resolve()

    @staticmethod
    def empty_catalog() -> dict[str, Any]:
        return {
            "schema_version": CATALOG_VERSION,
            "venues": [],
            "matches": [],
            "assets": [],
        }

    def load(self) -> dict[str, Any]:
        if not self.catalog_path.exists():
            catalog = self.empty_catalog()
            self._write(catalog)
            return catalog
        try:
            catalog = json.loads(self.catalog_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise MediaLibraryError("Media catalog is not valid JSON") from error
        if not isinstance(catalog, dict) or catalog.get("schema_version") != CATALOG_VERSION:
            raise MediaLibraryError("Media catalog schema is not supported")
        for key in ("venues", "matches", "assets"):
            if not isinstance(catalog.get(key), list):
                raise MediaLibraryError("Media catalog structure is invalid")
        return catalog

    def _write(self, catalog: dict[str, Any]) -> None:
        self.catalog_path.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=self.catalog_path.parent,
            prefix=f".{self.catalog_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            json.dump(catalog, temporary, indent=2, sort_keys=True)
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_path = Path(temporary.name)
        os.replace(temporary_path, self.catalog_path)

    def _save(self, catalog: dict[str, Any]) -> None:
        self._write(catalog)

    @staticmethod
    def _find(items: list[dict[str, Any]], item_id: str, label: str) -> dict[str, Any]:
        for item in items:
            if item.get("id") == item_id:
                return item
        raise MediaLibraryNotFoundError(f"{label} not found")

    def _relative_input_path(self, filename: Any) -> tuple[Path, str]:
        if not isinstance(filename, str) or not filename:
            raise MediaLibraryError("input_filename is required")
        candidate_name = Path(filename).name
        if candidate_name != filename or Path(candidate_name).suffix.lower() != ".mp4":
            raise MediaLibraryError("input_filename must name an existing input MP4")
        path = (self.input_dir / candidate_name).resolve()
        if path.parent != self.input_dir or not path.is_file():
            raise MediaLibraryNotFoundError("Input video not found")
        return path, path.relative_to(self.project_root).as_posix()

    def _managed_video_path(self, server_filename: Any) -> tuple[Path, str]:
        if not isinstance(server_filename, str) or not server_filename:
            raise MediaLibraryError("managed server filename is required")
        candidate_name = Path(server_filename).name
        if candidate_name != server_filename or Path(candidate_name).suffix.lower() != ".mp4":
            raise MediaLibraryError("managed server filename must be a safe MP4 filename")
        path = (self.managed_videos_dir / candidate_name).resolve()
        if path.parent != self.managed_videos_dir or not path.is_file():
            raise MediaLibraryNotFoundError("Managed video not found")
        return path, path.relative_to(self.project_root).as_posix()

    @staticmethod
    def _validate_kind(value: Any) -> str:
        if value not in ASSET_KINDS:
            raise MediaLibraryError("kind is not supported")
        return str(value)

    def _validate_match_and_venue(
        self,
        catalog: dict[str, Any],
        match_id: Any,
        venue_id: Any,
    ) -> tuple[str | None, str | None]:
        match = None
        if match_id is not None:
            if not isinstance(match_id, str):
                raise MediaLibraryError("match_id must be text")
            match = self._find(catalog["matches"], match_id, "Match")
        if venue_id is not None:
            if not isinstance(venue_id, str):
                raise MediaLibraryError("venue_id must be text")
            self._find(catalog["venues"], venue_id, "Venue")
        if match is not None and venue_id is not None and match.get("venue_id") not in (None, venue_id):
            raise MediaLibraryError("Asset venue conflicts with the assigned match venue")
        # A match-owned asset inherits venue through the match; do not duplicate it.
        return match_id, venue_id if match is None else None

    def list(self) -> dict[str, Any]:
        catalog = self.load()
        registered_paths = {
            asset.get("relative_path")
            for asset in catalog["assets"]
            if asset.get("storage_type") in {"legacy_input", "managed_original"}
        }
        legacy_unregistered = []
        if self.input_dir.exists():
            for path in sorted(self.input_dir.iterdir()):
                if not path.is_file() or path.suffix.lower() != ".mp4":
                    continue
                relative_path = path.resolve().relative_to(self.project_root).as_posix()
                if relative_path in registered_paths:
                    continue
                legacy_unregistered.append({
                    "filename": path.name,
                    "display_name": path.stem,
                    "storage_type": "legacy_input",
                    "relative_path": relative_path,
                    "technical": self.metadata_reader(path),
                })
        return {
            "schema_version": catalog["schema_version"],
            "venues": deepcopy(catalog["venues"]),
            "matches": deepcopy(catalog["matches"]),
            "assets": deepcopy(catalog["assets"]),
            "legacy_unregistered": legacy_unregistered,
        }

    def create_venue(self, data: dict[str, Any]) -> dict[str, Any]:
        catalog = self.load()
        now = _timestamp()
        venue = {
            "id": _identifier("venue"),
            "name": _required_text(data.get("name"), "name"),
            "pitch_length_m": _optional_positive(data.get("pitch_length_m"), "pitch_length_m"),
            "pitch_width_m": _optional_positive(data.get("pitch_width_m"), "pitch_width_m"),
            "created_at": now,
            "updated_at": now,
        }
        catalog["venues"].append(venue)
        self._save(catalog)
        return deepcopy(venue)

    def update_venue(self, venue_id: str, data: dict[str, Any]) -> dict[str, Any]:
        catalog = self.load()
        venue = self._find(catalog["venues"], venue_id, "Venue")
        if "name" in data:
            venue["name"] = _required_text(data["name"], "name")
        for field in ("pitch_length_m", "pitch_width_m"):
            if field in data:
                venue[field] = _optional_positive(data[field], field)
        venue["updated_at"] = _timestamp()
        self._save(catalog)
        return deepcopy(venue)

    def create_match(self, data: dict[str, Any]) -> dict[str, Any]:
        catalog = self.load()
        venue_id = data.get("venue_id")
        if venue_id is not None:
            self._find(catalog["venues"], venue_id, "Venue")
        now = _timestamp()
        match = {
            "id": _identifier("match"),
            "title": _required_text(data.get("title"), "title"),
            "date": _optional_text(data.get("date"), "date"),
            "venue_id": venue_id,
            "created_at": now,
            "updated_at": now,
        }
        catalog["matches"].append(match)
        self._save(catalog)
        return deepcopy(match)

    def update_match(self, match_id: str, data: dict[str, Any]) -> dict[str, Any]:
        catalog = self.load()
        match = self._find(catalog["matches"], match_id, "Match")
        if "title" in data:
            match["title"] = _required_text(data["title"], "title")
        if "date" in data:
            match["date"] = _optional_text(data["date"], "date")
        if "venue_id" in data:
            venue_id = data["venue_id"]
            if venue_id is not None:
                if not isinstance(venue_id, str):
                    raise MediaLibraryError("venue_id must be text")
                self._find(catalog["venues"], venue_id, "Venue")
            match["venue_id"] = venue_id
        match["updated_at"] = _timestamp()
        self._save(catalog)
        return deepcopy(match)

    def register_legacy_input(self, data: dict[str, Any]) -> dict[str, Any]:
        catalog = self.load()
        path, relative_path = self._relative_input_path(data.get("input_filename"))
        if any(
            asset.get("storage_type") in {"legacy_input", "managed_original"}
            and asset.get("relative_path") == relative_path
            for asset in catalog["assets"]
        ):
            raise MediaLibraryConflictError("This physical video is already registered")
        match_id, venue_id = self._validate_match_and_venue(
            catalog, data.get("match_id"), data.get("venue_id")
        )
        now = _timestamp()
        asset = {
            "id": _identifier("asset"),
            "display_name": _required_text(data.get("display_name"), "display_name"),
            "kind": self._validate_kind(data.get("kind")),
            "match_id": match_id,
            "venue_id": venue_id,
            "storage_type": "legacy_input",
            "relative_path": relative_path,
            "source_asset_id": None,
            "segment_start_seconds": None,
            "segment_end_seconds": None,
            "technical": {**self.metadata_reader(path), "file_size_bytes": path.stat().st_size},
            "created_at": now,
            "updated_at": now,
        }
        catalog["assets"].append(asset)
        self._save(catalog)
        return deepcopy(asset)

    def register_managed_original(self, data: dict[str, Any]) -> dict[str, Any]:
        """Register only a server-controlled file under ``managed_videos_dir``."""

        catalog = self.load()
        path, relative_path = self._managed_video_path(data.get("server_filename"))
        if any(
            asset.get("storage_type") in {"legacy_input", "managed_original"}
            and asset.get("relative_path") == relative_path
            for asset in catalog["assets"]
        ):
            raise MediaLibraryConflictError("This managed physical video is already registered")
        match_id, venue_id = self._validate_match_and_venue(
            catalog, data.get("match_id"), data.get("venue_id")
        )
        technical = data.get("technical")
        if not isinstance(technical, dict):
            raise MediaLibraryError("technical metadata is required")
        now = _timestamp()
        asset = {
            "id": _identifier("asset"),
            "display_name": _required_text(data.get("display_name"), "display_name"),
            "kind": self._validate_kind(data.get("kind")),
            "match_id": match_id,
            "venue_id": venue_id,
            "storage_type": "managed_original",
            "relative_path": relative_path,
            "source_asset_id": None,
            "segment_start_seconds": None,
            "segment_end_seconds": None,
            "technical": deepcopy(technical),
            "original_filename": _optional_text(data.get("original_filename"), "original_filename"),
            "created_at": now,
            "updated_at": now,
        }
        catalog["assets"].append(asset)
        self._save(catalog)
        return deepcopy(asset)

    def asset_by_relative_path(self, relative_path: str) -> dict[str, Any] | None:
        catalog = self.load()
        for asset in catalog["assets"]:
            if asset.get("relative_path") == relative_path:
                return deepcopy(asset)
        return None

    def resolve_asset_path(self, asset_id: str) -> Path:
        catalog = self.load()
        asset = self._find(catalog["assets"], asset_id, "Video asset")
        visited: set[str] = set()
        while asset.get("storage_type") == "segment_derived":
            if asset["id"] in visited:
                raise MediaLibraryError("Segment source chain is invalid")
            visited.add(asset["id"])
            asset = self._find(catalog["assets"], asset.get("source_asset_id"), "Segment source")
        relative_path = asset.get("relative_path")
        if not isinstance(relative_path, str):
            raise MediaLibraryError("Asset path is invalid")
        root = self.input_dir if asset.get("storage_type") == "legacy_input" else self.managed_videos_dir
        path = (self.project_root / relative_path).resolve()
        if path.parent != root or not path.is_file():
            raise MediaLibraryNotFoundError("Asset video not found")
        return path

    def update_asset(self, asset_id: str, data: dict[str, Any]) -> dict[str, Any]:
        catalog = self.load()
        asset = self._find(catalog["assets"], asset_id, "Video asset")
        if "display_name" in data:
            asset["display_name"] = _required_text(data["display_name"], "display_name")
        if "kind" in data:
            asset["kind"] = self._validate_kind(data["kind"])
        if "match_id" in data or "venue_id" in data:
            match_id = data.get("match_id", asset.get("match_id"))
            venue_id = data.get("venue_id", asset.get("venue_id"))
            asset["match_id"], asset["venue_id"] = self._validate_match_and_venue(catalog, match_id, venue_id)
        asset["updated_at"] = _timestamp()
        self._save(catalog)
        return deepcopy(asset)

    def create_segment(self, asset_id: str, data: dict[str, Any]) -> dict[str, Any]:
        catalog = self.load()
        source = self._find(catalog["assets"], asset_id, "Video asset")
        try:
            start = float(data.get("segment_start_seconds"))
            end = float(data.get("segment_end_seconds"))
        except (TypeError, ValueError) as error:
            raise MediaLibraryError("segment start and end seconds are required") from error
        if start < 0 or end <= start:
            raise MediaLibraryError("segment range must satisfy 0 <= start < end")
        duration = source.get("technical", {}).get("duration")
        if duration is not None and end > float(duration):
            raise MediaLibraryError("segment end exceeds source duration")
        now = _timestamp()
        segment = {
            "id": _identifier("asset"),
            "display_name": _required_text(data.get("display_name"), "display_name"),
            "kind": self._validate_kind(data.get("kind", "clip")),
            "match_id": source.get("match_id"),
            "venue_id": source.get("venue_id") if source.get("match_id") is None else None,
            "storage_type": "segment_derived",
            "relative_path": source.get("relative_path"),
            "source_asset_id": source["id"],
            "segment_start_seconds": start,
            "segment_end_seconds": end,
            "technical": {},
            "created_at": now,
            "updated_at": now,
        }
        catalog["assets"].append(segment)
        self._save(catalog)
        return deepcopy(segment)
