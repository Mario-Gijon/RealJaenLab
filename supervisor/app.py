"""Minimal local Supervisor for the RealJaen video foundation."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from supervisor.media_library import (
    MediaLibrary,
    MediaLibraryConflictError,
    MediaLibraryError,
    MediaLibraryNotFoundError,
)
from supervisor.media_uploads import (
    MediaUploadConflictError,
    MediaUploadError,
    MediaUploadManager,
    MediaUploadNotFoundError,
    parse_content_range,
)
from supervisor.field.calibration import CalibrationError, solve_calibration_payload
from supervisor.field.pitch import PitchModel
from supervisor.field.projection_support import build_projection_support_payload
from supervisor.field.camera_model_probe import probe_camera_models_payload


BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"
INPUT_DIR = BASE_DIR.parent / "input"
LIBRARY_DIR = BASE_DIR / "data" / "library"
_media_upload_manager: MediaUploadManager | None = None

app = FastAPI(title="RealJaen Supervisor")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


def read_video_metadata(video_path: Path) -> dict[str, float | int | None]:
    """Read the playback metadata required by the browser controls."""

    command = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=avg_frame_rate,nb_frames,duration,width,height",
        "-show_entries", "format=duration", "-of", "json", str(video_path),
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        probe = json.loads(result.stdout)
        stream = probe.get("streams", [{}])[0]
        numerator, denominator = stream.get("avg_frame_rate", "0/1").split("/", 1)
        fps = float(numerator) / float(denominator) if float(denominator) else None
        duration = float(stream.get("duration") or probe.get("format", {}).get("duration") or 0)
        frame_count = stream.get("nb_frames")
        return {
            "fps": fps,
            "duration": duration or None,
            "frame_count": int(frame_count) if frame_count and frame_count != "N/A" else None,
            "width": int(stream["width"]) if stream.get("width") else None,
            "height": int(stream["height"]) if stream.get("height") else None,
        }
    except (FileNotFoundError, subprocess.CalledProcessError, ValueError, json.JSONDecodeError):
        return {"fps": None, "duration": None, "frame_count": None, "width": None, "height": None}


def media_library() -> MediaLibrary:
    return MediaLibrary(
        catalog_path=LIBRARY_DIR / "catalog_v1.json",
        project_root=BASE_DIR.parent,
        input_dir=INPUT_DIR,
        metadata_reader=read_video_metadata,
        managed_videos_dir=LIBRARY_DIR / "videos",
    )


def media_uploads() -> MediaUploadManager:
    global _media_upload_manager
    if _media_upload_manager is None:
        _media_upload_manager = MediaUploadManager(LIBRARY_DIR, media_library())
    return _media_upload_manager


def media_library_error(error: MediaLibraryError) -> HTTPException:
    return HTTPException(status_code=404 if isinstance(error, MediaLibraryNotFoundError) else 400, detail=str(error))


def media_upload_error(error: MediaUploadError | MediaLibraryError) -> HTTPException:
    if isinstance(error, (MediaUploadNotFoundError, MediaLibraryNotFoundError)):
        status = 404
    elif isinstance(error, (MediaUploadConflictError, MediaLibraryConflictError)):
        status = 409
    else:
        status = 400
    return HTTPException(status_code=status, detail=str(error))


@app.get("/api/library")
def get_media_library() -> dict:
    try:
        return media_library().list()
    except MediaLibraryError as error:
        raise media_library_error(error) from error


@app.get("/api/field/pitch-lines")
def get_field_pitch_lines() -> dict:
    pitch = PitchModel()
    return pitch.geometry_dict()


@app.post("/api/field/calibration/solve")
def solve_field_calibration(payload: dict = Body()) -> dict:
    try:
        result = solve_calibration_payload(payload)
    except (CalibrationError, ValueError, TypeError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    try:
        projection_support = build_projection_support_payload(result, payload)
    except Exception as error:  # Support diagnostics must not invalidate a solve.
        projection_support = {
            "support_image_polygon": [],
            "visible_image_polygon": [],
            "projective_valid": False,
            "warnings": [f"No se pudo calcular el soporte de proyección: {error}"],
        }
    try:
        camera_model_probe = probe_camera_models_payload(result, payload)
    except Exception as error:  # The probe is diagnostic and must be non-fatal.
        camera_model_probe = {
            "models": {},
            "best_regularized_model": None,
            "warning": f"No se pudo ejecutar la prueba de modelos de cámara: {error}",
        }
    return {
        "asset_id": payload.get("asset_id"),
        "timestamp_seconds": payload.get("timestamp_seconds"),
        "frame_index": payload.get("frame_index"),
        "video_width": payload.get("video_width"),
        "video_height": payload.get("video_height"),
        "projection_support": projection_support,
        "camera_model_probe": camera_model_probe,
        **result,
    }


@app.post("/api/library/venues")
def create_library_venue(payload: dict = Body()) -> dict:
    try:
        return {"venue": media_library().create_venue(payload)}
    except MediaLibraryError as error:
        raise media_library_error(error) from error


@app.patch("/api/library/venues/{venue_id}")
def update_library_venue(venue_id: str, payload: dict = Body()) -> dict:
    try:
        return {"venue": media_library().update_venue(venue_id, payload)}
    except MediaLibraryError as error:
        raise media_library_error(error) from error


@app.post("/api/library/matches")
def create_library_match(payload: dict = Body()) -> dict:
    try:
        return {"match": media_library().create_match(payload)}
    except MediaLibraryError as error:
        raise media_library_error(error) from error


@app.patch("/api/library/matches/{match_id}")
def update_library_match(match_id: str, payload: dict = Body()) -> dict:
    try:
        return {"match": media_library().update_match(match_id, payload)}
    except MediaLibraryError as error:
        raise media_library_error(error) from error


@app.post("/api/library/assets/register")
def register_library_asset(payload: dict = Body()) -> dict:
    try:
        return {"asset": media_library().register_local_input(payload)}
    except MediaLibraryError as error:
        raise media_library_error(error) from error


@app.patch("/api/library/assets/{asset_id}")
def update_library_asset(asset_id: str, payload: dict = Body()) -> dict:
    try:
        return {"asset": media_library().update_asset(asset_id, payload)}
    except MediaLibraryError as error:
        raise media_library_error(error) from error


@app.post("/api/library/uploads")
def create_library_upload(payload: dict = Body()) -> dict:
    try:
        return media_uploads().create(payload)
    except (MediaUploadError, MediaLibraryError) as error:
        raise media_upload_error(error) from error


@app.get("/api/library/uploads")
def list_library_uploads() -> dict:
    return {"uploads": media_uploads().list()}


@app.get("/api/library/uploads/{upload_id}")
def get_library_upload(upload_id: str) -> dict:
    try:
        return media_uploads().get(upload_id)
    except MediaUploadError as error:
        raise media_upload_error(error) from error


@app.put("/api/library/uploads/{upload_id}/chunk")
async def upload_library_chunk(upload_id: str, request: Request) -> dict:
    try:
        start, end, total = parse_content_range(request.headers.get("content-range"))
        return await media_uploads().append_chunk(upload_id, start, end, total, request.stream())
    except (MediaUploadError, MediaLibraryError) as error:
        raise media_upload_error(error) from error


@app.post("/api/library/uploads/{upload_id}/complete")
def complete_library_upload(upload_id: str) -> dict:
    try:
        return {"asset": media_uploads().complete(upload_id)}
    except (MediaUploadError, MediaLibraryError) as error:
        raise media_upload_error(error) from error


@app.delete("/api/library/uploads/{upload_id}")
def abort_library_upload(upload_id: str) -> dict:
    try:
        media_uploads().abort(upload_id)
        return {"status": "aborted"}
    except MediaUploadError as error:
        raise media_upload_error(error) from error


@app.api_route("/api/library/assets/{asset_id}/video", methods=["GET", "HEAD"])
def serve_library_asset_video(asset_id: str) -> FileResponse:
    try:
        return FileResponse(media_library().resolve_asset_path(asset_id), media_type="video/mp4")
    except MediaLibraryError as error:
        raise media_library_error(error) from error
