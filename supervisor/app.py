import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from supervisor.field_calibration_strategy import solve_field_camera_adaptive
from supervisor.field_view_scanner import scan_field_view_states, summarize_field_view_scan
from supervisor.field_pitch_setup import (
    estimate_pitch_setup,
    next_pitch_measurement_frame,
    select_pitch_measurement_frames,
    summarize_pitch_setup,
)
from supervisor.field_camera_atlas import build_camera_atlas, summarize_camera_atlas
from supervisor.field_camera_atlas_structure import (
    analyze_camera_atlas_structure,
    summarize_camera_atlas_structure,
)
from supervisor.field_camera_atlas_coverage import (
    analyze_camera_atlas_coverage,
    summarize_camera_atlas_coverage,
)
from supervisor.field_anchor_selector import (
    select_manual_anchor_frames,
    summarize_anchor_selection,
)
from supervisor.field_camera_state_pipeline import recover_match_setup_cameras
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
from supervisor.video_sources import (
    VideoSource,
    VideoSourceError,
    VideoSourceNotFoundError,
    VideoSourceUnsupportedError,
    resolve_video_source,
)
from supervisor.field_auto_refinement import propose_field_auto_refinement
from supervisor.field_online_adaptation import (
    build_field_online_profile,
    suggest_local_field_searches,
    summarize_field_learning,
)
from supervisor.field_local_observer import observe_field_lines
from supervisor.field_propagation import propagate_field_homography
from supervisor.field_segment_refinement import refine_field_segment
from supervisor.field_spatial_gating import apply_field_spatial_gating
from supervisor.field_workflow import FieldWorkflowError, FieldWorkflowManager


BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"
INPUT_DIR = BASE_DIR.parent / "input"
REVIEW_DIR = BASE_DIR / "data" / "reviews"
LIBRARY_DIR = BASE_DIR / "data" / "library"
_media_upload_manager: MediaUploadManager | None = None
_field_workflow_manager: FieldWorkflowManager | None = None

app = FastAPI(title="RealJaen Supervisor")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


def read_video_metadata(video_path: Path) -> dict[str, float | int | None]:
    """Read only the playback metadata needed by the browser controls."""
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=avg_frame_rate,nb_frames,duration,width,height",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(video_path),
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


def available_videos() -> dict[str, Path]:
    if not INPUT_DIR.exists():
        return {}
    return {path.name: path for path in INPUT_DIR.iterdir() if path.is_file() and path.suffix.lower() == ".mp4"}


def media_library() -> MediaLibrary:
    return MediaLibrary(
        catalog_path=LIBRARY_DIR / "catalog_v1.json",
        project_root=BASE_DIR.parent,
        input_dir=INPUT_DIR,
        metadata_reader=read_video_metadata,
        managed_videos_dir=LIBRARY_DIR / "videos",
    )


def media_library_error(error: MediaLibraryError) -> HTTPException:
    status = 404 if isinstance(error, MediaLibraryNotFoundError) else 400
    return HTTPException(status_code=status, detail=str(error))


def media_uploads() -> MediaUploadManager:
    global _media_upload_manager
    if _media_upload_manager is None:
        _media_upload_manager = MediaUploadManager(LIBRARY_DIR, media_library())
    return _media_upload_manager


def field_workflows() -> FieldWorkflowManager:
    global _field_workflow_manager
    if _field_workflow_manager is None:
        _field_workflow_manager = FieldWorkflowManager(REVIEW_DIR, read_video_metadata)
    return _field_workflow_manager


def media_upload_error(error: MediaUploadError | MediaLibraryError) -> HTTPException:
    if isinstance(error, (MediaUploadNotFoundError, MediaLibraryNotFoundError)):
        status = 404
    elif isinstance(error, (MediaUploadConflictError, MediaLibraryConflictError)):
        status = 409
    else:
        status = 400
    return HTTPException(status_code=status, detail=str(error))


def field_workflow_error(error: FieldWorkflowError) -> HTTPException:
    return HTTPException(status_code=400, detail=str(error))


def workspace_video_source(reference: str) -> VideoSource:
    """Resolve legacy ``filename`` API values or managed asset ids safely."""

    try:
        return resolve_video_source(reference, library=media_library(), input_dir=INPUT_DIR)
    except VideoSourceNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except (VideoSourceUnsupportedError, VideoSourceError, MediaLibraryError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.get("/api/videos")
def list_videos() -> dict[str, list[dict[str, str | float | int | None]]]:
    videos = available_videos()
    return {
        "videos": [
            {"name": name, **read_video_metadata(path)}
            for name, path in sorted(videos.items())
        ]
    }


@app.get("/api/library")
def get_media_library() -> dict:
    try:
        return media_library().list()
    except MediaLibraryError as error:
        raise media_library_error(error) from error


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
        return {"asset": media_library().register_legacy_input(payload)}
    except MediaLibraryError as error:
        raise media_library_error(error) from error


@app.patch("/api/library/assets/{asset_id}")
def update_library_asset(asset_id: str, payload: dict = Body()) -> dict:
    try:
        return {"asset": media_library().update_asset(asset_id, payload)}
    except MediaLibraryError as error:
        raise media_library_error(error) from error


@app.post("/api/library/assets/{asset_id}/segments")
def create_library_segment(asset_id: str, payload: dict = Body()) -> dict:
    try:
        return {"asset": media_library().create_segment(asset_id, payload)}
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
        video_path = media_library().resolve_asset_path(asset_id)
    except MediaLibraryError as error:
        raise media_library_error(error) from error
    return FileResponse(video_path, media_type="video/mp4")


@app.get("/videos/{filename}")
def serve_video(filename: str) -> FileResponse:
    video_path = available_videos().get(filename)
    if video_path is None:
        raise HTTPException(status_code=404, detail="Video not found")
    return FileResponse(video_path, media_type="video/mp4")


def review_path_for(filename: str) -> tuple[Path, Path]:
    source = workspace_video_source(filename)
    return source.review_path(REVIEW_DIR), source.path


def video_identity(video_path: Path, source: VideoSource | None = None) -> dict[str, str | int | float | None]:
    identity: dict[str, str | int | float | None] = {
        "filename": video_path.name,
        "size_bytes": video_path.stat().st_size,
        **read_video_metadata(video_path),
    }
    if source is not None and source.source_type == "managed_original":
        identity.update({
            "filename": source.reference,
            "source_type": source.source_type,
            "asset_id": source.asset_id,
            "original_filename": source.original_filename,
            "physical_filename": video_path.name,
        })
    return identity


@app.get("/api/reviews/{filename}")
def load_review(filename: str) -> dict[str, dict | None]:
    review_path, _ = review_path_for(filename)
    if not review_path.exists():
        return {"review": None}
    try:
        return {"review": json.loads(review_path.read_text(encoding="utf-8"))}
    except json.JSONDecodeError as error:
        raise HTTPException(status_code=500, detail="Saved review is not valid JSON") from error


@app.get("/api/field-workflow/{video_reference}")
def get_field_workflow(video_reference: str) -> dict:
    source = workspace_video_source(video_reference)
    return field_workflows().get(source)


@app.post("/api/field-workflow/{video_reference}/camera/run")
def run_field_workflow_camera(video_reference: str) -> dict:
    source = workspace_video_source(video_reference)
    try:
        return field_workflows().start(source)
    except FieldWorkflowError as error:
        raise field_workflow_error(error) from error


@app.get("/api/field-camera-atlas/dense-preview/{filename}")
def load_dense_camera_atlas_preview(filename: str) -> dict:
    """Serve an existing Dense Camera Atlas only for safe preview use."""

    source = workspace_video_source(filename)
    per_video_path = REVIEW_DIR / source.result_basename("_camera_atlas_dense_v1.json")
    if per_video_path.exists():
        atlas_path = per_video_path
    elif source.source_type == "legacy_input" and source.legacy_filename == "benchmark_easy_2min.mp4":
        atlas_path = REVIEW_DIR / "camera_atlas_dense_v1.json"
    else:
        raise HTTPException(status_code=404, detail="Dense Camera Atlas preview not found")
    if not atlas_path.exists():
        raise HTTPException(status_code=404, detail="Dense Camera Atlas preview not found")
    try:
        atlas = json.loads(atlas_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise HTTPException(status_code=500, detail="Dense Camera Atlas preview is not valid JSON") from error
    if not isinstance(atlas, dict) or not isinstance(atlas.get("records"), list):
        raise HTTPException(status_code=500, detail="Dense Camera Atlas preview has no valid records")
    return atlas


@app.put("/api/reviews/{filename}")
def save_review(filename: str, review: dict = Body()) -> dict[str, str]:
    source = workspace_video_source(filename)
    review_path = source.review_path(REVIEW_DIR)
    video_path = source.path
    saved_review = {
        "format_version": 1,
        "video": video_identity(video_path, source),
        "field": review.get("field", {"annotations": []}),
        "people": {},
    }
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    review_path.write_text(json.dumps(saved_review, indent=2) + "\n", encoding="utf-8")
    return {"status": "saved"}


def points_are_non_collinear(points: np.ndarray) -> bool:
    if len(points) < 4:
        return False
    centered = points - points.mean(axis=0)
    singular_values = np.linalg.svd(centered, compute_uv=False)
    return len(singular_values) == 2 and singular_values[1] > singular_values[0] * 1e-4


@app.post("/api/homography")
def calculate_homography(payload: dict = Body()) -> dict[str, list | float]:
    correspondences = payload.get("correspondences", [])
    if len(correspondences) < 4:
        raise HTTPException(status_code=400, detail="Se requieren al menos cuatro correspondencias")

    try:
        world_points = np.array([[item["world"]["x"], item["world"]["y"]] for item in correspondences], dtype=np.float64)
        image_points = np.array([[item["image"]["x"], item["image"]["y"]] for item in correspondences], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as error:
        raise HTTPException(status_code=400, detail="Coordenadas de correspondencia no válidas") from error

    if not points_are_non_collinear(world_points) or not points_are_non_collinear(image_points):
        raise HTTPException(status_code=400, detail="Las correspondencias son geométricamente degeneradas")

    homography, _ = cv2.findHomography(world_points, image_points, method=0)
    if homography is None or not np.isfinite(homography).all() or abs(homography[2, 2]) < 1e-12:
        raise HTTPException(status_code=400, detail="No se pudo calcular una homografía válida")
    homography /= homography[2, 2]

    homogeneous_world = np.column_stack([world_points, np.ones(len(world_points))])
    projected = homogeneous_world @ homography.T
    projected = projected[:, :2] / projected[:, 2:3]
    reprojection_error = float(np.mean(np.linalg.norm(projected - image_points, axis=1)))
    return {"homography": homography.tolist(), "reprojection_error": reprojection_error}


@app.post("/api/field-calibration/solve")
def solve_calibration(payload: dict = Body()) -> dict:
    constraints = payload.get("constraints", [])
    grouped = {"point": [], "line": [], "arc": []}
    for constraint in constraints:
        constraint_type = constraint.get("type")
        if constraint_type in grouped:
            grouped[constraint_type].append(constraint)

    return solve_field_camera_adaptive(
        grouped["point"],
        grouped["line"],
        grouped["arc"],
        payload.get("image_width"),
        payload.get("image_height"),
        initial_homography=payload.get("initial_homography"),
        mode="auto",
    )


@app.post("/api/field-view-scan/analyze")
def analyze_field_view_scan(payload: dict = Body()) -> dict:
    filename = payload.get("filename")
    video_path = workspace_video_source(filename).path

    try:
        result = scan_field_view_states(video_path)
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"El escáner de vistas FIELD no pudo analizar el vídeo: {error}",
        ) from error
    return {
        "summary": summarize_field_view_scan(result),
        "scan": result,
    }


@app.post("/api/field-pitch-setup/select")
def select_field_pitch_measurements(payload: dict = Body()) -> dict:
    scan = payload.get("scan")
    if not isinstance(scan, dict):
        raise HTTPException(status_code=400, detail="El análisis de vistas del terreno no es válido")
    return select_pitch_measurement_frames(scan)


@app.post("/api/field-pitch-setup/estimate")
def estimate_field_pitch_setup(payload: dict = Body()) -> dict:
    observations = payload.get("observations")
    if not isinstance(observations, list):
        raise HTTPException(status_code=400, detail="Las mediciones HUMAN del terreno no son válidas")
    result = estimate_pitch_setup(observations)
    return {"result": result, "summary": summarize_pitch_setup(result)}


@app.post("/api/field-pitch-setup/next-frame")
def next_field_pitch_measurement(payload: dict = Body()) -> dict:
    scan = payload.get("scan")
    used_frames = payload.get("used_frames", [])
    if not isinstance(scan, dict) or not isinstance(used_frames, list):
        raise HTTPException(status_code=400, detail="La siguiente medición del terreno no es válida")
    return {"recommendation": next_pitch_measurement_frame(scan, used_frames)}


@app.post("/api/field-camera-state/recover-match-setup")
def recover_field_camera_state_match_setup(payload: dict = Body()) -> dict:
    calibrations = payload.get("calibrations")
    match_setup = payload.get("match_setup")
    pitch_model = payload.get("pitch_model")
    video_name = payload.get("video_name")
    if not isinstance(calibrations, list) or not isinstance(match_setup, dict) or not isinstance(pitch_model, dict):
        raise HTTPException(status_code=400, detail="La recuperación de Camera State no es válida")
    if not isinstance(video_name, str) or not video_name:
        raise HTTPException(status_code=400, detail="El vídeo de Camera State no es válido")
    video_source = workspace_video_source(video_name)
    video_path = video_source.path
    try:
        image_width = int(payload["image_width"])
        image_height = int(payload["image_height"])
    except (KeyError, TypeError, ValueError) as error:
        raise HTTPException(status_code=400, detail="La resolución de Camera State no es válida") from error
    return recover_match_setup_cameras(
        calibrations,
        match_setup,
        pitch_model,
        image_width,
        image_height,
        video_path=video_path,
        video_name=video_source.reference,
    )


@app.post("/api/field-camera-atlas/analyze")
def analyze_field_camera_atlas(payload: dict = Body()) -> dict:
    calibrations = payload.get("calibrations", [])
    propagations = payload.get("propagations", [])
    refined_segments = payload.get("refined_segments", [])
    video_name = payload.get("video_name")
    try:
        image_width = int(payload["image_width"])
        image_height = int(payload["image_height"])
    except (KeyError, TypeError, ValueError) as error:
        raise HTTPException(status_code=400, detail="Solicitud de Camera Atlas no válida") from error

    if not isinstance(calibrations, list):
        raise HTTPException(status_code=400, detail="Las calibraciones de Camera Atlas no son válidas")
    if not isinstance(propagations, list) or not isinstance(refined_segments, list):
        raise HTTPException(status_code=400, detail="Los probes de Camera Atlas no son válidos")
    if not isinstance(video_name, str) or not video_name:
        raise HTTPException(status_code=400, detail="El vídeo de Camera Atlas no es válido")
    video_source = workspace_video_source(video_name)

    calibrations = [
        calibration
        for calibration in calibrations
        if isinstance(calibration, dict)
        and calibration.get("purpose") != "pitch_setup"
    ]

    atlas = build_camera_atlas(
        calibrations,
        image_width,
        image_height,
        video_name=video_source.reference,
    )
    structure = analyze_camera_atlas_structure(atlas, image_width, image_height)
    coverage = analyze_camera_atlas_coverage(
        atlas,
        propagations,
        refined_segments,
        image_width,
        image_height,
        video_name=video_source.reference,
    )
    return {
        "summary": summarize_camera_atlas(atlas),
        "atlas": atlas,
        "structure_summary": summarize_camera_atlas_structure(structure),
        "structure": structure,
        "coverage_summary": summarize_camera_atlas_coverage(coverage),
        "coverage": coverage,
    }


@app.post("/api/field-anchor-selector/analyze")
def analyze_field_anchor_selector(payload: dict = Body()) -> dict:
    filename = payload.get("filename")
    video_source = workspace_video_source(filename)
    video_path = video_source.path

    calibrations = payload.get("calibrations", [])
    propagations = payload.get("propagations", [])
    refined_segments = payload.get("refined_segments", [])
    try:
        image_width = int(payload["image_width"])
        image_height = int(payload["image_height"])
    except (KeyError, TypeError, ValueError) as error:
        raise HTTPException(status_code=400, detail="Solicitud del selector de anchors no válida") from error

    if not all(isinstance(items, list) for items in (calibrations, propagations, refined_segments)):
        raise HTTPException(status_code=400, detail="Los datos del selector de anchors no son válidos")

    calibrations = [
        calibration
        for calibration in calibrations
        if isinstance(calibration, dict)
        and calibration.get("purpose") != "pitch_setup"
    ]

    atlas = build_camera_atlas(
        calibrations,
        image_width,
        image_height,
        video_name=video_source.reference,
    )
    coverage = analyze_camera_atlas_coverage(
        atlas,
        propagations,
        refined_segments,
        image_width,
        image_height,
        video_name=video_source.reference,
    )
    selection = select_manual_anchor_frames(
        video_path,
        atlas,
        coverage,
        propagations,
        refined_segments,
        image_width,
        image_height,
        video_name=video_source.reference,
    )
    return {
        "selection": selection,
        "selection_summary": summarize_anchor_selection(selection),
    }


@app.post("/api/field-adaptation/guidance")
def field_adaptation_guidance(payload: dict = Body()) -> dict:
    filename = payload.get("filename")
    review_path, _ = review_path_for(filename)
    try:
        image_width = int(payload["image_width"])
        image_height = int(payload["image_height"])
        current_homography = payload["homography"]
    except (KeyError, TypeError, ValueError) as error:
        raise HTTPException(status_code=400, detail="Solicitud de guía FIELD no válida") from error

    if review_path.exists():
        try:
            review = json.loads(review_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise HTTPException(status_code=500, detail="Saved review is not valid JSON") from error
    else:
        review = {}

    calibrations = review.get("field", {}).get("calibrations", [])
    if not isinstance(calibrations, list):
        calibrations = []
    profile = build_field_online_profile(calibrations, image_width, image_height)
    guidance = suggest_local_field_searches(
        profile,
        current_homography,
        image_width,
        image_height,
        only_seen_semantics=True,
    )
    guidance = apply_field_spatial_gating(
        guidance,
        current_homography,
        image_width,
        image_height,
    )
    corrected_view_frames = sorted({
        int(calibration.get("frame", 0))
        for calibration in calibrations
        if calibration.get("initialHomography") or calibration.get("initial_homography")
    })
    return {
        **guidance,
        "learning": summarize_field_learning(profile),
        "corrected_view_frames": corrected_view_frames,
    }


@app.post("/api/field-observer/lines")
def observe_field_lines_for_frame(payload: dict = Body()) -> dict:
    filename = payload.get("filename")
    review_path, video_path = review_path_for(filename)
    try:
        frame_index = int(payload["frame"])
        image_width = int(payload["image_width"])
        image_height = int(payload["image_height"])
        current_homography = payload["homography"]
    except (KeyError, TypeError, ValueError) as error:
        raise HTTPException(status_code=400, detail="Solicitud del observador FIELD no válida") from error

    if frame_index < 0:
        raise HTTPException(status_code=400, detail="El fotograma solicitado no es válido")
    if review_path.exists():
        try:
            review = json.loads(review_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise HTTPException(status_code=500, detail="Saved review is not valid JSON") from error
    else:
        review = {}

    calibrations = review.get("field", {}).get("calibrations", [])
    if not isinstance(calibrations, list):
        calibrations = []
    profile = build_field_online_profile(calibrations, image_width, image_height)
    guidance = suggest_local_field_searches(
        profile,
        current_homography,
        image_width,
        image_height,
        only_seen_semantics=True,
    )
    guidance = apply_field_spatial_gating(
        guidance,
        current_homography,
        image_width,
        image_height,
    )

    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise HTTPException(status_code=500, detail="OpenCV no pudo abrir el vídeo")
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if frame_count and frame_index >= frame_count:
            raise HTTPException(status_code=400, detail="El fotograma solicitado está fuera del vídeo")
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame_bgr = capture.read()
        if not ok or frame_bgr is None:
            raise HTTPException(status_code=400, detail="No se pudo leer el fotograma solicitado")
    finally:
        capture.release()

    corrected_view_frames = sorted({
        int(calibration.get("frame", 0))
        for calibration in calibrations
        if calibration.get("initialHomography") or calibration.get("initial_homography")
    })
    observer_result = observe_field_lines(frame_bgr, guidance)
    auto_refinement_preview = propose_field_auto_refinement(
        current_homography,
        observer_result,
        image_width,
        image_height,
    )
    return {
        "guidance": guidance,
        "observer": observer_result,
        "auto_refinement_preview": auto_refinement_preview,
        "learning": summarize_field_learning(profile),
        "corrected_view_frames": corrected_view_frames,
    }


@app.post("/api/field-propagation")
def propagate_calibration(payload: dict = Body()) -> dict:
    filename = payload.get("filename")
    video_path = workspace_video_source(filename).path

    try:
        anchor_frame = int(payload["anchor_frame"])
        max_frames = int(payload["max_frames"])
        direction = int(payload.get("direction", 1))
        homography = payload["homography"]
    except (KeyError, TypeError, ValueError) as error:
        raise HTTPException(status_code=400, detail="Solicitud de propagación no válida") from error

    return propagate_field_homography(
        video_path,
        anchor_frame,
        homography,
        max_frames=max_frames,
        direction=direction,
    )


@app.post("/api/field-segment/refine")
def refine_calibration_segment(payload: dict = Body()) -> dict:
    filename = payload.get("filename")
    video_path = workspace_video_source(filename).path

    try:
        left_anchor_frame = int(payload["left_anchor_frame"])
        right_anchor_frame = int(payload["right_anchor_frame"])
        left_homography = payload["left_homography"]
        right_homography = payload["right_homography"]
    except (KeyError, TypeError, ValueError) as error:
        raise HTTPException(status_code=400, detail="Solicitud de refinamiento no válida") from error

    return refine_field_segment(
        video_path,
        left_anchor_frame,
        left_homography,
        right_anchor_frame,
        right_homography,
    )
