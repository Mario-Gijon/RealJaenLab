import json
import subprocess
from pathlib import Path

from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles


BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"
INPUT_DIR = BASE_DIR.parent / "input"
REVIEW_DIR = BASE_DIR / "data" / "reviews"

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


@app.get("/api/videos")
def list_videos() -> dict[str, list[dict[str, str | float | int | None]]]:
    videos = available_videos()
    return {
        "videos": [
            {"name": name, **read_video_metadata(path)}
            for name, path in sorted(videos.items())
        ]
    }


@app.get("/videos/{filename}")
def serve_video(filename: str) -> FileResponse:
    video_path = available_videos().get(filename)
    if video_path is None:
        raise HTTPException(status_code=404, detail="Video not found")
    return FileResponse(video_path, media_type="video/mp4")


def review_path_for(filename: str) -> tuple[Path, Path]:
    video_path = available_videos().get(filename)
    if video_path is None:
        raise HTTPException(status_code=404, detail="Video not found")
    safe_stem = "".join(character if character.isalnum() or character in "-_" else "_" for character in video_path.stem)
    return REVIEW_DIR / f"{safe_stem or 'review'}.json", video_path


def video_identity(video_path: Path) -> dict[str, str | int | float | None]:
    return {
        "filename": video_path.name,
        "size_bytes": video_path.stat().st_size,
        **read_video_metadata(video_path),
    }


@app.get("/api/reviews/{filename}")
def load_review(filename: str) -> dict[str, dict | None]:
    review_path, _ = review_path_for(filename)
    if not review_path.exists():
        return {"review": None}
    try:
        return {"review": json.loads(review_path.read_text(encoding="utf-8"))}
    except json.JSONDecodeError as error:
        raise HTTPException(status_code=500, detail="Saved review is not valid JSON") from error


@app.put("/api/reviews/{filename}")
def save_review(filename: str, review: dict = Body()) -> dict[str, str]:
    review_path, video_path = review_path_for(filename)
    saved_review = {
        "format_version": 1,
        "video": video_identity(video_path),
        "field": review.get("field", {"annotations": []}),
        "people": {},
    }
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    review_path.write_text(json.dumps(saved_review, indent=2) + "\n", encoding="utf-8")
    return {"status": "saved"}
