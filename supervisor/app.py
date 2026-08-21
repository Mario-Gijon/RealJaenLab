import json
import subprocess
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles


BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"
INPUT_DIR = BASE_DIR.parent / "input"

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
        "stream=avg_frame_rate,nb_frames,duration",
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
        }
    except (FileNotFoundError, subprocess.CalledProcessError, ValueError, json.JSONDecodeError):
        return {"fps": None, "duration": None, "frame_count": None}


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
