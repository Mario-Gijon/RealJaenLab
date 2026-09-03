"""Background, cached integration for the validated view-diversity diagnostic."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from datetime import UTC, datetime
import json
from pathlib import Path
import subprocess
import sys
from threading import Lock
import time
from typing import Any

from supervisor.check_video_view_diversity_v1 import VERSION
from supervisor.video_sources import VideoSource


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _identity(source: VideoSource) -> dict[str, Any]:
    stat = source.path.stat()
    return {"reference": source.reference, "path": str(source.path), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _cache_path(review_dir: Path, source: VideoSource) -> Path:
    return Path(review_dir) / source.result_basename("_view_diversity_v1.json")


def analyze_video_view_diversity(video_path: Path, output_dir: Path) -> dict[str, Any]:
    """Run the single validated diagnostic implementation and return its report."""
    command = [sys.executable, "-m", "supervisor.check_video_view_diversity_v1", "--video", str(video_path), "--name", video_path.stem, "--output-dir", str(output_dir)]
    subprocess.run(command, check=True, capture_output=True, text=True)
    return json.loads((Path(output_dir) / "report.json").read_text(encoding="utf-8"))


class ViewDiversityManager:
    def __init__(self, review_dir: Path):
        self.review_dir = Path(review_dir)
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="supervisor-view-diversity")
        self.jobs: dict[str, Future] = {}
        self.lock = Lock()

    def _read(self, source: VideoSource) -> dict[str, Any] | None:
        path = _cache_path(self.review_dir, source)
        if not path.exists(): return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError): return None
        return value if isinstance(value, dict) else None

    def get(self, source: VideoSource) -> dict[str, Any]:
        cached = self._read(source)
        identity = _identity(source)
        if cached and cached.get("algorithm") == VERSION and cached.get("video_identity") == identity and cached.get("status") == "error":
            return cached
        if cached and cached.get("algorithm") == VERSION and cached.get("video_identity") == identity:
            state = dict(cached); state["status"] = "complete"; state["cache_hit"] = True
            return state
        with self.lock:
            future = self.jobs.get(source.reference)
        if future and not future.done():
            return {"status": "running", "cache_hit": False, "started_at": None}
        if cached:
            return {"status": "stale", "cache_hit": False, "error": "El vídeo ha cambiado; hay que volver a analizarlo."}
        return {"status": "ready", "cache_hit": False}

    def start(self, source: VideoSource) -> dict[str, Any]:
        current = self.get(source)
        if current.get("status") == "complete": return current
        with self.lock:
            future = self.jobs.get(source.reference)
            if future and not future.done(): return current
            started_at = _now()
            state = {"status": "running", "cache_hit": False, "started_at": started_at}
            self.jobs[source.reference] = self.executor.submit(self._run, source, started_at)
        return state

    def _run(self, source: VideoSource, started_at: str) -> None:
        started = time.perf_counter()
        output_dir = self.review_dir / f"{source.result_key}_view_diversity_v1"
        try:
            report = analyze_video_view_diversity(source.path, output_dir)
            recommendations = []
            for item in report.get("selection", {}).get("recommended", []):
                recommendations.append({
                    "frame": int(item["frame"]), "timestamp": item["time"],
                    "pan_label": item["pan"]["label"], "pan_value": item["pan"]["normalized"],
                    "zoom_label": item["zoom"]["label"], "zoom_value": item["zoom"]["normalized"],
                    "rank": int(item["recommendation_rank"]),
                })
            cache = {
                "status": "complete", "algorithm": report.get("version", VERSION),
                "video_identity": _identity(source), "recommendations": recommendations,
                "recommended_count": len(recommendations),
                "sampled_frames": report.get("sampling", {}).get("decoded"),
                "processing_duration_seconds": report.get("timings_seconds", {}).get("total_s", round(time.perf_counter() - started, 4)),
                "started_at": started_at, "completed_at": _now(),
            }
            path = _cache_path(self.review_dir, source); path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(cache, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        except Exception as error:
            path = _cache_path(self.review_dir, source); path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"status": "error", "algorithm": VERSION, "video_identity": _identity(source), "error": str(error), "processing_duration_seconds": round(time.perf_counter() - started, 4), "started_at": started_at, "completed_at": _now()}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
