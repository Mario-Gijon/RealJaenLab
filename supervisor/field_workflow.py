"""Persistent, single-process orchestration for the Supervisor V2 CAMERA flow."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from threading import Lock
from typing import Any, Callable

import numpy as np

from supervisor.field_camera_active_anchor import ActiveAnchorConfig, select_next_active_anchor
from supervisor.field_camera_anchor_relocalization import AnchorRelocalizationConfig, analyse_anchor_relocalization
from supervisor.field_camera_atlas_dense import DenseCameraAtlasConfig, build_dense_camera_atlas
from supervisor.field_camera_motion_global import GlobalCameraMotionConfig
from supervisor.field_camera_trajectory import CameraTrajectoryConfig, analyse_camera_trajectory


class FieldWorkflowError(ValueError):
    pass


def _now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _valid_homography(value: Any) -> bool:
    return isinstance(value, list) and len(value) == 3 and all(isinstance(row, list) and len(row) == 3 for row in value)


def _json_safe(value: Any) -> Any:
    """Convert CV diagnostics to JSON-safe values only at persistence edges."""
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def extract_human_anchors(review: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract only exact manual match-setup and active-anchor calibrations."""
    field = review.get("field") if isinstance(review, dict) else None
    if not isinstance(field, dict):
        return []
    completed = {int(frame) for frame in field.get("match_setup", {}).get("completed_frames", []) if isinstance(frame, (int, float))}
    by_frame: dict[int, dict[str, Any]] = {}
    for item in field.get("calibrations", []):
        if not isinstance(item, dict) or item.get("source") != "manual" or not _valid_homography(item.get("homography")):
            continue
        try:
            frame = int(item["frame"])
        except (KeyError, TypeError, ValueError):
            continue
        purpose = item.get("purpose")
        if purpose == "match_setup" and frame not in completed:
            continue
        if purpose not in {"match_setup", "active_anchor"}:
            continue
        # A later active anchor is the more explicit HUMAN correction.
        if frame not in by_frame or purpose == "active_anchor":
            by_frame[frame] = {"frame": frame, "homography": item["homography"], "purpose": purpose}
    return [by_frame[frame] for frame in sorted(by_frame)]


@dataclass
class WorkflowDependencies:
    trajectory: Callable = analyse_camera_trajectory
    relocalization: Callable = analyse_anchor_relocalization
    dense: Callable = build_dense_camera_atlas
    active_anchor: Callable = select_next_active_anchor


class FieldWorkflowManager:
    """One queued heavy CAMERA job at a time, keyed by logical video source."""

    def __init__(self, review_dir: Path, metadata_reader: Callable, dependencies: WorkflowDependencies | None = None):
        self.review_dir = Path(review_dir)
        self.metadata_reader = metadata_reader
        self.dependencies = dependencies or WorkflowDependencies()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="supervisor-camera")
        self.jobs: dict[str, Future] = {}
        self.lock = Lock()

    def _path(self, source, suffix: str) -> Path:
        return self.review_dir / source.result_basename(suffix)

    @staticmethod
    def _write(path: Path, value: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as temporary:
            json.dump(_json_safe(value), temporary, indent=2)
            temporary.write("\n"); temporary.flush(); os.fsync(temporary.fileno())
            name = temporary.name
        os.replace(name, path)

    def _load_review(self, source) -> dict[str, Any]:
        path = source.review_path(self.review_dir)
        if not path.exists():
            raise FieldWorkflowError("No existe una revisión guardada para este vídeo")
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise FieldWorkflowError("La revisión guardada no es válida") from error

    def _signature(self, source, review: dict[str, Any], metadata: dict[str, Any]) -> str:
        field = review.get("field", {})
        payload = {"source": source.reference, "size": source.path.stat().st_size, "frames": metadata.get("frame_count"), "pitch": field.get("pitch_model"), "anchors": extract_human_anchors(review)}
        return sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def _prepare(self, source) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], str]:
        review = self._load_review(source)
        field = review.get("field") if isinstance(review, dict) else None
        pitch = field.get("pitch_model") if isinstance(field, dict) else None
        if not isinstance(pitch, dict) or pitch.get("status") != "confirmed":
            raise FieldWorkflowError("Confirma el terreno antes de completar la cámara")
        try:
            length, width = float(pitch["length_m"]), float(pitch["width_m"])
        except (KeyError, TypeError, ValueError) as error:
            raise FieldWorkflowError("Las dimensiones del terreno no son válidas") from error
        if length <= 0 or width <= 0:
            raise FieldWorkflowError("Las dimensiones del terreno no son válidas")
        anchors = extract_human_anchors(review)
        if not anchors:
            raise FieldWorkflowError("Añade al menos una referencia HUMAN válida")
        metadata = self.metadata_reader(source.path)
        if not metadata.get("frame_count") or not metadata.get("width") or not metadata.get("height"):
            raise FieldWorkflowError("No se pudo leer la información técnica del vídeo")
        return review, metadata, anchors, self._signature(source, review, metadata)

    def _state_path(self, source) -> Path:
        return self._path(source, "_field_workflow_v1.json")

    def _read_state(self, source) -> dict[str, Any] | None:
        path = self._state_path(source)
        if not path.exists(): return None
        try: return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError): return None

    def _save_state(self, source, state: dict[str, Any]) -> dict[str, Any]:
        state["updated_at"] = _now(); self._write(self._state_path(source), state); return state

    def get(self, source) -> dict[str, Any]:
        state = self._read_state(source)
        if state is None:
            return {"version": 1, "source_reference": source.reference, "source_key": source.result_key, "camera": {"status": "ready", "phase": "ready", "progress_fraction": 0.0}}
        with self.lock:
            active = self.jobs.get(source.reference)
        camera = state.setdefault("camera", {})
        if camera.get("status") in {"queued", "running"} and (active is None or active.done()):
            camera.update({"status": "interrupted", "phase": "interrupted", "human_message": "El análisis se interrumpió; puedes reintentarlo."})
            self._save_state(source, state)
        if camera.get("status") == "complete":
            try:
                _, metadata, _, signature = self._prepare(source)
                if signature != state.get("input_signature"):
                    camera.update({"status": "ready", "phase": "stale", "human_message": "Las referencias o el terreno cambiaron; vuelve a completar la cámara.", "progress_fraction": 0.0})
                    state["input_signature"] = signature; state["metadata"] = metadata; self._save_state(source, state)
            except FieldWorkflowError:
                pass
        return state

    def start(self, source) -> dict[str, Any]:
        review, metadata, anchors, signature = self._prepare(source)
        with self.lock:
            future = self.jobs.get(source.reference)
        if future is not None and not future.done():
            # Do not call get() while holding self.lock: it also reads jobs.
            return self._read_state(source) or {"version": 1, "source_reference": source.reference, "source_key": source.result_key, "camera": {"status": "queued", "phase": "queued", "progress_fraction": 0.0}}
        with self.lock:
            # A request may have raced between the first lookup and this lock.
            future = self.jobs.get(source.reference)
            if future is not None and not future.done():
                return self._read_state(source) or {"version": 1, "source_reference": source.reference, "source_key": source.result_key, "camera": {"status": "queued", "phase": "queued", "progress_fraction": 0.0}}
            state = {"version": 1, "source_reference": source.reference, "source_key": source.result_key, "input_signature": signature, "metadata": metadata, "camera": {"status": "queued", "phase": "queued", "progress_fraction": 0.0, "human_message": "Preparando análisis de cámara…", "human_anchor_count": len(anchors), "total_frames": int(metadata["frame_count"]), "trajectory_keyframe_count": 0, "relocalized_keyframe_count": 0, "direct_unresolved_keyframe_count": 0, "dense_frame_count": 0, "dense_missing_frame_count": 0, "dense_coverage_fraction": 0.0, "recommended_human_frame": None}}
            self._save_state(source, state)
            self.jobs[source.reference] = self.executor.submit(self._run, source, state)
            return state

    def _run(self, source, state: dict[str, Any]) -> None:
        camera = state["camera"]
        persisted_progress = -1.0
        def progress(phase, fraction, message):
            nonlocal persisted_progress
            fraction = max(float(camera.get("progress_fraction") or 0.0), min(0.99, float(fraction)))
            changed_phase = camera.get("phase") != phase
            camera.update({"status": "running", "phase": phase, "progress_fraction": fraction, "human_message": message})
            if changed_phase or fraction - persisted_progress >= 0.005:
                persisted_progress = fraction
                self._save_state(source, state)
        def phase_failure(phase: str, result: Any) -> None:
            detail = result.get("status") if isinstance(result, dict) else "resultado inválido"
            camera.update({"status": "needs_attention", "phase": phase, "human_message": "No se pudo completar automáticamente la cámara.", "error": f"{phase}: {detail}"})
            self._save_state(source, state)
        try:
            review, metadata, anchors, signature = self._prepare(source)
            total, width, height = int(metadata["frame_count"]), int(metadata["width"]), int(metadata["height"])
            pitch = review["field"]["pitch_model"]
            progress("trajectory", 0.0, "Analizando movimiento de cámara…")
            def trajectory_progress(done, count, *_args):
                progress("trajectory", .35 * (float(done) / max(float(count), 1.0)), "Analizando movimiento de cámara…")
            trajectory = self.dependencies.trajectory(source.path, start_frame=0, end_frame=total - 1, motion_config=GlobalCameraMotionConfig(analysis_width=960, sample_step_frames=2), trajectory_config=CameraTrajectoryConfig(sample_step_frames=2), progress_callback=trajectory_progress)
            self._write(self._path(source, "_camera_trajectory_v1.json"), trajectory)
            if not isinstance(trajectory, dict) or trajectory.get("status") != "ok":
                phase_failure("trajectory", trajectory); return
            keyframes = trajectory.get("keyframes", []) if isinstance(trajectory, dict) else []
            targets = [{"frame": int(item["frame"]), "keyframe_index": index} for index, item in enumerate(keyframes) if isinstance(item, dict) and "frame" in item]
            camera["trajectory_keyframe_count"] = len(targets)
            progress("relocalization", .35, "Buscando vistas conocidas…")
            def relocalization_progress(phase, done, count, *_args):
                if phase == "targets": progress("relocalization", .35 + .35 * (float(done) / max(float(count), 1.0)), "Buscando vistas conocidas…")
            relocalization = self.dependencies.relocalization(source.path, anchors=anchors, targets=targets, config=AnchorRelocalizationConfig(analysis_width=960), exclude_same_frame=False, progress_callback=relocalization_progress)
            self._write(self._path(source, "_camera_anchor_relocalization_v1.json"), relocalization)
            if not isinstance(relocalization, dict) or relocalization.get("status") != "ok":
                phase_failure("relocalization", relocalization); return
            results = relocalization.get("results", []) if isinstance(relocalization, dict) else []
            soft = [{key: item.get(key) for key in ("frame", "homography", "confidence_tier", "anchor_frame", "score", "match_count", "inlier_count", "inlier_ratio", "median_reprojection_px", "direct_state_distance")} for item in results if isinstance(item, dict) and item.get("status") == "ok" and isinstance(item.get("homography"), list)]
            unresolved = [int(item["frame"]) for item in results if isinstance(item, dict) and item.get("status") != "ok" and "frame" in item]
            camera["relocalized_keyframe_count"] = len(soft); camera["direct_unresolved_keyframe_count"] = len(unresolved)
            progress("dense", .70, "Completando el vídeo…")
            def dense_progress(_phase, done, count, *_args):
                progress("dense", .70 + .29 * (float(done) / max(float(count), 1.0)), "Completando el vídeo…")
            dense = self.dependencies.dense(source.path, soft_anchors=soft, image_width=width, image_height=height, pitch_length_m=float(pitch["length_m"]), pitch_width_m=float(pitch["width_m"]), start_frame=0, end_frame=total - 1, config=DenseCameraAtlasConfig(), progress_callback=dense_progress)
            self._write(self._path(source, "_camera_atlas_dense_v1.json"), dense)
            if not isinstance(dense, dict) or dense.get("status") != "ok" or not isinstance(dense.get("records"), list):
                phase_failure("dense", dense); return
            records = dense.get("records", []) if isinstance(dense, dict) else []
            present = {int(item["frame"]) for item in records if isinstance(item, dict) and "frame" in item}
            missing_candidates = sorted(set(unresolved) - present)
            camera.update({"dense_frame_count": int(dense.get("derived_frame_count", len(records))) if isinstance(dense, dict) else 0, "dense_missing_frame_count": int(dense.get("missing_frame_count", total - len(records))) if isinstance(dense, dict) else total, "dense_coverage_fraction": float(dense.get("coverage_fraction", len(records) / max(total, 1))) if isinstance(dense, dict) else 0.0})
            state["artifacts"] = {"trajectory": self._path(source, "_camera_trajectory_v1.json").name, "relocalization": self._path(source, "_camera_anchor_relocalization_v1.json").name, "dense": self._path(source, "_camera_atlas_dense_v1.json").name}
            if dense.get("status") == "ok" and camera["dense_missing_frame_count"] == 0 and camera["dense_frame_count"] == total:
                camera.update({"status": "complete", "phase": "complete", "progress_fraction": 1.0, "human_message": "Cámara preparada"})
            elif missing_candidates:
                selection = self.dependencies.active_anchor(source.path, missing_candidates, config=ActiveAnchorConfig(analysis_width=960))
                recommendations = selection.get("recommendations", []) if isinstance(selection, dict) else []
                if recommendations:
                    recommendation = recommendations[0]; camera.update({"status": "needs_human", "phase": "needs_human", "progress_fraction": 1.0, "human_message": "Necesitamos una referencia manual en una vista nueva.", "recommended_human_frame": int(recommendation.get("frame")), "active_candidate_count": len(missing_candidates), "expected_gain": recommendation.get("expected_gain_count", recommendation.get("expected_gain"))})
                else:
                    camera.update({"status": "needs_attention", "phase": "needs_attention", "progress_fraction": 1.0, "human_message": "No se pudo completar automáticamente la cámara."})
            else:
                camera.update({"status": "needs_attention", "phase": "needs_attention", "progress_fraction": 1.0, "human_message": "No se pudo completar automáticamente la cámara."})
            state["input_signature"] = signature; self._save_state(source, state)
        except Exception as error:
            camera.update({"status": "error", "phase": "error", "human_message": "No se pudo completar la cámara.", "error": str(error)}); self._save_state(source, state)
