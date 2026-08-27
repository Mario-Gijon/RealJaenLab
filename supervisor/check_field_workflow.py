"""Fast state-machine checks for FIELD workflow orchestration (no CV)."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from supervisor.field_workflow import FieldWorkflowError, FieldWorkflowManager, WorkflowDependencies, extract_human_anchors


class Source:
    reference = "asset_test"
    result_key = "asset_test"
    def __init__(self, root: Path): self.path = root / "managed_upload.mp4"; self.path.write_bytes(b"video")
    def review_path(self, directory: Path) -> Path: return directory / "asset_test.json"
    def result_basename(self, suffix: str) -> str: return f"asset_test{suffix}"


def review(*, confirmed=True, anchors=True) -> dict:
    calibrations = [
        {"frame": 4, "source": "manual", "purpose": "match_setup", "homography": [[1,0,0],[0,1,0],[0,0,1]]},
        {"frame": 7, "source": "manual", "purpose": "match_setup", "homography": [[1,0,0],[0,1,0],[0,0,1]]},
        {"frame": 9, "source": "manual", "purpose": "active_anchor", "homography": [[1,0,0],[0,1,0],[0,0,1]]},
        {"frame": 3, "source": "manual", "purpose": "pitch_setup", "homography": [[1,0,0],[0,1,0],[0,0,1]]},
    ] if anchors else []
    return {"field": {"pitch_model": {"status": "confirmed" if confirmed else "default", "length_m": 105, "width_m": 68}, "match_setup": {"completed_frames": [4], "rejected_frames": [7]}, "calibrations": calibrations}}


def main() -> None:
    with TemporaryDirectory() as temporary:
        root = Path(temporary); reviews = root / "reviews"; reviews.mkdir(); source = Source(root)
        calls: dict[str, int] = {"trajectory": 0, "relocalization": 0, "dense": 0, "active": 0}
        def trajectory(*_args, **_kwargs): calls["trajectory"] += 1; return {"status": "ok", "keyframes": [{"frame": 0}, {"frame": 5}, {"frame": 9}]}
        def relocalization(_path, *, anchors, targets, **_kwargs):
            calls["relocalization"] += 1; assert [item["frame"] for item in anchors] == [4, 9]; assert [item["frame"] for item in targets] == [0, 5, 9]
            return {"status": "ok", "results": [{"frame": 0, "status": "ok", "homography": [[1,0,0],[0,1,0],[0,0,1]], "confidence_tier": "verified"}, {"frame": 5, "status": "unrelocalized"}, {"frame": 9, "status": "ok", "homography": [[1,0,0],[0,1,0],[0,0,1]], "confidence_tier": "high"}]}
        def dense(*_args, soft_anchors, **_kwargs):
            calls["dense"] += 1; assert len(soft_anchors) == 2; return {"status": "ok", "derived_frame_count": 10, "missing_frame_count": 0, "coverage_fraction": 1.0, "records": [{"frame": frame} for frame in range(10)]}
        def active(*_args, **_kwargs): calls["active"] += 1; return {"recommendations": [{"frame": 5, "expected_gain": 1}]}
        manager = FieldWorkflowManager(reviews, lambda _path: {"frame_count": 10, "width": 1920, "height": 1080}, WorkflowDependencies(trajectory, relocalization, dense, active))
        source.review_path(reviews).write_text(json.dumps(review()), encoding="utf-8")
        state = manager.start(source); manager.jobs[source.reference].result()
        complete = manager.get(source)["camera"]
        assert complete["status"] == "complete" and calls == {"trajectory": 1, "relocalization": 1, "dense": 1, "active": 0}
        assert (reviews / "asset_test_camera_atlas_dense_v1.json").exists()
        assert extract_human_anchors(review()) == extract_human_anchors(review())
        def incomplete_dense(*_args, **_kwargs):
            return {"status": "ok", "derived_frame_count": 9, "missing_frame_count": 1, "coverage_fraction": .9, "records": [{"frame": frame} for frame in range(10) if frame != 5]}
        needs_human = FieldWorkflowManager(reviews, lambda _path: {"frame_count": 10, "width": 1920, "height": 1080}, WorkflowDependencies(trajectory, relocalization, incomplete_dense, active))
        needs_human.start(source); needs_human.jobs[source.reference].result()
        assert needs_human.get(source)["camera"]["status"] == "needs_human"
        no_candidate = FieldWorkflowManager(reviews, lambda _path: {"frame_count": 10, "width": 1920, "height": 1080}, WorkflowDependencies(trajectory, relocalization, incomplete_dense, lambda *_args, **_kwargs: {"recommendations": []}))
        no_candidate.start(source); no_candidate.jobs[source.reference].result()
        assert no_candidate.get(source)["camera"]["status"] == "needs_attention"
        source.review_path(reviews).unlink()
        try: manager.start(source)
        except FieldWorkflowError: pass
        else: raise AssertionError("missing review accepted")
        source.review_path(reviews).write_text(json.dumps(review(confirmed=False)), encoding="utf-8")
        try: manager.start(source)
        except FieldWorkflowError: pass
        else: raise AssertionError("unconfirmed pitch accepted")
    print("check_field_workflow: ok")


if __name__ == "__main__": main()
