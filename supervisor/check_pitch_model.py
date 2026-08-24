"""Small regression check for the default parametric pitch geometry."""

from __future__ import annotations

import math

from supervisor.field_pitch_model import build_pitch_model


def _landmark(model, identifier):
    return next(item["world"] for item in model["landmarks"] if item["id"] == identifier)


def _assert_point(point, x, y):
    assert math.isclose(point["x"], x, abs_tol=1e-12), point
    assert math.isclose(point["y"], y, abs_tol=1e-12), point


def run_regression_check() -> None:
    model = build_pitch_model(105.0, 68.0)
    _assert_point(model["pitch_polygon"][0], -52.5, -34.0)
    _assert_point(model["pitch_polygon"][1], 52.5, -34.0)
    _assert_point(model["pitch_polygon"][2], 52.5, 34.0)
    _assert_point(model["pitch_polygon"][3], -52.5, 34.0)
    _assert_point(_landmark(model, "left_penalty_top"), -36.0, -20.16)
    _assert_point(_landmark(model, "left_penalty_bottom"), -36.0, 20.16)
    _assert_point(_landmark(model, "right_penalty_top"), 36.0, -20.16)
    _assert_point(_landmark(model, "right_penalty_bottom"), 36.0, 20.16)
    _assert_point(_landmark(model, "left_goal_top"), -47.0, -9.16)
    _assert_point(_landmark(model, "left_goal_bottom"), -47.0, 9.16)
    _assert_point(_landmark(model, "right_goal_top"), 47.0, -9.16)
    _assert_point(_landmark(model, "right_goal_bottom"), 47.0, 9.16)
    _assert_point(_landmark(model, "left_penalty_spot"), -41.5, 0.0)
    _assert_point(_landmark(model, "right_penalty_spot"), 41.5, 0.0)
    _assert_point(_landmark(model, "left_goal_post_top"), -52.5, -3.66)
    _assert_point(_landmark(model, "left_goal_post_bottom"), -52.5, 3.66)
    _assert_point(_landmark(model, "right_goal_post_top"), 52.5, -3.66)
    _assert_point(_landmark(model, "right_goal_post_bottom"), 52.5, 3.66)


if __name__ == "__main__":
    run_regression_check()
    print("Pitch model regression OK: 105 x 68")
