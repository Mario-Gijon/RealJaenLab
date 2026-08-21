"""Single-frame FIELD calibration solver boundary.

Mixed point/line/arc optimization belongs here when it is implemented later.
"""

import cv2
import numpy as np


def _points_are_non_collinear(points: np.ndarray) -> bool:
    if len(points) < 4:
        return False
    centered = points - points.mean(axis=0)
    singular_values = np.linalg.svd(centered, compute_uv=False)
    return len(singular_values) == 2 and singular_values[1] > singular_values[0] * 1e-4


def solve_field_calibration(
    point_constraints: list[dict],
    line_constraints: list[dict],
    arc_constraints: list[dict],
    image_width: int | None,
    image_height: int | None,
) -> dict:
    """Solve point-only calibration; reserve mixed constraints for a future solver."""
    if line_constraints or arc_constraints:
        return {
            "status": "not_implemented",
            "detail": "El solver mixto de puntos, líneas y arcos todavía no está conectado.",
        }
    if len(point_constraints) < 4:
        return {"status": "insufficient", "detail": "Se requieren al menos cuatro puntos no colineales."}

    try:
        world_points = np.array([[item["world"]["x"], item["world"]["y"]] for item in point_constraints], dtype=np.float64)
        image_points = np.array([[item["image"]["x"], item["image"]["y"]] for item in point_constraints], dtype=np.float64)
    except (KeyError, TypeError, ValueError):
        return {"status": "invalid", "detail": "Coordenadas de restricción no válidas."}

    if not _points_are_non_collinear(world_points) or not _points_are_non_collinear(image_points):
        return {"status": "degenerate", "detail": "Los puntos son geométricamente degenerados."}

    homography, _ = cv2.findHomography(world_points, image_points, method=0)
    if homography is None or not np.isfinite(homography).all() or abs(homography[2, 2]) < 1e-12:
        return {"status": "invalid", "detail": "No se pudo calcular una homografía válida."}
    homography /= homography[2, 2]
    homogeneous_world = np.column_stack([world_points, np.ones(len(world_points))])
    projected = homogeneous_world @ homography.T
    projected = projected[:, :2] / projected[:, 2:3]
    return {
        "status": "solved",
        "homography": homography.tolist(),
        "reprojection_error": float(np.mean(np.linalg.norm(projected - image_points, axis=1))),
    }
