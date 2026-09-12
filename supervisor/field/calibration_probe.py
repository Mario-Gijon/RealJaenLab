"""Interactive real-frame probe for RealJaenLab FIELD calibration.

This is a development tool for EXP-001, not the final web UI.

It extracts one native-resolution frame from a video with ffmpeg, lets the user
annotate semantic pitch lines with two clicks each, calls the production
``solve_calibration`` function, and overlays the complete projected pitch.

Example:

    python -m supervisor.field.calibration_probe \
        input/match.mp4 \
        --time 00:12:34.500

Controls
--------
1. Select a semantic line in the right panel.
2. Click two points along that real painted line in the frame.
3. Repeat for at least two x-constant and two y-constant pitch lines.
4. Press "Resolver".
5. Visually inspect markings that were NOT used for solving.

A new two-click annotation replaces the previous annotation for the currently
selected semantic line.
"""

from __future__ import annotations

import argparse
from io import BytesIO
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RadioButtons

from supervisor.field.calibration import (
    CalibrationError,
    LineAnnotation,
    solve_calibration,
)
from supervisor.field.pitch import PitchModel


def _parse_time(value: str) -> float:
    """Parse seconds or HH:MM:SS(.sss) into seconds."""
    value = value.strip()
    if not value:
        raise argparse.ArgumentTypeError("Time cannot be empty.")

    try:
        seconds = float(value)
    except ValueError:
        parts = value.split(":")
        if len(parts) not in (2, 3):
            raise argparse.ArgumentTypeError(
                "Use seconds or HH:MM:SS(.sss), for example 754.5 or 00:12:34.500."
            )

        try:
            numeric = [float(part) for part in parts]
        except ValueError as exc:
            raise argparse.ArgumentTypeError("Invalid time value.") from exc

        if len(numeric) == 2:
            minutes, seconds_part = numeric
            seconds = minutes * 60.0 + seconds_part
        else:
            hours, minutes, seconds_part = numeric
            seconds = hours * 3600.0 + minutes * 60.0 + seconds_part

    if seconds < 0:
        raise argparse.ArgumentTypeError("Time must be >= 0.")
    return seconds


def _extract_frame(video_path: Path, seconds: float):
    """Decode one frame at native video resolution using ffmpeg."""
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(video_path),
        "-ss",
        f"{seconds:.6f}",
        "-frames:v",
        "1",
        "-f",
        "image2pipe",
        "-vcodec",
        "png",
        "pipe:1",
    ]

    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "ffmpeg was not found. Install ffmpeg or ensure it is available in PATH."
        ) from exc
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(
            f"ffmpeg could not extract the requested frame: {detail or 'unknown error'}"
        ) from exc

    if not result.stdout:
        raise RuntimeError(
            "ffmpeg returned no frame. Check that the timestamp is inside the video."
        )

    return mpimg.imread(BytesIO(result.stdout), format="png")


class CalibrationProbe:
    def __init__(
        self,
        frame,
        *,
        video_path: Path,
        timestamp_seconds: float,
        pitch_length: float,
        pitch_width: float,
        output_path: Path | None,
    ) -> None:
        self.frame = frame
        self.video_path = video_path
        self.timestamp_seconds = timestamp_seconds
        self.pitch = PitchModel(pitch_length, pitch_width)
        self.output_path = output_path

        self.video_height = int(frame.shape[0])
        self.video_width = int(frame.shape[1])

        self.annotations: dict[str, LineAnnotation] = {}
        self.history: list[tuple[str, LineAnnotation | None]] = []
        self.pending_first_point: tuple[float, float] | None = None
        self.solution = None
        self.overlay_visible = True

        self.label_to_name = {
            line.label: line.name for line in self.pitch.straight_lines
        }
        self.name_to_line = {
            line.name: line for line in self.pitch.straight_lines
        }

        labels = [line.label for line in self.pitch.straight_lines]
        self.selected_label = labels[0]

        self.figure = plt.figure(figsize=(16, 9))
        self.image_ax = self.figure.add_axes([0.035, 0.075, 0.70, 0.88])
        self.image_ax.imshow(self.frame)
        self.image_ax.set_xlim(0, self.video_width)
        self.image_ax.set_ylim(self.video_height, 0)
        self.image_ax.set_title(self._base_title())
        self.image_ax.set_xlabel(
            f"Frame nativo: {self.video_width} × {self.video_height} px"
        )

        self.radio_ax = self.figure.add_axes([0.755, 0.39, 0.235, 0.56])
        self.radio = RadioButtons(
            self.radio_ax,
            labels,
            active=0,
            activecolor="tab:blue",
        )
        for label in self.radio.labels:
            label.set_fontsize(8)
        self.radio_ax.set_title("Línea semántica", fontsize=10)

        self.status_ax = self.figure.add_axes([0.755, 0.285, 0.235, 0.085])
        self.status_ax.axis("off")
        self.status_text = self.status_ax.text(
            0.0,
            1.0,
            "",
            va="top",
            ha="left",
            fontsize=9,
            wrap=True,
        )

        self.solve_ax = self.figure.add_axes([0.755, 0.225, 0.11, 0.045])
        self.solve_button = Button(self.solve_ax, "Resolver")

        self.undo_ax = self.figure.add_axes([0.88, 0.225, 0.11, 0.045])
        self.undo_button = Button(self.undo_ax, "Deshacer")

        self.remove_ax = self.figure.add_axes([0.755, 0.165, 0.11, 0.045])
        self.remove_button = Button(self.remove_ax, "Borrar línea")

        self.clear_ax = self.figure.add_axes([0.88, 0.165, 0.11, 0.045])
        self.clear_button = Button(self.clear_ax, "Borrar todo")

        self.overlay_ax = self.figure.add_axes([0.755, 0.105, 0.11, 0.045])
        self.overlay_button = Button(self.overlay_ax, "Overlay")

        self.save_ax = self.figure.add_axes([0.88, 0.105, 0.11, 0.045])
        self.save_button = Button(self.save_ax, "Guardar JSON")

        self.help_ax = self.figure.add_axes([0.755, 0.015, 0.235, 0.07])
        self.help_ax.axis("off")
        self.help_ax.text(
            0.0,
            1.0,
            "Selecciona línea → 2 clics.\n"
            "Mínimo práctico: 2 líneas de cada familia.\n"
            "Después comprueba líneas NO anotadas.",
            va="top",
            fontsize=8,
        )

        self.radio.on_clicked(self._on_select_line)
        self.solve_button.on_clicked(self._on_solve)
        self.undo_button.on_clicked(self._on_undo)
        self.remove_button.on_clicked(self._on_remove_selected)
        self.clear_button.on_clicked(self._on_clear)
        self.overlay_button.on_clicked(self._on_toggle_overlay)
        self.save_button.on_clicked(self._on_save)
        self.figure.canvas.mpl_connect("button_press_event", self._on_image_click)

        self._set_status(
            "Marca una línea con dos clics. "
            "Las coordenadas se guardan directamente en píxeles nativos."
        )
        self._redraw()

    def _base_title(self) -> str:
        return (
            f"{self.video_path.name} · t={self.timestamp_seconds:.3f}s · "
            f"campo {self.pitch.length:.1f}×{self.pitch.width:.1f} m"
        )

    @property
    def selected_name(self) -> str:
        return self.label_to_name[self.selected_label]

    def _set_status(self, text: str) -> None:
        self.status_text.set_text(text)
        self.figure.canvas.draw_idle()

    def _on_select_line(self, label: str) -> None:
        self.selected_label = label
        self.pending_first_point = None
        name = self.selected_name
        if name in self.annotations:
            self._set_status(
                f"{label} ya está anotada. Dos nuevos clics la reemplazarán."
            )
        else:
            self._set_status(f"{label}: haz dos clics sobre la línea real.")

    def _on_image_click(self, event) -> None:
        if event.inaxes is not self.image_ax:
            return
        if event.button != 1:
            return
        if event.xdata is None or event.ydata is None:
            return

        u = float(event.xdata)
        v = float(event.ydata)

        if not (0.0 <= u <= self.video_width and 0.0 <= v <= self.video_height):
            return

        if self.pending_first_point is None:
            self.pending_first_point = (u, v)
            self._set_status(
                f"{self.selected_label}: primer punto "
                f"({u:.1f}, {v:.1f}). Marca el segundo."
            )
            self._redraw()
            return

        first = self.pending_first_point
        self.pending_first_point = None

        previous = self.annotations.get(self.selected_name)
        annotation = LineAnnotation(
            self.selected_name,
            p1=_image_point(first),
            p2=_image_point((u, v)),
        )

        # Validation of coincident clicks is delegated to the solver too, but
        # catching it here makes the interactive tool less frustrating.
        distance_sq = (annotation.p2.u - annotation.p1.u) ** 2 + (
            annotation.p2.v - annotation.p1.v
        ) ** 2
        if distance_sq < 4.0:
            self._set_status("Los dos clics están demasiado juntos. Repite la línea.")
            self._redraw()
            return

        self.history.append((self.selected_name, previous))
        self.annotations[self.selected_name] = annotation
        self.solution = None
        self._set_status(
            f"{self.selected_label} anotada. "
            f"Total: {len(self.annotations)} líneas."
        )
        self._redraw()

    def _on_undo(self, _event) -> None:
        self.pending_first_point = None
        if not self.history:
            self._set_status("No hay acciones que deshacer.")
            return

        name, previous = self.history.pop()
        if previous is None:
            self.annotations.pop(name, None)
        else:
            self.annotations[name] = previous
        self.solution = None
        self._set_status(f"Deshecha la última anotación. Total: {len(self.annotations)}.")
        self._redraw()

    def _on_remove_selected(self, _event) -> None:
        self.pending_first_point = None
        name = self.selected_name
        previous = self.annotations.get(name)
        if previous is None:
            self._set_status(f"{self.selected_label} no está anotada.")
            return
        self.history.append((name, previous))
        del self.annotations[name]
        self.solution = None
        self._set_status(f"{self.selected_label} eliminada.")
        self._redraw()

    def _on_clear(self, _event) -> None:
        self.pending_first_point = None
        if not self.annotations:
            self._set_status("No hay anotaciones que borrar.")
            return

        # One snapshot entry per line makes repeated Undo useful after clear.
        for name, annotation in list(self.annotations.items()):
            self.history.append((name, annotation))
        self.annotations.clear()
        self.solution = None
        self._set_status("Anotaciones borradas.")
        self._redraw()

    def _on_solve(self, _event) -> None:
        self.pending_first_point = None
        try:
            self.solution = solve_calibration(
                list(self.annotations.values()),
                pitch_length=self.pitch.length,
                pitch_width=self.pitch.width,
                video_width=self.video_width,
                video_height=self.video_height,
            )
        except CalibrationError as exc:
            self.solution = None
            self._set_status(f"No se puede resolver: {exc}")
            self._redraw()
            return

        metrics = self.solution.metrics
        self._set_status(
            "Homografía resuelta. "
            f"Intersecciones: {len(self.solution.derived_correspondences)} · "
            f"mediana ajuste: {metrics['median_px']:.2f}px · "
            f"p95: {metrics['p95_px']:.2f}px. "
            "Ahora mira especialmente las marcas que NO anotaste."
        )

        print()
        print("=== RealJaen FIELD calibration probe ===")
        print(f"Video: {self.video_path}")
        print(f"Timestamp: {self.timestamp_seconds:.6f} s")
        print(f"Native frame: {self.video_width} x {self.video_height}")
        print(f"Annotations: {len(self.solution.annotations)}")
        print(
            "Derived intersections: "
            f"{len(self.solution.derived_correspondences)}"
        )
        print(
            "Line-fit metrics: "
            f"median={metrics['median_px']:.3f}px, "
            f"p95={metrics['p95_px']:.3f}px, "
            f"max={metrics['max_px']:.3f}px"
        )
        print("H_pitch_to_image:")
        for row in self.solution.homography_pitch_to_image:
            print("  " + " ".join(f"{value: .10f}" for value in row))
        print("H_image_to_pitch:")
        for row in self.solution.homography_image_to_pitch:
            print("  " + " ".join(f"{value: .10f}" for value in row))
        print()

        self._redraw()

    def _on_toggle_overlay(self, _event) -> None:
        self.overlay_visible = not self.overlay_visible
        self._redraw()

    def _on_save(self, _event) -> None:
        if self.output_path is None:
            self._set_status(
                "No se indicó --output. Ejecuta el probe con "
                "--output /tmp/calibracion.json para guardar."
            )
            return

        payload: dict[str, Any] = {
            "video": str(self.video_path),
            "timestamp_seconds": self.timestamp_seconds,
            "video_width": self.video_width,
            "video_height": self.video_height,
            "pitch_length": self.pitch.length,
            "pitch_width": self.pitch.width,
            "annotations": [
                annotation.as_dict()
                for annotation in self.annotations.values()
            ],
        }

        if self.solution is not None:
            payload["solution"] = self.solution.as_dict()

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        self._set_status(f"Guardado en {self.output_path}.")
        print(f"Calibration probe JSON saved to: {self.output_path}")

    def _redraw(self) -> None:
        # Keep the raster image, clear only development overlay artists.
        while self.image_ax.lines:
            self.image_ax.lines[-1].remove()
        while self.image_ax.texts:
            self.image_ax.texts[-1].remove()

        # Human annotations.
        for name, annotation in self.annotations.items():
            line = self.name_to_line[name]
            self.image_ax.plot(
                [annotation.p1.u, annotation.p2.u],
                [annotation.p1.v, annotation.p2.v],
                marker="o",
                linewidth=2.5,
            )
            midpoint_u = (annotation.p1.u + annotation.p2.u) / 2.0
            midpoint_v = (annotation.p1.v + annotation.p2.v) / 2.0
            self.image_ax.text(
                midpoint_u,
                midpoint_v,
                line.label,
                fontsize=7,
                bbox={"alpha": 0.65, "pad": 2},
            )

        # Pending first click.
        if self.pending_first_point is not None:
            self.image_ax.plot(
                [self.pending_first_point[0]],
                [self.pending_first_point[1]],
                marker="x",
                markersize=10,
                markeredgewidth=2,
            )

        # Complete pitch projection.
        if self.solution is not None and self.overlay_visible:
            geometry = self.solution.projected_geometry

            for projected_line in geometry["lines"]:
                points = projected_line["points"]
                self.image_ax.plot(
                    [points[0][0], points[1][0]],
                    [points[0][1], points[1][1]],
                    linewidth=1.4,
                    alpha=0.9,
                )

            for curve in geometry["curves"]:
                points = curve["points"]
                if not points:
                    continue
                xs = [point[0] for point in points]
                ys = [point[1] for point in points]
                if curve["closed"]:
                    xs.append(points[0][0])
                    ys.append(points[0][1])
                self.image_ax.plot(xs, ys, linewidth=1.2, alpha=0.9)

            for point in geometry["points"]:
                u, v = point["point"]
                self.image_ax.plot([u], [v], marker="+", markersize=7)

        self.image_ax.set_xlim(0, self.video_width)
        self.image_ax.set_ylim(self.video_height, 0)

        family_counts = {"x_constant": 0, "y_constant": 0}
        for name in self.annotations:
            family_counts[self.name_to_line[name].family] += 1

        self.image_ax.set_title(
            self._base_title()
            + "\n"
            + f"Anotaciones: {len(self.annotations)} · "
            + f"x_constant={family_counts['x_constant']} · "
            + f"y_constant={family_counts['y_constant']}"
        )
        self.figure.canvas.draw_idle()

    def show(self) -> None:
        plt.show()


def _image_point(point: tuple[float, float]):
    # Local import avoids leaking UI concerns into calibration.py.
    from supervisor.field.calibration import ImagePoint

    return ImagePoint(float(point[0]), float(point[1]))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract a real Veo frame, annotate semantic field lines, solve a "
            "homography, and inspect the complete projected pitch."
        )
    )
    parser.add_argument("video", type=Path, help="Path to an MP4 video.")
    parser.add_argument(
        "--time",
        type=_parse_time,
        default=0.0,
        help="Timestamp as seconds or HH:MM:SS(.sss). Default: 0.",
    )
    parser.add_argument(
        "--length",
        type=float,
        default=105.0,
        help="Pitch length in metres. Default: 105.",
    )
    parser.add_argument(
        "--width",
        type=float,
        default=68.0,
        help="Pitch width in metres. Default: 68.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Optional JSON output path, for example "
            "/tmp/realjaen_calibration_probe.json."
        ),
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    video_path = args.video.expanduser().resolve()

    if not video_path.is_file():
        raise SystemExit(f"Video not found: {video_path}")
    if video_path.suffix.lower() != ".mp4":
        raise SystemExit("The probe currently expects an MP4 video.")

    try:
        frame = _extract_frame(video_path, args.time)
        probe = CalibrationProbe(
            frame,
            video_path=video_path,
            timestamp_seconds=args.time,
            pitch_length=args.length,
            pitch_width=args.width,
            output_path=args.output,
        )
        probe.show()
    except (RuntimeError, CalibrationError, ValueError) as exc:
        print(f"Calibration probe error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
