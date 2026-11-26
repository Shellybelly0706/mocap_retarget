"""Reusable markers preview helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from utils.markers.data import (
    MocapData,
)


class MarkerViewer:
    """Matplotlib 3-D viewer shared by interactive playback and MP4 export."""

    def __init__(
        self,
        data: MocapData,
        show_labels: bool,
        track_subject: bool,
        width_px: int,
        height_px: int,
        dpi: int,
    ) -> None:
        import matplotlib.pyplot as plt

        self.data = data
        self.show_labels = show_labels
        self.track_subject = track_subject
        self.current_frame = 0
        self.playing = True
        self.fig = plt.figure(figsize=(width_px / dpi, height_px / dpi), dpi=dpi)
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.scatter = self.ax.scatter([], [], [], s=18, c="#00d9ff", depthshade=False)
        self.texts = [
            self.ax.text(0.0, 0.0, 0.0, "", fontsize=5, color="#222222")
            for _ in data.labels
        ]
        self.status = self.fig.text(0.01, 0.01, "", fontsize=8)
        self.direction_artists: list[object] = []
        self._configure_axes()
        self.update(0)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

    def _configure_axes(self) -> None:
        finite_points = self.data.markers[self.data.valid_mask]
        mins = np.min(finite_points, axis=0)
        maxs = np.max(finite_points, axis=0)
        self.fixed_center = (mins + maxs) / 2.0
        self.fixed_half_extent = max(float(np.max(maxs - mins)) / 2.0, 0.1) * 1.05
        # In a locomotion take, global equal limits can shrink a 1.7 m body to
        # a tiny cluster because the subject travels several metres. A robust
        # fixed body-scale extent lets the default camera follow translation
        # while preserving the same scale and X/Y/Z aspect on every frame.
        with np.errstate(all="ignore"):
            frame_spans = np.nanmax(self.data.markers, axis=1) - np.nanmin(
                self.data.markers, axis=1
            )
        self.tracking_half_extent = max(
            float(np.nanpercentile(frame_spans, 99.5)) / 2.0 * 1.08,
            0.1,
        )
        if self.track_subject:
            first_valid = self.data.valid_mask[0]
            center = np.median(self.data.markers[0, first_valid], axis=0)
            half_extent = self.tracking_half_extent
        else:
            center = self.fixed_center
            half_extent = self.fixed_half_extent
        self._set_view_center(center, half_extent)
        self.ax.set_box_aspect((1.0, 1.0, 1.0))
        self.ax.set_xlabel("X [m]")
        self.ax.set_ylabel("Y [m]")
        self.ax.set_zlabel("Z [m]")
        self.ax.grid(True, alpha=0.35)
        self.fig.subplots_adjust(left=0.02, right=0.96, bottom=0.06, top=0.92)

    def _set_view_center(self, center: np.ndarray, half_extent: float) -> None:
        self.ax.set_xlim(center[0] - half_extent, center[0] + half_extent)
        self.ax.set_ylim(center[1] - half_extent, center[1] + half_extent)
        self.ax.set_zlim(center[2] - half_extent, center[2] + half_extent)
        for artist in self.direction_artists:
            artist.remove()
        self.direction_artists.clear()
        arrow_origin = center - 0.82 * half_extent
        arrow_length = 0.18 * half_extent
        for direction, color, label in (
            ((1, 0, 0), "#d62728", "+X"),
            ((0, 1, 0), "#2ca02c", "+Y"),
            ((0, 0, 1), "#1f77b4", "+Z"),
        ):
            arrow = self.ax.quiver(
                *arrow_origin,
                *(np.asarray(direction) * arrow_length),
                color=color,
                arrow_length_ratio=0.18,
            )
            endpoint = arrow_origin + np.asarray(direction) * arrow_length * 1.12
            label_artist = self.ax.text(
                *endpoint, label, color=color, fontsize=8, fontweight="bold"
            )
            self.direction_artists.extend((arrow, label_artist))

    def update(self, frame: int) -> list[object]:
        frame = int(np.clip(frame, 0, self.data.markers.shape[0] - 1))
        self.current_frame = frame
        valid = self.data.valid_mask[frame]
        xyz = self.data.markers[frame, valid]
        if self.track_subject and xyz.size:
            self._set_view_center(np.median(xyz, axis=0), self.tracking_half_extent)
        if xyz.size:
            self.scatter._offsets3d = (xyz[:, 0], xyz[:, 1], xyz[:, 2])
        else:
            empty = np.array([], dtype=np.float64)
            self.scatter._offsets3d = (empty, empty, empty)
        for marker, text_artist in enumerate(self.texts):
            visible = bool(valid[marker] and self.show_labels)
            text_artist.set_visible(visible)
            if visible:
                point = self.data.markers[frame, marker]
                text_artist.set_position((point[0], point[1]))
                text_artist.set_3d_properties(point[2], zdir="z")
                text_artist.set_text(str(self.data.labels[marker]))
        self.ax.set_title(
            f"Frame {int(self.data.frame_ids[frame])}  |  "
            f"t = {float(self.data.timestamps[frame]):.3f} s  |  "
            f"valid = {int(np.count_nonzero(valid))}/{len(valid)}"
        )
        self.status.set_text(
            "Space/P: play-pause   ←/→: previous/next frame   "
            "L: toggle labels   Home/End: first/last"
        )
        return [self.scatter, self.ax.title, self.status, *self.texts]

    def _on_key(self, event: object) -> None:
        key = getattr(event, "key", "")
        if key in {" ", "space", "p"}:
            self.playing = not self.playing
            if hasattr(self, "animation"):
                if self.playing:
                    self.animation.event_source.start()
                else:
                    self.animation.event_source.stop()
        elif key == "right":
            self.playing = False
            if hasattr(self, "animation"):
                self.animation.event_source.stop()
            self.update(self.current_frame + 1)
            self.fig.canvas.draw_idle()
        elif key == "left":
            self.playing = False
            if hasattr(self, "animation"):
                self.animation.event_source.stop()
            self.update(self.current_frame - 1)
            self.fig.canvas.draw_idle()
        elif key and key.lower() == "l":
            self.show_labels = not self.show_labels
            self.update(self.current_frame)
            self.fig.canvas.draw_idle()
        elif key == "home":
            self.update(0)
            self.fig.canvas.draw_idle()
        elif key == "end":
            self.update(self.data.markers.shape[0] - 1)
            self.fig.canvas.draw_idle()

    def show(self) -> None:
        import matplotlib.animation as animation
        import matplotlib.pyplot as plt

        def advance(_: int) -> list[object]:
            if self.playing:
                next_frame = (self.current_frame + 1) % self.data.markers.shape[0]
                return self.update(next_frame)
            return []

        interval_ms = 1000.0 / self.data.frame_rate
        self.animation = animation.FuncAnimation(
            self.fig,
            advance,
            interval=interval_ms,
            blit=False,
            cache_frame_data=False,
        )
        plt.show()


def export_mp4(
    data: MocapData,
    destination: Path,
    show_labels: bool,
    track_subject: bool,
    video_fps: float,
    width_px: int,
    height_px: int,
    dpi: int,
) -> tuple[int, int, float]:
    """Render a real-time-duration MP4 via OpenCV, sampling only the preview."""
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError(
            "MP4 export requires the installed OpenCV (cv2) package"
        ) from error

    import matplotlib.pyplot as plt

    stride = max(1, int(round(data.frame_rate / video_fps)))
    actual_fps = data.frame_rate / stride
    frame_indices = list(range(0, data.markers.shape[0], stride))
    if frame_indices[-1] != data.markers.shape[0] - 1:
        frame_indices.append(data.markers.shape[0] - 1)

    viewer = MarkerViewer(data, show_labels, track_subject, width_px, height_px, dpi)
    viewer.fig.canvas.draw()
    canvas_width, canvas_height = viewer.fig.canvas.get_width_height()
    writer = cv2.VideoWriter(
        str(destination),
        cv2.VideoWriter_fourcc(*"mp4v"),
        actual_fps,
        (canvas_width, canvas_height),
    )
    if not writer.isOpened():
        plt.close(viewer.fig)
        raise RuntimeError("OpenCV could not open an MP4 writer with the mp4v codec")

    print(
        f"Rendering MP4: {len(frame_indices)} preview frames, stride={stride}, "
        f"fps={actual_fps:g}, size={canvas_width}x{canvas_height}",
        flush=True,
    )
    try:
        progress_step = max(1, len(frame_indices) // 10)
        for output_index, source_frame in enumerate(frame_indices):
            viewer.update(source_frame)
            viewer.fig.canvas.draw()
            rgba = np.asarray(viewer.fig.canvas.buffer_rgba())
            bgr = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
            writer.write(bgr)
            if (output_index + 1) % progress_step == 0 or output_index + 1 == len(
                frame_indices
            ):
                print(
                    f"  MP4 progress: {output_index + 1}/{len(frame_indices)}",
                    flush=True,
                )
    finally:
        writer.release()
        plt.close(viewer.fig)
    if not destination.is_file() or destination.stat().st_size == 0:
        raise RuntimeError(f"MP4 writer did not create a readable file: {destination}")
    return len(frame_indices), stride, actual_fps
