"""Reusable markers layout editor helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from utils.markers.layout import (
    get_indices,
    set_index,
)
from utils.markers.layout_io import (
    atomic_json_write,
    backup_once,
)
from utils.markers.layout_models import (
    DISPLAY_MARKER_OFFSET_M,
    MeshData,
)


class MarkerLayoutEditor:
    def __init__(
        self, layout_path: Path, layout: dict[str, Any], mesh: MeshData
    ) -> None:
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button, TextBox

        self.plt = plt
        self.layout_path = layout_path
        self.layout = layout
        self.mesh = mesh
        self.indices = get_indices(layout)
        self.labels = list(self.indices)
        self.current = self._first_unassigned_index()
        self.undo_stack: list[tuple[str, int | None, Any]] = []
        self.pending_duplicate: tuple[str, int, tuple[str, ...]] | None = None
        self.feedback = ""
        self.overlay_artists: list[Any] = []
        self.visible_marker_count = 0

        # Static triangle data used for exact camera-ray/surface intersection.
        # Picking the old all-vertex scatter could select a vertex on the back
        # of the body through the visible surface.
        triangle_vertices = mesh.vertices[mesh.faces]
        self.triangle_origins = triangle_vertices[:, 0]
        self.triangle_edge1 = triangle_vertices[:, 1] - triangle_vertices[:, 0]
        self.triangle_edge2 = triangle_vertices[:, 2] - triangle_vertices[:, 0]
        face_normals = np.cross(self.triangle_edge1, self.triangle_edge2)
        self.vertex_normals = np.zeros_like(mesh.vertices)
        for corner in range(3):
            np.add.at(self.vertex_normals, mesh.faces[:, corner], face_normals)
        normal_lengths = np.linalg.norm(self.vertex_normals, axis=1, keepdims=True)
        self.vertex_normals /= np.maximum(normal_lengths, 1e-12)
        self.mesh_center = (mesh.vertices.min(axis=0) + mesh.vertices.max(axis=0)) / 2

        metadata = self.layout.setdefault("layout_metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError("layout_metadata must be an object when present")
        existing_hash = metadata.get("model_sha256")
        if existing_hash and existing_hash != mesh.sha256:
            raise ValueError(
                "Refusing to edit a layout created against another model SHA-256"
            )
        metadata.update(
            {
                "model_path": str(mesh.path),
                "model_sha256": mesh.sha256,
                "topology_vertex_count": len(mesh.vertices),
            }
        )
        self.layout.setdefault("marker_sources", {})
        backup_once(self.layout_path)
        self._save()

        self.fig = plt.figure(figsize=(14, 9))
        self.ax = self.fig.add_axes([0.03, 0.06, 0.70, 0.90], projection="3d")
        self.ax.plot_trisurf(
            mesh.vertices[:, 0],
            mesh.vertices[:, 1],
            mesh.faces,
            mesh.vertices[:, 2],
            color="#b8c5d6",
            alpha=1.0,
            linewidth=0.0,
            edgecolor="none",
            shade=True,
        )
        self.filled_scatter = self.ax.scatter(
            [],
            [],
            [],
            s=70,
            c="#ff2020",
            edgecolors="white",
            linewidths=0.8,
            depthshade=False,
        )
        self.current_scatter = self.ax.scatter(
            [],
            [],
            [],
            s=140,
            c="#ffe600",
            edgecolors="black",
            linewidths=1.0,
            depthshade=False,
        )
        self._equal_axes()
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")
        self.status_artist = self.fig.text(
            0.75, 0.50, "", va="top", fontsize=10, wrap=True
        )
        self.help_artist = self.fig.text(
            0.75,
            0.94,
            "Shift+left-click: assign visible surface\n"
            "Mouse drag/wheel: rotate/zoom\n"
            "N/P: next/previous   S: skip\n"
            "U: undo   Delete/C: clear\n"
            "Y: confirm duplicate vertex\n"
            "Only markers on the visible body surface are shown",
            va="top",
            fontsize=10,
        )
        textbox_ax = self.fig.add_axes([0.75, 0.39, 0.22, 0.045])
        self.textbox = TextBox(
            textbox_ax, "marker #/name ", initial=str(self.current + 1)
        )
        self.textbox.on_submit(self._select_from_text)
        controls = [
            ("Previous", self.previous),
            ("Next", self.next),
            ("Skip", self.next),
            ("Undo", self.undo),
            ("Clear", self.clear_current),
        ]
        self.buttons = []
        for index, (name, callback) in enumerate(controls):
            button_ax = self.fig.add_axes(
                [0.75 + (index % 2) * 0.115, 0.30 - (index // 2) * 0.055, 0.105, 0.042]
            )
            button = Button(button_ax, name)
            button.on_clicked(callback)
            self.buttons.append(button)
        self.fig.canvas.mpl_connect("button_press_event", self._on_surface_click)
        self.fig.canvas.mpl_connect("button_release_event", self._on_view_changed)
        self.fig.canvas.mpl_connect("scroll_event", self._on_view_changed)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self._refresh()

    def _first_unassigned_index(self) -> int:
        for index, label in enumerate(self.labels):
            if self.indices[label] is None:
                return index
        return 0

    def _equal_axes(self) -> None:
        mins = self.mesh.vertices.min(axis=0)
        maxs = self.mesh.vertices.max(axis=0)
        center = (mins + maxs) / 2
        half = max(maxs - mins) / 2 * 1.05
        self.ax.set_xlim(center[0] - half, center[0] + half)
        self.ax.set_ylim(center[1] - half, center[1] + half)
        self.ax.set_zlim(center[2] - half, center[2] + half)
        self.ax.set_box_aspect((1, 1, 1))

    def _save(self) -> None:
        atomic_json_write(self.layout_path, self.layout)

    def _source_for(self, label: str) -> Any:
        return self.layout.get("marker_sources", {}).get(label)

    def _display_points(self, vertex_ids: Sequence[int]) -> np.ndarray:
        """Return visibly raised points without changing saved surface vertices."""
        ids = np.asarray(vertex_ids, dtype=np.int64)
        return (
            self.mesh.vertices[ids] + DISPLAY_MARKER_OFFSET_M * self.vertex_normals[ids]
        )

    def _camera_location(self) -> np.ndarray:
        """Return the 3-D camera location across supported Matplotlib versions."""
        self.ax.get_proj()
        get_camera_location = getattr(self.ax, "_get_camera_loc", None)
        if callable(get_camera_location):
            return np.asarray(get_camera_location(), dtype=float)

        # Matplotlib 3.5 stores the camera in normalized world coordinates as
        # ``Axes3D.eye`` but does not provide _get_camera_loc(). Undo the same
        # world transformation used by Axes3D.get_proj() to recover data-space
        # coordinates for ray casting.
        from mpl_toolkits.mplot3d import proj3d

        eye = getattr(self.ax, "eye", None)
        if eye is None:
            raise RuntimeError("Matplotlib did not expose a 3-D camera location")
        box_aspect = self.ax._roll_to_vertical(self.ax._box_aspect)
        world = proj3d.world_transformation(
            *self.ax.get_xlim3d(),
            *self.ax.get_ylim3d(),
            *self.ax.get_zlim3d(),
            pb_aspect=box_aspect,
        )
        homogeneous_eye = np.append(np.asarray(eye, dtype=float), 1.0)
        data_eye = np.linalg.solve(world, homogeneous_eye)
        if abs(data_eye[3]) <= 1e-12:
            raise RuntimeError("Matplotlib returned an invalid 3-D camera location")
        return data_eye[:3] / data_eye[3]

    def _vertex_is_visible(self, vertex_id: int) -> bool:
        """Return whether this surface vertex is the camera ray's first hit."""
        origin = self._camera_location()
        target = self.mesh.vertices[int(vertex_id)]
        direction = target - origin
        target_distance = float(np.linalg.norm(direction))
        if not np.isfinite(target_distance) or target_distance <= 1e-12:
            return False
        direction /= target_distance

        ray = np.broadcast_to(direction, self.triangle_edge2.shape)
        pvec = np.cross(ray, self.triangle_edge2)
        determinant = np.einsum("ij,ij->i", self.triangle_edge1, pvec)
        candidate = np.abs(determinant) > 1e-12
        inverse = np.zeros_like(determinant)
        inverse[candidate] = 1.0 / determinant[candidate]
        tvec = origin - self.triangle_origins
        barycentric_u = inverse * np.einsum("ij,ij->i", tvec, pvec)
        qvec = np.cross(tvec, self.triangle_edge1)
        barycentric_v = inverse * np.einsum("ij,ij->i", ray, qvec)
        distance = inverse * np.einsum("ij,ij->i", self.triangle_edge2, qvec)
        epsilon = 1e-8
        candidate &= barycentric_u >= -epsilon
        candidate &= barycentric_v >= -epsilon
        candidate &= barycentric_u + barycentric_v <= 1.0 + epsilon
        candidate &= distance > epsilon
        hits = distance[candidate]
        if not len(hits):
            return False

        # A small tolerance handles a marker vertex shared by several triangles
        # and floating-point error. Rear-side markers are separated from the
        # first hit by the body's thickness and therefore remain hidden.
        first_hit_distance = float(np.min(hits))
        return target_distance - first_hit_distance <= 0.008

    def _update_marker_overlay(self) -> None:
        """Overlay only markers lying on the camera-visible body surface."""
        from matplotlib import patheffects
        from mpl_toolkits.mplot3d import proj3d

        for artist in self.overlay_artists:
            artist.remove()
        self.overlay_artists.clear()

        assigned = [label for label in self.labels if self.indices[label] is not None]
        visible = [
            label
            for label in assigned
            if self._vertex_is_visible(int(self.indices[label]))
        ]
        self.visible_marker_count = len(visible)
        if not visible:
            return
        points = self._display_points([int(self.indices[label]) for label in visible])
        projection = self.ax.get_proj()
        for label, point in zip(visible, points):
            screen_x, screen_y, _ = proj3d.proj_transform(*point, projection)
            current = label == self.labels[self.current]
            dot_color = "#ffe600" if current else "#ff2020"
            label_color = "#111111" if current else "#c00000"
            dot = self.ax.text2D(
                screen_x,
                screen_y,
                "●",
                transform=self.ax.transData,
                color=dot_color,
                fontsize=13 if current else 10,
                fontweight="bold",
                zorder=10_000,
                clip_on=False,
            )
            dot.set_path_effects(
                [
                    patheffects.withStroke(
                        linewidth=2.0, foreground="black" if current else "white"
                    )
                ]
            )
            annotation = self.ax.annotate(
                label,
                xy=(screen_x, screen_y),
                xycoords=self.ax.transData,
                xytext=(5, 4),
                textcoords="offset points",
                color=label_color,
                fontsize=8,
                fontweight="bold",
                zorder=10_001,
                clip_on=False,
                bbox={
                    "boxstyle": "round,pad=0.12",
                    "facecolor": "#fff176" if current else "white",
                    "edgecolor": "none",
                    "alpha": 0.82,
                },
            )
            self.overlay_artists.extend((dot, annotation))

    def _on_view_changed(self, event: Any) -> None:
        if event.inaxes is self.ax:
            self._update_marker_overlay()
            self.fig.canvas.draw_idle()

    def _refresh(self) -> None:
        # Marker dots and labels are rendered by the visibility-tested 2-D
        # overlay below. Keeping the 3-D collections empty avoids Matplotlib's
        # collection depth sorting from revealing markers through the body.
        empty = np.array([])
        self.filled_scatter._offsets3d = (empty, empty, empty)
        label = self.labels[self.current]
        vertex = self.indices[label]
        self.current_scatter._offsets3d = (empty, empty, empty)
        assigned_count = sum(value is not None for value in self.indices.values())
        pending = ""
        if self.pending_duplicate:
            _, candidate, conflicts = self.pending_duplicate
            pending = (
                f"\n\nDUPLICATE PENDING: vertex {candidate} is already used by "
                f"{', '.join(conflicts)}. Press Y to confirm, or pick another vertex."
            )
        self.status_artist.set_text(
            f"Current {self.current + 1}/{len(self.labels)}\n{label}\n"
            f"vertex: {vertex}\nsource: {self._source_for(label)}\n\n"
            f"progress: {assigned_count}/{len(self.labels)} assigned{pending}"
            f"{self.feedback}"
        )
        self.ax.set_title(f"SMPL-X marker editor — {label}")
        self._update_marker_overlay()
        self.fig.canvas.draw_idle()

    def _select_from_text(self, text: str) -> None:
        value = text.strip()
        if not value:
            return
        try:
            index = int(value) - 1
        except ValueError:
            matches = [
                index for index, label in enumerate(self.labels) if label == value
            ]
            if not matches:
                matches = [
                    index
                    for index, label in enumerate(self.labels)
                    if label.endswith(value)
                ]
            if len(matches) != 1:
                self.status_artist.set_text(
                    f"Could not select unique marker: {value!r}"
                )
                self.fig.canvas.draw_idle()
                return
            index = matches[0]
        if 0 <= index < len(self.labels):
            self.current = index
            self.pending_duplicate = None
            self._refresh()

    def _visible_surface_vertex(self, event: Any) -> int | None:
        """Ray-cast through the click and return a vertex on the first hit face."""
        from mpl_toolkits.mplot3d import proj3d

        projected_xy = self.ax.transData.inverted().transform([event.x, event.y])
        projection = self.ax.get_proj()
        _, _, center_z = proj3d.proj_transform(*self.mesh_center, projection)
        target = np.asarray(
            proj3d.inv_transform(
                projected_xy[0],
                projected_xy[1],
                center_z,
                np.linalg.inv(projection),
            ),
            dtype=float,
        ).reshape(3)
        origin = self._camera_location()
        direction = target - origin
        direction_norm = np.linalg.norm(direction)
        if not np.isfinite(direction_norm) or direction_norm <= 1e-12:
            return None
        direction /= direction_norm

        # Vectorized Moller-Trumbore ray/triangle intersection. The minimum
        # positive ray distance is the visible surface, so rear faces can never
        # be selected through the body.
        ray = np.broadcast_to(direction, self.triangle_edge2.shape)
        pvec = np.cross(ray, self.triangle_edge2)
        determinant = np.einsum("ij,ij->i", self.triangle_edge1, pvec)
        candidate = np.abs(determinant) > 1e-12
        inverse = np.zeros_like(determinant)
        inverse[candidate] = 1.0 / determinant[candidate]
        tvec = origin - self.triangle_origins
        barycentric_u = inverse * np.einsum("ij,ij->i", tvec, pvec)
        qvec = np.cross(tvec, self.triangle_edge1)
        barycentric_v = inverse * np.einsum("ij,ij->i", ray, qvec)
        distance = inverse * np.einsum("ij,ij->i", self.triangle_edge2, qvec)
        epsilon = 1e-9
        candidate &= barycentric_u >= -epsilon
        candidate &= barycentric_v >= -epsilon
        candidate &= barycentric_u + barycentric_v <= 1.0 + epsilon
        candidate &= distance > epsilon
        hits = np.flatnonzero(candidate)
        if not len(hits):
            return None
        triangle_index = int(hits[np.argmin(distance[hits])])
        intersection = origin + distance[triangle_index] * direction
        face_vertices = self.mesh.faces[triangle_index]
        nearest = np.argmin(
            np.linalg.norm(self.mesh.vertices[face_vertices] - intersection, axis=1)
        )
        return int(face_vertices[nearest])

    def _on_surface_click(self, event: Any) -> None:
        key = (event.key or "").lower()
        if event.inaxes is not self.ax or event.button != 1 or "shift" not in key:
            return
        vertex = self._visible_surface_vertex(event)
        if vertex is None:
            self.feedback = "\n\nNo visible body surface under the click."
            self._refresh()
            return
        self.feedback = ""
        label = self.labels[self.current]
        conflicts = tuple(
            other
            for other, assigned_vertex in self.indices.items()
            if other != label and assigned_vertex == vertex
        )
        if conflicts:
            self.pending_duplicate = (label, vertex, conflicts)
            self._refresh()
            return
        self._assign(label, vertex, duplicate_confirmed=False)

    def _assign(self, label: str, vertex: int, duplicate_confirmed: bool) -> None:
        old_vertex = self.indices[label]
        old_source = self._source_for(label)
        self.undo_stack.append((label, old_vertex, old_source))
        self.indices[label] = int(vertex)
        set_index(self.layout, label, int(vertex))
        source = {
            "method": "manual_surface_click",
            "model_path": str(self.mesh.path),
            "model_sha256": self.mesh.sha256,
            "duplicate_vertex_confirmed": bool(duplicate_confirmed),
        }
        self.layout["marker_sources"][label] = source
        self.pending_duplicate = None
        self._save()
        self.next()

    def next(self, _event: Any = None) -> None:
        self.current = (self.current + 1) % len(self.labels)
        self.pending_duplicate = None
        self._refresh()

    def previous(self, _event: Any = None) -> None:
        self.current = (self.current - 1) % len(self.labels)
        self.pending_duplicate = None
        self._refresh()

    def clear_current(self, _event: Any = None) -> None:
        label = self.labels[self.current]
        old_vertex = self.indices[label]
        old_source = self._source_for(label)
        if old_vertex is None:
            return
        self.undo_stack.append((label, old_vertex, old_source))
        self.indices[label] = None
        set_index(self.layout, label, None)
        self.layout["marker_sources"].pop(label, None)
        self.pending_duplicate = None
        self._save()
        self._refresh()

    def undo(self, _event: Any = None) -> None:
        if not self.undo_stack:
            return
        label, old_vertex, old_source = self.undo_stack.pop()
        self.current = self.labels.index(label)
        self.indices[label] = old_vertex
        set_index(self.layout, label, old_vertex)
        if old_source is None:
            self.layout["marker_sources"].pop(label, None)
        else:
            self.layout["marker_sources"][label] = old_source
        self.pending_duplicate = None
        self._save()
        self._refresh()

    def _on_key(self, event: Any) -> None:
        key = event.key or ""
        if key.lower() == "n" or key == "right":
            self.next()
        elif key.lower() == "p" or key == "left":
            self.previous()
        elif key.lower() == "s":
            self.next()
        elif key.lower() == "u":
            self.undo()
        elif key.lower() == "c" or key == "delete":
            self.clear_current()
        elif key.lower() == "y" and self.pending_duplicate:
            label, vertex, _ = self.pending_duplicate
            self._assign(label, vertex, duplicate_confirmed=True)

    def show(self) -> None:
        self.plt.show()
