"""One transport and frame index for an embedded MuJoCo viewport and Neon video."""

from __future__ import annotations

import json
import time
import tkinter as tk
from tkinter import ttk

import cv2
import mujoco
import numpy as np
from PIL import Image, ImageOps, ImageTk

from utils.synchronization.neon_video import read_video_frame


class RecordingPlayer:
    def __init__(self, scene):
        self.scene = scene
        self.session = scene.session
        self.index = 0
        self.playing = False
        self.closed = False
        self.seeking = False
        self.anchor_time = 0.0
        self.anchor_wall = time.monotonic()
        self.video_state = {}
        self.capture = cv2.VideoCapture(
            str(self.session.directory / "front_camera_aligned.mp4")
        )
        if not self.capture.isOpened():
            raise ValueError("Cannot open aligned front-camera video")
        self.root = tk.Tk()
        self.root.title("Egocentric recording | OptiTrack + Neon")
        self.root.geometry("1320x780")
        self.root.minsize(960, 640)
        self.renderer = mujoco.Renderer(scene.model, width=640, height=480)
        self.speed = tk.StringVar(value="1.0")
        self.loop = tk.BooleanVar(value=True)
        self.axis_visible = tk.BooleanVar(value=True)
        self.imu_frame_visible = tk.BooleanVar(value=True)
        self.imu_orientation = tk.BooleanVar(value=scene.use_imu)
        self.skeleton_visible = tk.BooleanVar(value=bool(scene.options.geomgroup[2]))
        self.markers_visible = tk.BooleanVar(value=False)
        headers = ttk.Frame(self.root)
        headers.pack(fill="x", padx=10, pady=5)
        headers.columnconfigure(0, weight=1)
        headers.columnconfigure(1, weight=1)
        ttk.Label(headers, text=scene.label).grid(row=0, column=0, sticky="w")
        ttk.Label(headers, text="Neon front camera").grid(row=0, column=1, sticky="w")
        panels = ttk.Frame(self.root)
        panels.pack(fill="both", expand=True, padx=10)
        panels.rowconfigure(0, weight=1)
        self.panels = []
        for column in range(2):
            panels.columnconfigure(column, weight=1, uniform="views")
            panel = tk.Canvas(panels, background="#111820", highlightthickness=0)
            panel.grid(row=0, column=column, sticky="nsew", padx=2)
            self.panels.append(panel)
        self.images = [None, None]
        self.items = [
            panel.create_image(0, 0, anchor="center") for panel in self.panels
        ]
        self.slider = ttk.Scale(
            self.root,
            from_=0,
            to=len(self.session.times) - 1,
            command=self._slider_seek,
        )
        self.slider.pack(fill="x", padx=15, pady=6)
        controls = ttk.Frame(self.root)
        controls.pack(fill="x", padx=10)
        for label, callback in (
            ("Restart", lambda: self.seek(0)),
            ("Previous", lambda: self.seek(self.index - 1)),
            ("Play / Pause", self.toggle),
            ("Next", lambda: self.seek(self.index + 1)),
        ):
            ttk.Button(controls, text=label, command=callback).pack(side="left", padx=2)
        ttk.Label(controls, text="Speed").pack(side="left", padx=(15, 3))
        speed = ttk.Combobox(
            controls,
            textvariable=self.speed,
            values=("0.25", "0.5", "1.0", "2.0"),
            width=5,
            state="readonly",
        )
        speed.pack(side="left")
        speed.bind("<<ComboboxSelected>>", lambda _: self._anchor())
        ttk.Checkbutton(controls, text="Loop", variable=self.loop).pack(
            side="left", padx=5
        )
        ttk.Button(controls, text="Focus head", command=self.focus_head).pack(
            side="left", padx=3
        )
        ttk.Button(controls, text="Glasses fit…", command=self.mount_editor).pack(
            side="left", padx=3
        )
        debug = ttk.Frame(self.root)
        debug.pack(fill="x", padx=12, pady=5)
        for label, variable in (
            ("Camera axes (RGB = X/Y/Z)", self.axis_visible),
            ("IMU frame (top bar)", self.imu_frame_visible),
            ("Solved skeleton", self.skeleton_visible),
            ("Raw markers", self.markers_visible),
        ):
            ttk.Checkbutton(
                debug, text=label, variable=variable, command=self.refresh
            ).pack(side="left", padx=5)
        ttk.Checkbutton(
            debug,
            text="IMU orientation",
            variable=self.imu_orientation,
            command=self.refresh,
            state="normal" if scene.imu_sampled is not None else "disabled",
        ).pack(side="left", padx=5)
        self.status = ttk.Label(self.root)
        self.status.pack(anchor="w", padx=15, pady=3)
        ttk.Label(
            self.root,
            text="Clock agreement unverified. Nose position is geometric; IMU heading is registered to OptiTrack. IMU origin is a nominal camera-relative reference.",
        ).pack(anchor="w", padx=15)
        ttk.Label(
            self.root,
            text="Left view: drag to orbit, Shift-drag to pan, scroll to zoom. Space: play/pause. ←/→: step. Home: restart.",
        ).pack(anchor="w", padx=15, pady=(2, 8))
        self.panels[0].bind("<ButtonPress-1>", self._mouse_down)
        self.panels[0].bind("<B1-Motion>", self._mouse_move)
        self.panels[0].bind("<MouseWheel>", lambda e: self._zoom(-e.delta / 120 * 0.08))
        self.panels[0].bind("<Button-4>", lambda _: self._zoom(-0.08))
        self.panels[0].bind("<Button-5>", lambda _: self._zoom(0.08))
        self.root.bind("<space>", lambda _: self.toggle())
        self.root.bind("<Left>", lambda _: self.seek(self.index - 1))
        self.root.bind("<Right>", lambda _: self.seek(self.index + 1))
        self.root.bind("<Home>", lambda _: self.seek(0))
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.update_idletasks()
        self.refresh()
        self.root.after(16, self._tick)

    def _anchor(self):
        self.anchor_time = self.session.times[self.index]
        self.anchor_wall = time.monotonic()

    def toggle(self):
        if not self.playing and self.index == len(self.session.times) - 1:
            self.index = 0
        self.playing = not self.playing
        self._anchor()
        self.refresh()

    def seek(self, index):
        self.playing = False
        self.index = int(np.clip(index, 0, len(self.session.times) - 1))
        self._anchor()
        self.refresh()

    def _slider_seek(self, value):
        if not self.seeking:
            self.seek(round(float(value)))

    def _tick(self):
        if self.closed:
            return
        if self.playing:
            target = self.anchor_time + (time.monotonic() - self.anchor_wall) * float(
                self.speed.get()
            )
            if target > self.session.times[-1]:
                if self.loop.get():
                    self.index = 0
                    self._anchor()
                else:
                    self.index = len(self.session.times) - 1
                    self.playing = False
            else:
                self.index = max(
                    0,
                    int(np.searchsorted(self.session.times, target, side="right") - 1),
                )
            self.refresh()
        self.root.after(16, self._tick)

    def refresh(self):
        self.scene.options.sitegroup[5] = int(self.axis_visible.get())
        self.scene.options.sitegroup[4] = int(self.imu_frame_visible.get())
        self.scene.use_imu = self.imu_orientation.get()
        self.scene.options.geomgroup[2] = int(self.skeleton_visible.get())
        self.scene.options.geomgroup[3] = int(self.markers_visible.get())
        self.scene.set_frame(self.index)
        self.renderer.update_scene(
            self.scene.data, camera=self.scene.camera, scene_option=self.scene.options
        )
        left = Image.fromarray(self.renderer.render())
        video = read_video_frame(self.capture, self.index, self.video_state)
        right = Image.fromarray(cv2.cvtColor(video, cv2.COLOR_BGR2RGB))
        for i, source in enumerate((left, right)):
            panel = self.panels[i]
            width, height = max(1, panel.winfo_width()), max(1, panel.winfo_height())
            self.images[i] = ImageTk.PhotoImage(
                ImageOps.contain(source, (width, height))
            )
            panel.itemconfigure(self.items[i], image=self.images[i])
            panel.coords(self.items[i], width / 2, height / 2)
        self.seeking = True
        self.slider.set(self.index)
        self.seeking = False
        row = self.session.mapping[self.index]
        imu_status = (
            " | IMU orientation" if self.scene.use_imu else " | OptiTrack orientation"
        )
        if self.scene.use_imu:
            imu_status += f"; head/IMU residual {self.scene.imu_registration['per_frame_residual_degrees'][self.index]:.2f}°"
        self.status.configure(
            text=f"{'Playing' if self.playing else 'Paused'} | {self.index+1}/{len(self.session.times)} | {self.session.times[self.index]:.3f}s | source Neon {row['neon_source_frame']}, Motive {row['mocap_source_frame']} | Motive − Neon {float(row['mocap_delta_ms']):+.3f}ms"
            + imu_status
        )

    def focus_head(self):
        self.scene.focus_head()
        self.refresh()

    def _mouse_down(self, event):
        self.mouse = (event.x, event.y)

    def _mouse_move(self, event):
        previous = getattr(self, "mouse", (event.x, event.y))
        width = max(1, self.panels[0].winfo_width())
        height = max(1, self.panels[0].winfo_height())
        action = (
            mujoco.mjtMouse.mjMOUSE_MOVE_V
            if event.state & 1
            else mujoco.mjtMouse.mjMOUSE_ROTATE_V
        )
        mujoco.mjv_moveCamera(
            self.scene.model,
            action,
            (event.x - previous[0]) / width,
            (event.y - previous[1]) / height,
            self.renderer.scene,
            self.scene.camera,
        )
        self.mouse = (event.x, event.y)
        self.refresh()

    def _zoom(self, amount):
        mujoco.mjv_moveCamera(
            self.scene.model,
            mujoco.mjtMouse.mjMOUSE_ZOOM,
            0,
            amount,
            self.renderer.scene,
            self.scene.camera,
        )
        self.refresh()

    def mount_editor(self):
        self.playing = False
        window = tk.Toplevel(self.root)
        window.title("Provisional glasses mount")
        ttk.Label(
            window,
            text="Translation in head bind axes: X left, Y backward, Z up.\nRotation in CAD local axes. Inspect from several angles.",
        ).grid(row=0, column=0, columnspan=3, padx=12, pady=8)
        variables = []
        values = [
            v * 1000 for v in self.scene.mount.get("translation_m", [0, 0, 0])
        ] + self.scene.mount.get("rotation_deg_xyz", [0, 0, 0])
        for index, (label, value) in enumerate(
            zip(
                ("X mm", "Y mm", "Z mm", "RX degrees", "RY degrees", "RZ degrees"),
                values,
                strict=True,
            )
        ):
            variable = tk.DoubleVar(value=value)
            variables.append(variable)
            ttk.Label(window, text=label).grid(row=index + 1, column=0, padx=8, pady=3)
            ttk.Spinbox(
                window, from_=-180, to=180, increment=1, textvariable=variable, width=12
            ).grid(row=index + 1, column=1, padx=8)
        status = ttk.Label(window)

        def apply(save=False):
            try:
                settings = {
                    "translation_m": [v.get() / 1000 for v in variables[:3]],
                    "rotation_deg_xyz": [v.get() for v in variables[3:]],
                    "purpose": "visual nose-bridge fit; not calibrated camera-head extrinsics",
                    "avatar": self.scene.label,
                }
                self.scene.set_mount(settings)
                self.refresh()
                if save:
                    path = self.session.directory / "neon_mount.json"
                    path.write_text(json.dumps(settings, indent=2) + "\n")
                    status.configure(text=f"Saved {path.name}; reload with --mount")
            except (ValueError, tk.TclError) as exc:
                status.configure(text=str(exc))

        ttk.Button(window, text="Apply", command=apply).grid(row=7, column=0, pady=8)
        ttk.Button(window, text="Apply and save", command=lambda: apply(True)).grid(
            row=7, column=1
        )
        status.grid(row=8, column=0, columnspan=3, padx=8, pady=5)

    def close(self):
        if not self.closed:
            self.closed = True
            self.capture.release()
            self.renderer.close()
            self.root.destroy()

    def run(self):
        try:
            self.root.mainloop()
        finally:
            self.close()
