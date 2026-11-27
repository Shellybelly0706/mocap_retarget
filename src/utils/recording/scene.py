"""Build a MuJoCo tracking scene, optionally with a local neutral SMPL-X skin."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from utils.avatars.pose import anatomical_frames, quaternions_aligning_z
from utils.neon.frame_fit import TempleFitter, attach_temple_hinges, hinge_quaternion
from utils.neon.imu import imu_pose_in_cad, register_imu_world, sample_native_imu
from utils.neon.model import values
from utils.neon.mount import CAD_BRIDGE_M, attach_neon, load_mount, mount_transform
from utils.optitrack.skeleton import SKELETON_PARENTS
from utils.paths import ASSETS_DIR
from utils.recording.session import AlignedSession
from utils.smplx.fitted import load_fitted_motion
from utils.smplx.neutral import load_neutral_model, retarget_neutral


class RecordingScene:
    def __init__(
        self,
        session: AlignedSession,
        model_path: Path | None = None,
        mount_path: Path | None = None,
        use_imu: bool = False,
        fitted_result_path: Path | None = None,
        skeleton_source: str = "motive",
    ):
        if fitted_result_path and not model_path:
            raise ValueError("A fitted SMPL-X result requires its matching NPZ model")
        if skeleton_source not in {"motive", "smplx"}:
            raise ValueError("Skeleton source must be 'motive' or 'smplx'")
        if skeleton_source == "smplx" and not model_path:
            raise ValueError("The SMPL-X skeleton requires an SMPL-X avatar")
        self.session = session
        self.mount = load_mount(mount_path)
        self.skeleton_source = skeleton_source
        self.skeleton_label = (
            "SMPL-X fitted skeleton"
            if skeleton_source == "smplx" and fitted_result_path
            else (
                "SMPL-X retargeted skeleton"
                if skeleton_source == "smplx"
                else "Motive solved skeleton"
            )
        )
        self.skeleton_xray = skeleton_source == "smplx"
        self.label = (
            "MoSh++ fitted SMPL-X (marker-based reconstruction)"
            if fitted_result_path
            else (
                "SMPL-X neutral (visual retargeting)"
                if model_path
                else "OptiTrack skeleton"
            )
        )
        self.reference = min(
            int(np.searchsorted(session.times, 0.5)), len(session.times) - 1
        )
        self.xml = ET.Element("mujoco", model="aligned_egocentric_recording")
        ET.SubElement(self.xml, "compiler", angle="radian", inertiafromgeom="false")
        ET.SubElement(self.xml, "option", gravity="0 0 0")
        visual = ET.SubElement(self.xml, "visual")
        ET.SubElement(visual, "global", offwidth="1600", offheight="1200")
        ET.SubElement(visual, "map", znear="0.005", zfar="30")
        ET.SubElement(visual, "headlight", ambient="0.5 0.5 0.5", diffuse="0.7 0.7 0.7")
        asset = ET.SubElement(self.xml, "asset")
        ET.SubElement(
            asset,
            "texture",
            name="sky",
            type="skybox",
            builtin="gradient",
            width="256",
            height="256",
            rgb1="0.19 0.23 0.29",
            rgb2="0.06 0.08 0.12",
        )
        ET.SubElement(
            asset,
            "texture",
            name="grid",
            type="2d",
            builtin="checker",
            width="256",
            height="256",
            rgb1="0.16 0.19 0.23",
            rgb2="0.24 0.28 0.32",
        )
        ET.SubElement(
            asset, "material", name="floor", texture="grid", texrepeat="10 10"
        )
        world = ET.SubElement(self.xml, "worldbody")
        ET.SubElement(world, "light", pos="0 0 5", dir="0 0 -1")
        ET.SubElement(
            world,
            "geom",
            name="floor",
            type="plane",
            size="10 10 0.01",
            material="floor",
            contype="0",
            conaffinity="0",
        )
        self.tracks = []
        self.segment_geoms = []
        self.skeleton_geom_names = []
        bones = session.bones
        root_name = next(iter(bones))
        self.skeleton_names = []
        if skeleton_source == "motive":
            motive_names = list(bones)
            motive_indices = {name: index for index, name in enumerate(motive_names)}
            motive_parents = []
            for name in motive_names:
                parent = root_name if name == "Ab" else SKELETON_PARENTS.get(name)
                motive_parents.append(motive_indices.get(parent, -1))
            self._add_skeleton(
                world,
                "motive",
                motive_names,
                np.stack([bones[name][:, 4:7] for name in motive_names], axis=1),
                np.asarray(motive_parents),
                emphasized=fitted_result_path is not None,
            )
        self.marker_geoms = []
        for index, name in enumerate(session.marker_names):
            track = session.markers[:, index]
            body = self._body(world, f"marker_{index}", np.nan_to_num(track))
            geom_name = f"marker_geom_{index}"
            ET.SubElement(
                body,
                "geom",
                name=geom_name,
                type="sphere",
                size="0.005",
                rgba="1 0.55 0.15 1",
                group="3",
                contype="0",
                conaffinity="0",
            )
            self.marker_geoms.append((geom_name, np.isfinite(track).all(axis=1)))
        names = {name: index for index, name in enumerate(session.marker_names)}
        if not {"HeadTop", "HeadFront"} <= names.keys():
            raise ValueError("HeadTop and HeadFront are required to orient the glasses")
        head_top = session.markers[:, names["HeadTop"]]
        head_front = session.markers[:, names["HeadFront"]]
        if not np.isfinite(head_top).all() or not np.isfinite(head_front).all():
            raise ValueError(
                "Head markers have gaps; refusing to invent glasses orientations"
            )
        if model_path:
            if fitted_result_path:
                source_frames = np.asarray(
                    [int(row["mocap_source_frame"]) for row in session.mapping],
                    dtype=np.int64,
                )
                fitted = load_fitted_motion(
                    fitted_result_path,
                    model_path,
                    source_frames,
                    session.markers,
                    session.marker_names,
                )
                self.fitted_registration = {
                    "rotation": fitted.world_rotation,
                    "translation": fitted.world_translation,
                    "median_error_m": fitted.registration_median_error_m,
                    "p95_error_m": fitted.registration_p95_error_m,
                }
                avatar = fitted.avatar
                positions = fitted.positions
                rotations = fitted.rotations
                quats = fitted.quaternions
                scale = 1.0
            else:
                avatar = load_neutral_model(model_path)
                positions, rotations, scale = retarget_neutral(
                    avatar, bones, head_top, head_front, self.reference
                )
                quats = (
                    Rotation.from_matrix(rotations.reshape(-1, 3, 3))
                    .as_quat()[:, [3, 0, 1, 2]]
                    .reshape(len(positions), 55, 4)
                )
            if skeleton_source == "smplx":
                self._add_skeleton(
                    world,
                    "smplx",
                    [f"joint_{index:02d}" for index in range(55)],
                    positions,
                    avatar.parents,
                    emphasized=True,
                )
            joints = [
                self._body(
                    world,
                    f"smplx_joint_{index:02d}",
                    positions[:, index],
                    quats[:, index],
                )
                for index in range(55)
            ]
            skin = ET.SubElement(
                asset,
                "skin",
                name="smplx_neutral",
                vertex=values(avatar.vertices * scale),
                face=" ".join(map(str, avatar.faces.reshape(-1))),
                rgba="0.72 0.63 0.56 1",
                group="4",
            )
            for index in range(55):
                selected = np.flatnonzero(avatar.weights[:, index] > 1e-8)
                ET.SubElement(
                    skin,
                    "bone",
                    body=f"smplx_joint_{index:02d}",
                    bindpos=values(avatar.joints[index] * scale),
                    bindquat="1 0 0 0",
                    vertid=" ".join(map(str, selected)),
                    vertweight=values(avatar.weights[selected, index]),
                )
            head = joints[15]
            self.head_positions = positions[:, 15]
            self.head_rotations = rotations[:, 15]
            self.bridge_from_head = (avatar.nose_bridge - avatar.joints[15]) * scale
        else:
            # Canonical bind axes: +X left, +Y backward, +Z up.
            source_frame = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=float)
            head_frame = anatomical_frames(
                bones["LShoulder"][:, 4:7] - bones["RShoulder"][:, 4:7],
                head_top - bones["Head"][:, 4:7],
                head_front - bones["Head"][:, 4:7],
            )
            solved = Rotation.from_quat(bones["Head"][:, :4]).as_matrix()
            rotations = (
                solved
                @ solved[self.reference].T
                @ head_frame[self.reference]
                @ source_frame.T
            )
            quats = Rotation.from_matrix(rotations).as_quat()[:, [3, 0, 1, 2]]
            self.head_positions = bones["Head"][:, 4:7]
            self.head_rotations = rotations
            head = self._body(world, "head_reference", self.head_positions, quats)
            self.bridge_from_head = np.array([0, -0.09, 0.10])
            ET.SubElement(
                head,
                "geom",
                name="head_proxy",
                type="ellipsoid",
                pos="0 0 0.08",
                size="0.073 0.09 0.105",
                rgba="0.6 0.65 0.7 0.25",
                group="4",
                contype="0",
                conaffinity="0",
            )
        attach_neon(self.xml, head, self.bridge_from_head, self.mount)
        self.use_imu = use_imu
        self.imu_sampled = None
        self.imu_registration = None
        self.temple_fitter = None
        self.temple_fit = None
        if use_imu:
            stamps = np.array(
                [int(row["timestamp_ns"]) for row in session.mapping], dtype=np.int64
            )
            native_dir = Path(session.report["sources"]["neon_video"]["path"]).parent
            self.imu_sampled = sample_native_imu(native_dir, stamps)
            self.imu_world_rotations, self.imu_registration = register_imu_world(
                self.head_rotations, self.imu_sampled.rotations, session.times
            )
            self.imu_registration["head_frame"] = {
                "representation": self.label,
                "definition": "rendered head bind axes; solved OptiTrack Head with constant visual reference alignment",
                "visual_reference_frame": self.reference,
                "visual_reference_time_s": float(session.times[self.reference]),
            }
            reference = json.loads((ASSETS_DIR / "neon" / "geometry.json").read_text())[
                "camera_reference"
            ]
            _, self.rotation_cad_from_imu = imu_pose_in_cad(
                reference["position_cad_m"], reference["rotation_cad_from_opencv"]
            )
        if model_path:
            self.temple_fitter = TempleFitter(
                (avatar.vertices - avatar.joints[15]) * scale, avatar.faces
            )
            position, quaternion = self._mount_pose(self.reference)
            self.temple_fit = self.temple_fitter.fit(
                position, Rotation.from_quat(quaternion[[1, 2, 3, 0]]).as_matrix()
            )
            attach_temple_hinges(self.xml, self.temple_fit)
        self.model = mujoco.MjModel.from_xml_string(
            ET.tostring(self.xml, encoding="unicode")
        )
        self.data = mujoco.MjData(self.model)
        self.track_ids = [
            (self.model.body(name).mocapid[0], pos, quat)
            for name, pos, quat in self.tracks
        ]
        self.segment_geoms = [
            (self.model.geom(name).id, length) for name, length in self.segment_geoms
        ]
        self.marker_geoms = [
            (self.model.geom(name).id, valid) for name, valid in self.marker_geoms
        ]
        self.camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(self.camera)
        self.camera.lookat[:] = np.median(self.head_positions, axis=0) + [0, 0, -0.25]
        self.camera.distance = 2.2
        self.camera.azimuth = 150
        self.camera.elevation = -15
        self.options = mujoco.MjvOption()
        self.options.sitegroup[5] = 1
        self.options.sitegroup[4] = 1
        self.options.geomgroup[2] = int(
            model_path is None
            or fitted_result_path is not None
            or skeleton_source == "smplx"
        )
        self.options.geomgroup[3] = 0
        self.options.geomgroup[4] = 1
        self.options.skingroup[4] = 1
        self.options.flags[mujoco.mjtVisFlag.mjVIS_SKIN] = 1
        self.set_frame(0)

    def _add_skeleton(
        self,
        world,
        prefix: str,
        names: list[str],
        positions: np.ndarray,
        parents: np.ndarray,
        *,
        emphasized: bool,
    ):
        """Add one selected joint hierarchy to MuJoCo visualization group 2."""
        positions = np.asarray(positions, dtype=np.float64)
        parents = np.asarray(parents, dtype=np.int64)
        if positions.ndim != 3 or positions.shape[1:] != (len(names), 3):
            raise ValueError("Skeleton positions and names differ")
        if parents.shape != (len(names),):
            raise ValueError("Skeleton parents and names differ")
        self.skeleton_names = list(names)
        color = "0.02 0.15 1 1" if emphasized else "0.15 0.8 0.95 1"
        for index, name in enumerate(names):
            finger = prefix == "smplx" and index >= 25
            joint_radius = 0.0045 if finger else (0.010 if emphasized else 0.009)
            body = self._body(
                world, f"skeleton_{prefix}_node_{index:02d}", positions[:, index]
            )
            joint_geom = f"skeleton_{prefix}_joint_{index:02d}"
            ET.SubElement(
                body,
                "geom",
                name=joint_geom,
                type="sphere",
                size=str(joint_radius),
                rgba=color,
                group="2",
                contype="0",
                conaffinity="0",
            )
            self.skeleton_geom_names.append(joint_geom)
            parent = int(parents[index])
            if parent < 0:
                continue
            if parent >= index:
                raise ValueError(f"Invalid skeleton parent {parent} for joint {name}")
            vector = positions[:, index] - positions[:, parent]
            lengths = np.linalg.norm(vector, axis=1)
            if np.any(lengths < 1e-9):
                continue
            segment = self._body(
                world,
                f"skeleton_{prefix}_segment_{index:02d}",
                (positions[:, parent] + positions[:, index]) * 0.5,
                quaternions_aligning_z(vector),
            )
            geom_name = f"skeleton_{prefix}_link_{index:02d}"
            link_radius = 0.003 if finger else (0.006 if emphasized else 0.005)
            ET.SubElement(
                segment,
                "geom",
                name=geom_name,
                type="capsule",
                size=f"{link_radius} {lengths[0]/2}",
                rgba=color,
                group="2",
                contype="0",
                conaffinity="0",
            )
            self.skeleton_geom_names.append(geom_name)
            self.segment_geoms.append((geom_name, lengths / 2))

    def _body(self, world, name, positions, quats=None):
        if quats is None:
            quats = np.tile([1.0, 0, 0, 0], (len(positions), 1))
        self.tracks.append((name, positions, quats))
        return ET.SubElement(
            world,
            "body",
            name=name,
            mocap="true",
            pos=values(positions[0]),
            quat=values(quats[0]),
        )

    def set_frame(self, frame: int):
        self.frame = frame
        for mocap_id, positions, quaternions in self.track_ids:
            self.data.mocap_pos[mocap_id] = positions[frame]
            self.data.mocap_quat[mocap_id] = quaternions[frame]
        for geom_id, length in self.segment_geoms:
            self.model.geom_size[geom_id, 1] = length[frame]
        for geom_id, valid in self.marker_geoms:
            self.model.geom_rgba[geom_id, 3] = float(valid[frame])
        position, quaternion = self._mount_pose(frame)
        glasses = self.model.body("neon_glasses")
        self.model.body_pos[glasses.id] = position
        self.model.body_quat[glasses.id] = quaternion
        if self.temple_fit:
            for side in ("left", "right"):
                body = self.model.body(f"neon_temple_{side}")
                self.model.body_quat[body.id] = hinge_quaternion(
                    side, self.temple_fit[side]["opening_deg"]
                )
        mujoco.mj_forward(self.model, self.data)

    def _mount_pose(self, frame: int):
        if self.use_imu:
            rotation_delta = Rotation.from_euler(
                "xyz", self.mount.get("rotation_deg_xyz", [0, 0, 0]), degrees=True
            ).as_matrix()
            rotation_h_a = (
                self.head_rotations[frame].T
                @ self.imu_world_rotations[frame]
                @ self.rotation_cad_from_imu.T
                @ rotation_delta
            )
            position = (
                self.bridge_from_head
                + np.asarray(self.mount.get("translation_m", [0, 0, 0]))
                - rotation_h_a @ CAD_BRIDGE_M
            )
            quaternion = Rotation.from_matrix(rotation_h_a).as_quat()[[3, 0, 1, 2]]
        else:
            position, quaternion = mount_transform(self.bridge_from_head, self.mount)
        return position, quaternion

    def set_mount(self, settings: dict):
        mount_transform(
            self.bridge_from_head, settings
        )  # validate before changing state
        self.mount = settings
        if self.temple_fitter:
            position, quaternion = self._mount_pose(self.reference)
            self.temple_fit = self.temple_fitter.fit(
                position, Rotation.from_quat(quaternion[[1, 2, 3, 0]]).as_matrix()
            )
        self.set_frame(self.frame)

    def focus_head(self):
        self.camera.lookat[:] = self.data.body("neon_scene_camera_optical").xpos
        self.camera.distance = 0.45

    def save_visual_fit(self, path: Path):
        report = {
            "purpose": "approximate avatar/CAD visual fit, not calibrated extrinsics",
            "anchor_method": "eye height + nasal surface + CAD brow-bar standoff",
            "brow_bar_head_bind_m": self.bridge_from_head.tolist(),
            "mount_adjustment": self.mount,
            "orientation_mode": "registered IMU" if self.use_imu else "OptiTrack head",
            "geometry_fit_reference_frame": self.reference,
            "temples": self.temple_fit,
        }
        Path(path).write_text(json.dumps(report, indent=2) + "\n")

    def save_xml(self, path: Path):
        for name, positions, quaternions in self.tracks:
            body = self.xml.find(f"./worldbody/body[@name='{name}']")
            body.set("pos", values(positions[self.frame]))
            body.set("quat", values(quaternions[self.frame]))
        glasses = self.xml.find(".//body[@name='neon_glasses']")
        body = self.model.body("neon_glasses")
        glasses.set("pos", values(self.model.body_pos[body.id]))
        glasses.set("quat", values(self.model.body_quat[body.id]))
        if self.temple_fit:
            for side in ("left", "right"):
                name = f"neon_temple_{side}"
                element = self.xml.find(f".//body[@name='{name}']")
                element.set("quat", values(self.model.body(name).quat))
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        ET.indent(self.xml)
        ET.ElementTree(self.xml).write(path, encoding="utf-8", xml_declaration=True)
