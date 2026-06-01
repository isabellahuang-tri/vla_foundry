"""Rerun visualizer for MMT TAR files.

Example:
    python mmt_vis.py \
        --input_path s3://tri-mmt-data/lpp_data/vla_foundry/20251028_paper_towel/shards/ \
        --data_params vla_foundry/config_presets/data/mmt/mmt_data_params.yaml
"""

import argparse
import io
import json
import logging
import posixpath
import tarfile
from collections.abc import Iterator, Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath

import fsspec
import numpy as np
import rerun as rr
import torch
from PIL import Image, ImageDraw
from tqdm import tqdm

from vla_foundry.data.robotics.cv_utils import (
    intrinsics_3x3_to_4,
    scale_intrinsics_4_for_resize_and_crop,
    transform_points_to_camera_frame,
)
from vla_foundry.data.robotics.normalization import RoboticsNormalizer
from vla_foundry.data.robotics.utils import invert_homogeneous_transform, xyzrpy_to_T
from vla_foundry.inference.robotics.mmt.action_handlers import (
    ReferenceFrame,
    _get_position_cmd_config,
    is_position_action_field,
)
from vla_foundry.params.robotics.normalization_params import FieldNormalizationParams, NormalizationParams
from vla_foundry.visualizers import visualizer as vz

BALL_RADIUS: float = 0.004
VELOCITY_COMMAND_DT_S: float = 0.2

NDArray = np.ndarray


@dataclass(frozen=True)
class PredictionBundle:
    """Denormalized model prediction fields for one sample."""

    label: str
    action_fields: tuple[str, ...]
    lowdim: Mapping[str, NDArray]


VELOCITY_ACTION_FIELDS: Mapping[str, tuple[str, str]] = {
    "left_arm_action": ("left", "eef"),
    "right_arm_action": ("right", "eef"),
    "left_arm_action_at_gripper_tip": ("left", "gripper_tip"),
    "right_arm_action_at_gripper_tip": ("right", "gripper_tip"),
}

LEGACY_BIMANUAL_VELOCITY_FIELDS: Mapping[str, tuple[str, tuple[str, str]]] = {
    "arm_action": ("eef", ("left_arm_action", "right_arm_action")),
    "arm_action_at_gripper_tip": (
        "gripper_tip",
        ("left_arm_action_at_gripper_tip", "right_arm_action_at_gripper_tip"),
    ),
}


def setup_logging(level: int = logging.INFO) -> None:
    """Configure application logging.

    Args:
        level: Logging level (e.g., logging.INFO, logging.DEBUG). Defaults to logging.INFO.
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def is_tar_path(path: str) -> bool:
    """Return True if path points at a TAR-like archive name."""
    return path.lower().endswith((".tar", ".tar.gz", ".tgz"))


def list_tar_targets(input_path: str, recursive: bool = True) -> list[str]:
    """List TAR files from a local path, local directory, S3 URI, or S3 prefix."""
    fs, fs_path = fsspec.core.url_to_fs(input_path)
    if fs.isfile(fs_path):
        if not is_tar_path(fs_path):
            raise ValueError(f"Expected a TAR file, got: {input_path}")
        return [fs.unstrip_protocol(fs_path)]

    try:
        candidates = fs.find(fs_path) if recursive else fs.ls(fs_path, detail=False)
    except FileNotFoundError:
        logging.warning("Input path does not exist: %s", input_path)
        return []

    targets = [fs.unstrip_protocol(str(path)) for path in candidates if is_tar_path(str(path))]
    return sorted(targets)


def is_depth(name: str) -> bool:
    """Return True if the filename corresponds to a depth file."""
    n = name.lower()
    return (n.endswith(".png")) and ("depth" in n)


def is_image(name: str) -> bool:
    """Return True if the filename corresponds to an image file."""
    n = name.lower()
    return any(n.endswith(ext) for ext in (".jpg", ".jpeg", ".png")) and ("rgb" in n)


def is_lowdim(name: str) -> bool:
    """Return True if the filename corresponds to a low-dimensional npz payload."""
    n = name.lower()
    return n.endswith(".npz") and ("lowdim" in n)


class FileNameParser:
    """Parse member names from a TAR into (sample_id, data_type, ext).

    Expected format: "<id>.<data_type>.<ext>" (no leading directories).
    """

    @staticmethod
    def parse(name: str) -> tuple[str, str, str]:
        base = PurePosixPath(name).name  # Strip directories if present.
        parts = base.split(".")
        if len(parts) != 3:
            raise ValueError(f"Unexpected name format: {name}")
        sample_id, data_type, ext = parts
        return sample_id, data_type, ext


@dataclass
class NormalizerBundle:
    """Container for normalization configuration and utilities."""

    norm_dict: Mapping[str, object]
    normalizer: RoboticsNormalizer
    action_fields: tuple[str, ...] = ()

    @classmethod
    def from_paths(cls, data_params_path: str, stats_json_uri: str) -> "NormalizerBundle":
        """Create a NormalizerBundle from local YAML and (possibly remote) JSON stats."""
        from draccus.cfgparsing import load_config

        with open(data_params_path, encoding="utf-8") as f:
            loaded_config = load_config(f, file=data_params_path) or {}

        if not isinstance(loaded_config, Mapping):
            raise ValueError(f"Expected YAML mapping in {data_params_path}, got {type(loaded_config)}")

        data_params = loaded_config.get("data", loaded_config)
        if not isinstance(data_params, Mapping):
            raise ValueError(f"Expected data config mapping in {data_params_path}, got {type(data_params)}")

        # Load normalization stats (JSON may be local or remote via fsspec).
        fs, stats_json_path = fsspec.core.url_to_fs(stats_json_uri)
        with fs.open(stats_json_path, "r") as f:
            norm_dict = json.load(f)

        normalization_config = dict(data_params.get("normalization", {}) or {})
        raw_field_configs = normalization_config.get("field_configs")
        if isinstance(raw_field_configs, Mapping):
            normalization_config["field_configs"] = {
                str(field_name): (
                    field_config
                    if isinstance(field_config, FieldNormalizationParams)
                    else FieldNormalizationParams(**field_config)
                )
                for field_name, field_config in raw_field_configs.items()
            }

        normalization_params = NormalizationParams(**normalization_config)
        normalizer = RoboticsNormalizer(normalization_params=normalization_params, statistics_data=[norm_dict])
        action_fields = tuple(str(field_name) for field_name in (data_params.get("action_fields") or ()))
        return cls(norm_dict=norm_dict, normalizer=normalizer, action_fields=action_fields)

    def maybe_denormalize(self, key: str, arr: NDArray) -> NDArray:
        """Denormalize array if stats exist for key; otherwise return unchanged."""
        if key in self.norm_dict:
            tens = torch.from_numpy(arr)
            out = self.normalizer.denormalize_tensor(tens, key).numpy()
            return out
        return arr


class TarReader:
    """Stream TAR files from local or remote storage and assemble per-sample dictionaries."""

    def __init__(self, normalizer_bundle: NormalizerBundle, denormalize_lowdim: bool = True):
        self._norm = normalizer_bundle
        self._denormalize_lowdim = denormalize_lowdim

    def iter_samples_from_tar(self, tar_uri: str) -> Iterator[tuple[str, dict[str, object]]]:
        """Yield (sample_id, sample_dict) for each sample found in a single TAR."""
        data_dict: dict[str, dict[str, object]] = {}
        logging.debug("Opening TAR: %s", tar_uri)

        fs, tar_path = fsspec.core.url_to_fs(tar_uri)
        with fs.open(tar_path, "rb") as fo, tarfile.open(fileobj=fo, mode="r|*") as tf:
            for member in tf:
                if not member.isfile():
                    continue

                try:
                    sample_id, data_type, ext = FileNameParser.parse(member.name)
                except ValueError:
                    logging.debug("Skipping unexpected member name: %s", member.name)
                    continue

                ef = tf.extractfile(member)
                if ef is None:
                    continue

                raw = ef.read()
                entry = data_dict.setdefault(sample_id, {})

                try:
                    self._ingest_member(entry, member.name, data_type, ext, raw)
                except Exception as exc:  # Be resilient to partial TAR corruption.
                    logging.warning("Failed to ingest %s: %s", member.name, exc)

        yield from data_dict.items()

    def _ingest_member(
        self,
        entry: MutableMapping[str, object],
        name: str,
        data_type: str,
        ext: str,
        raw: bytes,
    ) -> None:
        if is_lowdim(name):
            self._ingest_lowdim(entry, data_type, raw)
        elif is_image(name):
            self._ingest_image(entry, data_type, raw)
        elif ext.lower() == "json":
            self._ingest_json(entry, data_type, raw)
        elif is_depth(name):
            self._ingest_depth(entry, data_type, raw)
        else:
            # Unknown payload; ignore.
            logging.debug("Ignoring payload: %s", name)

    def _ingest_lowdim(self, entry: MutableMapping[str, object], data_type: str, raw: bytes) -> None:
        with np.load(io.BytesIO(raw), allow_pickle=True) as npz:
            low_dim = {k: np.array(npz[k]) for k in npz.files}

        # Denormalize fields when we have stats.
        for field_name, array in list(low_dim.items()):
            try:
                low_dim[field_name] = (
                    self._norm.maybe_denormalize(field_name, array) if self._denormalize_lowdim else array
                )
            except Exception as exc:
                logging.debug("Denormalization failed for %s: %s", field_name, exc)

        entry[data_type] = low_dim  # Usually "lowdim".

    def _ingest_image(self, entry: MutableMapping[str, object], data_type: str, raw: bytes) -> None:
        if "rgb" not in data_type.lower():
            raise ValueError(f"Unknown image type in name: {data_type}")
        im = Image.open(io.BytesIO(raw)).convert("RGB")
        img_data = entry.setdefault("img_data", {})
        img_data[data_type] = np.asarray(im)

    def _ingest_json(self, entry: MutableMapping[str, object], data_type: str, raw: bytes) -> None:
        entry[data_type] = json.loads(raw.decode("utf-8"))

    def _ingest_depth(self, entry: MutableMapping[str, object], data_type: str, raw: bytes) -> None:
        depth = np.asarray(Image.open(io.BytesIO(raw)))
        img_data = entry.setdefault("img_data", {})
        img_data[data_type] = depth


class FrameAssembler:
    """Compute and package pose frames for visualization."""

    def __init__(self):
        # Fixed orientation tweak for head frame (kept from original script).
        self._R_head = np.array(
            [
                [0, 0, 1, 0],
                [-1, 0, 0, 0],
                [0, -1, 0, 0],
                [0, 0, 0, 1],
            ]
        )

    @staticmethod
    def _pose6_from_field(field_name: str, values: NDArray) -> NDArray:
        arr = np.asarray(values)
        if arr.ndim != 2:
            raise ValueError(f"'{field_name}' must be a 2D array, got shape {arr.shape}")
        if arr.shape[1] >= 7:
            return arr[:, 1:7]
        if arr.shape[1] >= 6:
            return arr[:, :6]
        raise ValueError(f"'{field_name}' must contain at least 6 pose values, got shape {arr.shape}")

    @staticmethod
    def _chest_head_pose6(values: NDArray) -> NDArray:
        arr = np.asarray(values)
        if arr.ndim != 2:
            raise ValueError(f"'chest_T_head_pose' must be a 2D array, got shape {arr.shape}")
        if arr.shape[1] >= 6:
            return arr[:, :6]
        if arr.shape[1] == 2:
            pose6 = np.zeros((arr.shape[0], 6), dtype=arr.dtype)
            pose6[:, 4:6] = arr
            return pose6
        raise ValueError(
            "'chest_T_head_pose' must contain either [pitch, yaw] or [x, y, z, roll, pitch, yaw], "
            f"got shape {arr.shape}"
        )

    @staticmethod
    def _chassis_chest_pose6(values: NDArray) -> NDArray:
        arr = np.asarray(values)
        if arr.ndim != 2:
            raise ValueError(f"'chassis_T_chest_pose' must be a 2D array, got shape {arr.shape}")
        if arr.shape[1] >= 6:
            return arr[:, :6]
        if arr.shape[1] == 1:
            pose6 = np.zeros((arr.shape[0], 6), dtype=arr.dtype)
            pose6[:, 2] = arr[:, 0]
            return pose6
        raise ValueError(
            "'chassis_T_chest_pose' must contain either [z] or [x, y, z, roll, pitch, yaw], "
            f"got shape {arr.shape}"
        )

    def _chest_eef_pose6(self, lowdim: Mapping[str, NDArray]) -> tuple[NDArray, NDArray]:
        left_eef = lowdim.get("chest_T_left_eef_pose")
        right_eef = lowdim.get("chest_T_right_eef_pose")
        if left_eef is not None and right_eef is not None:
            return (
                self._pose6_from_field("chest_T_left_eef_pose", left_eef),
                self._pose6_from_field("chest_T_right_eef_pose", right_eef),
            )

        chest_T_eef = lowdim.get("chest_T_eef_pose")
        if chest_T_eef is None:
            raise ValueError(
                "None of split 'chest_T_left_eef_pose'/'chest_T_right_eef_pose' or fallback "
                "'chest_T_eef_pose' found in lowdim data"
            )
        chest_T_eef = np.asarray(chest_T_eef)
        if chest_T_eef.ndim != 2 or chest_T_eef.shape[1] < 14:
            raise ValueError(f"'chest_T_eef_pose' must be a 2D array with at least 14 values, got {chest_T_eef.shape}")
        return chest_T_eef[:, 1:7], chest_T_eef[:, 8:14]

    def _chassis_eef_transforms(
        self,
        lowdim: Mapping[str, NDArray],
        chassis_T_chest_T: NDArray,
        pose_flip: NDArray,
    ) -> tuple[NDArray, NDArray]:
        left_eef = lowdim.get("chassis_T_left_eef_pose")
        right_eef = lowdim.get("chassis_T_right_eef_pose")
        if left_eef is not None and right_eef is not None:
            return (
                xyzrpy_to_T(self._pose6_from_field("chassis_T_left_eef_pose", left_eef)) @ pose_flip,
                xyzrpy_to_T(self._pose6_from_field("chassis_T_right_eef_pose", right_eef)) @ pose_flip,
            )

        chassis_T_eef = lowdim.get("chassis_T_eef_pose")
        if chassis_T_eef is not None:
            chassis_T_eef = np.asarray(chassis_T_eef)
            if chassis_T_eef.ndim != 2 or chassis_T_eef.shape[1] < 14:
                raise ValueError(
                    f"'chassis_T_eef_pose' must be a 2D array with at least 14 values, got {chassis_T_eef.shape}"
                )
            return (
                xyzrpy_to_T(chassis_T_eef[:, 1:7]) @ pose_flip,
                xyzrpy_to_T(chassis_T_eef[:, 8:14]) @ pose_flip,
            )

        left_pose6, right_pose6 = self._chest_eef_pose6(lowdim)
        chest_T_left = xyzrpy_to_T(left_pose6) @ pose_flip
        chest_T_right = xyzrpy_to_T(right_pose6) @ pose_flip
        return chassis_T_chest_T @ chest_T_left, chassis_T_chest_T @ chest_T_right

    @staticmethod
    def _camera_transform(lowdim: Mapping[str, NDArray], fallback: NDArray) -> NDArray:
        for field_name in ("extrinsics.rgb", "extrinsics.depth"):
            values = lowdim.get(field_name)
            if values is None:
                continue
            arr = np.asarray(values)
            if arr.ndim == 3 and arr.shape[1:] == (4, 4):
                return arr
            if arr.ndim == 2 and arr.shape == (4, 4):
                return np.tile(arr[None, :, :], (fallback.shape[0], 1, 1))
            raise ValueError(f"'{field_name}' must have shape (T, 4, 4) or (4, 4), got {arr.shape}")
        return fallback

    def chassis_frame_poses(self, lowdim: Mapping[str, NDArray]) -> dict[str, NDArray]:
        """Return mapping of chassis-anchored frame trajectories."""
        chest_T_head = lowdim.get("chest_T_head_pose")
        chassis_T_chest = lowdim.get("chassis_T_chest_pose")

        if chest_T_head is None:
            raise ValueError("'chest_T_head_pose' not found in lowdim data")
        if chassis_T_chest is None:
            raise ValueError("'chassis_T_chest_pose' not found in lowdim data")

        chassis_T_chest_T = xyzrpy_to_T(self._chassis_chest_pose6(chassis_T_chest))
        chest_T_head_T = xyzrpy_to_T(self._chest_head_pose6(chest_T_head))
        chassis_T_head_T = chassis_T_chest_T @ chest_T_head_T
        pose_flip = xyzrpy_to_T([0, 0, 0, 0, np.pi, 0])
        chassis_T_left, chassis_T_right = self._chassis_eef_transforms(lowdim, chassis_T_chest_T, pose_flip)
        chassis_T_head_camera = self._camera_transform(lowdim, chassis_T_head_T @ self._R_head)

        frames = {
            "chassis/left_eef": chassis_T_left,
            "chassis/right_eef": chassis_T_right,
            "chassis/head": chassis_T_head_T @ self._R_head,
            "chassis/head_camera": chassis_T_head_camera,
            "chassis/chest": chassis_T_chest_T,
        }

        for side in ("left", "right"):
            chassis_field = f"chassis_T_{side}_gripper_tip_pose"
            if chassis_field in lowdim:
                frames[f"chassis/{side}_gripper_tip"] = xyzrpy_to_T(
                    self._pose6_from_field(chassis_field, lowdim[chassis_field])
                ) @ pose_flip

        chassis_gripper_tip = lowdim.get("chassis_T_gripper_tip_pose")
        if chassis_gripper_tip is not None:
            arr = np.asarray(chassis_gripper_tip)
            if arr.ndim == 2 and arr.shape[1] >= 14:
                frames.setdefault("chassis/left_gripper_tip", xyzrpy_to_T(arr[:, 1:7]) @ pose_flip)
                frames.setdefault("chassis/right_gripper_tip", xyzrpy_to_T(arr[:, 8:14]) @ pose_flip)

        for side in ("left", "right"):
            chest_field = f"chest_T_{side}_gripper_tip_pose"
            frame_path = f"chassis/{side}_gripper_tip"
            if frame_path not in frames and chest_field in lowdim:
                chest_T_gripper_tip = xyzrpy_to_T(self._pose6_from_field(chest_field, lowdim[chest_field])) @ pose_flip
                frames[frame_path] = chassis_T_chest_T @ chest_T_gripper_tip

        chest_gripper_tip = lowdim.get("chest_T_gripper_tip_pose")
        if chest_gripper_tip is not None:
            arr = np.asarray(chest_gripper_tip)
            if arr.ndim == 2 and arr.shape[1] >= 14:
                frames.setdefault(
                    "chassis/left_gripper_tip",
                    chassis_T_chest_T @ (xyzrpy_to_T(arr[:, 1:7]) @ pose_flip),
                )
                frames.setdefault(
                    "chassis/right_gripper_tip",
                    chassis_T_chest_T @ (xyzrpy_to_T(arr[:, 8:14]) @ pose_flip),
                )

        return frames


class Plotter:
    """Logging helpers that call into the visualization backend."""

    @staticmethod
    def log_images(path: str, img_data: Mapping[str, NDArray]) -> None:
        vz.log_images(path, img_data)

    @staticmethod
    def log_coordinate_frames(frame_poses: Mapping[str, NDArray], axis_length: float = 0.3) -> None:
        for frame_name, Ts in frame_poses.items():
            for i, T in enumerate(Ts):
                rr.set_time("sample", sequence=i)
                vz.log_pose(frame_name, T[:3, 3], T[:3, :3], axis_length=axis_length)

    @staticmethod
    def _as_2d_array(field_name: str, values: NDArray) -> NDArray | None:
        arr = np.asarray(values)
        if arr.ndim == 1:
            arr = arr[None, :]
        if arr.ndim != 2:
            logging.warning("Skipping %s command visualization: expected 2D array, got %s", field_name, arr.shape)
            return None
        return arr

    @staticmethod
    def _extract_xyz(field_name: str, values: NDArray) -> NDArray | None:
        arr = Plotter._as_2d_array(field_name, values)
        if arr is None:
            return None

        dim = arr.shape[1]
        if dim >= 7:
            return arr[:, 1:4]
        if dim == 6:
            return arr[:, :3]
        if dim == 4:
            return arr[:, 1:4]
        if dim == 3:
            return arr[:, :3]

        logging.warning("Skipping %s command visualization: expected at least 3 xyz values, got %d", field_name, dim)
        return None

    @staticmethod
    def _log_command_trajectory(path: str, pts: NDArray) -> None:
        if pts.size == 0:
            return
        color = np.asarray(Plotter._trajectory_color(path, 0), dtype=np.uint8)
        vz.log_points3d(
            f"{path}/points",
            pts,
            radii=np.ones(pts.shape[0]) * BALL_RADIUS,
            colors=np.tile(color[None, :], (pts.shape[0], 1)),
        )
        if pts.shape[0] > 1:
            vz.log_line_strips3d(
                f"{path}/path",
                pts,
                colors=color[None, :],
                radii=np.asarray([BALL_RADIUS * 0.5], dtype=np.float32),
            )

    @staticmethod
    def _integrate_velocity_commands(anchor: NDArray, velocities: NDArray) -> NDArray:
        displacements = velocities * VELOCITY_COMMAND_DT_S
        endpoints = anchor + np.cumsum(displacements, axis=0)
        if endpoints.size == 0:
            return np.empty((0, 3), dtype=velocities.dtype)
        start = endpoints[0] - displacements[0]
        return np.concatenate([start[None, :], endpoints], axis=0)

    @staticmethod
    def _overlay_base_image(image: NDArray) -> NDArray:
        if image.ndim == 3 and image.shape[2] >= 3:
            if image.dtype == np.uint8:
                return np.array(image[:, :, :3], copy=True)
            return np.clip(image[:, :, :3], 0, 255).astype(np.uint8)

        depth = np.asarray(image)
        valid = np.isfinite(depth) & (depth > 0)
        if not np.any(valid):
            normalized = np.zeros(depth.shape, dtype=np.uint8)
        else:
            lo, hi = np.percentile(depth[valid], [2, 98])
            if hi <= lo:
                hi = lo + 1.0
            normalized = np.clip((depth.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0)
            normalized = (normalized * 255).astype(np.uint8)
        return np.repeat(normalized[:, :, None], 3, axis=2)

    @staticmethod
    def _trajectory_color(path: str, index: int) -> tuple[int, int, int]:
        if "/pred/" in path:
            if "position_command" in path:
                return (255, 80, 80) if "left" in path else (255, 180, 60)
            if "action" in path:
                return (255, 70, 70) if "left" in path else (255, 165, 40)
        if "/gt/" in path:
            if "position_command" in path:
                return (46, 204, 113) if "left" in path else (52, 152, 219)
            if "action" in path:
                return (230, 126, 34) if "left" in path else (155, 89, 182)
        if "position_command" in path:
            return (46, 204, 113) if "left" in path else (52, 152, 219)
        if "action" in path:
            return (230, 126, 34) if "left" in path else (155, 89, 182)
        palette = ((241, 196, 15), (26, 188, 156), (231, 76, 60), (149, 165, 166))
        return palette[index % len(palette)]

    @staticmethod
    def draw_projected_trajectories(
        image: NDArray,
        intrinsics: NDArray,
        camera_frame_trajectories: Mapping[str, NDArray],
    ) -> NDArray:
        out = Plotter._overlay_base_image(image)
        if not camera_frame_trajectories:
            return out

        fx, fy, cx, cy = np.asarray(intrinsics).reshape(4)
        pil_image = Image.fromarray(out)
        draw = ImageDraw.Draw(pil_image)
        height, width = out.shape[:2]
        total_visible = 0

        for i, (path, pts) in enumerate(camera_frame_trajectories.items()):
            pts = np.asarray(pts)
            if pts.ndim != 2 or pts.shape[1] != 3 or pts.size == 0:
                continue

            z = pts[:, 2]
            valid = np.isfinite(z) & (z > 0)
            if not np.any(valid):
                continue

            projected = np.full((pts.shape[0], 2), np.nan, dtype=np.float32)
            projected[valid, 0] = fx * (pts[valid, 0] / z[valid]) + cx
            projected[valid, 1] = fy * (pts[valid, 1] / z[valid]) + cy
            in_bounds = (
                valid
                & (projected[:, 0] >= 0)
                & (projected[:, 0] < width)
                & (projected[:, 1] >= 0)
                & (projected[:, 1] < height)
            )
            if not np.any(in_bounds):
                continue

            color = Plotter._trajectory_color(path, i)
            visible_indices = np.flatnonzero(in_bounds)
            coords = np.zeros((pts.shape[0], 2), dtype=np.int32)
            coords[visible_indices] = np.round(projected[visible_indices]).astype(np.int32)
            total_visible += len(visible_indices)

            for a, b in zip(visible_indices[:-1], visible_indices[1:], strict=False):
                if b != a + 1:
                    continue
                draw.line([tuple(coords[a]), tuple(coords[b])], fill=color, width=2)

            for j in visible_indices:
                radius = 5 if j == 0 else 3
                x, y = coords[j]
                draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color, outline=(0, 0, 0))

        if total_visible == 0:
            logging.debug("No projected trajectory points landed inside image bounds")
        return np.asarray(pil_image)

    @staticmethod
    def _points_in_chassis_frame(
        field_name: str,
        pts: NDArray,
        source_frame: str,
        frame_poses: Mapping[str, NDArray],
        start_index: int = 0,
    ) -> NDArray | None:
        if source_frame == "chassis":
            return pts

        if source_frame == "chest":
            chassis_T_chest = frame_poses.get("chassis/chest")
            if chassis_T_chest is None:
                logging.warning("Skipping %s: 'chassis/chest' frame is unavailable", field_name)
                return None
            transforms = np.asarray(chassis_T_chest)
            transforms = transforms[start_index:]
            n = min(pts.shape[0], transforms.shape[0])
            if n < pts.shape[0]:
                logging.warning(
                    "Truncating %s command visualization from %d to %d timesteps to match chest poses",
                    field_name,
                    pts.shape[0],
                    n,
                )
            points_h = np.concatenate([pts[:n], np.ones((n, 1), dtype=pts.dtype)], axis=1)
            return (transforms[:n] @ points_h[:, :, None])[:, :3, 0]

        logging.warning("Skipping %s: local-frame position commands are not visualized in chassis frame", field_name)
        return None

    @staticmethod
    def log_lift_action(lowdim: Mapping[str, NDArray], chest_T: NDArray, start_index: int = 0) -> None:
        lift = lowdim.get("lift_action")
        if lift is None:
            logging.debug("'lift_action' not found in lowdim data; skipping lift action visualization")
            return
        if start_index >= lift.shape[0] or start_index >= chest_T.shape[0]:
            logging.debug("Skipping lift action visualization: start index %d is outside available data", start_index)
            return

        rr.set_time("sample", sequence=start_index)
        empty = np.zeros([lift.shape[0], 2])
        lift_xyz = np.concatenate([empty, lift], axis=1)
        base = chest_T[start_index][:3, 3]
        pts = Plotter._integrate_velocity_commands(base, lift_xyz[start_index:])
        vz.log_points3d("chassis/lift_action", pts, radii=np.ones(pts.shape[0]) * BALL_RADIUS)

    @staticmethod
    def _velocity_anchor(
        frame_poses: Mapping[str, NDArray],
        side: str,
        tcp: str,
        start_index: int = 0,
    ) -> NDArray | None:
        preferred_path = f"chassis/{side}_{tcp}"
        fallback_path = f"chassis/{side}_eef"
        frame_path = preferred_path if preferred_path in frame_poses else fallback_path
        transforms = frame_poses.get(frame_path)
        if transforms is None:
            logging.warning("Skipping %s action visualization: frame '%s' is unavailable", side, frame_path)
            return None
        if start_index >= len(transforms):
            logging.warning(
                "Skipping %s action visualization: start index %d is outside frame '%s' with length %d",
                side,
                start_index,
                frame_path,
                len(transforms),
            )
            return None
        return transforms[start_index][:3, 3]

    @staticmethod
    def log_velocity_commands(
        lowdim: Mapping[str, NDArray],
        frame_poses: Mapping[str, NDArray],
        action_fields: tuple[str, ...] = (),
        start_index: int = 0,
        path_prefix: str = "chassis",
    ) -> dict[str, NDArray]:
        rr.set_time("sample", sequence=start_index)

        explicit_fields = bool(action_fields)
        fields_to_log = tuple(field for field in action_fields if field in VELOCITY_ACTION_FIELDS)
        if not fields_to_log and not explicit_fields:
            fields_to_log = tuple(field for field in VELOCITY_ACTION_FIELDS if field in lowdim)

        trajectories: dict[str, NDArray] = {}
        logged_any = False
        logged_split_fields: set[str] = set()
        for field_name in fields_to_log:
            values = lowdim.get(field_name)
            if values is None:
                continue

            side, tcp = VELOCITY_ACTION_FIELDS[field_name]
            deltas = Plotter._extract_xyz(field_name, values)
            if deltas is not None:
                deltas = deltas[start_index:]
            anchor = Plotter._velocity_anchor(frame_poses, side, tcp, start_index)
            if deltas is None or anchor is None:
                continue

            pts = Plotter._integrate_velocity_commands(anchor, deltas)
            path = f"{path_prefix}/{side}_{tcp}_action"
            Plotter._log_command_trajectory(path, pts)
            trajectories[path] = pts
            logged_any = True
            logged_split_fields.add(field_name)

        # Fallback: support legacy 14-dim bimanual action formats.
        for parent_field, (tcp, child_fields) in LEGACY_BIMANUAL_VELOCITY_FIELDS.items():
            if parent_field not in lowdim:
                continue
            if explicit_fields and parent_field not in action_fields and not any(
                child_field in action_fields for child_field in child_fields
            ):
                continue
            if any(child_field in logged_split_fields for child_field in child_fields):
                continue
            if not explicit_fields and any(child_field in lowdim for child_field in child_fields):
                continue

            arm_action = Plotter._as_2d_array(parent_field, lowdim[parent_field])
            if arm_action is not None and arm_action.shape[1] >= 14:
                for side, values in (("left", arm_action[:, :7]), ("right", arm_action[:, 7:14])):
                    deltas = Plotter._extract_xyz(f"{side}_arm_action", values)
                    if deltas is not None:
                        deltas = deltas[start_index:]
                    anchor = Plotter._velocity_anchor(frame_poses, side, tcp, start_index)
                    if deltas is None or anchor is None:
                        continue
                    pts = Plotter._integrate_velocity_commands(anchor, deltas)
                    path = f"{path_prefix}/{side}_{tcp}_action"
                    Plotter._log_command_trajectory(path, pts)
                    trajectories[path] = pts
                    logged_any = True

        if not logged_any:
            logging.debug("No velocity arm command fields found for visualization")
        return trajectories

    @staticmethod
    def log_position_commands(
        lowdim: Mapping[str, NDArray],
        frame_poses: Mapping[str, NDArray],
        action_fields: tuple[str, ...] = (),
        start_index: int = 0,
        path_prefix: str = "chassis",
    ) -> dict[str, NDArray]:
        rr.set_time("sample", sequence=start_index)

        explicit_fields = bool(action_fields)
        fields_to_log = tuple(field for field in action_fields if is_position_action_field(field))
        if not fields_to_log and not explicit_fields:
            fields_to_log = tuple(field for field in lowdim if is_position_action_field(field))

        trajectories: dict[str, NDArray] = {}
        logged_any = False
        for field_name in fields_to_log:
            values = lowdim.get(field_name)
            if values is None:
                logging.debug("Position action field '%s' is not present in lowdim data", field_name)
                continue

            config = _get_position_cmd_config(field_name)
            if config is None:
                continue

            part_name, reference_frame, tcp = config
            side = "left" if "left" in part_name else "right"
            source_frame = ReferenceFrame(reference_frame).name.lower()
            tcp_name = "eef" if tcp == "arm_tip" else tcp

            pts = Plotter._extract_xyz(field_name, values)
            if pts is None:
                continue
            pts = pts[start_index:]
            chassis_pts = Plotter._points_in_chassis_frame(field_name, pts, source_frame, frame_poses, start_index)
            if chassis_pts is None:
                continue

            path = f"{path_prefix}/{side}_{tcp_name}_position_command"
            Plotter._log_command_trajectory(path, chassis_pts)
            trajectories[path] = chassis_pts
            logged_any = True

        if not logged_any:
            logging.debug("No position command fields found for visualization")
        return trajectories

    def log_point_cloud(
        self,
        path: str,
        raw_depth: NDArray,
        depth_scale: float | NDArray,
        color_image: NDArray | None,
        intrinsics: NDArray,
        image_size: tuple[int, int],
    ) -> None:
        vz.log_point_cloud(path, raw_depth, depth_scale, color_image, intrinsics, image_size)


class RerunSampleVisualizer:
    """High-level visualizer that ties together reading, frames, and plotting."""

    def __init__(self, normalizer_bundle: NormalizerBundle, command_mode: str = "none"):
        self._reader = TarReader(normalizer_bundle)
        self._frames = FrameAssembler()
        self._plot = Plotter()
        self._action_fields = normalizer_bundle.action_fields
        self._command_mode = "action" if command_mode == "velocity" else command_mode
        logging.info("Action fields: %s", list(self._action_fields))
        logging.info("Command visualization mode: %s", self._command_mode)

    def _extract_depth_scale(self, metadata: Mapping[str, object]) -> float:
        """Extract scalar depth scale from heterogeneous metadata representations."""
        raw = metadata.get("depth_scale")
        if raw is None:
            raise ValueError("'depth_scale' not found in metadata for point cloud plotting")
        if isinstance(raw, (float, int)):
            return float(raw)
        if isinstance(raw, str):
            try:
                arr = np.fromstring(raw.strip("[]"), sep=",")
                return float(arr.reshape(-1)[0])
            except Exception as exc:
                raise ValueError(f"Could not parse depth_scale from string: {exc}") from exc
        if isinstance(raw, (list, tuple, np.ndarray)):
            return float(np.asarray(raw).reshape(-1)[0])
        raise ValueError(f"Unsupported depth_scale type: {type(raw)}")

    @staticmethod
    def _split_command_fields(action_fields: Sequence[str]) -> tuple[tuple[str, ...], tuple[str, ...]]:
        velocity_fields = tuple(
            field
            for field in action_fields
            if field in VELOCITY_ACTION_FIELDS or field in LEGACY_BIMANUAL_VELOCITY_FIELDS
        )
        position_fields = tuple(field for field in action_fields if is_position_action_field(field))
        return velocity_fields, position_fields

    def _select_command_fields(self) -> tuple[tuple[str, ...], tuple[str, ...]]:
        velocity_fields, position_fields = self._split_command_fields(self._action_fields)

        if self._command_mode == "action":
            position_fields = ()
        elif self._command_mode == "position":
            velocity_fields = ()

        self._validate_command_fields(self._action_fields, velocity_fields, position_fields, "data.action_fields")
        return velocity_fields, position_fields

    def _validate_command_fields(
        self,
        action_fields: Sequence[str],
        velocity_fields: tuple[str, ...],
        position_fields: tuple[str, ...],
        source_label: str,
    ) -> None:
        missing = []
        if self._command_mode in {"all", "action"} and not velocity_fields:
            missing.append("velocity/action")
        if self._command_mode in {"all", "position"} and not position_fields:
            missing.append("position")
        if not missing:
            return

        configured = ", ".join(action_fields) if action_fields else "<none>"
        raise ValueError(
            f"--command-mode {self._command_mode!r} requires non-empty {', '.join(missing)} fields in "
            f"{source_label}. Configured action_fields: {configured}"
        )

    def _select_prediction_fields(
        self,
        prediction: PredictionBundle,
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        velocity_fields, position_fields = self._split_command_fields(prediction.action_fields)
        if self._command_mode == "action":
            position_fields = ()
        elif self._command_mode == "position":
            velocity_fields = ()
        self._validate_command_fields(
            prediction.action_fields,
            velocity_fields,
            position_fields,
            f"prediction '{prediction.label}' action_fields",
        )
        return velocity_fields, position_fields

    @staticmethod
    def _anchor_index(metadata: Mapping[str, object] | None, num_steps: int) -> int:
        if metadata is None or num_steps <= 0:
            return 0
        try:
            anchor = int(metadata.get("anchor_timestep", 0)) - int(metadata.get("lowdim_start_timestep", 0))
        except (TypeError, ValueError):
            return 0
        return max(0, min(anchor, num_steps - 1))

    @staticmethod
    def _intrinsics4_from_lowdim(field_name: str, values: NDArray) -> NDArray:
        arr = np.asarray(values)
        if arr.ndim >= 3 and arr.shape[-2:] == (3, 3):
            intrinsics = arr.reshape(-1, 3, 3)[0]
            return intrinsics_3x3_to_4(intrinsics)
        if arr.ndim >= 1 and arr.shape[-1] == 4:
            return arr.reshape(-1, 4)[0]
        raise ValueError(f"'{field_name}' must have trailing shape (3, 3) or (4,), got {arr.shape}")

    def _intrinsics_for_dataset_image(
        self,
        lowdim: Mapping[str, NDArray],
        camera_name: str,
        dataset_image_size: tuple[int, int],
        original_image_size: tuple[int, int],
        fallback_camera_name: str | None = None,
    ) -> NDArray:
        camera_names = (camera_name,) if fallback_camera_name is None else (camera_name, fallback_camera_name)
        key_prefixes = (
            ("rescaled_intrinsics", "original_intrinsics")
            if dataset_image_size != original_image_size
            else ("original_intrinsics", "rescaled_intrinsics")
        )

        tried_keys = []
        for candidate_camera_name in camera_names:
            for prefix in key_prefixes:
                key = f"{prefix}.{candidate_camera_name}"
                tried_keys.append(key)
                if key not in lowdim:
                    continue
                intrinsics = self._intrinsics4_from_lowdim(key, lowdim[key])
                if prefix == "original_intrinsics" and dataset_image_size != original_image_size:
                    intrinsics = scale_intrinsics_4_for_resize_and_crop(
                        intrinsics,
                        original_image_size,
                        dataset_image_size,
                    )
                return intrinsics

            legacy_key = f"intrinsics.{candidate_camera_name}"
            tried_keys.append(legacy_key)
            if legacy_key in lowdim:
                intrinsics = self._intrinsics4_from_lowdim(legacy_key, lowdim[legacy_key])
                if dataset_image_size != original_image_size:
                    intrinsics = scale_intrinsics_4_for_resize_and_crop(
                        intrinsics,
                        original_image_size,
                        dataset_image_size,
                    )
                return intrinsics

        raise ValueError(
            f"No intrinsics found for camera '{camera_name}' matching displayed image size. Tried: {tried_keys}"
        )

    def _projected_trajectory_images(
        self,
        img_data: Mapping[str, NDArray],
        lowdim: Mapping[str, NDArray],
        original_image_sizes: Mapping[str, tuple[int, int] | list[int]],
        camera_frame_trajectories: Mapping[str, NDArray],
    ) -> dict[str, NDArray]:
        projected_images: dict[str, NDArray] = {}
        for image_key, image in img_data.items():
            camera_name = image_key.rsplit("_t", 1)[0]
            if camera_name not in original_image_sizes:
                logging.debug("Skipping projected overlay for %s: no original image size", image_key)
                continue

            height, width = image.shape[:2]
            intrinsics = self._intrinsics_for_dataset_image(
                lowdim,
                camera_name,
                (width, height),
                tuple(original_image_sizes[camera_name]),
                fallback_camera_name="rgb" if camera_name == "depth" else None,
            )
            projected_images[image_key] = self._plot.draw_projected_trajectories(
                image,
                intrinsics,
                camera_frame_trajectories,
            )
        return projected_images

    def visualize_sample(
        self,
        sample_id: str,
        payload: Mapping[str, object],
        prediction_bundles: Sequence[PredictionBundle] = (),
    ) -> None:
        """Visualize a single sample's payload."""
        img_data: Mapping[str, NDArray] | None = payload.get("img_data")  # type: ignore[assignment]
        lowdim: Mapping[str, NDArray] | None = payload.get("lowdim")  # type: ignore[assignment]
        metadata: Mapping[str, object] | None = payload.get("metadata")  # type: ignore[assignment]

        command_trajectories: dict[str, NDArray] = {}
        anchor_index = 0
        if lowdim is not None:
            frame_poses = self._frames.chassis_frame_poses(lowdim)
            anchor_index = self._anchor_index(metadata, len(frame_poses["chassis/chest"]))
            if img_data:
                rr.set_time("sample", sequence=anchor_index)
                self._plot.log_images("", img_data)
            self._plot.log_coordinate_frames(frame_poses)
            if self._command_mode != "none":
                velocity_fields, position_fields = self._select_command_fields()
                gt_path_prefix = "chassis/gt" if prediction_bundles else "chassis"
                if self._command_mode in {"all", "action"}:
                    self._plot.log_lift_action(lowdim, frame_poses["chassis/chest"], anchor_index)
                if velocity_fields:
                    command_trajectories.update(
                        self._plot.log_velocity_commands(
                            lowdim,
                            frame_poses,
                            velocity_fields,
                            anchor_index,
                            path_prefix=gt_path_prefix,
                        )
                    )
                if position_fields:
                    command_trajectories.update(
                        self._plot.log_position_commands(
                            lowdim,
                            frame_poses,
                            position_fields,
                            anchor_index,
                            path_prefix=gt_path_prefix,
                        )
                    )
                for prediction in prediction_bundles:
                    pred_velocity_fields, pred_position_fields = self._select_prediction_fields(prediction)
                    pred_path_prefix = f"chassis/pred/{prediction.label}"
                    if pred_velocity_fields:
                        command_trajectories.update(
                            self._plot.log_velocity_commands(
                                prediction.lowdim,
                                frame_poses,
                                pred_velocity_fields,
                                anchor_index,
                                path_prefix=pred_path_prefix,
                            )
                        )
                    if pred_position_fields:
                        command_trajectories.update(
                            self._plot.log_position_commands(
                                prediction.lowdim,
                                frame_poses,
                                pred_position_fields,
                                anchor_index,
                                path_prefix=pred_path_prefix,
                            )
                        )
        elif img_data:
            self._plot.log_images("", img_data)

        # Point cloud (requires metadata + depth + intrinsics + rgb).
        if metadata is None:
            raise ValueError("'metadata' not found for point cloud plotting, needed for depth_scale")
        depth_scale = self._extract_depth_scale(metadata)

        if lowdim is None:
            raise ValueError("'lowdim' not found for calibration data")

        if img_data is None or ("depth_t0" not in img_data):
            raise ValueError("Depth data ('depth_t0') not found in img_data")

        color_image = img_data.get("rgb_t0") if img_data else None
        raw_depth = img_data["depth_t0"]
        original_image_sizes = metadata["original_image_sizes"]
        depth_original_size = tuple(original_image_sizes.get("depth", original_image_sizes["rgb"]))
        depth_image_size = (raw_depth.shape[1], raw_depth.shape[0])
        depth_intrinsics = self._intrinsics_for_dataset_image(
            lowdim,
            "depth",
            depth_image_size,
            depth_original_size,
            fallback_camera_name="rgb",
        )
        rr.set_time("sample", sequence=anchor_index)
        self._plot.log_point_cloud(
            "chassis/head_camera/points",
            raw_depth,
            depth_scale,
            color_image,
            depth_intrinsics,
            depth_image_size,
        )
        logging.info("Plotted point cloud for sample %s", sample_id)

        if not command_trajectories:
            logging.info("No command trajectories logged for sample %s; skipping projected overlays", sample_id)
            return

        head_camera_T_chassis = invert_homogeneous_transform(frame_poses["chassis/head_camera"][anchor_index])
        camera_frame_trajectories = {
            path: transform_points_to_camera_frame(head_camera_T_chassis, pts)
            for path, pts in command_trajectories.items()
        }

        # Project command trajectories onto the decoded TAR images, not source-resolution images.
        img_data_with_traces = self._projected_trajectory_images(
            img_data,
            lowdim,
            original_image_sizes,
            camera_frame_trajectories,
        )
        if img_data_with_traces:
            self._plot.log_images("projected_trajectories", img_data_with_traces)

    def run(self, input_path: str) -> None:
        """Run the visualizer over one TAR file or all TAR files beneath a directory/prefix."""
        targets = list_tar_targets(input_path, recursive=True)
        if not targets:
            logging.warning("No TAR files found for given path: %s", input_path)
            return

        vz.init(run_name="MMT data visualizer")
        logging.info("Turn on looping in the Rerun viewer to see the EEF frames moving.")

        for tar_uri in tqdm(targets, desc="TARs"):
            for sample_id, payload in self._reader.iter_samples_from_tar(tar_uri):
                rr.set_time("sample", sequence=0)
                try:
                    self.visualize_sample(sample_id, payload)
                except Exception as exc:
                    logging.warning("Failed to visualize sample %s: %s", sample_id, exc)
                input("Sample visualized. Press Enter to continue...")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Visualize MMT TAR samples from a local path or URI using Rerun.")
    parser.add_argument(
        "--input_path",
        "--s3_path",
        dest="input_path",
        required=True,
        help="TAR file, directory, or URI/prefix containing TAR files. Local paths and s3:// paths are supported.",
    )
    parser.add_argument(
        "--data_params",
        required=True,
        help=("Path to data_params YAML (e.g., vla_foundry/config_presets/data/mmt/mmt_data_params.yaml)"),
    )
    parser.add_argument(
        "--command-mode",
        default="none",
        choices=["none", "all", "action", "position", "velocity"],
        help=(
            "Command visualization mode. 'none' skips all command reading/visualization; "
            "'all' logs action and position commands; 'velocity' is an alias for action."
        ),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Entry point."""
    args = parse_args(argv)
    setup_logging(getattr(logging, args.log_level))

    fs, input_fs_path = fsspec.core.url_to_fs(args.input_path)
    stats_path = (
        posixpath.join(posixpath.dirname(input_fs_path), "stats.json")
        if is_tar_path(input_fs_path)
        else posixpath.join(input_fs_path.rstrip("/"), "stats.json")
    )
    stats_uri = fs.unstrip_protocol(stats_path)
    norm = NormalizerBundle.from_paths(args.data_params, stats_uri)
    visualizer = RerunSampleVisualizer(norm, command_mode=args.command_mode)
    visualizer.run(args.input_path)


if __name__ == "__main__":
    main()
