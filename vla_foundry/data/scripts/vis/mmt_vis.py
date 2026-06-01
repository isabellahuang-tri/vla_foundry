"""Rerun visualizer for MMT TAR files on S3.

Example:
    python mmt_vis.py \
        --s3_path s3://tri-mmt-data/lpp_data/vla_foundry/20251028_paper_towel/shards/ \
        --data_params vla_foundry/config_presets/data/mmt/mmt_data_params.yaml
"""

import argparse
import io
import json
import logging
import tarfile
from collections.abc import Iterator, Mapping, MutableMapping
from dataclasses import dataclass
from pathlib import PurePosixPath

import fsspec
import numpy as np
import rerun as rr
import torch
import yaml
from matplotlib_wds_vis import list_s3_targets, open_s3
from PIL import Image
from tqdm import tqdm

from vla_foundry.data.robotics.cv_utils import (
    create_images_with_projected_trace,
    scale_intrinsics_4_for_resize_and_crop,
    transform_points_to_camera_frame,
)
from vla_foundry.data.robotics.normalization import RoboticsNormalizer
from vla_foundry.data.robotics.utils import invert_homogeneous_transform, xyzrpy_to_T
from vla_foundry.params.robotics.normalization_params import NormalizationParams
from vla_foundry.visualizers import visualizer as vz

BALL_RADIUS: float = 0.02

NDArray = np.ndarray


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

    norm_dict: Mapping[str, object] | None
    normalizer: RoboticsNormalizer | None

    @classmethod
    def from_paths(cls, data_params_path: str, stats_json_uri: str | None) -> "NormalizerBundle":
        """Create a NormalizerBundle from local YAML and (possibly remote) JSON stats.

        If stats_json_uri is None or the file doesn't exist, returns a bundle with no normalization.
        """
        with open(data_params_path, encoding="utf-8") as f:
            data_params = yaml.safe_load(f) or {}

        # Try to load normalization stats (JSON may be local or remote via fsspec)
        norm_dict = None
        normalizer = None

        if stats_json_uri is not None:
            try:
                fs, _ = fsspec.core.url_to_fs(stats_json_uri)
                with fs.open(stats_json_uri, "r") as f:
                    norm_dict = json.load(f)
                normalization_params = NormalizationParams(**(data_params.get("normalization", {}) or {}))
                normalizer = RoboticsNormalizer(normalization_params=normalization_params, statistics_data=[norm_dict])
                logging.info("Loaded normalization stats from %s", stats_json_uri)
            except FileNotFoundError:
                logging.warning("Stats file not found: %s. Skipping denormalization.", stats_json_uri)
            except Exception as exc:
                logging.warning("Failed to load stats from %s: %s. Skipping denormalization.", stats_json_uri, exc)

        if normalizer is None:
            logging.info("No normalization will be applied")

        return cls(norm_dict=norm_dict, normalizer=normalizer)

    def maybe_denormalize(self, key: str, arr: NDArray) -> NDArray:
        """Denormalize array if stats exist for key; otherwise return unchanged."""
        if self.norm_dict is not None and self.normalizer is not None and key in self.norm_dict:
            tens = torch.from_numpy(arr)
            out = self.normalizer.denormalize_tensor(tens, key).numpy()
            return out
        return arr


class S3TarReader:
    """Stream TAR files from S3 and assemble per-sample dictionaries."""

    def __init__(self, normalizer_bundle: NormalizerBundle):
        self._norm = normalizer_bundle

    def iter_samples_from_tar(self, tar_uri: str) -> Iterator[tuple[str, dict[str, object]]]:
        """Yield (sample_id, sample_dict) for each sample found in a single TAR."""
        data_dict: dict[str, dict[str, object]] = {}
        logging.debug("Opening TAR: %s", tar_uri)

        with open_s3(tar_uri) as fo, tarfile.open(fileobj=fo, mode="r|*") as tf:
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
                low_dim[field_name] = self._norm.maybe_denormalize(field_name, array)
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

    def chassis_frame_poses(self, lowdim: Mapping[str, NDArray]) -> dict[str, NDArray]:
        """Return mapping of chassis-anchored frame trajectories."""
        chest_T_eef = lowdim.get("chest_T_eef_pose")
        chest_T_head = lowdim.get("chest_T_head_pose")
        chassis_T_chest = lowdim.get("chassis_T_chest_pose")

        if chest_T_eef is None:
            raise ValueError("'chest_T_eef_pose' not found in lowdim data")
        if chest_T_head is None:
            raise ValueError("'chest_T_head_pose' not found in lowdim data")
        if chassis_T_chest is None:
            raise ValueError("'chassis_T_chest_pose' not found in lowdim data")

        chest_T_left = xyzrpy_to_T(chest_T_eef[:, 1:7]) @ xyzrpy_to_T([0, 0, 0, 0, np.pi, 0])
        chest_T_right = xyzrpy_to_T(chest_T_eef[:, 8:14]) @ xyzrpy_to_T([0, 0, 0, 0, np.pi, 0])

        chassis_T_chest_T = xyzrpy_to_T(chassis_T_chest[:, :6])
        chest_T_head_T = xyzrpy_to_T(chest_T_head[:, :6])
        chassis_T_head_T = chassis_T_chest_T @ chest_T_head_T

        return {
            "chassis/left_eef": chassis_T_chest_T @ chest_T_left,
            "chassis/right_eef": chassis_T_chest_T @ chest_T_right,
            "chassis/head": chassis_T_head_T @ self._R_head,
            "chassis/head_camera": [(chassis_T_head_T @ self._R_head)[0]],
            "chassis/chest": chassis_T_chest_T,
        }


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
    def log_lift_action(lowdim: Mapping[str, NDArray], chest_T: NDArray) -> None:
        lift = lowdim.get("lift_action")
        if lift is None:
            raise ValueError("'lift_action' not found in lowdim data")

        rr.set_time("sample", sequence=0)
        empty = np.zeros([lift.shape[0], 2])
        lift_xyz = np.concatenate([empty, lift], axis=1)
        base = chest_T[0][:3, 3]
        pts = np.cumsum(lift_xyz, axis=0) + base
        vz.log_points3d("chassis/lift_action", pts, radii=np.ones(pts.shape[0]) * BALL_RADIUS)

    @staticmethod
    def log_eef_actions(lowdim: Mapping[str, NDArray], frame_poses: Mapping[str, NDArray]) -> None:
        rr.set_time("sample", sequence=0)
        left_arm_action = lowdim.get("left_arm_action")
        right_arm_action = lowdim.get("right_arm_action")

        # Fallback: support legacy 14-dim arm_action format
        if left_arm_action is None and right_arm_action is None:
            arm_action = lowdim.get("arm_action")
            if arm_action is None:
                raise ValueError("None of 'left_arm_action', 'right_arm_action', or 'arm_action' found in lowdim data")
            left_arm_action = arm_action[:, :7]
            right_arm_action = arm_action[:, 7:]

        left_pts = None
        right_pts = None
        if left_arm_action is not None:
            left = left_arm_action[:, 1:7]
            left_pts = np.cumsum(left[:, :3], axis=0) + frame_poses["chassis/left_eef"][0][:3, 3]
        if right_arm_action is not None:
            right = right_arm_action[:, 1:7]
            right_pts = np.cumsum(right[:, :3], axis=0) + frame_poses["chassis/right_eef"][0][:3, 3]

        if left_pts is not None:
            vz.log_points3d(
                "chassis/left_eef_action",
                left_pts,
                radii=np.ones(left_pts.shape[0]) * BALL_RADIUS,
            )
        if right_pts is not None:
            vz.log_points3d(
                "chassis/right_eef_action",
                right_pts,
                radii=np.ones(right_pts.shape[0]) * BALL_RADIUS,
            )

    def log_point_cloud(
        self,
        path: str,
        raw_depth: NDArray,
        depth_scale: float | NDArray,
        color_image: NDArray | None,
        intrinsics_rgb: NDArray,
        original_image_size: tuple[int, int],
    ) -> None:
        vz.log_point_cloud(path, raw_depth, depth_scale, color_image, intrinsics_rgb, original_image_size)


class RerunSampleVisualizer:
    """High-level visualizer that ties together reading, frames, and plotting."""

    def __init__(self, normalizer_bundle: NormalizerBundle):
        self._norm = normalizer_bundle
        self._reader = S3TarReader(normalizer_bundle)
        self._frames = FrameAssembler()
        self._plot = Plotter()

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

    def visualize_sample(self, sample_id: str, payload: Mapping[str, object]) -> None:
        """Visualize a single sample's payload."""
        img_data: Mapping[str, NDArray] | None = payload.get("img_data")  # type: ignore[assignment]
        lowdim: Mapping[str, NDArray] | None = payload.get("lowdim")  # type: ignore[assignment]
        metadata: Mapping[str, object] | None = payload.get("metadata")  # type: ignore[assignment]

        if img_data:
            self._plot.log_images("", img_data)

        if lowdim is not None:
            frame_poses = self._frames.chassis_frame_poses(lowdim)
            self._plot.log_coordinate_frames(frame_poses)
            self._plot.log_lift_action(lowdim, frame_poses["chassis/chest"])
            self._plot.log_eef_actions(lowdim, frame_poses)

        # Point cloud (requires metadata + depth + intrinsics + rgb).
        if metadata is None:
            raise ValueError("'metadata' not found for point cloud plotting, needed for depth_scale")
        depth_scale = self._extract_depth_scale(metadata)

        if lowdim is None or "intrinsics.rgb" not in lowdim:
            raise ValueError("'intrinsics.rgb' not found in lowdim for point cloud plotting")
        intrinsics = lowdim["intrinsics.rgb"]

        if img_data is None or ("depth_t0" not in img_data):
            raise ValueError("Depth data ('depth_t0') not found in img_data")

        color_image = img_data.get("rgb_t0") if img_data else None
        raw_depth = img_data["depth_t0"]
        self._plot.log_point_cloud(
            "chassis/head_camera/points",
            raw_depth,
            depth_scale,
            color_image,
            intrinsics[0][0],
            # type: ignore[index]
            payload["metadata"]["original_image_sizes"]["rgb"],
        )
        logging.info("Plotted point cloud for sample %s", sample_id)

        # Apply transformation of 3D left and right arm trajectories into camera frame.
        left_pts = frame_poses["chassis/left_eef"][:, :3, 3]
        right_pts = frame_poses["chassis/right_eef"][:, :3, 3]
        head_camera_T_chassis = invert_homogeneous_transform(frame_poses["chassis/head_camera"][0])
        head_camera_t_left_pts = transform_points_to_camera_frame(head_camera_T_chassis, left_pts)
        head_camera_t_right_pts = transform_points_to_camera_frame(head_camera_T_chassis, right_pts)

        # Scale intrinsics to handle processing on original image (resize then crop).
        H, W, C = color_image.shape
        W0, H0 = payload["metadata"]["original_image_sizes"]["rgb"]

        # Convert intrinsics to (fx, fy, cx, cy) format if needed
        intr = np.asarray(intrinsics[0][0])
        if intr.shape == (3,):
            # Assume (fx, cx, cy) with fy=fx (square pixels)
            fx, cx, cy = intr
            intr_4 = np.array([fx, fx, cx, cy])
        elif intr.shape == (4,):
            intr_4 = intr
        elif intr.shape == (3, 3):
            intr_4 = np.array([intr[0, 0], intr[1, 1], intr[0, 2], intr[1, 2]])
        else:
            raise ValueError(f"Unexpected intrinsics shape: {intr.shape}")

        scaled_intrinsics = scale_intrinsics_4_for_resize_and_crop(
            intr_4,
            (W0, H0),
            (W, H),
        )

        # Currently visualizing the same trajectories for both t-1 and t0 images.
        # Debug: log what images we're processing
        logging.info("Processing images for trajectory projection:")
        for key, img in (img_data.items() if img_data else []):
            logging.info("  Input %s: shape=%s, dtype=%s", key, img.shape, img.dtype)

        img_data_with_traces = create_images_with_projected_trace(
            img_data,
            scaled_intrinsics,
            [head_camera_t_left_pts, head_camera_t_right_pts],
        )

        # Debug: log output
        logging.info("After trajectory projection:")
        for key, img in img_data_with_traces.items():
            logging.info("  Output %s: shape=%s, dtype=%s", key, img.shape, img.dtype)

        # Log with trajectories overlaid as a separate view for easier comparison
        self._plot.log_images("with_trajectories", img_data_with_traces)

    def run(self, s3_prefix: str) -> None:
        """Run the visualizer over all TAR files beneath the given S3 prefix."""
        targets = list_s3_targets(s3_prefix, recursive=True)
        if not targets:
            logging.warning("No .tar files found for given URI and options.")
            return

        # Sort targets to visualize in order (assuming TAR filenames are sequential)
        targets = sorted(targets)

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
    parser = argparse.ArgumentParser(description="Visualize TAR samples from an S3 prefix using Rerun.")
    parser.add_argument(
        "--s3_path",
        required=True,
        help="S3 prefix to TAR files. Use .../shards/ for shuffled data or .../episodes/ for ordered episodes.",
    )
    parser.add_argument(
        "--data_params",
        required=True,
        help=("Path to data_params YAML (e.g., vla_foundry/config_presets/data/mmt/mmt_data_params.yaml)"),
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

    # Try to find normalization stats in the shards/ directory (if it exists)
    stats_base_path = args.s3_path.replace("/episodes/", "/shards/").replace("/episodes", "/shards")
    # Check if path already ends with shards/ or episodes/; if not, try to append shards/
    if not any(x in stats_base_path for x in ["/shards/", "/episodes/", "/shards", "/episodes"]):
        stats_base_path = stats_base_path.rstrip("/") + "/shards"
    else:
        # Normalize to shards/ for stats lookup
        stats_base_path = stats_base_path.replace("/episodes/", "/shards/").replace("/episodes", "/shards")

    stats_uri = stats_base_path.rstrip("/") + "/stats.json"

    norm = NormalizerBundle.from_paths(args.data_params, stats_uri)
    visualizer = RerunSampleVisualizer(norm)
    visualizer.run(args.s3_path)


if __name__ == "__main__":
    main()
