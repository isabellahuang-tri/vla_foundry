"""Tests for RGB-only image selection in the robotics pipeline.

Robotics shards co-locate non-camera images (e.g. 16-bit depth PNGs used by
visualization / point-cloud tooling) alongside the RGB cameras the model
consumes. The pipeline must drop those before decode so they are never decoded
or augmented as RGB, instead of silently coercing them (which previously caused
a uint8/uint16 torch.stack crash).
"""

import io

import numpy as np
import pytest
import torch
from PIL import Image

from vla_foundry.data.augmentations.decode_and_augment import Augmentations
from vla_foundry.data.pipelines.robotics import _image_key_aliases, drop_unused_images
from vla_foundry.params.robotics.augmentation_params import (
    ColorJitterParams,
    CropParams,
    DataAugmentationParams,
    ImageAugmentationParams,
)


def _rgb_jpg_bytes(h=384, w=384):
    buf = io.BytesIO()
    Image.fromarray(np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)).save(buf, format="JPEG")
    return buf.getvalue()


def _depth_png_bytes(h=384, w=384):
    """16-bit single-channel PNG, same resolution as the RGB camera."""
    buf = io.BytesIO()
    Image.fromarray(np.random.randint(0, 4000, (h, w), dtype=np.uint16)).save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# _image_key_aliases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "field_key,expected",
    [
        ("rgb_t0.jpg", ("rgb_t0", "rgb_t0")),
        ("depth_t-1.png", ("depth_t-1", "depth_t-1")),
        ("observation.images.cam_t0.jpg", ("observation.images.cam_t0", "cam_t0")),
        ("observation.images.left_t-1.jpg", ("observation.images.left_t-1", "left_t-1")),
    ],
)
def test_image_key_aliases(field_key, expected):
    assert _image_key_aliases(field_key) == expected


# ---------------------------------------------------------------------------
# drop_unused_images
# ---------------------------------------------------------------------------


def test_drop_unused_images_removes_depth_keeps_rgb_and_data():
    sample = {
        "rgb_t-1.jpg": b"rgb-1",
        "rgb_t0.jpg": b"rgb0",
        "depth_t-1.png": b"depth-1",
        "depth_t0.png": b"depth0",
        "lowdim.npz": b"npz",
        "metadata.json": b"meta",
    }
    kept = drop_unused_images(sample, ["rgb_t-1", "rgb_t0"])

    assert set(kept) == {"rgb_t-1.jpg", "rgb_t0.jpg", "lowdim.npz", "metadata.json"}
    # Values for kept keys are passed through untouched.
    assert kept["rgb_t0.jpg"] == b"rgb0"


def test_drop_unused_images_matches_short_stem_for_dotted_keys():
    sample = {
        "observation.images.cam_t0.jpg": b"rgb",
        "observation.images.depth_t0.png": b"depth",
        "lowdim.npz": b"npz",
    }
    # image_names uses the short camera form.
    kept = drop_unused_images(sample, ["cam_t0"])

    assert "observation.images.cam_t0.jpg" in kept
    assert "observation.images.depth_t0.png" not in kept
    assert "lowdim.npz" in kept


def test_drop_unused_images_keeps_explicitly_requested_non_rgb():
    """If a camera is explicitly in image_names it survives selection (and would
    then trip the decoder's uint8 tripwire if it is not 8-bit)."""
    sample = {"rgb_t0.jpg": b"rgb", "depth_t0.png": b"depth"}
    kept = drop_unused_images(sample, ["rgb_t0", "depth_t0"])
    assert set(kept) == {"rgb_t0.jpg", "depth_t0.png"}


def test_drop_unused_images_keeps_all_when_image_names_empty():
    sample = {"rgb_t0.jpg": b"rgb", "depth_t0.png": b"depth", "lowdim.npz": b"npz"}
    assert drop_unused_images(sample, []) == sample
    assert drop_unused_images(sample, None) == sample


def test_drop_unused_images_preserves_non_image_fields():
    """TIFF point maps and point clouds are not image keys and must be kept."""
    sample = {
        "rgb_t0.jpg": b"rgb",
        "depth_t0.png": b"depth",
        "scene_right_0_point_map_t0.tiff": b"tiff",
        "point_cloud.npz": b"pc",
        "language_instructions.json": b"lang",
    }
    kept = drop_unused_images(sample, ["rgb_t0"])
    assert "depth_t0.png" not in kept
    assert {"rgb_t0.jpg", "scene_right_0_point_map_t0.tiff", "point_cloud.npz", "language_instructions.json"} <= set(
        kept
    )


# ---------------------------------------------------------------------------
# End-to-end: selection + decode/augment of an RGB+depth sample
# ---------------------------------------------------------------------------


def test_rgb_depth_sample_decodes_without_stack_crash_after_selection():
    """Regression: same-resolution RGB (uint8) + depth (uint16) used to crash in
    torch.stack. After dropping depth, decode+augment with crop+jitter succeeds."""
    sample = {
        "__key__": "s1",
        "rgb_t-1.jpg": _rgb_jpg_bytes(),
        "rgb_t0.jpg": _rgb_jpg_bytes(),
        "depth_t-1.png": _depth_png_bytes(),
        "depth_t0.png": _depth_png_bytes(),
    }
    augmentations = Augmentations(
        DataAugmentationParams(
            enabled=True,
            image=ImageAugmentationParams(
                crop=CropParams(enabled=True, mode="random", shape=[360, 360]),
                color_jitter=ColorJitterParams(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=(-0.05, 0.05), enabled=True
                ),
            ),
        )
    )

    kept = drop_unused_images(sample, ["rgb_t-1", "rgb_t0"])
    result = augmentations.decode_and_augment_sample(kept)

    for key in ("rgb_t-1.jpg", "rgb_t0.jpg"):
        assert isinstance(result[key], torch.Tensor)
        assert result[key].dtype == torch.uint8
        assert result[key].shape == (3, 360, 360)
    # Depth was never decoded.
    assert "depth_t0.png" not in result and "depth_t-1.png" not in result
