"""Canonical field layouts for MMT robotics inference.

Each entry defines:
  - full_dim: the full (pre-selection) dimension of the flattened field
  - groups: named index groups for sub-field access (e.g. left/right arm)
  - zzk_source (optional): how to extract this field from a ZZK status dict
    at inference time.  Omitted for action fields (they come from the model).
zzk_source types:
  - eef_pose: bimanual [gripper, pose(6)] x 2 — gripper from status scalar fields,
              pose from state matrix rows
  - single_arm_pose: single arm [gripper, pose(6)] — gripper from status scalar
                     field selected by ``side``, pose from state matrix row
  - state_row: a single row of the state matrix
  - status_field: a top-level key in the ZZK status dict
  - status_field_slice: a slice of a top-level key in the ZZK status dict

ZZK state format (from zzk_api_ctypes_client.cc):
  state[0][0:6]  = local_T_chassis (x, y, z, rx, ry, rz)
  state[2][0:6]  = chassis_T_left_arm_tip (x, y, z, rx, ry, rz)
  state[4][0:6]  = chassis_T_right_arm_tip (x, y, z, rx, ry, rz)
  state[5][0:6]  = chest_T_left_arm_tip (x, y, z, rx, ry, rz)
  state[6][0:6]  = chest_T_right_arm_tip (x, y, z, rx, ry, rz)
  state[7][0:6]  = chassis_T_left_gripper_tip (x, y, z, rx, ry, rz)
  state[8][0:6]  = chassis_T_right_gripper_tip (x, y, z, rx, ry, rz)
  state[9][0:6]  = chest_T_left_gripper_tip (x, y, z, rx, ry, rz)
  state[10][0:6] = chest_T_right_gripper_tip (x, y, z, rx, ry, rz)
  chest_T_head    = chest_T_head pose [x, y, z, rx, ry, rz]
  chassis_T_chest = chassis_T_chest pose [x, y, z, rx, ry, rz]
  wrench          = [left_fx, left_fy, left_fz, left_tx, left_ty, left_tz,
                     right_fx, right_fy, right_fz, right_tx, right_ty, right_tz]
  left_gripper_position  = scalar (status field)
  right_gripper_position = scalar (status field)
"""

ZZK_STATE_ROWS = 11
ZZK_STATE_COLS = 6

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Base field layouts — always required.
# These map directly to ZZK robot state or standalone model outputs.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
_BASE_FIELD_LAYOUTS = {
    # ── action fields (no zzk_source — produced by the model) ──
    "base_action": {
        "full_dim": 6,
        "groups": {"chassis": tuple(range(6))},
    },
    "head_action": {
        "full_dim": 6,
        "groups": {"head": tuple(range(6))},
    },
    "lift_action": {
        "full_dim": 6,
        "groups": {"lift": tuple(range(6))},
    },
    # ── proprioception fields (zzk_source → how to extract from ZZK status) ──
    "chassis_T_eef_pose": {
        "full_dim": 14,
        "groups": {
            "left_gripper": (0,),
            "left_pose": tuple(range(1, 7)),
            "right_gripper": (7,),
            "right_pose": tuple(range(8, 14)),
        },
        "zzk_source": {
            "type": "eef_pose",
            "left_pose_row": 2,
            "right_pose_row": 4,
        },
    },
    "chest_T_eef_pose": {
        "full_dim": 14,
        "groups": {
            "left_gripper": (0,),
            "left_pose": tuple(range(1, 7)),
            "right_gripper": (7,),
            "right_pose": tuple(range(8, 14)),
        },
        "zzk_source": {
            "type": "eef_pose",
            "left_pose_row": 5,
            "right_pose_row": 6,
        },
    },
    "chassis_T_gripper_tip_pose": {
        "full_dim": 14,
        "groups": {
            "left_gripper": (0,),
            "left_pose": tuple(range(1, 7)),
            "right_gripper": (7,),
            "right_pose": tuple(range(8, 14)),
        },
        "zzk_source": {
            "type": "eef_pose",
            "left_pose_row": 7,
            "right_pose_row": 8,
        },
    },
    # ── per-arm proprioception fields (single_arm_pose / single_arm_wrench) ──
    # chassis_T_eef_pose split
    "chassis_T_left_eef_pose": {
        "full_dim": 7,
        "groups": {"gripper": (0,), "pose": tuple(range(1, 7))},
        "zzk_source": {"type": "single_arm_pose", "side": "left", "pose_row": 2},
    },
    "chassis_T_right_eef_pose": {
        "full_dim": 7,
        "groups": {"gripper": (0,), "pose": tuple(range(1, 7))},
        "zzk_source": {"type": "single_arm_pose", "side": "right", "pose_row": 4},
    },
    # chest_T_eef_pose split
    "chest_T_left_eef_pose": {
        "full_dim": 7,
        "groups": {"gripper": (0,), "pose": tuple(range(1, 7))},
        "zzk_source": {"type": "single_arm_pose", "side": "left", "pose_row": 5},
    },
    "chest_T_right_eef_pose": {
        "full_dim": 7,
        "groups": {"gripper": (0,), "pose": tuple(range(1, 7))},
        "zzk_source": {"type": "single_arm_pose", "side": "right", "pose_row": 6},
    },
    # chassis_T_gripper_tip_pose split
    "chassis_T_left_gripper_tip_pose": {
        "full_dim": 7,
        "groups": {"gripper": (0,), "pose": tuple(range(1, 7))},
        "zzk_source": {"type": "single_arm_pose", "side": "left", "pose_row": 7},
    },
    "chassis_T_right_gripper_tip_pose": {
        "full_dim": 7,
        "groups": {"gripper": (0,), "pose": tuple(range(1, 7))},
        "zzk_source": {"type": "single_arm_pose", "side": "right", "pose_row": 8},
    },
    # chest_T_gripper_tip_pose split
    "chest_T_left_gripper_tip_pose": {
        "full_dim": 7,
        "groups": {"gripper": (0,), "pose": tuple(range(1, 7))},
        "zzk_source": {"type": "single_arm_pose", "side": "left", "pose_row": 9},
    },
    "chest_T_right_gripper_tip_pose": {
        "full_dim": 7,
        "groups": {"gripper": (0,), "pose": tuple(range(1, 7))},
        "zzk_source": {"type": "single_arm_pose", "side": "right", "pose_row": 10},
    },
    # wrench split
    "left_wrench": {
        "full_dim": 6,
        "groups": {"wrench": tuple(range(6))},
        "zzk_source": {"type": "status_field_slice", "key": "wrench", "start": 0, "end": 6},
    },
    "right_wrench": {
        "full_dim": 6,
        "groups": {"wrench": tuple(range(6))},
        "zzk_source": {"type": "status_field_slice", "key": "wrench", "start": 6, "end": 12},
    },
    "chest_T_gripper_tip_pose": {
        "full_dim": 14,
        "groups": {
            "left_gripper": (0,),
            "left_pose": tuple(range(1, 7)),
            "right_gripper": (7,),
            "right_pose": tuple(range(8, 14)),
        },
        "zzk_source": {
            "type": "eef_pose",
            "left_pose_row": 9,
            "right_pose_row": 10,
        },
    },
    "base_pose": {
        "full_dim": 6,
        "groups": {"base": tuple(range(6))},
        "zzk_source": {"type": "state_row", "row": 0},
    },
    "chest_T_head_pose": {
        "full_dim": 6,
        "groups": {"head_pose": tuple(range(6))},
        "zzk_source": {"type": "status_field", "key": "chest_T_head_pose", "expected_size": 6},
    },
    "chassis_T_chest_pose": {
        "full_dim": 6,
        "groups": {"chest_pose": tuple(range(6))},
        "zzk_source": {"type": "status_field", "key": "chassis_T_chest_pose", "expected_size": 6},
    },
    "wrench": {
        "full_dim": 12,
        "groups": {
            "left_wrench": tuple(range(6)),
            "right_wrench": tuple(range(6, 12)),
        },
        "zzk_source": {"type": "status_field", "key": "wrench", "expected_size": 12},
    },
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Legacy per-arm defaults — backward compatibility only.
# Needed for older checkpoints preprocessed WITHOUT lowdim_field_remap.
# New checkpoints with lowdim_field_remap auto-generate these via
# build_runtime_layouts(), so entries here are NOT used in that case.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
_LEGACY_PER_ARM_DEFAULTS = {
    # action splits
    "left_arm_action": {
        "full_dim": 7,
        "groups": {"gripper": (0,), "arm": tuple(range(1, 7))},
    },
    "right_arm_action": {
        "full_dim": 7,
        "groups": {"gripper": (0,), "arm": tuple(range(1, 7))},
    },
    "left_arm_action_at_gripper_tip": {
        "full_dim": 7,
        "groups": {"gripper": (0,), "arm": tuple(range(1, 7))},
    },
    "right_arm_action_at_gripper_tip": {
        "full_dim": 7,
        "groups": {"gripper": (0,), "arm": tuple(range(1, 7))},
    },
    # proprioception splits
    "chassis_T_left_eef_pose": {
        "full_dim": 7,
        "groups": {"gripper": (0,), "pose": tuple(range(1, 7))},
        "zzk_source": {"type": "single_arm_pose", "side": "left", "pose_row": 2},
    },
    "chassis_T_right_eef_pose": {
        "full_dim": 7,
        "groups": {"gripper": (0,), "pose": tuple(range(1, 7))},
        "zzk_source": {"type": "single_arm_pose", "side": "right", "pose_row": 4},
    },
    "chest_T_left_eef_pose": {
        "full_dim": 7,
        "groups": {"gripper": (0,), "pose": tuple(range(1, 7))},
        "zzk_source": {"type": "single_arm_pose", "side": "left", "pose_row": 5},
    },
    "chest_T_right_eef_pose": {
        "full_dim": 7,
        "groups": {"gripper": (0,), "pose": tuple(range(1, 7))},
        "zzk_source": {"type": "single_arm_pose", "side": "right", "pose_row": 6},
    },
    "chassis_T_left_gripper_tip_pose": {
        "full_dim": 7,
        "groups": {"gripper": (0,), "pose": tuple(range(1, 7))},
        "zzk_source": {"type": "single_arm_pose", "side": "left", "pose_row": 7},
    },
    "chassis_T_right_gripper_tip_pose": {
        "full_dim": 7,
        "groups": {"gripper": (0,), "pose": tuple(range(1, 7))},
        "zzk_source": {"type": "single_arm_pose", "side": "right", "pose_row": 8},
    },
    "chest_T_left_gripper_tip_pose": {
        "full_dim": 7,
        "groups": {"gripper": (0,), "pose": tuple(range(1, 7))},
        "zzk_source": {"type": "single_arm_pose", "side": "left", "pose_row": 9},
    },
    "chest_T_right_gripper_tip_pose": {
        "full_dim": 7,
        "groups": {"gripper": (0,), "pose": tuple(range(1, 7))},
        "zzk_source": {"type": "single_arm_pose", "side": "right", "pose_row": 10},
    },
    "left_wrench": {
        "full_dim": 6,
        "groups": {"wrench": tuple(range(6))},
        "zzk_source": {"type": "status_field_slice", "key": "wrench", "start": 0, "end": 6},
    },
    "right_wrench": {
        "full_dim": 6,
        "groups": {"wrench": tuple(range(6))},
        "zzk_source": {"type": "status_field_slice", "key": "wrench", "start": 6, "end": 12},
    },
}

# Combined dict used as the default base for build_runtime_layouts().
MMT_FIELD_LAYOUTS = {**_BASE_FIELD_LAYOUTS, **_LEGACY_PER_ARM_DEFAULTS}


def resolve_remap_indices(raw_indices: list) -> list[int]:
    """Resolve a list of ints and/or {start, end, step} dicts into flat indices."""
    result = []
    for item in raw_indices:
        if isinstance(item, dict):
            start = item.get("start", 0)
            end = item["end"]
            step = item.get("step", 1)
            result.extend(range(start, end, step))
        else:
            result.append(int(item))
    return result


def build_runtime_layouts(
    base_layouts: dict,
    preprocessing_config: dict,
) -> dict:
    """Merge base field layouts with auto-generated remap entries.

    For each entry in ``lowdim_field_remap``, a child layout is created with:
      - ``full_dim`` = number of remap indices
      - ``zzk_source`` with ``type: remap_slice`` (if the parent has a ``zzk_source``)

    Args:
        base_layouts: The static ``MMT_FIELD_LAYOUTS`` dict.
        preprocessing_config: Loaded from ``preprocessing_configs.yaml``.

    Returns:
        A new dict containing all base layouts plus auto-generated remap entries.
    """
    layouts = dict(base_layouts)

    remap_config = preprocessing_config.get("lowdim_field_remap") or {}
    for source_key, entries in remap_config.items():
        for entry in entries:
            child_name = entry["to"]
            indices = resolve_remap_indices(entry["indices"])
            child_layout = {"full_dim": len(indices)}

            parent_layout = base_layouts.get(source_key)
            if parent_layout and "zzk_source" in parent_layout:
                child_layout["zzk_source"] = {
                    "type": "remap_slice",
                    "parent": source_key,
                    "indices": indices,
                }

            layouts[child_name] = child_layout

    return layouts


def get_group_indices(field_name: str, group_name: str) -> tuple[int, ...]:
    if field_name not in MMT_FIELD_LAYOUTS:
        raise KeyError(f"Unknown MMT field layout '{field_name}'")
    groups = MMT_FIELD_LAYOUTS[field_name]["groups"]
    if group_name not in groups:
        raise KeyError(f"Unknown group '{group_name}' for MMT field layout '{field_name}'")
    return groups[group_name]
