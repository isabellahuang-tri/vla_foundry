"""Helper utilities for MMT inference lowdim decoding and ZZK action mapping."""

from enum import IntEnum

import numpy as np

from vla_foundry.inference.robotics.mmt.field_layouts import resolve_remap_indices


class ReferenceFrame(IntEnum):
    """Reference frame for position commands. Must match ZzkApiPositionReferenceFrame."""

    CHASSIS = 0
    CHEST = 1
    LOCAL = 2


REFERENCE_FRAME_MAP = {
    "chassis": ReferenceFrame.CHASSIS,
    "chest": ReferenceFrame.CHEST,
    "local": ReferenceFrame.LOCAL,
}


def _to_index_list(selection) -> list[int] | None:
    """Normalize a preprocessing index selection into a simple list form."""
    if selection is None:
        return None
    if isinstance(selection, int):
        return [selection]
    return [int(idx) for idx in selection]


def load_lowdim_index_selection(preprocessing_config: dict) -> dict[str, list[int] | None]:
    """Extract per-field index selections from a preprocessing config.

    Converts the ``mmt_lowdim_flatten_indices_selection`` block into a dict
    that maps each field name to a list of selected indices (or None).

    Example input (from preprocessing_configs.yaml)::

        {"mmt_lowdim_flatten_indices_selection": {
            "chest_T_eef_pose": [7, 8, 9, 10, 11, 12, 13],  # right arm only
            "base_action": [0, 1, 5],
            "lift_action": 2,                                 # single int
        }}

    Example output::

        {"chest_T_eef_pose": [7, 8, 9, 10, 11, 12, 13],
         "base_action": [0, 1, 5],
         "lift_action": [2]}
    """
    index_selection_config = preprocessing_config.get("mmt_lowdim_flatten_indices_selection") or {}
    return {name: _to_index_list(sel) for name, sel in index_selection_config.items()}


def load_lowdim_field_remap(preprocessing_config: dict) -> dict[str, tuple[str, list[int]]]:
    """Extract per-field remap info from a preprocessing config.

    Returns a dict mapping each remapped child field name to
    ``(parent_field_name, list_of_indices_into_parent)``.

    Example input (from preprocessing_configs.yaml)::

        {"lowdim_field_remap": {
            "arm_action": [
                {"to": "left_arm_action", "indices": [0, 1, 2, 3, 4, 5, 6]},
                {"to": "right_arm_action", "indices": [7, 8, 9, 10, 11, 12, 13]},
            ]
        }}

    Example output::

        {"left_arm_action": ("arm_action", [0, 1, 2, 3, 4, 5, 6]),
         "right_arm_action": ("arm_action", [7, 8, 9, 10, 11, 12, 13])}
    """
    remap_config = preprocessing_config.get("lowdim_field_remap") or {}
    result = {}
    for source_key, entries in remap_config.items():
        for entry in entries:
            result[entry["to"]] = (source_key, resolve_remap_indices(entry["indices"]))
    return result


def restore_full_values(values: np.ndarray, selection: list[int] | None, full_dim: int) -> np.ndarray:
    """Scatter selected values back into a zero-filled array of the original full dimension.

    Inverse of index selection: places each element of ``values`` at the
    corresponding position in ``selection``, leaving other positions as zero.

    Example::

        >>> restore_full_values(np.array([0.1, -0.2, 0.3]), [0, 1, 5], full_dim=6)
        array([0.1, -0.2, 0.0, 0.0, 0.0, 0.3], dtype=float32)
    """
    if selection is None:
        if len(values) == full_dim:
            return values.astype(np.float32, copy=False)
        raise ValueError(f"Cannot scatter values of dim {len(values)} into full dim {full_dim} without a selection.")

    full_values = np.zeros(full_dim, dtype=np.float32)
    if len(values) != len(selection):
        raise ValueError(f"Selection length {len(selection)} does not match values dim {len(values)}.")
    full_values[np.asarray(selection, dtype=np.int64)] = values
    return full_values


def _get_position_cmd_config(field_name: str) -> tuple[str, int, str] | None:
    """Derive position command config from field_layouts or field name convention.

    Returns (part_name, reference_frame, tcp) or None if not a position field.

    First checks for an explicit ``zzk_position_cmd`` entry in the layout.
    Otherwise, parses the field name convention:
        {frame}_T_{side}_{tcp_type}_pose
    Examples:
        chassis_T_left_eef_pose        -> ("left_arm",  CHASSIS, "arm_tip")
        chest_T_right_gripper_tip_pose -> ("right_arm", CHEST,   "gripper_tip")
    """
    from vla_foundry.inference.robotics.mmt.field_layouts import MMT_FIELD_LAYOUTS

    layout = MMT_FIELD_LAYOUTS.get(field_name)

    # Explicit config takes priority.
    if layout is not None and "zzk_position_cmd" in layout:
        cmd = layout["zzk_position_cmd"]
        ref_frame = REFERENCE_FRAME_MAP[cmd["reference_frame"]]
        return cmd["part_name"], ref_frame, cmd["tcp"]

    # Derive from naming convention: {frame}_T_{side}_{tcp_type}_pose
    if not field_name.endswith("_pose"):
        return None
    parts = field_name.split("_T_", 1)
    if len(parts) != 2:
        return None
    frame_str = parts[0]  # "chassis" or "chest"
    rest = parts[1]  # "left_eef_pose" or "right_gripper_tip_pose"
    if frame_str not in REFERENCE_FRAME_MAP:
        return None
    ref_frame = REFERENCE_FRAME_MAP[frame_str]
    if rest.startswith("left_"):
        side = "left"
        remainder = rest[len("left_") :]
    elif rest.startswith("right_"):
        side = "right"
        remainder = rest[len("right_") :]
    else:
        return None
    part_name = f"{side}_arm"
    if "gripper_tip" in remainder:
        tcp = "gripper_tip"
    elif "eef" in remainder:
        tcp = "arm_tip"
    else:
        return None
    return part_name, ref_frame, tcp


def is_position_action_field(field_name: str) -> bool:
    """Check if an action field is a position (pose) command rather than velocity."""
    return _get_position_cmd_config(field_name) is not None


class MmtActionMapper:
    """Maps model action fields into ZZK command dictionaries."""

    def __init__(
        self,
        lowdim_index_selection: dict[str, list[int] | None],
        runtime_layouts: dict | None = None,
    ):
        self.lowdim_index_selection = lowdim_index_selection
        self.runtime_layouts = runtime_layouts or {}
        self.action_field_handlers = {
            "left_arm_action": self.append_left_arm_action_command,
            "right_arm_action": self.append_right_arm_action_command,
            "left_arm_action_at_gripper_tip": self.append_left_arm_action_at_gripper_tip_command,
            "right_arm_action_at_gripper_tip": self.append_right_arm_action_at_gripper_tip_command,
            "base_action": self.append_base_action_command,
            "head_action": self.append_head_action_command,
            "lift_action": self.append_lift_action_command,
        }

    def decode_full_field_action(self, field_name: str, action_values: np.ndarray) -> np.ndarray:
        layout = self.runtime_layouts.get(field_name)
        if layout is None:
            raise KeyError(
                f"Unknown MMT field layout '{field_name}' "
                f"(not in runtime_layouts — check MMT_FIELD_LAYOUTS and lowdim_field_remap)"
            )
        full_dim = layout["full_dim"]
        selection = self.lowdim_index_selection.get(field_name)
        return restore_full_values(action_values, selection, full_dim)

    def _append_single_arm_command(
        self,
        field_name: str,
        side_name: str,
        zzk_arm_key: str,
        action_values: np.ndarray,
        zzk_action: dict,
        debug_parts: list[str],
    ) -> None:
        full_action = self.decode_full_field_action(field_name, action_values)
        gripper = float(full_action[0])
        arm = full_action[1:]
        zzk_action[zzk_arm_key] = arm.tolist()
        zzk_action[f"{side_name}_gripper"] = gripper
        debug_parts.append(f"{side_name}[{zzk_arm_key}](vx={arm[0]:.3f}, vy={arm[1]:.3f}, gripper={gripper:.3f})")

    def append_left_arm_action_command(
        self,
        action_values: np.ndarray,
        zzk_action: dict,
        debug_parts: list[str],
    ) -> None:
        self._append_single_arm_command(
            "left_arm_action",
            "left",
            "left_arm",
            action_values,
            zzk_action,
            debug_parts,
        )

    def append_right_arm_action_command(
        self,
        action_values: np.ndarray,
        zzk_action: dict,
        debug_parts: list[str],
    ) -> None:
        self._append_single_arm_command(
            "right_arm_action",
            "right",
            "right_arm",
            action_values,
            zzk_action,
            debug_parts,
        )

    def append_left_arm_action_at_gripper_tip_command(
        self,
        action_values: np.ndarray,
        zzk_action: dict,
        debug_parts: list[str],
    ) -> None:
        self._append_single_arm_command(
            "left_arm_action_at_gripper_tip",
            "left",
            "left_arm",
            action_values,
            zzk_action,
            debug_parts,
        )

    def append_right_arm_action_at_gripper_tip_command(
        self,
        action_values: np.ndarray,
        zzk_action: dict,
        debug_parts: list[str],
    ) -> None:
        self._append_single_arm_command(
            "right_arm_action_at_gripper_tip",
            "right",
            "right_arm",
            action_values,
            zzk_action,
            debug_parts,
        )

    def append_base_action_command(self, action_values: np.ndarray, zzk_action: dict, debug_parts: list[str]) -> None:
        full_base_action = self.decode_full_field_action("base_action", action_values)
        zzk_action["chassis"] = full_base_action.tolist()
        debug_parts.append(
            f"chassis(vx={full_base_action[0]:.3f}, vy={full_base_action[1]:.3f}, yaw={full_base_action[5]:.3f})"
        )

    def append_head_action_command(self, action_values: np.ndarray, zzk_action: dict, debug_parts: list[str]) -> None:
        full_head_action = self.decode_full_field_action("head_action", action_values)
        zzk_action["head"] = full_head_action.tolist()
        debug_parts.append(
            f"head(roll={full_head_action[3]:.3f}, pitch={full_head_action[4]:.3f}, yaw={full_head_action[5]:.3f})"
        )

    def append_lift_action_command(self, action_values: np.ndarray, zzk_action: dict, debug_parts: list[str]) -> None:
        full_lift_action = self.decode_full_field_action("lift_action", action_values)
        zzk_action["lift"] = full_lift_action.tolist()
        debug_parts.append(f"lift(z={full_lift_action[2]:.3f})")

    def build_position_command(self, field_name: str, action_values: np.ndarray) -> tuple[dict, str, dict]:
        """Build a ZZK position command dict for a pose action field.

        Returns (position_action, tcp, gripper_action) where:
          - position_action: arm pose payload for send_position_command.
          - tcp: tool center point string.
          - gripper_action: scalar gripper target keyed by ``{side}_gripper``,
            merged into position_action before send_position_command. The
            server routes the scalar to the correct joint axis using its
            kinematic model.
        The 7-dim action is [gripper, x, y, z, rx, ry, rz].
        """
        config = _get_position_cmd_config(field_name)
        if config is None:
            raise ValueError(f"No position command config for '{field_name}'")
        part_name, reference_frame, tcp = config
        full_action = self.decode_full_field_action(field_name, action_values)
        gripper = float(full_action[0])
        pose = full_action[1:].tolist()
        side = "left" if "left" in part_name else "right"
        position_action = {
            part_name: {"pose": pose, "reference_frame": reference_frame},
        }
        gripper_action = {f"{side}_gripper": gripper}
        return position_action, tcp, gripper_action
