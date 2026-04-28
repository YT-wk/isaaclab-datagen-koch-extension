from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

import carb

from .position_only_keyboard_device import PositionOnlyIKKeyboard, PositionOnlyIKKeyboardCfg


class MecanumPositionOnlyIKKeyboard(PositionOnlyIKKeyboard):
    """Keyboard device for mecanum-base + position-only IK arm teleoperation.

    The returned action vector is:
    ``[wheel_fl, wheel_fr, wheel_rl, wheel_rr, dx, dy, dz, wrist, gripper]``.

    Arm controls intentionally match :class:`PositionOnlyIKKeyboard` so the existing
    fixed-base keyboard workflow remains unchanged. Base controls are added on top:

    - Translation: Numpad ``8/2/4/6`` or arrow keys
    - Yaw: Numpad ``7/9``
    - Clear all keyboard state: ``L``
    """

    uses_keyboard_shortcuts = True

    def __init__(self, cfg: "MecanumPositionOnlyIKKeyboardCfg"):
        super().__init__(cfg)
        self._cfg = cfg
        self._base_command = np.zeros(3)
        self._wheel_velocity_signs = torch.tensor(cfg.wheel_velocity_signs, dtype=torch.float32)
        self._wheel_kinematics_span = float(cfg.wheel_base_half_length + cfg.wheel_base_half_width)

    def __str__(self) -> str:
        msg = f"Keyboard Controller for Mecanum Base + Position-Only IK: {self.__class__.__name__}\n"
        msg += f"\tKeyboard name: {self._input.get_keyboard_name(self._keyboard)}\n"
        msg += "\t----------------------------------------------\n"
        msg += "\tBase forward/backward: Numpad 8/2 or Arrow Up/Down\n"
        msg += "\tBase strafe right/left: Numpad 4/6 or Arrow Left/Right\n"
        msg += "\tBase yaw positive/negative: Numpad 7/9\n"
        msg += f"\tToggle gripper (open/close): {self._gripper_toggle_key}\n"
        msg += (
            f"\tMove arm along x-axis: {self._cfg.x_positive_key.upper()}/{self._cfg.x_negative_key.upper()}\n"
        )
        msg += (
            f"\tMove arm along y-axis: {self._cfg.y_positive_key.upper()}/{self._cfg.y_negative_key.upper()}\n"
        )
        msg += (
            f"\tMove arm along z-axis: {self._cfg.z_positive_key.upper()}/{self._cfg.z_negative_key.upper()}\n"
        )
        msg += (
            f"\tAdjust wrist joint: {self._cfg.wrist_positive_key.upper()}/{self._cfg.wrist_negative_key.upper()}\n"
        )
        msg += f"\tClear keyboard delta buffer: {self._clear_key}"
        return msg

    def reset(self):
        super().reset()
        self._base_command.fill(0.0)

    def advance(self) -> torch.Tensor:
        arm_command = super().advance()
        base_command = torch.tensor(self._base_command, dtype=torch.float32, device=self._sim_device)
        wheel_velocity_command = self._se2_to_mecanum_wheel_velocity(base_command)
        return torch.cat([wheel_velocity_command, arm_command], dim=0)

    def _on_keyboard_event(self, event, *args, **kwargs):
        key_name = self._extract_key_name(event)
        if key_name is None:
            return True

        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if key_name == self._clear_key:
                self.reset()
            if key_name == self._gripper_toggle_key:
                self._close_gripper = not self._close_gripper
            elif key_name in self._POS_INPUT_KEY_MAPPING:
                self._delta_pos += self._POS_INPUT_KEY_MAPPING[key_name]
            elif key_name in self._WRIST_INPUT_KEY_MAPPING:
                self._delta_wrist += self._WRIST_INPUT_KEY_MAPPING[key_name]
            elif key_name in self._BASE_INPUT_KEY_MAPPING:
                self._base_command += self._BASE_INPUT_KEY_MAPPING[key_name]

        if event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            if key_name in self._POS_INPUT_KEY_MAPPING:
                self._delta_pos -= self._POS_INPUT_KEY_MAPPING[key_name]
            elif key_name in self._WRIST_INPUT_KEY_MAPPING:
                self._delta_wrist -= self._WRIST_INPUT_KEY_MAPPING[key_name]
            elif key_name in self._BASE_INPUT_KEY_MAPPING:
                self._base_command -= self._BASE_INPUT_KEY_MAPPING[key_name]

        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if key_name in self._additional_callbacks:
                self._additional_callbacks[key_name]()

        return True

    def _create_key_bindings(self):
        super()._create_key_bindings()
        cfg = self._cfg
        self._BASE_INPUT_KEY_MAPPING = {
            "NUMPAD_8": np.asarray([1.0, 0.0, 0.0]) * cfg.base_vx_sensitivity,
            "UP": np.asarray([1.0, 0.0, 0.0]) * cfg.base_vx_sensitivity,
            "NUMPAD_2": np.asarray([-1.0, 0.0, 0.0]) * cfg.base_vx_sensitivity,
            "DOWN": np.asarray([-1.0, 0.0, 0.0]) * cfg.base_vx_sensitivity,
            "NUMPAD_4": np.asarray([0.0, 1.0, 0.0]) * cfg.base_vy_sensitivity,
            "LEFT": np.asarray([0.0, 1.0, 0.0]) * cfg.base_vy_sensitivity,
            "NUMPAD_6": np.asarray([0.0, -1.0, 0.0]) * cfg.base_vy_sensitivity,
            "RIGHT": np.asarray([0.0, -1.0, 0.0]) * cfg.base_vy_sensitivity,
            "NUMPAD_7": np.asarray([0.0, 0.0, 1.0]) * cfg.base_omega_sensitivity,
            "NUMPAD_9": np.asarray([0.0, 0.0, -1.0]) * cfg.base_omega_sensitivity,
        }

    def _se2_to_mecanum_wheel_velocity(self, base_command: torch.Tensor) -> torch.Tensor:
        vx, vy, wz = [float(value) for value in base_command.tolist()]
        yaw_term = self._wheel_kinematics_span * wz
        wheel_velocity = torch.tensor(
            [
                vx - vy - yaw_term,
                vx + vy + yaw_term,
                vx + vy - yaw_term,
                vx - vy + yaw_term,
            ],
            dtype=torch.float32,
            device=self._sim_device,
        )
        wheel_velocity = wheel_velocity / self._cfg.wheel_radius
        return wheel_velocity * self._wheel_velocity_signs.to(device=self._sim_device)


@dataclass
class MecanumPositionOnlyIKKeyboardCfg(PositionOnlyIKKeyboardCfg):
    """Configuration for the mecanum-base + position-only IK keyboard device."""

    base_vx_sensitivity: float = 0.4
    base_vy_sensitivity: float = 0.4
    base_omega_sensitivity: float = 0.8
    wheel_radius: float = 0.05
    wheel_base_half_length: float = 0.18
    wheel_base_half_width: float = 0.16
    wheel_velocity_signs: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    class_type: type = MecanumPositionOnlyIKKeyboard

    def __post_init__(self):
        if self.wheel_radius <= 0.0:
            raise ValueError(f"wheel_radius must be positive, got {self.wheel_radius}")
        if self.wheel_base_half_length < 0.0 or self.wheel_base_half_width < 0.0:
            raise ValueError(
                "wheel_base_half_length and wheel_base_half_width must be non-negative, "
                f"got {self.wheel_base_half_length} and {self.wheel_base_half_width}"
            )
        if len(self.wheel_velocity_signs) != 4:
            raise ValueError(
                "wheel_velocity_signs must contain four entries in front-left/front-right/rear-left/rear-right order, "
                f"got {len(self.wheel_velocity_signs)}"
            )
