from __future__ import annotations

import weakref
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch

import carb
import omni

from isaaclab.devices.device_base import DeviceBase, DeviceCfg


class PositionOnlyIKKeyboard(DeviceBase):
    """Keyboard device for position-only IK debugging.

    The device emits a 5D action vector:
    ``[dx, dy, dz, wrist, gripper]``.

    Compared with Isaac Lab's default ``Se3Keyboard``, this version intentionally removes
    rotational commands so it matches a position-only differential IK controller, while keeping
    one additional scalar channel for manually adjusting the last arm joint.
    """

    def __init__(self, cfg: PositionOnlyIKKeyboardCfg):
        self._cfg = cfg
        self.pos_sensitivity = cfg.pos_sensitivity
        self.wrist_sensitivity = cfg.wrist_sensitivity
        self.gripper_term = cfg.gripper_term
        self._sim_device = cfg.sim_device
        self._clear_key = cfg.clear_buffer_key.upper()
        self._gripper_toggle_key = cfg.gripper_toggle_key.upper()

        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        self._keyboard_sub = self._input.subscribe_to_keyboard_events(
            self._keyboard,
            lambda event, *args, obj=weakref.proxy(self): obj._on_keyboard_event(event, *args),
        )

        self._create_key_bindings()
        self._close_gripper = False
        self._delta_pos = np.zeros(3)
        self._delta_wrist = 0.0
        self._additional_callbacks: dict[str, Callable] = {}

    def __del__(self):
        self._input.unsubscribe_to_keyboard_events(self._keyboard, self._keyboard_sub)
        self._keyboard_sub = None

    def __str__(self) -> str:
        msg = f"Keyboard Controller for Position-Only IK: {self.__class__.__name__}\n"
        msg += f"\tKeyboard name: {self._input.get_keyboard_name(self._keyboard)}\n"
        msg += "\t----------------------------------------------\n"
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
        self._close_gripper = False
        self._delta_pos = np.zeros(3)
        self._delta_wrist = 0.0

    def add_callback(self, key: str, func: Callable):
        self._additional_callbacks[key.upper()] = func

    def advance(self) -> torch.Tensor:
        command = np.append(self._delta_pos.copy(), self._delta_wrist)
        if self.gripper_term:
            gripper_value = -1.0 if self._close_gripper else 1.0
            command = np.append(command, gripper_value)
        return torch.tensor(command, dtype=torch.float32, device=self._sim_device)

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

        if event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            if key_name in self._POS_INPUT_KEY_MAPPING:
                self._delta_pos -= self._POS_INPUT_KEY_MAPPING[key_name]
            elif key_name in self._WRIST_INPUT_KEY_MAPPING:
                self._delta_wrist -= self._WRIST_INPUT_KEY_MAPPING[key_name]

        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if key_name in self._additional_callbacks:
                self._additional_callbacks[key_name]()

        return True

    def _extract_key_name(self, event) -> str | None:
        """Normalize keyboard event input across different Kit/Carb event payload shapes."""
        raw_input = getattr(event, "input", None)
        if raw_input is None:
            return None
        if hasattr(raw_input, "name"):
            key_name = raw_input.name
        elif isinstance(raw_input, str):
            key_name = raw_input
        else:
            key_name = str(raw_input)
        return key_name.upper()

    def _create_key_bindings(self):
        cfg = self._cfg
        self._POS_INPUT_KEY_MAPPING = {
            cfg.x_positive_key.upper(): np.asarray([1.0, 0.0, 0.0]) * self.pos_sensitivity,
            cfg.x_negative_key.upper(): np.asarray([-1.0, 0.0, 0.0]) * self.pos_sensitivity,
            cfg.y_positive_key.upper(): np.asarray([0.0, 1.0, 0.0]) * self.pos_sensitivity,
            cfg.y_negative_key.upper(): np.asarray([0.0, -1.0, 0.0]) * self.pos_sensitivity,
            cfg.z_positive_key.upper(): np.asarray([0.0, 0.0, 1.0]) * self.pos_sensitivity,
            cfg.z_negative_key.upper(): np.asarray([0.0, 0.0, -1.0]) * self.pos_sensitivity,
        }
        self._WRIST_INPUT_KEY_MAPPING = {
            cfg.wrist_positive_key.upper(): self.wrist_sensitivity,
            cfg.wrist_negative_key.upper(): -self.wrist_sensitivity,
        }


@dataclass
class PositionOnlyIKKeyboardCfg(DeviceCfg):
    """Configuration for the position-only IK keyboard device."""

    gripper_term: bool = True
    pos_sensitivity: float = 0.05
    wrist_sensitivity: float = 0.05
    x_positive_key: str = "W"
    x_negative_key: str = "S"
    y_positive_key: str = "A"
    y_negative_key: str = "D"
    z_positive_key: str = "Q"
    z_negative_key: str = "E"
    wrist_positive_key: str = "Z"
    wrist_negative_key: str = "X"
    gripper_toggle_key: str = "K"
    clear_buffer_key: str = "L"
    retargeters: None = None
    class_type: type[DeviceBase] = PositionOnlyIKKeyboard
