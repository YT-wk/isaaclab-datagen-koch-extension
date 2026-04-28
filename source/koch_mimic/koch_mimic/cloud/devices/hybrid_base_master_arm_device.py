"""Hybrid teleop device for keyboard-driven mecanum base and streamed Koch master arm."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from isaaclab.devices import Se2Keyboard, Se2KeyboardCfg
from isaaclab.devices.device_base import DeviceBase, DeviceCfg

from .master_arm_stream_device import KochMasterArmStreamDevice, KochMasterArmStreamDeviceCfg


@dataclass
class KochHybridKeyboardMasterArmDeviceCfg(DeviceCfg):
    """Configuration for the hybrid mecanum-base and master-arm teleop device."""

    base_keyboard: Se2KeyboardCfg = field(default_factory=Se2KeyboardCfg)
    master_arm: KochMasterArmStreamDeviceCfg = field(default_factory=KochMasterArmStreamDeviceCfg)
    wheel_radius: float = 0.05
    wheel_base_half_length: float = 0.18
    wheel_base_half_width: float = 0.16
    wheel_velocity_signs: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    retargeters: list = field(default_factory=list)
    class_type: type[DeviceBase] | None = None

    def __post_init__(self) -> None:
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
        if self.class_type is None:
            self.class_type = KochHybridKeyboardMasterArmDevice


class KochHybridKeyboardMasterArmDevice(DeviceBase):
    """Combine keyboard base control with streamed master-arm control.

    The returned action vector is:
    ``[wheel_fl, wheel_fr, wheel_rl, wheel_rr, arm_j1, arm_j2, arm_j3, arm_j4, arm_j5, gripper]``.
    """

    uses_keyboard_shortcuts = True

    def __init__(self, cfg: KochHybridKeyboardMasterArmDeviceCfg):
        super().__init__(retargeters=None)
        self._cfg = cfg
        self._sim_device = cfg.sim_device
        self._callbacks: dict[str, Any] = {}

        # Keep the child devices on the same torch device as the environment.
        cfg.base_keyboard.sim_device = cfg.sim_device
        cfg.master_arm.sim_device = cfg.sim_device

        self._base_keyboard = Se2Keyboard(cfg.base_keyboard)
        self._master_arm = KochMasterArmStreamDevice(cfg.master_arm)
        self._wheel_velocity_signs = torch.tensor(cfg.wheel_velocity_signs, dtype=torch.float32)
        self._wheel_kinematics_span = float(cfg.wheel_base_half_length + cfg.wheel_base_half_width)

    def __del__(self):
        self.close()

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}(\n"
            f"  base={self._base_keyboard},\n"
            f"  arm={self._master_arm},\n"
            f"  wheel_radius={self._cfg.wheel_radius},\n"
            f"  wheel_base_half_length={self._cfg.wheel_base_half_length},\n"
            f"  wheel_base_half_width={self._cfg.wheel_base_half_width}\n"
            f")"
        )

    def reset(self):
        self._base_keyboard.reset()
        self._master_arm.reset()

    def close(self):
        if hasattr(self, "_base_keyboard") and self._base_keyboard is not None:
            try:
                del self._base_keyboard
            except Exception:
                pass
            self._base_keyboard = None
        if hasattr(self, "_master_arm") and self._master_arm is not None:
            self._master_arm.close()
            self._master_arm = None

    def add_callback(self, key: str, func):
        key_name = str(key).upper()
        self._callbacks[key_name] = func
        self._base_keyboard.add_callback(key_name, func)

    def advance(self) -> torch.Tensor:
        base_command = self._base_keyboard.advance().to(device=self._sim_device)
        arm_command = self._master_arm.advance().to(device=self._sim_device)
        wheel_velocity_command = self._se2_to_mecanum_wheel_velocity(base_command)
        return torch.cat([wheel_velocity_command, arm_command], dim=0)

    def _se2_to_mecanum_wheel_velocity(self, base_command: torch.Tensor) -> torch.Tensor:
        """Map base-frame ``[vx, vy, wz]`` commands to four wheel velocity targets."""
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


KochHybridKeyboardMasterArmDeviceCfg.class_type = KochHybridKeyboardMasterArmDevice
