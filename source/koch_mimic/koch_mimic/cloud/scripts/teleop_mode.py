"""Shared teleoperation mode helpers for the custom Koch scripts."""

from __future__ import annotations

import argparse
from dataclasses import dataclass


FIXED_KEYBOARD_TELEOP_DEVICE_NAME = "keyboard"
MOBILE_KEYBOARD_TELEOP_DEVICE_NAME = "keyboard_mecanum"
FIXED_REMOTE_MASTER_ARM_TELEOP_DEVICE_NAME = "external_master_arm"
MOBILE_REMOTE_MASTER_ARM_TELEOP_DEVICE_NAME = "external_master_arm_mecanum"


@dataclass(frozen=True)
class ResolvedTeleopMode:
    """Resolved teleoperation mode selected from the high-level CLI flags."""

    device_name: str
    fixed_base: bool
    arm_source: str

    @property
    def mobile_base(self) -> bool:
        return not self.fixed_base

    @property
    def remote_master_arm(self) -> bool:
        return self.arm_source == "remote_master_arm"


@dataclass(frozen=True)
class ResolvedMecanumBaseCfg:
    """Fully resolved mecanum-base control parameters."""

    wheel_joint_names: tuple[str, ...]
    wheel_velocity_signs: tuple[float, float, float, float]
    wheel_radius: float
    wheel_half_length: float
    wheel_half_width: float
    keyboard_cfg: object


def add_teleop_mode_args(parser: argparse.ArgumentParser) -> None:
    """Add the shared two-parameter teleop mode selection interface."""
    parser.add_argument(
        "--teleop_fixed_base",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override whether the robot base is fixed during teleoperation.",
    )
    parser.add_argument(
        "--arm_teleop_source",
        type=str,
        choices=("keyboard", "remote_master_arm"),
        default=None,
        help="Override whether the mounted arm is teleoperated from the keyboard or the remote master arm.",
    )


def add_remote_master_arm_args(parser: argparse.ArgumentParser) -> None:
    """Add the shared remote-master-arm bridge arguments."""
    parser.add_argument("--stream-host", type=str, default=None, help="TCP host to listen on.")
    parser.add_argument("--stream-port", type=int, default=None, help="TCP port to listen on.")
    parser.add_argument(
        "--joint-signs",
        type=str,
        default=None,
        help="Comma-separated sign flips for the 5 arm joints, for example: 1,-1,1,1,-1",
    )
    parser.add_argument(
        "--joint-offsets",
        type=str,
        default=None,
        help="Comma-separated additive offsets in radians for the 5 arm joints.",
    )
    parser.add_argument(
        "--gripper-close-delta",
        "--gripper-close-threshold",
        dest="gripper_close_delta",
        type=float,
        default=None,
        help=(
            "Leader gripper travel in radians that corresponds to moving from fully open to fully closed. "
            "The legacy alias --gripper-close-threshold is still accepted."
        ),
    )
    parser.add_argument(
        "--gripper-close-direction",
        type=str,
        choices=("positive", "negative"),
        default=None,
        help="Override whether the leader gripper closes when its angle moves in the positive or negative direction.",
    )
    parser.add_argument(
        "--zero-on-start",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Capture the first streamed frame as the leader home pose.",
    )
    parser.add_argument(
        "--stale-timeout",
        type=float,
        default=None,
        help="Warn and hold the last action when no fresh frame arrives within this time.",
    )


def add_mobile_base_args(parser: argparse.ArgumentParser) -> None:
    """Add the shared mobile-base configuration arguments."""
    parser.add_argument(
        "--base-wheel-joint-names",
        type=str,
        default=None,
        help="Comma-separated mecanum wheel joint names in front-left,front-right,rear-left,rear-right order.",
    )
    parser.add_argument(
        "--base-wheel-signs",
        type=str,
        default=None,
        help="Optional sign flips for the four wheel velocity targets in front-left,front-right,rear-left,rear-right order.",
    )
    parser.add_argument(
        "--base-wheel-radius",
        type=float,
        default=None,
        help="Wheel radius in meters for mecanum velocity conversion.",
    )
    parser.add_argument(
        "--base-wheel-half-length",
        type=float,
        default=None,
        help="Half of the front-to-rear wheel separation in meters.",
    )
    parser.add_argument(
        "--base-wheel-half-width",
        type=float,
        default=None,
        help="Half of the left-to-right wheel separation in meters.",
    )
    parser.add_argument(
        "--base-vx-sensitivity",
        type=float,
        default=None,
        help="Keyboard forward/backward speed scale for the mobile-base modes.",
    )
    parser.add_argument(
        "--base-vy-sensitivity",
        type=float,
        default=None,
        help="Keyboard lateral speed scale for the mobile-base modes.",
    )
    parser.add_argument(
        "--base-omega-sensitivity",
        type=float,
        default=None,
        help="Keyboard yaw speed scale for the mobile-base modes.",
    )


def parse_csv_floats(value: str, expected_len: int) -> tuple[float, ...]:
    """Parse a fixed-length comma-separated float list."""
    parts = [item.strip() for item in value.split(",") if item.strip()]
    if len(parts) != expected_len:
        raise ValueError(f"Expected {expected_len} comma-separated values, got {len(parts)} from: {value}")
    return tuple(float(item) for item in parts)


def parse_csv_strings(value: str, expected_len: int) -> tuple[str, ...]:
    """Parse a fixed-length comma-separated string list."""
    parts = tuple(item.strip() for item in value.split(",") if item.strip())
    if len(parts) != expected_len:
        raise ValueError(f"Expected {expected_len} comma-separated values, got {len(parts)} from: {value}")
    return parts


def normalize_arm_teleop_source(value: str) -> str:
    """Normalize arm teleop source names from CLI or config."""
    normalized = value.strip().lower()
    if normalized == "external_master_arm":
        normalized = "remote_master_arm"
    if normalized not in ("keyboard", "remote_master_arm"):
        raise ValueError(f"Unsupported arm teleop source: {value}")
    return normalized


def resolve_teleop_mode(args_cli, env_cfg) -> ResolvedTeleopMode:
    """Resolve the selected teleop mode from CLI overrides and env defaults."""
    teleop_fixed_base = (
        bool(args_cli.teleop_fixed_base)
        if args_cli.teleop_fixed_base is not None
        else bool(getattr(env_cfg, "teleop_fixed_base", True))
    )
    arm_source = (
        normalize_arm_teleop_source(args_cli.arm_teleop_source)
        if args_cli.arm_teleop_source is not None
        else normalize_arm_teleop_source(str(getattr(env_cfg, "teleop_arm_source", "keyboard")))
    )

    if arm_source == "keyboard":
        device_name = FIXED_KEYBOARD_TELEOP_DEVICE_NAME if teleop_fixed_base else MOBILE_KEYBOARD_TELEOP_DEVICE_NAME
    else:
        device_name = (
            FIXED_REMOTE_MASTER_ARM_TELEOP_DEVICE_NAME
            if teleop_fixed_base
            else MOBILE_REMOTE_MASTER_ARM_TELEOP_DEVICE_NAME
        )
    return ResolvedTeleopMode(device_name=device_name, fixed_base=teleop_fixed_base, arm_source=arm_source)


def resolve_mecanum_base_cfg(args_cli, env_cfg) -> ResolvedMecanumBaseCfg:
    """Resolve mecanum-base control parameters for the mobile-base modes."""
    from isaaclab.devices import Se2KeyboardCfg

    if args_cli.base_wheel_joint_names is not None:
        wheel_joint_names = parse_csv_strings(args_cli.base_wheel_joint_names, 4)
    else:
        wheel_joint_names = tuple(getattr(env_cfg, "koch_base_wheel_joint_names", ()))
    if len(wheel_joint_names) != 4:
        raise ValueError(
            "Mobile-base teleoperation requires four wheel joint names in "
            "front-left/front-right/rear-left/rear-right order. "
            f"Got: {wheel_joint_names}"
        )

    if args_cli.base_wheel_signs is not None:
        wheel_velocity_signs = parse_csv_floats(args_cli.base_wheel_signs, 4)
    else:
        wheel_velocity_signs = tuple(getattr(env_cfg, "koch_base_wheel_velocity_signs", (1.0, 1.0, 1.0, 1.0)))
    if len(wheel_velocity_signs) != 4:
        raise ValueError(f"Expected four wheel signs, got: {wheel_velocity_signs}")

    wheel_radius = (
        float(args_cli.base_wheel_radius)
        if args_cli.base_wheel_radius is not None
        else float(getattr(env_cfg, "koch_base_wheel_radius_m", 0.05))
    )
    wheel_half_length = (
        float(args_cli.base_wheel_half_length)
        if args_cli.base_wheel_half_length is not None
        else float(getattr(env_cfg, "koch_base_wheel_half_length_m", 0.18))
    )
    wheel_half_width = (
        float(args_cli.base_wheel_half_width)
        if args_cli.base_wheel_half_width is not None
        else float(getattr(env_cfg, "koch_base_wheel_half_width_m", 0.16))
    )

    keyboard_cfg = Se2KeyboardCfg(
        v_x_sensitivity=(
            float(args_cli.base_vx_sensitivity)
            if args_cli.base_vx_sensitivity is not None
            else float(getattr(env_cfg, "teleop_base_vx_sensitivity", 0.4))
        ),
        v_y_sensitivity=(
            float(args_cli.base_vy_sensitivity)
            if args_cli.base_vy_sensitivity is not None
            else float(getattr(env_cfg, "teleop_base_vy_sensitivity", 0.4))
        ),
        omega_z_sensitivity=(
            float(args_cli.base_omega_sensitivity)
            if args_cli.base_omega_sensitivity is not None
            else float(getattr(env_cfg, "teleop_base_omega_sensitivity", 0.8))
        ),
        sim_device=args_cli.device,
    )
    return ResolvedMecanumBaseCfg(
        wheel_joint_names=wheel_joint_names,
        wheel_velocity_signs=wheel_velocity_signs,  # type: ignore[arg-type]
        wheel_radius=wheel_radius,
        wheel_half_length=wheel_half_length,
        wheel_half_width=wheel_half_width,
        keyboard_cfg=keyboard_cfg,
    )


def resolve_external_master_arm_gripper_cfg(args_cli, env_cfg) -> tuple[float, float, float, str]:
    """Resolve gripper calibration for the remote master-arm path."""
    open_cmd = float(env_cfg.koch_gripper_open_command)
    close_cmd = float(env_cfg.koch_gripper_close_command)

    default_close_delta = getattr(env_cfg, "external_master_arm_gripper_close_delta", None)
    if default_close_delta is None:
        default_close_delta = abs(close_cmd - open_cmd)
        if default_close_delta <= 0.0:
            default_close_delta = 1.0
    close_delta = float(args_cli.gripper_close_delta) if args_cli.gripper_close_delta is not None else float(default_close_delta)

    default_direction = getattr(env_cfg, "external_master_arm_gripper_close_direction", "negative")
    close_direction = args_cli.gripper_close_direction if args_cli.gripper_close_direction is not None else str(default_direction)

    return open_cmd, close_cmd, close_delta, close_direction


def make_master_arm_device_cfg(args_cli, env_cfg):
    """Create the streamed master-arm device config for remote-arm teleop."""
    from koch_mimic.cloud.devices.master_arm_stream_device import (
        KochMasterArmStreamDevice,
        KochMasterArmStreamDeviceCfg,
    )

    joint_signs = (
        parse_csv_floats(args_cli.joint_signs, 5)
        if args_cli.joint_signs is not None
        else tuple(float(value) for value in getattr(env_cfg, "external_master_arm_joint_signs", (1, 1, 1, 1, 1)))
    )
    joint_offsets = (
        parse_csv_floats(args_cli.joint_offsets, 5)
        if args_cli.joint_offsets is not None
        else tuple(
            float(value) for value in getattr(env_cfg, "external_master_arm_joint_offsets", (0, 0, 0, 0, 0))
        )
    )
    gripper_open_command, gripper_close_command, gripper_close_delta, gripper_close_direction = (
        resolve_external_master_arm_gripper_cfg(args_cli, env_cfg)
    )
    stream_host = (
        str(args_cli.stream_host)
        if args_cli.stream_host is not None
        else str(getattr(env_cfg, "external_master_arm_stream_host", "127.0.0.1"))
    )
    stream_port = (
        int(args_cli.stream_port)
        if args_cli.stream_port is not None
        else int(getattr(env_cfg, "external_master_arm_stream_port", 55000))
    )
    zero_on_first_frame = (
        bool(args_cli.zero_on_start)
        if args_cli.zero_on_start is not None
        else bool(getattr(env_cfg, "external_master_arm_zero_on_first_frame", True))
    )
    stale_timeout = (
        float(args_cli.stale_timeout)
        if args_cli.stale_timeout is not None
        else float(getattr(env_cfg, "external_master_arm_stale_timeout", 1.0))
    )
    return KochMasterArmStreamDeviceCfg(
        sim_device=args_cli.device,
        host=stream_host,
        port=stream_port,
        joint_signs=joint_signs,
        joint_offsets=joint_offsets,
        zero_on_first_frame=zero_on_first_frame,
        gripper_open_command=gripper_open_command,
        gripper_close_command=gripper_close_command,
        gripper_close_delta=gripper_close_delta,
        gripper_close_direction=gripper_close_direction,
        stale_timeout=stale_timeout,
        class_type=KochMasterArmStreamDevice,
    )


def configure_teleop_env(env_cfg, args_cli) -> ResolvedTeleopMode:
    """Inject the requested teleop mode into the environment config."""
    from isaaclab.envs import ManagerBasedRLEnvCfg
    from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg, JointVelocityActionCfg
    from koch_mimic.cloud.tasks.koch_pick_place.mecanum_position_only_keyboard_device import (
        MecanumPositionOnlyIKKeyboardCfg,
    )

    from koch_mimic.cloud.devices.hybrid_base_master_arm_device import (
        KochHybridKeyboardMasterArmDevice,
        KochHybridKeyboardMasterArmDeviceCfg,
    )

    if not hasattr(env_cfg, "teleop_devices"):
        raise AttributeError("The selected environment config does not expose teleop_devices.")

    teleop_mode = resolve_teleop_mode(args_cli, env_cfg)
    mecanum_base_cfg: ResolvedMecanumBaseCfg | None = None

    if teleop_mode.mobile_base:
        if not isinstance(env_cfg, ManagerBasedRLEnvCfg):
            raise ValueError(f"{teleop_mode.device_name} is only supported for ManagerBasedRLEnv environments.")

        mecanum_base_cfg = resolve_mecanum_base_cfg(args_cli, env_cfg)
        env_cfg.robot_fix_root_link = False
        if hasattr(env_cfg, "scene") and hasattr(env_cfg.scene, "robot"):
            robot_spawn = getattr(env_cfg.scene.robot, "spawn", None)
            if robot_spawn is not None and getattr(robot_spawn, "articulation_props", None) is not None:
                robot_spawn.articulation_props.fix_root_link = False
        env_cfg.actions.base_action = JointVelocityActionCfg(
            asset_name="robot",
            joint_names=list(mecanum_base_cfg.wheel_joint_names),
            scale=1.0,
            offset=0.0,
            preserve_order=True,
            use_default_offset=False,
        )
    else:
        env_cfg.actions.base_action = None

    if teleop_mode.device_name == MOBILE_KEYBOARD_TELEOP_DEVICE_NAME:
        assert mecanum_base_cfg is not None
        env_cfg.teleop_devices.devices[MOBILE_KEYBOARD_TELEOP_DEVICE_NAME] = MecanumPositionOnlyIKKeyboardCfg(
            pos_sensitivity=float(getattr(env_cfg, "teleop_pos_sensitivity", 0.05)),
            wrist_sensitivity=float(getattr(env_cfg, "teleop_wrist_sensitivity", 0.05)),
            x_positive_key=str(getattr(env_cfg, "teleop_x_positive_key", "D")),
            x_negative_key=str(getattr(env_cfg, "teleop_x_negative_key", "A")),
            y_positive_key=str(getattr(env_cfg, "teleop_y_positive_key", "W")),
            y_negative_key=str(getattr(env_cfg, "teleop_y_negative_key", "S")),
            z_positive_key=str(getattr(env_cfg, "teleop_z_positive_key", "Q")),
            z_negative_key=str(getattr(env_cfg, "teleop_z_negative_key", "E")),
            wrist_positive_key=str(getattr(env_cfg, "teleop_wrist_positive_key", "Z")),
            wrist_negative_key=str(getattr(env_cfg, "teleop_wrist_negative_key", "X")),
            gripper_toggle_key=str(getattr(env_cfg, "teleop_gripper_toggle_key", "K")),
            clear_buffer_key=str(getattr(env_cfg, "teleop_clear_buffer_key", "L")),
            base_vx_sensitivity=mecanum_base_cfg.keyboard_cfg.v_x_sensitivity,
            base_vy_sensitivity=mecanum_base_cfg.keyboard_cfg.v_y_sensitivity,
            base_omega_sensitivity=mecanum_base_cfg.keyboard_cfg.omega_z_sensitivity,
            wheel_radius=mecanum_base_cfg.wheel_radius,
            wheel_base_half_length=mecanum_base_cfg.wheel_half_length,
            wheel_base_half_width=mecanum_base_cfg.wheel_half_width,
            wheel_velocity_signs=mecanum_base_cfg.wheel_velocity_signs,
            sim_device=args_cli.device,
        )
        return teleop_mode

    if not teleop_mode.remote_master_arm:
        return teleop_mode

    if not isinstance(env_cfg, ManagerBasedRLEnvCfg):
        raise ValueError(f"{teleop_mode.device_name} is only supported for ManagerBasedRLEnv environments.")
    if not hasattr(env_cfg, "koch_arm_joint_names") or not hasattr(env_cfg, "koch_gripper_joint_names"):
        raise AttributeError(
            "The selected environment config does not expose 'koch_arm_joint_names' and "
            "'koch_gripper_joint_names', so it cannot be controlled by the Koch master arm bridge."
        )

    master_arm_device_cfg = make_master_arm_device_cfg(args_cli, env_cfg)
    gripper_open_command = float(env_cfg.koch_gripper_open_command)
    gripper_close_command = float(env_cfg.koch_gripper_close_command)

    if teleop_mode.device_name == MOBILE_REMOTE_MASTER_ARM_TELEOP_DEVICE_NAME:
        assert mecanum_base_cfg is not None
        env_cfg.teleop_devices.devices[MOBILE_REMOTE_MASTER_ARM_TELEOP_DEVICE_NAME] = KochHybridKeyboardMasterArmDeviceCfg(
            sim_device=args_cli.device,
            base_keyboard=mecanum_base_cfg.keyboard_cfg,
            master_arm=master_arm_device_cfg,
            wheel_radius=mecanum_base_cfg.wheel_radius,
            wheel_base_half_length=mecanum_base_cfg.wheel_half_length,
            wheel_base_half_width=mecanum_base_cfg.wheel_half_width,
            wheel_velocity_signs=mecanum_base_cfg.wheel_velocity_signs,
            class_type=KochHybridKeyboardMasterArmDevice,
        )
    else:
        env_cfg.teleop_devices.devices[FIXED_REMOTE_MASTER_ARM_TELEOP_DEVICE_NAME] = master_arm_device_cfg

    env_cfg.actions.arm_action = JointPositionActionCfg(
        asset_name="robot",
        joint_names=list(env_cfg.koch_arm_joint_names),
        scale=1.0,
        offset=0.0,
        preserve_order=True,
        use_default_offset=False,
    )
    env_cfg.actions.wrist_action = None
    env_cfg.actions.gripper_action = JointPositionActionCfg(
        asset_name="robot",
        joint_names=list(env_cfg.koch_gripper_joint_names),
        scale=1.0,
        offset=0.0,
        preserve_order=True,
        use_default_offset=False,
        clip={
            name: (min(gripper_open_command, gripper_close_command), max(gripper_open_command, gripper_close_command))
            for name in env_cfg.koch_gripper_joint_names
        },
    )
    return teleop_mode


def teleop_mode_startup_messages(teleop_mode: ResolvedTeleopMode, args_cli) -> list[str]:
    """Build user-facing startup messages for the selected mode."""
    messages: list[str] = []
    if teleop_mode.remote_master_arm:
        messages.append(
            "Waiting for master-arm stream on "
            f"{args_cli.stream_host}:{args_cli.stream_port}. "
            "Start stream_koch_leader_over_ssh.py on the local machine after this script is ready."
        )
    if teleop_mode.device_name == MOBILE_REMOTE_MASTER_ARM_TELEOP_DEVICE_NAME:
        messages.append("Hybrid mode enabled: keyboard controls the mecanum base while the remote master arm controls the arm.")
    elif teleop_mode.device_name == MOBILE_KEYBOARD_TELEOP_DEVICE_NAME:
        messages.append("Hybrid keyboard mode enabled: keyboard controls both the mecanum base and the mounted arm.")
    return messages
