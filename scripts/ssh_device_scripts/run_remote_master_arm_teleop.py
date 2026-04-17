"""Cloud-side teleop entry for the custom Koch environment."""

from __future__ import annotations

import argparse
import logging
import os
from collections.abc import Callable

from isaaclab.app import AppLauncher


REMOTE_MASTER_ARM_DEVICE_NAME = "external_master_arm"
HYBRID_MECANUM_DEVICE_NAME = "external_master_arm_mecanum"


parser = argparse.ArgumentParser(
    description=(
        "Run the custom Koch Isaac Lab environment with a streamed master-arm device, "
        "or with a hybrid device that uses the keyboard for mecanum-base control and the "
        "streamed master arm for the 5-DoF arm plus gripper."
    )
)
parser.add_argument("--task", type=str, default="Isaac-Koch-Pick-Place-IK-Rel-Mimic-v0", help="Task name.")
parser.add_argument("--teleop_device", type=str, default=REMOTE_MASTER_ARM_DEVICE_NAME, help="Teleop device name.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--stream-host", type=str, default="127.0.0.1", help="TCP host to listen on.")
parser.add_argument("--stream-port", type=int, default=55000, help="TCP port to listen on.")
parser.add_argument(
    "--joint-signs",
    type=str,
    default="1,1,1,1,1",
    help="Comma-separated sign flips for the 5 arm joints, for example: 1,-1,1,1,-1",
)
parser.add_argument(
    "--joint-offsets",
    type=str,
    default="0,0,0,0,0",
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
    "--base-wheel-joint-names",
    type=str,
    default=None,
    help=(
        "Comma-separated mecanum wheel joint names in front-left,front-right,rear-left,rear-right order. "
        "Only used by external_master_arm_mecanum."
    ),
)
parser.add_argument(
    "--base-wheel-signs",
    type=str,
    default=None,
    help=(
        "Optional sign flips for the four wheel velocity targets in front-left,front-right,rear-left,rear-right "
        "order. Only used by external_master_arm_mecanum."
    ),
)
parser.add_argument(
    "--base-wheel-radius",
    type=float,
    default=None,
    help="Wheel radius in meters for mecanum velocity conversion. Only used by external_master_arm_mecanum.",
)
parser.add_argument(
    "--base-wheel-half-length",
    type=float,
    default=None,
    help="Half of the front-to-rear wheel separation in meters. Only used by external_master_arm_mecanum.",
)
parser.add_argument(
    "--base-wheel-half-width",
    type=float,
    default=None,
    help="Half of the left-to-right wheel separation in meters. Only used by external_master_arm_mecanum.",
)
parser.add_argument(
    "--base-vx-sensitivity",
    type=float,
    default=None,
    help="Keyboard forward/backward speed scale for the hybrid mecanum mode.",
)
parser.add_argument(
    "--base-vy-sensitivity",
    type=float,
    default=None,
    help="Keyboard lateral speed scale for the hybrid mecanum mode.",
)
parser.add_argument(
    "--base-omega-sensitivity",
    type=float,
    default=None,
    help="Keyboard yaw speed scale for the hybrid mecanum mode.",
)
parser.add_argument(
    "--no-zero-on-start",
    action="store_true",
    help="Do not capture the first streamed frame as the leader home pose.",
)
parser.add_argument(
    "--stale-timeout",
    type=float,
    default=1.0,
    help="Warn and hold the last action when no fresh frame arrives within this time.",
)
parser.add_argument(
    "--enable_pinocchio",
    action="store_true",
    default=False,
    help="Enable Pinocchio before launching Isaac Sim.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher_args = vars(args_cli)

if args_cli.enable_pinocchio:
    import pinocchio  # noqa: F401

app_launcher = AppLauncher(app_launcher_args)
simulation_app = app_launcher.app


import gymnasium as gym
import torch

from isaaclab.devices import Se2KeyboardCfg, Se3Keyboard, Se3KeyboardCfg
from isaaclab.devices.teleop_device_factory import create_teleop_device
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg, JointVelocityActionCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.manager_based.manipulation.lift import mdp
from isaaclab_tasks.utils import parse_env_cfg

from koch_hybrid_keyboard_master_arm_device import (
    KochHybridKeyboardMasterArmDevice,
    KochHybridKeyboardMasterArmDeviceCfg,
)
from koch_master_arm_stream_device import KochMasterArmStreamDevice, KochMasterArmStreamDeviceCfg

if args_cli.enable_pinocchio:
    import isaaclab_tasks.manager_based.locomanipulation.pick_place  # noqa: F401
    import isaaclab_tasks.manager_based.manipulation.pick_place  # noqa: F401


logger = logging.getLogger(__name__)


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


def resolve_external_master_arm_gripper_cfg(env_cfg: ManagerBasedRLEnvCfg) -> tuple[float, float, float, str]:
    """Resolve gripper calibration for the external master-arm path."""
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


def resolve_hybrid_base_cfg(
    env_cfg: ManagerBasedRLEnvCfg,
) -> tuple[tuple[str, ...], tuple[float, float, float, float], float, float, float, Se2KeyboardCfg]:
    """Resolve mecanum-base control parameters for the hybrid device."""
    if args_cli.base_wheel_joint_names is not None:
        wheel_joint_names = parse_csv_strings(args_cli.base_wheel_joint_names, 4)
    else:
        wheel_joint_names = tuple(getattr(env_cfg, "koch_base_wheel_joint_names", ()))
    if len(wheel_joint_names) != 4:
        raise ValueError(
            "Hybrid mecanum mode requires four wheel joint names in front-left/front-right/rear-left/rear-right order. "
            f"Got: {wheel_joint_names}"
        )

    if args_cli.base_wheel_signs is not None:
        wheel_signs = parse_csv_floats(args_cli.base_wheel_signs, 4)
    else:
        wheel_signs = tuple(getattr(env_cfg, "koch_base_wheel_velocity_signs", (1.0, 1.0, 1.0, 1.0)))
    if len(wheel_signs) != 4:
        raise ValueError(f"Expected four wheel signs, got: {wheel_signs}")

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
    return wheel_joint_names, wheel_signs, wheel_radius, wheel_half_length, wheel_half_width, keyboard_cfg


def make_master_arm_device_cfg(env_cfg: ManagerBasedRLEnvCfg) -> KochMasterArmStreamDeviceCfg:
    """Create the streamed master-arm device config shared by both remote modes."""
    joint_signs = parse_csv_floats(args_cli.joint_signs, 5)
    joint_offsets = parse_csv_floats(args_cli.joint_offsets, 5)
    gripper_open_command, gripper_close_command, gripper_close_delta, gripper_close_direction = (
        resolve_external_master_arm_gripper_cfg(env_cfg)
    )
    return KochMasterArmStreamDeviceCfg(
        sim_device=args_cli.device,
        host=args_cli.stream_host,
        port=args_cli.stream_port,
        joint_signs=joint_signs,
        joint_offsets=joint_offsets,
        zero_on_first_frame=not args_cli.no_zero_on_start,
        gripper_open_command=gripper_open_command,
        gripper_close_command=gripper_close_command,
        gripper_close_delta=gripper_close_delta,
        gripper_close_direction=gripper_close_direction,
        stale_timeout=args_cli.stale_timeout,
        class_type=KochMasterArmStreamDevice,
    )


def inject_remote_teleop_device(env_cfg: ManagerBasedRLEnvCfg) -> None:
    """Inject the requested remote teleop device into the environment config."""
    if args_cli.teleop_device not in (REMOTE_MASTER_ARM_DEVICE_NAME, HYBRID_MECANUM_DEVICE_NAME):
        raise ValueError(
            f"Unsupported teleop device '{args_cli.teleop_device}'. "
            f"Expected '{REMOTE_MASTER_ARM_DEVICE_NAME}' or '{HYBRID_MECANUM_DEVICE_NAME}'."
        )

    if not hasattr(env_cfg, "koch_arm_joint_names") or not hasattr(env_cfg, "koch_gripper_joint_names"):
        raise AttributeError(
            "The selected environment config does not expose 'koch_arm_joint_names' and "
            "'koch_gripper_joint_names', so it cannot be controlled by the Koch master arm bridge."
        )
    if not hasattr(env_cfg, "teleop_devices"):
        raise AttributeError("The selected environment config does not expose teleop_devices.")

    master_arm_device_cfg = make_master_arm_device_cfg(env_cfg)
    gripper_open_command = float(env_cfg.koch_gripper_open_command)
    gripper_close_command = float(env_cfg.koch_gripper_close_command)

    if args_cli.teleop_device == HYBRID_MECANUM_DEVICE_NAME:
        wheel_joint_names, wheel_signs, wheel_radius, wheel_half_length, wheel_half_width, keyboard_cfg = (
            resolve_hybrid_base_cfg(env_cfg)
        )
        env_cfg.robot_fix_root_link = False
        if hasattr(env_cfg, "scene") and hasattr(env_cfg.scene, "robot"):
            robot_spawn = getattr(env_cfg.scene.robot, "spawn", None)
            if robot_spawn is not None and getattr(robot_spawn, "articulation_props", None) is not None:
                robot_spawn.articulation_props.fix_root_link = False
        env_cfg.actions.base_action = JointVelocityActionCfg(
            asset_name="robot",
            joint_names=list(wheel_joint_names),
            scale=1.0,
            offset=0.0,
            preserve_order=True,
            use_default_offset=False,
        )
        env_cfg.teleop_devices.devices[HYBRID_MECANUM_DEVICE_NAME] = KochHybridKeyboardMasterArmDeviceCfg(
            sim_device=args_cli.device,
            base_keyboard=keyboard_cfg,
            master_arm=master_arm_device_cfg,
            wheel_radius=wheel_radius,
            wheel_base_half_length=wheel_half_length,
            wheel_base_half_width=wheel_half_width,
            wheel_velocity_signs=wheel_signs,
            class_type=KochHybridKeyboardMasterArmDevice,
        )
    else:
        env_cfg.actions.base_action = None
        env_cfg.teleop_devices.devices[REMOTE_MASTER_ARM_DEVICE_NAME] = master_arm_device_cfg
        if hasattr(env_cfg, "external_master_arm_device"):
            env_cfg.external_master_arm_device = master_arm_device_cfg

    # Both remote modes drive the mounted arm by absolute joint targets.
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


def setup_keyboard_shortcuts(callbacks: dict[str, Callable], teleop_interface: object) -> Se3Keyboard | None:
    """Enable keyboard-only hotkeys when the main teleop device is not keyboard based."""
    if args_cli.headless or os.environ.get("HEADLESS", "0") not in ("0", "", "False", "false"):
        return None

    if getattr(teleop_interface, "uses_keyboard_shortcuts", False):
        return None
    if "keyboard" in teleop_interface.__class__.__name__.lower():
        return None

    try:
        shortcut_listener = Se3Keyboard(
            Se3KeyboardCfg(pos_sensitivity=0.0, rot_sensitivity=0.0, gripper_term=False)
        )
    except Exception as exc:
        logger.warning(f"Failed to enable keyboard shortcut listener: {exc}")
        return None

    for key, callback in callbacks.items():
        shortcut_listener.add_callback(key, callback)

    print("Keyboard shortcuts enabled: press R to reset the environment.")
    return shortcut_listener


def main() -> None:
    """Script entry point."""
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.env_name = args_cli.task
    if not isinstance(env_cfg, ManagerBasedRLEnvCfg):
        raise ValueError(
            "Remote master-arm teleoperation is only supported for ManagerBasedRLEnv environments. "
            f"Received: {type(env_cfg).__name__}"
        )

    env_cfg.terminations.time_out = None
    if "Lift" in args_cli.task:
        env_cfg.commands.object_pose.resampling_time_range = (1.0e9, 1.0e9)
        env_cfg.terminations.object_reached_goal = DoneTerm(func=mdp.object_reached_goal)

    inject_remote_teleop_device(env_cfg)

    try:
        env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    except Exception as exc:
        logger.error(f"Failed to create environment: {exc}")
        simulation_app.close()
        return

    should_reset = False
    teleoperation_active = True

    def reset_env() -> None:
        nonlocal should_reset
        should_reset = True
        print("Reset requested")

    def start_teleoperation() -> None:
        nonlocal teleoperation_active
        teleoperation_active = True
        print("Teleoperation activated")

    def stop_teleoperation() -> None:
        nonlocal teleoperation_active
        teleoperation_active = False
        print("Teleoperation deactivated")

    teleop_callbacks = {
        "R": reset_env,
        "RESET": reset_env,
        "START": start_teleoperation,
        "STOP": stop_teleoperation,
    }

    teleop_interface = create_teleop_device(args_cli.teleop_device, env_cfg.teleop_devices.devices, teleop_callbacks)
    keyboard_shortcuts = setup_keyboard_shortcuts(teleop_callbacks, teleop_interface)

    print(f"Using teleop device: {teleop_interface}")
    print(
        "Waiting for master-arm stream on "
        f"{args_cli.stream_host}:{args_cli.stream_port}. "
        "Start koch_leader_ssh_streamer.py on the local machine after this script is ready."
    )
    if args_cli.teleop_device == HYBRID_MECANUM_DEVICE_NAME:
        print("Hybrid mode enabled: keyboard controls the mecanum base while the remote master arm controls the arm.")

    env.reset()
    teleop_interface.reset()

    while simulation_app.is_running():
        try:
            with torch.inference_mode():
                action = teleop_interface.advance()
                if teleoperation_active:
                    actions = action.repeat(env.num_envs, 1)
                    env.step(actions)
                else:
                    env.sim.render()

                if should_reset:
                    env.reset()
                    teleop_interface.reset()
                    should_reset = False
                    print("Environment reset complete")
        except Exception as exc:
            logger.error(f"Error during simulation step: {exc}")
            break

    env.close()
    del keyboard_shortcuts
    print("Environment closed")


if __name__ == "__main__":
    main()
    simulation_app.close()
