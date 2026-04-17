"""Record demonstrations for the custom Koch environment."""

from __future__ import annotations

import argparse
import contextlib
import logging
import os
import time
from collections.abc import Callable

from isaaclab.app import AppLauncher


REMOTE_MASTER_ARM_DEVICE_NAME = "external_master_arm"
HYBRID_MECANUM_DEVICE_NAME = "external_master_arm_mecanum"


parser = argparse.ArgumentParser(
    description="Record demonstrations for Isaac Lab environments with optional remote Koch teleoperation."
)
parser.add_argument("--task", type=str, required=True, help="Name of the task.")
parser.add_argument(
    "--teleop_device",
    type=str,
    default="keyboard",
    help=(
        "Teleop device. Built-ins: keyboard, spacemouse. "
        "Custom env-config devices are also supported, including external_master_arm and "
        "external_master_arm_mecanum."
    ),
)
parser.add_argument("--dataset_file", type=str, default="./datasets/dataset.hdf5", help="Output HDF5 file path.")
parser.add_argument("--step_hz", type=int, default=30, help="Environment stepping rate in Hz.")
parser.add_argument("--num_demos", type=int, default=0, help="Number of demonstrations to record. 0 means infinite.")
parser.add_argument(
    "--num_success_steps",
    type=int,
    default=10,
    help="Number of consecutive successful steps required before exporting a demo.",
)
parser.add_argument(
    "--stream-host",
    type=str,
    default="127.0.0.1",
    help="TCP host for the remote master-arm device to listen on inside the cloud server.",
)
parser.add_argument("--stream-port", type=int, default=55000, help="TCP port for the remote master-arm device.")
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

app_launcher_args = vars(args_cli).copy()
if "handtracking" in args_cli.teleop_device.lower():
    app_launcher_args["xr"] = True
    setattr(args_cli, "xr", True)

if args_cli.enable_pinocchio:
    import pinocchio  # noqa: F401

app_launcher = AppLauncher(app_launcher_args)
simulation_app = app_launcher.app


import gymnasium as gym
import omni.ui as ui
import torch

from isaaclab.devices import Se2KeyboardCfg, Se3Keyboard, Se3KeyboardCfg, Se3SpaceMouse, Se3SpaceMouseCfg
from isaaclab.devices.openxr import remove_camera_configs
from isaaclab.devices.teleop_device_factory import create_teleop_device
from isaaclab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg, JointVelocityActionCfg
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.envs.ui import EmptyWindow
from isaaclab.managers import DatasetExportMode

import isaaclab_mimic.envs  # noqa: F401
from isaaclab_mimic.ui.instruction_display import InstructionDisplay, show_subtask_instructions

if args_cli.enable_pinocchio:
    import isaaclab_tasks.manager_based.locomanipulation.pick_place  # noqa: F401
    import isaaclab_tasks.manager_based.manipulation.pick_place  # noqa: F401

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

from koch_hybrid_keyboard_master_arm_device import (
    KochHybridKeyboardMasterArmDevice,
    KochHybridKeyboardMasterArmDeviceCfg,
)
from koch_master_arm_stream_device import KochMasterArmStreamDevice, KochMasterArmStreamDeviceCfg


logger = logging.getLogger(__name__)


class RateLimiter:
    """Simple rate limiter to keep recording close to a fixed step rate."""

    def __init__(self, hz: int):
        self.hz = hz
        self.last_time = time.time()
        self.sleep_duration = 1.0 / hz
        self.render_period = min(0.033, self.sleep_duration)

    def sleep(self, env: gym.Env):
        next_wakeup_time = self.last_time + self.sleep_duration
        while time.time() < next_wakeup_time:
            time.sleep(self.render_period)
            env.sim.render()

        self.last_time = self.last_time + self.sleep_duration
        if self.last_time < time.time():
            while self.last_time < time.time():
                self.last_time += self.sleep_duration


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


def resolve_external_master_arm_gripper_cfg(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg,
) -> tuple[float, float, float, str]:
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
    """Create the streamed master-arm device config shared by the remote modes."""
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


def setup_output_directories() -> tuple[str, str]:
    """Prepare the recording output directory and dataset name."""
    output_dir = os.path.dirname(args_cli.dataset_file)
    output_file_name = os.path.splitext(os.path.basename(args_cli.dataset_file))[0]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    return output_dir, output_file_name


def maybe_inject_remote_teleop_device(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg) -> None:
    """Inject the requested remote teleop device into the environment config when needed."""
    if args_cli.teleop_device not in (REMOTE_MASTER_ARM_DEVICE_NAME, HYBRID_MECANUM_DEVICE_NAME):
        return

    if not isinstance(env_cfg, ManagerBasedRLEnvCfg):
        raise ValueError(f"{args_cli.teleop_device} is only supported for ManagerBasedRLEnv environments.")
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


def create_environment_config(
    output_dir: str, output_file_name: str
) -> tuple[ManagerBasedRLEnvCfg | DirectRLEnvCfg, object | None]:
    """Parse and complete the environment configuration."""
    try:
        env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
        env_cfg.env_name = args_cli.task.split(":")[-1]
    except Exception as exc:
        logger.error(f"Failed to parse environment configuration: {exc}")
        raise SystemExit(1) from exc

    success_term = None
    if hasattr(env_cfg.terminations, "success"):
        success_term = env_cfg.terminations.success
        env_cfg.terminations.success = None
    else:
        logger.warning("No success termination term was found in the environment.")

    if args_cli.xr:
        if not args_cli.enable_cameras:
            env_cfg = remove_camera_configs(env_cfg)
        env_cfg.sim.render.antialiasing_mode = "DLSS"

    env_cfg.terminations.time_out = None
    env_cfg.observations.policy.concatenate_terms = False

    env_cfg.recorders = ActionStateRecorderManagerCfg()
    env_cfg.recorders.dataset_export_dir_path = output_dir
    env_cfg.recorders.dataset_filename = output_file_name
    env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY

    maybe_inject_remote_teleop_device(env_cfg)
    return env_cfg, success_term


def create_environment(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg) -> gym.Env:
    """Instantiate the environment from its config."""
    try:
        return gym.make(args_cli.task, cfg=env_cfg).unwrapped
    except Exception as exc:
        logger.error(f"Failed to create environment: {exc}")
        raise SystemExit(1) from exc


def setup_teleop_device(callbacks: dict[str, Callable], env_cfg) -> object:
    """Create the requested teleop interface."""
    teleop_interface = None
    try:
        if hasattr(env_cfg, "teleop_devices") and args_cli.teleop_device in env_cfg.teleop_devices.devices:
            teleop_interface = create_teleop_device(args_cli.teleop_device, env_cfg.teleop_devices.devices, callbacks)
        else:
            logger.warning(f"No teleop device '{args_cli.teleop_device}' found in environment config. Creating default.")
            if args_cli.teleop_device.lower() == "keyboard":
                teleop_interface = Se3Keyboard(Se3KeyboardCfg(pos_sensitivity=0.2, rot_sensitivity=0.5))
            elif args_cli.teleop_device.lower() == "spacemouse":
                teleop_interface = Se3SpaceMouse(Se3SpaceMouseCfg(pos_sensitivity=0.2, rot_sensitivity=0.5))
            else:
                logger.error(f"Unsupported teleop device: {args_cli.teleop_device}")
                raise SystemExit(1)

            for key, callback in callbacks.items():
                teleop_interface.add_callback(key, callback)
    except Exception as exc:
        logger.error(f"Failed to create teleop device: {exc}")
        raise SystemExit(1) from exc

    if teleop_interface is None:
        logger.error("Failed to create teleop interface")
        raise SystemExit(1)

    return teleop_interface


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

    print("Keyboard shortcuts enabled: press R to reset the current recording episode.")
    return shortcut_listener


def setup_ui(label_text: str, env: gym.Env) -> InstructionDisplay:
    """Initialize the instruction UI used while recording."""
    instruction_display = InstructionDisplay(args_cli.xr)
    if not args_cli.xr:
        window = EmptyWindow(env, "Instruction")
        with window.ui_window_elements["main_vstack"]:
            demo_label = ui.Label(label_text)
            subtask_label = ui.Label("")
            instruction_display.set_labels(subtask_label, demo_label)
    return instruction_display


def process_success_condition(env: gym.Env, success_term: object | None, success_step_count: int) -> tuple[int, bool]:
    """Check the success condition and export the current episode when it is satisfied."""
    if success_term is None:
        return success_step_count, False

    if bool(success_term.func(env, **success_term.params)[0]):
        success_step_count += 1
        if success_step_count >= args_cli.num_success_steps:
            env.recorder_manager.record_pre_reset([0], force_export_or_skip=False)
            env.recorder_manager.set_success_to_episodes(
                [0], torch.tensor([[True]], dtype=torch.bool, device=env.device)
            )
            env.recorder_manager.export_episodes([0])
            print("Success condition met! Recording completed.")
            return success_step_count, True
    else:
        success_step_count = 0

    return success_step_count, False


def handle_reset(env: gym.Env, success_step_count: int, instruction_display: InstructionDisplay, label_text: str) -> int:
    """Reset the environment, recorder, and UI state."""
    print("Resetting environment...")
    env.sim.reset()
    env.recorder_manager.reset()
    env.reset()
    success_step_count = 0
    instruction_display.show_demo(label_text)
    return success_step_count


def run_simulation_loop(env: gym.Env, env_cfg, success_term: object | None, rate_limiter: RateLimiter | None) -> int:
    """Run the main recording loop."""
    current_recorded_demo_count = 0
    success_step_count = 0
    should_reset_recording_instance = False
    running_recording_instance = not args_cli.xr

    def reset_recording_instance():
        nonlocal should_reset_recording_instance
        should_reset_recording_instance = True
        print("Recording instance reset requested")

    def start_recording_instance():
        nonlocal running_recording_instance
        running_recording_instance = True
        print("Recording started")

    def stop_recording_instance():
        nonlocal running_recording_instance
        running_recording_instance = False
        print("Recording paused")

    teleoperation_callbacks = {
        "R": reset_recording_instance,
        "START": start_recording_instance,
        "STOP": stop_recording_instance,
        "RESET": reset_recording_instance,
    }

    teleop_interface = setup_teleop_device(teleoperation_callbacks, env_cfg)
    teleop_interface.add_callback("R", reset_recording_instance)
    keyboard_shortcuts = setup_keyboard_shortcuts(teleoperation_callbacks, teleop_interface)

    env.sim.reset()
    env.reset()
    teleop_interface.reset()

    if args_cli.teleop_device in (REMOTE_MASTER_ARM_DEVICE_NAME, HYBRID_MECANUM_DEVICE_NAME):
        print(
            "Waiting for master-arm stream on "
            f"{args_cli.stream_host}:{args_cli.stream_port}. "
            "Start koch_leader_ssh_streamer.py on the local machine after this script is ready."
        )
        if args_cli.teleop_device == HYBRID_MECANUM_DEVICE_NAME:
            print(
                "Hybrid mode enabled: keyboard controls the mecanum base while the remote master arm "
                "controls the arm and gripper."
            )

    label_text = f"Recorded {current_recorded_demo_count} successful demonstrations."
    instruction_display = setup_ui(label_text, env)
    subtasks = {}

    with contextlib.suppress(KeyboardInterrupt), torch.inference_mode():
        while simulation_app.is_running():
            action = teleop_interface.advance()
            actions = action.repeat(env.num_envs, 1)

            if running_recording_instance:
                obv = env.step(actions)
                if subtasks is not None:
                    if subtasks == {}:
                        subtasks = obv[0].get("subtask_terms")
                    elif subtasks:
                        show_subtask_instructions(instruction_display, subtasks, obv, env.cfg)
            else:
                env.sim.render()

            success_step_count, success_reset_needed = process_success_condition(env, success_term, success_step_count)
            if success_reset_needed:
                should_reset_recording_instance = True

            if env.recorder_manager.exported_successful_episode_count > current_recorded_demo_count:
                current_recorded_demo_count = env.recorder_manager.exported_successful_episode_count
                label_text = f"Recorded {current_recorded_demo_count} successful demonstrations."
                print(label_text)

            if args_cli.num_demos > 0 and env.recorder_manager.exported_successful_episode_count >= args_cli.num_demos:
                label_text = f"All {current_recorded_demo_count} demonstrations recorded.\nExiting the app."
                instruction_display.show_demo(label_text)
                print(label_text)
                target_time = time.time() + 0.8
                while time.time() < target_time:
                    if rate_limiter:
                        rate_limiter.sleep(env)
                    else:
                        env.sim.render()
                break

            if should_reset_recording_instance:
                success_step_count = handle_reset(env, success_step_count, instruction_display, label_text)
                should_reset_recording_instance = False

            if env.sim.is_stopped():
                break

            if rate_limiter:
                rate_limiter.sleep(env)

    del keyboard_shortcuts
    return current_recorded_demo_count


def main() -> None:
    """Script entry point."""
    if args_cli.xr:
        rate_limiter = None
        from isaaclab.ui.xr_widgets import TeleopVisualizationManager, XRVisualization

        XRVisualization.assign_manager(TeleopVisualizationManager)
    else:
        rate_limiter = RateLimiter(args_cli.step_hz)

    output_dir, output_file_name = setup_output_directories()
    env_cfg, success_term = create_environment_config(output_dir, output_file_name)
    env = create_environment(env_cfg)
    current_recorded_demo_count = run_simulation_loop(env, env_cfg, success_term, rate_limiter)
    env.close()
    print(f"Recording session completed with {current_recorded_demo_count} successful demonstrations")
    print(f"Demonstrations saved to: {args_cli.dataset_file}")


if __name__ == "__main__":
    main()
    simulation_app.close()
