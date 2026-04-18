"""Record demonstrations for the custom Koch environment."""

from __future__ import annotations

import argparse
import contextlib
import logging
import os
import time
from collections.abc import Callable

from isaaclab.app import AppLauncher

from koch_teleop_mode_utils import add_mobile_base_args, add_remote_master_arm_args, add_teleop_mode_args


parser = argparse.ArgumentParser(
    description=(
        "Record demonstrations for the custom Koch environment using two teleop mode flags: "
        "fixed/mobile base and keyboard/remote master-arm control."
    )
)
parser.add_argument("--task", type=str, required=True, help="Name of the task.")
add_teleop_mode_args(parser)
parser.add_argument("--dataset_file", type=str, default="./datasets/dataset.hdf5", help="Output HDF5 file path.")
parser.add_argument("--step_hz", type=int, default=30, help="Environment stepping rate in Hz.")
parser.add_argument("--num_demos", type=int, default=0, help="Number of demonstrations to record. 0 means infinite.")
parser.add_argument(
    "--num_success_steps",
    type=int,
    default=10,
    help="Number of consecutive successful steps required before exporting a demo.",
)
add_remote_master_arm_args(parser)
add_mobile_base_args(parser)
parser.add_argument(
    "--enable_pinocchio",
    action="store_true",
    default=False,
    help="Enable Pinocchio before launching Isaac Sim.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher_args = vars(args_cli).copy()

if args_cli.enable_pinocchio:
    import pinocchio  # noqa: F401

app_launcher = AppLauncher(app_launcher_args)
simulation_app = app_launcher.app


import gymnasium as gym
import omni.ui as ui
import torch

from isaaclab.devices import Se3Keyboard, Se3KeyboardCfg
from isaaclab.devices.openxr import remove_camera_configs
from isaaclab.devices.teleop_device_factory import create_teleop_device
from isaaclab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg
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

from koch_teleop_mode_utils import configure_teleop_env, teleop_mode_startup_messages


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


def setup_output_directories() -> tuple[str, str]:
    """Prepare the recording output directory and dataset name."""
    output_dir = os.path.dirname(args_cli.dataset_file)
    output_file_name = os.path.splitext(os.path.basename(args_cli.dataset_file))[0]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    return output_dir, output_file_name


def create_environment_config(
    output_dir: str, output_file_name: str
) -> tuple[ManagerBasedRLEnvCfg | DirectRLEnvCfg, object | None, object]:
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

    teleop_mode = configure_teleop_env(env_cfg, args_cli)
    return env_cfg, success_term, teleop_mode


def create_environment(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg) -> gym.Env:
    """Instantiate the environment from its config."""
    try:
        return gym.make(args_cli.task, cfg=env_cfg).unwrapped
    except Exception as exc:
        logger.error(f"Failed to create environment: {exc}")
        raise SystemExit(1) from exc


def setup_teleop_device(callbacks: dict[str, Callable], env_cfg, teleop_mode) -> object:
    """Create the requested teleop interface."""
    try:
        return create_teleop_device(teleop_mode.device_name, env_cfg.teleop_devices.devices, callbacks)
    except Exception as exc:
        logger.error(f"Failed to create teleop device: {exc}")
        raise SystemExit(1) from exc


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


def run_simulation_loop(env: gym.Env, env_cfg, success_term: object | None, teleop_mode, rate_limiter: RateLimiter | None) -> int:
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

    teleop_interface = setup_teleop_device(teleoperation_callbacks, env_cfg, teleop_mode)
    teleop_interface.add_callback("R", reset_recording_instance)
    keyboard_shortcuts = setup_keyboard_shortcuts(teleoperation_callbacks, teleop_interface)

    env.sim.reset()
    env.reset()
    teleop_interface.reset()

    for message in teleop_mode_startup_messages(teleop_mode, args_cli):
        print(message)

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
    env_cfg, success_term, teleop_mode = create_environment_config(output_dir, output_file_name)
    env = create_environment(env_cfg)
    current_recorded_demo_count = run_simulation_loop(env, env_cfg, success_term, teleop_mode, rate_limiter)
    env.close()
    print(f"Recording session completed with {current_recorded_demo_count} successful demonstrations")
    print(f"Demonstrations saved to: {args_cli.dataset_file}")


if __name__ == "__main__":
    main()
    simulation_app.close()
