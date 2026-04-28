"""Record demonstrations for the Koch Mimic task."""

from __future__ import annotations

import argparse
import contextlib
import logging
import os
from pathlib import Path
import sys
import time
from collections.abc import Callable, Sequence
from typing import Any

from isaaclab.app import AppLauncher

from koch_mimic.cloud.scripts.teleop_mode import (
    add_mobile_base_args,
    add_remote_master_arm_args,
    add_teleop_mode_args,
    configure_teleop_env,
    teleop_mode_startup_messages,
)
from koch_mimic.shared.configuration import (
    activate_runtime_config,
    get_config_section,
    option_was_provided,
    resolve_config_path,
)
from koch_mimic.shared.constants import CLOUD_PROFILE, DEFAULT_TASK_ID


logger = logging.getLogger(__name__)


def _csv_from_values(values: Sequence[float]) -> str:
    return ",".join(str(value) for value in values)


class RateLimiter:
    """Simple rate limiter to keep recording close to a fixed step rate."""

    def __init__(self, hz: int):
        self.hz = hz
        self.last_time = time.time()
        self.sleep_duration = 1.0 / hz
        self.render_period = min(0.033, self.sleep_duration)

    def sleep(self, env) -> None:
        next_wakeup_time = self.last_time + self.sleep_duration
        while time.time() < next_wakeup_time:
            time.sleep(self.render_period)
            env.sim.render()

        self.last_time = self.last_time + self.sleep_duration
        if self.last_time < time.time():
            while self.last_time < time.time():
                self.last_time += self.sleep_duration


class CameraPreviewWidget:
    """Minimal RGB image widget backed by an Omni UI byte image provider."""

    def __init__(self, image, label: str, widget_height: int = 240):
        self._provider = ui.ByteImageProvider()
        self._widget_height = widget_height
        self._label = label
        self._aspect_ratio = 1.0

        with ui.VStack(spacing=4):
            ui.Label(self._label)
            self._frame = ui.Frame(width=ui.Fraction(1), height=self._widget_height)
            with self._frame:
                self._image = ui.ImageWithProvider(self._provider)

        self.update_image(image)

    def update_image(self, image) -> None:
        image = np.ascontiguousarray(image)
        if image.ndim == 3 and image.shape[0] in (1, 3, 4):
            image = np.moveaxis(image, 0, -1)

        height, width = image.shape[:2]
        self._aspect_ratio = width / max(height, 1)

        if image.ndim == 2:
            image = np.repeat(image[..., None], 3, axis=2)
        if image.ndim == 3 and image.shape[2] == 1:
            image = np.repeat(image, 3, axis=2)
        if image.ndim == 3 and image.shape[2] == 3:
            alpha = np.full((height, width, 1), 255, dtype=np.uint8)
            image = np.concatenate((image, alpha), axis=2)

        self._frame.width = ui.Pixel(int(round(self._aspect_ratio * self._widget_height)))
        self._provider.set_bytes_data(image.flatten().data, [width, height])


class CameraPreviewPanel:
    """Small dockable UI windows that show live RGB previews from environment cameras."""

    def __init__(self, env, camera_names: list[str], EmptyWindow):
        self._plots: dict[str, CameraPreviewWidget] = {}
        self._windows: list[Any] = []

        for camera_name in camera_names:
            if camera_name not in env.scene.sensors:
                continue
            sensor = env.scene.sensors[camera_name]
            if "rgb" not in sensor.data.output:
                continue

            initial_image = sensor.data.output["rgb"][0].cpu().numpy()
            window = EmptyWindow(env, f"{camera_name} Preview")
            with window.ui_window_elements["main_vstack"]:
                self._plots[camera_name] = CameraPreviewWidget(
                    image=initial_image,
                    label=camera_name,
                    widget_height=240,
                )
            self._windows.append(window)

    def update(self, env) -> None:
        for camera_name, plot in self._plots.items():
            image = env.scene.sensors[camera_name].data.output["rgb"][0].cpu().numpy()
            plot.update_image(np.ascontiguousarray(image))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Record demonstrations for the Koch Mimic environment using fixed/mobile base teleop "
            "and keyboard/remote-master-arm control."
        )
    )
    parser.add_argument("--config", type=str, default=None, help="Optional extra cloud YAML overlay.")
    parser.add_argument("--task", type=str, default=None, help="Name of the task.")
    add_teleop_mode_args(parser)
    parser.add_argument("--dataset_file", type=str, default=None, help="Output HDF5 file path.")
    parser.add_argument("--step_hz", type=int, default=None, help="Environment stepping rate in Hz.")
    parser.add_argument("--num_demos", type=int, default=None, help="Number of demonstrations to record. 0 means infinite.")
    parser.add_argument(
        "--num_success_steps",
        type=int,
        default=None,
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
    return parser


def resolve_runtime_args(args: argparse.Namespace, argv: list[str]) -> None:
    config = activate_runtime_config(CLOUD_PROFILE, overlay_path=args.config, require_user_local=True)

    if args.task is None:
        args.task = str(get_config_section(config, "task", "default_task_id", default=DEFAULT_TASK_ID))

    if args.dataset_file is None:
        dataset_dir = resolve_config_path(
            str(get_config_section(config, "dataset", "output_dir", default="./datasets")),
            config,
        )
        dataset_filename = str(get_config_section(config, "dataset", "filename", default="koch_mimic_demos.hdf5"))
        args.dataset_file = str(Path(dataset_dir) / dataset_filename)

    if args.step_hz is None:
        args.step_hz = int(get_config_section(config, "demo", "step_hz", default=30))
    if args.num_demos is None:
        args.num_demos = int(get_config_section(config, "demo", "num_demos", default=0))
    if args.num_success_steps is None:
        args.num_success_steps = int(get_config_section(config, "demo", "num_success_steps", default=10))

    if args.stream_host is None:
        args.stream_host = str(get_config_section(config, "teleop", "stream", "host", default="127.0.0.1"))
    if args.stream_port is None:
        args.stream_port = int(get_config_section(config, "teleop", "stream", "port", default=55000))
    if args.joint_signs is None:
        args.joint_signs = _csv_from_values(
            get_config_section(config, "teleop", "stream", "joint_signs", default=[1, 1, 1, 1, 1])
        )
    if args.joint_offsets is None:
        args.joint_offsets = _csv_from_values(
            get_config_section(config, "teleop", "stream", "joint_offsets", default=[0, 0, 0, 0, 0])
        )
    if args.gripper_close_delta is None:
        args.gripper_close_delta = get_config_section(config, "teleop", "stream", "gripper_close_delta", default=None)
    if args.gripper_close_direction is None:
        args.gripper_close_direction = str(
            get_config_section(config, "teleop", "stream", "gripper_close_direction", default="positive")
        )
    if args.zero_on_start is None:
        args.zero_on_start = bool(
            get_config_section(config, "teleop", "stream", "zero_on_first_frame", default=True)
        )
    if args.stale_timeout is None:
        args.stale_timeout = float(get_config_section(config, "teleop", "stream", "stale_timeout", default=1.0))

    if not option_was_provided(argv, "--enable_pinocchio"):
        args.enable_pinocchio = bool(get_config_section(config, "app", "enable_pinocchio", default=False))

    if hasattr(args, "enable_cameras") and not option_was_provided(argv, "--enable_cameras"):
        args.enable_cameras = bool(get_config_section(config, "app", "enable_cameras", default=args.enable_cameras))


def setup_output_directories(dataset_file: str) -> tuple[str, str]:
    output_dir = os.path.dirname(dataset_file)
    output_file_name = os.path.splitext(os.path.basename(dataset_file))[0]
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    return output_dir, output_file_name


def main(argv: Sequence[str] | None = None) -> None:
    argv_list = list(sys.argv[1:] if argv is None else argv)
    parser = build_arg_parser()
    args = parser.parse_args(argv_list)
    resolve_runtime_args(args, argv_list)

    app_launcher_args = vars(args).copy()
    if args.enable_pinocchio:
        import pinocchio  # noqa: F401

    app_launcher = AppLauncher(app_launcher_args)
    simulation_app = app_launcher.app

    try:
        global gym, np, torch, ui  # noqa: PLW0603

        import gymnasium as gym
        import numpy as np
        import omni.ui as ui
        import torch

        from isaaclab.devices import Se3Keyboard, Se3KeyboardCfg
        from isaaclab.devices.openxr import remove_camera_configs
        from isaaclab.devices.teleop_device_factory import create_teleop_device
        from isaaclab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg
        from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
        from isaaclab.envs.ui import EmptyWindow
        from isaaclab.managers import DatasetExportMode

        from isaaclab_mimic.ui.instruction_display import InstructionDisplay, show_subtask_instructions

        import isaaclab_tasks  # noqa: F401
        import koch_mimic.cloud.tasks.koch_pick_place  # noqa: F401
        from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

        if args.enable_pinocchio:
            import isaaclab_tasks.manager_based.locomanipulation.pick_place  # noqa: F401
            import isaaclab_tasks.manager_based.manipulation.pick_place  # noqa: F401

        def create_environment_config(
            output_dir: str,
            output_file_name: str,
        ) -> tuple[ManagerBasedRLEnvCfg | DirectRLEnvCfg, object | None, object]:
            try:
                env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
                env_cfg.env_name = args.task.split(":")[-1]
            except Exception as exc:
                logger.error(f"Failed to parse environment configuration: {exc}")
                raise SystemExit(1) from exc

            success_term = None
            if hasattr(env_cfg.terminations, "success"):
                success_term = env_cfg.terminations.success
                env_cfg.terminations.success = None
            else:
                logger.warning("No success termination term was found in the environment.")

            if args.xr:
                if not args.enable_cameras:
                    env_cfg = remove_camera_configs(env_cfg)
                env_cfg.sim.render.antialiasing_mode = "DLSS"

            env_cfg.terminations.time_out = None
            env_cfg.observations.policy.concatenate_terms = False

            env_cfg.recorders = ActionStateRecorderManagerCfg()
            env_cfg.recorders.dataset_export_dir_path = output_dir
            env_cfg.recorders.dataset_filename = output_file_name
            env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY

            teleop_mode = configure_teleop_env(env_cfg, args)
            return env_cfg, success_term, teleop_mode

        def setup_teleop_device(callbacks: dict[str, Callable[[], None]], env_cfg, teleop_mode):
            try:
                return create_teleop_device(teleop_mode.device_name, env_cfg.teleop_devices.devices, callbacks)
            except Exception as exc:
                logger.error(f"Failed to create teleop device: {exc}")
                raise SystemExit(1) from exc

        def setup_keyboard_shortcuts(
            callbacks: dict[str, Callable[[], None]],
            teleop_interface: object,
        ) -> Se3Keyboard | None:
            if args.headless or os.environ.get("HEADLESS", "0") not in ("0", "", "False", "false"):
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

        def setup_ui(label_text: str, env):
            instruction_display = InstructionDisplay(args.xr)
            if not args.xr:
                window = EmptyWindow(env, "Instruction")
                with window.ui_window_elements["main_vstack"]:
                    demo_label = ui.Label(label_text)
                    subtask_label = ui.Label("")
                    instruction_display.set_labels(subtask_label, demo_label)
            return instruction_display

        def process_success_condition(env, success_term: object | None, success_step_count: int) -> tuple[int, bool]:
            if success_term is None:
                return success_step_count, False

            if bool(success_term.func(env, **success_term.params)[0]):
                success_step_count += 1
                if success_step_count >= args.num_success_steps:
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

        def handle_reset(env, success_step_count: int, instruction_display, label_text: str) -> int:
            print("Resetting environment...")
            env.sim.reset()
            env.recorder_manager.reset()
            env.reset()
            success_step_count = 0
            instruction_display.show_demo(label_text)
            return success_step_count

        output_dir, output_file_name = setup_output_directories(args.dataset_file)
        env_cfg, success_term, teleop_mode = create_environment_config(output_dir, output_file_name)

        try:
            env = gym.make(args.task, cfg=env_cfg).unwrapped
        except Exception as exc:
            logger.error(f"Failed to create environment: {exc}")
            raise SystemExit(1) from exc

        if args.xr:
            rate_limiter = None
            from isaaclab.ui.xr_widgets import TeleopVisualizationManager, XRVisualization

            XRVisualization.assign_manager(TeleopVisualizationManager)
        else:
            rate_limiter = RateLimiter(args.step_hz)

        current_recorded_demo_count = 0
        success_step_count = 0
        should_reset_recording_instance = False
        running_recording_instance = not args.xr

        def reset_recording_instance() -> None:
            nonlocal should_reset_recording_instance
            should_reset_recording_instance = True
            print("Recording instance reset requested")

        def start_recording_instance() -> None:
            nonlocal running_recording_instance
            running_recording_instance = True
            print("Recording started")

        def stop_recording_instance() -> None:
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

        for message in teleop_mode_startup_messages(teleop_mode, args):
            print(message)

        label_text = f"Recorded {current_recorded_demo_count} successful demonstrations."
        instruction_display = setup_ui(label_text, env)
        camera_preview_panel = None
        if not args.xr and not args.headless and hasattr(env.cfg, "image_obs_list"):
            camera_preview_panel = CameraPreviewPanel(env, list(env.cfg.image_obs_list), EmptyWindow)
        subtasks = {}

        with contextlib.suppress(KeyboardInterrupt), torch.inference_mode():
            while simulation_app.is_running():
                action = teleop_interface.advance()
                actions = action.repeat(env.num_envs, 1)

                if running_recording_instance:
                    obv = env.step(actions)
                    if camera_preview_panel is not None:
                        camera_preview_panel.update(env)
                    if subtasks == {}:
                        subtasks = obv[0].get("subtask_terms")
                    elif subtasks:
                        show_subtask_instructions(instruction_display, subtasks, obv, env.cfg)
                else:
                    env.sim.render()
                    if camera_preview_panel is not None:
                        camera_preview_panel.update(env)

                success_step_count, success_reset_needed = process_success_condition(
                    env, success_term, success_step_count
                )
                if success_reset_needed:
                    should_reset_recording_instance = True

                if env.recorder_manager.exported_successful_episode_count > current_recorded_demo_count:
                    current_recorded_demo_count = env.recorder_manager.exported_successful_episode_count
                    label_text = f"Recorded {current_recorded_demo_count} successful demonstrations."
                    print(label_text)

                if args.num_demos > 0 and env.recorder_manager.exported_successful_episode_count >= args.num_demos:
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

        env.close()
        del keyboard_shortcuts
        print(f"Recording session completed with {current_recorded_demo_count} successful demonstrations")
        print(f"Demonstrations saved to: {args.dataset_file}")
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
