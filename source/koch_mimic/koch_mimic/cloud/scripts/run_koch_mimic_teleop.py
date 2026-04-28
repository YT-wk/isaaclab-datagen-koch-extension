"""Cloud-side teleop entrypoint for the Koch Mimic task."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections.abc import Callable, Sequence

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
)
from koch_mimic.shared.constants import CLOUD_PROFILE, DEFAULT_TASK_ID


logger = logging.getLogger(__name__)


def _csv_from_values(values: Sequence[float]) -> str:
    return ",".join(str(value) for value in values)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the Koch IsaacLab environment with fixed/mobile base teleop and "
            "keyboard/remote-master-arm control."
        )
    )
    parser.add_argument("--config", type=str, default=None, help="Optional extra cloud YAML overlay.")
    parser.add_argument("--task", type=str, default=None, help="Task name.")
    add_teleop_mode_args(parser)
    parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
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
        import gymnasium as gym
        import torch

        from isaaclab.devices import Se3Keyboard, Se3KeyboardCfg
        from isaaclab.devices.teleop_device_factory import create_teleop_device
        from isaaclab.envs import ManagerBasedRLEnvCfg
        from isaaclab.managers import TerminationTermCfg as DoneTerm

        import isaaclab_tasks  # noqa: F401
        import koch_mimic.cloud.tasks.koch_pick_place  # noqa: F401
        from isaaclab_tasks.manager_based.manipulation.lift import mdp
        from isaaclab_tasks.utils import parse_env_cfg

        if args.enable_pinocchio:
            import isaaclab_tasks.manager_based.locomanipulation.pick_place  # noqa: F401
            import isaaclab_tasks.manager_based.manipulation.pick_place  # noqa: F401

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

            print("Keyboard shortcuts enabled: press R to reset the environment.")
            return shortcut_listener

        env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
        env_cfg.env_name = args.task
        if not isinstance(env_cfg, ManagerBasedRLEnvCfg):
            raise ValueError(
                "Koch teleoperation is only supported for ManagerBasedRLEnv environments. "
                f"Received: {type(env_cfg).__name__}"
            )

        env_cfg.terminations.time_out = None
        if "Lift" in args.task:
            env_cfg.commands.object_pose.resampling_time_range = (1.0e9, 1.0e9)
            env_cfg.terminations.object_reached_goal = DoneTerm(func=mdp.object_reached_goal)

        teleop_mode = configure_teleop_env(env_cfg, args)

        try:
            env = gym.make(args.task, cfg=env_cfg).unwrapped
        except Exception as exc:
            logger.error(f"Failed to create environment: {exc}")
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

        teleop_interface = create_teleop_device(teleop_mode.device_name, env_cfg.teleop_devices.devices, teleop_callbacks)
        keyboard_shortcuts = setup_keyboard_shortcuts(teleop_callbacks, teleop_interface)

        print(f"Using teleop device: {teleop_interface}")
        for message in teleop_mode_startup_messages(teleop_mode, args):
            print(message)

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
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
