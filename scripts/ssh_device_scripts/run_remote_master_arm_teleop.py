"""Cloud-side teleop entry for the custom Koch environment."""

from __future__ import annotations

import argparse
import logging
import os
from collections.abc import Callable

from isaaclab.app import AppLauncher

from koch_teleop_mode_utils import add_mobile_base_args, add_remote_master_arm_args, add_teleop_mode_args


parser = argparse.ArgumentParser(
    description=(
        "Run the custom Koch Isaac Lab environment with two teleop mode flags: "
        "fixed/mobile base and keyboard/remote master-arm control."
    )
)
parser.add_argument("--task", type=str, default="Isaac-Koch-Pick-Place-IK-Rel-Mimic-v0", help="Task name.")
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
args_cli = parser.parse_args()

app_launcher_args = vars(args_cli)

if args_cli.enable_pinocchio:
    import pinocchio  # noqa: F401

app_launcher = AppLauncher(app_launcher_args)
simulation_app = app_launcher.app


import gymnasium as gym
import torch

from isaaclab.devices import Se3Keyboard, Se3KeyboardCfg
from isaaclab.devices.teleop_device_factory import create_teleop_device
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.manager_based.manipulation.lift import mdp
from isaaclab_tasks.utils import parse_env_cfg

from koch_teleop_mode_utils import configure_teleop_env, teleop_mode_startup_messages

if args_cli.enable_pinocchio:
    import isaaclab_tasks.manager_based.locomanipulation.pick_place  # noqa: F401
    import isaaclab_tasks.manager_based.manipulation.pick_place  # noqa: F401


logger = logging.getLogger(__name__)


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
            "Koch teleoperation is only supported for ManagerBasedRLEnv environments. "
            f"Received: {type(env_cfg).__name__}"
        )

    env_cfg.terminations.time_out = None
    if "Lift" in args_cli.task:
        env_cfg.commands.object_pose.resampling_time_range = (1.0e9, 1.0e9)
        env_cfg.terminations.object_reached_goal = DoneTerm(func=mdp.object_reached_goal)

    teleop_mode = configure_teleop_env(env_cfg, args_cli)

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

    teleop_interface = create_teleop_device(teleop_mode.device_name, env_cfg.teleop_devices.devices, teleop_callbacks)
    keyboard_shortcuts = setup_keyboard_shortcuts(teleop_callbacks, teleop_interface)

    print(f"Using teleop device: {teleop_interface}")
    for message in teleop_mode_startup_messages(teleop_mode, args_cli):
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


if __name__ == "__main__":
    main()
    simulation_app.close()
