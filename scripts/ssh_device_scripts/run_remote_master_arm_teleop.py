"""云端远程遥操作入口。

这个脚本适合“实时遥操作”场景：本地 leader 机械臂持续发流，
云端 Isaac Lab 环境持续接收并驱动仿真机械臂。

它和 `record_demos_remote_master_arm.py` 的区别在于：
- 这里偏向实时遥操作，不负责录制数据集；
- 录制脚本会额外注入 recorder 配置，并按成功 episode 导出 HDF5。
"""

from __future__ import annotations

import argparse
import logging
import os
from collections.abc import Callable

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(
    description=(
        "Run the custom Koch Isaac Lab environment with a streamed master-arm device "
        "received over an SSH-tunneled TCP connection."
    )
)
# 这组参数本质上分三类：
# 1. 环境本身参数（task / num_envs / device）
# 2. 网络监听参数（stream-host / stream-port）
# 3. leader 到仿真关节的映射参数（joint-signs / joint-offsets / gripper-*）
parser.add_argument("--task", type=str, default="Isaac-Koch-Pick-Place-IK-Rel-Mimic-v0", help="Task name.")
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

from isaaclab.devices import Se3Keyboard, Se3KeyboardCfg
from isaaclab.devices.teleop_device_factory import create_teleop_device
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.manager_based.manipulation.lift import mdp
from isaaclab_tasks.utils import parse_env_cfg

from koch_master_arm_stream_device import KochMasterArmStreamDevice, KochMasterArmStreamDeviceCfg

if args_cli.enable_pinocchio:
    import isaaclab_tasks.manager_based.locomanipulation.pick_place  # noqa: F401
    import isaaclab_tasks.manager_based.manipulation.pick_place  # noqa: F401


logger = logging.getLogger(__name__)


def parse_csv_floats(value: str, expected_len: int) -> tuple[float, ...]:
    """解析逗号分隔的浮点数组参数。"""
    parts = [item.strip() for item in value.split(",") if item.strip()]
    if len(parts) != expected_len:
        raise ValueError(f"Expected {expected_len} comma-separated values, got {len(parts)} from: {value}")
    return tuple(float(item) for item in parts)


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


def setup_keyboard_shortcuts(callbacks: dict[str, Callable], teleop_interface: object) -> Se3Keyboard | None:
    """Enable keyboard-only hotkeys when the main teleop device is not keyboard based."""
    if args_cli.headless or os.environ.get("HEADLESS", "0") not in ("0", "", "False", "false"):
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
    """主函数。

    这里做的关键工作不是“读取 leader 数据”，而是：
    1. 重写环境动作配置，使其接收远程主手臂的 6 维动作。
    2. 注入 external_master_arm 设备。
    3. 在仿真主循环里从设备读取动作并送给环境。
    """
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

    joint_signs = parse_csv_floats(args_cli.joint_signs, 5)
    joint_offsets = parse_csv_floats(args_cli.joint_offsets, 5)
    gripper_open_command, gripper_close_command, gripper_close_delta, gripper_close_direction = (
        resolve_external_master_arm_gripper_cfg(env_cfg)
    )

    if not hasattr(env_cfg, "koch_arm_joint_names") or not hasattr(env_cfg, "koch_gripper_joint_names"):
        raise AttributeError(
            "The selected environment config does not expose 'koch_arm_joint_names' and "
            "'koch_gripper_joint_names', so it cannot be controlled by the Koch master arm bridge."
        )

    # 这里主动覆盖环境默认动作项。
    # 原本环境更偏向键盘/IK 交互，而远程主手臂直接输出的是“关节目标”。
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

    # 构造云端接收设备配置，并把映射参数全部注入进去。
    master_arm_device_cfg = KochMasterArmStreamDeviceCfg(
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

    if hasattr(env_cfg, "teleop_devices"):
        env_cfg.teleop_devices.devices["external_master_arm"] = master_arm_device_cfg
    if hasattr(env_cfg, "external_master_arm_device"):
        env_cfg.external_master_arm_device = master_arm_device_cfg

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

    # 这里固定选择 external_master_arm，因为这个脚本就是专门为它服务的。
    teleop_interface = create_teleop_device("external_master_arm", env_cfg.teleop_devices.devices, teleop_callbacks)
    keyboard_shortcuts = setup_keyboard_shortcuts(teleop_callbacks, teleop_interface)
    print(f"Using teleop device: {teleop_interface}")
    print(
        "Waiting for master-arm stream on "
        f"{args_cli.stream_host}:{args_cli.stream_port}. "
        "Start koch_leader_ssh_streamer.py on the local machine after this script is ready."
    )

    env.reset()
    teleop_interface.reset()

    while simulation_app.is_running():
        try:
            with torch.inference_mode():
                action = teleop_interface.advance()
                if teleoperation_active:
                    # teleop device 输出单环境动作，Isaac Lab step 需要 batch 维度。
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
