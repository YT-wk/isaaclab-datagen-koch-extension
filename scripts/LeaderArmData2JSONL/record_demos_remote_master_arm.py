"""远程主手臂版演示录制脚本。

这个脚本是在 Isaac Lab 官方 `record_demos.py` 的基础上扩出来的，
目的只有一个：在尽量保留官方录制流程的前提下，新增 `external_master_arm`
 这种远程真实 leader 设备。

因此它同时承担两类职责：
- 维持官方录制脚本的 recorder / success episode 导出逻辑；
- 在 teleop_device=external_master_arm 时，动态注入云端 TCP 接收设备。
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import os
import time
from collections.abc import Callable

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(
    description="Record demonstrations for Isaac Lab environments with optional remote Koch master-arm teleop."
)
# 录制脚本比纯遥操作脚本多了一层 recorder 配置，因此这里既支持普通 teleop，
# 也支持 external_master_arm 这种远程设备。
parser.add_argument("--task", type=str, required=True, help="Name of the task.")
parser.add_argument(
    "--teleop_device",
    type=str,
    default="keyboard",
    help=(
        "Teleop device. Built-ins: keyboard, spacemouse. "
        "Custom env-config devices are also supported, including external_master_arm."
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
    "--gripper-close-threshold",
    type=float,
    default=0.15,
    help="Leader gripper delta threshold in radians for switching from open to close.",
)
parser.add_argument(
    "--gripper-close-direction",
    type=str,
    choices=("positive", "negative"),
    default="negative",
    help="Whether the leader gripper closes when its angle moves in the positive or negative direction.",
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
if "handtracking" in args_cli.teleop_device.lower():
    app_launcher_args["xr"] = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym
import omni.ui as ui
import torch

from isaaclab.devices import Se3Keyboard, Se3KeyboardCfg, Se3SpaceMouse, Se3SpaceMouseCfg
from isaaclab.devices.openxr import remove_camera_configs
from isaaclab.devices.teleop_device_factory import create_teleop_device
from isaaclab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.actions.actions_cfg import AbsBinaryJointPositionActionCfg, JointPositionActionCfg
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

from koch_master_arm_stream_device import KochMasterArmStreamDevice, KochMasterArmStreamDeviceCfg


logger = logging.getLogger(__name__)


class RateLimiter:
    """简单的定频器，让录制过程更接近固定步频。"""

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
    """解析命令行里的逗号分隔浮点数组。"""
    parts = [item.strip() for item in value.split(",") if item.strip()]
    if len(parts) != expected_len:
        raise ValueError(f"Expected {expected_len} comma-separated values, got {len(parts)} from: {value}")
    return tuple(float(item) for item in parts)


def setup_output_directories() -> tuple[str, str]:
    """准备录制输出目录与数据集文件名。"""
    output_dir = os.path.dirname(args_cli.dataset_file)
    output_file_name = os.path.splitext(os.path.basename(args_cli.dataset_file))[0]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    return output_dir, output_file_name


def maybe_inject_external_master_arm(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg) -> None:
    """在需要时把 external_master_arm 注入到环境配置里。

    官方 `record_demos.py` 并不知道我们自定义的远程设备，因此这里在创建环境前
    动态把动作配置和 teleop 设备配置改写成适合远程主手臂的形式。
    """
    if args_cli.teleop_device != "external_master_arm":
        return

    if not isinstance(env_cfg, ManagerBasedRLEnvCfg):
        raise ValueError("external_master_arm is only supported for ManagerBasedRLEnv environments.")

    if not hasattr(env_cfg, "koch_arm_joint_names") or not hasattr(env_cfg, "koch_gripper_joint_names"):
        raise AttributeError(
            "The selected environment config does not expose 'koch_arm_joint_names' and "
            "'koch_gripper_joint_names', so it cannot be controlled by the Koch master arm bridge."
        )

    joint_signs = parse_csv_floats(args_cli.joint_signs, 5)
    joint_offsets = parse_csv_floats(args_cli.joint_offsets, 5)

    # 对远程主手臂来说，我们希望直接控制绝对关节目标，而不是走默认键盘路径。
    env_cfg.actions.arm_action = JointPositionActionCfg(
        asset_name="robot",
        joint_names=list(env_cfg.koch_arm_joint_names),
        scale=1.0,
        offset=0.0,
        preserve_order=True,
        use_default_offset=False,
    )
    env_cfg.actions.wrist_action = None
    env_cfg.actions.gripper_action = AbsBinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=list(env_cfg.koch_gripper_joint_names),
        open_command_expr={name: env_cfg.koch_gripper_open_command for name in env_cfg.koch_gripper_joint_names},
        close_command_expr={name: env_cfg.koch_gripper_close_command for name in env_cfg.koch_gripper_joint_names},
        threshold=0.5,
        positive_threshold=True,
    )

    master_arm_device_cfg = KochMasterArmStreamDeviceCfg(
        sim_device=args_cli.device,
        host=args_cli.stream_host,
        port=args_cli.stream_port,
        joint_signs=joint_signs,
        joint_offsets=joint_offsets,
        zero_on_first_frame=not args_cli.no_zero_on_start,
        gripper_close_threshold=args_cli.gripper_close_threshold,
        gripper_close_direction=args_cli.gripper_close_direction,
        stale_timeout=args_cli.stale_timeout,
        class_type=KochMasterArmStreamDevice,
    )

    if hasattr(env_cfg, "teleop_devices"):
        env_cfg.teleop_devices.devices["external_master_arm"] = master_arm_device_cfg
    if hasattr(env_cfg, "external_master_arm_device"):
        env_cfg.external_master_arm_device = master_arm_device_cfg


def create_environment_config(
    output_dir: str, output_file_name: str
) -> tuple[ManagerBasedRLEnvCfg | DirectRLEnvCfg, object | None]:
    """解析并补全环境配置。"""
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

    # 录制场景通常希望 episode 由成功条件决定，而不是被 time_out 硬切断。
    env_cfg.terminations.time_out = None
    env_cfg.observations.policy.concatenate_terms = False

    # 这里是录制脚本和纯遥操作脚本最大的不同点：
    # 需要给环境挂 recorder manager，并指定导出路径与导出策略。
    env_cfg.recorders = ActionStateRecorderManagerCfg()
    env_cfg.recorders.dataset_export_dir_path = output_dir
    env_cfg.recorders.dataset_filename = output_file_name
    env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY

    maybe_inject_external_master_arm(env_cfg)
    return env_cfg, success_term


def create_environment(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg) -> gym.Env:
    """根据配置实例化环境。"""
    try:
        return gym.make(args_cli.task, cfg=env_cfg).unwrapped
    except Exception as exc:
        logger.error(f"Failed to create environment: {exc}")
        raise SystemExit(1) from exc


def setup_teleop_device(callbacks: dict[str, Callable], env_cfg) -> object:
    """创建 teleop 设备。

    如果环境配置里已经声明了目标设备，就走配置化创建设备；
    否则再退回到 keyboard / spacemouse 这样的内置设备。
    """
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


def setup_ui(label_text: str, env: gym.Env) -> InstructionDisplay:
    """初始化录制时的提示界面。"""
    instruction_display = InstructionDisplay(args_cli.xr)
    if not args_cli.xr:
        window = EmptyWindow(env, "Instruction")
        with window.ui_window_elements["main_vstack"]:
            demo_label = ui.Label(label_text)
            subtask_label = ui.Label("")
            instruction_display.set_labels(subtask_label, demo_label)
    return instruction_display


def process_success_condition(env: gym.Env, success_term: object | None, success_step_count: int) -> tuple[int, bool]:
    """处理成功判据，并在满足条件时导出演示数据。"""
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
    """重置环境、recorder 和界面状态。"""
    print("Resetting environment...")
    env.sim.reset()
    env.recorder_manager.reset()
    env.reset()
    success_step_count = 0
    instruction_display.show_demo(label_text)
    return success_step_count


def run_simulation_loop(env: gym.Env, env_cfg, success_term: object | None, rate_limiter: RateLimiter | None) -> int:
    """执行主录制循环。"""
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

    # 录制开始前先做一次 reset，确保环境和 teleop 设备都处于干净状态。
    env.sim.reset()
    env.reset()
    teleop_interface.reset()

    if args_cli.teleop_device == "external_master_arm":
        print(
            "Waiting for master-arm stream on "
            f"{args_cli.stream_host}:{args_cli.stream_port}. "
            "Start koch_leader_ssh_streamer.py on the local machine after this script is ready."
        )

    label_text = f"Recorded {current_recorded_demo_count} successful demonstrations."
    instruction_display = setup_ui(label_text, env)
    subtasks = {}

    with contextlib.suppress(KeyboardInterrupt) and torch.inference_mode():
        while simulation_app.is_running():
            # 录制逻辑与普通遥操作一样，也是“设备给动作 -> 环境 step”。
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

    return current_recorded_demo_count


def main() -> None:
    """脚本入口。"""
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
