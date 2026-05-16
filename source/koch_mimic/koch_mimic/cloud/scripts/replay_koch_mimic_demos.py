"""Replay demonstrations for the Koch Mimic task."""

from __future__ import annotations

import argparse
import contextlib
import logging
import os
from pathlib import Path
import sys
from collections.abc import Sequence

from isaaclab.app import AppLauncher

from koch_mimic.shared.configuration import (
    activate_runtime_config,
    get_config_section,
    resolve_config_path,
)
from koch_mimic.shared.constants import CLOUD_PROFILE, DEFAULT_TASK_ID


logger = logging.getLogger(__name__)


is_paused = False


def play_cb() -> None:
    global is_paused
    is_paused = False


def pause_cb() -> None:
    global is_paused
    is_paused = True


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Replay demonstrations in the Koch Mimic Isaac Lab environment.")
    parser.add_argument("--config", type=str, default=None, help="Optional extra cloud YAML overlay.")
    parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to replay episodes.")
    parser.add_argument("--task", type=str, default=None, help="Force use of the specified task.")
    parser.add_argument(
        "--select_episodes",
        type=int,
        nargs="+",
        default=[],
        help="Episode indices to replay. Leave empty to replay all episodes in the dataset.",
    )
    parser.add_argument("--dataset_file", type=str, default=None, help="Dataset file to replay.")
    parser.add_argument(
        "--validate_states",
        action="store_true",
        default=False,
        help="Validate recorded states against runtime states. Only supported with --num_envs 1.",
    )
    parser.add_argument(
        "--validate_success_rate",
        action="store_true",
        default=False,
        help="Validate replay success rate using the task termination criteria.",
    )
    parser.add_argument(
        "--enable_pinocchio",
        action="store_true",
        default=False,
        help="Enable Pinocchio before launching Isaac Sim.",
    )
    parser.add_argument(
        "--pause_on_start",
        action="store_true",
        default=False,
        help='Pause before the first replayed action. Press "N" to start when a viewport is available.',
    )
    parser.add_argument(
        "--keep_alive_after_replay",
        action="store_true",
        default=False,
        help="Keep rendering after replay completes instead of closing immediately.",
    )
    AppLauncher.add_app_launcher_args(parser)
    return parser


def resolve_runtime_args(args: argparse.Namespace) -> None:
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


def compare_states(state_from_dataset, runtime_state, runtime_env_index: int) -> tuple[bool, str]:
    """Compare states from the dataset and runtime."""
    states_matched = True
    output_log = ""
    for asset_type in ["articulation", "rigid_object"]:
        for asset_name in runtime_state[asset_type].keys():
            for state_name in runtime_state[asset_type][asset_name].keys():
                runtime_asset_state = runtime_state[asset_type][asset_name][state_name][runtime_env_index]
                dataset_asset_state = state_from_dataset[asset_type][asset_name][state_name]
                if len(dataset_asset_state) != len(runtime_asset_state):
                    raise ValueError(f"State shape of {state_name} for asset {asset_name} do not match")
                for index in range(len(dataset_asset_state)):
                    if abs(dataset_asset_state[index] - runtime_asset_state[index]) > 0.01:
                        states_matched = False
                        output_log += (
                            f'\tState ["{asset_type}"]["{asset_name}"]["{state_name}"][{index}] do not match\r\n'
                        )
                        output_log += f"\t  Dataset:\t{dataset_asset_state[index]}\r\n"
                        output_log += f"\t  Runtime: \t{runtime_asset_state[index]}\r\n"
    return states_matched, output_log


def main(argv: Sequence[str] | None = None) -> None:
    global is_paused

    argv_list = list(sys.argv[1:] if argv is None else argv)
    parser = build_arg_parser()
    args = parser.parse_args(argv_list)
    resolve_runtime_args(args)

    app_launcher_args = vars(args).copy()
    if args.enable_pinocchio:
        import pinocchio  # noqa: F401

    app_launcher = AppLauncher(app_launcher_args)
    simulation_app = app_launcher.app

    try:
        import gymnasium as gym
        import torch

        from isaaclab.devices import Se3Keyboard, Se3KeyboardCfg
        from isaaclab.utils.datasets import EpisodeData, HDF5DatasetFileHandler

        if args.enable_pinocchio:
            import isaaclab_tasks.manager_based.locomanipulation.pick_place  # noqa: F401
            import isaaclab_tasks.manager_based.manipulation.pick_place  # noqa: F401

        import isaaclab_tasks  # noqa: F401
        import koch_mimic.cloud.tasks.koch_pick_place  # noqa: F401
        from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

        if not os.path.exists(args.dataset_file):
            raise FileNotFoundError(f"The dataset file {args.dataset_file} does not exist.")

        dataset_file_handler = HDF5DatasetFileHandler()
        dataset_file_handler.open(args.dataset_file)
        env_name = dataset_file_handler.get_env_name()
        episode_count = dataset_file_handler.get_num_episodes()

        if episode_count == 0:
            print("No episodes found in the dataset.")
            return

        episode_indices_to_replay = args.select_episodes
        if len(episode_indices_to_replay) == 0:
            episode_indices_to_replay = list(range(episode_count))

        if args.task is not None:
            env_name = args.task.split(":")[-1]
        if env_name is None:
            raise ValueError("Task/env name was not specified nor found in the dataset.")

        env_cfg = parse_env_cfg(env_name, device=args.device, num_envs=args.num_envs)
        env_cfg.env_name = env_name

        success_term = None
        if args.validate_success_rate:
            if hasattr(env_cfg.terminations, "success"):
                success_term = env_cfg.terminations.success
                env_cfg.terminations.success = None
            else:
                print(
                    "No success termination term was found in the environment."
                    " Will not be able to mark replayed demos as successful."
                )

        env_cfg.recorders = {}
        env_cfg.terminations = {}

        env = gym.make(args.task or env_name, cfg=env_cfg).unwrapped

        keyboard_listener = None
        if not args.headless and os.environ.get("HEADLESS", "0") in ("0", "", "False", "false"):
            keyboard_listener = Se3Keyboard(Se3KeyboardCfg(pos_sensitivity=0.1, rot_sensitivity=0.1))
            keyboard_listener.add_callback("N", play_cb)
            keyboard_listener.add_callback("B", pause_cb)
            print('Press "B" to pause and "N" to resume the replayed actions.')

        state_validation_enabled = False
        if args.validate_states and args.num_envs == 1:
            state_validation_enabled = True
        elif args.validate_states and args.num_envs > 1:
            print("Warning: State validation is only supported with a single environment. Skipping state validation.")

        if hasattr(env_cfg, "idle_action"):
            idle_action = env_cfg.idle_action.repeat(args.num_envs, 1)
        else:
            idle_action = torch.zeros(env.action_space.shape, device=env.device)

        env.reset()
        if keyboard_listener is not None:
            keyboard_listener.reset()
        if args.pause_on_start:
            is_paused = True
            print('Replay is paused before the first action. Press "N" to start and "B" to pause again.')
        else:
            is_paused = False

        episode_names = list(dataset_file_handler.get_episode_names())
        replayed_episode_count = 0
        recorded_episode_count = 0
        current_episode_indices = [None] * args.num_envs
        failed_demo_ids = []

        with contextlib.suppress(KeyboardInterrupt) and torch.inference_mode():
            while simulation_app.is_running() and not simulation_app.is_exiting():
                env_episode_data_map = {index: EpisodeData() for index in range(args.num_envs)}
                first_loop = True
                has_next_action = True
                episode_ended = [False] * args.num_envs
                while has_next_action:
                    actions = idle_action.clone()
                    has_next_action = False
                    for env_id in range(args.num_envs):
                        env_next_action = env_episode_data_map[env_id].get_next_action()
                        if env_next_action is None:
                            if (
                                (success_term is not None)
                                and (current_episode_indices[env_id] is not None)
                                and (not episode_ended[env_id])
                            ):
                                if bool(success_term.func(env, **success_term.params)[env_id]):
                                    recorded_episode_count += 1
                                    plural_suffix = "s" if recorded_episode_count > 1 else ""
                                    print(
                                        f"Successfully replayed {recorded_episode_count} episode{plural_suffix} out"
                                        f" of {replayed_episode_count} demos."
                                    )
                                else:
                                    if (
                                        current_episode_indices[env_id] is not None
                                        and current_episode_indices[env_id] not in failed_demo_ids
                                    ):
                                        failed_demo_ids.append(current_episode_indices[env_id])

                                episode_ended[env_id] = True

                            next_episode_index = None
                            while episode_indices_to_replay:
                                next_episode_index = episode_indices_to_replay.pop(0)
                                if next_episode_index < episode_count:
                                    episode_ended[env_id] = False
                                    break
                                next_episode_index = None

                            if next_episode_index is not None:
                                replayed_episode_count += 1
                                current_episode_indices[env_id] = next_episode_index
                                print(f"{replayed_episode_count:4}: Loading #{next_episode_index} episode to env_{env_id}")
                                episode_data = dataset_file_handler.load_episode(
                                    episode_names[next_episode_index], env.device
                                )
                                env_episode_data_map[env_id] = episode_data
                                initial_state = episode_data.get_initial_state()
                                env.reset_to(initial_state, torch.tensor([env_id], device=env.device), is_relative=True)
                                env_next_action = env_episode_data_map[env_id].get_next_action()
                                has_next_action = True
                            else:
                                continue
                        else:
                            has_next_action = True
                        actions[env_id] = env_next_action

                    if first_loop:
                        first_loop = False
                    else:
                        while is_paused:
                            env.sim.render()
                            continue

                    env.step(actions)

                    if state_validation_enabled:
                        state_from_dataset = env_episode_data_map[0].get_next_state()
                        if state_from_dataset is not None:
                            print(
                                f"Validating states at action-index:"
                                f" {env_episode_data_map[0].next_state_index - 1:4}",
                                end="",
                            )
                            current_runtime_state = env.scene.get_state(is_relative=True)
                            states_matched, comparison_log = compare_states(state_from_dataset, current_runtime_state, 0)
                            if states_matched:
                                print("\t- matched.")
                            else:
                                print("\t- mismatched.")
                                print(comparison_log)
                break

        plural_suffix = "s" if replayed_episode_count > 1 else ""
        print(f"Finished replaying {replayed_episode_count} episode{plural_suffix}.")

        if success_term is not None:
            print(f"Successfully replayed: {recorded_episode_count}/{replayed_episode_count}")
            if failed_demo_ids:
                print(f"\nFailed demo IDs ({len(failed_demo_ids)} total):")
                print(f"  {sorted(failed_demo_ids)}")

        if args.keep_alive_after_replay:
            print("Replay completed. Keeping the simulator open; close the window or press Ctrl+C to exit.")
            with contextlib.suppress(KeyboardInterrupt):
                while simulation_app.is_running() and not simulation_app.is_exiting():
                    env.sim.render()

        env.close()
        dataset_file_handler.close()

    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
