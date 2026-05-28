"""Inspect local Koch/IsaacLab HDF5 trajectory datasets.

This script is intentionally independent from Isaac Sim. It only needs h5py and
numpy, then reads HDF5 files under the local dataset directory.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections.abc import Mapping, MutableMapping
from pathlib import Path
from typing import Any


DEFAULT_DATASET_DIR = Path(r"D:\Codes\RoboticsProject\datasets\0527")
DEFAULT_FIELDS = (
    "actions",
    "processed_actions",
    "obs/eef_pos",
    "obs/eef_quat",
    "obs/gripper_pos",
    "obs/joint_pos",
    "obs/joint_vel",
    "obs/object_a_pose",
    "obs/object_b_pose",
)

IMAGE_FIELD_HINTS = ("rgb", "image", "camera", "depth", "segmentation")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect local HDF5 trajectory datasets without launching Isaac Sim."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help=f"Directory containing .hdf5 files. Default: {DEFAULT_DATASET_DIR}",
    )
    parser.add_argument(
        "--file",
        action="append",
        default=[],
        help=(
            "HDF5 file path or filename under --dataset-dir. Can be repeated. "
            "If omitted, every .hdf5 file in --dataset-dir is inspected."
        ),
    )
    parser.add_argument(
        "-n",
        "--num-trajectories",
        type=int,
        default=3,
        help="Number of trajectories/demos to read from each file. Use 0 or a negative value for all demos.",
    )
    parser.add_argument(
        "--demos",
        nargs="+",
        default=[],
        help='Specific demo names or indices to read, for example: --demos 0 2 demo_7. Overrides "-n".',
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=3,
        help="Number of timesteps/rows to print from each selected dataset.",
    )
    parser.add_argument(
        "--fields",
        nargs="+",
        default=[],
        help=(
            "Relative dataset paths inside each demo to preview. "
            "Default prints common low-dimensional fields such as actions and obs/joint_pos."
        ),
    )
    parser.add_argument(
        "--all-fields",
        action="store_true",
        help="Preview every non-image dataset inside each selected demo.",
    )
    parser.add_argument(
        "--include-images",
        action="store_true",
        help="Include image-like datasets in listings/previews. Image values are summarized, not fully printed.",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print min/max/mean/std for selected numeric datasets.",
    )
    parser.add_argument(
        "--tree",
        action="store_true",
        help="Print the full dataset tree for each selected demo.",
    )
    parser.add_argument(
        "--no-values",
        action="store_true",
        help="Only print shapes/dtypes and attributes; do not print data samples.",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=4,
        help="Floating-point precision used when printing sample arrays.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to save the inspected summary as JSON.",
    )
    parser.add_argument(
        "--mimic-diagnose",
        action="store_true",
        help=(
            "Print Koch Mimic-specific failure diagnostics using the effective cloud config "
            "(defaults, then cloud.user.local.yaml, then optional --cloud-config)."
        ),
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Repository root used to resolve configs. Default: auto-detect from this script path.",
    )
    parser.add_argument(
        "--cloud-config",
        type=Path,
        default=None,
        help="Optional cloud YAML overlay, equivalent to the cloud scripts' --config file.",
    )
    return parser


def import_hdf5_modules():
    try:
        import h5py  # type: ignore
        import numpy as np  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on the local Python environment
        message = (
            "Failed to import h5py/numpy in the current Python environment.\n"
            f"Original error: {type(exc).__name__}: {exc}\n\n"
            "Install compatible packages in the Python environment used to run this script, for example:\n"
            "  python -m pip install --upgrade --force-reinstall numpy h5py\n"
        )
        raise SystemExit(message) from exc
    return h5py, np


def import_yaml_module():
    try:
        import yaml  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on the local Python environment
        message = (
            "--mimic-diagnose requires PyYAML to read the project cloud config.\n"
            f"Original error: {type(exc).__name__}: {exc}\n\n"
            "Install it in the Python environment used to run this script, for example:\n"
            "  python -m pip install pyyaml\n"
        )
        raise SystemExit(message) from exc
    return yaml


def find_repo_root(start: Path | None = None) -> Path:
    search_start = (start or Path(__file__)).resolve()
    candidates = [search_start, *search_start.parents]
    for candidate in candidates:
        root = candidate if candidate.is_dir() else candidate.parent
        if (root / "configs").is_dir() and (root / "source" / "koch_mimic").is_dir():
            return root
    raise FileNotFoundError("Could not locate repo root containing configs/ and source/koch_mimic/.")


def read_yaml_mapping(path: Path, yaml: Any) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping in {path}, got {type(data).__name__}.")
    return data


def deep_merge(base: MutableMapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(dict(base))
    for key, value in overlay.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = deep_merge(dict(merged[key]), value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def get_nested(mapping: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    value: Any = mapping
    for key in keys:
        if not isinstance(value, Mapping) or key not in value:
            return default
        value = value[key]
    return value


def load_effective_cloud_config(args: argparse.Namespace) -> dict[str, Any]:
    yaml = import_yaml_module()
    repo_root = find_repo_root(args.repo_root)
    defaults_path = repo_root / "configs" / "cloud.defaults.yaml"
    user_local_path = repo_root / "configs" / "cloud.user.local.yaml"

    if not defaults_path.is_file():
        raise FileNotFoundError(f"Missing cloud defaults config: {defaults_path}")

    merged = read_yaml_mapping(defaults_path, yaml)
    loaded_paths = [str(defaults_path)]
    if user_local_path.is_file():
        merged = deep_merge(merged, read_yaml_mapping(user_local_path, yaml))
        loaded_paths.append(str(user_local_path))

    overlay_path = args.cloud_config
    if overlay_path is not None:
        overlay_path = overlay_path.expanduser().resolve()
        if not overlay_path.is_file():
            raise FileNotFoundError(f"Cloud overlay config does not exist: {overlay_path}")
        merged = deep_merge(merged, read_yaml_mapping(overlay_path, yaml))
        loaded_paths.append(str(overlay_path))

    return {
        "repo_root": str(repo_root),
        "loaded_paths": loaded_paths,
        "data": merged,
    }


def as_float_tuple(value: Any, fallback: tuple[float, ...]) -> tuple[float, ...]:
    if value is None:
        return fallback
    if isinstance(value, (list, tuple)):
        return tuple(float(item) for item in value)
    return fallback


def build_mimic_diagnosis_config(runtime_config: dict[str, Any]) -> dict[str, Any]:
    data = runtime_config["data"]
    return {
        "repo_root": runtime_config["repo_root"],
        "loaded_paths": runtime_config["loaded_paths"],
        "success": {
            "grasp_distance_threshold_m": float(
                get_nested(data, "success", "grasp_distance_threshold_m", default=0.12)
            ),
            "grasp_lift_height_threshold_m": float(
                get_nested(data, "success", "grasp_lift_height_threshold_m", default=0.015)
            ),
            "grasp_require_gripper_closed": bool(
                get_nested(data, "success", "grasp_require_gripper_closed", default=True)
            ),
            "grasp_lifted_requires_gripper_closed": bool(
                get_nested(data, "success", "grasp_lifted_requires_gripper_closed", default=True)
            ),
            "container_xy_half_size_m": as_float_tuple(
                get_nested(data, "success", "container_xy_half_size_m", default=None),
                (0.12, 0.12),
            ),
            "container_z_range_m": as_float_tuple(
                get_nested(data, "success", "container_z_range_m", default=None),
                (0.0, 0.20),
            ),
            "container_center_offset_m": as_float_tuple(
                get_nested(data, "success", "container_center_offset_m", default=None),
                (0.0, 0.0, 0.0),
            ),
            "require_gripper_open": bool(get_nested(data, "success", "require_gripper_open", default=True)),
        },
        "robot": {
            "gripper_open_command_rad": float(
                get_nested(data, "robot", "gripper_open_command_rad", default=1.3962634016)
            ),
            "gripper_close_command_rad": float(
                get_nested(data, "robot", "gripper_close_command_rad", default=-0.1745329252)
            ),
            "gripper_open_progress_threshold": float(
                get_nested(data, "robot", "gripper_open_progress_threshold", default=0.35)
            ),
            "gripper_closed_progress_threshold": float(
                get_nested(data, "robot", "gripper_closed_progress_threshold", default=0.65)
            ),
        },
        "mimic": {
            "generation_num_trials": int(get_nested(data, "mimic", "generation_num_trials", default=200)),
            "max_num_failures": int(get_nested(data, "mimic", "max_num_failures", default=50)),
            "action_noise": float(get_nested(data, "mimic", "action_noise", default=0.02)),
        },
    }


def print_mimic_diagnosis_config(config: dict[str, Any]) -> None:
    success = config["success"]
    robot = config["robot"]
    mimic = config["mimic"]
    print("\nEffective cloud config for Mimic diagnosis:")
    for path in config["loaded_paths"]:
        print(f"  loaded: {path}")
    print(
        "  success: "
        f"grasp_distance_threshold_m={success['grasp_distance_threshold_m']}, "
        f"container_xy_half_size_m={success['container_xy_half_size_m']}, "
        f"container_z_range_m={success['container_z_range_m']}, "
        f"container_center_offset_m={success['container_center_offset_m']}, "
        f"require_gripper_open={success['require_gripper_open']}, "
        f"grasp_lift_height_threshold_m={success['grasp_lift_height_threshold_m']}, "
        f"grasp_lifted_requires_gripper_closed={success['grasp_lifted_requires_gripper_closed']}"
    )
    print(
        "  robot: "
        f"gripper_open_command_rad={robot['gripper_open_command_rad']}, "
        f"gripper_close_command_rad={robot['gripper_close_command_rad']}, "
        f"gripper_open_progress_threshold={robot['gripper_open_progress_threshold']}, "
        f"gripper_closed_progress_threshold={robot['gripper_closed_progress_threshold']}"
    )
    print(
        "  mimic: "
        f"generation_num_trials={mimic['generation_num_trials']}, "
        f"max_num_failures={mimic['max_num_failures']}, "
        f"action_noise={mimic['action_noise']}"
    )


def safe_dataset_array(demo_group: Any, np: Any, *paths: str):
    for path in paths:
        obj = demo_group.get(path)
        if obj is not None:
            return np.asarray(obj[()])
    return None


def gripper_progress(gripper_pos: Any, config: dict[str, Any], np: Any):
    robot = config["robot"]
    open_val = float(robot["gripper_open_command_rad"])
    close_val = float(robot["gripper_close_command_rad"])
    travel = close_val - open_val
    if abs(travel) < 1e-12:
        return None
    return np.clip((gripper_pos - open_val) / travel, 0.0, 1.0)


def quat_apply_inverse(quat_wxyz: Any, vector: Any, np: Any):
    quat = np.asarray(quat_wxyz, dtype=np.float64)
    vec = np.asarray(vector, dtype=np.float64)
    w, x, y, z = quat
    q_vec = np.array([x, y, z], dtype=np.float64)
    # Apply inverse rotation without constructing a full matrix: q* v q.
    uv = np.cross(-q_vec, vec)
    uuv = np.cross(-q_vec, uv)
    return vec + 2.0 * (w * uv + uuv)


def compute_mimic_demo_diagnosis(demo_group: Any, np: Any, config: dict[str, Any]) -> dict[str, Any] | None:
    actions = safe_dataset_array(demo_group, np, "actions", "obs/actions")
    object_a_pose = safe_dataset_array(demo_group, np, "obs/object_a_pose", "states/rigid_object/cube_1/root_pose")
    object_b_pose = safe_dataset_array(demo_group, np, "obs/object_b_pose", "states/rigid_object/cube_2/root_pose")
    gripper_pos = safe_dataset_array(demo_group, np, "obs/gripper_pos")

    if actions is None and object_a_pose is None and object_b_pose is None and gripper_pos is None:
        return None

    diagnosis: dict[str, Any] = {}
    if actions is not None and actions.ndim >= 2:
        action_dim = int(actions.shape[-1])
        diagnosis["action_dim"] = action_dim
        diagnosis["action_min"] = np.min(actions, axis=0).astype(float).tolist()
        diagnosis["action_max"] = np.max(actions, axis=0).astype(float).tolist()
        diagnosis["action_mean"] = np.mean(actions, axis=0).astype(float).tolist()
        if action_dim == 5:
            wrist = actions[:, 3]
            diagnosis["wrist"] = {
                "min": float(np.min(wrist)),
                "max": float(np.max(wrist)),
                "mean": float(np.mean(wrist)),
                "std": float(np.std(wrist)),
                "all_zero": bool(np.allclose(wrist, 0.0)),
            }
        elif action_dim == 7:
            pos_delta = actions[:, :3]
            rot_delta = actions[:, 3:6]
            diagnosis["position_delta"] = {
                "min": np.min(pos_delta, axis=0).astype(float).tolist(),
                "max": np.max(pos_delta, axis=0).astype(float).tolist(),
                "mean": np.mean(pos_delta, axis=0).astype(float).tolist(),
                "max_norm": float(np.max(np.linalg.norm(pos_delta, axis=1))),
            }
            diagnosis["rotation_delta"] = {
                "min": np.min(rot_delta, axis=0).astype(float).tolist(),
                "max": np.max(rot_delta, axis=0).astype(float).tolist(),
                "mean": np.mean(rot_delta, axis=0).astype(float).tolist(),
                "max_norm": float(np.max(np.linalg.norm(rot_delta, axis=1))),
            }
        elif action_dim >= 6:
            wrist = actions[:, -2]
            diagnosis["wrist"] = {
                "min": float(np.min(wrist)),
                "max": float(np.max(wrist)),
                "mean": float(np.mean(wrist)),
                "std": float(np.std(wrist)),
                "all_zero": bool(np.allclose(wrist, 0.0)),
            }

    if object_a_pose is not None and object_a_pose.ndim >= 2 and object_a_pose.shape[-1] >= 3:
        object_a_pos = object_a_pose[:, :3]
        initial_z = float(object_a_pos[0, 2])
        max_z = float(np.max(object_a_pos[:, 2]))
        diagnosis["object_a"] = {
            "initial_z": initial_z,
            "final_z": float(object_a_pos[-1, 2]),
            "max_z": max_z,
            "max_lift": max_z - initial_z,
        }

    if object_a_pose is not None and gripper_pos is not None:
        if object_a_pose.ndim >= 2 and object_a_pose.shape[-1] >= 3:
            success_cfg = config["success"]
            object_a_pos = object_a_pose[:, :3]
            lift = object_a_pos[:, 2] - object_a_pos[0, 2]
            lifted = lift >= float(success_cfg["grasp_lift_height_threshold_m"])
            near = None
            if "obs/eef_pos" in demo_group:
                eef_pos = np.asarray(demo_group["obs/eef_pos"][()])
                if eef_pos.ndim >= 2 and eef_pos.shape[-1] >= 3 and eef_pos.shape[0] == object_a_pos.shape[0]:
                    distance = np.linalg.norm(object_a_pos - eef_pos[:, :3], axis=1)
                    near = distance <= float(success_cfg["grasp_distance_threshold_m"])
            progress = gripper_progress(gripper_pos.reshape(-1), config, np)
            if progress is not None:
                closed = progress >= float(config["robot"].get("gripper_closed_progress_threshold", 0.65))
                if near is None:
                    near = np.zeros_like(closed, dtype=bool)
                    min_distance = None
                else:
                    min_distance = float(np.min(distance))
                if bool(success_cfg["grasp_require_gripper_closed"]):
                    near_and_closed = np.logical_and(near, closed)
                    lifted_condition = np.logical_and(lifted, closed) if success_cfg["grasp_lifted_requires_gripper_closed"] else lifted
                    grasp_signal = np.logical_or(near_and_closed, lifted_condition)
                else:
                    grasp_signal = np.logical_or(near, lifted)
                diagnosis["grasp_signal"] = {
                    "near_frames": int(np.sum(near)),
                    "closed_frames": int(np.sum(closed)),
                    "lifted_frames": int(np.sum(lifted)),
                    "near_and_closed_frames": int(np.sum(np.logical_and(near, closed))),
                    "lifted_and_closed_frames": int(np.sum(np.logical_and(lifted, closed))),
                    "grasp_frames": int(np.sum(grasp_signal)),
                    "first_grasp_index": int(np.argmax(grasp_signal)) if np.any(grasp_signal) else None,
                    "min_eef_object_distance": min_distance,
                }

    if object_a_pose is not None and object_b_pose is not None:
        if object_a_pose.ndim >= 2 and object_b_pose.ndim >= 2 and object_a_pose.shape[-1] >= 3 and object_b_pose.shape[-1] >= 3:
            success_cfg = config["success"]
            offset = np.asarray(success_cfg["container_center_offset_m"], dtype=np.float64)
            rel_final_w = object_a_pose[-1, :3] - object_b_pose[-1, :3]
            if object_b_pose.shape[-1] >= 7:
                rel_final = quat_apply_inverse(object_b_pose[-1, 3:7], rel_final_w, np) - offset
            else:
                rel_final = rel_final_w - offset
            xy_half = np.asarray(success_cfg["container_xy_half_size_m"], dtype=np.float64)
            z_min, z_max = success_cfg["container_z_range_m"]
            inside_xy = bool(np.all(np.abs(rel_final[:2]) <= xy_half))
            inside_z = bool(z_min <= rel_final[2] <= z_max)
            diagnosis["container"] = {
                "final_object_a_minus_b": rel_final.astype(float).tolist(),
                "inside_xy": inside_xy,
                "inside_z": inside_z,
            }

    if gripper_pos is not None:
        progress = gripper_progress(gripper_pos.reshape(-1), config, np)
        if progress is not None:
            open_threshold = float(config["robot"]["gripper_open_progress_threshold"])
            is_open_final = bool(progress[-1] <= open_threshold)
            diagnosis["gripper"] = {
                "initial_pos": float(gripper_pos.reshape(-1)[0]),
                "final_pos": float(gripper_pos.reshape(-1)[-1]),
                "min_pos": float(np.min(gripper_pos)),
                "max_pos": float(np.max(gripper_pos)),
                "initial_progress": float(progress[0]),
                "final_progress": float(progress[-1]),
                "min_progress": float(np.min(progress)),
                "max_progress": float(np.max(progress)),
                "is_open_final": is_open_final,
            }

    container = diagnosis.get("container", {})
    gripper = diagnosis.get("gripper", {})
    require_open = bool(config["success"]["require_gripper_open"])
    success_by_config = bool(container.get("inside_xy", False) and container.get("inside_z", False))
    if require_open:
        success_by_config = success_by_config and bool(gripper.get("is_open_final", False))
    diagnosis["success_by_effective_config"] = success_by_config

    reasons = []
    if "object_a" in diagnosis:
        lift_threshold = float(config["success"]["grasp_lift_height_threshold_m"])
        if diagnosis["object_a"]["max_lift"] < lift_threshold:
            reasons.append(f"object_a max_lift below grasp threshold {lift_threshold:g} m")
    if container and not container.get("inside_xy", False):
        reasons.append("final object_a XY is outside container box")
    if container and not container.get("inside_z", False):
        reasons.append("final object_a Z is outside container range")
    if require_open and gripper and not gripper.get("is_open_final", False):
        reasons.append("final gripper is not open enough")
    diagnosis["failure_reasons"] = reasons
    return diagnosis


def format_numeric_list(values: list[float], precision: int) -> str:
    formatted = ", ".join(f"{value:.{precision}g}" for value in values)
    return f"[{formatted}]"


def print_mimic_demo_diagnosis(diagnosis: dict[str, Any], precision: int) -> None:
    print("    mimic diagnosis:")
    if "action_dim" in diagnosis:
        print(f"      action_dim: {diagnosis['action_dim']}")
        print(f"      action_min: {format_numeric_list(diagnosis['action_min'], precision)}")
        print(f"      action_max: {format_numeric_list(diagnosis['action_max'], precision)}")
    if "wrist" in diagnosis:
        wrist = diagnosis["wrist"]
        print(
            "      wrist: "
            f"min={wrist['min']:.{precision}g}, max={wrist['max']:.{precision}g}, "
            f"mean={wrist['mean']:.{precision}g}, std={wrist['std']:.{precision}g}, "
            f"all_zero={wrist['all_zero']}"
        )
    if "position_delta" in diagnosis:
        pos_delta = diagnosis["position_delta"]
        print(
            "      position_delta: "
            f"min={format_numeric_list(pos_delta['min'], precision)}, "
            f"max={format_numeric_list(pos_delta['max'], precision)}, "
            f"max_norm={pos_delta['max_norm']:.{precision}g}"
        )
    if "rotation_delta" in diagnosis:
        rot_delta = diagnosis["rotation_delta"]
        print(
            "      rotation_delta: "
            f"min={format_numeric_list(rot_delta['min'], precision)}, "
            f"max={format_numeric_list(rot_delta['max'], precision)}, "
            f"max_norm={rot_delta['max_norm']:.{precision}g}"
        )
    if "gripper" in diagnosis:
        gripper = diagnosis["gripper"]
        print(
            "      gripper: "
            f"final_pos={gripper['final_pos']:.{precision}g}, "
            f"final_progress={gripper['final_progress']:.{precision}g}, "
            f"is_open_final={gripper['is_open_final']}"
        )
    if "object_a" in diagnosis:
        obj = diagnosis["object_a"]
        print(
            "      object_a: "
            f"initial_z={obj['initial_z']:.{precision}g}, final_z={obj['final_z']:.{precision}g}, "
            f"max_z={obj['max_z']:.{precision}g}, max_lift={obj['max_lift']:.{precision}g}"
        )
    if "grasp_signal" in diagnosis:
        grasp = diagnosis["grasp_signal"]
        distance = grasp["min_eef_object_distance"]
        distance_text = "n/a" if distance is None else f"{distance:.{precision}g}"
        print(
            "      grasp_obj_a: "
            f"near_frames={grasp['near_frames']}, closed_frames={grasp['closed_frames']}, "
            f"lifted_frames={grasp['lifted_frames']}, near_and_closed_frames={grasp['near_and_closed_frames']}, "
            f"lifted_and_closed_frames={grasp['lifted_and_closed_frames']}, "
            f"grasp_frames={grasp['grasp_frames']}, first_grasp_index={grasp['first_grasp_index']}, "
            f"min_eef_object_distance={distance_text}"
        )
    if "container" in diagnosis:
        container = diagnosis["container"]
        print(
            "      container: "
            f"final_object_a_minus_b={format_numeric_list(container['final_object_a_minus_b'], precision)}, "
            f"inside_xy={container['inside_xy']}, inside_z={container['inside_z']}"
        )
    print(f"      success_by_effective_config: {diagnosis['success_by_effective_config']}")
    if diagnosis["failure_reasons"]:
        print(f"      failure_reasons: {'; '.join(diagnosis['failure_reasons'])}")


def summarize_mimic_file_diagnosis(demo_summaries: list[dict[str, Any]]) -> dict[str, Any] | None:
    diagnoses = [summary.get("mimic_diagnosis") for summary in demo_summaries if summary.get("mimic_diagnosis")]
    if not diagnoses:
        return None

    action_dims = sorted({int(item["action_dim"]) for item in diagnoses if "action_dim" in item})
    success_count = sum(1 for item in diagnoses if item.get("success_by_effective_config"))
    max_lifts = [float(item["object_a"]["max_lift"]) for item in diagnoses if "object_a" in item]
    wrist_all_zero_count = sum(1 for item in diagnoses if item.get("wrist", {}).get("all_zero"))

    summary: dict[str, Any] = {
        "diagnosed_demos": len(diagnoses),
        "success_by_effective_config": success_count,
        "action_dims": action_dims,
        "wrist_all_zero_count": wrist_all_zero_count,
    }
    if max_lifts:
        summary["object_a_max_lift"] = {
            "min": min(max_lifts),
            "max": max(max_lifts),
            "mean": sum(max_lifts) / len(max_lifts),
        }
    grasp_frame_counts = [
        int(item["grasp_signal"]["grasp_frames"]) for item in diagnoses if "grasp_signal" in item
    ]
    if grasp_frame_counts:
        summary["grasp_signal"] = {
            "demos_with_grasp": sum(1 for count in grasp_frame_counts if count > 0),
            "min_frames": min(grasp_frame_counts),
            "max_frames": max(grasp_frame_counts),
        }
    return summary


def print_mimic_file_summary(summary: dict[str, Any], precision: int) -> None:
    print("\n  Mimic diagnosis summary:")
    print(
        f"    diagnosed_demos={summary['diagnosed_demos']}, "
        f"success_by_effective_config={summary['success_by_effective_config']}, "
        f"action_dims={summary['action_dims']}, "
        f"wrist_all_zero_count={summary['wrist_all_zero_count']}"
    )
    if "object_a_max_lift" in summary:
        lift = summary["object_a_max_lift"]
        print(
            "    object_a max_lift: "
            f"min={lift['min']:.{precision}g}, max={lift['max']:.{precision}g}, "
            f"mean={lift['mean']:.{precision}g}"
        )
    if "grasp_signal" in summary:
        grasp = summary["grasp_signal"]
        print(
            "    grasp_obj_a signal: "
            f"demos_with_grasp={grasp['demos_with_grasp']}, "
            f"frame_count_range=[{grasp['min_frames']}, {grasp['max_frames']}]"
        )


def resolve_hdf5_files(dataset_dir: Path, file_args: list[str]) -> list[Path]:
    if file_args:
        files: list[Path] = []
        for item in file_args:
            path = Path(item)
            if not path.is_absolute():
                path = dataset_dir / path
            if path.is_dir():
                files.extend(sorted(path.glob("*.hdf5")))
                files.extend(sorted(path.glob("*.h5")))
            else:
                files.append(path)
    else:
        files = sorted(dataset_dir.glob("*.hdf5")) + sorted(dataset_dir.glob("*.h5"))

    unique_files: list[Path] = []
    seen: set[Path] = set()
    for path in files:
        resolved = path.resolve()
        if resolved in seen:
            continue
        if not resolved.exists():
            raise FileNotFoundError(f"HDF5 file does not exist: {resolved}")
        if not resolved.is_file():
            raise FileNotFoundError(f"Path is not a file: {resolved}")
        seen.add(resolved)
        unique_files.append(resolved)
    return unique_files


def natural_demo_key(name: str) -> tuple[int, int | str]:
    if name.startswith("demo_"):
        suffix = name.removeprefix("demo_")
        if suffix.isdigit():
            return (0, int(suffix))
    return (1, name)


def to_python(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, dict):
        return {str(k): to_python(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_python(v) for v in value]
    if hasattr(value, "item"):
        try:
            return to_python(value.item())
        except Exception:
            return str(value)
    return value


def decode_attr_value(value: Any) -> Any:
    value = to_python(value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("{") or stripped.startswith("["):
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                return value
    return value


def attrs_to_dict(obj: Any) -> dict[str, Any]:
    return {str(key): decode_attr_value(value) for key, value in obj.attrs.items()}


def demo_names_from_file(data_group: Any, h5py: Any) -> list[str]:
    demos = [name for name, obj in data_group.items() if isinstance(obj, h5py.Group)]
    return sorted(demos, key=natural_demo_key)


def select_demo_names(all_demos: list[str], explicit_demos: list[str], limit: int) -> list[str]:
    if explicit_demos:
        selected: list[str] = []
        available = set(all_demos)
        for item in explicit_demos:
            name = f"demo_{item}" if item.isdigit() else item
            if name not in available:
                raise KeyError(f"Requested demo '{item}' was not found. Available demos: {', '.join(all_demos)}")
            if name not in selected:
                selected.append(name)
        return selected

    if limit <= 0:
        return all_demos
    return all_demos[:limit]


def iter_datasets(group: Any, h5py: Any, prefix: str = ""):
    for key in sorted(group.keys(), key=natural_demo_key):
        obj = group[key]
        path = f"{prefix}/{key}" if prefix else key
        if isinstance(obj, h5py.Dataset):
            yield path, obj
        elif isinstance(obj, h5py.Group):
            yield from iter_datasets(obj, h5py, path)


def is_image_like(path: str, dataset: Any) -> bool:
    lower_path = path.lower()
    if any(hint in lower_path for hint in IMAGE_FIELD_HINTS):
        return True
    shape = tuple(dataset.shape)
    return len(shape) >= 3 and shape[-1] in (1, 3, 4)


def dataset_shape(dataset: Any) -> tuple[int, ...]:
    return tuple(int(dim) for dim in dataset.shape)


def dataset_preview(dataset: Any, np: Any, max_steps: int, precision: int, image_like: bool) -> str:
    shape = dataset_shape(dataset)
    if shape == ():
        value = dataset[()]
        return repr(to_python(value))

    if max_steps <= 0:
        return "<sample disabled because --max-steps <= 0>"

    row_count = shape[0]
    sample_count = min(max_steps, row_count)

    if image_like:
        sample = dataset[0]
        arr = np.asarray(sample)
        first_pixel = arr.reshape(-1, arr.shape[-1])[0].tolist() if arr.ndim >= 2 else arr.reshape(-1)[0].item()
        return f"<image-like sample shape={tuple(arr.shape)}, first_pixel={to_python(first_pixel)}>"

    sample = dataset[:sample_count]
    arr = np.asarray(sample)
    return np.array2string(
        arr,
        precision=precision,
        suppress_small=False,
        separator=", ",
        threshold=120,
        edgeitems=2,
    )


def dataset_stats(dataset: Any, np: Any, image_like: bool) -> dict[str, Any] | None:
    if image_like:
        return None
    if dataset.shape == ():
        return None
    if not np.issubdtype(dataset.dtype, np.number):
        return None
    if dataset.size == 0:
        return None

    arr = np.asarray(dataset[()])
    return {
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
    }


def format_stats(stats: dict[str, Any] | None) -> str:
    if stats is None:
        return ""
    return (
        f" stats(min={stats['min']:.6g}, max={stats['max']:.6g}, "
        f"mean={stats['mean']:.6g}, std={stats['std']:.6g})"
    )


def choose_fields(
    available: dict[str, Any],
    requested_fields: list[str],
    all_fields: bool,
    include_images: bool,
) -> list[str]:
    if all_fields:
        return [
            name
            for name, dataset in available.items()
            if include_images or not is_image_like(name, dataset)
        ]

    source_fields = requested_fields or list(DEFAULT_FIELDS)
    selected: list[str] = []
    for name in source_fields:
        if name not in available:
            print(f"  [warn] Field not found in this demo: {name}")
            continue
        if is_image_like(name, available[name]) and not include_images:
            print(f"  [warn] Skipping image-like field without --include-images: {name}")
            continue
        selected.append(name)
    return selected


def inspect_demo(
    demo_name: str,
    demo_group: Any,
    h5py: Any,
    np: Any,
    args: argparse.Namespace,
    mimic_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    print(f"\n  Demo: {demo_name}")
    demo_attrs = attrs_to_dict(demo_group)
    if demo_attrs:
        print(f"    attrs: {json.dumps(demo_attrs, ensure_ascii=False)}")

    available = dict(iter_datasets(demo_group, h5py))
    if args.tree:
        print("    tree:")
        for path, dataset in available.items():
            image_suffix = " image-like" if is_image_like(path, dataset) else ""
            print(f"      {path}: shape={dataset_shape(dataset)}, dtype={dataset.dtype}{image_suffix}")

    selected_fields = choose_fields(
        available,
        requested_fields=args.fields,
        all_fields=args.all_fields,
        include_images=args.include_images,
    )

    summary: dict[str, Any] = {
        "name": demo_name,
        "attrs": demo_attrs,
        "datasets": {},
    }

    if not selected_fields:
        print("    selected datasets: <none>")
        if mimic_config is not None:
            diagnosis = compute_mimic_demo_diagnosis(demo_group, np, mimic_config)
            if diagnosis is not None:
                print_mimic_demo_diagnosis(diagnosis, args.precision)
                summary["mimic_diagnosis"] = diagnosis
        return summary

    print("    selected datasets:")
    for field in selected_fields:
        dataset = available[field]
        image_like = is_image_like(field, dataset)
        stats = dataset_stats(dataset, np, image_like) if args.stats else None
        print(
            f"      {field}: shape={dataset_shape(dataset)}, dtype={dataset.dtype}"
            f"{' image-like' if image_like else ''}{format_stats(stats)}"
        )

        preview: str | None = None
        if not args.no_values:
            preview = dataset_preview(
                dataset,
                np=np,
                max_steps=args.max_steps,
                precision=args.precision,
                image_like=image_like,
            )
            indented_preview = "\n".join(f"        {line}" for line in preview.splitlines())
            print(indented_preview)

        summary["datasets"][field] = {
            "shape": dataset_shape(dataset),
            "dtype": str(dataset.dtype),
            "image_like": image_like,
            "stats": stats,
            "preview": preview,
        }
    if mimic_config is not None:
        diagnosis = compute_mimic_demo_diagnosis(demo_group, np, mimic_config)
        if diagnosis is not None:
            print_mimic_demo_diagnosis(diagnosis, args.precision)
            summary["mimic_diagnosis"] = diagnosis
    return summary


def inspect_file(
    path: Path,
    h5py: Any,
    np: Any,
    args: argparse.Namespace,
    mimic_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    print(f"\nFile: {path}")
    with h5py.File(path, "r") as h5_file:
        file_attrs = attrs_to_dict(h5_file)
        data_group = h5_file.get("data")

        summary: dict[str, Any] = {
            "file": str(path),
            "attrs": file_attrs,
            "data_attrs": {},
            "demo_count": 0,
            "selected_demos": [],
        }

        if file_attrs:
            print(f"  file attrs: {json.dumps(file_attrs, ensure_ascii=False)}")

        if data_group is None:
            print("  No /data group found.")
            return summary

        data_attrs = attrs_to_dict(data_group)
        summary["data_attrs"] = data_attrs
        if data_attrs:
            print(f"  /data attrs: {json.dumps(data_attrs, ensure_ascii=False)}")

        all_demos = demo_names_from_file(data_group, h5py)
        summary["demo_count"] = len(all_demos)
        if not all_demos:
            print("  No demo groups found under /data.")
            return summary

        lengths = []
        for demo_name in all_demos:
            demo_group = data_group[demo_name]
            if "num_samples" in demo_group.attrs:
                lengths.append(int(to_python(demo_group.attrs["num_samples"])))
            elif "actions" in demo_group:
                lengths.append(int(demo_group["actions"].shape[0]))

        print(f"  demos: {len(all_demos)}")
        if lengths:
            print(
                "  trajectory lengths: "
                f"min={min(lengths)}, max={max(lengths)}, total={sum(lengths)}, "
                f"mean={sum(lengths) / len(lengths):.2f}"
            )

        selected = select_demo_names(all_demos, args.demos, args.num_trajectories)
        print(f"  selected demos: {', '.join(selected)}")

        demo_summaries = []
        for demo_name in selected:
            demo_summaries.append(inspect_demo(demo_name, data_group[demo_name], h5py, np, args, mimic_config))
        summary["selected_demos"] = demo_summaries
        if mimic_config is not None:
            mimic_summary = summarize_mimic_file_diagnosis(demo_summaries)
            if mimic_summary is not None:
                print_mimic_file_summary(mimic_summary, args.precision)
                summary["mimic_diagnosis_summary"] = mimic_summary
        return summary


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    h5py, np = import_hdf5_modules()
    mimic_config = None
    if args.mimic_diagnose:
        mimic_config = build_mimic_diagnosis_config(load_effective_cloud_config(args))
        print_mimic_diagnosis_config(mimic_config)

    files = resolve_hdf5_files(args.dataset_dir, args.file)
    if not files:
        raise SystemExit(f"No .hdf5 or .h5 files found under: {args.dataset_dir}")

    summaries = [inspect_file(path, h5py, np, args, mimic_config) for path in files]

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nWrote JSON summary: {args.output_json}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        raise SystemExit(130)
