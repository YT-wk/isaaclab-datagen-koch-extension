"""Compatibility helpers for annotating Koch demos recorded with teleop action spaces."""

from __future__ import annotations


def configure_annotation_action_space_for_dataset(env_cfg, input_file: str | None) -> None:
    """Adapt annotation replay to the action dimensionality stored in an HDF5 dataset.

    Isaac Lab Mimic generation should use the task's Cartesian IK action space, but annotation must first replay
    the raw teleoperation actions exactly as recorded. Remote-master-arm demos in this project are direct joint
    targets, so their action dimension differs from the default Mimic IK action dimension.
    """
    action_dim = _read_first_demo_action_dim(input_file)
    if action_dim is None:
        return

    fixed_keyboard_dim = 5
    mobile_keyboard_dim = fixed_keyboard_dim + len(tuple(getattr(env_cfg, "koch_base_wheel_joint_names", ())))
    fixed_master_arm_dim = len(tuple(getattr(env_cfg, "koch_arm_joint_names", ()))) + len(
        tuple(getattr(env_cfg, "koch_gripper_joint_names", ()))
    )
    mobile_master_arm_dim = fixed_master_arm_dim + len(tuple(getattr(env_cfg, "koch_base_wheel_joint_names", ())))

    if action_dim in (fixed_keyboard_dim, mobile_keyboard_dim):
        _configure_mobile_base(env_cfg, enabled=(action_dim == mobile_keyboard_dim))
        _configure_position_only_keyboard(env_cfg)
        print(f"[Koch Mimic] Annotation replay action space: position-only IK ({action_dim}D).")
        return

    if action_dim in (fixed_master_arm_dim, mobile_master_arm_dim):
        _configure_mobile_base(env_cfg, enabled=(action_dim == mobile_master_arm_dim))
        _configure_direct_master_arm(env_cfg)
        print(f"[Koch Mimic] Annotation replay action space: direct master-arm joints ({action_dim}D).")
        return

    expected_dims = sorted({fixed_keyboard_dim, mobile_keyboard_dim, fixed_master_arm_dim, mobile_master_arm_dim})
    raise ValueError(
        f"Unsupported Koch demo action dimension {action_dim} in {input_file}. "
        f"Expected one of {expected_dims}."
    )


def _read_first_demo_action_dim(input_file: str | None) -> int | None:
    if not input_file:
        return None

    import h5py

    with h5py.File(input_file, "r") as dataset:
        data_group = dataset.get("data")
        if data_group is None:
            return None
        for demo_name in sorted(data_group.keys()):
            actions = data_group[demo_name].get("actions")
            if actions is not None and len(actions.shape) >= 2:
                return int(actions.shape[-1])
    return None


def _configure_mobile_base(env_cfg, enabled: bool) -> None:
    from isaaclab.envs.mdp.actions.actions_cfg import JointVelocityActionCfg

    env_cfg.teleop_fixed_base = not enabled
    env_cfg.robot_fix_root_link = not enabled
    robot_spawn = getattr(getattr(env_cfg.scene, "robot", None), "spawn", None)
    articulation_props = getattr(robot_spawn, "articulation_props", None)
    if articulation_props is not None:
        articulation_props.fix_root_link = not enabled

    if not enabled:
        env_cfg.actions.base_action = None
        return

    wheel_joint_names = tuple(getattr(env_cfg, "koch_base_wheel_joint_names", ()))
    if len(wheel_joint_names) != 4:
        raise ValueError(f"Mobile-base annotation replay requires four wheel joints, got: {wheel_joint_names}")
    env_cfg.actions.base_action = JointVelocityActionCfg(
        asset_name="robot",
        joint_names=list(wheel_joint_names),
        scale=1.0,
        offset=0.0,
        preserve_order=True,
        use_default_offset=False,
    )


def _configure_position_only_keyboard(env_cfg) -> None:
    from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
    from isaaclab.envs.mdp.actions.actions_cfg import RelativeJointPositionActionCfg

    from koch_mimic.cloud.tasks.koch_pick_place.latched_differential_ik_action import (
        LatchedDifferentialInverseKinematicsActionCfg,
    )

    env_cfg.actions.arm_action = LatchedDifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=list(env_cfg.koch_ik_joint_names),
        body_name=env_cfg.koch_ee_body_name,
        controller=DifferentialIKControllerCfg(
            command_type="position",
            use_relative_mode=True,
            ik_method="pinv",
            ik_params={"k_val": env_cfg.arm_ik_k_val},
        ),
        scale=env_cfg.arm_action_scale,
        zero_action_tolerance=1e-6,
        body_offset=LatchedDifferentialInverseKinematicsActionCfg.OffsetCfg(
            pos=list(env_cfg.koch_ee_offset),
            rot=env_cfg.koch_ee_offset_rot,
        ),
    )
    env_cfg.actions.wrist_action = RelativeJointPositionActionCfg(
        asset_name="robot",
        joint_names=list(env_cfg.koch_wrist_joint_names),
        scale=env_cfg.wrist_action_scale,
    )


def _configure_direct_master_arm(env_cfg) -> None:
    from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg

    gripper_open_command = float(env_cfg.koch_gripper_open_command)
    gripper_close_command = float(env_cfg.koch_gripper_close_command)

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
