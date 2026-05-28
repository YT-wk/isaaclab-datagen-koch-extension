# Copyright (c) 2026, Custom project example
# SPDX-License-Identifier: Apache-2.0

"""自定义 Koch 任务的 Mimic/SkillGen 配置。

这些配置会在 ``KochPickPlaceEnvCfg`` 的物理场景与任务定义之上，
补充 Isaac Lab Mimic 所需的标注和数据生成参数。这里的子任务定义决定了
示教轨迹如何被切分成“抓取”和“放置”两个阶段。
"""

from isaaclab.envs.mimic_env_cfg import MimicEnvCfg, SubTaskConfig
from isaaclab.utils import configclass

from koch_mimic.shared.configuration import get_active_runtime_config, get_config_section
from koch_mimic.shared.constants import CLOUD_PROFILE

from .env_cfg import KochPickPlaceEnvCfg
from .latched_differential_ik_action import PositionPriorityDifferentialInverseKinematicsActionCfg


@configclass
class KochPickPlaceDataGenEnvCfg(KochPickPlaceEnvCfg, MimicEnvCfg):
    """Koch 抓放任务的 Mimic 数据生成配置。"""

    def __post_init__(self):
        """补齐 Mimic 专用的数据生成与子任务配置。"""
        super().__post_init__()

        # 下面这些 datagen 参数决定 Mimic 如何搜索可复用的源片段，
        # 以及失败轨迹是否仍然保留下来供排查分析。
        runtime_config = get_active_runtime_config(CLOUD_PROFILE, require_user_local=False)
        self._configure_generation_action_space()
        self.datagen_config.name = str(
            get_config_section(runtime_config, "mimic", "datagen_name", default="koch_pick_place_mimic")
        )
        self.datagen_config.generation_guarantee = bool(
            get_config_section(runtime_config, "mimic", "generation_guarantee", default=True)
        )
        self.datagen_config.generation_keep_failed = bool(
            get_config_section(runtime_config, "mimic", "generation_keep_failed", default=True)
        )
        self.datagen_config.generation_num_trials = int(
            get_config_section(runtime_config, "mimic", "generation_num_trials", default=200)
        )
        self.datagen_config.generation_select_src_per_subtask = bool(
            get_config_section(runtime_config, "mimic", "generation_select_src_per_subtask", default=True)
        )
        self.datagen_config.generation_transform_first_robot_pose = bool(
            get_config_section(runtime_config, "mimic", "generation_transform_first_robot_pose", default=False)
        )
        self.datagen_config.generation_interpolate_from_last_target_pose = bool(
            get_config_section(runtime_config, "mimic", "generation_interpolate_from_last_target_pose", default=True)
        )
        self.datagen_config.generation_relative = bool(
            get_config_section(runtime_config, "mimic", "generation_relative", default=True)
        )
        self.datagen_config.max_num_failures = int(
            get_config_section(runtime_config, "mimic", "max_num_failures", default=50)
        )
        self.datagen_config.seed = int(get_config_section(runtime_config, "mimic", "seed", default=1))

        action_noise = float(get_config_section(runtime_config, "mimic", "action_noise", default=0.02))
        interpolation_steps = int(get_config_section(runtime_config, "mimic", "interpolation_steps", default=5))
        fixed_steps = int(get_config_section(runtime_config, "mimic", "fixed_steps", default=0))
        apply_noise_during_interpolation = bool(
            get_config_section(runtime_config, "mimic", "apply_noise_during_interpolation", default=False)
        )
        selection_nn_k = int(get_config_section(runtime_config, "mimic", "selection_nn_k", default=3))

        # 当前任务被显式拆成两个阶段：
        # 1. 抓取物体 A。
        # 2. 将物体 A 放到物体 B 上方。
        subtask_configs = [
            SubTaskConfig(
                object_ref="cube_1",
                subtask_term_signal="grasp_obj_a",
                subtask_term_offset_range=(0, 0),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": selection_nn_k},
                action_noise=action_noise,
                num_interpolation_steps=interpolation_steps,
                num_fixed_steps=fixed_steps,
                apply_noise_during_interpolation=apply_noise_during_interpolation,
                description="Grasp object A",
                next_subtask_description="Place object A on object B",
            ),
            SubTaskConfig(
                object_ref="cube_2",
                subtask_term_signal="place_obj_a_on_b",
                subtask_term_offset_range=(0, 0),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": selection_nn_k},
                action_noise=action_noise,
                num_interpolation_steps=interpolation_steps,
                num_fixed_steps=fixed_steps,
                apply_noise_during_interpolation=apply_noise_during_interpolation,
            ),
        ]
        self.subtask_configs["koch"] = subtask_configs

    def _configure_generation_action_space(self) -> None:
        mode = str(getattr(self, "mimic_generation_control_mode", "position_only"))
        if mode != "position_priority_pose_ik":
            return

        pose_joint_names = tuple(getattr(self, "mimic_pose_ik_joint_names", ())) or tuple(
            getattr(self, "koch_arm_joint_names", ())
        )
        self.actions.arm_action = PositionPriorityDifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=list(pose_joint_names),
            body_name=self.koch_ee_body_name,
            controller=self.actions.arm_action.controller.replace(command_type="pose", use_relative_mode=True),
            scale=self._generation_pose_action_scale(),
            zero_action_tolerance=getattr(self.actions.arm_action, "zero_action_tolerance", 1e-6),
            body_offset=PositionPriorityDifferentialInverseKinematicsActionCfg.OffsetCfg(
                pos=list(self.koch_ee_offset),
                rot=self.koch_ee_offset_rot,
            ),
            orientation_weight=float(getattr(self, "mimic_orientation_weight", 0.25)),
            orientation_max_step_rad=float(getattr(self, "mimic_orientation_max_step_rad", 0.15)),
            ik_damping=float(getattr(self, "mimic_ik_damping", 0.01)),
        )
        self.actions.wrist_action = None
        print(
            "[Koch Mimic] Generation action space: position-priority pose IK "
            f"(7D, joints={list(pose_joint_names)}, orientation_weight={self.mimic_orientation_weight})."
        )

    def _generation_pose_action_scale(self):
        scale = getattr(self, "arm_action_scale", 1.0)
        if isinstance(scale, (list, tuple)):
            if len(scale) == 6:
                return scale
            if len(scale) == 3:
                return tuple(scale) + (1.0, 1.0, 1.0)
        return (float(scale), float(scale), float(scale), 1.0, 1.0, 1.0)


@configclass
class KochPickPlaceSkillGenEnvCfg(KochPickPlaceDataGenEnvCfg):
    """Koch Mimic 环境的 SkillGen 变体配置。"""

    def __post_init__(self):
        """在基础 Mimic 配置上调整为 SkillGen 风格的回放参数。"""
        super().__post_init__()

        runtime_config = get_active_runtime_config(CLOUD_PROFILE, require_user_local=False)
        self.datagen_config.name = str(
            get_config_section(runtime_config, "mimic", "skillgen_name", default="koch_pick_place_skillgen")
        )
        self.datagen_config.use_skillgen = True
        self.datagen_config.generation_num_trials = int(
            get_config_section(runtime_config, "mimic", "skillgen_num_trials", default=100)
        )

        # SkillGen 更常直接在子任务切换状态之间跳转，
        # 而不是依赖 Mimic 原始轨迹生成的稠密插值。
        interpolation_steps = int(get_config_section(runtime_config, "mimic", "skillgen_interpolation_steps", default=0))
        for subtask_cfg in self.subtask_configs["koch"]:
            subtask_cfg.num_interpolation_steps = interpolation_steps
