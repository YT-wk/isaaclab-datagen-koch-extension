# Copyright (c) 2026, Custom project example
# SPDX-License-Identifier: Apache-2.0

"""自定义 Koch 任务的 Mimic/SkillGen 配置。

这些配置会在 ``MyCustomMimicEnvCfg`` 的物理场景与任务定义之上，
补充 Isaac Lab Mimic 所需的标注和数据生成参数。这里的子任务定义决定了
示教轨迹如何被切分成“抓取”和“放置”两个阶段。
"""

from isaaclab.envs.mimic_env_cfg import MimicEnvCfg, SubTaskConfig
from isaaclab.utils import configclass

from .my_env_cfg import MyCustomMimicEnvCfg


@configclass
class MyCustomMimicDataGenEnvCfg(MyCustomMimicEnvCfg, MimicEnvCfg):
    """Koch 抓放任务的 Mimic 数据生成配置。"""

    def __post_init__(self):
        """补齐 Mimic 专用的数据生成与子任务配置。"""
        super().__post_init__()

        # 下面这些 datagen 参数决定 Mimic 如何搜索可复用的源片段，
        # 以及失败轨迹是否仍然保留下来供排查分析。
        self.datagen_config.name = "koch_pick_place_mimic"
        self.datagen_config.generation_guarantee = True
        self.datagen_config.generation_keep_failed = True
        self.datagen_config.generation_num_trials = 200
        self.datagen_config.generation_select_src_per_subtask = True
        self.datagen_config.generation_transform_first_robot_pose = False
        self.datagen_config.generation_interpolate_from_last_target_pose = True
        self.datagen_config.generation_relative = True
        self.datagen_config.max_num_failures = 50
        self.datagen_config.seed = 1

        # 当前任务被显式拆成两个阶段：
        # 1. 抓取物体 A。
        # 2. 将物体 A 放到物体 B 上方。
        subtask_configs = [
            SubTaskConfig(
                object_ref="cube_1",
                subtask_term_signal="grasp_obj_a",
                subtask_term_offset_range=(0, 0),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.02,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="Grasp object A",
                next_subtask_description="Place object A on object B",
            ),
            SubTaskConfig(
                object_ref="cube_2",
                subtask_term_signal="place_obj_a_on_b",
                subtask_term_offset_range=(0, 0),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.02,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
            ),
        ]
        self.subtask_configs["koch"] = subtask_configs


@configclass
class MyCustomMimicSkillGenEnvCfg(MyCustomMimicDataGenEnvCfg):
    """Koch Mimic 环境的 SkillGen 变体配置。"""

    def __post_init__(self):
        """在基础 Mimic 配置上调整为 SkillGen 风格的回放参数。"""
        super().__post_init__()

        self.datagen_config.name = "koch_pick_place_skillgen"
        self.datagen_config.use_skillgen = True
        self.datagen_config.generation_num_trials = 100

        # SkillGen 更常直接在子任务切换状态之间跳转，
        # 而不是依赖 Mimic 原始轨迹生成的稠密插值。
        for subtask_cfg in self.subtask_configs["koch"]:
            subtask_cfg.num_interpolation_steps = 0
