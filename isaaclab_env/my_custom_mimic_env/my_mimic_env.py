# Copyright (c) 2026, Custom project example
# SPDX-License-Identifier: Apache-2.0

"""自定义 Koch 抓放任务的 Mimic 环境封装。

``my_env_cfg.py`` 中的基础环境负责定义物理场景以及面向 MDP 的观测和动作。
这个封装层额外告诉 Isaac Lab Mimic 应该如何：

1. 从观测缓冲区读取当前末端执行器位姿。
2. 将目标位姿转换成相对 IK 动作。
3. 为轨迹标注提供物体位姿与子任务完成信号。
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

import isaaclab.utils.math as PoseUtils
from isaaclab.envs import ManagerBasedRLMimicEnv


class MyCustomMimicEnv(ManagerBasedRLMimicEnv):
    """自定义 Koch 抓放环境对应的 Mimic 包装器。

    当前实现默认只有一个受控末端执行器，以及一个夹爪动作项。
    如果之后扩展成双臂任务，这个类里关于“单末端执行器”的假设会是最先需要泛化的地方。
    """

    def _get_eef_name(self) -> str:
        """返回 Mimic 子任务配置中使用的逻辑末端执行器名称。"""
        return list(self.cfg.subtask_configs.keys())[0]

    def _arm_uses_position_only_ik(self) -> bool:
        """判断当前环境是否启用了仅位置控制的 IK。"""
        return self.cfg.actions.arm_action.controller.command_type == "position"

    def _arm_action_dim(self) -> int:
        """返回当前 IK 动作项占用的动作维度。"""
        controller_cfg = self.cfg.actions.arm_action.controller
        if controller_cfg.command_type == "position":
            return 3
        if controller_cfg.command_type == "pose" and controller_cfg.use_relative_mode:
            return 6
        return 7

    def _has_manual_wrist_action(self) -> bool:
        """判断当前环境是否额外暴露了手动 wrist 关节动作。"""
        return hasattr(self.cfg.actions, "wrist_action") and self.cfg.actions.wrist_action is not None

    def get_robot_eef_pose(self, eef_name: str, env_ids: Sequence[int] | None = None) -> torch.Tensor:
        """从最新观测缓冲区读取 Mimic 所需的末端执行器位姿。

        返回位姿所使用的参考系与微分 IK 控制器保持一致。
        """
        if env_ids is None:
            env_ids = slice(None)

        # 底层环境已经把末端位姿以“位置 + 四元数”的形式写进了 policy 观测。
        eef_pos = self.obs_buf["policy"]["eef_pos"][env_ids]
        eef_quat = self.obs_buf["policy"]["eef_quat"][env_ids]  # wxyz
        return PoseUtils.make_pose(eef_pos, PoseUtils.matrix_from_quat(eef_quat))

    def target_eef_pose_to_action(
        self,
        target_eef_pose_dict: dict,
        gripper_action_dict: dict,
        action_noise_dict: dict | None = None,
        env_id: int = 0,
    ) -> torch.Tensor:
        """把 Mimic 给出的目标位姿转换成环境动作向量。

        当前环境默认使用相对 IK：
        - 如果是 position-only IK，则只取目标位置与当前位置的差值；
        - 如果是 pose IK，则同时计算位置和姿态的增量。
        """
        eef_name = self._get_eef_name()

        (target_eef_pose,) = target_eef_pose_dict.values()
        target_pos, target_rot = PoseUtils.unmake_pose(target_eef_pose)

        # Mimic 在“目标位姿 -> 动作”的转换过程中，一次只回放一个环境。
        curr_pose = self.get_robot_eef_pose(eef_name=eef_name, env_ids=[env_id])[0]
        curr_pos, curr_rot = PoseUtils.unmake_pose(curr_pose)

        delta_position = target_pos - curr_pos

        (gripper_action,) = gripper_action_dict.values()

        if self._arm_uses_position_only_ik():
            # position-only 调试版只对 XYZ 做相对控制，忽略目标姿态。
            arm_action = delta_position
        else:
            delta_rot_mat = target_rot.matmul(curr_rot.transpose(-1, -2))
            delta_quat = PoseUtils.quat_from_matrix(delta_rot_mat)
            delta_rotation = PoseUtils.axis_angle_from_quat(delta_quat)
            arm_action = torch.cat([delta_position, delta_rotation], dim=0)

        if action_noise_dict is not None:
            noise = action_noise_dict[eef_name] * torch.randn_like(arm_action)
            arm_action = torch.clamp(arm_action + noise, -1.0, 1.0)

        if self._has_manual_wrist_action():
            wrist_action = torch.zeros(1, dtype=arm_action.dtype, device=arm_action.device)
            return torch.cat([arm_action, wrist_action, gripper_action], dim=0)

        return torch.cat([arm_action, gripper_action], dim=0)

    def action_to_target_eef_pose(self, action: torch.Tensor) -> dict[str, torch.Tensor]:
        """把相对 IK 动作反解回目标末端执行器位姿。"""
        eef_name = self._get_eef_name()

        curr_pose = self.get_robot_eef_pose(eef_name=eef_name, env_ids=None)
        curr_pos, curr_rot = PoseUtils.unmake_pose(curr_pose)

        arm_action_dim = self._arm_action_dim()
        delta_position = action[:, :3]
        target_pos = curr_pos + delta_position

        if self._arm_uses_position_only_ik():
            # position-only 控制保持当前姿态不变，只更新目标位置。
            target_rot = curr_rot
        else:
            delta_rotation = action[:, 3:arm_action_dim]

            # 旋转动作部分采用轴角增量表示。
            delta_rotation_angle = torch.linalg.norm(delta_rotation, dim=-1, keepdim=True)
            delta_rotation_axis = delta_rotation / delta_rotation_angle

            near_zero = torch.isclose(delta_rotation_angle, torch.zeros_like(delta_rotation_angle)).squeeze(1)
            # 当旋转角接近 0 时，轴方向本身没有意义，这里显式置零以避免数值污染。
            delta_rotation_axis[near_zero] = torch.zeros_like(delta_rotation_axis)[near_zero]

            delta_quat = PoseUtils.quat_from_angle_axis(
                delta_rotation_angle.squeeze(1), delta_rotation_axis
            ).squeeze(0)
            delta_rot_mat = PoseUtils.matrix_from_quat(delta_quat)
            target_rot = torch.matmul(delta_rot_mat, curr_rot)

        return {eef_name: PoseUtils.make_pose(target_pos, target_rot).clone()}

    def actions_to_gripper_actions(self, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        """从环境动作张量末尾提取夹爪命令。"""
        return {self._get_eef_name(): actions[:, -1:]}

    def get_object_poses(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        """返回 Mimic/SkillGen 所需的全部非机器人对象位姿。

        这里读取相对坐标系下的场景状态，避免录制出的示教轨迹绑定到某个克隆环境的绝对原点。
        """
        if env_ids is None:
            env_ids = slice(None)

        scene_state = self.scene.get_state(is_relative=True)
        rigid_object_states = scene_state["rigid_object"]
        articulation_states = scene_state["articulation"]

        object_pose_matrix: dict[str, torch.Tensor] = {}

        for obj_name, obj_state in rigid_object_states.items():
            pos = obj_state["root_pose"][env_ids, :3]
            quat = obj_state["root_pose"][env_ids, 3:7]
            object_pose_matrix[obj_name] = PoseUtils.make_pose(pos, PoseUtils.matrix_from_quat(quat))

        for art_name, art_state in articulation_states.items():
            if art_name == "robot":
                continue
            pos = art_state["root_pose"][env_ids, :3]
            quat = art_state["root_pose"][env_ids, 3:7]
            object_pose_matrix[art_name] = PoseUtils.make_pose(pos, PoseUtils.matrix_from_quat(quat))

        return object_pose_matrix

    def get_subtask_term_signals(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        """向 Mimic 暴露任务专用的二值子任务完成信号。"""
        if env_ids is None:
            env_ids = slice(None)

        subtask_terms = self.obs_buf["subtask_terms"]
        return {
            "grasp_obj_a": subtask_terms["grasp_obj_a"][env_ids],
            "place_obj_a_on_b": subtask_terms["place_obj_a_on_b"][env_ids],
        }

    def get_expected_attached_object(self, eef_name: str, subtask_index: int, env_cfg) -> str | None:
        """SkillGen 使用的可选辅助接口。

        第二个子任务是放置阶段，因此这里默认认为第一阶段抓到的物体会在子任务切换点
        继续附着在夹爪上。
        """
        if eef_name not in env_cfg.subtask_configs:
            return None
        if subtask_index == 1:
            return env_cfg.subtask_configs[eef_name][0].object_ref
        return None
