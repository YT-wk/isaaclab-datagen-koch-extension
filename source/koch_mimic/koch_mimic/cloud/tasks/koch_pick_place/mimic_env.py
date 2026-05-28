# Copyright (c) 2026, Custom project example
# SPDX-License-Identifier: Apache-2.0

"""自定义 Koch 抓放任务的 Mimic 环境封装。

``env_cfg.py`` 中的基础环境负责定义物理场景以及面向 MDP 的观测和动作。
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


class KochPickPlaceMimicEnv(ManagerBasedRLMimicEnv):
    """自定义 Koch 抓放环境对应的 Mimic 包装器。

    当前实现默认只有一个受控末端执行器，以及一个夹爪动作项。
    如果之后扩展成双臂任务，这个类里关于“单末端执行器”的假设会是最先需要泛化的地方。
    """

    def _get_eef_name(self) -> str:
        """返回 Mimic 子任务配置中使用的逻辑末端执行器名称。"""
        return list(self.cfg.subtask_configs.keys())[0]

    def _base_action_dim(self) -> int:
        """Return the prefix dimension used by optional mobile-base actions."""
        base_action_cfg = getattr(self.cfg.actions, "base_action", None)
        if base_action_cfg is None:
            return 0
        return len(getattr(base_action_cfg, "joint_names", ()))

    def _arm_controller_cfg(self):
        """Return the IK controller config, or None for direct joint-control replay."""
        arm_action_cfg = getattr(self.cfg.actions, "arm_action", None)
        return getattr(arm_action_cfg, "controller", None)

    def _arm_uses_ik_action(self) -> bool:
        """Return whether the current arm action term is an IK action term."""
        return self._arm_controller_cfg() is not None

    def _arm_uses_position_only_ik(self) -> bool:
        """判断当前环境是否启用了仅位置控制的 IK。"""
        controller_cfg = self._arm_controller_cfg()
        return controller_cfg is not None and controller_cfg.command_type == "position"

    def _arm_action_dim(self) -> int:
        """返回当前 IK 动作项占用的动作维度。"""
        controller_cfg = self._arm_controller_cfg()
        if controller_cfg is None:
            return len(getattr(self.cfg, "koch_arm_joint_names", ()))
        if controller_cfg.command_type == "position":
            return 3
        if controller_cfg.command_type == "pose" and controller_cfg.use_relative_mode:
            return 6
        return 7

    def _arm_action_scale_tensor(self, action: torch.Tensor) -> torch.Tensor:
        """Return the configured IK action scale as a tensor broadcastable to ``action``."""
        arm_action_cfg = getattr(self.cfg.actions, "arm_action", None)
        scale = getattr(arm_action_cfg, "scale", 1.0)
        scale_tensor = torch.as_tensor(scale, dtype=action.dtype, device=action.device).flatten()
        if scale_tensor.numel() == 0:
            raise ValueError("arm_action.scale must contain at least one value.")
        if scale_tensor.numel() == 1:
            scale_tensor = scale_tensor.repeat(action.shape[-1])
        if scale_tensor.numel() != action.shape[-1]:
            raise ValueError(
                f"arm_action.scale has {scale_tensor.numel()} values, but the IK action has "
                f"{action.shape[-1]} dimensions."
            )
        if torch.any(torch.isclose(scale_tensor, torch.zeros_like(scale_tensor))):
            raise ValueError("arm_action.scale must be non-zero for Mimic target/action conversion.")
        return scale_tensor

    def _raw_arm_action_from_processed(self, processed_action: torch.Tensor) -> torch.Tensor:
        """Convert an IK controller-space delta back to the raw action expected by env.step()."""
        return processed_action / self._arm_action_scale_tensor(processed_action)

    def _processed_arm_action_from_raw(self, raw_action: torch.Tensor) -> torch.Tensor:
        """Convert a raw IK action into the actual controller-space delta applied by the action term."""
        return raw_action * self._arm_action_scale_tensor(raw_action)

    def _has_manual_wrist_action(self) -> bool:
        """判断当前环境是否额外暴露了手动 wrist 关节动作。"""
        return hasattr(self.cfg.actions, "wrist_action") and self.cfg.actions.wrist_action is not None

    def _wrist_action_scale_tensor(self, action: torch.Tensor) -> torch.Tensor:
        """Return the configured wrist action scale as a tensor broadcastable to ``action``."""
        wrist_action_cfg = getattr(self.cfg.actions, "wrist_action", None)
        scale = getattr(wrist_action_cfg, "scale", 1.0)
        scale_tensor = torch.as_tensor(scale, dtype=action.dtype, device=action.device).flatten()
        if scale_tensor.numel() == 0:
            raise ValueError("wrist_action.scale must contain at least one value.")
        if scale_tensor.numel() == 1:
            scale_tensor = scale_tensor.repeat(action.shape[-1])
        if scale_tensor.numel() != action.shape[-1]:
            raise ValueError(
                f"wrist_action.scale has {scale_tensor.numel()} values, but the wrist action has "
                f"{action.shape[-1]} dimensions."
            )
        if torch.any(torch.isclose(scale_tensor, torch.zeros_like(scale_tensor))):
            raise ValueError("wrist_action.scale must be non-zero for Mimic wrist conversion.")
        return scale_tensor

    def _raw_wrist_action_from_processed(self, processed_action: torch.Tensor) -> torch.Tensor:
        """Convert a controller-space wrist delta to the raw relative joint action."""
        return processed_action / self._wrist_action_scale_tensor(processed_action)

    def _current_wrist_joint_pos(self, env_id: int, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        """Return the current manual wrist joint position for one environment."""
        robot = self.scene["robot"]
        wrist_joint_ids, _ = robot.find_joints(self.cfg.koch_wrist_joint_names)
        if not wrist_joint_ids:
            raise ValueError(f"No wrist joints found for names: {self.cfg.koch_wrist_joint_names}")
        return robot.data.joint_pos[env_id, wrist_joint_ids].to(dtype=dtype, device=device)

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
        gripper_action = gripper_action.flatten()
        wrist_processed_action = None
        if self._has_manual_wrist_action():
            wrist_action_dim = len(getattr(self.cfg, "koch_wrist_joint_names", ()))
            if gripper_action.numel() >= wrist_action_dim + 2:
                wrist_value = gripper_action[:wrist_action_dim]
                wrist_is_absolute = bool(gripper_action[wrist_action_dim].item() > 0.5)
                gripper_action = gripper_action[-1:]
                if wrist_is_absolute:
                    wrist_bias = torch.full_like(
                        wrist_value,
                        float(getattr(self.cfg, "mimic_wrist_target_bias_rad", 0.0)),
                    )
                    wrist_value = wrist_value + wrist_bias
                    wrist_current = self._current_wrist_joint_pos(
                        env_id,
                        dtype=wrist_value.dtype,
                        device=wrist_value.device,
                    )
                    wrist_processed_action = wrist_value - wrist_current
                else:
                    wrist_processed_action = wrist_value
            elif gripper_action.numel() > 1:
                wrist_processed_action = gripper_action[:-1]
                gripper_action = gripper_action[-1:]
        elif gripper_action.numel() > 1:
            gripper_action = gripper_action[-1:]

        if self._arm_uses_position_only_ik():
            processed_arm_action = delta_position
            # position-only 调试版只对 XYZ 做相对控制，忽略目标姿态。
            processed_arm_action = delta_position
        else:
            delta_rot_mat = target_rot.matmul(curr_rot.transpose(-1, -2))
            delta_quat = PoseUtils.quat_from_matrix(delta_rot_mat)
            delta_rotation = PoseUtils.axis_angle_from_quat(delta_quat)
            processed_arm_action = torch.cat([delta_position, delta_rotation], dim=0)

        if action_noise_dict is not None:
            noise = action_noise_dict[eef_name] * torch.randn_like(processed_arm_action)
            processed_arm_action = processed_arm_action + noise

        # ``LatchedDifferentialInverseKinematicsAction`` multiplies raw actions by
        # ``arm_action.scale`` before sending them to the IK controller. Mimic target
        # poses are expressed in controller-space deltas, especially after annotating
        # direct joint-control master-arm demos, so invert that scale here.
        arm_action = self._raw_arm_action_from_processed(processed_arm_action)
        gripper_action = gripper_action.to(dtype=arm_action.dtype, device=arm_action.device)

        if self._has_manual_wrist_action():
            if wrist_processed_action is None:
                wrist_action = torch.zeros(
                    len(getattr(self.cfg, "koch_wrist_joint_names", ())),
                    dtype=arm_action.dtype,
                    device=arm_action.device,
                )
            else:
                wrist_action = self._raw_wrist_action_from_processed(
                    wrist_processed_action.to(dtype=arm_action.dtype, device=arm_action.device)
                )
            play_action = torch.cat([arm_action, wrist_action, gripper_action], dim=0)
        else:
            play_action = torch.cat([arm_action, gripper_action], dim=0)

        base_action_dim = self._base_action_dim()
        if base_action_dim > 0:
            base_action = torch.zeros(base_action_dim, dtype=play_action.dtype, device=play_action.device)
            play_action = torch.cat([base_action, play_action], dim=0)
        return play_action

    def action_to_target_eef_pose(self, action: torch.Tensor) -> dict[str, torch.Tensor]:
        """把相对 IK 动作反解回目标末端执行器位姿。"""
        eef_name = self._get_eef_name()

        curr_pose = self.get_robot_eef_pose(eef_name=eef_name, env_ids=None)
        if not self._arm_uses_ik_action():
            # Direct joint-target teleop demos do not encode a Cartesian controller target.
            # During annotation, use the replayed end-effector pose trajectory itself as the Mimic target path.
            return {eef_name: curr_pose.clone()}

        curr_pos, curr_rot = PoseUtils.unmake_pose(curr_pose)

        arm_action_dim = self._arm_action_dim()
        base_action_dim = self._base_action_dim()
        raw_arm_action = action[:, base_action_dim : base_action_dim + arm_action_dim]
        processed_arm_action = self._processed_arm_action_from_raw(raw_arm_action)
        delta_position = processed_arm_action[:, :3]
        target_pos = curr_pos + delta_position

        if self._arm_uses_position_only_ik():
            target_rot = curr_rot
            # position-only 控制保持当前姿态不变，只更新目标位置。
            target_rot = curr_rot
        else:
            delta_rotation = processed_arm_action[:, 3:arm_action_dim]

            # 旋转动作部分采用轴角增量表示。
            delta_rotation_angle = torch.linalg.norm(delta_rotation, dim=-1, keepdim=True)
            safe_delta_rotation_angle = torch.clamp(delta_rotation_angle, min=1.0e-9)
            delta_rotation_axis = delta_rotation / safe_delta_rotation_angle

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
        gripper_actions = actions[..., -1:]
        base_action_dim = self._base_action_dim()
        arm_action_dim = self._arm_action_dim()
        direct_joint_action_dim = len(getattr(self.cfg, "koch_arm_joint_names", ())) + len(
            getattr(self.cfg, "koch_gripper_joint_names", ())
        )
        mobile_direct_joint_action_dim = len(getattr(self.cfg, "koch_base_wheel_joint_names", ())) + direct_joint_action_dim

        if actions.shape[-1] in (direct_joint_action_dim, mobile_direct_joint_action_dim):
            direct_base_action_dim = actions.shape[-1] - direct_joint_action_dim
            wrist_action_start = direct_base_action_dim + len(getattr(self.cfg, "koch_ik_joint_names", ()))
            wrist_actions = actions[..., wrist_action_start:-1]
            open_command = float(self.cfg.koch_gripper_open_command)
            close_command = float(self.cfg.koch_gripper_close_command)
            threshold = 0.5 * (open_command + close_command)
            if open_command <= close_command:
                is_open = gripper_actions <= threshold
            else:
                is_open = gripper_actions >= threshold
            gripper_actions = torch.where(is_open, torch.ones_like(gripper_actions), -torch.ones_like(gripper_actions))
            if self._has_manual_wrist_action() and wrist_actions.shape[-1] > 0:
                wrist_is_absolute = torch.ones((*wrist_actions.shape[:-1], 1), dtype=actions.dtype, device=actions.device)
                gripper_actions = torch.cat([wrist_actions, wrist_is_absolute, gripper_actions], dim=-1)
        elif self._has_manual_wrist_action():
            wrist_start = base_action_dim + arm_action_dim
            wrist_end = actions.shape[-1] - 1
            if wrist_end > wrist_start:
                wrist_actions = actions[..., wrist_start:wrist_end]
                wrist_cfg = getattr(self.cfg.actions, "wrist_action", None)
                wrist_scale = torch.as_tensor(
                    getattr(wrist_cfg, "scale", 1.0),
                    dtype=actions.dtype,
                    device=actions.device,
                ).flatten()
                if wrist_scale.numel() == 1:
                    wrist_scale = wrist_scale.repeat(wrist_actions.shape[-1])
                wrist_processed_actions = wrist_actions * wrist_scale
                wrist_is_absolute = torch.zeros(
                    (*wrist_processed_actions.shape[:-1], 1), dtype=actions.dtype, device=actions.device
                )
                gripper_actions = torch.cat([wrist_processed_actions, wrist_is_absolute, gripper_actions], dim=-1)

        return {self._get_eef_name(): gripper_actions}

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
