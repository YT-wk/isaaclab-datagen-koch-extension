from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction
from isaaclab.utils import configclass


class LatchedDifferentialInverseKinematicsAction(DifferentialInverseKinematicsAction):
    """零输入时保持上一帧末端目标的微分 IK 动作项。

    Isaac Lab 默认的相对式 IK 会把“零命令”解释成“把当前末端位姿设为新目标”。
    在遥操作场景里，这会让机械臂在重力下发生的微小下沉被不断吸收到目标里，
    视觉上就像末端在无人控制时缓缓往下掉。

    这个本地版本在处理后的动作接近零时，会继续沿用上一帧的期望末端位姿；
    只有收到非零命令时，才按原本的相对式 IK 逻辑刷新目标。
    """

    cfg: "LatchedDifferentialInverseKinematicsActionCfg"

    def __init__(self, cfg: "LatchedDifferentialInverseKinematicsActionCfg", env):
        super().__init__(cfg, env)
        self._latched_ee_pos_des = torch.zeros_like(self._ik_controller.ee_pos_des)
        self._latched_ee_quat_des = torch.zeros_like(self._ik_controller.ee_quat_des)
        self._latched_target_initialized = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def process_actions(self, actions: torch.Tensor):
        # 预处理逻辑保持与 Isaac Lab 默认实现一致，方便后续继续对齐上游改动。
        self._raw_actions[:] = actions
        self._processed_actions[:] = self.raw_actions * self._scale
        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions, min=self._clip[:, :, 0], max=self._clip[:, :, 1]
            )

        ee_pos_curr, ee_quat_curr = self._compute_frame_pose()

        # reset 之后的第一帧如果没有输入，应当锁定“当前末端姿态”，而不是沿用上一回合残留目标。
        init_mask = ~self._latched_target_initialized
        if torch.any(init_mask):
            self._latched_ee_pos_des[init_mask] = ee_pos_curr[init_mask]
            self._latched_ee_quat_des[init_mask] = ee_quat_curr[init_mask]
            self._latched_target_initialized[init_mask] = True

        # 先按标准相对式 IK 算一遍目标，再把零输入环境恢复到“锁定的上一目标”。
        self._ik_controller.set_command(self._processed_actions, ee_pos_curr, ee_quat_curr)

        zero_action_mask = torch.all(torch.abs(self._processed_actions) <= self.cfg.zero_action_tolerance, dim=1)
        if torch.any(zero_action_mask):
            self._ik_controller.ee_pos_des[zero_action_mask] = self._latched_ee_pos_des[zero_action_mask]
            self._ik_controller.ee_quat_des[zero_action_mask] = self._latched_ee_quat_des[zero_action_mask]

        update_mask = ~zero_action_mask
        if torch.any(update_mask):
            self._latched_ee_pos_des[update_mask] = self._ik_controller.ee_pos_des[update_mask]
            self._latched_ee_quat_des[update_mask] = self._ik_controller.ee_quat_des[update_mask]

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        env_ids = slice(None) if env_ids is None else env_ids
        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = 0.0
        self._latched_target_initialized[env_ids] = False


@configclass
class LatchedDifferentialInverseKinematicsActionCfg(DifferentialInverseKinematicsActionCfg):
    """本地锁定式微分 IK 动作项配置。"""

    class_type: type = LatchedDifferentialInverseKinematicsAction
    zero_action_tolerance: float = 1e-6
