"""Koch pick-place task package and gym registration."""

from __future__ import annotations

import gymnasium as gym

from koch_mimic.shared.constants import DEFAULT_TASK_ID, LEGACY_TASK_ID

from .env_cfg import KochPickPlaceEnvCfg
from .mimic_env import KochPickPlaceMimicEnv
from .mimic_env_cfg import KochPickPlaceDataGenEnvCfg, KochPickPlaceSkillGenEnvCfg


def _register_task(task_id: str, env_cfg_entry_point: str) -> None:
    if task_id in gym.registry:
        return
    gym.register(
        id=task_id,
        entry_point=f"{__name__}.mimic_env:KochPickPlaceMimicEnv",
        kwargs={"env_cfg_entry_point": env_cfg_entry_point},
        disable_env_checker=True,
    )


_register_task(DEFAULT_TASK_ID, f"{__name__}.mimic_env_cfg:KochPickPlaceDataGenEnvCfg")
_register_task(LEGACY_TASK_ID, f"{__name__}.mimic_env_cfg:KochPickPlaceDataGenEnvCfg")

__all__ = [
    "KochPickPlaceEnvCfg",
    "KochPickPlaceDataGenEnvCfg",
    "KochPickPlaceSkillGenEnvCfg",
    "KochPickPlaceMimicEnv",
]
