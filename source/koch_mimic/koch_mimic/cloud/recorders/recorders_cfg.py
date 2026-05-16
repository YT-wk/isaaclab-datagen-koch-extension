"""Recorder configs for Koch Mimic cloud demo datasets."""

from __future__ import annotations

from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.managers.manager_term_cfg import RecorderTermCfg
from isaaclab.utils import configclass

from .recorders import PreStepRGBCameraObservationsRecorder


@configclass
class PreStepRGBCameraObservationsRecorderCfg(RecorderTermCfg):
    """Configuration for recording the `rgb_camera` observation group."""

    class_type = PreStepRGBCameraObservationsRecorder


@configclass
class KochActionStateRecorderManagerCfg(ActionStateRecorderManagerCfg):
    """Isaac Lab action/state recorder extended with RGB camera observations."""

    record_pre_step_rgb_camera_observations = PreStepRGBCameraObservationsRecorderCfg()
