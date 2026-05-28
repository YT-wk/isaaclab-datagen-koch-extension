"""Project-local recorder terms extending Isaac Lab demo capture."""

from __future__ import annotations

from isaaclab.managers.recorder_manager import RecorderTerm


class PreStepRGBCameraObservationsRecorder(RecorderTerm):
    """Record the `rgb_camera` observation group before each action is applied.

    The base Isaac Lab action/state recorder only stores ``obs_buf["policy"]``. For this project we also
    want camera frames from teleoperation and Mimic generation so the HDF5 dataset can be used for both
    low-dimensional replay and visuomotor training.
    """

    def record_pre_step(self):
        if not hasattr(self._env, "obs_buf"):
            return None, None
        if "rgb_camera" not in self._env.obs_buf:
            return None, None
        return "obs/rgb_camera", self._env.obs_buf["rgb_camera"]
