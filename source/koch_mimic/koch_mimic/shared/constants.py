"""Package-wide constants."""

from __future__ import annotations

CLOUD_PROFILE = "cloud"
LOCAL_PROFILE = "local"

DEFAULT_TASK_ID = "Isaac-Koch-Mimic-PickPlace-v0"
LEGACY_TASK_ID = "Isaac-Koch-Pick-Place-IK-Rel-Mimic-v0"

JOINT_STREAM_SCHEMA_VERSION = 1
DEFAULT_LEADER_JOINT_ORDER = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)

CONFIG_FILENAMES = {
    CLOUD_PROFILE: {
        "defaults": "cloud.defaults.yaml",
        "example": "cloud.user.example.yaml",
        "user_local": "cloud.user.local.yaml",
    },
    LOCAL_PROFILE: {
        "defaults": "local.defaults.yaml",
        "example": "local.user.example.yaml",
        "user_local": "local.user.local.yaml",
    },
}

