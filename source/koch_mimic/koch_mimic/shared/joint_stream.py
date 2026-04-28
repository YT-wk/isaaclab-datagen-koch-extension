"""Shared JSONL schema helpers for the leader-arm joint stream."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
import time
from typing import Any, Mapping

from .constants import DEFAULT_LEADER_JOINT_ORDER, JOINT_STREAM_SCHEMA_VERSION


@dataclass(frozen=True)
class JointSample:
    """Normalized single-joint sample used by the local leader-arm pipeline."""

    motor_id: int
    ticks: int
    degrees: float
    radians: float


def jsonl_dumps(message: Mapping[str, Any]) -> str:
    """Serialize a message as one JSONL line."""
    return json.dumps(dict(message), ensure_ascii=False, separators=(",", ":")) + "\n"


def parse_jsonl_message(line: bytes | str) -> dict[str, Any]:
    """Parse one JSONL line into a mapping."""
    payload = line.decode("utf-8") if isinstance(line, bytes) else line
    data = json.loads(payload)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object, got {type(data).__name__}")
    return data


def build_session_start_message(
    session_id: str,
    serial_port: str,
    baudrate: int,
    *,
    joint_order: tuple[str, ...] = DEFAULT_LEADER_JOINT_ORDER,
) -> dict[str, Any]:
    """Create a session-start message for the cloud receiver."""
    return {
        "type": "session_start",
        "schema_version": JOINT_STREAM_SCHEMA_VERSION,
        "session_id": session_id,
        "timestamp": datetime.now().astimezone().isoformat(timespec="milliseconds"),
        "serial_port": serial_port,
        "baudrate": baudrate,
        "joint_order": list(joint_order),
    }


def build_joint_frame_message(
    session_id: str,
    sequence: int,
    serial_port: str,
    baudrate: int,
    joint_positions: Mapping[str, JointSample],
    *,
    joint_order: tuple[str, ...] = DEFAULT_LEADER_JOINT_ORDER,
) -> dict[str, Any]:
    """Create one streaming joint frame."""
    ordered_samples = [joint_positions[name] for name in joint_order]
    return {
        "type": "joint_frame",
        "schema_version": JOINT_STREAM_SCHEMA_VERSION,
        "session_id": session_id,
        "sequence": sequence,
        "timestamp": datetime.now().astimezone().isoformat(timespec="milliseconds"),
        "timestamp_ns": time.time_ns(),
        "serial_port": serial_port,
        "baudrate": baudrate,
        "joint_order": list(joint_order),
        "joint_position_ticks": [sample.ticks for sample in ordered_samples],
        "joint_position_deg": [round(sample.degrees, 6) for sample in ordered_samples],
        "joint_position_rad": [round(sample.radians, 6) for sample in ordered_samples],
    }


def joint_map_from_frame(
    message: Mapping[str, Any],
    *,
    expected_joint_order: tuple[str, ...] = DEFAULT_LEADER_JOINT_ORDER,
) -> dict[str, float]:
    """Extract `joint_name -> radian` from a joint-frame message."""
    joint_order = message.get("joint_order")
    joint_position_rad = message.get("joint_position_rad")
    if not isinstance(joint_order, list) or not isinstance(joint_position_rad, list):
        raise ValueError("Joint frame is missing `joint_order` or `joint_position_rad`.")
    if len(joint_order) != len(joint_position_rad):
        raise ValueError("Joint frame has mismatched `joint_order` and `joint_position_rad` lengths.")
    joint_map = {str(name): float(value) for name, value in zip(joint_order, joint_position_rad, strict=False)}
    missing = [name for name in expected_joint_order if name not in joint_map]
    if missing:
        raise ValueError(f"Joint frame is missing expected joints: {missing}")
    return joint_map
