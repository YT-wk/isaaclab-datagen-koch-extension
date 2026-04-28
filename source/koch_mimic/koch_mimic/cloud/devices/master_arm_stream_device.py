"""TCP-backed teleop device for a streamed Koch master arm."""

from __future__ import annotations

import socket
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import torch

from isaaclab.devices.device_base import DeviceBase, DeviceCfg

from koch_mimic.shared.constants import DEFAULT_LEADER_JOINT_ORDER
from koch_mimic.shared.joint_stream import joint_map_from_frame, parse_jsonl_message


@dataclass
class KochMasterArmStreamDeviceCfg(DeviceCfg):
    """Configuration for the streamed Koch master-arm bridge."""

    host: str = "127.0.0.1"
    port: int = 55000
    expected_joint_order: tuple[str, ...] = DEFAULT_LEADER_JOINT_ORDER
    joint_signs: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0, 1.0)
    joint_offsets: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0)
    zero_on_first_frame: bool = True
    gripper_open_command: float = 1.0
    gripper_close_command: float = 0.0
    gripper_close_delta: float = 1.0
    gripper_close_direction: str = "negative"
    stale_timeout: float = 1.0
    socket_accept_timeout: float = 0.5
    verbose: bool = True
    retargeters: list = field(default_factory=list)
    class_type: type[DeviceBase] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.gripper_close_delta <= 0.0:
            raise ValueError(f"gripper_close_delta must be positive, got {self.gripper_close_delta}")
        if self.class_type is None:
            self.class_type = KochMasterArmStreamDevice


class KochMasterArmStreamDevice(DeviceBase):
    """Convert streamed leader joint frames into Isaac Lab actions.

    The returned action vector is:
    ``[arm_j1, arm_j2, arm_j3, arm_j4, arm_j5, gripper_position_target]``.
    The first five entries are absolute arm joint position targets in radians.
    The last entry is a continuous gripper joint target in radians.
    """

    def __init__(self, cfg: KochMasterArmStreamDeviceCfg):
        super().__init__(retargeters=None)
        self._cfg = cfg
        self._sim_device = cfg.sim_device
        self._callbacks: dict[str, Any] = {}
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._server_socket: socket.socket | None = None
        self._client_socket: socket.socket | None = None
        self._thread = threading.Thread(target=self._server_loop, daemon=True)

        self._expected_arm_joint_order = tuple(cfg.expected_joint_order[:5])
        self._expected_gripper_joint = cfg.expected_joint_order[5]
        self._joint_signs = torch.tensor(cfg.joint_signs, dtype=torch.float32)
        self._joint_offsets = torch.tensor(cfg.joint_offsets, dtype=torch.float32)
        self._gripper_open_command = float(cfg.gripper_open_command)
        self._gripper_close_command = float(cfg.gripper_close_command)
        self._gripper_close_delta = float(cfg.gripper_close_delta)
        self._gripper_command_min = min(self._gripper_open_command, self._gripper_close_command)
        self._gripper_command_max = max(self._gripper_open_command, self._gripper_close_command)

        self._home_arm_rad: torch.Tensor | None = None
        self._home_gripper_rad: float | None = None
        self._last_action = torch.zeros(6, dtype=torch.float32)
        self._last_action[-1] = self._gripper_open_command
        self._last_frame_timestamp = 0.0
        self._last_session_id: str | None = None
        self._thread.start()

    def __del__(self):
        self.close()

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}(listen={self._cfg.host}:{self._cfg.port}, "
            f"zero_on_first_frame={self._cfg.zero_on_first_frame}, "
            f"gripper_close_direction={self._cfg.gripper_close_direction}, "
            f"gripper_close_delta={self._cfg.gripper_close_delta})"
        )

    def reset(self):
        with self._lock:
            self._last_action = torch.zeros(6, dtype=torch.float32)
            self._last_action[-1] = self._gripper_open_command

    def add_callback(self, key: str, func):
        self._callbacks[str(key).upper()] = func

    def advance(self) -> torch.Tensor:
        with self._lock:
            action = self._last_action.clone()
            last_frame_timestamp = self._last_frame_timestamp

        if last_frame_timestamp == 0.0:
            return action.to(device=self._sim_device)

        if self._cfg.stale_timeout > 0.0:
            age = time.monotonic() - last_frame_timestamp
            if age > self._cfg.stale_timeout and self._cfg.verbose:
                print(f"[KochMasterArmStreamDevice] No fresh frame for {age:.2f}s. Holding the last action.")
                with self._lock:
                    self._last_frame_timestamp = time.monotonic()

        return action.to(device=self._sim_device)

    def close(self):
        self._stop_event.set()
        if self._client_socket is not None:
            try:
                self._client_socket.close()
            except OSError:
                pass
            self._client_socket = None
        if self._server_socket is not None:
            try:
                self._server_socket.close()
            except OSError:
                pass
            self._server_socket = None

    def _server_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
                    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    server.bind((self._cfg.host, self._cfg.port))
                    server.listen(1)
                    server.settimeout(self._cfg.socket_accept_timeout)
                    self._server_socket = server
                    if self._cfg.verbose:
                        print(f"[KochMasterArmStreamDevice] Listening on {self._cfg.host}:{self._cfg.port}")

                    while not self._stop_event.is_set():
                        try:
                            client, address = server.accept()
                        except socket.timeout:
                            continue

                        self._client_socket = client
                        if self._cfg.verbose:
                            print(f"[KochMasterArmStreamDevice] Client connected from {address}")
                        try:
                            self._handle_client(client)
                        finally:
                            try:
                                client.close()
                            except OSError:
                                pass
                            self._client_socket = None
                            if self._cfg.verbose:
                                print("[KochMasterArmStreamDevice] Client disconnected")
            except OSError as exc:
                if self._stop_event.is_set():
                    break
                print(f"[KochMasterArmStreamDevice] Socket error: {exc}. Retrying...")
                time.sleep(1.0)

    def _handle_client(self, client: socket.socket) -> None:
        client.settimeout(1.0)
        buffer = b""
        while not self._stop_event.is_set():
            try:
                chunk = client.recv(4096)
            except socket.timeout:
                continue
            except OSError:
                break

            if not chunk:
                break

            buffer += chunk
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                if not line.strip():
                    continue
                try:
                    message = parse_jsonl_message(line)
                except ValueError as exc:
                    if self._cfg.verbose:
                        print(f"[KochMasterArmStreamDevice] Ignoring invalid JSON line: {exc}")
                    continue
                self._handle_message(message)

    def _handle_message(self, message: dict[str, Any]) -> None:
        message_type = message.get("type")
        if message_type == "session_start":
            self._handle_session_start(message)
            return
        if message_type == "joint_frame":
            self._handle_joint_frame(message)

    def _handle_session_start(self, message: dict[str, Any]) -> None:
        session_id = message.get("session_id")
        with self._lock:
            self._last_session_id = session_id
            if self._cfg.zero_on_first_frame:
                self._home_arm_rad = None
                self._home_gripper_rad = None
        if self._cfg.verbose:
            print(f"[KochMasterArmStreamDevice] Session started: {session_id}")

    def _handle_joint_frame(self, message: dict[str, Any]) -> None:
        try:
            joint_map = joint_map_from_frame(message, expected_joint_order=self._cfg.expected_joint_order)
        except ValueError as exc:
            if self._cfg.verbose:
                print(f"[KochMasterArmStreamDevice] Ignoring frame: {exc}")
            return

        leader_arm_rad = torch.tensor(
            [joint_map[name] for name in self._expected_arm_joint_order], dtype=torch.float32
        )
        leader_gripper_rad = float(joint_map[self._expected_gripper_joint])

        with self._lock:
            if self._cfg.zero_on_first_frame and self._home_arm_rad is None:
                self._home_arm_rad = leader_arm_rad.clone()
                self._home_gripper_rad = leader_gripper_rad
                if self._cfg.verbose:
                    print("[KochMasterArmStreamDevice] Captured leader home pose from first frame")

            home_arm = self._home_arm_rad if self._home_arm_rad is not None else torch.zeros_like(leader_arm_rad)
            home_gripper = self._home_gripper_rad if self._home_gripper_rad is not None else 0.0

            arm_action = (leader_arm_rad - home_arm) * self._joint_signs + self._joint_offsets
            gripper_delta = leader_gripper_rad - home_gripper
            if self._cfg.gripper_close_direction.lower() == "positive":
                close_progress = gripper_delta / self._gripper_close_delta
            else:
                close_progress = -gripper_delta / self._gripper_close_delta

            close_progress = max(0.0, min(1.0, close_progress))
            gripper_action = self._gripper_open_command + close_progress * (
                self._gripper_close_command - self._gripper_open_command
            )
            gripper_action = max(self._gripper_command_min, min(self._gripper_command_max, gripper_action))

            self._last_action = torch.cat([arm_action, torch.tensor([gripper_action], dtype=torch.float32)])
            self._last_frame_timestamp = time.monotonic()


KochMasterArmStreamDeviceCfg.class_type = KochMasterArmStreamDevice
