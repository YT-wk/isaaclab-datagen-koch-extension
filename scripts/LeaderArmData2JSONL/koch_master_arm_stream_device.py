"""Koch leader 云端接收设备。

这个模块运行在云端 Isaac Lab 所在机器上，职责是把本地 leader 发来的 JSONL
关节流转换成 Isaac Lab teleop device 能消费的动作向量。

它本质上是一个“TCP 设备适配层”：
- 上游协议：`koch_leader_ssh_streamer.py` 发来的 `session_start` / `joint_frame`
- 下游协议：Isaac Lab 期望的 `DeviceBase.advance()` 动作张量
"""

from __future__ import annotations

import json
import socket
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import torch

from isaaclab.devices.device_base import DeviceBase, DeviceCfg


DEFAULT_LEADER_JOINT_ORDER = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)


@dataclass
class KochMasterArmStreamDeviceCfg(DeviceCfg):
    """云端 TCP 设备配置。

    这里把“leader 到仿真机械臂”的映射参数都集中放在配置里，
    方便通过命令行注入，而不需要硬编码在设备逻辑内部。
    """

    host: str = "127.0.0.1"
    port: int = 55000
    expected_joint_order: tuple[str, ...] = DEFAULT_LEADER_JOINT_ORDER
    joint_signs: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0, 1.0)
    joint_offsets: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0)
    zero_on_first_frame: bool = True
    gripper_close_threshold: float = 0.15
    gripper_close_direction: str = "negative"
    stale_timeout: float = 1.0
    socket_accept_timeout: float = 0.5
    verbose: bool = True
    retargeters: list = field(default_factory=list)
    class_type: type[DeviceBase] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.class_type is None:
            self.class_type = KochMasterArmStreamDevice


class KochMasterArmStreamDevice(DeviceBase):
    """把 leader 关节流转换成 Isaac Lab 动作向量的设备类。

    The device listens on a local TCP port inside the cloud server. The local machine connects to it
    through the SSH tunnel created by ``koch_leader_ssh_streamer.py`` and sends JSONL messages of type
    ``session_start`` and ``joint_frame``.

    The returned action vector is:
    ``[arm_j1, arm_j2, arm_j3, arm_j4, arm_j5, gripper_binary]``
    where the first five entries are absolute joint position targets in radians and the last entry is
    a continuous scalar for ``AbsBinaryJointPositionAction`` (1.0 = open, 0.0 = close).
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
        # 监听线程和仿真线程解耦，避免网络收包阻塞物理仿真主循环。
        self._thread = threading.Thread(target=self._server_loop, daemon=True)

        # 前 5 个关节映射到臂部绝对关节位置，第 6 个关节单独走二值夹爪逻辑。
        self._expected_arm_joint_order = tuple(cfg.expected_joint_order[:5])
        self._expected_gripper_joint = cfg.expected_joint_order[5]
        self._joint_signs = torch.tensor(cfg.joint_signs, dtype=torch.float32)
        self._joint_offsets = torch.tensor(cfg.joint_offsets, dtype=torch.float32)

        # home_* 用于“首帧归零”模式：把第一帧 leader 姿态当作仿真参考原点。
        self._home_arm_rad: torch.Tensor | None = None
        self._home_gripper_rad: float | None = None
        self._last_action = torch.zeros(6, dtype=torch.float32)
        self._last_action[-1] = 1.0
        self._last_frame_timestamp = 0.0
        self._last_session_id: str | None = None
        self._thread.start()

    def __del__(self):
        self.close()

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}(listen={self._cfg.host}:{self._cfg.port}, "
            f"zero_on_first_frame={self._cfg.zero_on_first_frame}, "
            f"gripper_close_direction={self._cfg.gripper_close_direction})"
        )

    def reset(self):
        """重置设备内部缓存动作。"""
        with self._lock:
            self._last_action = torch.zeros(6, dtype=torch.float32)
            self._last_action[-1] = 1.0

    def add_callback(self, key: str, func):
        """兼容 Isaac Lab teleop device 的回调注册接口。"""
        self._callbacks[str(key).upper()] = func

    def advance(self) -> torch.Tensor:
        """返回当前动作。

        仿真主循环每一步都会调用一次这个函数，因此这里必须尽量轻量：
        只读缓存，不做任何网络 IO。
        """
        with self._lock:
            action = self._last_action.clone()
            last_frame_timestamp = self._last_frame_timestamp

        if last_frame_timestamp == 0.0:
            return action.to(device=self._sim_device)

        if self._cfg.stale_timeout > 0.0:
            age = time.monotonic() - last_frame_timestamp
            if age > self._cfg.stale_timeout and self._cfg.verbose:
                print(
                    f"[KochMasterArmStreamDevice] No fresh frame for {age:.2f}s. "
                    "Holding the last action."
                )
                # Only print once per stale period.
                with self._lock:
                    self._last_frame_timestamp = time.monotonic()

        return action.to(device=self._sim_device)

    def close(self):
        """关闭监听线程相关资源。"""
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
        """后台监听线程。

        设备持续监听一个本地 TCP 端口，等待本地电脑通过 SSH 隧道连进来。
        """
        while not self._stop_event.is_set():
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
                    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    server.bind((self._cfg.host, self._cfg.port))
                    server.listen(1)
                    server.settimeout(self._cfg.socket_accept_timeout)
                    self._server_socket = server
                    if self._cfg.verbose:
                        print(
                            f"[KochMasterArmStreamDevice] Listening on {self._cfg.host}:{self._cfg.port}"
                        )

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
        """按行读取 JSONL，并逐条分发给消息处理器。"""
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
                    message = json.loads(line.decode("utf-8"))
                except json.JSONDecodeError as exc:
                    if self._cfg.verbose:
                        print(f"[KochMasterArmStreamDevice] Ignoring invalid JSON line: {exc}")
                    continue
                self._handle_message(message)

    def _handle_message(self, message: dict[str, Any]) -> None:
        """按照消息类型分发。"""
        message_type = message.get("type")
        if message_type == "session_start":
            self._handle_session_start(message)
            return
        if message_type == "joint_frame":
            self._handle_joint_frame(message)

    def _handle_session_start(self, message: dict[str, Any]) -> None:
        """处理一次新会话开始。

        如果启用了首帧归零，这里会清空 home pose，等待下一帧重新采集。
        """
        session_id = message.get("session_id")
        with self._lock:
            self._last_session_id = session_id
            if self._cfg.zero_on_first_frame:
                self._home_arm_rad = None
                self._home_gripper_rad = None
        if self._cfg.verbose:
            print(f"[KochMasterArmStreamDevice] Session started: {session_id}")

    def _handle_joint_frame(self, message: dict[str, Any]) -> None:
        """把一帧 leader 关节数据转换成 Isaac Lab 动作。

        转换公式分成两部分：
        - arm_action = (leader - home) * sign + offset
        - gripper 根据与首帧的相对变化量映射成开/合二值状态
        """
        joint_order = message.get("joint_order")
        joint_position_rad = message.get("joint_position_rad")
        if not isinstance(joint_order, list) or not isinstance(joint_position_rad, list):
            return
        if len(joint_order) != len(joint_position_rad):
            return

        joint_map = {str(name): float(value) for name, value in zip(joint_order, joint_position_rad, strict=False)}
        missing = [name for name in self._cfg.expected_joint_order if name not in joint_map]
        if missing:
            if self._cfg.verbose:
                print(f"[KochMasterArmStreamDevice] Missing joints in frame: {missing}")
            return

        leader_arm_rad = torch.tensor(
            [joint_map[name] for name in self._expected_arm_joint_order], dtype=torch.float32
        )
        leader_gripper_rad = float(joint_map[self._expected_gripper_joint])

        with self._lock:
            if self._cfg.zero_on_first_frame and self._home_arm_rad is None:
                # 记录第一帧 leader 姿态，之后所有动作都相对这个姿态来解释。
                self._home_arm_rad = leader_arm_rad.clone()
                self._home_gripper_rad = leader_gripper_rad
                if self._cfg.verbose:
                    print("[KochMasterArmStreamDevice] Captured leader home pose from first frame")

            home_arm = self._home_arm_rad if self._home_arm_rad is not None else torch.zeros_like(leader_arm_rad)
            home_gripper = self._home_gripper_rad if self._home_gripper_rad is not None else leader_gripper_rad

            # arm_action 是“绝对关节目标”，因此这里直接输出映射后的关节角目标。
            arm_action = (leader_arm_rad - home_arm) * self._joint_signs + self._joint_offsets
            gripper_delta = leader_gripper_rad - home_gripper
            if self._cfg.gripper_close_direction.lower() == "positive":
                gripper_is_closed = gripper_delta >= self._cfg.gripper_close_threshold
            else:
                gripper_is_closed = gripper_delta <= -self._cfg.gripper_close_threshold

            # 目前夹爪仍然使用二值开合，而不是连续角度。
            gripper_action = 0.0 if gripper_is_closed else 1.0
            self._last_action = torch.cat([arm_action, torch.tensor([gripper_action], dtype=torch.float32)])
            self._last_frame_timestamp = time.monotonic()


# 再做一层类属性兜底，确保从不同路径实例化配置时都能拿到正确的构造器。
KochMasterArmStreamDeviceCfg.class_type = KochMasterArmStreamDevice
