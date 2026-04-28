"""Koch leader 本地采集端。

这个脚本运行在连接真实 leader 机械臂的本地电脑上，负责完成三件事：
1. 通过 USB 串口读取 6 个 Dynamixel 关节的当前角度。
2. 将读取结果封装成统一的 JSONL 消息格式。
3. 通过 SSH 隧道把消息转发到云端 Isaac Sim / Isaac Lab 所在机器。

整条链路刻意拆成“本地采集端”和“云端接收端”两部分：
- 本地只负责稳定读串口、稳定发流，不掺杂仿真逻辑。
- 云端只负责把 leader 关节角映射成仿真动作，不直接接触真实硬件。

这样做的好处是职责清晰，出了问题也容易定位：如果串口读不到，就查本地；
如果 SSH 隧道或仿真动作不对，就查云端脚本。
"""

from __future__ import annotations

import argparse
import getpass
import glob
import math
import os
import sys
import time
from datetime import datetime
from importlib import import_module
from pathlib import Path
from typing import Any

from koch_mimic.shared.configuration import (
    activate_runtime_config,
    get_config_section,
    option_was_provided,
    resolve_config_path,
)
from koch_mimic.shared.constants import LOCAL_PROFILE
from koch_mimic.shared.joint_stream import (
    JointSample,
    build_joint_frame_message,
    build_session_start_message,
    jsonl_dumps,
)

paramiko = None


# SSH 连接与远程 TCP 接收端的默认参数。
# 这些值服务于“本地 leader -> SSH 隧道 -> 云端 TCP 设备”这条固定链路。
DEFAULT_SSH_HOST = "183.147.142.40"
DEFAULT_SSH_PORT = 31339
DEFAULT_SSH_USERNAME = "root"
DEFAULT_STREAM_HOST = "127.0.0.1"
DEFAULT_STREAM_PORT = 55000

# Leader 串口与 Dynamixel 侧的默认配置。
DEFAULT_BAUDRATE = 1_000_000
DEFAULT_SAMPLE_HZ = 60.0
PRESENT_POSITION_ADDR = 132
PRESENT_POSITION_LEN = 4
EXPECTED_MODEL_NUMBER = 1190
DXL_IDS_TO_PROBE = (1, 2, 3, 4, 5, 6)

# 统一规定发送到云端时的关节顺序。
# 云端接收端会按这个顺序解析 joint_position_rad / deg / ticks。
MOTORS = (
    ("shoulder_pan", 1),
    ("shoulder_lift", 2),
    ("elbow_flex", 3),
    ("wrist_flex", 4),
    ("wrist_roll", 5),
    ("gripper", 6),
)


def decode_signed_32bit(value: int) -> int:
    """把 Dynamixel 返回的 32 位原始值解释成有符号整数。"""
    if value & 0x80000000:
        return value - 0x100000000
    return value


def tick_to_degree(value: int) -> float:
    """把编码器 ticks 转成角度制。"""
    return value * 360.0 / 4096.0


def tick_to_radian(value: int) -> float:
    """把编码器 ticks 转成弧度制。"""
    return value * (2.0 * math.pi) / 4096.0


def sort_port_name(port_name: str) -> tuple[int, int | str]:
    upper = port_name.upper()
    if upper.startswith("COM") and upper[3:].isdigit():
        return (0, int(upper[3:]))
    return (1, port_name)


def list_candidate_ports(patterns: Sequence[str] | None = None) -> list[str]:
    """枚举可能的串口设备。

    Windows 下从注册表里取 COM 口；
    Linux / macOS 下则根据常见的 USB 串口路径模式去扫描。
    """
    ports: set[str] = set()

    if sys.platform.startswith("win"):
        try:
            import winreg

            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DEVICEMAP\SERIALCOMM") as key:
                index = 0
                while True:
                    try:
                        _, value, _ = winreg.EnumValue(key, index)
                    except OSError:
                        break
                    if isinstance(value, str):
                        ports.add(value)
                    index += 1
        except OSError:
            pass
    else:
        glob_patterns = tuple(patterns) if patterns is not None else (
            "/dev/ttyUSB*",
            "/dev/ttyACM*",
            "/dev/tty.usbserial*",
            "/dev/tty.usbmodem*",
            "/dev/cu.usbserial*",
            "/dev/cu.usbmodem*",
        )
        for pattern in glob_patterns:
            ports.update(glob.glob(pattern))

    return sorted(ports, key=sort_port_name)


def ensure_dir(path: Path) -> None:
    """确保日志输出目录存在。"""
    path.mkdir(parents=True, exist_ok=True)


def remove_file_if_exists(path: Path | None) -> None:
    """删除本地缓存文件；不存在时静默跳过。"""
    if path is None:
        return
    try:
        path.unlink(missing_ok=True)
    except TypeError:
        if path.exists():
            path.unlink()


def require_paramiko() -> Any:
    """延迟导入 paramiko。

    这么做的目的是让纯串口调试阶段不强依赖 SSH 库，
    只有真正开始建立 SSH 隧道时才检查依赖。
    """
    global paramiko
    if paramiko is None:
        try:
            paramiko = import_module("paramiko")
        except ImportError as exc:  # pragma: no cover - dependency error is handled at runtime
            raise RuntimeError("Missing paramiko. Install it with: pip install paramiko") from exc
    if paramiko is None:
        raise RuntimeError("Missing paramiko. Install it with: pip install paramiko")
    return paramiko


def load_dynamixel_sdk() -> Any:
    """延迟导入 Dynamixel SDK。"""
    try:
        import dynamixel_sdk as dxl  # type: ignore
    except ImportError as exc:  # pragma: no cover - dependency error is handled at runtime
        raise RuntimeError("Missing dynamixel_sdk. Install it with: pip install dynamixel-sdk") from exc
    return dxl

class KochLeaderUSBReader:
    """负责和真实 leader 机械臂的 Dynamixel 总线通信。"""

    def __init__(
        self,
        port: str | None,
        baudrate: int,
        motor_layout: tuple[tuple[str, int], ...],
        *,
        probe_ids: tuple[int, ...],
        expected_model_number: int,
        port_globs: tuple[str, ...],
    ):
        self.port = port
        self.baudrate = baudrate
        self.motor_layout = motor_layout
        self.probe_ids = probe_ids
        self.expected_model_number = expected_model_number
        self.port_globs = port_globs
        self.dxl = load_dynamixel_sdk()
        self.packet_handler = self.dxl.PacketHandler(2.0)
        self.port_handler = None
        self.sync_reader = None

    def connect(self) -> str:
        """打开串口，并把所有目标电机加入 GroupSyncRead。"""
        selected_port = self.port
        if not selected_port or selected_port.lower() == "auto":
            selected_port = self._discover_port()

        self.port_handler = self.dxl.PortHandler(selected_port)
        if not self.port_handler.openPort():
            raise RuntimeError(f"Failed to open serial port {selected_port}")
        if not self.port_handler.setBaudRate(self.baudrate):
            self.port_handler.closePort()
            raise RuntimeError(f"Failed to set {selected_port} to baudrate {self.baudrate}")

        self.sync_reader = self.dxl.GroupSyncRead(
            self.port_handler,
            self.packet_handler,
            PRESENT_POSITION_ADDR,
            PRESENT_POSITION_LEN,
        )
        for _, motor_id in self.motor_layout:
            if not self.sync_reader.addParam(motor_id):
                raise RuntimeError(f"Failed to register motor id {motor_id} for sync read")

        self.port = selected_port
        return selected_port

    def _discover_port(self) -> str:
        """自动寻找最像 Koch leader 的串口。

        这里不是简单找“第一个 COM 口”，而是逐个串口做 ping 探测，
        选出能找到最多目标电机的那个端口。
        """
        candidates = list_candidate_ports(self.port_globs)
        if not candidates:
            raise RuntimeError("No serial port found. Please specify one manually, for example: --port COM5")

        best_port: str | None = None
        best_score = -1
        for candidate in candidates:
            score = self._probe_port(candidate)
            if score > best_score:
                best_port = candidate
                best_score = score

        if best_port is None or best_score <= 0:
            raise RuntimeError(
                "Auto discovery failed. No Koch leader bus was found on the available ports. "
                f"Scanned ports: {', '.join(candidates)}"
            )

        return best_port

    def _probe_port(self, port_name: str) -> int:
        """评估某个串口是否是目标机械臂所连接的总线。

        返回值越大，说明在该端口上探测到的目标电机越多。
        """
        port_handler = self.dxl.PortHandler(port_name)
        score = 0
        try:
            if not port_handler.openPort():
                return 0
            if not port_handler.setBaudRate(self.baudrate):
                return 0

            for motor_id in self.probe_ids:
                try:
                    model_number, comm_result, error = self.packet_handler.ping(port_handler, motor_id)
                except Exception:
                    continue

                if comm_result != self.dxl.COMM_SUCCESS or error != 0:
                    continue
                if model_number == self.expected_model_number:
                    score += 1
        finally:
            try:
                port_handler.closePort()
            except Exception:
                pass

        return score

    def read_joint_positions(self) -> dict[str, JointSample]:
        """读取当前所有关节位置。

        这里使用 GroupSyncRead 一次性读回全部关节，
        以减少逐关节轮询带来的延迟和抖动。
        """
        if self.port_handler is None or self.sync_reader is None:
            raise RuntimeError("Robot is not connected yet. Call connect() first.")

        comm_result = self.sync_reader.txRxPacket()
        if comm_result != self.dxl.COMM_SUCCESS:
            error_text = self.packet_handler.getTxRxResult(comm_result)
            raise RuntimeError(f"Failed to sync read present positions: {error_text}")

        result: dict[str, JointSample] = {}
        for motor_name, motor_id in self.motor_layout:
            is_available = self.sync_reader.isAvailable(motor_id, PRESENT_POSITION_ADDR, PRESENT_POSITION_LEN)
            if not is_available:
                raise RuntimeError(f"Present position is not available for motor id {motor_id} ({motor_name})")

            raw_value = self.sync_reader.getData(motor_id, PRESENT_POSITION_ADDR, PRESENT_POSITION_LEN)
            ticks = decode_signed_32bit(raw_value)
            result[motor_name] = JointSample(
                motor_id=motor_id,
                ticks=ticks,
                degrees=tick_to_degree(ticks),
                radians=tick_to_radian(ticks),
            )

        return result

    def close(self) -> None:
        """关闭同步读对象和串口。"""
        if self.sync_reader is not None:
            try:
                self.sync_reader.clearParam()
            except Exception:
                pass
            self.sync_reader = None

        if self.port_handler is not None:
            try:
                self.port_handler.closePort()
            finally:
                self.port_handler = None


class OptionalJsonlLogger:
    """可选的本地 JSONL 日志器。

    它的作用不是主链路必须的一环，而是方便排查：
    即使云端暂时没监听，本地也能保留采样原始记录。
    """

    def __init__(self, path: Path | None):
        self.path = path
        self._file = None

    def open(self) -> None:
        if self.path is None:
            return
        ensure_dir(self.path.parent)
        self._file = self.path.open("a", encoding="utf-8", buffering=1)

    def write(self, line: str) -> None:
        if self._file is not None:
            self._file.write(line)

    def close(self) -> None:
        if self._file is not None:
            try:
                self._file.close()
            finally:
                self._file = None


class SSHRealtimeJSONLStreamer:
    """通过 SSH 隧道把 JSONL 实时发往云端 TCP 接收端。"""

    def __init__(
        self,
        host: str,
        port: int,
        username: str,
        password: str,
        remote_stream_host: str,
        remote_stream_port: int,
        connect_timeout: float,
        keepalive_interval: float,
    ):
        self.paramiko = require_paramiko()
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.remote_stream_host = remote_stream_host
        self.remote_stream_port = remote_stream_port
        self.connect_timeout = connect_timeout
        self.keepalive_interval = keepalive_interval
        self._ssh = None
        self._channel = None

    @property
    def target(self) -> str:
        return f"{self.remote_stream_host}:{self.remote_stream_port}"

    def connect(self) -> None:
        """建立 SSH 连接，并在 SSH 内部打开 direct-tcpip 通道。

        这里不是登录后执行远程命令，而是直接把本地采样数据
        透传到云端某个已经在监听的 TCP 端口。
        """
        self.close()

        ssh = self.paramiko.SSHClient()
        ssh.set_missing_host_key_policy(self.paramiko.AutoAddPolicy())
        ssh.connect(
            hostname=self.host,
            port=self.port,
            username=self.username,
            password=self.password,
            timeout=self.connect_timeout,
            banner_timeout=self.connect_timeout,
            auth_timeout=self.connect_timeout,
            look_for_keys=False,
            allow_agent=False,
        )

        transport = ssh.get_transport()
        if transport is None or not transport.is_active():
            ssh.close()
            raise RuntimeError("SSH transport is not active after connect")

        keepalive_seconds = max(1, int(round(self.keepalive_interval)))
        transport.set_keepalive(keepalive_seconds)

        try:
            channel = transport.open_channel(
                kind="direct-tcpip",
                dest_addr=(self.remote_stream_host, self.remote_stream_port),
                src_addr=("127.0.0.1", 0),
                timeout=self.connect_timeout,
            )
        except Exception as exc:
            ssh.close()
            raise RuntimeError(
                "Failed to open an SSH-tunneled TCP stream to "
                f"{self.remote_stream_host}:{self.remote_stream_port}. "
                "Make sure Isaac Sim or its receiver is already listening on that port."
            ) from exc

        channel.settimeout(self.connect_timeout)
        self._ssh = ssh
        self._channel = channel

    def send_line(self, line: str) -> None:
        """发送一行 JSONL 消息。

        上层保证每个消息都以换行结尾，云端就能按行拆包。
        """
        if self._channel is None or self._ssh is None:
            self.connect()

        payload = line.encode("utf-8")
        try:
            self._channel.sendall(payload)
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        """关闭 SSH 通道和 SSH 会话。"""
        if self._channel is not None:
            try:
                self._channel.close()
            finally:
                self._channel = None
        if self._ssh is not None:
            try:
                self._ssh.close()
            finally:
                self._ssh = None


def build_session_start(
    session_id: str,
    port_name: str,
    baudrate: int,
    motor_layout: tuple[tuple[str, int], ...],
) -> dict[str, Any]:
    """构建一条会话开始消息。

    云端接收端会用它来重置“首帧归零”等会话级状态。
    """
    return build_session_start_message(
        session_id=session_id,
        serial_port=port_name,
        baudrate=baudrate,
        joint_order=tuple(name for name, _ in motor_layout),
    )


def build_frame(
    session_id: str,
    sequence: int,
    port_name: str,
    baudrate: int,
    joint_positions: dict[str, JointSample],
    motor_layout: tuple[tuple[str, int], ...],
) -> dict[str, Any]:
    """构建一条关节数据帧。

    同时发送 ticks / deg / rad 三种表达方式，是为了兼顾：
    - 云端控制通常直接使用 rad；
    - 人工排查更适合看 deg；
    - 底层调试时往往还需要原始 ticks。
    """
    return build_joint_frame_message(
        session_id=session_id,
        sequence=sequence,
        serial_port=port_name,
        baudrate=baudrate,
        joint_positions=joint_positions,
        joint_order=tuple(name for name, _ in motor_layout),
    )


def build_arg_parser() -> argparse.ArgumentParser:
    """定义命令行参数。"""
    parser = argparse.ArgumentParser(
        description=(
            "Read Koch leader joint data from USB and stream it in real time over an "
            "SSH-tunneled TCP JSONL connection."
        )
    )
    parser.add_argument("--config", default=None, help="Optional extra local YAML overlay.")
    parser.add_argument("--port", default="COM5", help="USB serial port, for example COM5. Default: auto.")
    parser.add_argument("--baudrate", type=int, default=DEFAULT_BAUDRATE, help="Dynamixel baudrate.")
    parser.add_argument("--host", default=DEFAULT_SSH_HOST, help="SSH server host.")
    parser.add_argument("--ssh-port", type=int, default=DEFAULT_SSH_PORT, help="SSH server port.")
    parser.add_argument("--username", default=DEFAULT_SSH_USERNAME, help="SSH username.")
    parser.add_argument(
        "--password",
        default=os.getenv("KOCH_SSH_PASSWORD"),
        help="SSH password. If omitted, you will be prompted, or use KOCH_SSH_PASSWORD.",
    )
    parser.add_argument(
        "--stream-host",
        default=DEFAULT_STREAM_HOST,
        help="TCP host on the remote server that Isaac Sim listens on. Usually 127.0.0.1.",
    )
    parser.add_argument(
        "--stream-port",
        type=int,
        default=DEFAULT_STREAM_PORT,
        help="TCP port on the remote server that Isaac Sim listens on.",
    )
    parser.add_argument("--sample-hz", type=float, default=DEFAULT_SAMPLE_HZ, help="Sampling frequency in Hz.")
    parser.add_argument("--connect-timeout", type=float, default=5.0, help="SSH and stream connect timeout in seconds.")
    parser.add_argument(
        "--keepalive-interval",
        type=float,
        default=10.0,
        help="SSH keepalive interval in seconds.",
    )
    parser.add_argument(
        "--reconnect-backoff",
        type=float,
        default=1.0,
        help="Delay before retrying after the stream connection drops, in seconds.",
    )
    parser.add_argument(
        "--status-interval",
        type=float,
        default=0.5,
        help="Console status print interval in seconds.",
    )
    parser.add_argument("--max-samples", type=int, default=0, help="Maximum number of samples. 0 means run forever.")
    parser.add_argument("--once", action="store_true", help="Read and stream one sample, then exit.")
    parser.add_argument(
        "--local-log",
        default="",
        help="Optional local JSONL log file path. Leave empty to disable disk logging.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """主循环。

    循环逻辑很直接：
    1. 先连本地串口，再连云端 SSH/TCP。
    2. 按固定采样频率读取 leader 关节。
    3. 生成 JSONL 帧，先写本地日志，再尝试推送到云端。
    4. 云端断开时不阻塞采样，等待退避时间后继续用“最新帧”重连。
    """
    argv_list = list(sys.argv[1:] if argv is None else argv)
    args = build_arg_parser().parse_args(argv_list)
    config = activate_runtime_config(LOCAL_PROFILE, overlay_path=args.config, require_user_local=True)
    motor_layout = tuple(
        (str(name), int(motor_id))
        for name, motor_id in dict(get_config_section(config, "leader_arm", "motor_layout", default=dict(MOTORS))).items()
    )
    probe_ids = tuple(
        int(value) for value in get_config_section(config, "leader_arm", "probe_ids", default=list(DXL_IDS_TO_PROBE))
    )
    expected_model_number = int(
        get_config_section(config, "leader_arm", "expected_model_number", default=EXPECTED_MODEL_NUMBER)
    )
    port_globs = tuple(
        str(value)
        for value in get_config_section(
            config,
            "discovery",
            "port_globs",
            default=[
                "/dev/ttyUSB*",
                "/dev/ttyACM*",
                "/dev/tty.usbserial*",
                "/dev/tty.usbmodem*",
                "/dev/cu.usbserial*",
                "/dev/cu.usbmodem*",
            ],
        )
    )

    if not option_was_provided(argv_list, "--port"):
        args.port = str(get_config_section(config, "leader_arm", "serial_port", default=args.port))
    if not option_was_provided(argv_list, "--baudrate"):
        args.baudrate = int(get_config_section(config, "leader_arm", "baudrate", default=args.baudrate))
    if not option_was_provided(argv_list, "--sample-hz"):
        args.sample_hz = float(get_config_section(config, "leader_arm", "sample_hz", default=args.sample_hz))
    if not option_was_provided(argv_list, "--max-samples"):
        args.max_samples = int(get_config_section(config, "leader_arm", "max_samples", default=args.max_samples))
    if not option_was_provided(argv_list, "--status-interval"):
        args.status_interval = float(
            get_config_section(config, "leader_arm", "status_interval", default=args.status_interval)
        )
    if not option_was_provided(argv_list, "--reconnect-backoff"):
        args.reconnect_backoff = float(
            get_config_section(config, "leader_arm", "reconnect_backoff", default=args.reconnect_backoff)
        )
    if not option_was_provided(argv_list, "--host"):
        args.host = str(get_config_section(config, "ssh", "host", default=args.host))
    if not option_was_provided(argv_list, "--ssh-port"):
        args.ssh_port = int(get_config_section(config, "ssh", "port", default=args.ssh_port))
    if not option_was_provided(argv_list, "--username"):
        args.username = str(get_config_section(config, "ssh", "username", default=args.username))
    if not option_was_provided(argv_list, "--stream-host"):
        args.stream_host = str(
            get_config_section(config, "ssh", "remote_stream_host", default=args.stream_host)
        )
    if not option_was_provided(argv_list, "--stream-port"):
        args.stream_port = int(
            get_config_section(config, "ssh", "remote_stream_port", default=args.stream_port)
        )
    if not option_was_provided(argv_list, "--connect-timeout"):
        args.connect_timeout = float(
            get_config_section(config, "ssh", "connect_timeout", default=args.connect_timeout)
        )
    if not option_was_provided(argv_list, "--keepalive-interval"):
        args.keepalive_interval = float(
            get_config_section(config, "ssh", "keepalive_interval", default=args.keepalive_interval)
        )
    if not option_was_provided(argv_list, "--local-log"):
        args.local_log = str(get_config_section(config, "logging", "local_jsonl_path", default=args.local_log))
    cleanup_local_log_on_exit = bool(
        get_config_section(config, "logging", "cleanup_local_jsonl_on_exit", default=True)
    )

    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    password_env = str(get_config_section(config, "ssh", "password_env", default="KOCH_SSH_PASSWORD"))
    config_password = get_config_section(config, "ssh", "password", default=None)
    password = args.password or config_password or os.getenv(password_env) or getpass.getpass("SSH password: ")
    local_log_path = Path(resolve_config_path(args.local_log, config)).expanduser().resolve() if args.local_log else None

    reader = KochLeaderUSBReader(
        port=args.port,
        baudrate=args.baudrate,
        motor_layout=motor_layout,
        probe_ids=probe_ids,
        expected_model_number=expected_model_number,
        port_globs=port_globs,
    )
    streamer = SSHRealtimeJSONLStreamer(
        host=args.host,
        port=args.ssh_port,
        username=args.username,
        password=password,
        remote_stream_host=args.stream_host,
        remote_stream_port=args.stream_port,
        connect_timeout=args.connect_timeout,
        keepalive_interval=args.keepalive_interval,
    )
    logger = OptionalJsonlLogger(local_log_path)
    logger.open()

    print("Connecting to Koch leader...")
    selected_port = reader.connect()
    print(f"Connected serial port: {selected_port} @ {args.baudrate}")
    print(f"Connecting realtime stream through SSH to {streamer.target} ...")
    streamer.connect()
    print(f"Realtime stream is ready: ssh://{args.username}@{args.host}:{args.ssh_port} -> {streamer.target}")

    period_sec = 0.0 if args.sample_hz <= 0 else 1.0 / args.sample_hz
    sequence = 0
    next_status_deadline = time.monotonic()
    reconnect_deadline = 0.0
    last_stream_error: str | None = None
    session_announcement_pending = True

    try:
        while True:
            loop_started_at = time.monotonic()
            joint_positions = reader.read_joint_positions()
            frame = build_frame(session_id, sequence, selected_port, args.baudrate, joint_positions, motor_layout)
            frame_line = jsonl_dumps(frame)
            logger.write(frame_line)

            now = time.monotonic()
            stream_state = "ok"

            if now >= reconnect_deadline:
                try:
                    # 每次新建连接后都先发一条 session_start，提醒云端重置会话状态。
                    if session_announcement_pending:
                        session_start = build_session_start(session_id, selected_port, args.baudrate, motor_layout)
                        session_line = jsonl_dumps(session_start)
                        streamer.send_line(session_line)
                        session_announcement_pending = False

                    streamer.send_line(frame_line)
                    if last_stream_error is not None:
                        print("Realtime stream recovered. Isaac Sim should receive new frames again.")
                    last_stream_error = None
                except Exception as exc:
                    # 注意这里不会停掉本地采样，只是进入“等待重连”状态。
                    # 这样当云端恢复监听时，leader 的最新姿态还能立刻继续推送。
                    current_error = str(exc)
                    if current_error != last_stream_error:
                        print(f"Realtime stream disconnected. Retrying with latest frames only: {current_error}")
                    last_stream_error = current_error
                    session_announcement_pending = True
                    reconnect_deadline = now + max(args.reconnect_backoff, 0.1)
                    streamer.close()
                    stream_state = "reconnecting"
            else:
                stream_state = "waiting-reconnect"

            if now >= next_status_deadline:
                # 控制台状态输出只用于人眼观察，不参与协议本身。
                summary = ", ".join(f"{name}={sample.ticks}" for name, sample in joint_positions.items())
                print(f"[{frame['timestamp']}] seq={sequence} stream={stream_state} {summary}")
                next_status_deadline = now + max(args.status_interval, 0.2)

            sequence += 1
            if args.once or (args.max_samples > 0 and sequence >= args.max_samples):
                break

            if period_sec > 0:
                elapsed = time.monotonic() - loop_started_at
                sleep_time = period_sec - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
    except KeyboardInterrupt:
        print("Received Ctrl+C. Exiting safely...")
    finally:
        logger.close()
        streamer.close()
        reader.close()
        if cleanup_local_log_on_exit:
            remove_file_if_exists(local_log_path)
            if local_log_path is not None:
                print(f"Removed local JSONL cache: {local_log_path}")

    if local_log_path is not None and not cleanup_local_log_on_exit:
        print(f"Local JSONL log: {local_log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
