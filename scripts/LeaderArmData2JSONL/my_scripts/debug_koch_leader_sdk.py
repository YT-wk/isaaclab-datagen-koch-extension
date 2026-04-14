from __future__ import annotations

import argparse
import json
import sys
import time
from contextlib import contextmanager
from pathlib import Path

# 把项目根目录加入 sys.path，确保从 my_scripts 直接运行时仍能导入上一级模块。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dynamixel import DynamixelMotorsBus, OperatingMode
from motors_bus import Motor, MotorNormMode

try:
    from serial.tools import list_ports
except ImportError:  # pragma: no cover
    list_ports = None


DEFAULT_BAUDRATE = 1_000_000
DEFAULT_IDS = (1, 2, 3, 4, 5, 6)
DEFAULT_BAUDRATE_SCAN = (57_600, 115_200, 1_000_000, 2_000_000, 3_000_000, 4_000_000)
EXPECTED_MODEL_NUMBER = 1190

OPERATING_MODE_NAMES = {
    0: "CURRENT",
    1: "VELOCITY",
    3: "POSITION",
    4: "EXTENDED_POSITION",
    5: "CURRENT_POSITION",
    16: "PWM",
}

KOCH_MOTOR_NAMES = {
    1: "shoulder_pan",
    2: "shoulder_lift",
    3: "elbow_flex",
    4: "wrist_flex",
    5: "wrist_roll",
    6: "gripper",
}
KOCH_NON_GRIPPER_IDS = (1, 2, 3, 4, 5)
KOCH_GRIPPER_ID = 6
KOCH_RETURN_DELAY_MIN = 0


def parse_int_list(value: str) -> list[int]:
    result: list[int] = []
    for part in value.split(","):
        token = part.strip()
        if not token:
            continue
        if "-" in token:
            start_str, end_str = token.split("-", 1)
            start = int(start_str)
            end = int(end_str)
            step = 1 if end >= start else -1
            result.extend(list(range(start, end + step, step)))
        else:
            result.append(int(token))
    if not result:
        raise ValueError(f"Failed to parse integer list from: {value}")
    return result


def format_register_value(name: str, value: int) -> str:
    if name == "Present_Position":
        deg = ticks_to_degrees(value)
        rad = ticks_to_radians(value)
        return f"{value} ticks ({deg:.2f} deg, {rad:.3f} rad)"
    if name == "Present_Current":
        return f"{value}"
    if name == "Present_Input_Voltage":
        return f"{value / 10.0:.1f} V"
    if name == "Present_Temperature":
        return f"{value} C"
    if name == "Torque_Enable":
        return "ON" if value else "OFF"
    if name == "Operating_Mode":
        return f"{value} ({OPERATING_MODE_NAMES.get(value, 'UNKNOWN')})"
    return str(value)


def list_serial_ports() -> list[str]:
    if list_ports is None:
        return []
    return [port.device for port in list_ports.comports()]


def ticks_to_degrees(value: int) -> float:
    return value * 360.0 / 4096.0


def ticks_to_radians(value: int) -> float:
    return value * 2.0 * 3.141592653589793 / 4096.0


def build_action_frame(positions: dict[int, int]) -> dict[str, dict[str, float | int]]:
    action: dict[str, dict[str, float | int]] = {}
    for motor_id, raw_ticks in positions.items():
        motor_name = KOCH_MOTOR_NAMES.get(motor_id, f"motor_{motor_id}")
        action[f"{motor_name}.pos"] = {
            "ticks": raw_ticks,
            "degrees": round(ticks_to_degrees(raw_ticks), 3),
            "radians": round(ticks_to_radians(raw_ticks), 6),
        }
    return action


def motor_name_from_id(motor_id: int) -> str:
    return KOCH_MOTOR_NAMES.get(motor_id, f"motor_{motor_id}")


def build_koch_motors(ids: list[int] | tuple[int, ...]) -> dict[str, Motor]:
    motors: dict[str, Motor] = {}
    for motor_id in ids:
        norm_mode = MotorNormMode.RANGE_0_100 if motor_id == KOCH_GRIPPER_ID else MotorNormMode.RANGE_M100_100
        motors[motor_name_from_id(motor_id)] = Motor(motor_id, "xl330-m077", norm_mode)
    return motors


def motor_names_from_ids(ids: list[int] | tuple[int, ...]) -> list[str]:
    return [motor_name_from_id(motor_id) for motor_id in ids]


def make_bus(port: str, ids: list[int] | tuple[int, ...]) -> DynamixelMotorsBus:
    return DynamixelMotorsBus(port=port, motors=build_koch_motors(ids), calibration={})


@contextmanager
def connected_bus(
    port: str,
    baudrate: int,
    ids: list[int] | tuple[int, ...],
    *,
    handshake: bool = False,
    timeout_ms: int | None = None,
):
    bus = make_bus(port, ids)
    bus.connect(handshake=handshake)
    try:
        bus.set_baudrate(baudrate)
        if timeout_ms is not None:
            bus.set_timeout(timeout_ms)
        yield bus
    finally:
        if bus.is_connected:
            bus.disconnect(disable_torque=False)


def sync_read_positions_via_bus(
    bus: DynamixelMotorsBus,
    ids: list[int] | tuple[int, ...],
) -> dict[int, int]:
    raw_positions = bus.sync_read("Present_Position", motor_names_from_ids(ids), normalize=False)
    return {bus.motors[motor_name].id: int(position) for motor_name, position in raw_positions.items()}


def probe_motor_models(
    bus: DynamixelMotorsBus,
    ids: list[int] | tuple[int, ...],
    *,
    verbose: bool = False,
) -> dict[int, int]:
    online: dict[int, int] = {}
    for motor_id in ids:
        if verbose:
            print(f"  pinging id={motor_id} ...", flush=True)
        model_number = bus.ping(motor_name_from_id(motor_id))
        if model_number is not None:
            online[motor_id] = int(model_number)
            if verbose:
                print(f"  id={motor_id}: model={model_number}")
        elif verbose:
            print(f"  id={motor_id}: no response")
    return online


def configure_like_koch_leader(bus: DynamixelMotorsBus, gripper_open_raw: int | None) -> None:
    bus.disable_torque()
    bus.configure_motors(return_delay_time=KOCH_RETURN_DELAY_MIN)

    for motor_name in motor_names_from_ids(KOCH_NON_GRIPPER_IDS):
        bus.write("Operating_Mode", motor_name, OperatingMode.EXTENDED_POSITION.value, normalize=False)

    gripper_name = motor_name_from_id(KOCH_GRIPPER_ID)
    bus.write("Operating_Mode", gripper_name, OperatingMode.CURRENT_POSITION.value, normalize=False)
    bus.enable_torque(gripper_name)

    if gripper_open_raw is not None:
        bus.write("Goal_Position", gripper_name, gripper_open_raw, normalize=False)


def print_ports() -> None:
    ports = list_serial_ports()
    if not ports:
        print("No serial ports found.")
        return
    print("Available serial ports:")
    for port in ports:
        print(f"  - {port}")


def run_scan(ports: list[str], baudrates: list[int], ids: list[int]) -> int:
    found_any = False
    for port in ports:
        print(f"\n=== Scan port {port} ===")
        for baudrate in baudrates:
            bus = make_bus(port, ids)
            try:
                bus.connect(handshake=False)
                bus.set_baudrate(baudrate)
            except Exception as exc:
                print(f"  baudrate={baudrate}: open failed -> {exc}")
                continue

            try:
                broadcast = bus.broadcast_ping()
                if broadcast:
                    found_any = True
                    print(f"  baudrate={baudrate}: broadcast ping -> {broadcast}")
                else:
                    print(f"  baudrate={baudrate}: broadcast ping -> none")

                hits = {}
                for motor_id in ids:
                    model_number = bus.ping(motor_id)
                    if model_number is not None:
                        hits[motor_id] = model_number
                if hits:
                    found_any = True
                    print(f"  baudrate={baudrate}: targeted ping -> {hits}")
            except Exception as exc:
                print(f"  baudrate={baudrate}: ping failed -> {exc}")
            finally:
                if bus.is_connected:
                    bus.disconnect(disable_torque=False)

    return 0 if found_any else 1


def run_status(
    port: str,
    baudrate: int,
    ids: list[int],
    *,
    timeout_ms: int,
    skip_broadcast: bool,
    skip_sync_read: bool,
) -> int:
    with connected_bus(port, baudrate, ids, handshake=False, timeout_ms=timeout_ms) as bus:
        print(f"Connected: {port} @ {baudrate}")
        print(f"Bus timeout: {timeout_ms} ms")
        print("Step 1/3: targeted ping per motor ...", flush=True)
        online_models = probe_motor_models(bus, ids, verbose=True)
        online_ids = [motor_id for motor_id in ids if motor_id in online_models]
        print(f"Online IDs: {online_ids}" if online_ids else "Online IDs: none")

        if not skip_broadcast:
            print("Step 2/3: broadcast_ping ...", flush=True)
            try:
                broadcast = bus.broadcast_ping()
                print(f"Broadcast ping: {broadcast}")
            except Exception as exc:
                print(f"Broadcast ping failed: {exc}")
        else:
            print("Step 2/3: broadcast_ping skipped")

        position_map = {}
        if not skip_sync_read and online_ids:
            print(f"Step 3/3: sync_read Present_Position on online IDs {online_ids} ...", flush=True)
            try:
                position_map = sync_read_positions_via_bus(bus, online_ids)
                print(f"Sync read ok: {sorted(position_map)}")
            except Exception as exc:
                print(f"Sync read positions failed, fallback to single read: {exc}")
        elif not online_ids:
            print("Step 3/3: sync_read skipped because no motors responded to targeted ping")
        else:
            print("Step 3/3: sync_read skipped")

        registers = [
            "Model_Number",
            "Torque_Enable",
            "Operating_Mode",
            "Drive_Mode",
            "Present_Current",
            "Present_Input_Voltage",
            "Present_Temperature",
            "Hardware_Error_Status",
        ]

        print("Detail readout:", flush=True)
        for motor_id in ids:
            motor_name = motor_name_from_id(motor_id)
            print(f"\n[{motor_name}] id={motor_id}")
            model_number = online_models.get(motor_id)
            if model_number is None:
                print("  ping: no response")
                continue
            print(
                f"  ping: model={model_number}"
                + (" (xl330-m077 expected)" if model_number == EXPECTED_MODEL_NUMBER else "")
            )

            for register_name in registers:
                try:
                    value = int(bus.read(register_name, motor_name, normalize=False))
                    print(f"  {register_name}: {format_register_value(register_name, value)}")
                except Exception as exc:
                    print(f"  {register_name}: ERROR -> {exc}")

            try:
                position = (
                    position_map[motor_id]
                    if motor_id in position_map
                    else int(bus.read("Present_Position", motor_name, normalize=False))
                )
                print(f"  Present_Position: {format_register_value('Present_Position', position)}")
            except Exception as exc:
                print(f"  Present_Position: ERROR -> {exc}")
    return 0


def run_monitor(port: str, baudrate: int, ids: list[int], hz: float, *, timeout_ms: int) -> int:
    period = 0.0 if hz <= 0 else 1.0 / hz
    with connected_bus(port, baudrate, ids, handshake=False, timeout_ms=timeout_ms) as bus:
        print(f"Monitoring Present_Position on {port} @ {baudrate}. Press Ctrl+C to stop.")
        try:
            while True:
                started = time.perf_counter()
                positions = sync_read_positions_via_bus(bus, ids)
                summary = ", ".join(
                    f"{motor_name_from_id(motor_id)}={positions[motor_id]}"
                    for motor_id in ids
                    if motor_id in positions
                )
                print(summary)
                if period > 0:
                    sleep_time = period - (time.perf_counter() - started)
                    if sleep_time > 0:
                        time.sleep(sleep_time)
        except KeyboardInterrupt:
            print("Stopped.")
            return 0


def run_leader_configure(port: str, baudrate: int, gripper_open_raw: int | None) -> int:
    with connected_bus(port, baudrate, DEFAULT_IDS, handshake=False) as bus:
        configure_like_koch_leader(bus, gripper_open_raw)
        print(f"Koch leader style configure finished on {port} @ {baudrate}")
        print("  - motors 1-5 -> Operating_Mode = EXTENDED_POSITION (4)")
        print("  - motor 6    -> Operating_Mode = CURRENT_POSITION (5)")
        print("  - Return_Delay_Time -> 0 for all motors")
        print("  - gripper torque -> ON")
        if gripper_open_raw is not None:
            print(f"  - gripper Goal_Position -> {gripper_open_raw}")
        else:
            print("  - gripper Goal_Position -> unchanged (not provided)")
    return 0


def run_leader_action(port: str, baudrate: int, ids: list[int], *, timeout_ms: int) -> int:
    with connected_bus(port, baudrate, ids, handshake=False, timeout_ms=timeout_ms) as bus:
        positions = sync_read_positions_via_bus(bus, ids)
        action = build_action_frame(positions)
        print(json.dumps(action, indent=2, ensure_ascii=False))
    return 0


def run_leader_monitor(port: str, baudrate: int, ids: list[int], hz: float, *, timeout_ms: int) -> int:
    period = 0.0 if hz <= 0 else 1.0 / hz
    with connected_bus(port, baudrate, ids, handshake=False, timeout_ms=timeout_ms) as bus:
        print(f"Monitoring KochLeader-style action on {port} @ {baudrate}. Press Ctrl+C to stop.")
        try:
            while True:
                started = time.perf_counter()
                positions = sync_read_positions_via_bus(bus, ids)
                summary = ", ".join(
                    f"{motor_name_from_id(motor_id)}.pos={positions[motor_id]}"
                    for motor_id in ids
                    if motor_id in positions
                )
                print(summary)
                if period > 0:
                    sleep_time = period - (time.perf_counter() - started)
                    if sleep_time > 0:
                        time.sleep(sleep_time)
        except KeyboardInterrupt:
            print("Stopped.")
            return 0


def run_torque(port: str, baudrate: int, ids: list[int], enabled: bool, *, timeout_ms: int) -> int:
    with connected_bus(port, baudrate, ids, handshake=False, timeout_ms=timeout_ms) as bus:
        if enabled:
            bus.enable_torque(ids)
        else:
            bus.disable_torque(ids)
        for motor_id in ids:
            print(f"id={motor_id}: torque -> {'ON' if enabled else 'OFF'}")
    return 0


def run_led(port: str, baudrate: int, ids: list[int], enabled: bool, *, timeout_ms: int) -> int:
    with connected_bus(port, baudrate, ids, handshake=False, timeout_ms=timeout_ms) as bus:
        value = 1 if enabled else 0
        for motor_id in ids:
            bus.write("LED", motor_name_from_id(motor_id), value, normalize=False)
            print(f"id={motor_id}: LED -> {'ON' if enabled else 'OFF'}")
    return 0


def run_mode(port: str, baudrate: int, ids: list[int], mode_value: int, *, timeout_ms: int) -> int:
    with connected_bus(port, baudrate, ids, handshake=False, timeout_ms=timeout_ms) as bus:
        bus.disable_torque(ids)
        for motor_id in ids:
            bus.write("Operating_Mode", motor_name_from_id(motor_id), mode_value, normalize=False)
            print(f"id={motor_id}: Operating_Mode -> {mode_value} ({OPERATING_MODE_NAMES.get(mode_value, 'UNKNOWN')})")
    return 0


def run_nudge(port: str, baudrate: int, motor_id: int, delta_ticks: int, *, timeout_ms: int) -> int:
    with connected_bus(port, baudrate, [motor_id], handshake=False, timeout_ms=timeout_ms) as bus:
        motor_name = motor_name_from_id(motor_id)
        current_position = int(bus.read("Present_Position", motor_name, normalize=False))
        target_position = current_position + delta_ticks
        torque_enabled = int(bus.read("Torque_Enable", motor_name, normalize=False))
        if not torque_enabled:
            bus.enable_torque(motor_name)
        bus.write("Goal_Position", motor_name, target_position, normalize=False)
        print(
            f"id={motor_id}: Present_Position={current_position}, "
            f"Goal_Position={target_position} (delta={delta_ticks})"
        )
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Debug Koch leader / Dynamixel motors with DynamixelMotorsBus and KochLeader-style helpers."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    ports_parser = subparsers.add_parser("ports", help="List available serial ports.")
    ports_parser.set_defaults(handler=lambda args: (print_ports(), 0)[1])

    scan_parser = subparsers.add_parser("scan", help="Scan serial ports, baudrates and motor IDs.")
    scan_parser.add_argument("--ports", default="", help="Comma-separated serial ports. Empty means auto-detect.")
    scan_parser.add_argument(
        "--baudrates",
        default=",".join(str(v) for v in DEFAULT_BAUDRATE_SCAN),
        help="Comma-separated baudrates.",
    )
    scan_parser.add_argument(
        "--ids",
        default="1-6",
        help="Motor IDs to targeted-ping, for example 1-6 or 1,2,3,4,5,6.",
    )
    scan_parser.set_defaults(handler=None)

    status_parser = subparsers.add_parser("status", help="Read model, mode, torque and sensor values.")
    status_parser.add_argument("--port", required=True, help="Serial port, for example COM3.")
    status_parser.add_argument("--baudrate", type=int, default=DEFAULT_BAUDRATE, help="Baudrate.")
    status_parser.add_argument("--ids", default="1-6", help="Motor IDs to inspect.")
    status_parser.add_argument("--timeout-ms", type=int, default=300, help="Per-packet timeout in milliseconds.")
    status_parser.add_argument(
        "--skip-broadcast",
        action="store_true",
        help="Skip broadcast ping. Useful when the adapter/driver stalls on broadcast packets.",
    )
    status_parser.add_argument(
        "--skip-sync-read",
        action="store_true",
        help="Skip initial sync_read and fall back to per-motor reads only.",
    )
    status_parser.set_defaults(handler=None)

    monitor_parser = subparsers.add_parser(
        "monitor",
        help="Continuously read Present_Position through DynamixelMotorsBus.sync_read().",
    )
    monitor_parser.add_argument("--port", required=True, help="Serial port, for example COM3.")
    monitor_parser.add_argument("--baudrate", type=int, default=DEFAULT_BAUDRATE, help="Baudrate.")
    monitor_parser.add_argument("--ids", default="1-6", help="Motor IDs to read.")
    monitor_parser.add_argument("--hz", type=float, default=10.0, help="Refresh rate in Hz.")
    monitor_parser.add_argument("--timeout-ms", type=int, default=300, help="Per-packet timeout in milliseconds.")
    monitor_parser.set_defaults(handler=None)

    leader_configure_parser = subparsers.add_parser(
        "leader-configure",
        help="Configure motors the same way as koch_leader.KochLeader.configure().",
    )
    leader_configure_parser.add_argument("--port", required=True, help="Serial port, for example COM3.")
    leader_configure_parser.add_argument("--baudrate", type=int, default=DEFAULT_BAUDRATE, help="Baudrate.")
    leader_configure_parser.add_argument(
        "--gripper-open-raw",
        type=int,
        default=None,
        help="Optional raw Goal_Position for the gripper after switching to current-position mode.",
    )
    leader_configure_parser.set_defaults(handler=None)

    leader_action_parser = subparsers.add_parser(
        "leader-action",
        help="Read KochLeader-style action once and print JSON.",
    )
    leader_action_parser.add_argument("--port", required=True, help="Serial port, for example COM3.")
    leader_action_parser.add_argument("--baudrate", type=int, default=DEFAULT_BAUDRATE, help="Baudrate.")
    leader_action_parser.add_argument("--ids", default="1-6", help="Motor IDs to read.")
    leader_action_parser.add_argument(
        "--timeout-ms", type=int, default=300, help="Per-packet timeout in milliseconds."
    )
    leader_action_parser.set_defaults(handler=None)

    leader_monitor_parser = subparsers.add_parser(
        "leader-monitor",
        help="Continuously read KochLeader-style action, similar to get_action().",
    )
    leader_monitor_parser.add_argument("--port", required=True, help="Serial port, for example COM3.")
    leader_monitor_parser.add_argument("--baudrate", type=int, default=DEFAULT_BAUDRATE, help="Baudrate.")
    leader_monitor_parser.add_argument("--ids", default="1-6", help="Motor IDs to read.")
    leader_monitor_parser.add_argument("--hz", type=float, default=30.0, help="Refresh rate in Hz.")
    leader_monitor_parser.add_argument(
        "--timeout-ms", type=int, default=300, help="Per-packet timeout in milliseconds."
    )
    leader_monitor_parser.set_defaults(handler=None)

    torque_parser = subparsers.add_parser("torque", help="Enable or disable torque on selected motors.")
    torque_parser.add_argument("--port", required=True, help="Serial port, for example COM3.")
    torque_parser.add_argument("--baudrate", type=int, default=DEFAULT_BAUDRATE, help="Baudrate.")
    torque_parser.add_argument("--ids", default="1-6", help="Motor IDs to modify.")
    torque_parser.add_argument("--state", choices=("on", "off"), required=True, help="Torque target state.")
    torque_parser.add_argument("--timeout-ms", type=int, default=300, help="Per-packet timeout in milliseconds.")
    torque_parser.set_defaults(handler=None)

    led_parser = subparsers.add_parser("led", help="Turn motor LED on or off.")
    led_parser.add_argument("--port", required=True, help="Serial port, for example COM3.")
    led_parser.add_argument("--baudrate", type=int, default=DEFAULT_BAUDRATE, help="Baudrate.")
    led_parser.add_argument("--ids", default="1-6", help="Motor IDs to modify.")
    led_parser.add_argument("--state", choices=("on", "off"), required=True, help="LED target state.")
    led_parser.add_argument("--timeout-ms", type=int, default=300, help="Per-packet timeout in milliseconds.")
    led_parser.set_defaults(handler=None)

    mode_parser = subparsers.add_parser("mode", help="Set Operating_Mode after disabling torque.")
    mode_parser.add_argument("--port", required=True, help="Serial port, for example COM3.")
    mode_parser.add_argument("--baudrate", type=int, default=DEFAULT_BAUDRATE, help="Baudrate.")
    mode_parser.add_argument("--ids", default="1-6", help="Motor IDs to modify.")
    mode_parser.add_argument("--value", type=int, required=True, help="Operating mode value, e.g. 3/4/5.")
    mode_parser.add_argument("--timeout-ms", type=int, default=300, help="Per-packet timeout in milliseconds.")
    mode_parser.set_defaults(handler=None)

    nudge_parser = subparsers.add_parser("nudge", help="Move one motor by a small delta in ticks.")
    nudge_parser.add_argument("--port", required=True, help="Serial port, for example COM3.")
    nudge_parser.add_argument("--baudrate", type=int, default=DEFAULT_BAUDRATE, help="Baudrate.")
    nudge_parser.add_argument("--motor-id", type=int, required=True, help="Motor ID to move.")
    nudge_parser.add_argument("--delta", type=int, required=True, help="Goal position delta in ticks.")
    nudge_parser.add_argument("--timeout-ms", type=int, default=300, help="Per-packet timeout in milliseconds.")
    nudge_parser.set_defaults(handler=None)

    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.command == "scan":
        port_list = [token.strip() for token in args.ports.split(",") if token.strip()] or list_serial_ports()
        if not port_list:
            print("No serial ports found for scanning.")
            return 1
        return run_scan(port_list, parse_int_list(args.baudrates), parse_int_list(args.ids))

    if args.command == "status":
        return run_status(
            args.port,
            args.baudrate,
            parse_int_list(args.ids),
            timeout_ms=args.timeout_ms,
            skip_broadcast=args.skip_broadcast,
            skip_sync_read=args.skip_sync_read,
        )

    if args.command == "monitor":
        return run_monitor(args.port, args.baudrate, parse_int_list(args.ids), args.hz, timeout_ms=args.timeout_ms)

    if args.command == "leader-configure":
        return run_leader_configure(args.port, args.baudrate, args.gripper_open_raw)

    if args.command == "leader-action":
        return run_leader_action(
            args.port,
            args.baudrate,
            parse_int_list(args.ids),
            timeout_ms=args.timeout_ms,
        )

    if args.command == "leader-monitor":
        return run_leader_monitor(
            args.port,
            args.baudrate,
            parse_int_list(args.ids),
            args.hz,
            timeout_ms=args.timeout_ms,
        )

    if args.command == "torque":
        return run_torque(
            args.port,
            args.baudrate,
            parse_int_list(args.ids),
            args.state == "on",
            timeout_ms=args.timeout_ms,
        )

    if args.command == "led":
        return run_led(
            args.port,
            args.baudrate,
            parse_int_list(args.ids),
            args.state == "on",
            timeout_ms=args.timeout_ms,
        )

    if args.command == "mode":
        return run_mode(
            args.port,
            args.baudrate,
            parse_int_list(args.ids),
            args.value,
            timeout_ms=args.timeout_ms,
        )

    if args.command == "nudge":
        return run_nudge(
            args.port,
            args.baudrate,
            args.motor_id,
            args.delta,
            timeout_ms=args.timeout_ms,
        )

    if args.command == "ports":
        print_ports()
        return 0

    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
