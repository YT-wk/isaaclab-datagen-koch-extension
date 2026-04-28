"""Minimal serial-read monitor for the Koch leader arm."""

from __future__ import annotations

import argparse
import time
from collections.abc import Sequence

from koch_mimic.local.leader_arm import Motor, MotorNormMode
from koch_mimic.local.leader_arm.dynamixel import DynamixelMotorsBus
from koch_mimic.shared.configuration import activate_runtime_config, get_config_section
from koch_mimic.shared.constants import LOCAL_PROFILE


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Continuously read Koch leader positions from the serial bus.")
    parser.add_argument("--config", type=str, default=None, help="Optional extra local YAML overlay.")
    parser.add_argument("--port", type=str, default=None, help="Serial port, for example COM5.")
    parser.add_argument("--baudrate", type=int, default=None, help="Dynamixel baudrate.")
    parser.add_argument("--read-timeout-ms", type=int, default=None, help="Per-packet timeout in milliseconds.")
    return parser


def build_motors(motor_layout: dict[str, int]) -> dict[str, Motor]:
    motors: dict[str, Motor] = {}
    for name, motor_id in motor_layout.items():
        norm_mode = MotorNormMode.RANGE_0_100 if name == "gripper" else MotorNormMode.RANGE_M100_100
        motors[name] = Motor(int(motor_id), "xl330-m077", norm_mode)
    return motors


def ticks_to_degrees(ticks: int) -> float:
    return ticks * 360.0 / 4096.0


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    config = activate_runtime_config(LOCAL_PROFILE, overlay_path=args.config, require_user_local=True)

    motor_layout = dict(get_config_section(config, "leader_arm", "motor_layout", default={}))
    port = args.port or str(get_config_section(config, "leader_arm", "serial_port", default="auto"))
    baudrate = int(args.baudrate or get_config_section(config, "leader_arm", "baudrate", default=1_000_000))
    read_timeout_ms = int(
        args.read_timeout_ms or get_config_section(config, "leader_arm", "read_timeout_ms", default=200)
    )

    motors_bus = DynamixelMotorsBus(
        port=port,
        motors=build_motors(motor_layout),
        calibration={},
    )
    motor_names = list(motors_bus.motors)

    try:
        print(f"Opening serial port {port}...")
        motors_bus.connect(handshake=False)
        motors_bus.set_baudrate(baudrate)
        motors_bus.set_timeout(read_timeout_ms)
        print(f"Connected to DYNAMIXEL bus on {port} at {baudrate} baud.")
        print(f"Read timeout set to {read_timeout_ms} ms.")

        while True:
            raw_positions = motors_bus.sync_read("Present_Position", motor_names, normalize=False)
            temperatures = motors_bus.sync_read("Present_Temperature", motor_names, normalize=False)

            position_degrees = {
                name: round(ticks_to_degrees(int(raw_positions[name])), 2) for name in motor_names
            }

            print(f"positions_raw={raw_positions}")
            print(f"positions_deg={position_degrees}")
            print(f"temperatures_c={temperatures}")
            print("-" * 60)
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Interrupted by user.")
    except Exception as exc:
        print(f"Serial test failed: {exc}")
        raise
    finally:
        if motors_bus.is_connected:
            motors_bus.disconnect(disable_torque=False)
        print("Disconnected from DYNAMIXEL bus.")


if __name__ == "__main__":
    main()
