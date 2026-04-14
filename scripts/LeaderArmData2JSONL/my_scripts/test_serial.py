import time
import sys
from pathlib import Path

# 把项目根目录加入 sys.path，确保从 my_scripts 直接运行时仍能导入上一级模块。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dynamixel import DynamixelMotorsBus
from motors_bus import Motor, MotorNormMode
from utils.configs import DynamixelMotorsBusConfig


BAUDRATE = 1_000_000
READ_TIMEOUT_MS = 200

lead_motor_config = {
    "shoulder_pan": (1, "xl330-m077"),
    "shoulder_lift": (2, "xl330-m077"),
    "elbow_flex": (3, "xl330-m077"),
    "wrist_flex": (4, "xl330-m077"),
    "wrist_roll": (5, "xl330-m077"),
    "gripper": (6, "xl330-m077"),
}

config = DynamixelMotorsBusConfig(
    port="COM5",
    motors=lead_motor_config,
    mock=False,
)


def build_motors(cfg: DynamixelMotorsBusConfig) -> dict[str, Motor]:
    motors: dict[str, Motor] = {}
    for name, (motor_id, model) in cfg.motors.items():
        norm_mode = MotorNormMode.RANGE_0_100 if name == "gripper" else MotorNormMode.RANGE_M100_100
        motors[name] = Motor(motor_id, model, norm_mode)
    return motors


def ticks_to_degrees(ticks: int) -> float:
    return ticks * 360.0 / 4096.0


def main() -> None:
    motors_bus = DynamixelMotorsBus(
        port=config.port,
        motors=build_motors(config),
        calibration={},
    )
    motor_names = list(motors_bus.motors)

    try:
        print(f"Opening serial port {config.port}...")
        motors_bus.connect(handshake=False)
        motors_bus.set_baudrate(BAUDRATE)
        motors_bus.set_timeout(READ_TIMEOUT_MS)
        print(f"Connected to DYNAMIXEL bus on {config.port} at {BAUDRATE} baud.")
        print(f"Read timeout set to {READ_TIMEOUT_MS} ms.")

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
