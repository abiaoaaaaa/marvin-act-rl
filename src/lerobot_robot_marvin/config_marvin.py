from dataclasses import dataclass
import sys

from lerobot.robots.config import RobotConfig


_marvin_dataclass = dataclass(kw_only=True) if sys.version_info >= (3, 10) else dataclass


@RobotConfig.register_subclass("marvin")
@_marvin_dataclass
class MarvinRobotConfig(RobotConfig):
    host: str = "127.0.0.1"
    port: int = 8765
    timeout_s: float = 1.0
    connect_timeout_s: float = 30.0
    reset_timeout_s: float = 120.0
    fail_safe_on_invalid_action: bool = True
