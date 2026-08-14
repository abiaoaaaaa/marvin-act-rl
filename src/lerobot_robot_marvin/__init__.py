"""Marvin deployment adapter.

The bridge is intentionally usable in the HIL-SERL environment where LeRobot
is not installed.  When LeRobot is present, importing this package also
registers the third-party RobotConfig.
"""

try:
    from .config_marvin import MarvinRobotConfig
    from .marvin import MarvinRobot
except ImportError:
    # HIL-SERL may contain an unrelated/incomplete LeRobot checkout (and its
    # torch build can be intentionally incompatible).  Bridge-only modules
    # must remain importable there.  The lerobot_xlb environment imports these
    # classes normally and performs plugin registration.
    pass
else:
    __all__ = ["MarvinRobot", "MarvinRobotConfig"]
