"""Single source of truth for the Marvin policy-side schema."""

CAMERA_KEYS = ("side_policy", "wrist_1buchong", "wrist_1", "wrist_2")
IMAGE_SHAPE = (128, 128, 3)
FPS = 10

STATE_NAMES = (
    "gripper_pose",
    "last_action.dx",
    "last_action.dy",
    "last_action.dz",
    "last_action.drz",
    "last_action.gripper",
    "tcp_force.x",
    "tcp_force.y",
    "tcp_force.z",
    "tcp_pose.x",
    "tcp_pose.y",
    "tcp_pose.z",
    "tcp_pose.rx",
    "tcp_pose.ry",
    "tcp_pose.rz",
    "tcp_torque.x",
    "tcp_torque.y",
    "tcp_torque.z",
    "tcp_vel.x",
    "tcp_vel.y",
    "tcp_vel.z",
    "tcp_vel.rx",
    "tcp_vel.ry",
    "tcp_vel.rz",
)
ACTION_NAMES = ("dx", "dy", "dz", "drz", "gripper")

# LeRobot 59ab2862 filters rollout scalar keys by these suffixes.  They are
# dictionary transport names only; vector order and values remain unchanged.
ROBOT_STATE_KEYS = tuple(f"{name}.pos" for name in STATE_NAMES)
ROBOT_ACTION_KEYS = tuple(f"{name}.vel" for name in ACTION_NAMES)

ACTION_LOW = (-1.0,) * 5
ACTION_HIGH = (1.0,) * 5

