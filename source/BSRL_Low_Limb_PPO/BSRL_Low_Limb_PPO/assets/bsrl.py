import os  # noqa: I001

import isaaclab.sim as sim_utils
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils import configclass

from BSRL_Low_Limb_PPO.assets.delayed_implicit_actuator import DelayedPowerLimitedPDActuatorCfg

BSRL_MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "bsrl"))

BSRL_DEFAULT_ROOT_HEIGHT = 0.8665

BSRL_ACTUATOR_MIN_DELAY = 0
BSRL_ACTUATOR_MAX_DELAY = 4

BSRL_SMALL_MOTOR_PEAK_TORQUE = 112.0
BSRL_SMALL_MOTOR_PEAK_SPEED = 250.0 * 2.0 * 3.141592653589793 / 60.0
BSRL_SMALL_MOTOR_PEAK_POWER = 1250.0

BSRL_LARGE_MOTOR_PEAK_TORQUE = 216.0
BSRL_LARGE_MOTOR_PEAK_SPEED = 144.5 * 2.0 * 3.141592653589793 / 60.0
BSRL_LARGE_MOTOR_PEAK_POWER = 1050.0

BSRL_ACTION_SCALE_MULTIPLIER = {
    "joint_.*_hip_pitch": 0.25,
    "joint_.*_hip_roll": 0.25,
    "joint_.*_hip_yaw": 0.25,
    "joint_.*_knee_pitch": 0.25,
    "joint_.*_ankle_roll": 0.25,
    "joint_.*_ankle_pitch": 0.25,
}


@configclass
class BSRLArticulationCfg(ArticulationCfg):
    joint_sdk_names: list[str] = None
    soft_joint_pos_limit_factor = 0.9


BSRL_CFG = BSRLArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{BSRL_MODEL_DIR}/urdf/export/export.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, BSRL_DEFAULT_ROOT_HEIGHT),
        joint_pos={
            "joint_.*_hip_yaw": 0.0,
            "joint_.*_hip_roll": 0.0,
            "joint_.*_hip_pitch": -0.2,
            "joint_.*_knee_pitch": 0.4,
            "joint_.*_ankle_pitch": -0.2,
            "joint_.*_ankle_roll": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "large_motors": DelayedPowerLimitedPDActuatorCfg(
            min_delay=BSRL_ACTUATOR_MIN_DELAY,
            max_delay=BSRL_ACTUATOR_MAX_DELAY,
            joint_names_expr=[
                "joint_.*_hip_pitch",
                "joint_.*_hip_roll",
                "joint_.*_knee_pitch",
            ],
            effort_limit=BSRL_LARGE_MOTOR_PEAK_TORQUE,
            effort_limit_sim=BSRL_LARGE_MOTOR_PEAK_TORQUE,
            velocity_limit=BSRL_LARGE_MOTOR_PEAK_SPEED,
            velocity_limit_sim=BSRL_LARGE_MOTOR_PEAK_SPEED,
            peak_power=BSRL_LARGE_MOTOR_PEAK_POWER,
            armature=0.01,
            stiffness={
                "joint_.*_hip_pitch": 100,
                "joint_.*_hip_roll": 100,
                "joint_.*_knee_pitch": 150,
            },
            damping={
                "joint_.*_hip_pitch": 2,
                "joint_.*_hip_roll": 2,
                "joint_.*_knee_pitch": 4,
            },
        ),
        "small_motors": DelayedPowerLimitedPDActuatorCfg(
            min_delay=BSRL_ACTUATOR_MIN_DELAY,
            max_delay=BSRL_ACTUATOR_MAX_DELAY,
            joint_names_expr=[
                "joint_.*_hip_yaw",
                "joint_.*_ankle_roll",
                "joint_.*_ankle_pitch",
            ],
            effort_limit=BSRL_SMALL_MOTOR_PEAK_TORQUE,
            effort_limit_sim=BSRL_SMALL_MOTOR_PEAK_TORQUE,
            velocity_limit=BSRL_SMALL_MOTOR_PEAK_SPEED,
            velocity_limit_sim=BSRL_SMALL_MOTOR_PEAK_SPEED,
            peak_power=BSRL_SMALL_MOTOR_PEAK_POWER,
            armature=0.01,
            stiffness={
                "joint_.*_hip_yaw": 100,
                "joint_.*_ankle_roll": 40,
                "joint_.*_ankle_pitch": 40,
            },
            damping={
                "joint_.*_hip_yaw": 2,
                "joint_.*_ankle_roll": 2,
                "joint_.*_ankle_pitch": 2,
            },
        ),
    },
    joint_sdk_names=[
        "joint_right_hip_yaw",
        "joint_right_hip_roll",
        "joint_right_hip_pitch",
        "joint_right_knee_pitch",
        "joint_right_ankle_pitch",
        "joint_right_ankle_roll",
        "joint_left_hip_yaw",
        "joint_left_hip_roll",
        "joint_left_hip_pitch",
        "joint_left_knee_pitch",
        "joint_left_ankle_pitch",
        "joint_left_ankle_roll",
    ],
)


def _build_action_scale(robot_cfg: ArticulationCfg) -> dict[str, float]:
    action_scale = {}
    for actuator_cfg in robot_cfg.actuators.values():
        for name in actuator_cfg.joint_names_expr:
            action_scale[name] = BSRL_ACTION_SCALE_MULTIPLIER.get(name, 0.25)
    return action_scale


BSRL_ACTION_SCALE = _build_action_scale(BSRL_CFG)
