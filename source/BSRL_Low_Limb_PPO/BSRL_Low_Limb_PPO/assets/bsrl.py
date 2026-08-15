import os  # noqa: I001

import isaaclab.sim as sim_utils
from isaaclab.assets.articulation import ArticulationCfg

from bsrl_low_limb_ppo.assets.delayed_implicit_actuator import DelayedImplicitActuatorCfg
# BSRL_Low_Limb_PPO

BSRL_MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "bsrl"))

# 机器人默认高度
BSRL_DEFAULT_ROOT_HEIGHT = 0.8665

# 训练时为每个环境随机采样 0~4 个物理步的执行器命令延迟。
BSRL_ACTUATOR_MIN_DELAY = 0
BSRL_ACTUATOR_MAX_DELAY = 4

BSRL_CFG = ArticulationCfg(
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
            enabled_self_collisions=False,
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
        "legs": DelayedImplicitActuatorCfg(
            min_delay=BSRL_ACTUATOR_MIN_DELAY,
            max_delay=BSRL_ACTUATOR_MAX_DELAY,
            joint_names_expr=[
                "joint_.*_hip_pitch",
                "joint_.*_hip_roll",
                "joint_.*_hip_yaw",
                "joint_.*_knee_pitch",
            ],
            effort_limit_sim=1000.0,
            velocity_limit_sim=10.0,
            armature=0.01,
            stiffness={
                "joint_.*_hip_pitch": 100,
                "joint_.*_hip_roll": 100,
                "joint_.*_hip_yaw": 100,
                "joint_.*_knee_pitch": 150,
            },
            damping={
                "joint_.*_hip_pitch": 2,
                "joint_.*_hip_roll": 2,
                "joint_.*_hip_yaw": 2,
                "joint_.*_knee_pitch": 4,
            },
        ),
        "feet": DelayedImplicitActuatorCfg(
            min_delay=BSRL_ACTUATOR_MIN_DELAY,
            max_delay=BSRL_ACTUATOR_MAX_DELAY,
            joint_names_expr=[
                "joint_.*_ankle_roll",
                "joint_.*_ankle_pitch",
            ],
            effort_limit_sim=1000.0,
            velocity_limit_sim=10.0,
            armature=0.01,
            stiffness={
                "joint_.*_ankle_roll": 40,
                "joint_.*_ankle_pitch": 40,
            },
            damping={
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
