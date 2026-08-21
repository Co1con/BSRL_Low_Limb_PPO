import math  # noqa: I001

import isaaclab.sim as sim_utils
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from BSRL_Low_Limb_PPO.assets.bsrl import BSRL_CFG, BSRL_DEFAULT_ROOT_HEIGHT
from BSRL_Low_Limb_PPO.tasks.manager_based.bsrl_low_limb_ppo import mdp

from BSRL_Low_Limb_PPO.tasks.manager_based.bsrl_low_limb_ppo.bsrl_baseline_velocity_env_cfg import RobotEnvCfg


BSRL_FOOT_NAME = ["link_left_ankle_roll", "link_right_ankle_roll"]


@configclass
class HopfObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, noise=Unoise(n_min=-1.5, n_max=1.5))
        joint_effort = ObsTerm(func=mdp.joint_effort, scale=0.01)
        last_action = ObsTerm(func=mdp.last_action)

        hopf_master_xy = ObsTerm(func=mdp.hopf_master_xy, params={"command_name": "base_velocity"})

        def __post_init__(self):
            # self.history_length = 5
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


    @configclass
    class CriticCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05)
        joint_effort = ObsTerm(func=mdp.joint_effort, scale=0.01)
        last_action = ObsTerm(func=mdp.last_action)
        height_scanner = ObsTerm(func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 5.0),
        )

        hopf_master_xy = ObsTerm(func=mdp.hopf_master_xy, params={"command_name": "base_velocity"})

        # def __post_init__(self):
        #     self.history_length = 5

    critic: CriticCfg = CriticCfg()


@configclass
class HopfCommandsCfg:
    base_velocity = mdp.UniformLevelVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=False,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(0.3, 0.5), lin_vel_y=(-0.0, 0.0), ang_vel_z=(-0.0, 0.0)
        ),
        limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.2, 1.0), lin_vel_y=(-0.0, 0.0), ang_vel_z=(-0.0, 0.0)
        ),
    )


@configclass
class HopfRewardsCfg:
    # Task
    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=0.5,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )

    # Base
    alive = RewTerm(func=mdp.is_alive, weight=0.15)
    base_linear_velocity = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    base_angular_velocity = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-5.0)
    base_height = RewTerm(
        func=mdp.base_height_l2,
        weight=-10.0,
        params={
            "target_height": BSRL_DEFAULT_ROOT_HEIGHT,
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("height_scanner"),
        },
    )

    # Joints
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.001)
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.05)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-5.0)
    energy = RewTerm(func=mdp.energy, weight=-2e-5)
    joint_deviation_hips = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["joint_.*_hip_roll", "joint_.*_hip_yaw"])},
    )
    hopf_joint_tracking = RewTerm(
        func=mdp.hopf_joint_tracking,
        weight=0.5,
        params={
            "command_name": "base_velocity",
            "std": 0.15,
            "command_threshold": 0.1,
        },
    )

    # Feet
    feet_double_support = RewTerm(
        func=mdp.long_double_support_penalty,
        weight=-0.5,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=BSRL_FOOT_NAME),
            "command_threshold": 0.1,
            "allowed_time": 0.1
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=BSRL_FOOT_NAME),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=BSRL_FOOT_NAME),
        },
    )
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=1.0,
        params={            
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=BSRL_FOOT_NAME),
            "threshold": 0.5,
        },
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "threshold": 1,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["^(?!.*link_.*_ankle_roll).*"]),
        },
    )


@configclass
class HopfEventCfg:
    ## startup
    # 随机化材质
    # physics_material = EventTerm(
    #     func=mdp.randomize_rigid_body_material,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
    #         "static_friction_range": (0.6, 1.2),
    #         "dynamic_friction_range": (0.6, 1.2),
    #         "restitution_range": (0.0, 0.05),
    #         "num_buckets": 64,
    #     },
    # )
    # # 随机化质量
    # add_base_mass = EventTerm(
    #     func=mdp.randomize_rigid_body_mass,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
    #         "mass_distribution_params": (-5.0, 5.0),
    #         "operation": "add",
    #     },
    # )
    # # 随机化质心位置
    # base_com = EventTerm(
    #     func=mdp.randomize_rigid_body_com,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
    #         "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.01, 0.01)},
    #     },
    # )

    ## reset
    # 随机外力/力矩
    # base_external_force_torque = EventTerm(
    #     func=mdp.apply_external_force_torque,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
    #         "force_range": (0.0, 0.0),
    #         "torque_range": (-0.0, 0.0),
    #     },
    # )
    # 随机重置位姿
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
            # "velocity_range": {
                # "x": (-0.5, 0.5),
                # "y": (-0.5, 0.5),
                # "z": (-0.5, 0.5),
                # "roll": (-0.5, 0.5),
                # "pitch": (-0.5, 0.5),
                # "yaw": (-0.5, 0.5),
            # },
        },
    )
    # 随机大小
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (-0.0, 0.0),
        },
    )
    ## interval
    # 随机施加速度以推动机器人
    # push_robot = EventTerm(
    #     func=mdp.push_by_setting_velocity,
    #     mode="interval",
    #     interval_range_s=(10.0, 15.0),
    #     params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    # )


@configclass
class HopfCurriculumCfg:
    # terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
    lin_vel_cmd_levels = CurrTerm(mdp.lin_vel_cmd_levels)


@configclass
class RobotHopfEnvCfg(RobotEnvCfg):
    # 观测、动作、指令配置
    observations: HopfObservationsCfg = HopfObservationsCfg()
    commands: HopfCommandsCfg = HopfCommandsCfg()
    # MDP相关配置
    rewards: HopfRewardsCfg = HopfRewardsCfg()
    events: HopfEventCfg = HopfEventCfg()
    curriculum: HopfCurriculumCfg = HopfCurriculumCfg()


@configclass
class RobotHopfPlayEnvCfg(RobotHopfEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 8
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
        # self.commands.base_velocity.ranges = mdp.UniformLevelVelocityCommandCfg.Ranges(
        #     lin_vel_x=(0.0, 0.6), lin_vel_y=(-0.0, 0.0), ang_vel_z=(-0.0, 0.0)
        # ),


        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 2
            self.scene.terrain.terrain_generator.num_cols = 10
