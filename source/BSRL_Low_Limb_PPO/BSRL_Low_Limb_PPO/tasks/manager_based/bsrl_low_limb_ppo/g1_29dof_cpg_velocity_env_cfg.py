"""CPG velocity task adapted to the 29-DoF Unitree G1."""

from isaaclab.assets import ArticulationCfg
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.utils import configclass

from BSRL_Low_Limb_PPO.assets.G1.unitree import UNITREE_G1_29DOF_CFG
from BSRL_Low_Limb_PPO.tasks.manager_based.bsrl_low_limb_ppo import mdp
from BSRL_Low_Limb_PPO.tasks.manager_based.bsrl_low_limb_ppo.bsrl_baseline_velocity_env_cfg import RobotSceneCfg
from BSRL_Low_Limb_PPO.tasks.manager_based.bsrl_low_limb_ppo.bsrl_cpg_velocity_env_cfg import (
    CPGRewardsCfg,
    RobotCPGEnvCfg,
)


G1_FOOT_NAMES = ["left_ankle_roll_link", "right_ankle_roll_link"]
G1_CPG_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "left_knee_joint",
    "right_hip_pitch_joint",
    "right_knee_joint",
)
G1_DEFAULT_ROOT_HEIGHT = 0.8


@configclass
class G1RobotSceneCfg(RobotSceneCfg):
    robot: ArticulationCfg = UNITREE_G1_29DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )


@configclass
class G1CPGRewardsCfg(CPGRewardsCfg):
    base_height = RewTerm(
        func=mdp.base_height_l2,
        weight=-10.0,
        params={
            "target_height": G1_DEFAULT_ROOT_HEIGHT,
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("height_scanner"),
        },
    )
    joint_deviation_hips = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_roll_joint", ".*_hip_yaw_joint"])},
    )
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=[".*_shoulder_.*_joint", ".*_elbow_joint", ".*_wrist_.*_joint"]
            )
        },
    )
    joint_deviation_waist = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["waist_.*_joint"])},
    )
    cpg_tracking = RewTerm(
        func=mdp.cpg_joint_tracking,
        weight=0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "command_name": "base_velocity",
            "command_threshold": 0.05,
            "joint_names": G1_CPG_JOINT_NAMES,
        },
    )
    feet_double_support = RewTerm(
        func=mdp.long_double_support_penalty,
        weight=-0.5,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=G1_FOOT_NAMES),
            "command_threshold": 0.1,
            "allowed_time": 0.1,
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.4,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=G1_FOOT_NAMES),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=G1_FOOT_NAMES),
        },
    )
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=1.0,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=G1_FOOT_NAMES),
            "threshold": 0.5,
        },
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "threshold": 1,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["^(?!.*ankle_roll_link$).*"]),
        },
    )


@configclass
class G1CPGEnvCfg(RobotCPGEnvCfg):
    scene: G1RobotSceneCfg = G1RobotSceneCfg(num_envs=4096, env_spacing=2.5)
    rewards: G1CPGRewardsCfg = G1CPGRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        self.terminations.base_contact.params["sensor_cfg"] = SceneEntityCfg(
            "contact_forces", body_names="torso_link"
        )
        self.terminations.base_height.params["minimum_height"] = 0.2
        for event_name in ("add_base_mass", "base_com", "base_external_force_torque"):
            getattr(self.events, event_name).params["asset_cfg"] = SceneEntityCfg("robot", body_names="torso_link")


@configclass
class G1CPGPlayEnvCfg(G1CPGEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.commands.base_velocity.ranges = mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 1.5), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0)
        )
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 2
            self.scene.terrain.terrain_generator.num_cols = 10
