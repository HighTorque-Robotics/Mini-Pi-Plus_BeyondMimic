import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from whole_body_tracking.assets import ASSET_DIR

ARMATURE_4438 = 0.008419
ARMATURE_5047 = 0.044277
ARMATURE_6056 = 0.066666

NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10Hz
DAMPING_RATIO = 2.0

STIFFNESS_4438 = 50
STIFFNESS_5047 = 65
STIFFNESS_6056 = 80

DAMPING_4438 = 1.1
DAMPING_5047 = 1.5
DAMPING_6056 = 1.1


HI_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        replace_cylinders_with_capsules=True,
        asset_path=f"{ASSET_DIR}/hightorque/hi/urdf/hi_25dof.urdf",
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
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.67),
        joint_pos={
            ".*_hip_pitch_joint": -0.25,
            ".*_calf_joint": 0.65,
            ".*_ankle_pitch_joint": -0.4,
            ".*_elbow_joint": 0.0,
            "l_shoulder_roll_joint": 0.0,
            "l_shoulder_pitch_joint": 0.0,
            "r_shoulder_roll_joint": 0.0,
            "r_shoulder_pitch_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_thigh_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_hip_calf_joint",
            ],
            effort_limit_sim={
                ".*_hip_thigh_joint": 30.0,
                ".*_hip_roll_joint": 30.0,
                ".*_hip_pitch_joint": 30.0,
                ".*_hip_calf_joint": 30.0,
            },
            velocity_limit_sim={
                ".*_hip_thigh_joint": 4.2,
                ".*_hip_roll_joint": 4.2,
                ".*_hip_pitch_joint": 4.2,
                ".*_hip_calf_joint": 4.2,
            },
            stiffness={
                ".*_hip_pitch_joint": STIFFNESS_6056,
                ".*_hip_roll_joint": STIFFNESS_6056,
                ".*_hip_thigh_joint": STIFFNESS_6056,
                ".*_hip_calf_joint": STIFFNESS_6056,
            },
            damping={
                ".*_hip_pitch_joint": DAMPING_6056,
                ".*_hip_roll_joint": DAMPING_6056,
                ".*_hip_thigh_joint": DAMPING_6056,
                ".*_hip_calf_joint": DAMPING_6056,
            },
            armature={
                ".*_hip_pitch_joint": ARMATURE_6056,
                ".*_hip_roll_joint": ARMATURE_6056,
                ".*_hip_thigh_joint": ARMATURE_6056,
                ".*_hip_calf_joint": ARMATURE_6056,
            },
        ),
        "feet": ImplicitActuatorCfg(
            effort_limit_sim=30.0,
            velocity_limit_sim=7.0,
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            stiffness=STIFFNESS_6056,
            damping=DAMPING_6056,
            armature=ARMATURE_6056,
        ),
        "waist_yaw": ImplicitActuatorCfg(
            effort_limit_sim=30,
            velocity_limit_sim=32.0,
            joint_names_expr=["waist_yaw_joint"],
            stiffness=STIFFNESS_6056,
            damping=DAMPING_6056,
            armature=ARMATURE_6056,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_upper_arm_joint",
                ".*_elbow_joint",
                ".*_wrist_joint",
            ],
            effort_limit_sim={
                ".*_shoulder_pitch_joint": 21.0,
                ".*_shoulder_roll_joint": 21.0,
                ".*_upper_arm_joint": 21.0,
                ".*_elbow_joint": 21.0,
                ".*_wrist_joint": 10.0,
            },
            velocity_limit_sim={
                ".*_shoulder_pitch_joint": 4.2,
                ".*_shoulder_roll_joint": 4.2,
                ".*_upper_arm_joint": 4.2,
                ".*_elbow_joint": 4.2,
                ".*_wrist_joint": 5.0,
            },
            stiffness={
                ".*_shoulder_pitch_joint": STIFFNESS_5047,
                ".*_shoulder_roll_joint": STIFFNESS_5047,
                ".*_upper_arm_joint": STIFFNESS_5047,
                ".*_elbow_joint": STIFFNESS_5047,
                ".*_wrist_joint": STIFFNESS_4438,
            },
            damping={
                ".*_shoulder_pitch_joint": DAMPING_5047,
                ".*_shoulder_roll_joint": DAMPING_5047,
                ".*_upper_arm_joint": DAMPING_5047,
                ".*_elbow_joint": DAMPING_5047,
                ".*_wrist_joint": DAMPING_4438,
            },
            armature={
                ".*_shoulder_pitch_joint": ARMATURE_5047,
                ".*_shoulder_roll_joint": ARMATURE_5047,
                ".*_upper_arm_joint": ARMATURE_5047,
                ".*_elbow_joint": ARMATURE_5047,
                ".*_wrist_joint": ARMATURE_4438,
            },
        ),
    },
)

HI_ACTION_SCALE = {}
for a in HI_CFG.actuators.values():
    e = a.effort_limit_sim
    s = a.stiffness
    names = a.joint_names_expr
    if not isinstance(e, dict):
        e = {n: e for n in names}
    if not isinstance(s, dict):
        s = {n: s for n in names}
    for n in names:
        if n in e and n in s and s[n]:
            HI_ACTION_SCALE[n] = 0.25 * e[n] / s[n]
