from isaaclab.utils import configclass
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
import whole_body_tracking.tasks.tracking.mdp as mdp

from whole_body_tracking.robots.hi import HI_CFG, HI_ACTION_SCALE
from whole_body_tracking.tasks.tracking.config.hi.agents.rsl_rl_ppo_cfg import LOW_FREQ_SCALE
from whole_body_tracking.tasks.tracking.tracking_env_cfg import TrackingEnvCfg



@configclass
class HiFlatEnvCfg(TrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        
        self.scene.robot = HI_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = HI_ACTION_SCALE
        
        # 指定参与运动跟踪的关节（排除锁定的头部关节）
        self.actions.joint_pos.joint_names = [
            "l_hip_pitch_joint",
            "l_hip_roll_joint", 
            "l_hip_thigh_joint",
            "l_hip_calf_joint",
            "l_ankle_pitch_joint",
            "l_ankle_roll_joint",
            "r_hip_pitch_joint",
            "r_hip_roll_joint",
            "r_hip_thigh_joint", 
            "r_hip_calf_joint",
            "r_ankle_pitch_joint",
            "r_ankle_roll_joint",
            "waist_yaw_joint",
            "l_shoulder_pitch_joint",
            "l_shoulder_roll_joint",
            "l_upper_arm_joint",
            "l_elbow_joint",
            "l_wrist_joint",
            "r_shoulder_pitch_joint",
            "r_shoulder_roll_joint",
            "r_upper_arm_joint",
            "r_elbow_joint",
            "r_wrist_joint",
        ]
        
        # motion command也需要知道参与跟踪的关节（与动作空间一致）
        self.commands.motion.joint_names = [
            "l_hip_pitch_joint",
            "l_hip_roll_joint", 
            "l_hip_thigh_joint",
            "l_hip_calf_joint",
            "l_ankle_pitch_joint",
            "l_ankle_roll_joint",
            "r_hip_pitch_joint",
            "r_hip_roll_joint",
            "r_hip_thigh_joint", 
            "r_hip_calf_joint",
            "r_ankle_pitch_joint",
            "r_ankle_roll_joint",
            "waist_yaw_joint",
            "l_shoulder_pitch_joint",
            "l_shoulder_roll_joint",
            "l_upper_arm_joint",
            "l_elbow_joint",
            "l_wrist_joint",
            "r_shoulder_pitch_joint",
            "r_shoulder_roll_joint",
            "r_upper_arm_joint",
            "r_elbow_joint",
            "r_wrist_joint",
        ]
        
        self.commands.motion.anchor_body_name = "waist_yaw_link"
        self.commands.motion.body_names = [
            "base_link",
            "l_hip_roll_link",
            "l_hip_calf_link",
            "l_ankle_roll_link",
            "r_hip_roll_link",
            "r_hip_calf_link",
            "r_ankle_roll_link",

            "waist_yaw_link",

            "l_shoulder_roll_link",
            "l_elbow_link",
            "l_wrist_link",
            "r_shoulder_roll_link",
            "r_elbow_link",
            "r_wrist_link",
        ]
        
        # 相机设置：自由视角，不跟随机器人
        self.viewer.eye = (3.0, 3.0, 2.0)  # 相机位置
        self.viewer.lookat = (0.0, 0.0, 1.0)  # 看向位置
        self.viewer.origin_type = "world"  # 世界坐标系，不跟随机器人
        self.viewer.asset_name = None  # 不绑定到特定资产
        
        # 关闭调试可视化显示
        self.commands.motion.debug_vis = False  # 关闭motion命令的调试可视化
        self.scene.contact_forces.debug_vis = False  # 关闭接触力可视化
        
        # HI机器人专用奖励权重调整
        # 增强位置跟踪奖励（HI机器人惯性更大，需要更强的引导）
        self.rewards.motion_global_anchor_pos.weight = 0.8  # 从0.5增加到0.8
        self.rewards.motion_body_pos.weight = 1.5  # 从1.0增加到1.5
        
        # 减少动作惩罚（HI机器人需要更大的动作幅度完成舞蹈）
        self.rewards.action_rate_l2.weight = -5e-2  # 从-1e-1减少到-5e-2
        
        # 调整速度跟踪权重（适应HI机器人的动力学特性）
        self.rewards.motion_body_lin_vel.weight = 0.8  # 从1.0降到0.8，减少对线速度的严格要求
        self.rewards.motion_body_ang_vel.weight = 0.6  # 从1.0降到0.6，减少对角速度的严格要求
        
        # 保持训练时的默认设置（自适应采样等）
        # 如需演示模式，请使用 Tracking-Flat-Hi-Play-v0
        
        # HI机器人专用termination调整
        # 1. 放宽anchor位置偏差阈值（HI机器人质量大，控制精度相对较低）
        self.terminations.anchor_pos.params["threshold"] = 0.35  # 从0.25增加到0.35
        
        # 2. 放宽anchor方向偏差阈值（适应HI机器人的动力学特性）
        self.terminations.anchor_ori.params["threshold"] = 0.7  # 从0.8降低到0.7（更严格，因为HI机器人倾倒风险更高）
        
        # 3. 更新末端执行器body_names为HI机器人的对应链接
        self.terminations.ee_body_pos.params["body_names"] = [
            "l_ankle_roll_link",  # HI机器人的脚踝
            "r_ankle_roll_link", 
            "l_wrist_link",      # HI机器人的手腕（注意名称不同）
            "r_wrist_link",
        ]
        # 4. 适当放宽末端执行器位置偏差阈值
        self.terminations.ee_body_pos.params["threshold"] = 0.3  # 从0.25增加到0.3
        
        # 5. 更新不期望接触的body名称，排除HI机器人的末端执行器
        self.rewards.undesired_contacts.params["sensor_cfg"] = SceneEntityCfg(
            "contact_forces",
            body_names=[
                r"^(?!l_ankle_roll_link$)(?!r_ankle_roll_link$)(?!l_wrist_link$)(?!r_wrist_link$).+$"
            ],
        )
        
        # 修复base_com事件配置，使用base_link而不是torso_link
        self.events.base_com = EventTerm(
            func=mdp.randomize_rigid_body_com,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="waist_yaw_link"),
                "com_range": {"x": (-0.025, 0.025), "y": (-0.05, 0.05), "z": (-0.05, 0.05)},
            },
        )

@configclass
class HiFlatWoStateEstimationEnvCfg(HiFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.motion_anchor_pos_b = None
        self.observations.policy.base_lin_vel = None


@configclass
class HiFlatLowFreqEnvCfg(HiFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.decimation = round(self.decimation / LOW_FREQ_SCALE)
        self.rewards.action_rate_l2.weight *= LOW_FREQ_SCALE