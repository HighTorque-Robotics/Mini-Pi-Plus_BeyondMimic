from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Literal, Callable

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.envs.mdp.events import _randomize_prop_by_op, push_by_setting_velocity
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def randomize_joint_default_pos(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    pos_distribution_params: tuple[float, float] | None = None,
    operation: Literal["add", "scale", "abs"] = "abs",
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """
    Randomize the joint default positions which may be different from URDF due to calibration errors.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # save nominal value for export
    asset.data.default_joint_pos_nominal = torch.clone(asset.data.default_joint_pos[0])

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    # resolve joint indices
    if asset_cfg.joint_ids == slice(None):
        joint_ids = slice(None)  # for optimization purposes
    else:
        joint_ids = torch.tensor(asset_cfg.joint_ids, dtype=torch.int, device=asset.device)

    if pos_distribution_params is not None:
        pos = asset.data.default_joint_pos.to(asset.device).clone()
        pos = _randomize_prop_by_op(
            pos, pos_distribution_params, env_ids, joint_ids, operation=operation, distribution=distribution
        )[env_ids][:, joint_ids]

        if env_ids != slice(None) and joint_ids != slice(None):
            env_ids = env_ids[:, None]
        asset.data.default_joint_pos[env_ids, joint_ids] = pos
        # update the offset in action since it is not updated automatically
        env.action_manager.get_term("joint_pos")._offset[env_ids, joint_ids] = pos


def randomize_rigid_body_com(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    com_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg,
):
    """Randomize the center of mass (CoM) of rigid bodies by adding a random value sampled from the given ranges.

    .. note::
        This function uses CPU tensors to assign the CoM. It is recommended to use this function
        only during the initialization of the environment.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    # resolve body indices
    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")

    # sample random CoM values
    range_list = [com_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
    ranges = torch.tensor(range_list, device="cpu")
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 3), device="cpu").unsqueeze(1)

    # get the current com of the bodies (num_assets, num_bodies)
    coms = asset.root_physx_view.get_coms().clone()

    # Randomize the com in range
    coms[:, body_ids, :3] += rand_samples

    # Set the new coms
    asset.root_physx_view.set_coms(coms, env_ids)


def conditional_push_by_setting_velocity(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    velocity_range: dict[str, tuple[float, float]],
    condition_func: Callable[[ManagerBasedEnv, torch.Tensor], torch.Tensor] | None = None,
    condition_params: dict | None = None,
):
    """有条件地推动机器人，只对满足特定条件的环境执行推动。
    
    这个函数是 `push_by_setting_velocity` 的扩展版本，允许根据自定义条件
    选择性地对环境执行推动操作。
    
    Args:
        env: 环境实例
        env_ids: 要考虑推动的环境ID。如果为None，则考虑所有环境
        velocity_range: 推动速度范围字典，包含线速度和角速度的范围
        condition_func: 条件判断函数，接收(env, env_ids)参数，返回布尔张量
        condition_params: 传递给条件函数的额外参数
        
    Returns:
        None
        
    Examples:
        # 示例1: 只推动机器人高度低于某个阈值的环境
        def height_condition(env, env_ids):
            robot = env.scene["robot"]
            base_pos = robot.data.root_pos_w[env_ids, 2]  # Z坐标
            return base_pos < 0.8  # 高度低于0.8m
            
        # 示例2: 只推动机器人速度低于某个阈值的环境  
        def velocity_condition(env, env_ids):
            robot = env.scene["robot"]
            base_vel = torch.norm(robot.data.root_lin_vel_w[env_ids, :2], dim=1)  # XY平面速度
            return base_vel < 0.5  # 速度低于0.5m/s
            
        # 示例3: 只推动跟踪误差大于某个阈值的环境
        def tracking_error_condition(env, env_ids):
            command_manager = env.command_manager
            motion_command = command_manager.get_command("motion")
            # 计算跟踪误差（这里需要根据具体实现调整）
            error = compute_tracking_error(env, env_ids, motion_command)
            return error > 0.3  # 误差大于0.3
    """
    # 如果没有指定环境ID，则考虑所有环境
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=env.device)
    
    # 如果没有条件函数，则对所有指定环境执行推动
    if condition_func is None:
        push_env_ids = env_ids
    else:
        # 应用条件函数筛选环境
        if condition_params is None:
            condition_params = {}
        
        # 调用条件函数获取满足条件的环境掩码
        condition_mask = condition_func(env, env_ids, **condition_params)
        
        # 筛选出满足条件的环境ID
        push_env_ids = env_ids[condition_mask]
    
    # 如果有满足条件的环境，则执行推动
    if len(push_env_ids) > 0:
        push_by_setting_velocity(env, push_env_ids, velocity_range)


# 预定义的条件函数示例

def height_based_condition(
    env: ManagerBasedEnv, 
    env_ids: torch.Tensor, 
    height_threshold: float = 0.8,
    below_threshold: bool = True
) -> torch.Tensor:
    """基于机器人高度的条件函数
    
    Args:
        env: 环境实例
        env_ids: 环境ID
        height_threshold: 高度阈值
        below_threshold: True表示高度低于阈值时推动，False表示高度高于阈值时推动
        
    Returns:
        满足条件的环境掩码
    """
    robot = env.scene["robot"]
    base_height = robot.data.root_pos_w[env_ids, 2]
    
    if below_threshold:
        return base_height < height_threshold
    else:
        return base_height > height_threshold


def velocity_based_condition(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    velocity_threshold: float = 0.5,
    below_threshold: bool = True
) -> torch.Tensor:
    """基于机器人速度的条件函数
    
    Args:
        env: 环境实例
        env_ids: 环境ID
        velocity_threshold: 速度阈值
        below_threshold: True表示速度低于阈值时推动，False表示速度高于阈值时推动
        
    Returns:
        满足条件的环境掩码
    """
    robot = env.scene["robot"]
    base_vel_magnitude = torch.norm(robot.data.root_lin_vel_w[env_ids, :2], dim=1)
    
    if below_threshold:
        return base_vel_magnitude < velocity_threshold
    else:
        return base_vel_magnitude > velocity_threshold

def ori_error_condition(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    command_name: str = "motion",
    error_threshold: float = 0.3,
    above_threshold: bool = True
) -> torch.Tensor:
    """基于方向误差的条件函数
    
    Args:
        env: 环境实例
        env_ids: 环境ID
        command_name: 运动命令名称
        error_threshold: 误差阈值
        above_threshold: True表示误差大于阈值时推动，False表示误差小于阈值时推动
        
    Returns:
        满足条件的环境掩码
    """
    # 获取运动命令
    command_manager = env.command_manager
    motion_command = command_manager.get_term(command_name)
    
    # 获取机器人当前状态
    ori_error = motion_command.ori_error[env_ids]
    
    if above_threshold:
        return ori_error > error_threshold
    else:
        return ori_error < error_threshold


def tracking_error_condition(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    command_name: str = "motion",
    error_threshold: float = 0.3,
    above_threshold: bool = True
) -> torch.Tensor:
    """基于跟踪误差的条件函数
    
    Args:
        env: 环境实例
        env_ids: 环境ID
        command_name: 运动命令名称
        error_threshold: 误差阈值
        above_threshold: True表示误差大于阈值时推动，False表示误差小于阈值时推动
        
    Returns:
        满足条件的环境掩码
    """
    # 获取运动命令
    command_manager = env.command_manager
    motion_command = command_manager.get_command(command_name)
    
    # 获取机器人当前状态
    robot = env.scene["robot"]
    current_pos = robot.data.root_pos_w[env_ids]
    
    # 获取目标位置（这里简化为锚点位置，实际可能需要更复杂的计算）
    target_pos = motion_command.anchor_pos_w[env_ids]
    
    # 计算位置误差
    pos_error = torch.norm(current_pos - target_pos, dim=1)
    
    if above_threshold:
        return pos_error > error_threshold
    else:
        return pos_error < error_threshold


def random_condition(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    probability: float = 0.5
) -> torch.Tensor:
    """随机条件函数，以指定概率选择环境
    
    Args:
        env: 环境实例
        env_ids: 环境ID
        probability: 选择概率
        
    Returns:
        满足条件的环境掩码
    """
    return torch.rand(len(env_ids), device=env.device) < probability


def update_force_curriculum(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    command_name: str = "motion",
    update_interval: int = 10,
    force_reduction_rate: float = 10.0,
    min_force: float = 0.0,
    max_force: float = 500.0,
    standing_base_force: float = 50.0,
    ori_threshold: float = 0.3,
) -> None:
    """力课程学习函数 - 根据机器人表现动态调整辅助力
    
    这个函数作为课程学习项被调用，根据机器人的跟踪表现动态调整辅助力：
    - 对于表现良好的非站立环境，减少辅助力
    - 对于表现不佳的环境，保持或增加辅助力
    
    Args:
        env: 环境实例
        env_ids: 环境ID（通常为所有环境）
        command_name: 运动命令名称
        update_interval: 更新间隔（步数）
        force_reduction_rate: 力减少率
        min_force: 最小辅助力
        max_force: 最大辅助力
        standing_base_force: 站立环境基础力
        ori_threshold: 方向误差阈值
    """
    # 获取运动命令
    command_manager = env.command_manager
    motion_command = command_manager.get_term(command_name)
    
    # 检查是否需要更新（基于步数间隔）
    # if not hasattr(motion_command, '_force_curriculum_counter'):
    #     motion_command._force_curriculum_counter = 0
    
    # motion_command._force_curriculum_counter += 1
    
    # # 只在指定间隔更新
    # if motion_command._force_curriculum_counter % update_interval != 0:
    #     return
    
    # 调用 MotionCommand 的 update_force_curriculum 方法
    motion_command.update_force_curriculum(env_ids)
   
