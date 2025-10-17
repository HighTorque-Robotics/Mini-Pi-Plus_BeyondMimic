from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

from whole_body_tracking.tasks.tracking.mdp.commands import MotionCommand
from whole_body_tracking.tasks.tracking.mdp.rewards import _get_body_indexes


def bad_anchor_pos(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    return torch.norm(command.anchor_pos_w - command.robot_anchor_pos_w, dim=1) > threshold


def bad_anchor_pos_z_only(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    return torch.abs(command.anchor_pos_w[:, -1] - command.robot_anchor_pos_w[:, -1]) > threshold


def bad_anchor_ori(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, command_name: str, threshold: float
) -> torch.Tensor:
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    command: MotionCommand = env.command_manager.get_term(command_name)
    motion_projected_gravity_b = math_utils.quat_rotate_inverse(command.anchor_quat_w, asset.data.GRAVITY_VEC_W)

    robot_projected_gravity_b = math_utils.quat_rotate_inverse(command.robot_anchor_quat_w, asset.data.GRAVITY_VEC_W)

    return (motion_projected_gravity_b[:, 2] - robot_projected_gravity_b[:, 2]).abs() > threshold


def bad_motion_body_pos(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    body_indexes = _get_body_indexes(command, body_names)
    error = torch.norm(command.body_pos_relative_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes], dim=-1)
    return torch.any(error > threshold, dim=-1)


def bad_motion_body_pos_z_only(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    body_indexes = _get_body_indexes(command, body_names)
    error = torch.abs(command.body_pos_relative_w[:, body_indexes, -1] - command.robot_body_pos_w[:, body_indexes, -1])
    return torch.any(error > threshold, dim=-1)

def bad_anchor_pos_z_only_time(env: ManagerBasedRLEnv, command_name: str, threshold: float,timeout_threshold:float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    reset=torch.abs(command.anchor_pos_w[:, -1] - command.robot_anchor_pos_w[:, -1]) > threshold
    command.bad_pos_starttime[~reset] = env.episode_length_buf[~reset].clone().float()
    command.getup_timeout=env.episode_length_buf - command.bad_pos_starttime > timeout_threshold
    command.bad_pos_starttime[reset&command.getup_timeout]=0.0
    command.bad_pos_starttime[env.episode_length_buf >= env.max_episode_length]=0.0
    under_force=command.force.squeeze(-1)<20
    return reset&command.getup_timeout&under_force

def bad_anchor_pos_z_only_condition(env: ManagerBasedRLEnv, command_name: str, threshold: float,timeout_threshold:float,condition_threshold:float) -> torch.Tensor:
    
    
    command: MotionCommand = env.command_manager.get_term(command_name)
    condition_good_pro=command.anchor_conditions_good.float().mean()
    
    reset=torch.abs(command.anchor_pos_w[:, -1] - command.robot_anchor_pos_w[:, -1]) > threshold
    command.bad_pos_starttime[~reset] = env.episode_length_buf[~reset].clone().float()
    command.getup_time=env.episode_length_buf - command.bad_pos_starttime
    command.getup_timeout=command.getup_time > timeout_threshold
    command.bad_pos_starttime[env.episode_length_buf >= env.max_episode_length]=0.0
    if condition_good_pro < condition_threshold:
        return command.not_reach_anchor_conditions_good
    command.bad_pos_starttime[reset&command.getup_timeout]=0.0
    
    return reset&command.getup_timeout

def bad_anchor_pos_z_only_falltest_condition(env: ManagerBasedRLEnv, command_name: str, threshold: float,timeout_threshold:float,condition_threshold:float) -> torch.Tensor:
    
    
    command: MotionCommand = env.command_manager.get_term(command_name)
    condition_good_pro=command.anchor_conditions_good.float().mean()
    
    command.bad_pos_starttime[command.first_fall] = env.episode_length_buf[command.first_fall].clone().float()
    command.first_fall[command.robot_anchor_pos_w[:, -1]<0.12]=False
    command.first_fall[command.anchor_conditions_good]=True
    command.getup_time=env.episode_length_buf - command.bad_pos_starttime
    command.getup_timeout=command.getup_time > timeout_threshold
    command.bad_pos_starttime[env.episode_length_buf >= env.max_episode_length]=0.0
    if condition_good_pro < condition_threshold:
        return command.not_reach_anchor_conditions_good
    command.bad_pos_starttime[command.getup_timeout]=0.0
    
    return command.getup_timeout