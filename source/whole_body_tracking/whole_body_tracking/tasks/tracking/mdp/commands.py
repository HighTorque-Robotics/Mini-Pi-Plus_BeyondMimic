from __future__ import annotations

import math
import numpy as np
import os
import torch
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    quat_apply,
    quat_error_magnitude,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
    sample_uniform,
    yaw_quat,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class MotionLoader:
    def __init__(self, motion_file: str, body_indexes: Sequence[int], device: str = "cpu"):
        assert os.path.isfile(motion_file), f"Invalid file path: {motion_file}"
        data = np.load(motion_file)
        self.fps = data["fps"]
        self.joint_pos = torch.tensor(data["joint_pos"], dtype=torch.float32, device=device)
        self.joint_vel = torch.tensor(data["joint_vel"], dtype=torch.float32, device=device)
        self._body_pos_w = torch.tensor(data["body_pos_w"], dtype=torch.float32, device=device)
        self._body_quat_w = torch.tensor(data["body_quat_w"], dtype=torch.float32, device=device)
        self._body_lin_vel_w = torch.tensor(data["body_lin_vel_w"], dtype=torch.float32, device=device)
        self._body_ang_vel_w = torch.tensor(data["body_ang_vel_w"], dtype=torch.float32, device=device)
        self._body_indexes = body_indexes
        self.time_step_total = self.joint_pos.shape[0]

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self._body_pos_w[:, self._body_indexes]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self._body_quat_w[:, self._body_indexes]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self._body_lin_vel_w[:, self._body_indexes]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self._body_ang_vel_w[:, self._body_indexes]


class MotionCommand(CommandTerm):
    cfg: MotionCommandCfg

    def __init__(self, cfg: MotionCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]
        self.robot_anchor_body_index = self.robot.body_names.index(self.cfg.anchor_body_name)
        self.motion_anchor_body_index = self.cfg.body_names.index(self.cfg.anchor_body_name)
        self.body_indexes = torch.tensor(
            self.robot.find_bodies(self.cfg.body_names, preserve_order=True)[0], dtype=torch.long, device=self.device
        )

        self.motion = MotionLoader(self.cfg.motion_file, self.body_indexes, device=self.device)
        self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.body_pos_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 3, device=self.device)
        self.body_quat_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 4, device=self.device)
        self.body_quat_relative_w[:, :, 0] = 1.0

        self.bin_count = int(self.motion.time_step_total // (1 / (env.cfg.decimation * env.cfg.sim.dt))) + 1
        self.bin_failed_count = torch.zeros(self.bin_count, dtype=torch.float, device=self.device)
        self._current_bin_failed = torch.zeros(self.bin_count, dtype=torch.float, device=self.device)
        self.kernel = torch.tensor(
            [self.cfg.adaptive_lambda**i for i in range(self.cfg.adaptive_kernel_size)], device=self.device
        )
        self.kernel = self.kernel / self.kernel.sum()

        self.metrics["error_anchor_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_lin_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_ang_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_entropy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_top1_prob"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_top1_bin"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["ori_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["head_height"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["projected_gravity_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["anchor_conditions_good"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["force"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["force1"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["force2"] = torch.zeros(self.num_envs, device=self.device)



        
        self.anchor_conditions_good = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.not_reach_anchor_conditions_good = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.ori_error = torch.zeros(self.num_envs, device=self.device)
        self.last_base_height = torch.zeros(self.num_envs, device=self.device)
        self.last_feet_contact_forces = torch.zeros(self.num_envs, device=self.device)
        self.force=self.cfg.force * torch.ones(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False).unsqueeze(1)
        self.ang_pitch=self.cfg.ang_pitch * torch.ones(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False).unsqueeze(1)
        self.robot_projected_gravity_w=quat_apply(self.robot_anchor_quat_w, self.robot.data.GRAVITY_VEC_W)
        self.target_gravity_w=torch.tensor([0, 0 ,-1], device=self.device)
        self.head_height = torch.zeros(self.num_envs, device=self.device)
        self.projected_gravity_error_w = torch.zeros(self.num_envs, device=self.device)
        self.bad_pos_starttime = torch.zeros(self.num_envs, device=self.device)
        self.getup_timeout=torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.first_fall=torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

        self.getup_time=torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:  # TODO Consider again if this is the best observation
        return torch.cat([self.joint_pos, self.joint_vel], dim=1)

    @property
    def joint_pos(self) -> torch.Tensor:
        return self.motion.joint_pos[self.time_steps]

    @property
    def joint_vel(self) -> torch.Tensor:
        return self.motion.joint_vel[self.time_steps]

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self.motion.body_pos_w[self.time_steps] + self._env.scene.env_origins[:, None, :]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self.motion.body_quat_w[self.time_steps]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self.motion.body_lin_vel_w[self.time_steps]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self.motion.body_ang_vel_w[self.time_steps]

    @property
    def anchor_pos_w(self) -> torch.Tensor:
        if hasattr(self._env, 'scene') and hasattr(self._env.scene, 'env_origins'):
            return self.motion.body_pos_w[self.time_steps, self.motion_anchor_body_index] + self._env.scene.env_origins
        else:
            return self.motion.body_pos_w[self.time_steps, self.motion_anchor_body_index]

    @property
    def anchor_quat_w(self) -> torch.Tensor:
        return self.motion.body_quat_w[self.time_steps, self.motion_anchor_body_index]

    @property
    def anchor_lin_vel_w(self) -> torch.Tensor:
        return self.motion.body_lin_vel_w[self.time_steps, self.motion_anchor_body_index]

    @property
    def anchor_ang_vel_w(self) -> torch.Tensor:
        return self.motion.body_ang_vel_w[self.time_steps, self.motion_anchor_body_index]

    @property
    def robot_joint_pos(self) -> torch.Tensor:
        return self.robot.data.joint_pos

    @property
    def robot_joint_vel(self) -> torch.Tensor:
        return self.robot.data.joint_vel

    @property
    def robot_body_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.body_indexes]

    @property
    def robot_body_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.body_indexes]

    @property
    def robot_body_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.body_indexes]

    @property
    def robot_body_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.body_indexes]

    @property
    def robot_anchor_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.robot_anchor_body_index]

    def _update_metrics(self):
        self.metrics["error_anchor_pos"] = torch.norm(self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1)
        self.metrics["error_anchor_rot"] = quat_error_magnitude(self.anchor_quat_w, self.robot_anchor_quat_w)
        self.metrics["error_anchor_lin_vel"] = torch.norm(self.anchor_lin_vel_w - self.robot_anchor_lin_vel_w, dim=-1)
        self.metrics["error_anchor_ang_vel"] = torch.norm(self.anchor_ang_vel_w - self.robot_anchor_ang_vel_w, dim=-1)

        self.metrics["error_body_pos"] = torch.norm(self.body_pos_relative_w - self.robot_body_pos_w, dim=-1).mean(
            dim=-1
        )
        self.metrics["error_body_rot"] = quat_error_magnitude(self.body_quat_relative_w, self.robot_body_quat_w).mean(
            dim=-1
        )

        self.metrics["error_body_lin_vel"] = torch.norm(self.body_lin_vel_w - self.robot_body_lin_vel_w, dim=-1).mean(
            dim=-1
        )
        self.metrics["error_body_ang_vel"] = torch.norm(self.body_ang_vel_w - self.robot_body_ang_vel_w, dim=-1).mean(
            dim=-1
        )

        self.metrics["error_joint_pos"] = torch.norm(self.joint_pos - self.robot_joint_pos, dim=-1)
        self.metrics["error_joint_vel"] = torch.norm(self.joint_vel - self.robot_joint_vel, dim=-1)
        
        pos_error = torch.abs(self.anchor_pos_w[:, -1] - self.robot_anchor_pos_w[:, -1])
        pos_good = pos_error <= self.cfg.anchor_pos_threshold
        
        motion_projected_gravity_b = quat_apply(quat_inv(self.anchor_quat_w), self.robot.data.GRAVITY_VEC_W)
        robot_projected_gravity_b = quat_apply(quat_inv(self.robot_anchor_quat_w), self.robot.data.GRAVITY_VEC_W)
        self.robot_projected_gravity_w=quat_apply(self.robot_anchor_quat_w, self.robot.data.GRAVITY_VEC_W)
        self.ori_error = (motion_projected_gravity_b[:, 2] - robot_projected_gravity_b[:, 2]).abs()
        ori_good = self.ori_error <= self.cfg.anchor_ori_threshold
        
        self.anchor_conditions_good = pos_good & ori_good
        self.metrics["anchor_conditions_good"][:] = self.anchor_conditions_good.float()
        self.metrics["ori_error"][:] = self.ori_error
        self.metrics["force"][:] = self.force.squeeze(-1)
        self.metrics["force1"][:self.num_envs//4] = self.force[:self.num_envs//4].squeeze(-1)
        self.metrics["force2"][self.num_envs//4:self.num_envs//2] = self.force[self.num_envs//4:self.num_envs//2].squeeze(-1)
        self.metrics["head_height"][:] = self.head_height
        self.metrics["projected_gravity_error"][:] = self.projected_gravity_error_w

    def _adaptive_sampling(self, env_ids: Sequence[int]):
        episode_failed = self._env.termination_manager.terminated[env_ids]
        if torch.any(episode_failed):
            current_bin_index = torch.clamp(
                (self.time_steps * self.bin_count) // max(self.motion.time_step_total, 1), 0, self.bin_count - 1
            )
            fail_bins = current_bin_index[env_ids][episode_failed]
            self._current_bin_failed[:] = torch.bincount(fail_bins, minlength=self.bin_count)

        # Sample
        sampling_probabilities = self.bin_failed_count + self.cfg.adaptive_uniform_ratio / float(self.bin_count)
        sampling_probabilities = torch.nn.functional.pad(
            sampling_probabilities.unsqueeze(0).unsqueeze(0),
            (0, self.cfg.adaptive_kernel_size - 1),  # Non-causal kernel
            mode="replicate",
        )
        sampling_probabilities = torch.nn.functional.conv1d(sampling_probabilities, self.kernel.view(1, 1, -1)).view(-1)

        sampling_probabilities = sampling_probabilities / sampling_probabilities.sum()

        sampled_bins = torch.multinomial(sampling_probabilities, len(env_ids), replacement=True)

        self.time_steps[env_ids] = (
            (sampled_bins + sample_uniform(0.0, 1.0, (len(env_ids),), device=self.device))
            / self.bin_count
            * (self.motion.time_step_total - 1)
        ).long()
        self.time_steps[env_ids] = (sampled_bins / self.bin_count * (self.motion.time_step_total - 1)).long()

        # Metrics
        H = -(sampling_probabilities * (sampling_probabilities + 1e-12).log()).sum()
        H_norm = H / math.log(self.bin_count)
        pmax, imax = sampling_probabilities.max(dim=0)
        self.metrics["sampling_entropy"][:] = H_norm
        self.metrics["sampling_top1_prob"][:] = pmax
        self.metrics["sampling_top1_bin"][:] = imax.float() / self.bin_count

    
    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return
        self._adaptive_sampling(env_ids)

        root_pos = self.body_pos_w[:, 0].clone()
        root_ori = self.body_quat_w[:, 0].clone()
        root_lin_vel = self.body_lin_vel_w[:, 0].clone()
        root_ang_vel = self.body_ang_vel_w[:, 0].clone()

        range_list = [self.cfg.pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_pos[env_ids] += rand_samples[:, 0:3]
        orientations_delta = quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
        root_ori[env_ids] = quat_mul(orientations_delta, root_ori[env_ids])
        range_list = [self.cfg.velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_lin_vel[env_ids] += rand_samples[:, :3]
        root_ang_vel[env_ids] += rand_samples[:, 3:]

        joint_pos = self.joint_pos.clone()
        joint_vel = self.joint_vel.clone()

        joint_pos += sample_uniform(*self.cfg.joint_position_range, joint_pos.shape, joint_pos.device)
        soft_joint_pos_limits = self.robot.data.soft_joint_pos_limits[env_ids]
        joint_pos[env_ids] = torch.clip(
            joint_pos[env_ids], soft_joint_pos_limits[:, :, 0], soft_joint_pos_limits[:, :, 1]
        )
        if self.cfg.force_curriculum_enabled:
        
            # 将机器人分为三组：站立(1/2)、伏卧(1/4)、仰卧(1/4)
            num_standing = int(self.num_envs * 0.5)
            num_prone = int(self.num_envs * 0.25)
            num_supine = self.num_envs - num_standing - num_prone
            
            # 将env_ids转换为tensor以便进行比较操作
            env_ids_tensor = torch.tensor(env_ids, device=self.device)
            
            # 创建状态标志
            standing_env_flag = env_ids_tensor < num_standing
            prone_env_flag = (env_ids_tensor >= num_standing) & (env_ids_tensor < num_standing + num_prone)
            supine_env_flag = env_ids_tensor >= num_standing + num_prone
            
            # 伏卧状态 (prone) - 趴着，面朝下
            if torch.any(prone_env_flag):
                prone_env_ids = env_ids_tensor[prone_env_flag]
                joint_pos[prone_env_ids] = 0.0
                joint_vel[prone_env_ids] = 0.0
                root_pos[prone_env_ids, 2] = 0.25  # 降低高度
                # 绕X轴旋转180度，面朝下
                prone_quat = quat_from_euler_xyz(
                    torch.zeros(len(prone_env_ids), device=self.device),
                    torch.tensor(-math.pi/2, device=self.device).expand(len(prone_env_ids)),

                    torch.zeros(len(prone_env_ids), device=self.device)
                )
                root_ori[prone_env_ids] = prone_quat
                root_lin_vel[prone_env_ids] = 0.0
                root_ang_vel[prone_env_ids] = 0.0
            
            # 仰卧状态 (supine) - 躺着，面朝上
            if torch.any(supine_env_flag):
                supine_env_ids = env_ids_tensor[supine_env_flag]
                joint_pos[supine_env_ids] = 0.0
                joint_vel[supine_env_ids] = 0.0
                root_pos[supine_env_ids, 2] = 0.25  # 降低高度
                # 绕X轴旋转180度，然后绕Z轴旋转180度，面朝上
                supine_quat = quat_from_euler_xyz(
                    torch.zeros(len(supine_env_ids), device=self.device),
                    torch.tensor(math.pi/2, device=self.device).expand(len(supine_env_ids)),
                    torch.zeros(len(supine_env_ids), device=self.device),
                )
                root_ori[supine_env_ids] = supine_quat
                root_lin_vel[supine_env_ids] = 0.0
                root_ang_vel[supine_env_ids] = 0.0

        self.robot.write_joint_state_to_sim(joint_pos[env_ids], joint_vel[env_ids], env_ids=env_ids)
        self.robot.write_root_state_to_sim(
            torch.cat([root_pos[env_ids], root_ori[env_ids], root_lin_vel[env_ids], root_ang_vel[env_ids]], dim=-1),
            env_ids=env_ids,
        )
    def update_force_curriculum(self, env_ids: torch.Tensor):
        num_standing =0#int(self.num_envs * 0.5)

        if len(env_ids) == 0:
            return
        # 在传入的env_ids中找到满足anchor_conditions_good且为非站立环境的ID
        anchor_conditions_good_mask = self.anchor_conditions_good[env_ids]
        non_standing_mask = env_ids >= num_standing
        combined_mask = anchor_conditions_good_mask & non_standing_mask&(~self.getup_timeout[env_ids])
        increased_force_mask =non_standing_mask&(self.getup_timeout[env_ids])
        non_standing_env_ids = env_ids[combined_mask]
        increased_force_env_ids = env_ids[increased_force_mask]
        if len(non_standing_env_ids) == 0:
            return
        self.force[non_standing_env_ids] = (self.force[non_standing_env_ids] - 5).clamp(0, np.inf)
        if len(increased_force_env_ids) > 0:
            self.force[increased_force_env_ids] = (self.force[increased_force_env_ids] + 5).clamp(0, 100)
       

    def _update_command(self):
        self.time_steps += 1
        env_ids = torch.where(self.time_steps >= self.motion.time_step_total)[0]
        self._resample_command(env_ids)

        anchor_pos_w_repeat = self.anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        anchor_quat_w_repeat = self.anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_pos_w_repeat = self.robot_anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_quat_w_repeat = self.robot_anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)

        delta_pos_w = robot_anchor_pos_w_repeat
        delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]
        delta_ori_w = yaw_quat(quat_mul(robot_anchor_quat_w_repeat, quat_inv(anchor_quat_w_repeat)))

        self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w)
        self.body_pos_relative_w = delta_pos_w + quat_apply(delta_ori_w, self.body_pos_w - anchor_pos_w_repeat)

        self.bin_failed_count = (
            self.cfg.adaptive_alpha * self._current_bin_failed + (1 - self.cfg.adaptive_alpha) * self.bin_failed_count
        )
        self._current_bin_failed.zero_()
        if self.cfg.force_curriculum_enabled:
        # 力的应用现在通过课程学习系统管理，这里只需要应用当前的力值
            self.apply_force_to_robot(self.force)
       
    def apply_force_to_robot(self, force_magnitude: torch.Tensor):
        """Apply upward force along z-axis to robot base for Isaac Sim.
        
        Args:
            env_ids: Environment IDs to apply force to
            force_magnitude: Magnitude of upward force in Newtons
        """
        need_to_pull = torch.where(self.anchor_conditions_good==False)[0]
        
        if len(need_to_pull) == 0:
            return
       
        shoulder_body_indices = self.robot.find_bodies(".*shoulder_pitch_link")[0]
        
        # Create force tensor for multiple bodies: [len(need_to_pull), len(shoulder_body_indices), 3]
        num_bodies = len(shoulder_body_indices)
        force_tensor = torch.zeros(len(need_to_pull), num_bodies, 3, device=self.device)
        
        # Apply same force magnitude to all shoulder bodies (vectorized)
        force_tensor[:, :, 2] = force_magnitude[need_to_pull].squeeze(-1).unsqueeze(1)  # z-axis upward force
        
      
        self.robot.set_external_force_and_torque(
            forces=force_tensor,  # Shape: (len(need_to_pull), num_bodies, 3)
            torques=torch.zeros_like(force_tensor),  # No torques
            body_ids=[shoulder_body_indices],  # Only apply to torso body
            env_ids=need_to_pull,
            is_global=True  # Apply force in global/world frame

        )
        
        # IMPORTANT: Must call write_data_to_sim to actually apply the forces
        self.robot.write_data_to_sim()

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "current_anchor_visualizer"):
                self.current_anchor_visualizer = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/current/anchor")
                )
                self.goal_anchor_visualizer = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/anchor")
                )

                self.current_body_visualizers = []
                self.goal_body_visualizers = []
                for name in self.cfg.body_names:
                    self.current_body_visualizers.append(
                        VisualizationMarkers(
                            self.cfg.body_visualizer_cfg.replace(prim_path="/Visuals/Command/current/" + name)
                        )
                    )
                    self.goal_body_visualizers.append(
                        VisualizationMarkers(
                            self.cfg.body_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/" + name)
                        )
                    )
                
                # Add force visualizer
                from isaaclab.markers.config import RED_ARROW_X_MARKER_CFG
                self.force_visualizer = VisualizationMarkers(
                    RED_ARROW_X_MARKER_CFG.replace(
                        prim_path="/Visuals/Command/force",
                        markers={
                            "arrow": RED_ARROW_X_MARKER_CFG.markers["arrow"].replace(
                                scale=(0.5, 0.5, 6.1)  # Arrow for force vector
                            )
                        }
                    )
                )

            self.current_anchor_visualizer.set_visibility(True)
            self.goal_anchor_visualizer.set_visibility(True)
            self.force_visualizer.set_visibility(True)
            for i in range(len(self.cfg.body_names)):
                self.current_body_visualizers[i].set_visibility(True)
                self.goal_body_visualizers[i].set_visibility(True)

        else:
            if hasattr(self, "current_anchor_visualizer"):
                self.current_anchor_visualizer.set_visibility(False)
                self.goal_anchor_visualizer.set_visibility(False)
                self.force_visualizer.set_visibility(False)
                for i in range(len(self.cfg.body_names)):
                    self.current_body_visualizers[i].set_visibility(False)
                    self.goal_body_visualizers[i].set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return
        
        # 安全检查：确保环境和场景仍然可访问
        if not hasattr(self._env, 'scene'):
            return

        self.current_anchor_visualizer.visualize(self.robot_anchor_pos_w, self.robot_anchor_quat_w)
        self.goal_anchor_visualizer.visualize(self.anchor_pos_w, self.anchor_quat_w)

        # Visualize force vectors for robots that need pulling
        need_to_pull = torch.where(self.anchor_conditions_good==False)[0]
        if torch.any(need_to_pull):
            # Get shoulder body indices where force is actually applied
            shoulder_body_indices = self.robot.find_bodies(".*shoulder_pitch_link")[0]
            
            # Get positions where force is applied (shoulder body positions)
            shoulder_positions = self.robot.data.body_pos_w[need_to_pull][:, shoulder_body_indices]  # [num_envs, num_shoulders, 3]
            
            # Flatten to show all shoulder positions
            num_envs = len(need_to_pull)
            num_shoulders = len(shoulder_body_indices)
            force_positions = shoulder_positions.reshape(-1, 3)  # [num_envs * num_shoulders, 3]
            
            # Create quaternions to rotate x-axis arrow to z-axis (90 degrees around y-axis)
            force_orientations = torch.zeros(force_positions.shape[0], 4, device=self.device)
            force_orientations[:, 0] = 0.7071  # cos(45°) for 90° rotation around y-axis
            force_orientations[:, 2] = 0.7071  # sin(45°) for 90° rotation around y-axis
            self.force_visualizer.visualize(force_positions, force_orientations)
        else:
            # Hide force visualizer when no forces are applied - use a dummy position far away
            dummy_pos = torch.tensor([[1000.0, 1000.0, 1000.0]], device=self.device)  # Far away position
            dummy_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=self.device)  # Identity quaternion
            self.force_visualizer.visualize(dummy_pos, dummy_quat)

        for i in range(len(self.cfg.body_names)):
            self.current_body_visualizers[i].visualize(self.robot_body_pos_w[:, i], self.robot_body_quat_w[:, i])
            self.goal_body_visualizers[i].visualize(self.body_pos_relative_w[:, i], self.body_quat_relative_w[:, i])


@configclass
class MotionCommandCfg(CommandTermCfg):
    """Configuration for the motion command."""

    class_type: type = MotionCommand

    asset_name: str = MISSING

    motion_file: str = MISSING
    anchor_body_name: str = MISSING
    body_names: list[str] = MISSING

    pose_range: dict[str, tuple[float, float]] = {}
    velocity_range: dict[str, tuple[float, float]] = {}

    joint_position_range: tuple[float, float] = (-0.52, 0.52)

    adaptive_kernel_size: int = 3
    adaptive_lambda: float = 0.8
    adaptive_uniform_ratio: float = 0.1
    adaptive_alpha: float = 0.001
    
    anchor_pos_threshold: float = 0.25
    anchor_ori_threshold: float = 0.3

    anchor_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    anchor_visualizer_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)

    body_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    body_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)

    force: float = 100.0
    enable_force_curriculum: bool = False
    ang_pitch: float = 10.0
    
    # 力课程学习参数
    force_curriculum_enabled: bool = False
    force_curriculum_update_interval: int = 10  # 每N步更新一次力课程学习
    force_reduction_rate: float = 10.0  # 每次减少的力大小
    min_force: float = 0.0  # 最小辅助力
    max_force: float = 500.0  # 最大辅助力
    standing_base_force: float = 50.0  # 站立环境的基础力
