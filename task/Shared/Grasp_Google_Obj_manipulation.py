# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import math
from abc import abstractmethod

import numpy as np
import torch
from omni.isaac.core.prims import RigidPrimView, XFormPrim
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.stage import add_reference_to_stage, get_current_stage
from omni.isaac.core.utils.torch import *
from omniisaacgymenvs.tasks.base.rl_task import RLTask

class GraspGoogleObjManipulationTask(RLTask):
    def __init__(self, name, env, offset=None) -> None:

        GraspGoogleObjManipulationTask.update_config(self)

        RLTask.__init__(self, name, env)

        self.x_unit_tensor = torch.tensor([1, 0, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.y_unit_tensor = torch.tensor([0, 1, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.z_unit_tensor = torch.tensor([0, 0, 1], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))

        self.reset_goal_buf = self.reset_buf.clone()
        self.successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.consecutive_successes = torch.zeros(1, dtype=torch.float, device=self.device)
        self.randomization_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        self.av_factor = torch.tensor(self.av_factor, dtype=torch.float, device=self.device)
        self.total_successes = 0
        self.total_resets = 0

    def update_config(self):
        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"]

        self.dist_reward_scale = self._task_cfg["env"]["distRewardScale"]
        self.rot_reward_scale = self._task_cfg["env"]["rotRewardScale"]
        self.action_penalty_scale = self._task_cfg["env"]["actionPenaltyScale"]
        self.success_tolerance = self._task_cfg["env"]["successTolerance"]
        self.reach_goal_bonus = self._task_cfg["env"]["reachGoalBonus"]
        self.fall_dist = self._task_cfg["env"]["fallDistance"]
        self.fall_penalty = self._task_cfg["env"]["fallPenalty"]
        self.rot_eps = self._task_cfg["env"]["rotEps"]
        self.vel_obs_scale = self._task_cfg["env"]["velObsScale"]

        self.reset_position_noise = self._task_cfg["env"]["resetPositionNoise"]
        self.reset_rotation_noise = self._task_cfg["env"]["resetRotationNoise"]
        self.reset_dof_pos_noise = self._task_cfg["env"]["resetDofPosRandomInterval"]
        self.reset_dof_vel_noise = self._task_cfg["env"]["resetDofVelRandomInterval"]

        self.hand_dof_speed_scale = self._task_cfg["env"]["dofSpeedScale"]
        self.use_relative_control = self._task_cfg["env"]["useRelativeControl"]
        self.act_moving_average = self._task_cfg["env"]["actionsMovingAverage"]

        self.max_episode_length = self._task_cfg["env"]["episodeLength"]
        self.reset_time = self._task_cfg["env"].get("resetTime", -1.0)
        self.print_success_stat = self._task_cfg["env"]["printNumSuccesses"]
        self.max_consecutive_successes = self._task_cfg["env"]["maxConsecutiveSuccesses"]
        self.av_factor = self._task_cfg["env"].get("averFactor", 0.1)

        self.dt = 1.0 / 60
        control_freq_inv = self._task_cfg["env"].get("controlFrequencyInv", 1)
        if self.reset_time > 0.0:
            self.max_episode_length = int(round(self.reset_time / (control_freq_inv * self.dt)))
            print("Reset time: ", self.reset_time)
            print("New episode length: ", self.max_episode_length)

    def set_up_scene(self, scene) -> None:
        self._stage = get_current_stage()
        self._assets_root_path = get_assets_root_path()

        self.get_starting_positions()
        self.get_hand()

        self.object_start_translation = self.hand_start_translation.clone()
        self.object_start_translation[1] += self.pose_dy
        self.object_start_translation[2] += self.pose_dz
        self.object_start_orientation = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)

        self.goal_displacement_tensor = torch.tensor([-0.2, -0.06, 0.12], device=self.device)
        self.goal_start_translation = self.object_start_translation + self.goal_displacement_tensor
        self.goal_start_translation[2] -= 0.04
        self.goal_start_orientation = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)

        self.get_object(self.hand_start_translation, self.pose_dy, self.pose_dz)
        self.get_goal()

        super().set_up_scene(scene, filter_collisions=False)

        self._hands = self.get_hand_view(scene)
        scene.add(self._hands)
        self._objects = RigidPrimView(
            prim_paths_expr="/World/envs/env_.*/object/object",
            name="object_view",
            reset_xform_properties=False,
            masses=torch.tensor([0.07087] * self._num_envs, device=self.device),
        )
        scene.add(self._objects)
        self._goals = RigidPrimView(
            prim_paths_expr="/World/envs/env_.*/goal/object", name="goal_view", reset_xform_properties=False
        )
        self._goals._non_root_link = True  # hack to ignore kinematics
        scene.add(self._goals)

        if self._dr_randomizer.randomize:
            self._dr_randomizer.apply_on_startup_domain_randomization(self)

    def initialize_views(self, scene):
        RLTask.initialize_views(self, scene)

        if scene.object_exists("shadow_hand_view"):
            scene.remove_object("shadow_hand_view", registry_only=True)
        if scene.object_exists("finger_view"):
            scene.remove_object("finger_view", registry_only=True)
        if scene.object_exists("allegro_hand_view"):
            scene.remove_object("allegro_hand_view", registry_only=True)
        if scene.object_exists("goal_view"):
            scene.remove_object("goal_view", registry_only=True)
        if scene.object_exists("object_view"):
            scene.remove_object("object_view", registry_only=True)

        self.get_starting_positions()
        self.object_start_translation = self.hand_start_translation.clone()
        self.object_start_translation[1] += self.pose_dy
        self.object_start_translation[2] += self.pose_dz
        self.object_start_orientation = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)

        self.goal_displacement_tensor = torch.tensor([-0.2, -0.06, 0.12], device=self.device)
        self.goal_start_translation = self.object_start_translation + self.goal_displacement_tensor
        self.goal_start_translation[2] -= 0.04
        self.goal_start_orientation = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)

        self._hands = self.get_hand_view(scene)
        scene.add(self._hands)
        self._objects = RigidPrimView(
            prim_paths_expr="/World/envs/env_.*/object/object",
            name="object_view",
            reset_xform_properties=False,
            masses=torch.tensor([0.07087] * self._num_envs, device=self.device),
        )
        scene.add(self._objects)
        self._goals = RigidPrimView(
            prim_paths_expr="/World/envs/env_.*/goal/object", name="goal_view", reset_xform_properties=False
        )
        self._goals._non_root_link = True  # hack to ignore kinematics
        scene.add(self._goals)

        if self._dr_randomizer.randomize:
            self._dr_randomizer.apply_on_startup_domain_randomization(self)

    @abstractmethod
    def get_hand(self):
        pass

    @abstractmethod
    def get_hand_view(self):
        pass

    @abstractmethod
    def get_observations(self):
        pass
    
    
