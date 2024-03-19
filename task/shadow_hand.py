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

import numpy as np
import torch
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.torch import *
from task.shadow_hand import RLTask
from robots.articulations.shadow_hand import ShadowHand
from robots.articulations.views.shadow_hand_view import ShadowHandView
from task.Shared.Grasp_Google_Obj_manipulation import GraspGoogleObjManipulationTask

class ShadowHandTask(GraspGoogleObjManipulationTask):
    def __init__(self, name, sim_config, env, offset=None) -> None:

        self.update_config(sim_config)
        GraspGoogleObjManipulationTask.__init__(self, name=name, env=env)
        return

    def update_config(self, sim_config):
        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config

        self.object_type = self._task_cfg["env"]["objectType"]
        assert self.object_type in ["block"]

        self.obs_type = self._task_cfg["env"]["observationType"]
        if not (self.obs_type in ["openai", "full_no_vel", "full", "full_state"]):
            raise Exception(
                "Unknown type of observations!\nobservationType should be one of: [openai, full_no_vel, full, full_state]"
            )
        print("Obs type:", self.obs_type)
        self.num_obs_dict = {
            "openai": 42,
            "full_no_vel": 77,
            "full": 157,
            "full_state": 187,
        }

        self.asymmetric_obs = self._task_cfg["env"]["asymmetric_observations"]
        self.use_vel_obs = False

        self.fingertip_obs = True
        self.fingertips = [
            "robot0:ffdistal",
            "robot0:mfdistal",
            "robot0:rfdistal",
            "robot0:lfdistal",
            "robot0:thdistal",
        ]
        self.num_fingertips = len(self.fingertips)

        self.object_scale = torch.tensor([1.0, 1.0, 1.0])
        self.force_torque_obs_scale = 10.0

        num_states = 0
        if self.asymmetric_obs:
            num_states = 187

        self._num_observations = self.num_obs_dict[self.obs_type]
        self._num_actions = 20
        self._num_states = num_states
        InHandManipulationTask.update_config(self)

    def get_starting_positions(self):
        self.hand_start_translation = torch.tensor([0.0, 0.0, 0.5], device=self.device)
        self.hand_start_orientation = torch.tensor([0.0, 1.0, 0.0, 0.0], device=self.device) # Quaternion for 180 degrees rotation around X-axis
        self.pose_dy, self.pose_dz = -0.39, 0.10

    def get_hand(self):
        shadow_hand = ShadowHand(
            prim_path=self.default_zero_env_path + "/shadow_hand",
            name="shadow_hand",
            translation=self.hand_start_translation,
            orientation=self.hand_start_orientation,
        )
        self._sim_config.apply_articulation_settings(
            "shadow_hand",
            get_prim_at_path(shadow_hand.prim_path),
            self._sim_config.parse_actor_config("shadow_hand"),
        )
        shadow_hand.set_shadow_hand_properties(stage=self._stage, shadow_hand_prim=shadow_hand.prim)
        shadow_hand.set_motor_control_mode(stage=self._stage, shadow_hand_path=shadow_hand.prim_path)

    def get_hand_view(self, scene):
        hand_view = ShadowHandView(prim_paths_expr="/World/envs/.*/shadow_hand", name="shadow_hand_view")
        scene.add(hand_view._fingers)
        return hand_view

    def get_observations(self):
        self.get_object_goal_observations()

        self.fingertip_pos, self.fingertip_rot = self._hands._fingers.get_world_poses(clone=False)
        self.fingertip_pos -= self._env_pos.repeat((1, self.num_fingertips)).reshape(
            self.num_envs * self.num_fingertips, 3
        )
        self.fingertip_velocities = self._hands._fingers.get_velocities(clone=False)

        self.hand_dof_pos = self._hands.get_joint_positions(clone=False)
        self.hand_dof_vel = self._hands.get_joint_velocities(clone=False)

        if self.obs_type == "full_state" or self.asymmetric_obs:
            self.vec_sensor_tensor = self._hands.get_measured_joint_forces(
                joint_indices=self._hands._sensor_indices
            ).view(self._num_envs, -1)

        if self.obs_type == "openai":
            self.compute_fingertip_observations(True)
        elif self.obs_type == "full_no_vel":
            self.compute_full_observations(True)
        elif self.obs_type == "full":
            self.compute_full_observations()
        elif self.obs_type == "full_state":
            self.compute_full_state(False)
        else:
            print("Unkown observations type!")

        if self.asymmetric_obs:
            self.compute_full_state(True)

        observations = {self._hands.name: {"obs_buf": self.obs_buf}}
        return observations
