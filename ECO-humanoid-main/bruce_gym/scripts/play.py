# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2021 ETH Zurich, Nikita Rudin
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
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
#
# Copyright (c) 2024 Beijing RobotEra TECHNOLOGY CO.,LTD. All rights reserved.
# Copyright (c) 2026 ECO Authors. All rights reserved.
import csv
import os
import cv2
import numpy as np
from isaacgym import gymapi
from bruce_gym import LEGGED_GYM_ROOT_DIR
import time
# import isaacgym
from bruce_gym.envs import *
from bruce_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger
from bruce_gym.rotor_energy import (
    BRUCE_YAW_JOINT_INDICES,
    bruce_rotor_names_from_dof_names,
)
from isaacgym.torch_utils import *
from PIL import Image

import torch
from tqdm import tqdm
from datetime import datetime

from pynput import keyboard

speed_x = 0.15
speed_y = 0.0
speed_w = 0.0


def _flatten_log_for_csv(values):
    row = {}
    for key, value in values.items():
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        if isinstance(value, np.ndarray):
            flat_value = value.reshape(-1)
            for idx, item in enumerate(flat_value):
                row[f"{key}_{idx}"] = item.item() if hasattr(item, "item") else item
        elif isinstance(value, (list, tuple)):
            for idx, item in enumerate(value):
                row[f"{key}_{idx}"] = item.item() if hasattr(item, "item") else item
        else:
            row[key] = value.item() if hasattr(value, "item") else value
    return row


def _resolve_foot_contact_indices(feet_names):
    if "ankle_pitch_link_l" in feet_names and "ankle_pitch_link_r" in feet_names:
        return (
            feet_names.index("ankle_pitch_link_l"),
            feet_names.index("ankle_pitch_link_r"),
        )
    if len(feet_names) >= 2:
        return 0, 1
    raise ValueError(f"Expected at least two feet, got: {feet_names}")


def _resolve_dof_pair(dof_names, left_name, right_name, fallback):
    if left_name in dof_names and right_name in dof_names:
        return dof_names.index(left_name), dof_names.index(right_name)
    return fallback


def _resolve_yaw_energy_indices(left_dof_idx, right_dof_idx):
    yaw_dof_indices = list(BRUCE_YAW_JOINT_INDICES)
    if left_dof_idx in yaw_dof_indices and right_dof_idx in yaw_dof_indices:
        return (
            yaw_dof_indices.index(left_dof_idx),
            yaw_dof_indices.index(right_dof_idx),
        )
    return 0, 1


def on_press(key):
    global speed_x, speed_y, speed_w
    try:
        if key.char == 'w':
            speed_x = max(speed_x-0.1, -1.0)
        elif key.char == 's':
            speed_x = min(speed_x+0.1, 1.0)
        elif key.char == 'a':
            speed_y = max(speed_y-0.1, -1.0)
        elif key.char == 'd':
            speed_y = min(speed_y+0.1, 1.0)
        elif key.char == 'j':
            speed_w = max(speed_w-0.1, -1.0)
        elif key.char == 'k':
            speed_w = min(speed_w+0.1, 1.0)
    except AttributeError:
        pass

def play(args):
    global speed_x, speed_y, speed_w

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 1)
    env_cfg.sim.max_gpu_contact_pairs = 2**10
    #env_cfg.terrain.mesh_type = 'trimesh'
    env_cfg.terrain.mesh_type = 'plane'
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False     
    env_cfg.terrain.max_init_terrain_level = 5
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.push_robots = False 
    env_cfg.domain_rand.joint_angle_noise = 0.
    env_cfg.noise.curriculum = False
    env_cfg.noise.noise_level = 0.5
    env_cfg.domain_rand.disturbance = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.delay = False
    env_cfg.domain_rand.randomize_kp = False
    env_cfg.domain_rand.randomize_kd = False

    train_cfg.seed = 123145
    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    env.set_camera(env_cfg.viewer.pos, env_cfg.viewer.lookat)
    env.gym.set_light_parameters(env.sim, 0, gymapi.Vec3(1.0, 1.0, 1.0),  gymapi.Vec3(0.05, 0.05, 0.05),  gymapi.Vec3(0.0, 0.0, 1.0))
    env.gym.set_light_parameters(env.sim, 1, gymapi.Vec3(0.5, 0.5, 0.5),  gymapi.Vec3(0.2, 0.2, 0.2),  gymapi.Vec3(1.0, -2.0, 0.0))

    obs = env.get_observations()

    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    csv_file_path = time.strftime('simulation_isaac_data_' + train_cfg.runner.load_run.split('/')[-1]+'_%Y%m%d_%H%M%S.csv')

    with open(csv_file_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_header = None
        logger = Logger(env.dt)
        robot_index = 0 # which robot is used for logging
        joint_index = 4 # which joint is used for logging
        left_foot_idx, right_foot_idx = _resolve_foot_contact_indices(
            getattr(env, "feet_names", [])
        )
        left_hip_yaw_idx, right_hip_yaw_idx = _resolve_dof_pair(
            list(env.dof_names), "hip_yaw_l", "hip_yaw_r", fallback=(0, 5)
        )
        left_yaw_energy_idx, right_yaw_energy_idx = _resolve_yaw_energy_indices(
            left_hip_yaw_idx, right_hip_yaw_idx
        )
        rotor_names = bruce_rotor_names_from_dof_names(env.dof_names)
        stop_state_log = 50000 # number of steps before plotting states
        listener = keyboard.Listener(on_press=on_press)
        listener.start()
        if RENDER:
            camera_properties = gymapi.CameraProperties()
            camera_properties.width = 1920
            camera_properties.height = 1080
            h1 = env.gym.create_camera_sensor(env.envs[0], camera_properties)
            camera_offset = gymapi.Vec3(1, -1, 0.5)
            camera_rotation = gymapi.Quat.from_axis_angle(gymapi.Vec3(-0.3, 0.2, 1),
                                                        np.deg2rad(135))

            actor_handle = env.gym.get_actor_handle(env.envs[0], 0)
            body_handle = env.gym.get_actor_rigid_body_handle(env.envs[0], actor_handle, 0)
            env.gym.attach_camera_to_body(
                h1, env.envs[0], body_handle,
                gymapi.Transform(camera_offset, camera_rotation),
                gymapi.FOLLOW_POSITION)

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_dir = os.path.join(LEGGED_GYM_ROOT_DIR, 'videos')
            experiment_dir = os.path.join(LEGGED_GYM_ROOT_DIR, 'videos', train_cfg.runner.experiment_name)
            args.run_name = 'v1'
            dir = os.path.join(experiment_dir, datetime.now().strftime('%b%d_%H-%M-%S')+ args.run_name + '.mp4')
            if not os.path.exists(video_dir):
                os.mkdir(video_dir)
            if not os.path.exists(experiment_dir):
                os.mkdir(experiment_dir)
            video = cv2.VideoWriter(dir, fourcc, 50.0, (1920, 1080))
        cost_sum = 0

        cur_cost_sum = torch.zeros(
            env.num_envs, dtype=torch.float, device=env.device
        )
        lin_vel_x_sum = 0
        episode_id = 0
        episode_step = 0
        for step_env in range(stop_state_log):
            actions = policy(obs.detach())
            if FIX_COMMAND:
                env.commands[:, 0] = speed_x
                env.commands[:, 1] = speed_y
                env.commands[:, 2] = speed_w
                env.commands[:, 3] = 0.
            env.gait_frequency = torch.ones_like(env.gait_frequency)
            command_zeros = env.commands[:, 0] == 0.0
            env.gait_frequency[command_zeros] = 0.0
            obs, critic_obs, rews, dones, infos, cost= env.step(actions.detach())

            cost_value = cost[0][robot_index].item() if isinstance(cost, list) else cost[robot_index].item()
            cost_sum += cost_value
            done = bool(dones[robot_index].item())
            lin_vel_x = env.base_lin_vel[robot_index, 0].item()
            lin_vel_x_sum += lin_vel_x
            if step_env%500 == 0:
                print("avg_lin_vel_x", lin_vel_x_sum/500, "cost_sum", cost_sum/500)
                lin_vel_x_sum = 0
                cost_sum = 0

            if RENDER:
                env.gym.fetch_results(env.sim, True)
                env.gym.step_graphics(env.sim)
                env.gym.render_all_camera_sensors(env.sim)
                img = env.gym.get_camera_image(env.sim, env.envs[0], h1, gymapi.IMAGE_COLOR)
                img = np.reshape(img, (1080, 1920, 4))
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                # video.write(img[..., :3])

            foot_force_z = env.contact_forces[robot_index, env.feet_indices, 2]
            foot_contact = foot_force_z > 5.
            gait_phase = torch.remainder(env._get_phase()[robot_index], 1.0).item()

            log_dict = {


                    'enery_0_gym':  env.torques[robot_index, 0].item() * env.dof_vel[robot_index, 0].item(),
                    'enery_1_gym':  env.torques[robot_index, 1].item() * env.dof_vel[robot_index, 1].item(),
                    'enery_2_gym':  env.torques[robot_index, 2].item() * env.dof_vel[robot_index, 2].item(),
                    'enery_3_gym':  env.torques[robot_index, 3].item() * env.dof_vel[robot_index, 3].item(),
                    'enery_4_gym':  env.torques[robot_index, 4].item() * env.dof_vel[robot_index, 4].item(),
                    'enery_5_gym':  env.torques[robot_index, 5].item() * env.dof_vel[robot_index, 5].item(),
                    'enery_6_gym':  env.torques[robot_index, 6].item() * env.dof_vel[robot_index, 6].item(),
                    'enery_7_gym':  env.torques[robot_index, 7].item() * env.dof_vel[robot_index, 7].item(),
                    'enery_8_gym':  env.torques[robot_index, 8].item() * env.dof_vel[robot_index, 8].item(),
                    'enery_9_gym':  env.torques[robot_index, 9].item() * env.dof_vel[robot_index, 9].item(),

                    'dof_pos_target0_gym': env.actions[robot_index, 0].item() * env.cfg.control.action_scale + env.default_dof_pos[0][0].item(),
                    'dof_pos0_gym': env.dof_pos[robot_index, 0].item(),
                    'dof_pos_target1_gym': env.actions[robot_index, 1].item() * env.cfg.control.action_scale + env.default_dof_pos[0][1].item(),
                    'dof_pos1_gym': env.dof_pos[robot_index, 1].item(),
                    'dof_pos_target2_gym': env.actions[robot_index, 2].item() * env.cfg.control.action_scale + env.default_dof_pos[0][2].item(),
                    'dof_pos2_gym': env.dof_pos[robot_index, 2].item(),
                    'dof_pos_target3_gym': env.actions[robot_index, 3].item() * env.cfg.control.action_scale + env.default_dof_pos[0][3].item(),
                    'dof_pos3_gym': env.dof_pos[robot_index, 3].item(),
                    'dof_pos_target4_gym': env.actions[robot_index, 4].item() * env.cfg.control.action_scale + env.default_dof_pos[0][4].item(),
                    'dof_pos4_gym': env.dof_pos[robot_index, 4].item(),
                    'dof_pos_target5_gym': env.actions[robot_index, 5].item() * env.cfg.control.action_scale + env.default_dof_pos[0][5].item(),
                    'dof_pos5_gym': env.dof_pos[robot_index, 5].item(),
                    'dof_pos_target6_gym': env.actions[robot_index, 6].item() * env.cfg.control.action_scale + env.default_dof_pos[0][6].item(),
                    'dof_pos6_gym': env.dof_pos[robot_index, 6].item(),
                    'dof_pos_target7_gym': env.actions[robot_index, 7].item() * env.cfg.control.action_scale + env.default_dof_pos[0][7].item(),
                    'dof_pos7_gym': env.dof_pos[robot_index, 7].item(),
                    'dof_pos_target8_gym': env.actions[robot_index, 8].item() * env.cfg.control.action_scale + env.default_dof_pos[0][8].item(),
                    'dof_pos8_gym': env.dof_pos[robot_index, 8].item(),
                    'dof_pos_target9_gym': env.actions[robot_index, 9].item() * env.cfg.control.action_scale + env.default_dof_pos[0][9].item(),
                    'dof_pos9_gym': env.dof_pos[robot_index, 9].item(),


                    'dof_vel0_gym': env.dof_vel[robot_index, 0].item(),
                    'dof_vel1_gym': env.dof_vel[robot_index, 1].item(),
                    'dof_vel2_gym': env.dof_vel[robot_index, 2].item(),
                    'dof_vel3_gym': env.dof_vel[robot_index, 3].item(),
                    'dof_vel4_gym': env.dof_vel[robot_index, 4].item(),
                    'dof_vel5_gym': env.dof_vel[robot_index, 5].item(),
                    'dof_vel6_gym': env.dof_vel[robot_index, 6].item(),
                    'dof_vel7_gym': env.dof_vel[robot_index, 7].item(),
                    'dof_vel8_gym': env.dof_vel[robot_index, 8].item(),
                    'dof_vel9_gym': env.dof_vel[robot_index, 9].item(),


                    'dof_torque_0': env.torques[robot_index, 0].item(),
                    'dof_torque_1': env.torques[robot_index, 1].item(),
                    'dof_torque_2': env.torques[robot_index, 2].item(),
                    'dof_torque_3': env.torques[robot_index, 3].item(),
                    'dof_torque_4': env.torques[robot_index, 4].item(),
                    'dof_torque_5': env.torques[robot_index, 5].item(),

                    'dof_torque_6': env.torques[robot_index, 6].item(),
                    'dof_torque_7': env.torques[robot_index, 7].item(),
                    'dof_torque_8': env.torques[robot_index, 8].item(),
                    'dof_torque_9': env.torques[robot_index, 9].item(),


                    'command_x': env.commands[robot_index, 0].item(),
                    'command_y': env.commands[robot_index, 1].item(),
                    'command_yaw': env.commands[robot_index, 2].item(),
                    'episode_id': episode_id,
                    'episode_step': episode_step,
                    'done': float(done),
                    'gait_phase': gait_phase,
                    'left_contact_state': foot_contact[left_foot_idx].float().item(),
                    'right_contact_state': foot_contact[right_foot_idx].float().item(),
                    'left_contact_force_z': foot_force_z[left_foot_idx].item(),
                    'right_contact_force_z': foot_force_z[right_foot_idx].item(),
                    'hip_yaw_power_l': env.torques[robot_index, left_hip_yaw_idx].item()
                    * env.dof_vel[robot_index, left_hip_yaw_idx].item(),
                    'hip_yaw_power_r': env.torques[robot_index, right_hip_yaw_idx].item()
                    * env.dof_vel[robot_index, right_hip_yaw_idx].item(),
                    'base_vel_x': env.base_lin_vel[robot_index, 0].item(),
                    'base_vel_y': env.base_lin_vel[robot_index, 1].item(),
                    'base_vel_z': env.base_lin_vel[robot_index, 2].item(),
                    'contact_forces_z': foot_force_z.cpu().numpy(),
                    'contact_vel_z': torch.norm(env.rigid_state[robot_index, env.feet_indices, 7:10], dim=-1).cpu().numpy(),
                    'contact_period': [
                        env._get_gait_phase()[robot_index, left_foot_idx].item(),
                        env._get_gait_phase()[robot_index, right_foot_idx].item(),
                    ],

                    'base_euler0_gym': env.base_euler_xyz[robot_index,0].item(),
                    'base_euler1_gym': env.base_euler_xyz[robot_index,1].item(),
                    'base_euler2_gym': env.base_euler_xyz[robot_index,2].item(),
                    'height': env.root_states[robot_index, 2].item(),
                # 'base_ang0_gazebo': env_gazebo.base_ang_vel[robot_index, 0].item(),
                # 'base_ang1_gazebo': env_gazebo.base_ang_vel[robot_index, 1].item(),
                # 'base_ang2_gazebo': env_gazebo.base_ang_vel[robot_index, 2].item(),

            }
            if hasattr(env, "rotor_power"):
                for motor_idx, motor_name in enumerate(rotor_names):
                    log_dict.update({
                        f'{motor_name}_output_torque': env.rotor_output_torque[robot_index, motor_idx].item(),
                        f'{motor_name}_output_velocity': env.rotor_output_velocity[robot_index, motor_idx].item(),
                        f'{motor_name}_rotor_torque': env.rotor_torque[robot_index, motor_idx].item(),
                        f'{motor_name}_rotor_velocity': env.rotor_velocity[robot_index, motor_idx].item(),
                        f'{motor_name}_rotor_power': env.rotor_power[robot_index, motor_idx].item(),
                        f'{motor_name}_drive_energy': env.rotor_drive_energy_per_motor[robot_index, motor_idx].item(),
                        f'{motor_name}_brake_energy': env.rotor_brake_energy_per_motor[robot_index, motor_idx].item(),
                    })
                log_dict.update({
                    'rotor_drive_energy': env.rotor_drive_energy[robot_index].item(),
                    'rotor_brake_energy': env.rotor_brake_energy[robot_index].item(),
                    'hip_yaw_drive_energy_l': env.yaw_joint_drive_energy[
                        robot_index, left_yaw_energy_idx
                    ].item(),
                    'hip_yaw_drive_energy_r': env.yaw_joint_drive_energy[
                        robot_index, right_yaw_energy_idx
                    ].item(),
                    'hip_yaw_brake_energy_l': env.yaw_joint_brake_energy[
                        robot_index, left_yaw_energy_idx
                    ].item(),
                    'hip_yaw_brake_energy_r': env.yaw_joint_brake_energy[
                        robot_index, right_yaw_energy_idx
                    ].item(),
                })
            if args.task == 'kuavo_ppo':
                log_dict.update({
                    'dof_pos_target10_gym': env.actions[robot_index, 10].item() * env.cfg.control.action_scale + env.default_dof_pos[0][10].item(),
                    'dof_pos10_gym': env.dof_pos[robot_index, 10].item(),
                    'dof_pos_target11_gym': env.actions[robot_index, 11].item() * env.cfg.control.action_scale + env.default_dof_pos[0][11].item(),
                    'dof_pos11_gym': env.dof_pos[robot_index, 11].item(),
                    'dof_vel10_gym': env.dof_vel[robot_index, 10].item(),
                    'dof_vel11_gym': env.dof_vel[robot_index, 11].item(),
                    'dof_torque_10': env.torques[robot_index, 10].item(),
                    'dof_torque_11': env.torques[robot_index, 11].item(),
                    })
                
            for i in range(len(env.reward_functions)):
                name = env.reward_names[i]
                rew = env.reward_functions[i]() * env.reward_scales[name]
                log_dict.update(
                    {'rew_' +name: rew[robot_index].item()}
                )
            csv_row = _flatten_log_for_csv({
                'step': step_env,
                'l_ankele_pos_z': env.rigid_state[
                    robot_index, env.feet_indices[left_foot_idx], 2
                ].item(),
                **log_dict,
            })
            if csv_header is None:
                csv_header = list(csv_row.keys())
                csv_writer.writerow(csv_header)
            csv_writer.writerow([csv_row.get(key, "") for key in csv_header])
            logger.log_states(log_dict)
            episode_step += 1
            if done:
                episode_id += 1
                episode_step = 0

            # ====================== Log states ======================
            if infos["episode"]:
                num_episodes = torch.sum(env.reset_buf).item()
                if num_episodes>0:
                    logger.log_rewards(infos["episode"], num_episodes)
        print('cost_sum', cost_sum)
        logger.print_rewards()
        logger.plot_states()
        
        if RENDER:
            video.release()

if __name__ == '__main__':
    EXPORT_POLICY = True
    RENDER = True
    FIX_COMMAND = True
    args = get_args()
    play(args)
