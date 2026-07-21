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

import os
import copy
import torch
import numpy as np
import random
from isaacgym import gymapi
from isaacgym import gymutil

from bruce_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from bruce_gym.gpu_auto_select import select_idle_gpu, selected_gpu_from_env
from bruce_gym.rotor_energy import (
    SUPPORTED_ENERGY_COST_MODES,
    parse_reducer_rated_torque_spec,
)


def class_to_dict(obj) -> dict:
    if not hasattr(obj, "__dict__"):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_"):
            continue
        element = []
        val = getattr(obj, key)
        if isinstance(val, list):
            for item in val:
                element.append(class_to_dict(item))
        else:
            element = class_to_dict(val)
        result[key] = element
    return result


def update_class_from_dict(obj, dict):
    for key, val in dict.items():
        attr = getattr(obj, key, None)
        if isinstance(attr, type):
            update_class_from_dict(attr, val)
        else:
            setattr(obj, key, val)
    return


def set_seed(seed):
    if seed == -1:
        seed = np.random.randint(0, 10000)
    print("Setting seed: {}".format(seed))

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_sim_params(args, cfg):
    # code from Isaac Gym Preview 2
    # initialize sim params
    sim_params = gymapi.SimParams()

    # set some values from args
    if args.physics_engine == gymapi.SIM_FLEX:
        if args.device != "cpu":
            print("WARNING: Using Flex with GPU instead of PHYSX!")
    elif args.physics_engine == gymapi.SIM_PHYSX:
        sim_params.physx.use_gpu = args.use_gpu
        sim_params.physx.num_subscenes = args.subscenes
    sim_params.use_gpu_pipeline = args.use_gpu_pipeline

    # if sim options are provided in cfg, parse them and update/override above:
    if "sim" in cfg:
        gymutil.parse_sim_config(cfg["sim"], sim_params)

    # Override num_threads if passed on the command line
    if args.physics_engine == gymapi.SIM_PHYSX and args.num_threads > 0:
        sim_params.physx.num_threads = args.num_threads

    return sim_params


def get_load_path(root, load_run=-1, checkpoint=-1):
    if load_run == -1 or str(load_run) == "-1":
        if not os.path.isdir(root):
            raise ValueError("No runs in this directory: " + root)
        runs = os.listdir(root)
        # TODO sort by date to handle change of month
        runs.sort()
        if "exported" in runs:
            runs.remove("exported")
        if len(runs) == 0:
            raise ValueError("No runs in this directory: " + root)
        load_run = os.path.join(root, runs[-1])
    elif os.path.isabs(load_run):
        load_run = load_run
    else:
        load_run = os.path.join(root, load_run)

    if not os.path.isdir(load_run):
        raise ValueError("Run directory does not exist: " + load_run)

    if checkpoint == -1:
        models = [file for file in os.listdir(load_run) if "model" in file]
        models.sort(key=lambda m: "{0:0>15}".format(m))
        if len(models) == 0:
            raise ValueError("No model checkpoints in directory: " + load_run)
        model = models[-1]
    else:
        model = "model_{}.pt".format(checkpoint)

    load_path = os.path.join(load_run, model)
    if not os.path.isfile(load_path):
        raise ValueError("Checkpoint file does not exist: " + load_path)
    return load_path


def update_cfg_from_args(env_cfg, cfg_train, args):
    # seed
    if env_cfg is not None:
        # num envs
        if args.num_envs is not None:
            env_cfg.env.num_envs = args.num_envs
        if args.seed is not None:
            env_cfg.seed = args.seed
        if args.cost_limit1 is not None:
            env_cfg.env.cost_limit1 = args.cost_limit1
        if args.energy_cost_mode is not None:
            if args.energy_cost_mode not in SUPPORTED_ENERGY_COST_MODES:
                raise ValueError(
                    f"Unsupported energy_cost_mode '{args.energy_cost_mode}'. "
                    f"Supported modes are: {SUPPORTED_ENERGY_COST_MODES}"
                )
            env_cfg.env.energy_cost_mode = args.energy_cost_mode
        reducer_rated_torque = getattr(args, "reducer_rated_torque", None)
        if reducer_rated_torque is not None:
            env_cfg.env.reducer_rated_torque = parse_reducer_rated_torque_spec(
                reducer_rated_torque
            )
        train_command_x = getattr(args, "train_command_x", None)
        if train_command_x is not None:
            env_cfg.commands.ranges.lin_vel_x = [train_command_x, train_command_x]
            print(
                "Overriding training commands.ranges.lin_vel_x to "
                f"[{train_command_x}, {train_command_x}]"
            )
    if cfg_train is not None:
        if args.cost_limit1 is not None:
            cfg_train.algorithm.cost_limit1 = args.cost_limit1
        if args.lambda_lr1 is not None :
            cfg_train.algorithm.lambda_lr1 = args.lambda_lr1
        if args.ipo_kappa1 is not None:
            cfg_train.algorithm.ipo_kappa1 = args.ipo_kappa1
        if args.ipo_kappa2 is not None:
            cfg_train.algorithm.ipo_kappa2 = args.ipo_kappa2
        if args.p3o_kappa1 is not None:
            cfg_train.algorithm.p3o_kappa1 = args.p3o_kappa1
        if args.p3o_kappa2 is not None:
            cfg_train.algorithm.p3o_kappa2 = args.p3o_kappa2
        if args.cost_critic_learning_rate is not None:
            cfg_train.algorithm.cost_critic_learning_rate = args.cost_critic_learning_rate
        if args.seed is not None:
            cfg_train.seed = args.seed
        # alg runner parameters
        if args.max_iterations is not None:
            cfg_train.runner.max_iterations = args.max_iterations
        if args.resume:
            cfg_train.runner.resume = args.resume
        if args.experiment_name is not None:
            cfg_train.runner.experiment_name = args.experiment_name
        if args.run_name is not None:
            cfg_train.runner.run_name = args.run_name
        if args.load_run is not None:
            cfg_train.runner.load_run = args.load_run
        if args.checkpoint is not None:
            cfg_train.runner.checkpoint = args.checkpoint

    return env_cfg, cfg_train


def get_args():
    custom_parameters = [
        {
            "name": "--task",
            "type": str,
            "default": "XBotL_free",
            "help": "Resume training or start testing from a checkpoint. Overrides config file if provided.",
        },
        {
            "name": "--resume",
            "action": "store_true",
            "default": False,
            "help": "Resume training from a checkpoint",
        },
        {
            "name": "--reset_optimizer_on_resume",
            "action": "store_true",
            "default": False,
            "help": "When resuming, load model weights but reset optimizer states.",
        },
        {
            "name": "--reset_lagrange_on_resume",
            "action": "store_true",
            "default": False,
            "help": "When resuming, load model weights but reset Lagrange multipliers.",
        },
        {
            "name": "--experiment_name",
            "type": str,
            "help": "Name of the experiment to run or load. Overrides config file if provided.",
        },
        {
            "name": "--run_name",
            "type": str,
            "help": "Name of the run. Overrides config file if provided.",
        },
        {
            "name": "--load_run",
            "type": str,
            "help": "Name of the run to load when resume=True. If -1: will load the last run. Overrides config file if provided.",
        },
        {
            "name": "--checkpoint",
            "type": int,
            "help": "Saved model checkpoint number. If -1: will load the last checkpoint. Overrides config file if provided.",
        },
        {
            "name": "--headless",
            "action": "store_true",
            "default": False,
            "help": "Force display off at all times",
        },
        {
            "name": "--horovod",
            "action": "store_true",
            "default": False,
            "help": "Use horovod for multi-gpu training",
        },
        {
            "name": "--rl_device",
            "type": str,
            "default": "cuda:0",
            "help": "Device used by the RL algorithm, (cpu, gpu, cuda:0, cuda:1 etc..)",
        },
        {
            "name": "--auto_select_gpu",
            "action": "store_true",
            "default": False,
            "help": "Automatically select an idle NVIDIA GPU for simulation and RL.",
        },
        {
            "name": "--auto_gpu_min_free_memory_mb",
            "type": int,
            "default": 4096,
            "help": "Minimum free GPU memory for automatic GPU selection.",
        },
        {
            "name": "--auto_gpu_max_utilization",
            "type": int,
            "default": 20,
            "help": "Maximum GPU utilization percentage for automatic GPU selection.",
        },
        {
            "name": "--num_envs",
            "type": int,
            "help": "Number of environments to create. Overrides config file if provided.",
        },
        {
            "name": "--seed",
            "type": int,
            "help": "Random seed. Overrides config file if provided.",
        },
        {
            "name": "--lambda_lr1",
            "type": float,
            "help": "Lambda for the learning rate of the first layer of the critic. Overrides config file if provided.",
        },
        {
            "name": "--cost_critic_learning_rate",
            "type": float,
            "help": "Learning rate of the cost critic. Overrides config file if provided.",
        },
        {
            "name": "--ipo_kappa1",
            "type": float,
            "help": "Kappa1 for the IPO algorithm. Overrides config file if provided.",
        },
        {
            "name": "--ipo_kappa2",
            "type": float,
            "help": "Kappa2 for the IPO algorithm. Overrides config file if provided.",
        },
        {
            "name": "--p3o_kappa1",
            "type": float,
            "help": "Kappa1 for the P3O algorithm. Overrides config file if provided.",
        },
        {
            "name": "--p3o_kappa2",
            "type": float,
            "help": "Kappa2 for the P3O algorithm. Overrides config file if provided.",
        },
        {
            "name": "--max_iterations",
            "type": int,
            "help": "Maximum number of training iterations. Overrides config file if provided.",
        },
        {
            "name": "--push_vector",
            "type": float,
            "help": "Minimum push force in gazebo.",
        },
        {
            "name": "--cost_limit1",
            "type": float,
            "help": "cost limit1.",
        },
        {
            "name": "--energy_cost_mode",
            "type": str,
            "help": f"Energy cost mode. Supported: {SUPPORTED_ENERGY_COST_MODES}.",
        },
        {
            "name": "--reducer_rated_torque",
            "type": str,
            "help": (
                "Reducer output-side rated torque T_N in N m: one shared value "
                "or eight comma-separated motor values. Required for "
                "reducer_corrected_8."
            ),
        },
        {
            "name": "--num_eval_episodes",
            "type": int,
            "default": 20,
            "help": "Number of evaluation episodes.",
        },
        {
            "name": "--calibration_episodes",
            "type": int,
            "default": 100,
            "help": "Number of complete episodes for training-cost calibration.",
        },
        {
            "name": "--calibration_limit_fraction",
            "type": float,
            "default": 0.95,
            "help": "Fraction of baseline mean cost used for the recommended limit.",
        },
        {
            "name": "--command_x",
            "type": float,
            "default": 0.1,
            "help": "Fixed evaluation command in x velocity.",
        },
        {
            "name": "--train_command_x",
            "type": float,
            "help": "Fixed training command in x velocity. Overrides commands.ranges.lin_vel_x when provided.",
        },
        {
            "name": "--command_y",
            "type": float,
            "default": 0.0,
            "help": "Fixed evaluation command in y velocity.",
        },
        {
            "name": "--command_yaw",
            "type": float,
            "default": 0.0,
            "help": "Fixed evaluation command in yaw velocity.",
        },
        {
            "name": "--eval_gait_phase_offset",
            "type": float,
            "default": 0.0,
            "help": (
                "Evaluation-only gait clock offset in cycles. "
                "Use 0.5 to swap the initial left/right gait role."
            ),
        },
        {
            "name": "--eval_mirrored_policy",
            "action": "store_true",
            "default": False,
            "help": (
                "Evaluate the left-right mirrored counterpart of the policy "
                "by mirroring observations and actions."
            ),
        },
        {
            "name": "--output_dir",
            "type": str,
            "help": "Directory for evaluation CSV and metadata outputs.",
        },
        {
            "name": "--no_hard_exit_after_eval",
            "action": "store_true",
            "default": False,
            "help": "Disable os._exit(0) after successful energy evaluation output.",
        },
        {
            "name": "--no_eval_progress",
            "action": "store_true",
            "default": False,
            "help": "Disable progress output during energy evaluation.",
        },
        {
            "name": "--eval_progress_interval",
            "type": int,
            "default": 50,
            "help": "Refresh energy evaluation progress every N simulator steps.",
        },
        {
            "name": "--make_eval_report",
            "action": "store_true",
            "default": False,
            "help": "Generate an offline evaluation review report after energy evaluation.",
        },
        {
            "name": "--eval_report_dir",
            "type": str,
            "help": "Directory for the offline evaluation review report.",
        },
        {
            "name": "--no_eval_report_plots",
            "action": "store_true",
            "default": False,
            "help": "Skip PNG plot generation in the evaluation review report.",
        },
        {
            "name": "--review_dir",
            "type": str,
            "help": "Evaluation review directory containing representative_episodes.json.",
        },
        {
            "name": "--record_episode_ids",
            "type": str,
            "help": "Comma-separated episode ids to record instead of reading review_dir.",
        },
        {
            "name": "--record_episode_labels",
            "type": str,
            "help": "Comma-separated labels for record_episode_ids.",
        },
        {
            "name": "--max_video_episodes",
            "type": int,
            "default": 3,
            "help": "Maximum number of selected episodes to record.",
        },
        {
            "name": "--video_dir",
            "type": str,
            "help": "Directory for recorded episode videos. Defaults to review_dir.",
        },
        {
            "name": "--video_width",
            "type": int,
            "default": 1280,
            "help": "Recorded video width.",
        },
        {
            "name": "--video_height",
            "type": int,
            "default": 720,
            "help": "Recorded video height.",
        },
        {
            "name": "--video_fps",
            "type": int,
            "default": 100,
            "help": "Recorded video frame rate.",
        },
        {
            "name": "--video_stride",
            "type": int,
            "default": 1,
            "help": "Record one video frame every N policy steps.",
        },
        {
            "name": "--camera_offset",
            "type": str,
            "default": "1.0,-1.0,0.5",
            "help": "Camera offset from the robot body as x,y,z.",
        },
        {
            "name": "--camera_axis",
            "type": str,
            "default": "-0.3,0.2,1.0",
            "help": "Camera rotation axis as x,y,z.",
        },
        {
            "name": "--camera_angle_deg",
            "type": float,
            "default": 135.0,
            "help": "Camera rotation angle in degrees.",
        },
        {
            "name": "--no_hard_exit_after_video",
            "action": "store_true",
            "default": False,
            "help": "Disable os._exit(0) after successful video recording output.",
        },
        {
            "name": "--no_web_video_conversion",
            "action": "store_true",
            "default": False,
            "help": "Skip H.264 browser-compatible _web.mp4 conversion after recording.",
        },
        {
            "name": "--web_video_crf",
            "type": int,
            "default": 23,
            "help": "H.264 CRF quality for browser-compatible _web.mp4 conversion.",
        },
        {
            "name": "--web_video_preset",
            "type": str,
            "default": "veryfast",
            "help": "H.264 preset for browser-compatible _web.mp4 conversion.",
        },
    ]
    # parse arguments
    args = gymutil.parse_arguments(
        description="RL Policy", custom_parameters=custom_parameters
    )

    args.reducer_rated_torque = parse_reducer_rated_torque_spec(
        args.reducer_rated_torque
    )

    if args.auto_select_gpu:
        selected_gpu = selected_gpu_from_env()
        if selected_gpu is None:
            selected_gpu = select_idle_gpu(
                min_free_memory_mb=args.auto_gpu_min_free_memory_mb,
                max_utilization=args.auto_gpu_max_utilization,
            )

        cuda_device_id = selected_gpu["cuda_device_id"]
        if args.sim_device_type == "cuda":
            args.compute_device_id = cuda_device_id
        if str(args.rl_device).startswith("cuda"):
            args.rl_device = f"cuda:{cuda_device_id}"

        if selected_gpu_from_env() is None:
            print(
                "Auto-selected GPU "
                f"{selected_gpu['index']} as cuda:{cuda_device_id} "
                f"(free {selected_gpu['memory_free_mb']} MiB / "
                f"{selected_gpu['memory_total_mb']} MiB, "
                f"util {selected_gpu['utilization_gpu']}%)."
            )
        else:
            print(
                "Using auto-selected GPU "
                f"{selected_gpu['index']} as cuda:{cuda_device_id}."
            )

    # name allignment
    args.sim_device_id = args.compute_device_id
    args.sim_device = args.sim_device_type
    if args.sim_device == "cuda":
        args.sim_device += f":{args.sim_device_id}"
    return args


def export_policy_as_jit(actor_critic, path):
    os.makedirs(path, exist_ok=True)
    path = os.path.join(path, "policy_1.pt")
    model = copy.deepcopy(actor_critic.actor).to("cpu")
    traced_script_module = torch.jit.script(model)
    traced_script_module.save(path)
