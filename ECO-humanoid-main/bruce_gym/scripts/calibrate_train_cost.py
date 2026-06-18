# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2026 ECO Authors. All rights reserved.

"""Calibrate cost limits under the policy's randomized training distribution."""

import csv
import json
import math
import os
import sys
import time

from bruce_gym.gpu_auto_select import apply_auto_gpu_selection_from_argv

apply_auto_gpu_selection_from_argv()

import isaacgym  # noqa: F401
import torch
from tqdm import tqdm

from bruce_gym import LEGGED_GYM_ROOT_DIR
from bruce_gym.cost_calibration import summarize_episode_costs
from bruce_gym.envs import *  # noqa: F401,F403
from bruce_gym.rotor_energy import ROTOR_POSITIVE_8, SUPPORTED_ENERGY_COST_MODES
from bruce_gym.utils import get_args, task_registry
from bruce_gym.utils.helpers import class_to_dict


DEFAULT_CALIBRATION_ENVS = 1024


def _write_csv(path, rows):
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _set_calibration_config(env_cfg, args):
    configured_num_envs = int(env_cfg.env.num_envs)
    env_cfg.env.num_envs = (
        int(args.num_envs)
        if args.num_envs is not None
        else min(configured_num_envs, DEFAULT_CALIBRATION_ENVS)
    )
    env_cfg.commands.curriculum = False
    env_cfg.commands.ranges.lin_vel_x = [args.command_x, args.command_x]

    mode = args.energy_cost_mode or ROTOR_POSITIVE_8
    if mode not in SUPPORTED_ENERGY_COST_MODES:
        raise ValueError(
            f"Unsupported energy_cost_mode '{mode}'. "
            f"Supported modes are: {SUPPORTED_ENERGY_COST_MODES}"
        )
    env_cfg.env.energy_cost_mode = mode
    return mode


def _default_output_dir(args, train_cfg, mode):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    checkpoint = train_cfg.runner.checkpoint
    name = (
        f"{timestamp}_{args.task}_{mode}_train_distribution_"
        f"vx{args.command_x:g}_model{checkpoint}"
    )
    return os.path.join(LEGGED_GYM_ROOT_DIR, "energy_calibrations", name)


def _collect_complete_episodes(env, policy, args):
    target_episodes = int(args.calibration_episodes)
    if target_episodes <= 0:
        raise ValueError("calibration_episodes must be positive.")

    device = env.device
    episode_cost = torch.zeros(env.num_envs, dtype=torch.float, device=device)
    episode_body_distance = torch.zeros_like(episode_cost)
    episode_steps = torch.zeros(
        env.num_envs, dtype=torch.long, device=device
    )
    episode_rows = []

    env.commands[:, 0] = args.command_x
    env.compute_observations()
    obs = env.get_observations()

    episode_batches = math.ceil(target_episodes / env.num_envs)
    max_steps = int(env.max_episode_length * (episode_batches + 2))
    progress = tqdm(
        total=target_episodes,
        desc="Calibrating training cost",
        unit="episode",
        dynamic_ncols=True,
        file=sys.stdout,
    )

    try:
        with torch.inference_mode():
            for _ in range(max_steps):
                # Lock only the forward command. Lateral/yaw/heading behavior and
                # every training randomization remain under the environment config.
                env.commands[:, 0] = args.command_x
                actions = policy(obs.detach())
                obs, _, _, dones, _, costs = env.step(actions.detach())

                episode_cost += costs[0].reshape(-1)
                episode_body_distance += env.base_lin_vel[:, 0] * env.dt
                episode_steps += 1

                done_ids = (dones > 0).nonzero(as_tuple=False).flatten()
                if done_ids.numel() == 0:
                    continue

                remaining = target_episodes - len(episode_rows)
                record_ids = done_ids
                if done_ids.numel() > remaining:
                    # Simultaneous timeouts are common. Sample evenly across the
                    # batch instead of biasing calibration toward low env IDs,
                    # which may share nearby terrain assignments.
                    positions = torch.linspace(
                        0,
                        done_ids.numel() - 1,
                        steps=remaining,
                        device=done_ids.device,
                    ).round().long()
                    record_ids = done_ids[positions]

                batch_costs = episode_cost[record_ids].detach().cpu().tolist()
                batch_distances = (
                    episode_body_distance[record_ids].detach().cpu().tolist()
                )
                batch_steps = episode_steps[record_ids].detach().cpu().tolist()
                batch_timeouts = (
                    env.time_out_buf[record_ids].detach().cpu().tolist()
                )
                batch_env_ids = record_ids.detach().cpu().tolist()

                for env_id, cost1, distance, steps, timeout in zip(
                    batch_env_ids,
                    batch_costs,
                    batch_distances,
                    batch_steps,
                    batch_timeouts,
                ):
                    if len(episode_rows) >= target_episodes:
                        break
                    duration_s = int(steps) * env.dt
                    is_timeout = bool(timeout)
                    episode_rows.append(
                        {
                            "episode_id": len(episode_rows),
                            "env_id": int(env_id),
                            "episode_outcome": (
                                "success" if is_timeout else "fall"
                            ),
                            "timeout": float(is_timeout),
                            "fall": float(not is_timeout),
                            "steps": int(steps),
                            "duration_s": duration_s,
                            "cost1": float(cost1),
                            "cost1_per_second": float(cost1)
                            / max(duration_s, 1e-8),
                            "body_frame_distance_x": float(distance),
                            "mean_body_frame_velocity_x": float(distance)
                            / max(duration_s, 1e-8),
                        }
                    )
                    progress.update(1)

                episode_cost[done_ids] = 0.0
                episode_body_distance[done_ids] = 0.0
                episode_steps[done_ids] = 0

                if len(episode_rows) >= target_episodes:
                    break
            else:
                raise RuntimeError(
                    "Calibration ended before collecting the requested number "
                    f"of episodes: {len(episode_rows)}/{target_episodes}."
                )
    finally:
        progress.close()

    return episode_rows


def calibrate(args):
    if args.task == "XBotL_free":
        args.task = "bruce_ppolag"
    if not 0.0 < args.calibration_limit_fraction <= 1.0:
        raise ValueError(
            "calibration_limit_fraction must be in the interval (0, 1]."
        )

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    mode = _set_calibration_config(env_cfg, args)
    train_cfg.runner.resume = True
    # Calibration needs policy weights only. Avoid restoring optimizer and
    # Lagrange state from checkpoints created under a different cost scale.
    args.reset_optimizer_on_resume = True
    args.reset_lagrange_on_resume = True

    env, env_cfg = task_registry.make_env(
        name=args.task, args=args, env_cfg=env_cfg
    )
    runner, train_cfg = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg
    )
    policy = runner.get_inference_policy(device=env.device)

    episode_rows = _collect_complete_episodes(env, policy, args)
    summary = summarize_episode_costs(
        episode_rows, limit_fraction=args.calibration_limit_fraction
    )

    output_dir = os.path.abspath(
        args.output_dir or _default_output_dir(args, train_cfg, mode)
    )
    os.makedirs(output_dir, exist_ok=True)
    metadata = {
        "task": args.task,
        "load_run": train_cfg.runner.load_run,
        "checkpoint": train_cfg.runner.checkpoint,
        "seed": env_cfg.seed,
        "energy_cost_mode": env.energy_cost_mode,
        "calibration_profile": "training_distribution_fixed_lin_vel_x",
        "calibration_episodes": args.calibration_episodes,
        "calibration_limit_fraction": args.calibration_limit_fraction,
        "num_envs": env.num_envs,
        "command_x": args.command_x,
        "heading_command": bool(env_cfg.commands.heading_command),
        "terrain": class_to_dict(env_cfg.terrain),
        "noise": class_to_dict(env_cfg.noise),
        "domain_rand": class_to_dict(env_cfg.domain_rand),
        "policy_dt": env.dt,
        "sim_dt": env.sim_params.dt,
        "cost_accumulation": (
            "sum of returned cost1 over each complete episode, matching "
            "the population used by Train/mean_cost1"
        ),
        "checkpoint_state_loaded": "policy_weights_only",
        "output_dir": output_dir,
        "hard_exit_after_calibration": not args.no_hard_exit_after_eval,
    }

    _write_csv(os.path.join(output_dir, "episode_costs.csv"), episode_rows)
    with open(os.path.join(output_dir, "calibration_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print("\nCalibration summary:")
    print(json.dumps(summary, indent=2))
    print(f"Wrote training-distribution calibration to: {output_dir}")
    sys.stdout.flush()
    sys.stderr.flush()

    if not args.no_hard_exit_after_eval:
        os._exit(0)
    return summary


if __name__ == "__main__":
    calibrate(get_args())
