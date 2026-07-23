# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2026 ECO Authors. All rights reserved.

"""Calibrate cost limits under the policy's randomized training distribution."""

import csv
import json
import os
import sys

from bruce_gym.gpu_auto_select import apply_auto_gpu_selection_from_argv

apply_auto_gpu_selection_from_argv()

import isaacgym  # noqa: F401
import torch
from tqdm import tqdm

from bruce_gym import LEGGED_GYM_ROOT_DIR
from bruce_gym.cost_calibration import (
    evenly_spaced_indices,
    summarize_episode_costs,
)
from bruce_gym.envs import *  # noqa: F401,F403
from bruce_gym.paired_evaluation import canonical_fingerprint
from bruce_gym.rotor_energy import (
    CONTROL_ENERGY_COST_MODES,
    REDUCER_CORRECTED_8,
    REDUCER_POSITIVE_EFFICIENCY_OFFSET,
    REDUCER_POSITIVE_EFFICIENCY_SCALE,
    REDUCER_RATED_TORQUE_NM,
    ROTOR_MIXED_8,
    SUPPORTED_ENERGY_COST_MODES,
    bruce_rotor_names_from_dof_names,
    energy_costs_from_policy_step_buffers,
)
from bruce_gym.naming import policy_id, reducer_rated_torque_suffix
from bruce_gym.utils import get_args, task_registry
from bruce_gym.utils.helpers import class_to_dict


DEFAULT_CALIBRATION_ENVS = 1024


def _write_csv(path, rows):
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _selected_tensor_rows(env, cohort_ids, attribute):
    value = getattr(env, attribute, None)
    if not isinstance(value, torch.Tensor):
        return None
    return value.index_select(0, cohort_ids).detach().cpu().tolist()


def _build_scenario_manifest(env, cohort_ids, seed, command_x):
    tensor_attributes = (
        "terrain_levels",
        "terrain_types",
        "env_origins",
        "commands",
        "root_states",
        "dof_pos",
        "dof_vel",
        "Kp_factors",
        "Kd_factors",
        "payload",
        "com_displacement",
        "euler_rand",
        "friction_coeffs",
        "restitution_coeffs",
        "joint_friction_coeffs",
        "joint_armature_coeffs",
        "body_mass",
    )
    tensors = {}
    for attribute in tensor_attributes:
        rows = _selected_tensor_rows(env, cohort_ids, attribute)
        if rows is not None:
            tensors[attribute] = rows
    manifest = {
        "schema_version": 1,
        "seed": int(seed),
        "command_x": float(command_x),
        "num_envs": int(env.num_envs),
        "cohort_env_ids": cohort_ids.detach().cpu().tolist(),
        "initial_tensors": tensors,
        "scope": (
            "Initial scenario pairing. Runtime noise, delay, motor-strength, "
            "push, and disturbance streams remain seeded but are not replayed "
            "from this manifest."
        ),
    }
    manifest["sha256"] = canonical_fingerprint(manifest)
    return manifest


def _set_calibration_config(env_cfg, args):
    configured_num_envs = int(env_cfg.env.num_envs)
    env_cfg.env.num_envs = (
        int(args.num_envs)
        if args.num_envs is not None
        else min(configured_num_envs, DEFAULT_CALIBRATION_ENVS)
    )
    env_cfg.commands.curriculum = False
    env_cfg.commands.ranges.lin_vel_x = [args.command_x, args.command_x]

    mode = args.energy_cost_mode or ROTOR_MIXED_8
    if mode not in SUPPORTED_ENERGY_COST_MODES:
        raise ValueError(
            f"Unsupported energy_cost_mode '{mode}'. "
            f"Supported modes are: {SUPPORTED_ENERGY_COST_MODES}"
        )
    env_cfg.env.energy_cost_mode = mode
    return mode


def _default_output_dir(args, train_cfg, mode):
    checkpoint = train_cfg.runner.checkpoint
    suffix = None
    if mode == REDUCER_CORRECTED_8:
        suffix = reducer_rated_torque_suffix(
            args.reducer_rated_torque or REDUCER_RATED_TORQUE_NM
        )
    name = policy_id(args.command_x, mode, args.seed, checkpoint, suffix)
    group = f"train_dist{int(args.calibration_episodes)}"
    return os.path.join(LEGGED_GYM_ROOT_DIR, "energy_calibrations", group, name)


def _collect_complete_episodes(env, policy, args):
    target_episodes = int(args.calibration_episodes)
    if target_episodes <= 0:
        raise ValueError("calibration_episodes must be positive.")
    if target_episodes > env.num_envs:
        raise ValueError(
            "calibration_episodes must not exceed num_envs. Fixed-cohort "
            "sampling requires one independent environment per episode."
        )

    device = env.device
    episode_cost = torch.zeros(env.num_envs, dtype=torch.float, device=device)
    reported_cost_modes = list(CONTROL_ENERGY_COST_MODES)
    if env.energy_cost_mode not in reported_cost_modes:
        reported_cost_modes.append(env.energy_cost_mode)
    episode_costs_by_mode = {
        mode: torch.zeros_like(episode_cost)
        for mode in reported_cost_modes
    }
    episode_body_distance = torch.zeros_like(episode_cost)
    episode_steps = torch.zeros(
        env.num_envs, dtype=torch.long, device=device
    )
    episode_rows = []
    cohort_ids = torch.tensor(
        evenly_spaced_indices(env.num_envs, target_episodes),
        dtype=torch.long,
        device=device,
    )
    pending_cohort = torch.zeros(
        env.num_envs, dtype=torch.bool, device=device
    )
    pending_cohort[cohort_ids] = True

    env.commands[:, 0] = args.command_x
    evaluation_seed = (
        args.seed if args.seed is not None else getattr(env.cfg, "seed", 0)
    )
    scenario_manifest = _build_scenario_manifest(
        env, cohort_ids, evaluation_seed, args.command_x
    )
    env.compute_observations()
    obs = env.get_observations()

    max_steps = int(env.max_episode_length * 2)
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
                step_costs_by_mode = energy_costs_from_policy_step_buffers(
                    env.joint_power,
                    env.joint_drive_energy_per_joint,
                    env.joint_brake_energy_per_joint,
                    env.rotor_drive_energy,
                    env.rotor_brake_energy,
                    (
                        env.reducer_corrected_energy
                        if env.reducer_rated_torque is not None
                        else None
                    ),
                )
                for mode, step_cost in step_costs_by_mode.items():
                    episode_costs_by_mode[mode] += step_cost.reshape(-1)
                episode_body_distance += env.base_lin_vel[:, 0] * env.dt
                episode_steps += 1

                done_ids = (dones > 0).nonzero(as_tuple=False).flatten()
                if done_ids.numel() == 0:
                    continue

                # Record exactly one first complete episode from each member of
                # a cohort selected before simulation. This prevents early falls
                # from dominating a "first N completions" sample.
                record_ids = done_ids[pending_cohort[done_ids]]

                if record_ids.numel() > 0:
                    batch_costs = (
                        episode_cost[record_ids].detach().cpu().tolist()
                    )
                    batch_distances = (
                        episode_body_distance[record_ids].detach().cpu().tolist()
                    )
                    batch_steps = (
                        episode_steps[record_ids].detach().cpu().tolist()
                    )
                    batch_timeouts = (
                        env.time_out_buf[record_ids].detach().cpu().tolist()
                    )
                    batch_env_ids = record_ids.detach().cpu().tolist()
                    batch_costs_by_mode = {
                        mode: values[record_ids].detach().cpu().tolist()
                        for mode, values in episode_costs_by_mode.items()
                    }

                    for batch_index, (
                        env_id,
                        cost1,
                        distance,
                        steps,
                        timeout,
                    ) in enumerate(
                        zip(
                            batch_env_ids,
                            batch_costs,
                            batch_distances,
                            batch_steps,
                            batch_timeouts,
                        )
                    ):
                        duration_s = int(steps) * env.dt
                        is_timeout = bool(timeout)
                        row = {
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
                        for mode in reported_cost_modes:
                            row[f"cost1_{mode}"] = float(
                                batch_costs_by_mode[mode][batch_index]
                            )
                        episode_rows.append(row)
                        progress.update(1)
                    pending_cohort[record_ids] = False

                episode_cost[done_ids] = 0.0
                for values in episode_costs_by_mode.values():
                    values[done_ids] = 0.0
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

    return episode_rows, scenario_manifest


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

    env, env_cfg = task_registry.make_env(
        name=args.task, args=args, env_cfg=env_cfg
    )
    runner, train_cfg = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg
    )
    policy = runner.get_inference_policy(device=env.device)

    episode_rows, scenario_manifest = _collect_complete_episodes(
        env, policy, args
    )
    summary = summarize_episode_costs(
        episode_rows, limit_fraction=args.calibration_limit_fraction
    )
    cost_mode_summaries = {}
    reported_cost_modes = list(CONTROL_ENERGY_COST_MODES)
    if mode not in reported_cost_modes:
        reported_cost_modes.append(mode)
    for reported_mode in reported_cost_modes:
        mode_rows = []
        for row in episode_rows:
            mode_row = dict(row)
            mode_row["cost1"] = row[f"cost1_{reported_mode}"]
            mode_rows.append(mode_row)
        cost_mode_summaries[reported_mode] = summarize_episode_costs(
            mode_rows, limit_fraction=args.calibration_limit_fraction
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
        "reducer_rated_torque": (
            env.reducer_rated_torque.detach().cpu().tolist()
            if env.reducer_rated_torque is not None
            else None
        ),
        "reducer_positive_efficiency_scale": REDUCER_POSITIVE_EFFICIENCY_SCALE,
        "reducer_positive_efficiency_offset": REDUCER_POSITIVE_EFFICIENCY_OFFSET,
        "reducer_efficiency_input": "abs(output_torque) / rated_torque",
        "reducer_negative_power_efficiency": 1.0,
        "rotor_names": list(bruce_rotor_names_from_dof_names(env.dof_names)),
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
        "sampling_method": (
            "one first complete episode from each member of an evenly spaced "
            "fixed environment cohort"
        ),
        "reported_cost_modes": reported_cost_modes,
        "scenario_manifest": "scenario_manifest.json",
        "scenario_fingerprint_sha256": scenario_manifest["sha256"],
        "scenario_pairing_scope": scenario_manifest["scope"],
        "checkpoint_state_loaded": "policy_weights_only",
        "output_dir": output_dir,
        "hard_exit_after_calibration": not args.no_hard_exit_after_eval,
    }

    _write_csv(os.path.join(output_dir, "episode_costs.csv"), episode_rows)
    with open(os.path.join(output_dir, "calibration_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(output_dir, "cost_mode_summaries.json"), "w") as f:
        json.dump(cost_mode_summaries, f, indent=2)
    with open(os.path.join(output_dir, "scenario_manifest.json"), "w") as f:
        json.dump(scenario_manifest, f, indent=2)
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
