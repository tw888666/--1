# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2026 ECO Authors. All rights reserved.

import csv
import json
import os
import time
from collections import defaultdict

import torch

from bruce_gym import LEGGED_GYM_ROOT_DIR
from bruce_gym.envs import *  # noqa: F401,F403
from bruce_gym.rotor_energy import BRUCE_ROTOR_NAMES, SUPPORTED_ENERGY_COST_MODES
from bruce_gym.utils import get_args, task_registry


JOINT_NAMES = (
    "hip_yaw_l",
    "hip_pitch_l",
    "hip_roll_l",
    "knee_pitch_l",
    "ankle_pitch_l",
    "hip_yaw_r",
    "hip_pitch_r",
    "hip_roll_r",
    "knee_pitch_r",
    "ankle_pitch_r",
)


def _to_float(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().item()
    if hasattr(value, "item"):
        return value.item()
    return float(value)


def _tensor_list(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().reshape(-1).tolist()
    return list(value)


def _write_dicts_csv(path, rows):
    fieldnames = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with open(path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _set_eval_config(env_cfg, args):
    env_cfg.env.num_envs = 1
    env_cfg.env.episode_length_s = 24
    env_cfg.terrain.mesh_type = "plane"
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.measure_heights = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.disturbance = False
    env_cfg.domain_rand.delay = False
    env_cfg.domain_rand.randomize_kp = False
    env_cfg.domain_rand.randomize_kd = False
    if hasattr(env_cfg.domain_rand, "randomize_motor_strength"):
        env_cfg.domain_rand.randomize_motor_strength = False
    if args.energy_cost_mode is not None:
        env_cfg.env.energy_cost_mode = args.energy_cost_mode


def _set_fixed_command(env, args):
    env.commands[:, 0] = args.command_x
    env.commands[:, 1] = args.command_y
    env.commands[:, 2] = args.command_yaw
    if env.commands.shape[1] > 3:
        env.commands[:, 3] = 0.0


def _phase_label(left_contact, right_contact):
    if left_contact and right_contact:
        return "double_support"
    if left_contact:
        return "left_support"
    if right_contact:
        return "right_support"
    return "flight_or_transition"


def _add_motor_fields(row, prefix, values):
    for idx, motor_name in enumerate(BRUCE_ROTOR_NAMES):
        row[f"{motor_name}_{prefix}"] = values[idx]


def _add_joint_fields(row, prefix, values):
    for idx, joint_name in enumerate(JOINT_NAMES):
        row[f"{joint_name}_{prefix}"] = values[idx]


def _step_energy_snapshot(env, robot_index):
    rotor_pos = _tensor_list(env.rotor_drive_energy_per_motor[robot_index])
    rotor_neg = _tensor_list(env.rotor_brake_energy_per_motor[robot_index])
    yaw_pos = _tensor_list(env.yaw_joint_drive_energy[robot_index])
    yaw_neg = _tensor_list(env.yaw_joint_brake_energy[robot_index])
    joint_pos = _tensor_list(env.joint_drive_energy_per_joint[robot_index])
    joint_neg = _tensor_list(env.joint_brake_energy_per_joint[robot_index])

    return {
        "rotor_pos": rotor_pos,
        "rotor_neg": rotor_neg,
        "yaw_pos": yaw_pos,
        "yaw_neg": yaw_neg,
        "joint_pos": joint_pos,
        "joint_neg": joint_neg,
        "rotor_pos_total": sum(rotor_pos),
        "rotor_neg_total": sum(rotor_neg),
        "yaw_pos_total": sum(yaw_pos),
        "yaw_neg_total": sum(yaw_neg),
        "joint_pos_total": sum(joint_pos),
        "joint_neg_total": sum(joint_neg),
    }


def _new_episode_accumulator():
    return {
        "steps": 0,
        "distance_x": 0.0,
        "rotor_pos": [0.0] * len(BRUCE_ROTOR_NAMES),
        "rotor_neg": [0.0] * len(BRUCE_ROTOR_NAMES),
        "yaw_pos": [0.0, 0.0],
        "yaw_neg": [0.0, 0.0],
        "joint_pos": [0.0] * len(JOINT_NAMES),
        "joint_neg": [0.0] * len(JOINT_NAMES),
    }


def _accumulate_episode(acc, snapshot, base_vel_x, dt):
    acc["steps"] += 1
    acc["distance_x"] += base_vel_x * dt
    for key in ("rotor_pos", "rotor_neg", "yaw_pos", "yaw_neg", "joint_pos", "joint_neg"):
        for idx, value in enumerate(snapshot[key]):
            acc[key][idx] += value


def _episode_summary_row(episode_id, acc, timeout, dt):
    duration_s = acc["steps"] * dt
    row = {
        "episode_id": episode_id,
        "timeout": float(timeout),
        "fall": float(not timeout),
        "duration_s": duration_s,
        "distance_x": acc["distance_x"],
        "mean_velocity_x": acc["distance_x"] / max(duration_s, 1e-8),
        "rotor_positive_energy_8": sum(acc["rotor_pos"]),
        "rotor_negative_energy_8": sum(acc["rotor_neg"]),
        "yaw_positive_energy_2": sum(acc["yaw_pos"]),
        "yaw_negative_energy_2": sum(acc["yaw_neg"]),
        "joint_positive_energy_10": sum(acc["joint_pos"]),
        "joint_negative_energy_10": sum(acc["joint_neg"]),
    }
    _add_motor_fields(row, "positive_energy", acc["rotor_pos"])
    _add_motor_fields(row, "negative_energy", acc["rotor_neg"])
    _add_joint_fields(row, "positive_energy", acc["joint_pos"])
    _add_joint_fields(row, "negative_energy", acc["joint_neg"])
    return row


def _init_phase_accumulators():
    return defaultdict(
        lambda: {
            "valid_steps": 0,
            "rotor_positive_energy_8": 0.0,
            "rotor_negative_energy_8": 0.0,
            "joint_positive_energy_10": 0.0,
            "joint_negative_energy_10": 0.0,
        }
    )


def _record_phase(phase_acc, phase, snapshot):
    phase_acc[phase]["valid_steps"] += 1
    phase_acc[phase]["rotor_positive_energy_8"] += snapshot["rotor_pos_total"]
    phase_acc[phase]["rotor_negative_energy_8"] += snapshot["rotor_neg_total"]
    phase_acc[phase]["joint_positive_energy_10"] += snapshot["joint_pos_total"]
    phase_acc[phase]["joint_negative_energy_10"] += snapshot["joint_neg_total"]


def _phase_summary_rows(phase_acc, dt):
    rows = []
    for phase, values in sorted(phase_acc.items()):
        row = {"phase": phase, **values}
        row["duration_s"] = values["valid_steps"] * dt
        rows.append(row)
    return rows


def _default_output_dir(args):
    mode = args.energy_cost_mode or "cfg"
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    name = f"{timestamp}_{args.task}_{mode}_vx{args.command_x:g}"
    return os.path.join(LEGGED_GYM_ROOT_DIR, "energy_evaluations", name)


def evaluate(args):
    if args.task == "XBotL_free":
        args.task = "bruce_ppolag"
    if args.energy_cost_mode is not None and args.energy_cost_mode not in SUPPORTED_ENERGY_COST_MODES:
        raise ValueError(
            f"Unsupported energy_cost_mode '{args.energy_cost_mode}'. "
            f"Supported modes are: {SUPPORTED_ENERGY_COST_MODES}"
        )

    output_dir = args.output_dir or _default_output_dir(args)
    os.makedirs(output_dir, exist_ok=True)

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    _set_eval_config(env_cfg, args)
    train_cfg.runner.resume = True

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    _set_fixed_command(env, args)
    env.compute_observations()
    obs = env.get_observations()

    runner, train_cfg = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg
    )
    policy = runner.get_inference_policy(device=env.device)

    metadata = {
        "task": args.task,
        "load_run": train_cfg.runner.load_run,
        "checkpoint": train_cfg.runner.checkpoint,
        "energy_cost_mode": env.energy_cost_mode,
        "num_eval_episodes": args.num_eval_episodes,
        "command_x": args.command_x,
        "command_y": args.command_y,
        "command_yaw": args.command_yaw,
        "policy_dt": env.dt,
        "sim_dt": env.sim_params.dt,
        "output_dir": output_dir,
    }

    step_rows = []
    episode_rows = []
    phase_acc = _init_phase_accumulators()
    episode_acc = _new_episode_accumulator()
    episode_id = 0
    episode_step = 0
    robot_index = 0
    max_steps = int(args.num_eval_episodes * env.max_episode_length * 3)

    with torch.inference_mode():
        for global_step in range(max_steps):
            _set_fixed_command(env, args)
            actions = policy(obs.detach())
            obs, _, _, dones, infos, costs = env.step(actions.detach())

            done = bool(dones[robot_index].item())
            timeout = bool(env.time_out_buf[robot_index].item())
            valid_state = not done
            snapshot = _step_energy_snapshot(env, robot_index)
            base_vel_x = _to_float(env.base_lin_vel[robot_index, 0])
            _accumulate_episode(episode_acc, snapshot, base_vel_x, env.dt)

            row = {
                "global_step": global_step,
                "episode_id": episode_id,
                "episode_step": episode_step,
                "done": float(done),
                "timeout": float(timeout),
                "valid_state": float(valid_state),
                "cost1": _to_float(costs[0][robot_index]),
                "command_x": args.command_x,
                "command_y": args.command_y,
                "command_yaw": args.command_yaw,
                "rotor_positive_energy_8": snapshot["rotor_pos_total"],
                "rotor_negative_energy_8": snapshot["rotor_neg_total"],
                "yaw_positive_energy_2": snapshot["yaw_pos_total"],
                "yaw_negative_energy_2": snapshot["yaw_neg_total"],
                "joint_positive_energy_10": snapshot["joint_pos_total"],
                "joint_negative_energy_10": snapshot["joint_neg_total"],
            }
            _add_motor_fields(row, "positive_energy", snapshot["rotor_pos"])
            _add_motor_fields(row, "negative_energy", snapshot["rotor_neg"])
            _add_motor_fields(row, "power", _tensor_list(env.rotor_power[robot_index]))
            _add_motor_fields(row, "torque", _tensor_list(env.rotor_torque[robot_index]))
            _add_motor_fields(row, "velocity", _tensor_list(env.rotor_velocity[robot_index]))
            _add_joint_fields(row, "positive_energy", snapshot["joint_pos"])
            _add_joint_fields(row, "negative_energy", snapshot["joint_neg"])
            _add_joint_fields(row, "power", _tensor_list(env.joint_power[robot_index]))

            if valid_state:
                foot_force_z = env.contact_forces[robot_index, env.feet_indices, 2]
                left_contact = bool((foot_force_z[0] > 5.0).item())
                right_contact = bool((foot_force_z[1] > 5.0).item())
                phase = _phase_label(left_contact, right_contact)
                row.update(
                    {
                        "gait_phase": _to_float(torch.remainder(env._get_phase()[robot_index], 1.0)),
                        "phase_label": phase,
                        "left_contact_state": float(left_contact),
                        "right_contact_state": float(right_contact),
                        "left_contact_force_z": _to_float(foot_force_z[0]),
                        "right_contact_force_z": _to_float(foot_force_z[1]),
                        "base_vel_x": base_vel_x,
                        "base_vel_y": _to_float(env.base_lin_vel[robot_index, 1]),
                        "base_vel_z": _to_float(env.base_lin_vel[robot_index, 2]),
                    }
                )
                _record_phase(phase_acc, phase, snapshot)
            step_rows.append(row)

            episode_step += 1
            if done:
                episode_rows.append(
                    _episode_summary_row(episode_id, episode_acc, timeout, env.dt)
                )
                episode_id += 1
                episode_step = 0
                episode_acc = _new_episode_accumulator()
                if episode_id >= args.num_eval_episodes:
                    break
        else:
            raise RuntimeError(
                f"Evaluation stopped after {max_steps} steps before collecting "
                f"{args.num_eval_episodes} episodes."
            )

    with open(os.path.join(output_dir, "metadata.json"), "w") as jsonfile:
        json.dump(metadata, jsonfile, indent=2)
    _write_dicts_csv(os.path.join(output_dir, "episode_summary.csv"), episode_rows)
    _write_dicts_csv(os.path.join(output_dir, "step_timeseries.csv"), step_rows)
    _write_dicts_csv(
        os.path.join(output_dir, "phase_summary.csv"),
        _phase_summary_rows(phase_acc, env.dt),
    )

    print(f"Wrote energy evaluation outputs to: {output_dir}")


if __name__ == "__main__":
    evaluate(get_args())
