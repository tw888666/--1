# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2026 ECO Authors. All rights reserved.

import csv
import json
import os
import sys
import time
from collections import defaultdict

import isaacgym  # noqa: F401
import torch

from bruce_gym import LEGGED_GYM_ROOT_DIR
from bruce_gym.envs import *  # noqa: F401,F403
from bruce_gym.rotor_energy import (
    BRUCE_EXPECTED_DOF_NAMES,
    BRUCE_ROTOR_NAMES,
    SUPPORTED_ENERGY_COST_MODES,
    bruce_joint_names_from_dof_names,
    bruce_rotor_names_from_dof_names,
)
from bruce_gym.utils import get_args, task_registry


JOINT_NAMES = BRUCE_EXPECTED_DOF_NAMES
ROTOR_NAMES = BRUCE_ROTOR_NAMES


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
    env_cfg.commands.heading_command = False
    env_cfg.commands.ranges.lin_vel_x = [args.command_x, args.command_x]
    env_cfg.commands.ranges.lin_vel_y = [args.command_y, args.command_y]
    env_cfg.commands.ranges.ang_vel_yaw = [args.command_yaw, args.command_yaw]
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.disturbance = False
    env_cfg.domain_rand.delay = False
    env_cfg.domain_rand.randomize_dof_init = False
    env_cfg.domain_rand.randomize_euler = False
    env_cfg.domain_rand.randomize_payload_mass = False
    env_cfg.domain_rand.randomize_com_displacement = False
    env_cfg.domain_rand.randomize_link_mass = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.randomize_restitution = False
    env_cfg.domain_rand.randomize_joint_friction = False
    env_cfg.domain_rand.randomize_joint_armature = False
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
    for idx, motor_name in enumerate(ROTOR_NAMES):
        row[f"{motor_name}_{prefix}"] = values[idx]


def _add_joint_fields(row, prefix, values):
    for idx, joint_name in enumerate(JOINT_NAMES):
        row[f"{joint_name}_{prefix}"] = values[idx]


def _resolve_foot_contact_indices(feet_names):
    if "ankle_pitch_link_l" in feet_names and "ankle_pitch_link_r" in feet_names:
        return (
            feet_names.index("ankle_pitch_link_l"),
            feet_names.index("ankle_pitch_link_r"),
        )
    if len(feet_names) >= 2:
        return 0, 1
    raise ValueError(f"Expected at least two BRUCE feet, got: {feet_names}")


def _pre_step_state(env, robot_index, left_foot_idx, right_foot_idx):
    foot_force_z = env.contact_forces[robot_index, env.feet_indices, 2]
    left_contact = bool((foot_force_z[left_foot_idx] > 5.0).item())
    right_contact = bool((foot_force_z[right_foot_idx] > 5.0).item())
    base_vel = _tensor_list(env.base_lin_vel[robot_index])
    root_pos = _tensor_list(env.root_states[robot_index, :3])
    return {
        "gait_phase": _to_float(torch.remainder(env._get_phase()[robot_index], 1.0)),
        "phase_label": _phase_label(left_contact, right_contact),
        "left_contact_state": float(left_contact),
        "right_contact_state": float(right_contact),
        "left_contact_force_z": _to_float(foot_force_z[left_foot_idx]),
        "right_contact_force_z": _to_float(foot_force_z[right_foot_idx]),
        "pre_step_base_vel_x": base_vel[0],
        "pre_step_base_vel_y": base_vel[1],
        "pre_step_base_vel_z": base_vel[2],
        "pre_step_root_pos_x": root_pos[0],
        "pre_step_root_pos_y": root_pos[1],
        "pre_step_root_pos_z": root_pos[2],
        "base_vel_x": base_vel[0],
        "base_vel_y": base_vel[1],
        "base_vel_z": base_vel[2],
    }


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
        "dynamic_steps": 0,
        "distance_x": 0.0,
        "path_length_xy": 0.0,
        "body_frame_distance_x": 0.0,
        "rotor_pos": [0.0] * len(ROTOR_NAMES),
        "rotor_neg": [0.0] * len(ROTOR_NAMES),
        "yaw_pos": [0.0, 0.0],
        "yaw_neg": [0.0, 0.0],
        "joint_pos": [0.0] * len(JOINT_NAMES),
        "joint_neg": [0.0] * len(JOINT_NAMES),
    }


def _accumulate_episode(
    acc, snapshot, pre_step_state, post_step_root_pos, dynamic_state_valid, dt
):
    world_dx = post_step_root_pos[0] - pre_step_state["pre_step_root_pos_x"]
    world_dy = post_step_root_pos[1] - pre_step_state["pre_step_root_pos_y"]
    acc["steps"] += 1
    acc["distance_x"] += world_dx
    acc["path_length_xy"] += (world_dx * world_dx + world_dy * world_dy) ** 0.5
    if dynamic_state_valid:
        acc["dynamic_steps"] += 1
        acc["body_frame_distance_x"] += pre_step_state["pre_step_base_vel_x"] * dt
    for key in (
        "rotor_pos",
        "rotor_neg",
        "yaw_pos",
        "yaw_neg",
        "joint_pos",
        "joint_neg",
    ):
        for idx, value in enumerate(snapshot[key]):
            acc[key][idx] += value


def _episode_summary_row(episode_id, acc, timeout, dt):
    duration_s = acc["steps"] * dt
    dynamic_duration_s = acc["dynamic_steps"] * dt
    episode_outcome = "success" if timeout else "fall"
    row = {
        "episode_id": episode_id,
        "episode_outcome": episode_outcome,
        "timeout": float(timeout),
        "fall": float(not timeout),
        "duration_s": duration_s,
        "dynamic_duration_s": dynamic_duration_s,
        "dynamic_steps": acc["dynamic_steps"],
        "distance_x": acc["distance_x"],
        "mean_velocity_x": acc["distance_x"] / max(duration_s, 1e-8),
        "path_length_xy": acc["path_length_xy"],
        "mean_path_velocity_xy": acc["path_length_xy"] / max(duration_s, 1e-8),
        "body_frame_distance_x": acc["body_frame_distance_x"],
        "mean_body_frame_velocity_x": acc["body_frame_distance_x"]
        / max(dynamic_duration_s, 1e-8),
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


def _empty_phase_stats():
    return {
        "episodes": 0,
        "valid_steps": 0,
        "rotor_pos": [0.0] * len(ROTOR_NAMES),
        "rotor_neg": [0.0] * len(ROTOR_NAMES),
        "yaw_pos": [0.0, 0.0],
        "yaw_neg": [0.0, 0.0],
        "joint_pos": [0.0] * len(JOINT_NAMES),
        "joint_neg": [0.0] * len(JOINT_NAMES),
    }


def _init_phase_accumulators():
    return defaultdict(_empty_phase_stats)


def _add_energy_lists(values, snapshot):
    for target_key, snapshot_key in (
        ("rotor_pos", "rotor_pos"),
        ("rotor_neg", "rotor_neg"),
        ("yaw_pos", "yaw_pos"),
        ("yaw_neg", "yaw_neg"),
        ("joint_pos", "joint_pos"),
        ("joint_neg", "joint_neg"),
    ):
        for idx, value in enumerate(snapshot[snapshot_key]):
            values[target_key][idx] += value


def _record_phase(phase_acc, phase, snapshot):
    phase_acc[phase]["valid_steps"] += 1
    _add_energy_lists(phase_acc[phase], snapshot)


def _merge_phase_accumulators(target, source, episode_outcome):
    for phase, source_values in source.items():
        values = target[(episode_outcome, phase)]
        values["episodes"] += 1
        values["valid_steps"] += source_values["valid_steps"]
        for key in (
            "rotor_pos",
            "rotor_neg",
            "yaw_pos",
            "yaw_neg",
            "joint_pos",
            "joint_neg",
        ):
            for idx, value in enumerate(source_values[key]):
                values[key][idx] += value


def _phase_summary_rows(phase_acc, dt):
    rows = []
    for (episode_outcome, phase), values in sorted(phase_acc.items()):
        row = {
            "episode_outcome": episode_outcome,
            "phase": phase,
            "episodes": values["episodes"],
            "valid_steps": values["valid_steps"],
            "rotor_positive_energy_8": sum(values["rotor_pos"]),
            "rotor_negative_energy_8": sum(values["rotor_neg"]),
            "yaw_positive_energy_2": sum(values["yaw_pos"]),
            "yaw_negative_energy_2": sum(values["yaw_neg"]),
            "joint_positive_energy_10": sum(values["joint_pos"]),
            "joint_negative_energy_10": sum(values["joint_neg"]),
        }
        row["duration_s"] = values["valid_steps"] * dt
        _add_motor_fields(row, "positive_energy", values["rotor_pos"])
        _add_motor_fields(row, "negative_energy", values["rotor_neg"])
        _add_joint_fields(row, "positive_energy", values["joint_pos"])
        _add_joint_fields(row, "negative_energy", values["joint_neg"])
        rows.append(row)
    return rows


def _default_output_dir(args):
    mode = args.energy_cost_mode or "cfg"
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    name = f"{timestamp}_{args.task}_{mode}_vx{args.command_x:g}"
    return os.path.join(LEGGED_GYM_ROOT_DIR, "energy_evaluations", name)


def evaluate(args):
    global JOINT_NAMES, ROTOR_NAMES

    if args.task == "XBotL_free":
        args.task = "bruce_ppolag"
    if (
        args.energy_cost_mode is not None
        and args.energy_cost_mode not in SUPPORTED_ENERGY_COST_MODES
    ):
        raise ValueError(
            f"Unsupported energy_cost_mode '{args.energy_cost_mode}'. "
            f"Supported modes are: {SUPPORTED_ENERGY_COST_MODES}"
        )

    output_dir = os.path.abspath(args.output_dir or _default_output_dir(args))
    os.makedirs(output_dir, exist_ok=True)

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    _set_eval_config(env_cfg, args)
    train_cfg.runner.resume = True

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    runner, train_cfg = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg
    )
    policy = runner.get_inference_policy(device=env.device)
    _set_fixed_command(env, args)
    env.compute_observations()
    obs = env.get_observations()

    feet_names = getattr(env, "feet_names", [])
    body_names = getattr(env, "body_names", [])
    print("DOF_NAMES:", env.dof_names)
    print("FEET_NAMES:", feet_names)
    JOINT_NAMES = bruce_joint_names_from_dof_names(env.dof_names)
    ROTOR_NAMES = bruce_rotor_names_from_dof_names(env.dof_names)
    left_foot_idx, right_foot_idx = _resolve_foot_contact_indices(feet_names)

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
        "dof_names": list(env.dof_names),
        "body_names": list(body_names),
        "feet_names": list(feet_names),
        "left_foot_name": feet_names[left_foot_idx],
        "right_foot_name": feet_names[right_foot_idx],
        "rotor_names": list(ROTOR_NAMES),
        "joint_names": list(JOINT_NAMES),
        "state_rows_sample": "pre_step",
        "valid_state_semantics": "pre_step_dynamic_state_valid",
        "pre_step_dynamic_state_valid": (
            "false for episode_step 0 because automatic reset does not refresh "
            "base_lin_vel/contact_forces until the next physics step"
        ),
        "distance_x_source": "world_root_position_delta_x",
        "path_length_xy_source": "world_root_position_delta_xy",
        "body_frame_distance_x_source": (
            "pre_step_base_lin_vel_x integrated only when "
            "pre_step_dynamic_state_valid is true"
        ),
        "instantaneous_motor_fields_sample": "last_physics_substep",
        "phase_summary_energy_source": (
            "integrated_policy_step_energy for pre_step_dynamic_state_valid rows"
        ),
        "hard_exit_after_eval": not args.no_hard_exit_after_eval,
    }

    step_rows = []
    episode_rows = []
    phase_acc = _init_phase_accumulators()
    episode_phase_acc = _init_phase_accumulators()
    episode_acc = _new_episode_accumulator()
    episode_id = 0
    episode_step = 0
    robot_index = 0
    max_steps = int(args.num_eval_episodes * env.max_episode_length * 3)

    with torch.inference_mode():
        for global_step in range(max_steps):
            _set_fixed_command(env, args)
            pre_step_state = _pre_step_state(
                env, robot_index, left_foot_idx, right_foot_idx
            )
            actions = policy(obs.detach())
            obs, _, _, dones, infos, costs = env.step(actions.detach())

            done = bool(dones[robot_index].item())
            timeout = bool(env.time_out_buf[robot_index].item())
            pre_step_state_valid = True
            pre_step_dynamic_state_valid = episode_step > 0
            post_step_state_valid = not done
            post_step_root_pos = _tensor_list(
                env.pre_reset_root_states[robot_index, :3]
            )
            world_dx = post_step_root_pos[0] - pre_step_state["pre_step_root_pos_x"]
            world_dy = post_step_root_pos[1] - pre_step_state["pre_step_root_pos_y"]
            world_dz = post_step_root_pos[2] - pre_step_state["pre_step_root_pos_z"]
            world_path_xy = (world_dx * world_dx + world_dy * world_dy) ** 0.5
            body_frame_delta_x = (
                pre_step_state["pre_step_base_vel_x"] * env.dt
                if pre_step_dynamic_state_valid
                else 0.0
            )
            snapshot = _step_energy_snapshot(env, robot_index)
            _accumulate_episode(
                episode_acc,
                snapshot,
                pre_step_state,
                post_step_root_pos,
                pre_step_dynamic_state_valid,
                env.dt,
            )
            if pre_step_dynamic_state_valid:
                _record_phase(
                    episode_phase_acc, pre_step_state["phase_label"], snapshot
                )

            row = {
                "global_step": global_step,
                "episode_id": episode_id,
                "episode_step": episode_step,
                "done": float(done),
                "timeout": float(timeout),
                "valid_state": float(pre_step_dynamic_state_valid),
                "pre_step_state_valid": float(pre_step_state_valid),
                "pre_step_dynamic_state_valid": float(pre_step_dynamic_state_valid),
                "post_step_state_valid": float(post_step_state_valid),
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
                "post_step_root_pos_x": post_step_root_pos[0],
                "post_step_root_pos_y": post_step_root_pos[1],
                "post_step_root_pos_z": post_step_root_pos[2],
                "step_world_delta_x": world_dx,
                "step_world_delta_y": world_dy,
                "step_world_delta_z": world_dz,
                "step_world_path_xy": world_path_xy,
                "body_frame_delta_x": body_frame_delta_x,
            }
            row.update(pre_step_state)
            _add_motor_fields(row, "positive_energy", snapshot["rotor_pos"])
            _add_motor_fields(row, "negative_energy", snapshot["rotor_neg"])
            _add_motor_fields(row, "power", _tensor_list(env.rotor_power[robot_index]))
            _add_motor_fields(row, "torque", _tensor_list(env.rotor_torque[robot_index]))
            _add_motor_fields(
                row, "velocity", _tensor_list(env.rotor_velocity[robot_index])
            )
            _add_joint_fields(row, "positive_energy", snapshot["joint_pos"])
            _add_joint_fields(row, "negative_energy", snapshot["joint_neg"])
            _add_joint_fields(row, "power", _tensor_list(env.joint_power[robot_index]))

            if post_step_state_valid:
                row.update(
                    {
                        "post_step_base_vel_x": _to_float(
                            env.base_lin_vel[robot_index, 0]
                        ),
                        "post_step_base_vel_y": _to_float(
                            env.base_lin_vel[robot_index, 1]
                        ),
                        "post_step_base_vel_z": _to_float(
                            env.base_lin_vel[robot_index, 2]
                        ),
                    }
                )
            step_rows.append(row)

            episode_step += 1
            if done:
                episode_outcome = "success" if timeout else "fall"
                episode_rows.append(
                    _episode_summary_row(episode_id, episode_acc, timeout, env.dt)
                )
                _merge_phase_accumulators(
                    phase_acc, episode_phase_acc, episode_outcome
                )
                episode_id += 1
                episode_step = 0
                episode_acc = _new_episode_accumulator()
                episode_phase_acc = _init_phase_accumulators()
                if episode_id >= args.num_eval_episodes:
                    break
        else:
            raise RuntimeError(
                f"Evaluation stopped after {max_steps} steps before collecting "
                f"{args.num_eval_episodes} episodes."
            )

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "metadata.json"), "w") as jsonfile:
        json.dump(metadata, jsonfile, indent=2)
    _write_dicts_csv(os.path.join(output_dir, "episode_summary.csv"), episode_rows)
    _write_dicts_csv(os.path.join(output_dir, "step_timeseries.csv"), step_rows)
    _write_dicts_csv(
        os.path.join(output_dir, "phase_summary.csv"),
        _phase_summary_rows(phase_acc, env.dt),
    )

    print(f"Wrote energy evaluation outputs to: {output_dir}")
    sys.stdout.flush()
    sys.stderr.flush()
    if not args.no_hard_exit_after_eval:
        os._exit(0)


if __name__ == "__main__":
    evaluate(get_args())
