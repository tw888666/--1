# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2026 ECO Authors. All rights reserved.

import json
import os
import re

from bruce_gym.gpu_auto_select import apply_auto_gpu_selection_from_argv

apply_auto_gpu_selection_from_argv()

import cv2
import numpy as np
import torch
from isaacgym import gymapi

from bruce_gym.envs import *  # noqa: F401,F403
from bruce_gym.scripts.evaluate_energy import _set_eval_config, _set_fixed_command
from bruce_gym.utils import get_args, task_registry


def _parse_int_list(value):
    if not value:
        return []
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _parse_label_list(value):
    if not value:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def _safe_label(label):
    label = re.sub(r"[^A-Za-z0-9_.-]+", "_", label.strip())
    return label or "episode"


def _load_targets_from_review(review_dir, limit):
    path = os.path.join(review_dir, "representative_episodes.json")
    with open(path) as jsonfile:
        payload = json.load(jsonfile)
    targets = []
    for item in payload.get("representatives", []):
        targets.append((str(item["label"]), int(item["episode_id"])))
    return targets[:limit]


def _resolve_targets(args):
    episode_ids = _parse_int_list(args.record_episode_ids)
    if episode_ids:
        labels = _parse_label_list(args.record_episode_labels)
        if labels and len(labels) != len(episode_ids):
            raise ValueError("--record_episode_labels must match --record_episode_ids length.")
        if not labels:
            labels = [f"episode_{episode_id:04d}" for episode_id in episode_ids]
        return list(zip(labels, episode_ids))[: args.max_video_episodes]

    if not args.review_dir:
        raise ValueError("Set --review_dir or --record_episode_ids.")
    return _load_targets_from_review(args.review_dir, args.max_video_episodes)


def _parse_vec3(value, default):
    if not value:
        return default
    parts = [float(part.strip()) for part in value.split(",")]
    if len(parts) != 3:
        raise ValueError("Camera vector arguments must look like x,y,z.")
    return gymapi.Vec3(parts[0], parts[1], parts[2])


def _create_camera(env, args):
    camera_properties = gymapi.CameraProperties()
    camera_properties.width = args.video_width
    camera_properties.height = args.video_height
    camera_handle = env.gym.create_camera_sensor(env.envs[0], camera_properties)

    camera_offset = _parse_vec3(
        args.camera_offset,
        gymapi.Vec3(1.0, -1.0, 0.5),
    )
    camera_axis = _parse_vec3(
        args.camera_axis,
        gymapi.Vec3(-0.3, 0.2, 1.0),
    )
    camera_rotation = gymapi.Quat.from_axis_angle(
        camera_axis,
        np.deg2rad(args.camera_angle_deg),
    )
    actor_handle = env.gym.get_actor_handle(env.envs[0], 0)
    body_handle = env.gym.get_actor_rigid_body_handle(env.envs[0], actor_handle, 0)
    env.gym.attach_camera_to_body(
        camera_handle,
        env.envs[0],
        body_handle,
        gymapi.Transform(camera_offset, camera_rotation),
        gymapi.FOLLOW_POSITION,
    )
    return camera_handle


def _open_writers(video_dir, labels, args):
    os.makedirs(video_dir, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writers = []
    for label in labels:
        path = os.path.join(video_dir, f"episode_{_safe_label(label)}.mp4")
        writer = cv2.VideoWriter(
            path,
            fourcc,
            float(args.video_fps),
            (args.video_width, args.video_height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Failed to open video writer: {path}")
        writers.append((path, writer))
        print(f"Recording {label} to {path}")
    return writers


def _write_frame(env, camera_handle, writers, args):
    env.gym.fetch_results(env.sim, True)
    env.gym.step_graphics(env.sim)
    env.gym.render_all_camera_sensors(env.sim)
    image = env.gym.get_camera_image(
        env.sim,
        env.envs[0],
        camera_handle,
        gymapi.IMAGE_COLOR,
    )
    image = np.reshape(image, (args.video_height, args.video_width, 4))
    image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    for _, writer in writers:
        writer.write(image)


def _release_writers(writers):
    for _, writer in writers:
        writer.release()


def record(args):
    targets = _resolve_targets(args)
    if not targets:
        raise ValueError("No episode targets selected for video recording.")

    video_dir = args.video_dir or args.review_dir or os.getcwd()
    labels_by_episode = {}
    for label, episode_id in targets:
        labels_by_episode.setdefault(episode_id, []).append(label)
    max_target_episode = max(labels_by_episode)

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    _set_eval_config(env_cfg, args)
    train_cfg.runner.resume = True

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    runner, _ = task_registry.make_alg_runner(
        env=env,
        name=args.task,
        args=args,
        train_cfg=train_cfg,
    )
    policy = runner.get_inference_policy(device=env.device)
    _set_fixed_command(env, args)
    env.compute_observations()
    obs = env.get_observations()
    camera_handle = _create_camera(env, args)

    episode_id = 0
    episode_step = 0
    active_episode = None
    active_writers = []
    recorded = set()
    max_steps = int((max_target_episode + 1) * env.max_episode_length * 3)

    try:
        with torch.inference_mode():
            for global_step in range(max_steps):
                _set_fixed_command(env, args)
                actions = policy(obs.detach())
                obs, _, _, dones, _, _ = env.step(actions.detach())

                if episode_id in labels_by_episode:
                    if active_episode != episode_id:
                        active_episode = episode_id
                        active_writers = _open_writers(
                            video_dir,
                            labels_by_episode[episode_id],
                            args,
                        )
                    if episode_step % max(1, args.video_stride) == 0:
                        _write_frame(env, camera_handle, active_writers, args)

                done = bool(dones[0].item())
                episode_step += 1
                if done:
                    if active_writers:
                        _release_writers(active_writers)
                        recorded.add(episode_id)
                        active_writers = []
                        active_episode = None
                    episode_id += 1
                    episode_step = 0
                    if episode_id > max_target_episode and set(labels_by_episode) <= recorded:
                        break
            else:
                missing = sorted(set(labels_by_episode) - recorded)
                raise RuntimeError(
                    f"Stopped after {max_steps} steps before recording episodes {missing}."
                )
    finally:
        _release_writers(active_writers)

    print(f"Wrote {len(recorded)} episode video(s) to: {video_dir}")


if __name__ == "__main__":
    record(get_args())
