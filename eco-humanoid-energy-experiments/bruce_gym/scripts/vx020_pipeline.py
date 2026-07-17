# SPDX-License-Identifier: BSD-3-Clause

"""Resumable controller for the seed-0 vx=0.2 control experiment pipeline."""

import argparse
import csv
import fcntl
import json
import math
import os
import re
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EXPECTED_PYTHON = Path(
    "/home/xy.chen/miniconda3/envs/bruce_gym/bin/python"
).resolve()
WORKFLOW_DIR = ROOT / "workflow_runs" / "vx020_seed0"
STATE_PATH = WORKFLOW_DIR / "state.json"
LOCK_PATH = WORKFLOW_DIR / "controller.lock"
SUMMARY_JSON_PATH = WORKFLOW_DIR / "summary.json"
SUMMARY_MD_PATH = WORKFLOW_DIR / "summary.md"
LOG_DIR = WORKFLOW_DIR / "logs"

BASE_RUN = ROOT / "pretrained_weights" / "eco_ppolag_0.2vel_8000cost_seed123"
FIXED_ROOT = ROOT / "energy_evaluations" / "fixed20"
TRAIN_DIST_ROOT = ROOT / "energy_calibrations" / "train_dist100"

TARGET_COMMAND_X = 0.2
VELOCITY_MIN = 0.16
VELOCITY_MAX = 0.24
FIXED_EPISODES = 20
TRAIN_DIST_EPISODES = 100
TRAIN_DIST_ENVS = 1024
DEFAULT_MIN_FREE_MIB = 8192
INITIAL_START_TIMEOUT_SECONDS = 600
STATE_VERSION = 1

CONTROL_ORDER = ("rp8", "ra8", "jp8", "ja8", "lja10")
INITIAL_FIXED_ALIASES = ("rp8", "jp8")
TRAIN_ALIASES = ("ra8", "ja8", "lja10")
NEW_FIXED_ALIASES = ("ra8", "ja8", "lja10")

CONTROL_SPECS = {
    "rp8": {
        "mode": "rotor_positive_8",
        "run_name": "vx020_rp8_s00_model4001",
        "cost_limit1": 49.157738095283506,
        "train_gpu": None,
        "depends_on": None,
        "existing_run": ROOT
        / "logs"
        / "exp"
        / "20260710_1236_vx020_rp8_s00_model4001",
        "initial_log": ROOT
        / "fixed_eval_logs"
        / "eval_rp8_truevx020_vx020_gpu3.log",
    },
    "ra8": {
        "mode": "rotor_abs_8",
        "run_name": "vx020_ra8_s00_model4001",
        "cost_limit1": 81.79582343673707,
        "train_gpu": 0,
        "depends_on": None,
        "existing_run": None,
        "initial_log": None,
    },
    "jp8": {
        "mode": "joint_positive_8",
        "run_name": "vx020_jp8_s00_model4001",
        "cost_limit1": 45.27336714076996,
        "train_gpu": None,
        "depends_on": None,
        "existing_run": ROOT
        / "logs"
        / "exp"
        / "20260710_1240_vx020_jp8_s00_model4001",
        "initial_log": ROOT
        / "fixed_eval_logs"
        / "eval_jp8_truevx020_vx020_gpu3.log",
    },
    "ja8": {
        "mode": "joint_abs_8",
        "run_name": "vx020_ja8_s00_model4001",
        "cost_limit1": 72.39956164360046,
        "train_gpu": 1,
        "depends_on": None,
        "existing_run": None,
        "initial_log": None,
    },
    "lja10": {
        "mode": "legacy_joint_abs_10",
        "run_name": "vx020_lja10_s00_model4001",
        "cost_limit1": 7557.11452758789,
        "train_gpu": 0,
        "depends_on": "ra8",
        "existing_run": None,
        "initial_log": None,
    },
}

FATAL_LOG_PATTERNS = (
    "Traceback (most recent call last)",
    "CUDA out of memory",
    "RuntimeError:",
    "Exception:",
)


def _now():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _atomic_write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(str(temporary), str(path))


def _atomic_write_text(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(value)
    os.replace(str(temporary), str(path))


def _read_json(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_csv(path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _mean_finite(rows, field, errors):
    values = []
    for row_number, row in enumerate(rows, start=1):
        try:
            value = float(row[field])
        except (KeyError, TypeError, ValueError):
            errors.append("row {} has invalid {}".format(row_number, field))
            continue
        if not math.isfinite(value):
            errors.append("row {} has non-finite {}".format(row_number, field))
            continue
        values.append(value)
    return statistics.mean(values) if len(values) == len(rows) and values else None


def fixed_output_dir(alias):
    return FIXED_ROOT / CONTROL_SPECS[alias]["run_name"]


def train_dist_output_dir(alias):
    return TRAIN_DIST_ROOT / CONTROL_SPECS[alias]["run_name"]


def validate_fixed_evaluation(output_dir, expected_mode):
    """Validate one completed 20-episode fixed-command evaluation."""

    output_dir = Path(output_dir)
    metadata_path = output_dir / "metadata.json"
    episodes_path = output_dir / "episode_summary.csv"
    result = {
        "complete": metadata_path.is_file() and episodes_path.is_file(),
        "passed": False,
        "errors": [],
        "output_dir": str(output_dir),
    }
    if not result["complete"]:
        return result

    try:
        metadata = _read_json(metadata_path)
        rows = _read_csv(episodes_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        result["errors"].append("cannot read evaluation output: {}".format(exc))
        return result

    if metadata.get("energy_cost_mode") != expected_mode:
        result["errors"].append(
            "energy_cost_mode is {!r}, expected {!r}".format(
                metadata.get("energy_cost_mode"), expected_mode
            )
        )
    try:
        command_x = float(metadata.get("command_x"))
    except (TypeError, ValueError):
        command_x = math.nan
    if not math.isclose(command_x, TARGET_COMMAND_X, abs_tol=1e-9):
        result["errors"].append(
            "command_x is {!r}, expected {}".format(
                metadata.get("command_x"), TARGET_COMMAND_X
            )
        )
    try:
        checkpoint = int(metadata.get("checkpoint"))
    except (TypeError, ValueError):
        checkpoint = None
    if checkpoint != 4001:
        result["errors"].append(
            "checkpoint is {!r}, expected 4001".format(metadata.get("checkpoint"))
        )
    if len(rows) != FIXED_EPISODES:
        result["errors"].append(
            "episode count is {}, expected {}".format(len(rows), FIXED_EPISODES)
        )

    success_count = sum(
        row.get("episode_outcome") == "success" for row in rows
    )
    if success_count != FIXED_EPISODES:
        result["errors"].append(
            "success count is {}, expected {}".format(
                success_count, FIXED_EPISODES
            )
        )

    mean_velocity_x = _mean_finite(rows, "mean_velocity_x", result["errors"])
    mean_body_velocity_x = _mean_finite(
        rows, "mean_body_frame_velocity_x", result["errors"]
    )
    for name, value in (
        ("mean_velocity_x", mean_velocity_x),
        ("mean_body_frame_velocity_x", mean_body_velocity_x),
    ):
        if value is not None and not VELOCITY_MIN <= value <= VELOCITY_MAX:
            result["errors"].append(
                "{} is {:.6f}, expected [{:.2f}, {:.2f}]".format(
                    name, value, VELOCITY_MIN, VELOCITY_MAX
                )
            )

    result.update(
        {
            "episode_count": len(rows),
            "success_count": success_count,
            "success_rate": success_count / len(rows) if rows else 0.0,
            "mean_velocity_x": mean_velocity_x,
            "mean_body_frame_velocity_x": mean_body_velocity_x,
            "metadata": {
                "command_x": metadata.get("command_x"),
                "checkpoint": metadata.get("checkpoint"),
                "energy_cost_mode": metadata.get("energy_cost_mode"),
            },
        }
    )
    result["passed"] = not result["errors"]
    return result


def validate_training_distribution(output_dir, expected_mode):
    """Validate and summarize one 100-episode training-distribution run."""

    output_dir = Path(output_dir)
    metadata_path = output_dir / "metadata.json"
    episodes_path = output_dir / "episode_costs.csv"
    summary_path = output_dir / "calibration_summary.json"
    result = {
        "complete": all(
            path.is_file()
            for path in (metadata_path, episodes_path, summary_path)
        ),
        "passed": False,
        "errors": [],
        "output_dir": str(output_dir),
    }
    if not result["complete"]:
        return result

    try:
        metadata = _read_json(metadata_path)
        rows = _read_csv(episodes_path)
        summary = _read_json(summary_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        result["errors"].append(
            "cannot read training-distribution output: {}".format(exc)
        )
        return result

    if metadata.get("energy_cost_mode") != expected_mode:
        result["errors"].append(
            "energy_cost_mode is {!r}, expected {!r}".format(
                metadata.get("energy_cost_mode"), expected_mode
            )
        )
    try:
        command_x = float(metadata.get("command_x"))
    except (TypeError, ValueError):
        command_x = math.nan
    if not math.isclose(command_x, TARGET_COMMAND_X, abs_tol=1e-9):
        result["errors"].append(
            "command_x is {!r}, expected {}".format(
                metadata.get("command_x"), TARGET_COMMAND_X
            )
        )
    try:
        checkpoint = int(metadata.get("checkpoint"))
    except (TypeError, ValueError):
        checkpoint = None
    if checkpoint != 4001:
        result["errors"].append(
            "checkpoint is {!r}, expected 4001".format(metadata.get("checkpoint"))
        )
    if len(rows) != TRAIN_DIST_EPISODES:
        result["errors"].append(
            "episode count is {}, expected {}".format(
                len(rows), TRAIN_DIST_EPISODES
            )
        )

    mean_body_velocity_x = _mean_finite(
        rows, "mean_body_frame_velocity_x", result["errors"]
    )
    success_count = sum(
        row.get("episode_outcome") == "success" for row in rows
    )
    result.update(
        {
            "episode_count": len(rows),
            "success_count": success_count,
            "success_rate": success_count / len(rows) if rows else 0.0,
            "mean_body_frame_velocity_x": mean_body_velocity_x,
            "cost1_mean": summary.get("cost1_all_episodes", {}).get("mean"),
            "cost1_median": summary.get("cost1_all_episodes", {}).get("median"),
            "recommended_cost_limit1": summary.get(
                "recommended_cost_limit1_all_episodes"
            ),
        }
    )
    result["passed"] = not result["errors"]
    return result


def build_train_command(alias, python_executable=None):
    spec = CONTROL_SPECS[alias]
    return [
        python_executable or sys.executable,
        "-u",
        "-m",
        "bruce_gym.scripts.train",
        "--task=bruce_ppolag",
        "--headless",
        "--resume",
        "--experiment_name=exp",
        "--load_run={}".format(BASE_RUN),
        "--checkpoint=3000",
        "--max_iterations=1000",
        "--num_envs=8192",
        "--sim_device=cuda:0",
        "--rl_device=cuda:0",
        "--seed=0",
        "--run_name={}".format(spec["run_name"]),
        "--energy_cost_mode={}".format(spec["mode"]),
        "--cost_limit1={}".format(spec["cost_limit1"]),
    ]


def build_fixed_eval_command(alias, run_dir, python_executable=None):
    spec = CONTROL_SPECS[alias]
    return [
        python_executable or sys.executable,
        "-u",
        "-m",
        "bruce_gym.scripts.evaluate_energy",
        "--task=bruce_ppolag",
        "--headless",
        "--resume",
        "--experiment_name=exp",
        "--load_run={}".format(run_dir),
        "--checkpoint=4001",
        "--energy_cost_mode={}".format(spec["mode"]),
        "--command_x=0.2",
        "--command_y=0.0",
        "--command_yaw=0.0",
        "--num_eval_episodes=20",
        "--output_dir={}".format(fixed_output_dir(alias)),
        "--make_eval_report",
        "--sim_device=cuda:0",
        "--rl_device=cuda:0",
        "--seed=0",
    ]


def build_train_dist_command(alias, run_dir, python_executable=None):
    spec = CONTROL_SPECS[alias]
    return [
        python_executable or sys.executable,
        "-u",
        "-m",
        "bruce_gym.scripts.calibrate_train_cost",
        "--task=bruce_ppolag",
        "--headless",
        "--resume",
        "--experiment_name=exp",
        "--load_run={}".format(run_dir),
        "--checkpoint=4001",
        "--energy_cost_mode={}".format(spec["mode"]),
        "--command_x=0.2",
        "--calibration_episodes=100",
        "--calibration_limit_fraction=0.95",
        "--num_envs=1024",
        "--output_dir={}".format(train_dist_output_dir(alias)),
        "--sim_device=cuda:0",
        "--rl_device=cuda:0",
        "--seed=0",
    ]


def new_state():
    return {
        "version": STATE_VERSION,
        "status": "running",
        "phase": "waiting_initial_fixed",
        "reason": None,
        "controller_pid": None,
        "created_at": _now(),
        "updated_at": _now(),
        "jobs": {},
        "run_dirs": {
            alias: str(CONTROL_SPECS[alias]["existing_run"])
            for alias in INITIAL_FIXED_ALIASES
        },
        "fixed_evaluations": {},
        "training_distributions": {},
        "initial_missing_since": {},
    }


def load_state(path=STATE_PATH):
    path = Path(path)
    if not path.is_file():
        return new_state()
    state = _read_json(path)
    if state.get("version") != STATE_VERSION:
        raise RuntimeError(
            "Unsupported state version: {}".format(state.get("version"))
        )
    return state


def save_state(state, path=STATE_PATH):
    state["updated_at"] = _now()
    _atomic_write_json(Path(path), state)


def prepare_stopped_state_for_resume(state):
    """Remove intentionally stopped jobs once their processes have exited."""

    stopped_jobs = {
        job_id: job
        for job_id, job in state.get("jobs", {}).items()
        if job.get("status") == "stopped"
    }
    alive = [
        "{}(pid={})".format(job_id, job.get("pid"))
        for job_id, job in stopped_jobs.items()
        if _pid_alive(job.get("pid"))
    ]
    if alive:
        return False, "stopped child processes are still exiting: {}".format(
            ", ".join(alive)
        )

    for job_id, job in stopped_jobs.items():
        if job.get("kind") == "train":
            state.get("run_dirs", {}).pop(job.get("alias"), None)
        state["jobs"].pop(job_id, None)
    state["status"] = "running"
    state["reason"] = None
    return True, None


def prepare_blocked_training_state_for_retry(state):
    """Keep complete cleanup-crash runs and reset incomplete training jobs."""

    if state.get("phase") != "training_controls":
        return False, "--retry-failed only supports the training_controls phase"
    failed_jobs = {
        job_id: job
        for job_id, job in state.get("jobs", {}).items()
        if job.get("status") == "failed"
    }
    if not failed_jobs:
        return False, "blocked training state has no failed jobs to retry"
    if any(job.get("kind") != "train" for job in failed_jobs.values()):
        return False, "--retry-failed refuses non-training failures"

    alive = [
        "{}(pid={})".format(job_id, job.get("pid"))
        for job_id, job in failed_jobs.items()
        if _pid_alive(job.get("pid"))
    ]
    if alive:
        return False, "failed child processes are still alive: {}".format(
            ", ".join(alive)
        )

    for job_id, job in failed_jobs.items():
        artifact_ok, _ = _job_artifact_result(job)
        cleanup_complete = (
            job.get("returncode") == -signal.SIGSEGV
            and artifact_ok
            and _training_reached_final_iteration(job.get("log_path"))
        )
        if cleanup_complete:
            job["status"] = "completed"
            job["reason"] = (
                "accepted SIGSEGV after final iteration and model_4001.pt save"
            )
            state["run_dirs"][job["alias"]] = job["run_dir"]
            continue
        state["jobs"].pop(job_id, None)
        state.get("run_dirs", {}).pop(job.get("alias"), None)

    state["status"] = "running"
    state["reason"] = None
    return True, None


def _pid_alive(pid):
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
    except (OSError, ValueError):
        return False
    return True


def _process_with_argument_exists(argument):
    encoded = str(argument).encode("utf-8")
    for cmdline_path in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            cmdline = cmdline_path.read_bytes()
        except OSError:
            continue
        if encoded in cmdline and b"evaluate_energy" in cmdline:
            return True
    return False


def _log_has_fatal_error(path):
    path = Path(path)
    if not path.is_file():
        return None
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    for pattern in FATAL_LOG_PATTERNS:
        if pattern in text:
            return pattern
    return None


def query_free_memory_mib():
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.free",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    free_memory = {}
    for line in completed.stdout.splitlines():
        if not line.strip():
            continue
        index, free_mib = (part.strip() for part in line.split(",", 1))
        free_memory[int(index)] = int(free_mib)
    return free_memory


def select_available_gpu(gpus, free_memory, busy_gpus, min_free_mib):
    for gpu in gpus:
        if gpu not in busy_gpus and free_memory.get(gpu, 0) >= min_free_mib:
            return gpu
    return None


def _child_environment(gpu):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    conda_bin = str(Path(sys.executable).resolve().parent)
    current_path = env.get("PATH")
    env["PATH"] = (
        conda_bin
        if not current_path
        else conda_bin + os.pathsep + current_path
    )
    current_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(ROOT)
        if not current_pythonpath
        else str(ROOT) + os.pathsep + current_pythonpath
    )
    conda_lib = str(Path(sys.executable).resolve().parents[1] / "lib")
    current_ld_path = env.get("LD_LIBRARY_PATH")
    env["LD_LIBRARY_PATH"] = (
        conda_lib
        if not current_ld_path
        else conda_lib + os.pathsep + current_ld_path
    )
    return env


def _job_id(kind, alias):
    return "{}_{}".format(kind, alias)


def _job_log_path(kind, alias):
    return LOG_DIR / "{}_{}.log".format(kind, alias)


def _active_jobs(state):
    return {
        job_id: job
        for job_id, job in state["jobs"].items()
        if job.get("status") == "running"
    }


def _busy_gpus(state):
    return {
        int(job["gpu"])
        for job in _active_jobs(state).values()
        if job.get("gpu") is not None
    }


def _launch_job(state, kind, alias, gpu, command, output_dir=None):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = _job_log_path(kind, alias)
    with log_path.open("ab") as log_handle:
        header = (
            "\n[{timestamp}] START gpu={gpu}\n{command}\n".format(
                timestamp=_now(), gpu=gpu, command=" ".join(command)
            )
        )
        log_handle.write(header.encode("utf-8"))
        log_handle.flush()
        process = subprocess.Popen(
            command,
            cwd=str(ROOT),
            env=_child_environment(gpu),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    job = {
        "kind": kind,
        "alias": alias,
        "status": "running",
        "pid": process.pid,
        "gpu": gpu,
        "command": command,
        "log_path": str(log_path),
        "output_dir": str(output_dir) if output_dir else None,
        "run_dir": None,
        "started_at": _now(),
        "ended_at": None,
        "returncode": None,
        "reason": None,
    }
    state["jobs"][_job_id(kind, alias)] = job
    return process


def _parse_training_run_dir(log_path):
    path = Path(log_path)
    if not path.is_file():
        return None
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    matches = re.findall(r"^log_dir\s+(.+)$", text, flags=re.MULTILINE)
    return Path(matches[-1].strip()) if matches else None


def _training_reached_final_iteration(log_path):
    if not log_path:
        return False
    path = Path(log_path)
    if not path.is_file():
        return False
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    return bool(re.search(r"Learning iteration\s+4000/4001", text))


def _job_artifact_result(job):
    alias = job["alias"]
    kind = job["kind"]
    if kind == "train":
        run_dir = job.get("run_dir") or _parse_training_run_dir(job["log_path"])
        if not run_dir:
            return False, "training log did not report log_dir"
        job["run_dir"] = str(run_dir)
        checkpoint = Path(run_dir) / "model_4001.pt"
        if not checkpoint.is_file():
            return False, "missing checkpoint {}".format(checkpoint)
        return True, None
    if kind == "fixed_eval":
        result = validate_fixed_evaluation(
            job["output_dir"], CONTROL_SPECS[alias]["mode"]
        )
        if not result["complete"]:
            return False, "fixed evaluation output is incomplete"
        if not result["passed"]:
            return False, "; ".join(result["errors"])
        return True, None
    if kind == "train_dist":
        result = validate_training_distribution(
            job["output_dir"], CONTROL_SPECS[alias]["mode"]
        )
        if not result["complete"]:
            return False, "training-distribution output is incomplete"
        if not result["passed"]:
            return False, "; ".join(result["errors"])
        return True, None
    return False, "unknown job kind {!r}".format(kind)


def _poll_jobs(state, processes):
    changed = False
    for job_id, job in list(_active_jobs(state).items()):
        if job["kind"] == "train" and not job.get("run_dir"):
            run_dir = _parse_training_run_dir(job["log_path"])
            if run_dir:
                job["run_dir"] = str(run_dir)
                state["run_dirs"][job["alias"]] = str(run_dir)
                changed = True

        process = processes.get(job_id)
        if process is not None:
            returncode = process.poll()
            if returncode is None:
                continue
        elif _pid_alive(job.get("pid")):
            continue
        else:
            returncode = None

        artifact_ok, reason = _job_artifact_result(job)
        if job["kind"] == "fixed_eval":
            state["fixed_evaluations"][job["alias"]] = (
                validate_fixed_evaluation(
                    job["output_dir"], CONTROL_SPECS[job["alias"]]["mode"]
                )
            )
        elif job["kind"] == "train_dist":
            state["training_distributions"][job["alias"]] = (
                validate_training_distribution(
                    job["output_dir"], CONTROL_SPECS[job["alias"]]["mode"]
                )
            )
        if process is not None and returncode != 0:
            cleanup_complete = (
                job["kind"] == "train"
                and returncode == -signal.SIGSEGV
                and artifact_ok
                and _training_reached_final_iteration(job.get("log_path"))
            )
            if cleanup_complete:
                reason = (
                    "accepted SIGSEGV after final iteration and "
                    "model_4001.pt save"
                )
            else:
                artifact_ok = False
                reason = "process exited with code {}{}".format(
                    returncode, ": " + reason if reason else ""
                )
        job["returncode"] = returncode
        job["ended_at"] = _now()
        job["status"] = "completed" if artifact_ok else "failed"
        job["reason"] = reason
        if artifact_ok and job["kind"] == "train":
            state["run_dirs"][job["alias"]] = job["run_dir"]
        processes.pop(job_id, None)
        changed = True
    return changed


def _stage_has_failed_job(state, kind):
    return any(
        job.get("kind") == kind and job.get("status") == "failed"
        for job in state["jobs"].values()
    )


def _stage_alias_completed(state, kind, alias):
    job = state["jobs"].get(_job_id(kind, alias))
    return bool(job and job.get("status") == "completed")


def _all_stage_aliases_completed(state, kind, aliases):
    return all(_stage_alias_completed(state, kind, alias) for alias in aliases)


def _block_after_active_jobs_finish(state, kind):
    if not _stage_has_failed_job(state, kind):
        return False
    if any(job["kind"] == kind for job in _active_jobs(state).values()):
        return True
    failures = [
        "{}: {}".format(job["alias"], job.get("reason"))
        for job in state["jobs"].values()
        if job.get("kind") == kind and job.get("status") == "failed"
    ]
    state["status"] = "blocked"
    state["reason"] = "{} stage failed: {}".format(kind, " | ".join(failures))
    return True


def _ensure_completed_output_job(state, kind, alias, output_dir, validator):
    job_id = _job_id(kind, alias)
    if job_id in state["jobs"]:
        return
    result = validator(output_dir, CONTROL_SPECS[alias]["mode"])
    if result["complete"]:
        if kind == "fixed_eval":
            state["fixed_evaluations"][alias] = result
        elif kind == "train_dist":
            state["training_distributions"][alias] = result
        state["jobs"][job_id] = {
            "kind": kind,
            "alias": alias,
            "status": "completed" if result["passed"] else "failed",
            "pid": None,
            "gpu": None,
            "command": None,
            "log_path": None,
            "output_dir": str(output_dir),
            "run_dir": None,
            "started_at": None,
            "ended_at": _now(),
            "returncode": None,
            "reason": None if result["passed"] else "; ".join(result["errors"]),
        }


def _refresh_initial_fixed(state, now_epoch=None):
    now_epoch = time.time() if now_epoch is None else now_epoch
    missing_since = state.setdefault("initial_missing_since", {})
    missing = []
    completed = set()
    for alias in INITIAL_FIXED_ALIASES:
        result = validate_fixed_evaluation(
            fixed_output_dir(alias), CONTROL_SPECS[alias]["mode"]
        )
        if result["complete"]:
            completed.add(alias)
            missing_since.pop(alias, None)
            state["fixed_evaluations"][alias] = result
            if not result["passed"]:
                state["status"] = "blocked"
                state["reason"] = "{} fixed evaluation failed: {}".format(
                    alias, "; ".join(result["errors"])
                )
                return False
            continue

        missing.append(alias)
        fatal = _log_has_fatal_error(CONTROL_SPECS[alias]["initial_log"])
        process_running = _process_with_argument_exists(fixed_output_dir(alias))
        if process_running:
            missing_since.pop(alias, None)
            continue
        if fatal and not process_running:
            state["status"] = "blocked"
            state["reason"] = (
                "{} initial fixed evaluation ended with {}".format(alias, fatal)
            )
            return False
        eligible_to_start = alias == "rp8" or "rp8" in completed
        if eligible_to_start:
            if alias not in missing_since:
                missing_since[alias] = now_epoch
            elapsed = now_epoch - float(missing_since[alias])
            if elapsed >= INITIAL_START_TIMEOUT_SECONDS:
                state["status"] = "blocked"
                state["reason"] = (
                    "{} initial fixed evaluation did not start within {} seconds"
                ).format(alias, INITIAL_START_TIMEOUT_SECONDS)
                return False
    if missing:
        state["reason"] = "waiting for initial fixed evaluations: {}".format(
            ", ".join(missing)
        )
        return False
    state["reason"] = None
    return True


def _schedule_training(state, processes, free_memory, min_free_mib):
    if _block_after_active_jobs_finish(state, "train"):
        return
    busy_gpus = _busy_gpus(state)
    for alias in TRAIN_ALIASES:
        if _stage_alias_completed(state, "train", alias):
            continue
        if _job_id("train", alias) in state["jobs"]:
            continue
        dependency = CONTROL_SPECS[alias]["depends_on"]
        if dependency and not _stage_alias_completed(state, "train", dependency):
            continue
        gpu = CONTROL_SPECS[alias]["train_gpu"]
        if gpu in busy_gpus or free_memory.get(gpu, 0) < min_free_mib:
            continue
        process = _launch_job(
            state, "train", alias, gpu, build_train_command(alias)
        )
        processes[_job_id("train", alias)] = process
        busy_gpus.add(gpu)


def _schedule_parallel_stage(
    state,
    processes,
    kind,
    aliases,
    free_memory,
    min_free_mib,
):
    if _block_after_active_jobs_finish(state, kind):
        return
    busy_gpus = _busy_gpus(state)
    for alias in aliases:
        if _stage_alias_completed(state, kind, alias):
            continue
        if _job_id(kind, alias) in state["jobs"]:
            continue
        gpu = select_available_gpu((3, 4), free_memory, busy_gpus, min_free_mib)
        if gpu is None:
            return
        run_dir = state["run_dirs"].get(alias)
        if not run_dir:
            state["status"] = "blocked"
            state["reason"] = "missing trained run directory for {}".format(alias)
            return
        if kind == "fixed_eval":
            output_dir = fixed_output_dir(alias)
            command = build_fixed_eval_command(alias, run_dir)
        else:
            output_dir = train_dist_output_dir(alias)
            command = build_train_dist_command(alias, run_dir)
        process = _launch_job(
            state, kind, alias, gpu, command, output_dir=output_dir
        )
        processes[_job_id(kind, alias)] = process
        busy_gpus.add(gpu)


def _refresh_all_fixed_results(state):
    passed = True
    for alias in CONTROL_ORDER:
        result = validate_fixed_evaluation(
            fixed_output_dir(alias), CONTROL_SPECS[alias]["mode"]
        )
        state["fixed_evaluations"][alias] = result
        if not result["complete"] or not result["passed"]:
            passed = False
    return passed


def _refresh_distribution_results(state):
    for alias in CONTROL_ORDER:
        state["training_distributions"][alias] = validate_training_distribution(
            train_dist_output_dir(alias), CONTROL_SPECS[alias]["mode"]
        )


def _format_number(value, digits=4):
    if value is None:
        return "N/A"
    return ("{:.%df}" % digits).format(float(value))


def build_summary(state):
    return {
        "status": state["status"],
        "phase": state["phase"],
        "reason": state.get("reason"),
        "command_x": TARGET_COMMAND_X,
        "fixed_velocity_range": [VELOCITY_MIN, VELOCITY_MAX],
        "fixed_evaluations": state.get("fixed_evaluations", {}),
        "training_distributions": state.get("training_distributions", {}),
        "run_dirs": state.get("run_dirs", {}),
        "updated_at": state.get("updated_at"),
    }


def build_summary_markdown(state):
    lines = [
        "# vx=0.2 seed-0 control pipeline",
        "",
        "- Status: `{}`".format(state["status"]),
        "- Phase: `{}`".format(state["phase"]),
        "- Reason: {}".format(state.get("reason") or "N/A"),
        "- Updated: `{}`".format(state.get("updated_at")),
        "",
        "## Fixed-command evaluation",
        "",
        "| Group | Mode | Episodes | Success | World vx | Body vx | Gate |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for alias in CONTROL_ORDER:
        result = state.get("fixed_evaluations", {}).get(alias, {})
        lines.append(
            "| {alias} | `{mode}` | {episodes} | {success} | {world} | "
            "{body} | {gate} |".format(
                alias=alias,
                mode=CONTROL_SPECS[alias]["mode"],
                episodes=result.get("episode_count", "N/A"),
                success=result.get("success_count", "N/A"),
                world=_format_number(result.get("mean_velocity_x")),
                body=_format_number(result.get("mean_body_frame_velocity_x")),
                gate="PASS" if result.get("passed") else "WAIT/FAIL",
            )
        )
    lines.extend(
        [
            "",
            "## Training-distribution evaluation",
            "",
            "| Group | Episodes | Success rate | Body vx | Cost mean | Cost median |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for alias in CONTROL_ORDER:
        result = state.get("training_distributions", {}).get(alias, {})
        success_rate = result.get("success_rate")
        lines.append(
            "| {alias} | {episodes} | {success} | {body} | {cost_mean} | "
            "{cost_median} |".format(
                alias=alias,
                episodes=result.get("episode_count", "N/A"),
                success=(
                    _format_number(100.0 * success_rate, 1) + "%"
                    if success_rate is not None
                    else "N/A"
                ),
                body=_format_number(result.get("mean_body_frame_velocity_x")),
                cost_mean=_format_number(result.get("cost1_mean")),
                cost_median=_format_number(result.get("cost1_median")),
            )
        )
    lines.append("")
    return "\n".join(lines)


def write_summaries(state):
    _atomic_write_json(SUMMARY_JSON_PATH, build_summary(state))
    _atomic_write_text(SUMMARY_MD_PATH, build_summary_markdown(state))


def _preflight():
    errors = []
    if Path(sys.executable).resolve() != EXPECTED_PYTHON:
        errors.append(
            "pipeline must run with bruce_gym Python: {} (current: {})".format(
                EXPECTED_PYTHON, Path(sys.executable).resolve()
            )
        )
    if not (BASE_RUN / "model_3000.pt").is_file():
        errors.append("missing base checkpoint: {}".format(BASE_RUN / "model_3000.pt"))
    for alias in INITIAL_FIXED_ALIASES:
        checkpoint = CONTROL_SPECS[alias]["existing_run"] / "model_4001.pt"
        if not checkpoint.is_file():
            errors.append("missing existing checkpoint: {}".format(checkpoint))
    try:
        free_memory = query_free_memory_mib()
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        errors.append("cannot query nvidia-smi: {}".format(exc))
        free_memory = {}
    for gpu in (0, 1, 3, 4):
        if gpu not in free_memory:
            errors.append("GPU {} is not visible to nvidia-smi".format(gpu))
    if errors:
        raise RuntimeError("preflight failed:\n- " + "\n- ".join(errors))


class PipelineController:
    def __init__(self, poll_seconds, min_free_mib, retry_failed=False):
        self.poll_seconds = poll_seconds
        self.min_free_mib = min_free_mib
        self.retry_failed = retry_failed
        self.processes = {}
        self.stop_requested = False
        self.state = None

    def _handle_stop(self, signum, frame):
        del signum, frame
        self.stop_requested = True

    def _terminate_children(self):
        for job in _active_jobs(self.state).values():
            pid = job.get("pid")
            if not _pid_alive(pid):
                continue
            try:
                os.killpg(int(pid), signal.SIGTERM)
            except ProcessLookupError:
                continue
            job["status"] = "stopped"
            job["reason"] = "stopped by controller request"
            job["ended_at"] = _now()

    def _print_heartbeat(self):
        running = [
            "{}:{}@GPU{}".format(job["kind"], job["alias"], job["gpu"])
            for job in _active_jobs(self.state).values()
        ]
        print(
            "[{}] phase={} status={} jobs={} reason={}".format(
                _now(),
                self.state["phase"],
                self.state["status"],
                ",".join(running) or "none",
                self.state.get("reason") or "none",
            ),
            flush=True,
        )

    def _tick(self):
        _poll_jobs(self.state, self.processes)
        if self.state["status"] == "blocked":
            return

        phase = self.state["phase"]
        if phase == "waiting_initial_fixed":
            if _refresh_initial_fixed(self.state):
                self.state["phase"] = "training_controls"
        elif phase == "training_controls":
            if _all_stage_aliases_completed(
                self.state, "train", TRAIN_ALIASES
            ):
                self.state["phase"] = "fixed_evaluating_controls"
                self.state["reason"] = None
            else:
                free_memory = query_free_memory_mib()
                _schedule_training(
                    self.state,
                    self.processes,
                    free_memory,
                    self.min_free_mib,
                )
        elif phase == "fixed_evaluating_controls":
            for alias in NEW_FIXED_ALIASES:
                _ensure_completed_output_job(
                    self.state,
                    "fixed_eval",
                    alias,
                    fixed_output_dir(alias),
                    validate_fixed_evaluation,
                )
            if _all_stage_aliases_completed(
                self.state, "fixed_eval", NEW_FIXED_ALIASES
            ):
                self.state["phase"] = "validating_all_fixed"
                self.state["reason"] = None
            else:
                free_memory = query_free_memory_mib()
                _schedule_parallel_stage(
                    self.state,
                    self.processes,
                    "fixed_eval",
                    NEW_FIXED_ALIASES,
                    free_memory,
                    self.min_free_mib,
                )
        elif phase == "validating_all_fixed":
            if _refresh_all_fixed_results(self.state):
                self.state["phase"] = "training_distribution"
                self.state["reason"] = None
            else:
                failures = []
                for alias, result in self.state["fixed_evaluations"].items():
                    if not result.get("complete"):
                        failures.append("{} incomplete".format(alias))
                    elif not result.get("passed"):
                        failures.append(
                            "{}: {}".format(alias, "; ".join(result["errors"]))
                        )
                self.state["status"] = "blocked"
                self.state["reason"] = "fixed evaluation gate failed: {}".format(
                    " | ".join(failures)
                )
        elif phase == "training_distribution":
            for alias in CONTROL_ORDER:
                _ensure_completed_output_job(
                    self.state,
                    "train_dist",
                    alias,
                    train_dist_output_dir(alias),
                    validate_training_distribution,
                )
            if _all_stage_aliases_completed(
                self.state, "train_dist", CONTROL_ORDER
            ):
                _refresh_distribution_results(self.state)
                self.state["phase"] = "complete"
                self.state["status"] = "complete"
                self.state["reason"] = None
            else:
                free_memory = query_free_memory_mib()
                _schedule_parallel_stage(
                    self.state,
                    self.processes,
                    "train_dist",
                    CONTROL_ORDER,
                    free_memory,
                    self.min_free_mib,
                )
        elif phase == "complete":
            self.state["status"] = "complete"
        else:
            self.state["status"] = "blocked"
            self.state["reason"] = "unknown pipeline phase {!r}".format(phase)

    def run(self):
        _preflight()
        WORKFLOW_DIR.mkdir(parents=True, exist_ok=True)
        with LOCK_PATH.open("a+") as lock_handle:
            try:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError("another vx020 pipeline controller is running")

            self.state = load_state()
            if self.state["status"] == "blocked":
                if not self.retry_failed:
                    raise RuntimeError(
                        "pipeline is blocked: {}".format(self.state.get("reason"))
                    )
                resumed, reason = prepare_blocked_training_state_for_retry(
                    self.state
                )
                if not resumed:
                    raise RuntimeError(reason)
            if self.state["status"] == "complete":
                print("vx020 pipeline is already complete")
                return 0
            if self.state["status"] == "stopped":
                resumed, reason = prepare_stopped_state_for_resume(self.state)
                if not resumed:
                    raise RuntimeError(reason)

            self.state["controller_pid"] = os.getpid()
            self.state["status"] = "running"
            signal.signal(signal.SIGTERM, self._handle_stop)
            signal.signal(signal.SIGINT, self._handle_stop)
            save_state(self.state)

            while not self.stop_requested:
                try:
                    self._tick()
                except (OSError, subprocess.SubprocessError, ValueError) as exc:
                    self.state["reason"] = "temporary scheduler error: {}".format(exc)
                    print(self.state["reason"], file=sys.stderr, flush=True)
                save_state(self.state)
                write_summaries(self.state)
                self._print_heartbeat()
                if self.state["status"] in ("blocked", "complete"):
                    return 0 if self.state["status"] == "complete" else 2
                time.sleep(self.poll_seconds)

            self._terminate_children()
            self.state["status"] = "stopped"
            self.state["reason"] = "controller stopped by user"
            save_state(self.state)
            write_summaries(self.state)
            return 130


def print_status(state):
    print("status: {}".format(state.get("status")))
    print("phase: {}".format(state.get("phase")))
    print("reason: {}".format(state.get("reason") or "none"))
    controller_pid = state.get("controller_pid")
    print(
        "controller: pid={} alive={}".format(
            controller_pid, _pid_alive(controller_pid)
        )
    )
    print("jobs:")
    if not state.get("jobs"):
        print("  none")
    for job_id, job in sorted(state.get("jobs", {}).items()):
        print(
            "  {} status={} gpu={} pid={} reason={}".format(
                job_id,
                job.get("status"),
                job.get("gpu"),
                job.get("pid"),
                job.get("reason") or "none",
            )
        )
    print("fixed evaluations:")
    for alias in CONTROL_ORDER:
        result = state.get("fixed_evaluations", {}).get(alias, {})
        print(
            "  {} complete={} pass={} success={}/{} world_vx={} body_vx={}".format(
                alias,
                result.get("complete", False),
                result.get("passed", False),
                result.get("success_count", "N/A"),
                result.get("episode_count", "N/A"),
                _format_number(result.get("mean_velocity_x")),
                _format_number(result.get("mean_body_frame_velocity_x")),
            )
        )


def dry_run():
    print("vx020 pipeline stage graph:")
    print(
        "  waiting_initial_fixed -> training_controls -> "
        "fixed_evaluating_controls -> validating_all_fixed -> "
        "training_distribution -> complete"
    )
    print(
        "fixed gate: 20/20 success and world/body mean vx in [{:.2f}, {:.2f}]".format(
            VELOCITY_MIN, VELOCITY_MAX
        )
    )
    print("\ntraining commands:")
    for alias in TRAIN_ALIASES:
        gpu = CONTROL_SPECS[alias]["train_gpu"]
        print("  [{} GPU{}] {}".format(alias, gpu, " ".join(build_train_command(alias))))
    print("\nfixed evaluation commands (GPU3/GPU4 scheduled dynamically):")
    for alias in NEW_FIXED_ALIASES:
        placeholder = "<run_dir:{}>".format(alias)
        print("  [{}] {}".format(alias, " ".join(build_fixed_eval_command(alias, placeholder))))
    print("\ntraining-distribution commands (GPU3/GPU4 scheduled dynamically):")
    for alias in CONTROL_ORDER:
        run_dir = CONTROL_SPECS[alias]["existing_run"] or "<run_dir:{}>".format(alias)
        print("  [{}] {}".format(alias, " ".join(build_train_dist_command(alias, run_dir))))
    return 0


def _build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run", help="run or resume the controller")
    run_parser.add_argument("--poll-seconds", type=float, default=30.0)
    run_parser.add_argument(
        "--min-free-mib", type=int, default=DEFAULT_MIN_FREE_MIB
    )
    run_parser.add_argument(
        "--retry-failed",
        action="store_true",
        help=(
            "resume a blocked training stage, retaining only complete "
            "model_4001 cleanup-crash runs"
        ),
    )
    subparsers.add_parser("status", help="show persisted controller state")
    subparsers.add_parser("stop", help="stop the controller and active child jobs")
    subparsers.add_parser("dry-run", help="print all planned commands without running GPU jobs")
    return parser


def main(argv=None):
    args = _build_parser().parse_args(argv)
    if args.command == "dry-run":
        return dry_run()
    if args.command == "status":
        if not STATE_PATH.is_file():
            print("status: not_started")
            print("phase: waiting_initial_fixed")
            print("controller: not running")
            return 0
        state = load_state()
        print_status(state)
        return 0
    if args.command == "stop":
        state = load_state()
        pid = state.get("controller_pid")
        if not _pid_alive(pid):
            print("vx020 pipeline controller is not running")
            return 1
        os.kill(int(pid), signal.SIGTERM)
        print("sent SIGTERM to vx020 pipeline controller pid {}".format(pid))
        return 0
    if args.poll_seconds <= 0:
        raise SystemExit("--poll-seconds must be positive")
    if args.min_free_mib <= 0:
        raise SystemExit("--min-free-mib must be positive")
    try:
        return PipelineController(
            args.poll_seconds,
            args.min_free_mib,
            retry_failed=args.retry_failed,
        ).run()
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
