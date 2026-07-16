# SPDX-License-Identifier: BSD-3-Clause

"""Resumable paired multi-seed evaluation for the vx=0.2 controls."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from bruce_gym.naming import policy_id
from bruce_gym.paired_evaluation import (
    build_paired_summary,
    validate_calibration_output,
)


ROOT = Path(__file__).resolve().parents[2]
EXPECTED_PYTHON = Path(
    "/home/xy.chen/miniconda3/envs/bruce_gym/bin/python"
).resolve()
WORKFLOW_DIR = ROOT / "workflow_runs" / "vx020_paired_eval"
STATE_PATH = WORKFLOW_DIR / "state.json"
LOCK_PATH = WORKFLOW_DIR / "controller.lock"
STOP_PATH = WORKFLOW_DIR / "stop.request"
LOG_DIR = WORKFLOW_DIR / "logs"
SUMMARY_JSON_PATH = WORKFLOW_DIR / "summary.json"
SUMMARY_MD_PATH = WORKFLOW_DIR / "summary.md"

SOURCE_PIPELINE_SUMMARY = ROOT / "workflow_runs" / "vx020_seed0" / "summary.json"
BASE_RUN = ROOT / "pretrained_weights" / "eco_ppolag_0.2vel_8000cost_seed123"
OUTPUT_ROOT = ROOT / "energy_calibrations" / "paired100"

TARGET_COMMAND_X = 0.2
CALIBRATION_EPISODES = 100
CALIBRATION_ENVS = 1024
DEFAULT_EVAL_SEEDS = tuple(range(6))
DEFAULT_GPU = 3
DEFAULT_MIN_FREE_MIB = 8192
STATE_VERSION = 1

CONTROL_ORDER = ("rp8", "ra8", "jp8", "ja8", "lja10")
CONTROL_SPECS = {
    "rp8": {
        "mode": "rotor_positive_8",
        "cost_limit1": 49.157738095283506,
    },
    "ra8": {
        "mode": "rotor_abs_8",
        "cost_limit1": 81.79582343673707,
    },
    "jp8": {
        "mode": "joint_positive_8",
        "cost_limit1": 45.27336714076996,
    },
    "ja8": {
        "mode": "joint_abs_8",
        "cost_limit1": 72.39956164360046,
    },
    "lja10": {
        "mode": "legacy_joint_abs_10",
        "cost_limit1": 7557.11452758789,
    },
}


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
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_eval_seeds(value):
    if isinstance(value, (tuple, list)):
        seeds = [int(item) for item in value]
    else:
        text = str(value).strip()
        if not text:
            raise ValueError("evaluation seeds must not be empty")
        seeds = []
        for part in text.split(","):
            part = part.strip()
            if "-" in part:
                start_text, end_text = part.split("-", 1)
                start = int(start_text)
                end = int(end_text)
                if end < start:
                    raise ValueError("evaluation seed ranges must be ascending")
                seeds.extend(range(start, end + 1))
            else:
                seeds.append(int(part))
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("evaluation seeds must be non-empty and unique")
    if any(seed < 0 for seed in seeds):
        raise ValueError("evaluation seeds must be non-negative")
    return tuple(seeds)


def load_control_run_dirs(summary_path=SOURCE_PIPELINE_SUMMARY):
    summary_path = Path(summary_path)
    if not summary_path.is_file():
        raise ValueError(f"vx020 pipeline summary does not exist: {summary_path}")
    summary = _read_json(summary_path)
    if summary.get("status") != "complete":
        raise ValueError("vx020 source pipeline is not complete")
    run_dirs = summary.get("run_dirs", {})
    missing = [alias for alias in CONTROL_ORDER if alias not in run_dirs]
    if missing:
        raise ValueError(
            "vx020 source summary is missing run directories: "
            + ", ".join(missing)
        )
    return {alias: Path(run_dirs[alias]).resolve() for alias in CONTROL_ORDER}


def baseline_output_dir(seed):
    return OUTPUT_ROOT / policy_id(
        TARGET_COMMAND_X, "rotor_positive_8", seed, 3000
    )


def control_output_dir(alias, seed):
    return OUTPUT_ROOT / policy_id(
        TARGET_COMMAND_X, CONTROL_SPECS[alias]["mode"], seed, 4001
    )


def build_job_specs(eval_seeds, run_dirs):
    jobs = []
    for seed in eval_seeds:
        jobs.append(
            {
                "id": f"baseline_s{seed}",
                "kind": "baseline",
                "alias": "baseline",
                "mode": "rotor_positive_8",
                "eval_seed": int(seed),
                "load_run": str(BASE_RUN),
                "checkpoint": 3000,
                "output_dir": str(baseline_output_dir(seed)),
            }
        )
        for alias in CONTROL_ORDER:
            jobs.append(
                {
                    "id": f"{alias}_s{seed}",
                    "kind": "control",
                    "alias": alias,
                    "mode": CONTROL_SPECS[alias]["mode"],
                    "eval_seed": int(seed),
                    "load_run": str(run_dirs[alias]),
                    "checkpoint": 4001,
                    "output_dir": str(control_output_dir(alias, seed)),
                }
            )
    return jobs


def build_calibration_command(job, python_executable=EXPECTED_PYTHON):
    return [
        str(python_executable),
        "-u",
        "-m",
        "bruce_gym.scripts.calibrate_train_cost",
        "--task=bruce_ppolag",
        "--headless",
        "--resume",
        f"--load_run={job['load_run']}",
        f"--checkpoint={job['checkpoint']}",
        f"--energy_cost_mode={job['mode']}",
        f"--command_x={TARGET_COMMAND_X}",
        f"--calibration_episodes={CALIBRATION_EPISODES}",
        "--calibration_limit_fraction=0.95",
        f"--num_envs={CALIBRATION_ENVS}",
        f"--seed={job['eval_seed']}",
        f"--output_dir={job['output_dir']}",
        "--sim_device=cuda:0",
        "--rl_device=cuda:0",
    ]


def _job_state(spec):
    return {
        **spec,
        "status": "pending",
        "pid": None,
        "returncode": None,
        "log_path": str(LOG_DIR / f"{spec['id']}.log"),
        "started_at": None,
        "finished_at": None,
        "reason": None,
        "validation": None,
    }


def new_state(eval_seeds, gpu, min_free_mib, run_dirs):
    return {
        "version": STATE_VERSION,
        "status": "running",
        "phase": "paired_evaluation",
        "reason": None,
        "created_at": _now(),
        "updated_at": _now(),
        "controller_pid": os.getpid(),
        "heartbeat_at": _now(),
        "config": {
            "eval_seeds": list(eval_seeds),
            "gpu": int(gpu),
            "min_free_mib": int(min_free_mib),
            "source_pipeline_summary": str(SOURCE_PIPELINE_SUMMARY),
            "pairing_scope": "initial_scenario_manifest",
        },
        "run_dirs": {key: str(value) for key, value in run_dirs.items()},
        "jobs": [
            _job_state(spec) for spec in build_job_specs(eval_seeds, run_dirs)
        ],
    }


def save_state(state, path=STATE_PATH):
    state["updated_at"] = _now()
    _atomic_write_json(Path(path), state)


def load_state(path=STATE_PATH):
    state = _read_json(path)
    if state.get("version") != STATE_VERSION:
        raise ValueError(
            f"unsupported state version {state.get('version')!r}"
        )
    return state


def _baseline_job(state, seed):
    job_id = f"baseline_s{seed}"
    for job in state["jobs"]:
        if job["id"] == job_id:
            return job
    raise ValueError(f"missing baseline job for eval seed {seed}")


def validate_job(state, job):
    expected_fingerprint = None
    if job["kind"] == "control":
        baseline = _baseline_job(state, job["eval_seed"])
        validation = baseline.get("validation") or {}
        expected_fingerprint = validation.get(
            "scenario_fingerprint_sha256"
        )
        if not expected_fingerprint:
            return {
                "complete": False,
                "passed": False,
                "errors": ["baseline scenario fingerprint is unavailable"],
                "output_dir": job["output_dir"],
            }
    return validate_calibration_output(
        job["output_dir"],
        expected_mode=job["mode"],
        expected_seed=job["eval_seed"],
        expected_checkpoint=job["checkpoint"],
        expected_scenario_fingerprint=expected_fingerprint,
    )


def _pid_alive(pid):
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
    except (OSError, ProcessLookupError, ValueError):
        return False
    return True


def _gpu_free_memory_mib(gpu):
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.free",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    memory = {}
    for line in result.stdout.splitlines():
        index_text, free_text = [part.strip() for part in line.split(",", 1)]
        memory[int(index_text)] = int(free_text)
    if gpu not in memory:
        raise ValueError(f"GPU {gpu} was not reported by nvidia-smi")
    return memory[gpu]


def _child_environment(gpu):
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    python_bin = str(Path(sys.executable).resolve().parent)
    current_path = environment.get("PATH", "")
    environment["PATH"] = python_bin + os.pathsep + current_path
    return environment


def _mark_failed(state, job, reason):
    job["status"] = "failed"
    job["finished_at"] = _now()
    job["reason"] = reason
    state["status"] = "blocked"
    state["phase"] = "blocked"
    state["reason"] = f"{job['id']}: {reason}"


def _finish_job(state, job, returncode):
    job["pid"] = None
    job["returncode"] = returncode
    validation = validate_job(state, job)
    job["validation"] = validation
    if returncode not in (None, 0):
        _mark_failed(state, job, f"process exited with code {returncode}")
        return False
    if not validation.get("passed"):
        _mark_failed(
            state,
            job,
            "; ".join(validation.get("errors") or ["validation failed"]),
        )
        return False
    job["status"] = "completed"
    job["finished_at"] = _now()
    job["reason"] = None
    return True


def _adopt_existing_outputs(state, ignore_invalid_job_ids=()):
    ignore_invalid_job_ids = set(ignore_invalid_job_ids)
    for job in state["jobs"]:
        if job["status"] == "completed":
            continue
        if job["status"] == "running" and _pid_alive(job.get("pid")):
            break
        output_dir = Path(job["output_dir"])
        if not output_dir.exists():
            if job["status"] == "running":
                _mark_failed(state, job, "stale process has no output")
            break
        if job["id"] in ignore_invalid_job_ids:
            break
        validation = validate_job(state, job)
        job["validation"] = validation
        if validation.get("passed"):
            job["status"] = "completed"
            job["pid"] = None
            job["finished_at"] = _now()
            job["reason"] = None
            continue
        _mark_failed(
            state,
            job,
            "existing output is invalid: "
            + "; ".join(validation.get("errors") or []),
        )
        break


def _prepare_retry(state):
    failed = [job for job in state["jobs"] if job["status"] == "failed"]
    if len(failed) != 1:
        return False, "retry requires exactly one failed job"
    job = failed[0]
    job["status"] = "pending"
    job["pid"] = None
    job["returncode"] = None
    job["reason"] = None
    job["validation"] = None
    state["status"] = "running"
    state["phase"] = "paired_evaluation"
    state["reason"] = None
    return True, job["id"]


def _prepare_stopped(state):
    for job in state["jobs"]:
        if job["status"] in ("running", "stopped"):
            job["status"] = "pending"
            job["pid"] = None
            job["returncode"] = None
            job["reason"] = None
    state["status"] = "running"
    state["phase"] = "paired_evaluation"
    state["reason"] = None


def _start_job(state, job):
    gpu = int(state["config"]["gpu"])
    min_free = int(state["config"]["min_free_mib"])
    free_memory = _gpu_free_memory_mib(gpu)
    if free_memory < min_free:
        return None, f"GPU {gpu} free memory {free_memory} MiB < {min_free} MiB"
    command = build_calibration_command(job, python_executable=sys.executable)
    log_path = Path(job["log_path"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("ab")
    try:
        process = subprocess.Popen(
            command,
            cwd=str(ROOT),
            env=_child_environment(gpu),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    finally:
        log_handle.close()
    job["status"] = "running"
    job["pid"] = process.pid
    job["started_at"] = _now()
    job["reason"] = None
    return process, None


def _stop_active_job(state):
    for job in state["jobs"]:
        if job["status"] != "running":
            continue
        pid = job.get("pid")
        if _pid_alive(pid):
            try:
                os.killpg(int(pid), signal.SIGTERM)
            except (OSError, ProcessLookupError):
                pass
        job["status"] = "stopped"
        job["pid"] = None
        job["finished_at"] = _now()
        job["reason"] = "stop requested"
    state["status"] = "stopped"
    state["phase"] = "stopped"
    state["reason"] = "stop requested"
    state["controller_pid"] = None


def _format_percent(value):
    return "N/A" if value is None else f"{100.0 * value:.2f}%"


def _summary_markdown(summary):
    lines = [
        "# vx=0.2 paired multi-seed evaluation",
        "",
        "- Pairing scope: `initial_scenario_manifest`",
        "- Evaluation seeds: `" + ", ".join(map(str, summary["evaluation_seeds"])) + "`",
        "- Baseline: one `model_3000` trajectory per evaluation seed, with all five costs recomputed together",
        "",
        "| Group | Mode | Episodes | Success (95% CI) | Baseline success | Paired delta (95% CI) | Body vx | Cost mean | Cost vs baseline | Success cost vs limit |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    controls = summary["controls"]
    for alias in CONTROL_ORDER:
        mode = CONTROL_SPECS[alias]["mode"]
        result = controls[mode]
        success_ci = result["success_rate_ci95_wilson"]
        delta_ci = result["paired_success_rate_delta_ci95_bootstrap"]
        lines.append(
            "| {alias} | `{mode}` | {episodes} | {success} ({success_lo}, {success_hi}) | {baseline} | {delta} ({delta_lo}, {delta_hi}) | {velocity:.4f} | {cost:.4f} | {cost_delta} | {success_limit} |".format(
                alias=alias,
                mode=mode,
                episodes=result["episode_count"],
                success=_format_percent(result["success_rate"]),
                success_lo=_format_percent(success_ci[0]),
                success_hi=_format_percent(success_ci[1]),
                baseline=_format_percent(result["baseline_success_rate"]),
                delta=_format_percent(result["paired_success_rate_delta"]),
                delta_lo=_format_percent(delta_ci[0]),
                delta_hi=_format_percent(delta_ci[1]),
                velocity=result["weighted_body_velocity_x"],
                cost=result["cost1_mean"],
                cost_delta=_format_percent(result["cost1_change_from_baseline"]),
                success_limit=_format_percent(
                    result["cost1_success_relative_to_limit"]
                ),
            )
        )
    lines.extend(
        [
            "",
            "The scenario fingerprint covers initial terrain, commands, robot state, and sampled physical parameters. Runtime stochastic streams remain seed-controlled but are not replayed from the manifest.",
            "",
        ]
    )
    return "\n".join(lines)


def write_summary(state):
    baseline_dirs = {}
    control_dirs = {
        CONTROL_SPECS[alias]["mode"]: {} for alias in CONTROL_ORDER
    }
    for job in state["jobs"]:
        if job["status"] != "completed":
            raise ValueError(f"job {job['id']} is not complete")
        seed = int(job["eval_seed"])
        if job["kind"] == "baseline":
            baseline_dirs[seed] = job["output_dir"]
        else:
            control_dirs[job["mode"]][seed] = job["output_dir"]
    cost_limits = {
        spec["mode"]: spec["cost_limit1"] for spec in CONTROL_SPECS.values()
    }
    summary = build_paired_summary(
        baseline_dirs, control_dirs, cost_limits=cost_limits
    )
    summary["generated_at"] = _now()
    summary["output_root"] = str(OUTPUT_ROOT)
    _atomic_write_json(SUMMARY_JSON_PATH, summary)
    _atomic_write_text(SUMMARY_MD_PATH, _summary_markdown(summary))
    return summary


def _validate_inputs(run_dirs, gpu=None):
    errors = []
    if Path(sys.executable).resolve() != EXPECTED_PYTHON:
        errors.append(
            f"expected conda bruce_gym Python {EXPECTED_PYTHON}, got "
            f"{Path(sys.executable).resolve()}"
        )
    base_checkpoint = BASE_RUN / "model_3000.pt"
    if not base_checkpoint.is_file():
        errors.append(f"missing baseline checkpoint: {base_checkpoint}")
    for alias, run_dir in run_dirs.items():
        checkpoint = run_dir / "model_4001.pt"
        if not checkpoint.is_file():
            errors.append(f"missing {alias} checkpoint: {checkpoint}")
    if gpu is not None and int(gpu) == 5:
        errors.append("GPU 5 is unavailable on this server")
    if gpu is not None and int(gpu) < 0:
        errors.append("GPU index must be non-negative")
    return errors


def dry_run(args):
    eval_seeds = parse_eval_seeds(args.eval_seeds)
    run_dirs = load_control_run_dirs()
    errors = _validate_inputs(run_dirs, gpu=args.gpu)
    jobs = build_job_specs(eval_seeds, run_dirs)
    print(
        f"paired evaluation: {len(eval_seeds)} seeds, {len(jobs)} jobs, "
        f"GPU {args.gpu}, sequential"
    )
    for job in jobs:
        command = build_calibration_command(job)
        print(f"[{job['id']}] " + " ".join(map(str, command)))
    if errors:
        print("dry-run FAILED:")
        for error in errors:
            print(f"  - {error}")
        return 1
    print("dry-run PASS")
    return 0


def run_controller(args):
    eval_seeds = parse_eval_seeds(args.eval_seeds)
    run_dirs = load_control_run_dirs()
    errors = _validate_inputs(run_dirs, gpu=args.gpu)
    if errors:
        raise ValueError("; ".join(errors))

    WORKFLOW_DIR.mkdir(parents=True, exist_ok=True)
    lock_handle = LOCK_PATH.open("a+")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError("another paired-evaluation controller is running")

    retry_job_id = None
    if STATE_PATH.is_file():
        state = load_state()
        configured_seeds = tuple(state["config"]["eval_seeds"])
        if configured_seeds != eval_seeds:
            raise ValueError(
                f"existing state uses eval seeds {configured_seeds}, "
                f"requested {eval_seeds}"
            )
        if int(state["config"]["gpu"]) != int(args.gpu):
            raise ValueError("existing state uses a different GPU")
        if state["status"] == "complete":
            print(f"workflow already complete: {SUMMARY_MD_PATH}")
            return 0
        if state["status"] == "blocked":
            if not args.retry_failed:
                raise RuntimeError(
                    "workflow is blocked; inspect status and use "
                    "run --retry-failed after fixing the cause"
                )
            prepared, retry_job_id = _prepare_retry(state)
            if not prepared:
                raise RuntimeError(retry_job_id)
        elif state["status"] == "stopped":
            _prepare_stopped(state)
    else:
        state = new_state(eval_seeds, args.gpu, args.min_free_mib, run_dirs)

    if STOP_PATH.exists():
        STOP_PATH.unlink()
    state["controller_pid"] = os.getpid()
    state["heartbeat_at"] = _now()
    _adopt_existing_outputs(
        state,
        ignore_invalid_job_ids=(retry_job_id,) if retry_job_id else (),
    )
    save_state(state)
    if state["status"] == "blocked":
        raise RuntimeError(state["reason"])

    active_process = None
    active_job = None
    last_update = 0.0
    while True:
        if STOP_PATH.exists():
            _stop_active_job(state)
            save_state(state)
            print("stop request handled")
            return 0

        if active_job is not None:
            returncode = active_process.poll()
            if returncode is None:
                if time.time() - last_update >= 30.0:
                    print(
                        f"[{_now()}] running {active_job['id']} "
                        f"pid={active_job['pid']}"
                    )
                    last_update = time.time()
                state["heartbeat_at"] = _now()
                save_state(state)
                time.sleep(5.0)
                continue
            completed = _finish_job(state, active_job, returncode)
            save_state(state)
            print(
                f"[{_now()}] {active_job['id']} "
                f"{'completed' if completed else 'failed'}"
            )
            active_process = None
            active_job = None
            if not completed:
                return 1

        resumed_jobs = [
            job for job in state["jobs"] if job["status"] == "running"
        ]
        if resumed_jobs:
            resumed_job = resumed_jobs[0]
            if _pid_alive(resumed_job.get("pid")):
                state["heartbeat_at"] = _now()
                save_state(state)
                if time.time() - last_update >= 30.0:
                    print(
                        f"[{_now()}] attached to {resumed_job['id']} "
                        f"pid={resumed_job['pid']}"
                    )
                    last_update = time.time()
                time.sleep(5.0)
                continue
            completed = _finish_job(state, resumed_job, returncode=None)
            save_state(state)
            if not completed:
                return 1
            continue

        pending = [job for job in state["jobs"] if job["status"] == "pending"]
        if not pending:
            write_summary(state)
            state["status"] = "complete"
            state["phase"] = "complete"
            state["reason"] = None
            state["controller_pid"] = None
            save_state(state)
            print(f"paired evaluation complete: {SUMMARY_MD_PATH}")
            return 0

        next_job = pending[0]
        process, wait_reason = _start_job(state, next_job)
        if process is None:
            state["reason"] = wait_reason
            state["heartbeat_at"] = _now()
            save_state(state)
            if time.time() - last_update >= 30.0:
                print(f"[{_now()}] waiting: {wait_reason}")
                last_update = time.time()
            time.sleep(15.0)
            continue
        state["reason"] = None
        save_state(state)
        print(
            f"[{_now()}] started {next_job['id']} pid={process.pid} "
            f"GPU={state['config']['gpu']}"
        )
        active_process = process
        active_job = next_job


def status_command(_args):
    if not STATE_PATH.is_file():
        print("status: not_started")
        return 0
    state = load_state()
    counts = {}
    for job in state["jobs"]:
        counts[job["status"]] = counts.get(job["status"], 0) + 1
    print(f"status: {state['status']}")
    print(f"phase: {state['phase']}")
    print(f"reason: {state.get('reason') or 'N/A'}")
    controller_pid = state.get("controller_pid")
    print(
        f"controller: pid={controller_pid} alive={_pid_alive(controller_pid)}"
    )
    print(
        "jobs: "
        + ", ".join(f"{key}={value}" for key, value in sorted(counts.items()))
    )
    for job in state["jobs"]:
        if job["status"] in ("running", "failed", "stopped"):
            print(
                f"  {job['id']}: {job['status']} pid={job.get('pid')} "
                f"reason={job.get('reason') or 'N/A'}"
            )
    if state["status"] == "complete":
        print(f"summary: {SUMMARY_MD_PATH}")
    return 0


def stop_command(_args):
    if not STATE_PATH.is_file():
        print("status: not_started")
        return 0
    _atomic_write_text(STOP_PATH, _now() + "\n")
    print(f"stop requested: {STOP_PATH}")
    return 0


def summarize_command(_args):
    if not STATE_PATH.is_file():
        raise RuntimeError("workflow state does not exist")
    state = load_state()
    summary = write_summary(state)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"wrote: {SUMMARY_MD_PATH}")
    return 0


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--eval-seeds", default="0-5")
    run_parser.add_argument("--gpu", type=int, default=DEFAULT_GPU)
    run_parser.add_argument(
        "--min-free-mib", type=int, default=DEFAULT_MIN_FREE_MIB
    )
    run_parser.add_argument("--retry-failed", action="store_true")

    dry_parser = subparsers.add_parser("dry-run")
    dry_parser.add_argument("--eval-seeds", default="0-5")
    dry_parser.add_argument("--gpu", type=int, default=DEFAULT_GPU)

    subparsers.add_parser("status")
    subparsers.add_parser("stop")
    subparsers.add_parser("summarize")
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    if args.command == "run":
        return run_controller(args)
    if args.command == "dry-run":
        return dry_run(args)
    if args.command == "status":
        return status_command(args)
    if args.command == "stop":
        return stop_command(args)
    if args.command == "summarize":
        return summarize_command(args)
    raise ValueError(f"unknown command: {args.command}")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
