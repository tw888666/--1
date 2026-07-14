# SPDX-License-Identifier: BSD-3-Clause

"""Pure-Python validation and statistics for paired vx=0.2 evaluation."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import statistics
from pathlib import Path

from bruce_gym.rotor_energy import CONTROL_ENERGY_COST_MODES


TARGET_COMMAND_X = 0.2
EXPECTED_EPISODES = 100
EXPECTED_NUM_ENVS = 1024


def _read_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_csv(path: Path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def canonical_fingerprint(value) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_calibration_output(
    output_dir,
    expected_mode,
    expected_seed,
    expected_checkpoint,
    expected_scenario_fingerprint=None,
):
    output_dir = Path(output_dir)
    errors = []
    paths = {
        "metadata": output_dir / "metadata.json",
        "rows": output_dir / "episode_costs.csv",
        "summary": output_dir / "calibration_summary.json",
        "mode_summaries": output_dir / "cost_mode_summaries.json",
        "manifest": output_dir / "scenario_manifest.json",
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        return {
            "complete": False,
            "passed": False,
            "errors": ["missing files: {}".format(", ".join(missing))],
            "output_dir": str(output_dir),
        }

    try:
        metadata = _read_json(paths["metadata"])
        rows = _read_csv(paths["rows"])
        summary = _read_json(paths["summary"])
        mode_summaries = _read_json(paths["mode_summaries"])
        manifest = _read_json(paths["manifest"])
    except (OSError, ValueError, csv.Error, json.JSONDecodeError) as exc:
        return {
            "complete": False,
            "passed": False,
            "errors": [f"failed to read output: {exc}"],
            "output_dir": str(output_dir),
        }

    expected_metadata = {
        "checkpoint": int(expected_checkpoint),
        "seed": int(expected_seed),
        "energy_cost_mode": expected_mode,
        "calibration_profile": "training_distribution_fixed_lin_vel_x",
        "calibration_episodes": EXPECTED_EPISODES,
        "num_envs": EXPECTED_NUM_ENVS,
    }
    for key, expected in expected_metadata.items():
        if metadata.get(key) != expected:
            errors.append(
                f"metadata {key}={metadata.get(key)!r}, expected {expected!r}"
            )
    try:
        command_x = float(metadata.get("command_x"))
    except (TypeError, ValueError):
        command_x = math.nan
    if not math.isclose(command_x, TARGET_COMMAND_X, abs_tol=1e-12):
        errors.append(
            f"metadata command_x={metadata.get('command_x')!r}, "
            f"expected {TARGET_COMMAND_X}"
        )

    reported_modes = tuple(metadata.get("reported_cost_modes", ()))
    if reported_modes != tuple(CONTROL_ENERGY_COST_MODES):
        errors.append(
            "reported_cost_modes do not match the five control cost modes"
        )
    if set(mode_summaries) != set(CONTROL_ENERGY_COST_MODES):
        errors.append("cost_mode_summaries does not contain all five modes")

    manifest_sha = manifest.get("sha256")
    manifest_payload = dict(manifest)
    manifest_payload.pop("sha256", None)
    try:
        computed_sha = canonical_fingerprint(manifest_payload)
    except (TypeError, ValueError) as exc:
        computed_sha = None
        errors.append(f"scenario manifest is not canonical JSON: {exc}")
    if not manifest_sha or manifest_sha != computed_sha:
        errors.append("scenario manifest sha256 does not match its content")
    if metadata.get("scenario_fingerprint_sha256") != manifest_sha:
        errors.append("metadata scenario fingerprint does not match manifest")
    if manifest.get("seed") != int(expected_seed):
        errors.append("scenario manifest seed does not match evaluation seed")
    if expected_scenario_fingerprint is not None:
        if manifest_sha != expected_scenario_fingerprint:
            errors.append(
                "scenario fingerprint does not match the baseline for this seed"
            )

    if len(rows) != EXPECTED_EPISODES:
        errors.append(
            f"episode count {len(rows)}, expected {EXPECTED_EPISODES}"
        )
    env_ids = []
    success_count = 0
    total_duration = 0.0
    total_distance = 0.0
    selected_costs = []
    success_costs = []
    for row_number, row in enumerate(rows, start=1):
        try:
            env_id = int(row["env_id"])
            duration = float(row["duration_s"])
            distance = float(row["body_frame_distance_x"])
            selected_cost = float(row["cost1"])
            reconstructed_cost = float(row[f"cost1_{expected_mode}"])
            all_mode_costs = [
                float(row[f"cost1_{mode}"])
                for mode in CONTROL_ENERGY_COST_MODES
            ]
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"row {row_number} has invalid fields: {exc}")
            continue
        values = [duration, distance, selected_cost, reconstructed_cost]
        values.extend(all_mode_costs)
        if not all(math.isfinite(value) for value in values):
            errors.append(f"row {row_number} has non-finite values")
            continue
        if duration <= 0.0:
            errors.append(f"row {row_number} has non-positive duration")
        if not math.isclose(
            selected_cost, reconstructed_cost, rel_tol=1e-5, abs_tol=1e-4
        ):
            errors.append(
                f"row {row_number} selected cost does not match "
                f"cost1_{expected_mode}"
            )
        outcome = row.get("episode_outcome")
        if outcome not in ("success", "fall"):
            errors.append(f"row {row_number} has invalid outcome {outcome!r}")
        env_ids.append(env_id)
        total_duration += duration
        total_distance += distance
        selected_costs.append(selected_cost)
        if outcome == "success":
            success_count += 1
            success_costs.append(selected_cost)

    if len(env_ids) != len(set(env_ids)):
        errors.append("episode rows contain duplicate env_id values")
    manifest_env_ids = manifest.get("cohort_env_ids")
    if sorted(env_ids) != sorted(manifest_env_ids or []):
        errors.append("episode env_id values do not match scenario manifest")
    if summary.get("episode_count") != len(rows):
        errors.append("calibration summary episode count does not match CSV")
    if summary.get("success_count") != success_count:
        errors.append("calibration summary success count does not match CSV")

    return {
        "complete": True,
        "passed": not errors,
        "errors": errors,
        "output_dir": str(output_dir),
        "scenario_fingerprint_sha256": manifest_sha,
        "episode_count": len(rows),
        "success_count": success_count,
        "success_rate": success_count / len(rows) if rows else None,
        "weighted_body_velocity_x": (
            total_distance / total_duration if total_duration > 0.0 else None
        ),
        "cost1_mean": (
            statistics.fmean(selected_costs) if selected_costs else None
        ),
        "cost1_success_mean": (
            statistics.fmean(success_costs) if success_costs else None
        ),
    }


def wilson_interval(successes, total, z=1.959963984540054):
    if total <= 0:
        raise ValueError("total must be positive")
    if not 0 <= successes <= total:
        raise ValueError("successes must be between zero and total")
    proportion = successes / total
    z2 = z * z
    denominator = 1.0 + z2 / total
    center = (proportion + z2 / (2.0 * total)) / denominator
    radius = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z2 / (4.0 * total * total)
        )
        / denominator
    )
    return center - radius, center + radius


def paired_bootstrap_mean_interval(
    values, confidence=0.95, samples=10000, seed=20260714
):
    values = [float(value) for value in values]
    if not values:
        raise ValueError("values must not be empty")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one")
    if samples <= 0:
        raise ValueError("samples must be positive")
    generator = random.Random(seed)
    count = len(values)
    means = []
    for _ in range(samples):
        means.append(
            sum(values[generator.randrange(count)] for _ in range(count))
            / count
        )
    means.sort()
    tail = (1.0 - confidence) / 2.0
    lower_index = max(0, min(samples - 1, int(tail * samples)))
    upper_index = max(
        0, min(samples - 1, int((1.0 - tail) * samples) - 1)
    )
    return means[lower_index], means[upper_index]


def _mode_cost(row, mode):
    return float(row[f"cost1_{mode}"])


def build_paired_summary(baseline_dirs, control_dirs, cost_limits):
    """Aggregate baseline and control runs keyed by evaluation seed."""

    baseline_rows = {}
    fingerprints = {}
    for seed, output_dir in sorted(baseline_dirs.items()):
        output_dir = Path(output_dir)
        rows = _read_csv(output_dir / "episode_costs.csv")
        metadata = _read_json(output_dir / "metadata.json")
        baseline_rows[int(seed)] = {
            int(row["env_id"]): row for row in rows
        }
        fingerprints[int(seed)] = metadata["scenario_fingerprint_sha256"]

    controls = {}
    for mode, dirs_by_seed in control_dirs.items():
        all_rows = []
        all_baseline_rows = []
        paired_success_differences = []
        per_seed = []
        for seed, output_dir in sorted(dirs_by_seed.items()):
            rows = _read_csv(Path(output_dir) / "episode_costs.csv")
            by_env = {int(row["env_id"]): row for row in rows}
            reference = baseline_rows[int(seed)]
            if set(by_env) != set(reference):
                raise ValueError(
                    f"mode {mode} seed {seed} has mismatched env_id values"
                )
            successes = sum(
                row["episode_outcome"] == "success" for row in rows
            )
            baseline_successes = sum(
                row["episode_outcome"] == "success"
                for row in reference.values()
            )
            for env_id, row in by_env.items():
                current = 1 if row["episode_outcome"] == "success" else 0
                baseline = (
                    1
                    if reference[env_id]["episode_outcome"] == "success"
                    else 0
                )
                paired_success_differences.append(current - baseline)
            all_rows.extend(rows)
            all_baseline_rows.extend(reference.values())
            per_seed.append(
                {
                    "eval_seed": int(seed),
                    "success_count": successes,
                    "baseline_success_count": baseline_successes,
                    "success_rate_delta": (
                        successes - baseline_successes
                    )
                    / len(rows),
                }
            )

        successes = sum(
            row["episode_outcome"] == "success" for row in all_rows
        )
        baseline_successes = sum(
            row["episode_outcome"] == "success"
            for row in all_baseline_rows
        )
        total = len(all_rows)
        durations = [float(row["duration_s"]) for row in all_rows]
        distances = [float(row["body_frame_distance_x"]) for row in all_rows]
        current_costs = [_mode_cost(row, mode) for row in all_rows]
        current_success_costs = [
            _mode_cost(row, mode)
            for row in all_rows
            if row["episode_outcome"] == "success"
        ]
        baseline_costs = [
            _mode_cost(row, mode) for row in all_baseline_rows
        ]
        success_ci = wilson_interval(successes, total)
        paired_ci = paired_bootstrap_mean_interval(
            paired_success_differences
        )
        cost_mean = statistics.fmean(current_costs)
        baseline_cost_mean = statistics.fmean(baseline_costs)
        limit = float(cost_limits[mode])
        controls[mode] = {
            "episode_count": total,
            "success_count": successes,
            "success_rate": successes / total,
            "success_rate_ci95_wilson": list(success_ci),
            "baseline_success_count": baseline_successes,
            "baseline_success_rate": baseline_successes / total,
            "paired_success_rate_delta": statistics.fmean(
                paired_success_differences
            ),
            "paired_success_rate_delta_ci95_bootstrap": list(paired_ci),
            "weighted_body_velocity_x": sum(distances) / sum(durations),
            "cost_limit1": limit,
            "cost1_mean": cost_mean,
            "cost1_median": statistics.median(current_costs),
            "cost1_success_mean": (
                statistics.fmean(current_success_costs)
                if current_success_costs
                else None
            ),
            "baseline_cost1_mean": baseline_cost_mean,
            "cost1_change_from_baseline": (
                cost_mean / baseline_cost_mean - 1.0
            ),
            "cost1_relative_to_limit": cost_mean / limit - 1.0,
            "cost1_success_relative_to_limit": (
                statistics.fmean(current_success_costs) / limit - 1.0
                if current_success_costs
                else None
            ),
            "per_seed": per_seed,
        }

    return {
        "evaluation_seeds": sorted(baseline_rows),
        "pairing_scope": "initial_scenario_manifest",
        "scenario_fingerprints": fingerprints,
        "controls": controls,
    }
