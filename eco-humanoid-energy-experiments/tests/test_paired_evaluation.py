import csv
import json
import tempfile
import unittest
from pathlib import Path

from bruce_gym.paired_evaluation import (
    canonical_fingerprint,
    build_paired_summary,
    paired_bootstrap_mean_interval,
    validate_calibration_output,
    wilson_interval,
)
from bruce_gym.rotor_energy import CONTROL_ENERGY_COST_MODES
from bruce_gym.scripts import vx020_paired_eval as pipeline


def _write_output(
    output_dir,
    mode="rotor_positive_8",
    seed=0,
    checkpoint=4001,
    successes=80,
    costs=None,
    manifest_marker="shared",
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    costs = costs or {
        "rotor_positive_8": 1.0,
        "rotor_abs_8": 2.0,
        "joint_positive_8": 3.0,
        "joint_abs_8": 4.0,
        "legacy_joint_abs_10": 5.0,
    }
    env_ids = list(range(100))
    manifest = {
        "schema_version": 1,
        "seed": seed,
        "command_x": 0.2,
        "num_envs": 1024,
        "cohort_env_ids": env_ids,
        "initial_tensors": {"marker": manifest_marker},
        "scope": "initial scenario pairing",
    }
    manifest["sha256"] = canonical_fingerprint(manifest)
    metadata = {
        "checkpoint": checkpoint,
        "seed": seed,
        "energy_cost_mode": mode,
        "calibration_profile": "training_distribution_fixed_lin_vel_x",
        "calibration_episodes": 100,
        "num_envs": 1024,
        "command_x": 0.2,
        "reported_cost_modes": list(CONTROL_ENERGY_COST_MODES),
        "scenario_fingerprint_sha256": manifest["sha256"],
    }
    rows = []
    for env_id in env_ids:
        row = {
            "episode_id": env_id,
            "env_id": env_id,
            "episode_outcome": "success" if env_id < successes else "fall",
            "timeout": 1.0 if env_id < successes else 0.0,
            "fall": 0.0 if env_id < successes else 1.0,
            "steps": 100,
            "duration_s": 1.0,
            "cost1": costs[mode],
            "cost1_per_second": costs[mode],
            "body_frame_distance_x": 0.2,
            "mean_body_frame_velocity_x": 0.2,
        }
        for reported_mode in CONTROL_ENERGY_COST_MODES:
            row[f"cost1_{reported_mode}"] = costs[reported_mode]
        rows.append(row)

    with (output_dir / "episode_costs.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    (output_dir / "scenario_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    (output_dir / "calibration_summary.json").write_text(
        json.dumps({"episode_count": 100, "success_count": successes}),
        encoding="utf-8",
    )
    (output_dir / "cost_mode_summaries.json").write_text(
        json.dumps({mode: {} for mode in CONTROL_ENERGY_COST_MODES}),
        encoding="utf-8",
    )
    return manifest["sha256"]


class CalibrationOutputValidationTests(unittest.TestCase):
    def test_accepts_complete_paired_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            fingerprint = _write_output(tmp)

            result = validate_calibration_output(
                tmp,
                expected_mode="rotor_positive_8",
                expected_seed=0,
                expected_checkpoint=4001,
                expected_scenario_fingerprint=fingerprint,
            )

            self.assertTrue(result["passed"], result["errors"])
            self.assertEqual(result["success_count"], 80)
            self.assertAlmostEqual(result["weighted_body_velocity_x"], 0.2)

    def test_rejects_scenario_fingerprint_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_output(tmp)

            result = validate_calibration_output(
                tmp,
                expected_mode="rotor_positive_8",
                expected_seed=0,
                expected_checkpoint=4001,
                expected_scenario_fingerprint="different",
            )

            self.assertFalse(result["passed"])
            self.assertIn("baseline", " ".join(result["errors"]))

    def test_rejects_selected_cost_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_output(tmp)
            path = Path(tmp) / "episode_costs.csv"
            with path.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            rows[0]["cost1"] = "999"
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

            result = validate_calibration_output(
                tmp,
                expected_mode="rotor_positive_8",
                expected_seed=0,
                expected_checkpoint=4001,
            )

            self.assertFalse(result["passed"])
            self.assertIn("selected cost", " ".join(result["errors"]))


class PairedStatisticsTests(unittest.TestCase):
    def test_wilson_interval_contains_observed_rate(self):
        lower, upper = wilson_interval(80, 100)

        self.assertLess(lower, 0.8)
        self.assertGreater(upper, 0.8)
        self.assertAlmostEqual(lower, 0.7112, places=3)
        self.assertAlmostEqual(upper, 0.8666, places=3)

    def test_bootstrap_is_deterministic_and_handles_zero_delta(self):
        first = paired_bootstrap_mean_interval([0.0] * 100, samples=200)
        second = paired_bootstrap_mean_interval([0.0] * 100, samples=200)

        self.assertEqual(first, (0.0, 0.0))
        self.assertEqual(first, second)

    def test_builds_multi_seed_paired_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline_dirs = {}
            control_dirs = {"joint_abs_8": {}}
            for seed in (0, 1):
                baseline = root / f"baseline_{seed}"
                control = root / f"control_{seed}"
                _write_output(
                    baseline,
                    mode="rotor_positive_8",
                    seed=seed,
                    checkpoint=3000,
                    successes=80,
                )
                _write_output(
                    control,
                    mode="joint_abs_8",
                    seed=seed,
                    checkpoint=4001,
                    successes=85,
                    costs={
                        "rotor_positive_8": 1.0,
                        "rotor_abs_8": 2.0,
                        "joint_positive_8": 2.5,
                        "joint_abs_8": 3.0,
                        "legacy_joint_abs_10": 5.0,
                    },
                )
                baseline_dirs[seed] = baseline
                control_dirs["joint_abs_8"][seed] = control

            summary = build_paired_summary(
                baseline_dirs,
                control_dirs,
                cost_limits={"joint_abs_8": 4.0},
            )

            result = summary["controls"]["joint_abs_8"]
            self.assertEqual(result["episode_count"], 200)
            self.assertEqual(result["success_count"], 170)
            self.assertAlmostEqual(result["paired_success_rate_delta"], 0.05)
            self.assertAlmostEqual(result["cost1_change_from_baseline"], -0.25)


class PipelineConstructionTests(unittest.TestCase):
    def test_parses_seed_ranges(self):
        self.assertEqual(pipeline.parse_eval_seeds("0-2,5"), (0, 1, 2, 5))
        with self.assertRaises(ValueError):
            pipeline.parse_eval_seeds("2-0")
        with self.assertRaises(ValueError):
            pipeline.parse_eval_seeds("0,0")

    def test_builds_one_baseline_and_five_controls_per_seed(self):
        run_dirs = {
            alias: Path("/tmp") / alias for alias in pipeline.CONTROL_ORDER
        }

        jobs = pipeline.build_job_specs((0, 1), run_dirs)

        self.assertEqual(len(jobs), 12)
        self.assertEqual(jobs[0]["id"], "baseline_s0")
        self.assertEqual(jobs[1]["id"], "rp8_s0")
        self.assertEqual(jobs[6]["id"], "baseline_s1")
        self.assertTrue(
            jobs[0]["output_dir"].endswith(
                "energy_calibrations/paired100/vx020_rp8_s00_model3000"
            )
        )
        self.assertTrue(
            jobs[4]["output_dir"].endswith(
                "energy_calibrations/paired100/vx020_ja8_s00_model4001"
            )
        )

    def test_command_locks_vx_seed_gpu_local_device_and_output(self):
        job = {
            "load_run": "/tmp/run",
            "checkpoint": 4001,
            "mode": "joint_abs_8",
            "eval_seed": 4,
            "output_dir": "/tmp/output",
        }

        command = pipeline.build_calibration_command(
            job, python_executable="python"
        )

        self.assertIn("--command_x=0.2", command)
        self.assertIn("--seed=4", command)
        self.assertIn("--calibration_episodes=100", command)
        self.assertIn("--num_envs=1024", command)
        self.assertIn("--sim_device=cuda:0", command)
        self.assertIn("--output_dir=/tmp/output", command)

    def test_retry_requeues_only_failed_job(self):
        run_dirs = {
            alias: Path("/tmp") / alias for alias in pipeline.CONTROL_ORDER
        }
        state = pipeline.new_state((0,), 3, 8192, run_dirs)
        state["status"] = "blocked"
        state["phase"] = "blocked"
        state["jobs"][2]["status"] = "failed"

        prepared, job_id = pipeline._prepare_retry(state)

        self.assertTrue(prepared)
        self.assertEqual(job_id, "ra8_s0")
        self.assertEqual(state["jobs"][2]["status"], "pending")
        self.assertEqual(state["status"], "running")

    def test_rejects_unavailable_gpu5(self):
        errors = pipeline._validate_inputs({}, gpu=5)
        self.assertIn("GPU 5", " ".join(errors))


if __name__ == "__main__":
    unittest.main()
