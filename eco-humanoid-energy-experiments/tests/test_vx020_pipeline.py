# SPDX-License-Identifier: BSD-3-Clause

import csv
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from bruce_gym.scripts import vx020_pipeline as pipeline


def _write_fixed_output(
    output_dir,
    mode="rotor_positive_8",
    episodes=20,
    success_count=20,
    world_vx=0.2,
    body_vx=0.2,
    command_x=0.2,
    checkpoint=4001,
):
    output_dir.mkdir(parents=True)
    (output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "energy_cost_mode": mode,
                "command_x": command_x,
                "checkpoint": checkpoint,
            }
        ),
        encoding="utf-8",
    )
    with (output_dir / "episode_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "episode_outcome",
                "mean_velocity_x",
                "mean_body_frame_velocity_x",
            ),
        )
        writer.writeheader()
        for index in range(episodes):
            writer.writerow(
                {
                    "episode_outcome": (
                        "success" if index < success_count else "fall"
                    ),
                    "mean_velocity_x": world_vx,
                    "mean_body_frame_velocity_x": body_vx,
                }
            )


def _write_distribution_output(output_dir, mode="rotor_positive_8", episodes=100):
    output_dir.mkdir(parents=True)
    (output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "energy_cost_mode": mode,
                "command_x": 0.2,
                "checkpoint": 4001,
            }
        ),
        encoding="utf-8",
    )
    with (output_dir / "episode_costs.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "episode_outcome",
                "mean_body_frame_velocity_x",
                "cost1",
            ),
        )
        writer.writeheader()
        for index in range(episodes):
            writer.writerow(
                {
                    "episode_outcome": "success" if index < 80 else "fall",
                    "mean_body_frame_velocity_x": 0.2,
                    "cost1": 10.0,
                }
            )
    (output_dir / "calibration_summary.json").write_text(
        json.dumps(
            {
                "cost1_all_episodes": {"mean": 10.0, "median": 9.5},
                "recommended_cost_limit1_all_episodes": 9.5,
            }
        ),
        encoding="utf-8",
    )


class FixedEvaluationValidationTests(unittest.TestCase):
    def test_accepts_values_on_velocity_boundaries(self):
        for velocity in (0.16, 0.24):
            with self.subTest(velocity=velocity), tempfile.TemporaryDirectory() as tmp:
                output_dir = Path(tmp) / "eval"
                _write_fixed_output(
                    output_dir, world_vx=velocity, body_vx=velocity
                )

                result = pipeline.validate_fixed_evaluation(
                    output_dir, "rotor_positive_8"
                )

                self.assertTrue(result["complete"])
                self.assertTrue(result["passed"], result["errors"])

    def test_rejects_out_of_range_speed_and_fall(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "eval"
            _write_fixed_output(
                output_dir, success_count=19, world_vx=0.159, body_vx=0.241
            )

            result = pipeline.validate_fixed_evaluation(
                output_dir, "rotor_positive_8"
            )

            self.assertFalse(result["passed"])
            self.assertIn("success count", " ".join(result["errors"]))
            self.assertIn("mean_velocity_x", " ".join(result["errors"]))
            self.assertIn(
                "mean_body_frame_velocity_x", " ".join(result["errors"])
            )

    def test_rejects_missing_episode_and_metadata_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "eval"
            _write_fixed_output(
                output_dir,
                mode="joint_abs_8",
                episodes=19,
                success_count=19,
                command_x=0.1,
                checkpoint=4000,
            )

            result = pipeline.validate_fixed_evaluation(
                output_dir, "rotor_positive_8"
            )

            errors = " ".join(result["errors"])
            self.assertFalse(result["passed"])
            self.assertIn("energy_cost_mode", errors)
            self.assertIn("command_x", errors)
            self.assertIn("checkpoint", errors)
            self.assertIn("episode count", errors)

    def test_rejects_non_finite_speed(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "eval"
            _write_fixed_output(output_dir, world_vx=math.nan)

            result = pipeline.validate_fixed_evaluation(
                output_dir, "rotor_positive_8"
            )

            self.assertFalse(result["passed"])
            self.assertIn("non-finite", " ".join(result["errors"]))


class TrainingDistributionValidationTests(unittest.TestCase):
    def test_summarizes_complete_distribution(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "distribution"
            _write_distribution_output(output_dir)

            result = pipeline.validate_training_distribution(
                output_dir, "rotor_positive_8"
            )

            self.assertTrue(result["passed"], result["errors"])
            self.assertEqual(result["episode_count"], 100)
            self.assertEqual(result["success_count"], 80)
            self.assertAlmostEqual(result["success_rate"], 0.8)
            self.assertAlmostEqual(result["mean_body_frame_velocity_x"], 0.2)
            self.assertAlmostEqual(result["cost1_mean"], 10.0)

    def test_rejects_incomplete_distribution(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "distribution"
            _write_distribution_output(output_dir, episodes=99)

            result = pipeline.validate_training_distribution(
                output_dir, "rotor_positive_8"
            )

            self.assertFalse(result["passed"])
            self.assertIn("episode count", " ".join(result["errors"]))


class CommandConstructionTests(unittest.TestCase):
    def test_training_commands_lock_vx_and_preserve_optimizer(self):
        command = pipeline.build_train_command("ra8", python_executable="python")

        self.assertIn("--train_command_x=0.2", command)
        self.assertIn("--num_envs=8192", command)
        self.assertIn("--max_iterations=1000", command)
        self.assertIn("--reset_lagrange_on_resume", command)
        self.assertNotIn("--reset_optimizer_on_resume", command)
        self.assertIn("--energy_cost_mode=rotor_abs_8", command)
        self.assertIn("--cost_limit1=81.79582343673707", command)

    def test_fixed_and_distribution_commands_use_model_4001(self):
        fixed = pipeline.build_fixed_eval_command(
            "ja8", "/tmp/ja8", python_executable="python"
        )
        distribution = pipeline.build_train_dist_command(
            "ja8", "/tmp/ja8", python_executable="python"
        )

        self.assertIn("--checkpoint=4001", fixed)
        self.assertIn("--num_eval_episodes=20", fixed)
        self.assertIn("--make_eval_report", fixed)
        self.assertIn("--checkpoint=4001", distribution)
        self.assertIn("--calibration_episodes=100", distribution)
        self.assertIn("--num_envs=1024", distribution)


class SchedulingAndStateTests(unittest.TestCase):
    def test_gpu_selection_respects_memory_and_busy_set(self):
        selected = pipeline.select_available_gpu(
            (3, 4), {3: 12000, 4: 9000}, {3}, 8192
        )
        self.assertEqual(selected, 4)
        self.assertIsNone(
            pipeline.select_available_gpu(
                (3, 4), {3: 8000, 4: 8191}, set(), 8192
            )
        )

    def test_state_round_trip_supports_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.json"
            state = pipeline.new_state()
            state["phase"] = "training_controls"
            state["jobs"]["train_ra8"] = {
                "kind": "train",
                "alias": "ra8",
                "status": "completed",
            }

            pipeline.save_state(state, path)
            loaded = pipeline.load_state(path)

            self.assertEqual(loaded["phase"], "training_controls")
            self.assertEqual(
                loaded["jobs"]["train_ra8"]["status"], "completed"
            )

    def test_failed_job_blocks_only_after_active_sibling_finishes(self):
        state = pipeline.new_state()
        state["jobs"] = {
            "train_ra8": {
                "kind": "train",
                "alias": "ra8",
                "status": "failed",
                "reason": "OOM",
            },
            "train_ja8": {
                "kind": "train",
                "alias": "ja8",
                "status": "running",
                "reason": None,
            },
        }

        self.assertTrue(pipeline._block_after_active_jobs_finish(state, "train"))
        self.assertEqual(state["status"], "running")

        state["jobs"]["train_ja8"]["status"] = "completed"
        self.assertTrue(pipeline._block_after_active_jobs_finish(state, "train"))
        self.assertEqual(state["status"], "blocked")
        self.assertIn("ra8", state["reason"])

    def test_stopped_jobs_are_removed_for_safe_resume(self):
        state = pipeline.new_state()
        state["status"] = "stopped"
        state["run_dirs"]["ra8"] = "/tmp/partial-ra8"
        state["jobs"]["train_ra8"] = {
            "kind": "train",
            "alias": "ra8",
            "status": "stopped",
            "pid": 999999999,
        }

        resumed, reason = pipeline.prepare_stopped_state_for_resume(state)

        self.assertTrue(resumed, reason)
        self.assertEqual(state["status"], "running")
        self.assertNotIn("train_ra8", state["jobs"])
        self.assertNotIn("ra8", state["run_dirs"])

    def test_existing_failed_fixed_output_is_kept_in_summary_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "eval"
            _write_fixed_output(
                output_dir,
                mode="rotor_abs_8",
                world_vx=0.05,
                body_vx=0.05,
            )
            state = pipeline.new_state()

            pipeline._ensure_completed_output_job(
                state,
                "fixed_eval",
                "ra8",
                output_dir,
                pipeline.validate_fixed_evaluation,
            )

            self.assertEqual(
                state["jobs"]["fixed_eval_ra8"]["status"], "failed"
            )
            self.assertFalse(state["fixed_evaluations"]["ra8"]["passed"])
            self.assertAlmostEqual(
                state["fixed_evaluations"]["ra8"]["mean_velocity_x"], 0.05
            )

    def test_missing_initial_evaluation_blocks_after_start_timeout(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = pipeline.new_state()
            with mock.patch.object(pipeline, "FIXED_ROOT", Path(tmp)), mock.patch.object(
                pipeline, "_process_with_argument_exists", return_value=False
            ):
                self.assertFalse(
                    pipeline._refresh_initial_fixed(state, now_epoch=100.0)
                )
                self.assertEqual(state["status"], "running")

                self.assertFalse(
                    pipeline._refresh_initial_fixed(
                        state,
                        now_epoch=100.0 + pipeline.INITIAL_START_TIMEOUT_SECONDS,
                    )
                )

            self.assertEqual(state["status"], "blocked")
            self.assertIn("rp8", state["reason"])
            self.assertIn("did not start", state["reason"])


if __name__ == "__main__":
    unittest.main()
