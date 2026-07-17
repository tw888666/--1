import csv
import json
import os
import tempfile
import unittest
from unittest import mock

from bruce_gym.eval_review import (
    _dynamic_state_rows,
    _evaluation_variant_description,
    _localized_gait_energy_rows,
    _phase_binned_series,
    _steady_state_rows,
    _write_csv_dicts,
    build_gait_stage_joint_energy,
    generate_review,
    select_representatives,
)


def _write_csv(path, rows):
    fieldnames = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with open(path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


class EvalReviewTests(unittest.TestCase):
    def test_evaluation_variant_description_is_chinese(self):
        self.assertEqual(
            _evaluation_variant_description(
                {"gait_phase_offset": 0.5, "mirrored_policy": True}
            ),
            "步态相位偏移：0.5周期｜策略：左右镜像策略",
        )

    def test_csv_writer_explicitly_uses_utf8_for_chinese_headers(self):
        mocked_open = mock.mock_open()
        with mock.patch("builtins.open", mocked_open):
            _write_csv_dicts("/tmp/table.csv", [{"步态阶段": "左腿支撑"}])

        mocked_open.assert_called_once_with(
            "/tmp/table.csv", "w", encoding="utf-8", newline=""
        )

    def test_gait_stage_joint_energy_is_normalized_per_cycle(self):
        rows = [
            {
                "time_s": str(index * 0.1),
                "gait_phase": str(index * 0.1),
                "hip_pitch_l_positive_energy": "2.0",
                "hip_pitch_l_negative_energy": "1.0",
                "pre_step_dynamic_state_valid": "1",
            }
            for index in range(10)
        ]

        table_rows, joint_names = build_gait_stage_joint_energy(
            rows,
            {"policy_dt": 0.1, "joint_names": ["hip_pitch_l"]},
        )

        self.assertEqual(joint_names, ["hip_pitch_l"])
        self.assertAlmostEqual(table_rows[0]["hip_pitch_l"], 3.0)
        self.assertAlmostEqual(
            sum(row["hip_pitch_l"] for row in table_rows),
            30.0,
        )
        localized_rows, localized_joint_fields = _localized_gait_energy_rows(
            table_rows, joint_names
        )
        self.assertEqual(localized_rows[0]["步态阶段"], "左脚落地与重心转移")
        self.assertEqual(localized_joint_fields, ["左髋俯仰"])

    def test_steady_phase_series_uses_middle_third_and_wraps_phase(self):
        rows = [
            {
                "time_s": str(time_s),
                "gait_phase": str(phase),
                "base_vel_x": str(value),
                "pre_step_dynamic_state_valid": "1",
            }
            for time_s, phase, value in [
                (0.0, 0.0, 10.0),
                (3.0, 0.25, 20.0),
                (4.0, 1.25, 30.0),
                (6.0, 0.75, 40.0),
                (9.0, 0.0, 50.0),
            ]
        ]

        steady_rows = _steady_state_rows(rows)
        _, values = _phase_binned_series(steady_rows, "base_vel_x", bins=4)

        self.assertEqual(steady_rows, rows[1:4])
        self.assertEqual(values[1], 25.0)
        self.assertEqual(values[3], 40.0)

    def test_dynamic_state_rows_excludes_stale_reset_boundary_sample(self):
        rows = [
            {"episode_step": 0, "pre_step_dynamic_state_valid": "0", "base_vel_x": "0.2"},
            {"episode_step": 1, "pre_step_dynamic_state_valid": "1", "base_vel_x": "0.0"},
        ]

        self.assertEqual(_dynamic_state_rows(rows), [rows[1]])

    def test_select_representatives_prefers_successful_metric_and_falls_for_worst(self):
        rows = [
            {"episode_id": 0, "success": 1.0, "fall": 0.0, "e_mix_per_m": 3.0},
            {"episode_id": 1, "success": 1.0, "fall": 0.0, "e_mix_per_m": 1.0},
            {"episode_id": 2, "success": 1.0, "fall": 0.0, "e_mix_per_m": 2.0},
            {"episode_id": 3, "success": 0.0, "fall": 1.0, "e_mix_per_m": 0.5},
        ]

        reps = select_representatives(rows)
        reps_by_label = {rep["label"]: rep for rep in reps}

        self.assertEqual(reps_by_label["best"]["episode_id"], 1)
        self.assertEqual(reps_by_label["median"]["episode_id"], 2)
        self.assertEqual(reps_by_label["worst"]["episode_id"], 3)

    def test_generate_review_writes_summary_representatives_and_episode_csvs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            metadata = {
                "task": "bruce_ppolag",
                "load_run": "run_a",
                "checkpoint": 3000,
                "command_x": 0.1,
                "policy_dt": 0.01,
                "joint_names": ["hip_yaw_l", "knee_pitch_l"],
            }
            with open(os.path.join(tmpdir, "metadata.json"), "w") as jsonfile:
                json.dump(metadata, jsonfile)

            _write_csv(
                os.path.join(tmpdir, "episode_summary.csv"),
                [
                    {
                        "episode_id": 0,
                        "episode_outcome": "success",
                        "timeout": 1,
                        "fall": 0,
                        "duration_s": 2.0,
                        "distance_x": 1.0,
                        "mean_velocity_x": 0.5,
                        "rotor_positive_energy_8": 4.0,
                        "rotor_negative_energy_8": 2.0,
                        "rotor_mixed_energy_8": 5.0,
                        "reducer_corrected_energy_8": 4.0,
                        "joint_positive_energy_10": 6.0,
                        "joint_negative_energy_10": 3.0,
                        "hip_yaw_l_positive_energy": 1.0,
                        "hip_yaw_l_negative_energy": 0.5,
                        "knee_pitch_l_positive_energy": 2.0,
                        "knee_pitch_l_negative_energy": 0.25,
                    },
                    {
                        "episode_id": 1,
                        "episode_outcome": "success",
                        "timeout": 1,
                        "fall": 0,
                        "duration_s": 2.0,
                        "distance_x": 2.0,
                        "mean_velocity_x": 1.0,
                        "rotor_positive_energy_8": 6.0,
                        "rotor_negative_energy_8": 2.0,
                        "rotor_mixed_energy_8": 7.0,
                        "reducer_corrected_energy_8": 6.0,
                        "joint_positive_energy_10": 7.0,
                        "joint_negative_energy_10": 3.0,
                        "hip_yaw_l_positive_energy": 1.0,
                        "hip_yaw_l_negative_energy": 0.5,
                        "knee_pitch_l_positive_energy": 2.0,
                        "knee_pitch_l_negative_energy": 0.25,
                    },
                ],
            )
            _write_csv(
                os.path.join(tmpdir, "step_timeseries.csv"),
                [
                    {
                        "episode_id": 0,
                        "episode_step": 0,
                        "command_x": 0.1,
                        "base_vel_x": 0.1,
                        "post_step_root_pos_z": 0.45,
                        "left_contact_state": 1,
                        "right_contact_state": 0,
                        "rotor_positive_energy_8": 1.0,
                        "rotor_negative_energy_8": 0.5,
                        "rotor_mixed_energy_8": 1.25,
                        "left_hip_motor_0_power": 2.0,
                        "hip_yaw_l_power": 1.0,
                    },
                    {
                        "episode_id": 1,
                        "episode_step": 0,
                        "command_x": 0.1,
                        "base_vel_x": 0.2,
                        "post_step_root_pos_z": 0.46,
                        "left_contact_state": 1,
                        "right_contact_state": 1,
                        "rotor_positive_energy_8": 1.0,
                        "rotor_negative_energy_8": 0.25,
                        "rotor_mixed_energy_8": 1.125,
                        "left_hip_motor_0_power": 3.0,
                        "hip_yaw_l_power": 1.5,
                    },
                ],
            )

            result = generate_review(tmpdir, make_plots=False)

            self.assertTrue(os.path.exists(result["summary_csv"]))
            self.assertTrue(os.path.exists(result["report_md"]))
            self.assertTrue(
                os.path.exists(os.path.join(result["report_dir"], "episodes", "episode_0000.csv"))
            )
            with open(result["summary_csv"], newline="") as csvfile:
                rows = list(csv.DictReader(csvfile))
            self.assertAlmostEqual(float(rows[0]["e_mix_per_m"]), 5.0)
            self.assertAlmostEqual(float(rows[1]["e_mix_per_m"]), 3.5)
            self.assertAlmostEqual(float(rows[0]["e_reducer_corrected_per_m"]), 4.0)
            self.assertAlmostEqual(float(rows[1]["e_reducer_corrected_per_m"]), 3.0)


if __name__ == "__main__":
    unittest.main()
