# SPDX-License-Identifier: BSD-3-Clause

import math
import unittest

from bruce_gym.cost_calibration import (
    evenly_spaced_indices,
    summarize_episode_costs,
)


class CostCalibrationSummaryTests(unittest.TestCase):
    def test_summarizes_all_and_success_episodes(self):
        rows = [
            {"cost1": 10.0, "episode_outcome": "success"},
            {"cost1": 50.0, "episode_outcome": "fall"},
            {"cost1": 30.0, "episode_outcome": "success"},
        ]

        summary = summarize_episode_costs(rows, limit_fraction=0.95)

        self.assertEqual(summary["episode_count"], 3)
        self.assertEqual(summary["success_count"], 2)
        self.assertEqual(summary["fall_count"], 1)
        self.assertAlmostEqual(summary["success_rate"], 2.0 / 3.0)
        self.assertAlmostEqual(
            summary["cost1_all_episodes"]["mean"], 30.0
        )
        self.assertAlmostEqual(
            summary["cost1_all_episodes"]["median"], 30.0
        )
        self.assertAlmostEqual(
            summary["cost1_all_episodes"]["population_std"],
            math.sqrt(800.0 / 3.0),
        )
        self.assertAlmostEqual(
            summary["cost1_success_episodes"]["mean"], 20.0
        )
        self.assertAlmostEqual(
            summary["recommended_cost_limit1_all_episodes"], 28.5
        )
        self.assertAlmostEqual(
            summary["recommended_cost_limit1_success_episodes"], 19.0
        )
        self.assertEqual(
            summary["recommendation_status"],
            "review_required_falls_present",
        )

    def test_supports_no_successful_episodes(self):
        summary = summarize_episode_costs(
            [{"cost1": 5.0, "episode_outcome": "fall"}]
        )

        self.assertEqual(summary["success_count"], 0)
        self.assertIsNone(summary["cost1_success_episodes"]["mean"])
        self.assertAlmostEqual(
            summary["provisional_cost_limit1_all_episodes"], 4.75
        )
        self.assertIsNone(summary["recommended_cost_limit1_all_episodes"])
        self.assertIsNone(
            summary["recommended_cost_limit1_success_episodes"]
        )
        self.assertEqual(
            summary["recommendation_status"],
            "invalid_no_successful_episodes",
        )

    def test_rejects_empty_episode_data(self):
        with self.assertRaisesRegex(ValueError, "At least one complete episode"):
            summarize_episode_costs([])

    def test_rejects_invalid_limit_fraction(self):
        for fraction in (0.0, -0.1, 1.1):
            with self.subTest(fraction=fraction):
                with self.assertRaisesRegex(ValueError, "limit_fraction"):
                    summarize_episode_costs(
                        [{"cost1": 1.0, "episode_outcome": "success"}],
                        limit_fraction=fraction,
                    )

    def test_non_positive_mean_cost_has_no_recommended_limit(self):
        summary = summarize_episode_costs(
            [
                {"cost1": -2.0, "episode_outcome": "success"},
                {"cost1": 1.0, "episode_outcome": "success"},
            ]
        )

        self.assertEqual(
            summary["recommendation_status"],
            "invalid_non_positive_mean_cost",
        )
        self.assertIsNone(summary["recommended_cost_limit1_all_episodes"])
        self.assertIsNone(summary["recommended_cost_limit1_success_episodes"])

    def test_evenly_spaced_indices_cover_full_environment_range(self):
        indices = evenly_spaced_indices(1024, 100)

        self.assertEqual(len(indices), 100)
        self.assertEqual(len(set(indices)), 100)
        self.assertEqual(indices[0], 0)
        self.assertEqual(indices[-1], 1023)

    def test_evenly_spaced_indices_reject_invalid_sample_size(self):
        with self.assertRaisesRegex(ValueError, "sample_count"):
            evenly_spaced_indices(10, 11)


if __name__ == "__main__":
    unittest.main()
