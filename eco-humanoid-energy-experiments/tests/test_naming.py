import unittest

from bruce_gym.naming import (
    energy_cost_alias,
    policy_id,
    reducer_efficiency_suffix,
    velocity_token,
)


class NamingTests(unittest.TestCase):
    def test_velocity_token_uses_hundredths(self):
        self.assertEqual(velocity_token(0.1), "vx010")
        self.assertEqual(velocity_token(0.2), "vx020")

    def test_cost_aliases_are_short_and_stable(self):
        self.assertEqual(energy_cost_alias("rotor_positive_8"), "rp8")
        self.assertEqual(energy_cost_alias("rotor_abs_8"), "ra8")
        self.assertEqual(energy_cost_alias("joint_positive_8"), "jp8")
        self.assertEqual(energy_cost_alias("joint_abs_8"), "ja8")
        self.assertEqual(energy_cost_alias("legacy_joint_abs_10"), "lja10")
        self.assertEqual(energy_cost_alias("rotor_mixed_8_alpha050"), "rm8a05")
        self.assertEqual(energy_cost_alias("reducer_corrected_8"), "rc8")

    def test_reducer_efficiency_suffix_is_explicit_for_shared_values(self):
        self.assertEqual(
            reducer_efficiency_suffix(0.9, 0.5),
            "etam090_etag050",
        )

    def test_policy_id_matches_directory_rule(self):
        self.assertEqual(
            policy_id(0.2, "rotor_positive_8", 0, 4001),
            "vx020_rp8_s00_model4001",
        )


if __name__ == "__main__":
    unittest.main()
