import unittest

from bruce_gym.energy_reporting import rotor_energy_totals


class EnergyReportingTests(unittest.TestCase):
    def test_rotor_energy_totals_include_mixed_energy(self):
        totals = rotor_energy_totals(
            rotor_positive_energy=[1.0, 2.0, 3.0],
            rotor_negative_energy=[0.5, 1.5, 2.5],
        )

        self.assertAlmostEqual(totals["rotor_positive_energy_8"], 6.0)
        self.assertAlmostEqual(totals["rotor_negative_energy_8"], 4.5)
        self.assertAlmostEqual(totals["rotor_abs_energy_8"], 10.5)
        self.assertAlmostEqual(totals["rotor_mixed_energy_8"], 8.25)


if __name__ == "__main__":
    unittest.main()
