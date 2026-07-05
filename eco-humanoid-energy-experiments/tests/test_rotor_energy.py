import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import torch

from bruce_gym.rotor_energy import (
    BRUCE_DRIVE_JOINT_INDICES,
    BRUCE_EXPECTED_DOF_NAMES,
    BRUCE_LEFT_FIRST_DOF_NAMES,
    BRUCE_RIGHT_FIRST_DOF_NAMES,
    JOINT_ABS_8,
    JOINT_POSITIVE_8,
    LEGACY_JOINT_ABS_10,
    ROTOR_ABS_8,
    ROTOR_POSITIVE_8,
    SUPPORTED_ENERGY_COST_MODES,
    assert_bruce_dof_order,
    bruce_joint_names_from_dof_names,
    bruce_rotor_names_from_dof_names,
    compute_bruce_energy_terms,
    compute_bruce_rotor_power,
    energy_cost_from_terms,
    make_bruce_transmission_tensors,
)


class BruceRotorEnergyTest(unittest.TestCase):
    def setUp(self):
        self.dtype = torch.float64
        self.transmission = make_bruce_transmission_tensors(
            torch.device("cpu"), self.dtype, gear_ratio=9.0
        )

    def test_transmission_inverse_maps_velocity_back_to_joint_space(self):
        torques = torch.zeros(2, 10, dtype=self.dtype)
        velocities = torch.zeros(2, 10, dtype=self.dtype)
        velocities[:, 1:5] = torch.tensor(
            [[0.1, -0.2, 0.3, -0.4], [-0.7, 0.5, 0.2, -0.1]],
            dtype=self.dtype,
        )
        result = compute_bruce_rotor_power(torques, velocities, self.transmission)

        hip_velocity = result["output_velocity"][:, 0:2]
        lower_velocity = result["output_velocity"][:, 2:4]
        recovered_hip_qdot = hip_velocity @ self.transmission["hip_j"].T
        recovered_lower_qdot = lower_velocity @ self.transmission["lower_j"].T

        self.assertTrue(torch.allclose(recovered_hip_qdot, velocities[:, 1:3]))
        self.assertTrue(torch.allclose(recovered_lower_qdot, velocities[:, 3:5]))

    def test_rotor_power_conserves_drive_joint_power(self):
        generator = torch.Generator().manual_seed(7)
        torques = torch.randn(32, 10, dtype=self.dtype, generator=generator)
        velocities = torch.randn(32, 10, dtype=self.dtype, generator=generator)

        terms = compute_bruce_energy_terms(
            torques, velocities, sim_dt=0.001, transmission=self.transmission
        )

        joint_power = (
            torques[:, BRUCE_DRIVE_JOINT_INDICES]
            * velocities[:, BRUCE_DRIVE_JOINT_INDICES]
        ).sum(dim=-1)
        rotor_power = terms["rotor_power"].sum(dim=-1)
        rel_error = torch.abs(joint_power - rotor_power) / torch.clamp(
            torch.abs(joint_power), min=1e-8
        )

        self.assertLess(torch.max(rel_error).item(), 1e-6)

    def test_gear_ratio_does_not_change_ideal_mechanical_power(self):
        torques = torch.tensor([[0.0, 1.0, -2.0, 3.0, -4.0, 0.0, 2.0, 1.0, -3.0, 5.0]], dtype=self.dtype)
        velocities = torch.tensor([[0.0, -0.5, 0.25, 1.5, -1.0, 0.0, 0.75, -0.25, 2.0, -1.25]], dtype=self.dtype)

        output_terms = compute_bruce_rotor_power(torques, velocities, self.transmission)
        output_power = (
            output_terms["output_torque"] * output_terms["output_velocity"]
        ).sum(dim=-1)
        rotor_power = output_terms["rotor_power"].sum(dim=-1)

        self.assertTrue(torch.allclose(output_power, rotor_power))

    def test_positive_and_negative_energy_are_integrated_separately(self):
        torques = torch.tensor([[0.0, 1.0, -2.0, 3.0, -4.0, 0.0, 2.0, 1.0, -3.0, 5.0]], dtype=self.dtype)
        velocities = torch.tensor([[0.0, -0.5, 0.25, 1.5, -1.0, 0.0, 0.75, -0.25, 2.0, -1.25]], dtype=self.dtype)
        dt = 0.001
        substeps = 10

        terms = compute_bruce_energy_terms(
            torques, velocities, sim_dt=dt, transmission=self.transmission
        )
        rotor_power = terms["rotor_power"]
        expected_drive = torch.clamp(rotor_power, min=0.0).sum(dim=-1) * dt * substeps
        expected_brake = torch.clamp(-rotor_power, min=0.0).sum(dim=-1) * dt * substeps

        drive = terms["rotor_drive_energy"] * substeps
        brake = terms["rotor_brake_energy"] * substeps

        self.assertTrue(torch.allclose(drive, expected_drive))
        self.assertTrue(torch.allclose(brake, expected_brake))
        self.assertTrue(torch.all(drive >= 0.0))
        self.assertTrue(torch.all(brake >= 0.0))

    def test_supported_cost_modes(self):
        torques = torch.tensor([[1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0, 9.0, -10.0]], dtype=self.dtype)
        velocities = torch.tensor([[0.5, 0.25, -0.5, 0.75, -1.0, 1.25, -1.5, 1.75, -2.0, 2.25]], dtype=self.dtype)
        terms = compute_bruce_energy_terms(
            torques, velocities, sim_dt=0.001, transmission=self.transmission
        )

        legacy = torch.abs(torques * velocities).sum(dim=-1)
        joint_power = torques[:, BRUCE_DRIVE_JOINT_INDICES] * velocities[:, BRUCE_DRIVE_JOINT_INDICES]
        joint_abs = torch.abs(joint_power).sum(dim=-1) * 0.001
        joint_positive = torch.clamp(joint_power, min=0.0).sum(dim=-1) * 0.001
        rotor_abs = terms["rotor_drive_energy"] + terms["rotor_brake_energy"]

        self.assertTrue(torch.allclose(energy_cost_from_terms(terms, LEGACY_JOINT_ABS_10), legacy))
        self.assertTrue(torch.allclose(energy_cost_from_terms(terms, JOINT_ABS_8), joint_abs))
        self.assertTrue(torch.allclose(energy_cost_from_terms(terms, JOINT_POSITIVE_8), joint_positive))
        self.assertTrue(torch.allclose(energy_cost_from_terms(terms, ROTOR_ABS_8), rotor_abs))
        self.assertTrue(torch.allclose(energy_cost_from_terms(terms, ROTOR_POSITIVE_8), terms["rotor_drive_energy"]))

    def test_rotor_mixed_cost_penalizes_negative_power_with_half_weight(self):
        torques = torch.tensor([[1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0, 9.0, -10.0]], dtype=self.dtype)
        velocities = torch.tensor([[0.5, 0.25, -0.5, 0.75, -1.0, 1.25, -1.5, 1.75, -2.0, 2.25]], dtype=self.dtype)
        terms = compute_bruce_energy_terms(
            torques, velocities, sim_dt=0.001, transmission=self.transmission
        )

        expected = terms["rotor_drive_energy"] + 0.5 * terms["rotor_brake_energy"]

        self.assertIn("rotor_mixed_8", SUPPORTED_ENERGY_COST_MODES)
        self.assertIn("rotor_mixed_8_alpha050", SUPPORTED_ENERGY_COST_MODES)
        self.assertTrue(torch.allclose(energy_cost_from_terms(terms, "rotor_mixed_8"), expected))
        self.assertTrue(torch.allclose(energy_cost_from_terms(terms, "rotor_mixed_8_alpha050"), expected))

    def test_all_ten_joint_energies_are_reported(self):
        torques = torch.tensor(
            [[1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0, 9.0, -10.0]],
            dtype=self.dtype,
        )
        velocities = torch.tensor(
            [[0.5, 0.25, -0.5, 0.75, -1.0, 1.25, -1.5, 1.75, -2.0, 2.25]],
            dtype=self.dtype,
        )
        dt = 0.001
        terms = compute_bruce_energy_terms(
            torques, velocities, sim_dt=dt, transmission=self.transmission
        )
        joint_power = torques * velocities

        self.assertEqual(terms["joint_drive_energy_per_joint"].shape[-1], 10)
        self.assertEqual(terms["joint_brake_energy_per_joint"].shape[-1], 10)
        self.assertEqual(terms["drive_joint_drive_energy_per_joint"].shape[-1], 8)
        self.assertTrue(
            torch.allclose(
                terms["joint_drive_energy_per_joint"],
                torch.clamp(joint_power, min=0.0) * dt,
            )
        )
        self.assertTrue(
            torch.allclose(
                terms["joint_brake_energy_per_joint"],
                torch.clamp(-joint_power, min=0.0) * dt,
            )
        )

    def test_joint_positive_negative_energy_reconstructs_power_integral(self):
        torques = torch.tensor([[1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0, 9.0, -10.0]], dtype=self.dtype)
        velocities = torch.tensor([[0.5, 0.25, -0.5, 0.75, -1.0, 1.25, -1.5, 1.75, -2.0, 2.25]], dtype=self.dtype)
        dt = 0.001
        terms = compute_bruce_energy_terms(
            torques, velocities, sim_dt=dt, transmission=self.transmission
        )
        signed_integral = (torques * velocities).sum(dim=-1) * dt
        abs_integral = torch.abs(torques * velocities).sum(dim=-1) * dt

        self.assertTrue(
            torch.allclose(
                terms["joint_drive_energy"] + terms["joint_brake_energy"],
                abs_integral,
            )
        )
        self.assertTrue(
            torch.allclose(
                terms["joint_drive_energy"] - terms["joint_brake_energy"],
                signed_integral,
            )
        )

    def test_bruce_dof_order_assertion(self):
        assert_bruce_dof_order(BRUCE_RIGHT_FIRST_DOF_NAMES)
        assert_bruce_dof_order(BRUCE_LEFT_FIRST_DOF_NAMES)
        with self.assertRaises(ValueError):
            assert_bruce_dof_order(
                (
                    "hip_yaw_r",
                    "hip_pitch_r",
                    "hip_roll_r",
                    "knee_pitch_r",
                    "ankle_pitch_r",
                    "hip_pitch_l",
                    "hip_yaw_l",
                    "hip_roll_l",
                    "knee_pitch_l",
                    "ankle_pitch_l",
                )
            )

    def test_runtime_names_follow_dof_order(self):
        self.assertEqual(
            bruce_joint_names_from_dof_names(BRUCE_LEFT_FIRST_DOF_NAMES),
            BRUCE_LEFT_FIRST_DOF_NAMES,
        )
        self.assertEqual(
            bruce_rotor_names_from_dof_names(BRUCE_LEFT_FIRST_DOF_NAMES),
            (
                "left_hip_motor_0",
                "left_hip_motor_1",
                "left_lower_motor_0",
                "left_lower_motor_1",
                "right_hip_motor_0",
                "right_hip_motor_1",
                "right_lower_motor_0",
                "right_lower_motor_1",
            ),
        )
        self.assertEqual(
            bruce_rotor_names_from_dof_names(BRUCE_RIGHT_FIRST_DOF_NAMES),
            (
                "right_hip_motor_0",
                "right_hip_motor_1",
                "right_lower_motor_0",
                "right_lower_motor_1",
                "left_hip_motor_0",
                "left_hip_motor_1",
                "left_lower_motor_0",
                "left_lower_motor_1",
            ),
        )

    def test_expected_dof_order_matches_bruce_urdf(self):
        repo_root = Path(__file__).resolve().parents[1]
        urdf_path = repo_root / "resources" / "robots" / "bruce" / "bruce.urdf"
        root = ET.parse(urdf_path).getroot()
        movable_joint_names = [
            joint.attrib["name"]
            for joint in root.findall("joint")
            if joint.attrib.get("type") != "fixed"
        ]

        self.assertEqual(tuple(movable_joint_names[:10]), BRUCE_EXPECTED_DOF_NAMES)


if __name__ == "__main__":
    unittest.main()
