# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2026 ECO Authors. All rights reserved.

import math
from typing import Dict, Optional

import torch


BRUCE_DRIVE_JOINT_INDICES = (1, 2, 3, 4, 6, 7, 8, 9)
BRUCE_YAW_JOINT_INDICES = (0, 5)
BRUCE_RIGHT_FIRST_DOF_NAMES = (
    "hip_yaw_r",
    "hip_pitch_r",
    "hip_roll_r",
    "knee_pitch_r",
    "ankle_pitch_r",
    "hip_yaw_l",
    "hip_pitch_l",
    "hip_roll_l",
    "knee_pitch_l",
    "ankle_pitch_l",
)
BRUCE_LEFT_FIRST_DOF_NAMES = (
    "hip_yaw_l",
    "hip_pitch_l",
    "hip_roll_l",
    "knee_pitch_l",
    "ankle_pitch_l",
    "hip_yaw_r",
    "hip_pitch_r",
    "hip_roll_r",
    "knee_pitch_r",
    "ankle_pitch_r",
)
BRUCE_SUPPORTED_DOF_ORDERS = (BRUCE_RIGHT_FIRST_DOF_NAMES, BRUCE_LEFT_FIRST_DOF_NAMES)
BRUCE_EXPECTED_DOF_NAMES = BRUCE_RIGHT_FIRST_DOF_NAMES


def _rotor_names_from_side_blocks(side_blocks):
    names = []
    for side in side_blocks:
        names.extend(
            (
                f"{side}_hip_motor_0",
                f"{side}_hip_motor_1",
                f"{side}_lower_motor_0",
                f"{side}_lower_motor_1",
            )
        )
    return tuple(names)


BRUCE_ROTOR_NAMES = _rotor_names_from_side_blocks(("right", "left"))

LEGACY_JOINT_ABS_10 = "legacy_joint_abs_10"
JOINT_ABS_8 = "joint_abs_8"
JOINT_POSITIVE_8 = "joint_positive_8"
ROTOR_ABS_8 = "rotor_abs_8"
ROTOR_POSITIVE_8 = "rotor_positive_8"
ROTOR_MIXED_8 = "rotor_mixed_8"
ROTOR_MIXED_8_ALPHA050 = "rotor_mixed_8_alpha050"
REDUCER_CORRECTED_8 = "reducer_corrected_8"
ROTOR_MIXED_BRAKE_ALPHA = 0.5
REDUCER_POSITIVE_EFFICIENCY_SCALE = 0.905
REDUCER_POSITIVE_EFFICIENCY_OFFSET = 0.2735

SUPPORTED_ENERGY_COST_MODES = (
    LEGACY_JOINT_ABS_10,
    JOINT_ABS_8,
    JOINT_POSITIVE_8,
    ROTOR_ABS_8,
    ROTOR_POSITIVE_8,
    ROTOR_MIXED_8,
    ROTOR_MIXED_8_ALPHA050,
    REDUCER_CORRECTED_8,
)

CONTROL_ENERGY_COST_MODES = (
    ROTOR_POSITIVE_8,
    ROTOR_ABS_8,
    JOINT_POSITIVE_8,
    JOINT_ABS_8,
    LEGACY_JOINT_ABS_10,
)


def compute_rotor_mixed_energy(rotor_positive_energy, rotor_negative_energy):
    return rotor_positive_energy + ROTOR_MIXED_BRAKE_ALPHA * rotor_negative_energy


def parse_reducer_rated_torque_spec(value):
    """Parse one shared rated torque or eight per-motor values."""

    if value is None:
        return None
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
        if not parts:
            raise ValueError("Reducer rated torque specification must not be empty.")
        values = [float(part) for part in parts]
    elif isinstance(value, (int, float)):
        values = [float(value)]
    else:
        values = [float(item) for item in value]

    if len(values) not in (1, len(BRUCE_ROTOR_NAMES)):
        raise ValueError(
            "Reducer rated torque must contain one shared value or "
            f"{len(BRUCE_ROTOR_NAMES)} per-motor values; got {len(values)}."
        )
    if any(not math.isfinite(item) or item <= 0.0 for item in values):
        raise ValueError(
            "Reducer rated torque values must be finite and positive."
        )
    return values[0] if len(values) == 1 else tuple(values)


def make_reducer_rated_torque_tensor(
    values,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build and validate positive output-side rated torques for eight motors."""

    values = parse_reducer_rated_torque_spec(values)
    if values is None:
        raise ValueError(
            f"reducer_rated_torque is required when "
            f"energy_cost_mode={REDUCER_CORRECTED_8}."
        )
    tensor = torch.as_tensor(values, device=device, dtype=dtype)
    if tensor.ndim == 0:
        tensor = tensor.repeat(len(BRUCE_ROTOR_NAMES))
    if tuple(tensor.shape) != (len(BRUCE_ROTOR_NAMES),):
        raise ValueError(
            "reducer_rated_torque must be scalar or have shape "
            f"({len(BRUCE_ROTOR_NAMES)},), got {tuple(tensor.shape)}."
        )
    if not torch.isfinite(tensor).all().item() or torch.any(tensor <= 0.0).item():
        raise ValueError("reducer_rated_torque values must be finite and positive.")
    return tensor


def reducer_positive_efficiency(rated_torque_ratio: torch.Tensor) -> torch.Tensor:
    """Return eta(x)=0.905*x/(x+0.2735), x=abs(T)/T_N."""

    return (
        REDUCER_POSITIVE_EFFICIENCY_SCALE
        * rated_torque_ratio
        / (rated_torque_ratio + REDUCER_POSITIVE_EFFICIENCY_OFFSET)
    )


def compute_reducer_corrected_energy_per_motor(
    rotor_power: torch.Tensor,
    output_torque: torch.Tensor,
    rated_torque: torch.Tensor,
    sim_dt: float,
) -> torch.Tensor:
    """Compute P+ / eta(abs(T)/T_N) + P- for one physics substep."""

    positive_power = torch.clamp(rotor_power, min=0.0)
    negative_power_magnitude = torch.clamp(-rotor_power, min=0.0)
    rated_torque_ratio = torch.abs(output_torque) / rated_torque
    drive_mask = positive_power > 0.0
    safe_ratio = torch.where(
        drive_mask, rated_torque_ratio, torch.ones_like(rated_torque_ratio)
    )
    # Use the algebraically equivalent form only on the positive-drive mask to
    # avoid evaluating 0/0 when both torque and positive power are zero.
    positive_input_power = torch.where(
        drive_mask,
        positive_power
        * (safe_ratio + REDUCER_POSITIVE_EFFICIENCY_OFFSET)
        / (REDUCER_POSITIVE_EFFICIENCY_SCALE * safe_ratio),
        torch.zeros_like(positive_power),
    )
    dt = torch.as_tensor(sim_dt, device=rotor_power.device, dtype=rotor_power.dtype)
    return (positive_input_power + negative_power_magnitude) * dt


def compute_reducer_corrected_energy(
    rotor_power: torch.Tensor,
    output_torque: torch.Tensor,
    rated_torque: torch.Tensor,
    sim_dt: float,
) -> torch.Tensor:
    """Sum the nonlinear positive-power and absolute-negative-power cost."""

    return compute_reducer_corrected_energy_per_motor(
        rotor_power, output_torque, rated_torque, sim_dt
    ).sum(dim=-1)


def make_bruce_transmission_tensors(
    device: torch.device,
    dtype: torch.dtype,
    gear_ratio: float = 9.0,
) -> Dict[str, torch.Tensor]:
    hip_j = torch.tensor(
        [[-0.5, -0.5], [0.5, -0.5]], device=device, dtype=dtype
    )
    lower_j = torch.tensor(
        [[-1.0, 0.0], [1.0, 1.0]], device=device, dtype=dtype
    )
    return {
        "hip_j": hip_j,
        "hip_j_inv": torch.linalg.inv(hip_j),
        "lower_j": lower_j,
        "lower_j_inv": torch.linalg.inv(lower_j),
        "gear_ratio": torch.tensor(gear_ratio, device=device, dtype=dtype),
    }


def _select_last_dim(values: torch.Tensor, indices) -> torch.Tensor:
    index = torch.tensor(indices, device=values.device, dtype=torch.long)
    return values.index_select(-1, index)


def bruce_dof_order(dof_names):
    actual = tuple(dof_names[: len(BRUCE_EXPECTED_DOF_NAMES)])
    if actual in BRUCE_SUPPORTED_DOF_ORDERS:
        return actual
    expected_text = " or ".join(
        f"[{', '.join(order)}]" for order in BRUCE_SUPPORTED_DOF_ORDERS
    )
    actual_text = ", ".join(actual)
    raise ValueError(
        "Unexpected BRUCE DOF order for rotor energy mapping. "
        f"Expected {expected_text}, got [{actual_text}]."
    )


def assert_bruce_dof_order(dof_names) -> None:
    bruce_dof_order(dof_names)


def bruce_joint_names_from_dof_names(dof_names):
    return bruce_dof_order(dof_names)


def bruce_rotor_names_from_dof_names(dof_names):
    order = bruce_dof_order(dof_names)
    side_blocks = []
    for block_start in (0, 5):
        side_suffix = order[block_start].rsplit("_", 1)[-1]
        side_blocks.append("left" if side_suffix == "l" else "right")
    return _rotor_names_from_side_blocks(side_blocks)


def _map_pair(
    joint_torque: torch.Tensor,
    joint_velocity: torch.Tensor,
    transmission_j: torch.Tensor,
    transmission_j_inv: torch.Tensor,
    gear_ratio: torch.Tensor,
):
    output_velocity = joint_velocity @ transmission_j_inv.T
    output_torque = joint_torque @ transmission_j
    rotor_velocity = output_velocity * gear_ratio
    rotor_torque = output_torque / gear_ratio
    rotor_power = rotor_torque * rotor_velocity
    return output_torque, output_velocity, rotor_torque, rotor_velocity, rotor_power


def compute_bruce_rotor_power(
    joint_torques: torch.Tensor,
    joint_velocities: torch.Tensor,
    transmission: Optional[Dict[str, torch.Tensor]] = None,
    gear_ratio: float = 9.0,
) -> Dict[str, torch.Tensor]:
    if joint_torques.shape[-1] < 10 or joint_velocities.shape[-1] < 10:
        raise ValueError("BRUCE rotor mapping requires at least 10 leg joints.")

    if transmission is None:
        transmission = make_bruce_transmission_tensors(
            joint_torques.device, joint_torques.dtype, gear_ratio
        )

    pairs = (
        (1, 2, transmission["hip_j"], transmission["hip_j_inv"]),
        (3, 4, transmission["lower_j"], transmission["lower_j_inv"]),
        (6, 7, transmission["hip_j"], transmission["hip_j_inv"]),
        (8, 9, transmission["lower_j"], transmission["lower_j_inv"]),
    )

    output_torques = []
    output_velocities = []
    rotor_torques = []
    rotor_velocities = []
    rotor_powers = []
    gear = transmission["gear_ratio"]

    for idx0, idx1, matrix, matrix_inv in pairs:
        pair_torque = joint_torques[..., (idx0, idx1)]
        pair_velocity = joint_velocities[..., (idx0, idx1)]
        mapped = _map_pair(pair_torque, pair_velocity, matrix, matrix_inv, gear)
        output_torques.append(mapped[0])
        output_velocities.append(mapped[1])
        rotor_torques.append(mapped[2])
        rotor_velocities.append(mapped[3])
        rotor_powers.append(mapped[4])

    return {
        "output_torque": torch.cat(output_torques, dim=-1),
        "output_velocity": torch.cat(output_velocities, dim=-1),
        "rotor_torque": torch.cat(rotor_torques, dim=-1),
        "rotor_velocity": torch.cat(rotor_velocities, dim=-1),
        "rotor_power": torch.cat(rotor_powers, dim=-1),
    }


def compute_bruce_energy_terms(
    joint_torques: torch.Tensor,
    joint_velocities: torch.Tensor,
    sim_dt: float,
    transmission: Optional[Dict[str, torch.Tensor]] = None,
    gear_ratio: float = 9.0,
    reducer_rated_torque: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    rotor = compute_bruce_rotor_power(
        joint_torques, joint_velocities, transmission, gear_ratio
    )

    joint_power_all = joint_torques[..., :10] * joint_velocities[..., :10]
    drive_joint_power = (
        _select_last_dim(joint_torques, BRUCE_DRIVE_JOINT_INDICES)
        * _select_last_dim(joint_velocities, BRUCE_DRIVE_JOINT_INDICES)
    )
    yaw_power = (
        _select_last_dim(joint_torques, BRUCE_YAW_JOINT_INDICES)
        * _select_last_dim(joint_velocities, BRUCE_YAW_JOINT_INDICES)
    )

    dt = torch.as_tensor(sim_dt, device=joint_torques.device, dtype=joint_torques.dtype)
    rotor_power = rotor["rotor_power"]
    rotor_drive_energy_per_motor = torch.clamp(rotor_power, min=0.0) * dt
    rotor_brake_energy_per_motor = torch.clamp(-rotor_power, min=0.0) * dt
    joint_drive_energy_per_joint = torch.clamp(joint_power_all, min=0.0) * dt
    joint_brake_energy_per_joint = torch.clamp(-joint_power_all, min=0.0) * dt
    drive_joint_drive_energy_per_joint = torch.clamp(drive_joint_power, min=0.0) * dt
    drive_joint_brake_energy_per_joint = torch.clamp(-drive_joint_power, min=0.0) * dt

    terms = {
        **rotor,
        "joint_power_all": joint_power_all,
        "drive_joint_power": drive_joint_power,
        "yaw_power": yaw_power,
        "rotor_drive_energy_per_motor": rotor_drive_energy_per_motor,
        "rotor_brake_energy_per_motor": rotor_brake_energy_per_motor,
        "rotor_drive_energy": rotor_drive_energy_per_motor.sum(dim=-1),
        "rotor_brake_energy": rotor_brake_energy_per_motor.sum(dim=-1),
        "joint_drive_energy_per_joint": joint_drive_energy_per_joint,
        "joint_brake_energy_per_joint": joint_brake_energy_per_joint,
        "joint_drive_energy": joint_drive_energy_per_joint.sum(dim=-1),
        "joint_brake_energy": joint_brake_energy_per_joint.sum(dim=-1),
        "drive_joint_drive_energy_per_joint": drive_joint_drive_energy_per_joint,
        "drive_joint_brake_energy_per_joint": drive_joint_brake_energy_per_joint,
        "drive_joint_drive_energy": drive_joint_drive_energy_per_joint.sum(dim=-1),
        "drive_joint_brake_energy": drive_joint_brake_energy_per_joint.sum(dim=-1),
        "legacy_joint_abs_10_cost": torch.abs(joint_power_all).sum(dim=-1),
    }
    if reducer_rated_torque is not None:
        reducer_corrected_energy_per_motor = (
            compute_reducer_corrected_energy_per_motor(
                rotor_power,
                rotor["output_torque"],
                reducer_rated_torque,
                sim_dt,
            )
        )
        terms["reducer_corrected_energy_per_motor"] = (
            reducer_corrected_energy_per_motor
        )
        terms["reducer_corrected_energy"] = (
            reducer_corrected_energy_per_motor.sum(dim=-1)
        )
    terms["joint_abs_energy"] = terms["joint_drive_energy"] + terms["joint_brake_energy"]
    terms["drive_joint_abs_energy"] = (
        terms["drive_joint_drive_energy"] + terms["drive_joint_brake_energy"]
    )
    terms["rotor_abs_energy"] = terms["rotor_drive_energy"] + terms["rotor_brake_energy"]
    terms["rotor_mixed_energy"] = compute_rotor_mixed_energy(
        terms["rotor_drive_energy"], terms["rotor_brake_energy"]
    )
    return terms


def energy_cost_from_terms(
    terms: Dict[str, torch.Tensor],
    mode: str,
) -> torch.Tensor:
    if mode == LEGACY_JOINT_ABS_10:
        return terms["legacy_joint_abs_10_cost"]
    if mode == JOINT_ABS_8:
        return terms["drive_joint_abs_energy"]
    if mode == JOINT_POSITIVE_8:
        return terms["drive_joint_drive_energy"]
    if mode == ROTOR_ABS_8:
        return terms["rotor_abs_energy"]
    if mode == ROTOR_POSITIVE_8:
        return terms["rotor_drive_energy"]
    if mode in (ROTOR_MIXED_8, ROTOR_MIXED_8_ALPHA050):
        return terms["rotor_mixed_energy"]
    if mode == REDUCER_CORRECTED_8:
        if "reducer_corrected_energy" not in terms:
            raise ValueError(
                "Reducer-corrected energy requires reducer_rated_torque."
            )
        return terms["reducer_corrected_energy"]
    raise ValueError(f"Unsupported energy cost mode: {mode}")


def energy_costs_from_policy_step_buffers(
    joint_power_all: torch.Tensor,
    joint_drive_energy_per_joint: torch.Tensor,
    joint_brake_energy_per_joint: torch.Tensor,
    rotor_drive_energy: torch.Tensor,
    rotor_brake_energy: torch.Tensor,
    reducer_corrected_energy: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """Reconstruct every control cost from one shared policy-step trajectory."""

    drive_joint_positive = _select_last_dim(
        joint_drive_energy_per_joint, BRUCE_DRIVE_JOINT_INDICES
    ).sum(dim=-1)
    drive_joint_negative = _select_last_dim(
        joint_brake_energy_per_joint, BRUCE_DRIVE_JOINT_INDICES
    ).sum(dim=-1)
    costs = {
        ROTOR_POSITIVE_8: rotor_drive_energy,
        ROTOR_ABS_8: rotor_drive_energy + rotor_brake_energy,
        JOINT_POSITIVE_8: drive_joint_positive,
        JOINT_ABS_8: drive_joint_positive + drive_joint_negative,
        LEGACY_JOINT_ABS_10: torch.abs(joint_power_all[..., :10]).sum(
            dim=-1
        ),
    }
    if reducer_corrected_energy is not None:
        costs[REDUCER_CORRECTED_8] = reducer_corrected_energy
    return costs
