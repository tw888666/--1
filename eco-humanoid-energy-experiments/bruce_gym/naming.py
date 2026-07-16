"""Shared naming helpers for experiment artifacts."""

ENERGY_COST_ALIASES = {
    "rotor_positive_8": "rp8",
    "rotor_abs_8": "ra8",
    "joint_positive_8": "jp8",
    "joint_abs_8": "ja8",
    "legacy_joint_abs_10": "lja10",
    "rotor_mixed_8_alpha050": "rm8a05",
    "rotor_mixed_8": "rm8",
}


def velocity_token(command_x) -> str:
    """Return vx token such as vx010 or vx020 from a velocity in m/s."""
    return f"vx{int(round(float(command_x) * 100)):03d}"


def seed_token(seed) -> str:
    value = 0 if seed is None else int(seed)
    return f"s{value:02d}"


def checkpoint_token(checkpoint) -> str:
    if checkpoint is None or int(checkpoint) < 0:
        return "modellatest"
    return f"model{int(checkpoint)}"


def energy_cost_alias(mode: str) -> str:
    return ENERGY_COST_ALIASES.get(mode, mode)


def policy_id(command_x, energy_cost_mode: str, seed, checkpoint, suffix=None) -> str:
    parts = [
        velocity_token(command_x),
        energy_cost_alias(energy_cost_mode),
        seed_token(seed),
        checkpoint_token(checkpoint),
    ]
    if suffix:
        parts.append(str(suffix))
    return "_".join(parts)
