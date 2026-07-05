from bruce_gym.rotor_energy import compute_rotor_mixed_energy


def rotor_energy_totals(rotor_positive_energy, rotor_negative_energy):
    positive_total = sum(rotor_positive_energy)
    negative_total = sum(rotor_negative_energy)
    return {
        "rotor_positive_energy_8": positive_total,
        "rotor_negative_energy_8": negative_total,
        "rotor_mixed_energy_8": compute_rotor_mixed_energy(
            positive_total, negative_total
        ),
    }
