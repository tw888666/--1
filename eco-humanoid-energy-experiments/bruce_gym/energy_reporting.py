from bruce_gym.rotor_energy import compute_rotor_mixed_energy, parse_efficiency_spec


def _expanded_efficiencies(values, count):
    values = parse_efficiency_spec(values)
    if values is None:
        return None
    if isinstance(values, tuple):
        if len(values) != count:
            raise ValueError(
                f"Expected {count} per-motor efficiencies, got {len(values)}."
            )
        return values
    return (values,) * count


def rotor_energy_totals(
    rotor_positive_energy,
    rotor_negative_energy,
    reducer_motoring_efficiency=None,
    reducer_generating_efficiency=None,
):
    positive_total = sum(rotor_positive_energy)
    negative_total = sum(rotor_negative_energy)
    abs_total = positive_total + negative_total
    totals = {
        "rotor_positive_energy_8": positive_total,
        "rotor_negative_energy_8": negative_total,
        "rotor_abs_energy_8": abs_total,
        "rotor_mixed_energy_8": compute_rotor_mixed_energy(
            positive_total, negative_total
        ),
    }
    if (
        reducer_motoring_efficiency is not None
        and reducer_generating_efficiency is not None
    ):
        motoring = _expanded_efficiencies(
            reducer_motoring_efficiency, len(rotor_positive_energy)
        )
        generating = _expanded_efficiencies(
            reducer_generating_efficiency, len(rotor_negative_energy)
        )
        totals["reducer_corrected_energy_8"] = sum(
            positive / eta_mot - eta_gen * negative
            for positive, negative, eta_mot, eta_gen in zip(
                rotor_positive_energy,
                rotor_negative_energy,
                motoring,
                generating,
            )
        )
    return totals
