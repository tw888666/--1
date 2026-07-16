# Energy Calibration Outputs

This directory stores training-distribution cost calibration outputs.

## Current Layout

Organized training-distribution calibrations use:

```text
train_dist<episodes>/<policy_id>/
```

The current main cohort is:

```text
train_dist100/
  vx010_rp8_s00_model4000
  vx010_ra8_s00_model4000
  vx010_jp8_s00_model4000
  vx010_ja8_s00_model4000
  vx010_lja10_s00_model4000
  vx020_rp8_s00_model4001
  vx020_ra8_s00_model4001
  vx020_jp8_s00_model4001
  vx020_ja8_s00_model4001
  vx020_lja10_s00_model4001
```

Policy id format:

```text
vx<command_x*100>_<cost_alias>_s<seed>_model<checkpoint>
```

Cost aliases:

```text
rp8    = rotor_positive_8
ra8    = rotor_abs_8
jp8    = joint_positive_8
ja8    = joint_abs_8
lja10  = legacy_joint_abs_10
rm8a05 = rotor_mixed_8_alpha050
```

Each policy directory contains:

```text
metadata.json
episode_costs.csv
calibration_summary.json
```

Use `metadata.json` as the source of truth for `load_run`, `checkpoint`,
`energy_cost_mode`, `command_x`, and `seed`.

## Legacy Outputs

Top-level `calib_*` and `paired_*` directories are older or different-scope
calibration runs. Keep them separate from `train_dist100/` unless their metadata
matches the same protocol.
