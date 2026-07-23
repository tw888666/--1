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
  vx010_rm8a05_s00_model4000
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
rc8    = reducer_corrected_8（默认 T_N=2.1 N·m）
```

Each policy directory contains:

```text
metadata.json
episode_costs.csv
calibration_summary.json
```

Use `metadata.json` as the source of truth for `load_run`, `checkpoint`,
`energy_cost_mode`, `command_x`, and `seed`.

## Paired Calibrations

Paired multi-seed vx=0.2 calibration outputs use:

```text
paired100/<policy_id>/
```

The paired cohort includes the model-3000 baseline and model-4001 controls:

```text
paired100/
  vx020_rp8_s00_model3000
  vx020_rp8_s00_model4001
  vx020_ra8_s00_model4001
  vx020_jp8_s00_model4001
  vx020_ja8_s00_model4001
  vx020_lja10_s00_model4001
```

Seeds use the same `sNN` token, so the full paired set continues through
`s05` where available.

## Legacy Outputs

Older or different-scope calibration runs are kept under `legacy/` and renamed
with the same policy-id vocabulary where possible. Keep them separate from the
current cohorts unless their metadata matches the same protocol.
