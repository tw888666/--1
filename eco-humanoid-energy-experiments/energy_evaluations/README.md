# Energy Evaluation Outputs

This directory stores fixed-command policy evaluation outputs used for energy analysis.

## Current Layout

Organized fixed-command evaluations use:

```text
fixed<episodes>/<policy_id>/
```

The current main cohort is:

```text
fixed20/
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
episode_summary.csv
phase_summary.csv
step_timeseries.csv
eval_report/
```

Use `metadata.json` as the source of truth for `load_run`, `checkpoint`,
`energy_cost_mode`, `command_x`, and `seed`.

## Special Fixed Evaluations

Short or modified fixed-command checks use the same policy id plus a suffix:

```text
fixed1/
  vx020_ra8_s00_model4001_mirror
  vx020_ja8_s00_model4001_mirror

fixed5/
  vx020_ra8_s00_model4001_phase050
  vx020_ja8_s00_model4001_phase050
```

## Legacy Outputs

Older or different-scope runs are kept under `legacy/` and renamed with the
same policy-id vocabulary where possible. Keep them separate from the current
cohort unless their metadata matches the same protocol.
