# Energy Evaluation Outputs

This directory stores policy evaluation outputs used for energy analysis.

## Fixed Command Evaluation

`fixed_command_vx010_20ep/` contains flat, deterministic fixed-command evaluations:

```text
command_x = 0.10 m/s
command_y = 0.0 m/s
command_yaw = 0.0 rad/s
episodes = 20
```

The organized model-4000 keep-optimizer cohort is:

```text
fixed_command_vx010_20ep/
  model4000_keepopt/
    rotor_positive_8/seed0..seed5/
    rotor_abs_8/seed0..seed5/
    joint_positive_8/seed0..seed5/
    legacy_joint_abs_10/seed0..seed5/
```

Each `seedN/` directory contains:

```text
metadata.json
episode_summary.csv
phase_summary.csv
step_timeseries.csv
```

These files are the right source for joint energy, phase energy, and negative-power decomposition plots.

## Legacy Top-Level Outputs

Top-level evaluation directories that remain here are older or different-scope runs, such as model-3000 baseline checks, vx015/vx020 checks, or checkpoint sweeps. Keep them separate from the organized fixed-command model-4000 cohort unless their metadata matches the same protocol.
