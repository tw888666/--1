# Energy Calibration Outputs

This directory stores training-distribution cost calibration outputs.

## Current Organized Layout

`train_dist_vx010/model4000_keepopt/` contains the six-seed model-4000 keep-optimizer calibration cohort:

```text
train_dist_vx010/
  model4000_keepopt/
    rotor_positive_8/seed0..seed5/
    rotor_abs_8/seed0..seed5/
    joint_positive_8/seed0..seed5/
    legacy_joint_abs_10/seed0..seed5/
```

Each `seedN/` directory contains:

```text
metadata.json
episode_costs.csv
calibration_summary.json
```

## Legacy Top-Level Outputs

Top-level `calib_*` directories that remain here are older or different-scope calibration runs, such as model-3000 baseline calibration, vx020 checks, or non-keepopt/cohort runs. Do not mix them with the organized `model4000_keepopt` cohort unless their metadata matches the same protocol.
