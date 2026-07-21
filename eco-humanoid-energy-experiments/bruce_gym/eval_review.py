# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2026 ECO Authors. All rights reserved.

import csv
import json
import math
import os
import re
import statistics
from collections import defaultdict


DEFAULT_METRIC = "e_mix_per_m"
EPSILON_DISTANCE_M = 1e-8
GAIT_STAGES = (
    {
        "stage": "left_touchdown_transfer",
        "phase": "[0.95,1.00)+[0.00,0.05)",
        "ranges": ((0.95, 1.0), (0.0, 0.05)),
        "width": 0.10,
    },
    {
        "stage": "left_early_stance_right_swing",
        "phase": "[0.05,0.25)",
        "ranges": ((0.05, 0.25),),
        "width": 0.20,
    },
    {
        "stage": "left_late_stance_right_placement",
        "phase": "[0.25,0.45)",
        "ranges": ((0.25, 0.45),),
        "width": 0.20,
    },
    {
        "stage": "right_touchdown_transfer",
        "phase": "[0.45,0.55)",
        "ranges": ((0.45, 0.55),),
        "width": 0.10,
    },
    {
        "stage": "right_early_stance_left_swing",
        "phase": "[0.55,0.75)",
        "ranges": ((0.55, 0.75),),
        "width": 0.20,
    },
    {
        "stage": "right_late_stance_left_placement",
        "phase": "[0.75,0.95)",
        "ranges": ((0.75, 0.95),),
        "width": 0.20,
    },
)
GAIT_STAGE_NAMES_ZH = {
    "left_touchdown_transfer": "左脚落地与重心转移",
    "left_early_stance_right_swing": "左腿前支撑、右腿摆动",
    "left_late_stance_right_placement": "左腿后支撑、右脚落位",
    "right_touchdown_transfer": "右脚落地与重心转移",
    "right_early_stance_left_swing": "右腿前支撑、左腿摆动",
    "right_late_stance_left_placement": "右腿后支撑、左脚落位",
}
JOINT_NAMES_ZH = {
    "hip_yaw_l": "左髋偏航",
    "hip_pitch_l": "左髋俯仰",
    "hip_roll_l": "左髋侧倾",
    "knee_pitch_l": "左膝俯仰",
    "ankle_pitch_l": "左踝俯仰",
    "hip_yaw_r": "右髋偏航",
    "hip_pitch_r": "右髋俯仰",
    "hip_roll_r": "右髋侧倾",
    "knee_pitch_r": "右膝俯仰",
    "ankle_pitch_r": "右踝俯仰",
}
ENERGY_COST_MODE_NAMES_ZH = {
    "legacy_joint_abs_10": "旧版10关节绝对能量",
    "joint_abs_8": "8驱动关节绝对能量",
    "joint_positive_8": "8驱动关节正向能量",
    "rotor_abs_8": "8电机转子绝对能量",
    "rotor_positive_8": "8电机转子正向能量",
}


def _to_float(value, default=0.0):
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _evaluation_variant_description(metadata):
    parts = []
    phase_offset = _to_float(metadata.get("gait_phase_offset"), default=0.0) % 1.0
    if abs(phase_offset) > 1e-9:
        parts.append(f"步态相位偏移：{phase_offset:g}周期")
    if metadata.get("mirrored_policy"):
        parts.append("策略：左右镜像策略")
    return "｜".join(parts)


def _to_int(value, default=0):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _safe_div(numerator, denominator, default=math.inf):
    if abs(denominator) <= EPSILON_DISTANCE_M:
        return default
    return numerator / denominator


def _finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def _mean(values, default=0.0):
    finite_values = [value for value in values if _finite(value)]
    if not finite_values:
        return default
    return sum(finite_values) / len(finite_values)


def _rms(values, default=0.0):
    finite_values = [value for value in values if _finite(value)]
    if not finite_values:
        return default
    return math.sqrt(sum(value * value for value in finite_values) / len(finite_values))


def _read_csv_dicts(path):
    with open(path, encoding="utf-8", newline="") as csvfile:
        return list(csv.DictReader(csvfile))


def _write_csv_dicts(path, rows, preferred_fields=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fields = []
    seen = set()
    for field in preferred_fields or []:
        if field not in seen:
            seen.add(field)
            fields.append(field)
    for row in rows:
        for field in row.keys():
            if field not in seen:
                seen.add(field)
                fields.append(field)

    with open(path, "w", encoding="utf-8", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _read_metadata(path):
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as jsonfile:
        return json.load(jsonfile)


def _group_steps_by_episode(step_rows):
    grouped = defaultdict(list)
    for row in step_rows:
        grouped[_to_int(row.get("episode_id"))].append(row)
    return dict(grouped)


def _row_series(rows, key):
    return [_to_float(row.get(key), default=math.nan) for row in rows]


def _dynamic_state_rows(rows):
    if not any("pre_step_dynamic_state_valid" in row for row in rows):
        return rows
    return [
        row
        for row in rows
        if _to_float(row.get("pre_step_dynamic_state_valid")) > 0.5
    ]


def _steady_state_rows(rows):
    rows = _dynamic_state_rows(rows)
    timed_rows = [
        row
        for row in rows
        if _finite(_to_float(row.get("time_s"), default=math.nan))
    ]
    if len(timed_rows) < 3:
        return timed_rows

    start_time = _to_float(timed_rows[0].get("time_s"))
    end_time = _to_float(timed_rows[-1].get("time_s"))
    duration = end_time - start_time
    if duration < 6.0:
        return timed_rows

    steady_start = start_time + duration / 3.0
    steady_end = start_time + 2.0 * duration / 3.0
    steady_rows = [
        row
        for row in timed_rows
        if steady_start <= _to_float(row.get("time_s")) <= steady_end
    ]
    return steady_rows or timed_rows


def _phase_binned_series(rows, key, bins=50):
    totals = [0.0] * bins
    counts = [0] * bins
    for row in rows:
        phase = _to_float(row.get("gait_phase"), default=math.nan)
        value = _to_float(row.get(key), default=math.nan)
        if not _finite(phase) or not _finite(value):
            continue
        bin_index = min(int((phase % 1.0) * bins), bins - 1)
        totals[bin_index] += value
        counts[bin_index] += 1
    centers = [(index + 0.5) / bins for index in range(bins)]
    values = [
        totals[index] / counts[index] if counts[index] else math.nan
        for index in range(bins)
    ]
    return centers, values


def _estimate_cycle_time(rows, policy_dt):
    phase_steps = []
    for previous, current in zip(rows, rows[1:]):
        previous_phase = _to_float(previous.get("gait_phase"), default=math.nan)
        current_phase = _to_float(current.get("gait_phase"), default=math.nan)
        if not _finite(previous_phase) or not _finite(current_phase):
            continue
        phase_step = (current_phase - previous_phase) % 1.0
        if 0.0 < phase_step < 0.25:
            phase_steps.append(phase_step)
    if not phase_steps:
        return 0.55
    return policy_dt / statistics.median(phase_steps)


def build_gait_stage_joint_energy(rows, metadata):
    rows = _steady_state_rows(rows)
    if not rows:
        return [], []

    joint_names = list(metadata.get("joint_names") or [])
    if not joint_names:
        joint_names = [
            column[: -len("_power")]
            for column in _columns_with_suffix(rows, "_power", exclude="motor")
        ]
    policy_dt = _to_float(metadata.get("policy_dt"), default=0.01)
    cycle_time = _estimate_cycle_time(rows, policy_dt)

    table_rows = []
    for stage in GAIT_STAGES:
        stage_rows = []
        for row in rows:
            phase = _to_float(row.get("gait_phase"), default=math.nan) % 1.0
            if any(start <= phase < end for start, end in stage["ranges"]):
                stage_rows.append(row)

        expected_steps = stage["width"] * cycle_time / policy_dt
        table_row = {"stage": stage["stage"], "phase": stage["phase"]}
        for joint_name in joint_names:
            per_step_energy = [
                _to_float(row.get(f"{joint_name}_positive_energy"))
                + _to_float(row.get(f"{joint_name}_negative_energy"))
                for row in stage_rows
            ]
            table_row[joint_name] = _mean(per_step_energy) * expected_steps
        table_row["total"] = sum(table_row[joint_name] for joint_name in joint_names)
        table_rows.append(table_row)
    return table_rows, joint_names


def _localized_gait_energy_rows(table_rows, joint_names):
    localized_rows = []
    localized_joint_fields = [
        JOINT_NAMES_ZH.get(joint_name, joint_name) for joint_name in joint_names
    ]
    for row in table_rows:
        localized_row = {
            "步态阶段": GAIT_STAGE_NAMES_ZH.get(row["stage"], row["stage"]),
            "相位范围": row["phase"],
        }
        for joint_name, localized_field in zip(joint_names, localized_joint_fields):
            localized_row[localized_field] = row[joint_name]
        localized_row["合计"] = row["total"]
        localized_rows.append(localized_row)
    return localized_rows, localized_joint_fields


def _cumulative(rows, key):
    total = 0.0
    values = []
    for row in rows:
        total += _to_float(row.get(key), default=0.0)
        values.append(total)
    return values


def _columns_with_suffix(rows, suffix, include=None, exclude=None):
    if not rows:
        return []
    columns = []
    for column in rows[0].keys():
        if not column.endswith(suffix):
            continue
        if include and include not in column:
            continue
        if exclude and exclude in column:
            continue
        columns.append(column)
    return columns


def _per_row_rms(rows, columns):
    values = []
    for row in rows:
        values.append(_rms([_to_float(row.get(column), math.nan) for column in columns]))
    return values


def _contact_fractions(rows):
    valid_rows = [
        row
        for row in rows
        if row.get("left_contact_state") != "" and row.get("right_contact_state") != ""
    ]
    if not valid_rows:
        return {
            "left_contact_fraction": 0.0,
            "right_contact_fraction": 0.0,
            "double_support_fraction": 0.0,
            "flight_fraction": 0.0,
        }

    left = [_to_float(row.get("left_contact_state")) > 0.5 for row in valid_rows]
    right = [_to_float(row.get("right_contact_state")) > 0.5 for row in valid_rows]
    count = len(valid_rows)
    return {
        "left_contact_fraction": sum(left) / count,
        "right_contact_fraction": sum(right) / count,
        "double_support_fraction": sum(l and r for l, r in zip(left, right)) / count,
        "flight_fraction": sum((not l) and (not r) for l, r in zip(left, right))
        / count,
    }


def _episode_metric_row(episode_row, step_rows, metadata):
    row = dict(episode_row)
    episode_id = _to_int(row.get("episode_id"))
    distance_x = _to_float(row.get("distance_x"))
    command_x = _to_float(
        row.get("command_x"),
        _to_float(metadata.get("command_x"), default=0.0),
    )
    duration_s = _to_float(row.get("duration_s"))
    success = row.get("episode_outcome") == "success" or _to_float(row.get("timeout")) > 0.5
    fall = row.get("episode_outcome") == "fall" or _to_float(row.get("fall")) > 0.5

    velocity_errors = [
        _to_float(step.get("base_vel_x"), default=math.nan)
        - _to_float(step.get("command_x"), default=command_x)
        for step in step_rows
    ]
    root_heights = _row_series(step_rows, "post_step_root_pos_z")
    rotor_power_columns = _columns_with_suffix(step_rows, "_power", include="motor")
    joint_power_columns = _columns_with_suffix(step_rows, "_power", exclude="motor")
    rotor_torque_columns = _columns_with_suffix(step_rows, "_torque", include="motor")

    row.update(
        {
            "episode_id": episode_id,
            "success": float(success),
            "fall": float(fall),
            "command_x": command_x,
            "e_pos_per_m": _safe_div(
                _to_float(row.get("rotor_positive_energy_8")), distance_x
            ),
            "e_neg_per_m": _safe_div(
                _to_float(row.get("rotor_negative_energy_8")), distance_x
            ),
            "e_mix_per_m": _safe_div(
                _to_float(row.get("rotor_mixed_energy_8")), distance_x
            ),
            "joint_pos_per_m": _safe_div(
                _to_float(row.get("joint_positive_energy_10")), distance_x
            ),
            "joint_neg_per_m": _safe_div(
                _to_float(row.get("joint_negative_energy_10")), distance_x
            ),
            "velocity_error_rms": _rms(velocity_errors),
            "mean_abs_velocity_error": _mean([abs(value) for value in velocity_errors]),
            "root_height_min": min(root_heights) if root_heights else 0.0,
            "root_height_mean": _mean(root_heights),
            "rotor_power_rms_mean": _mean(_per_row_rms(step_rows, rotor_power_columns)),
            "joint_power_rms_mean": _mean(_per_row_rms(step_rows, joint_power_columns)),
            "rotor_torque_rms_mean": _mean(_per_row_rms(step_rows, rotor_torque_columns)),
            "review_duration_s": duration_s,
        }
    )
    reducer_corrected = row.get("reducer_corrected_energy_8")
    if reducer_corrected not in (None, ""):
        row["e_reducer_corrected_per_m"] = _safe_div(
            _to_float(reducer_corrected), distance_x
        )
    row.update(_contact_fractions(step_rows))
    return row


def build_summary_rows(episode_rows, step_rows, metadata=None):
    metadata = metadata or {}
    grouped_steps = _group_steps_by_episode(step_rows)
    return [
        _episode_metric_row(row, grouped_steps.get(_to_int(row.get("episode_id")), []), metadata)
        for row in episode_rows
    ]


def select_representatives(summary_rows, metric=DEFAULT_METRIC):
    finite_rows = [
        row for row in summary_rows if _finite(_to_float(row.get(metric), default=math.inf))
    ]
    successful_rows = [row for row in finite_rows if _to_float(row.get("success")) > 0.5]
    cohort = successful_rows or finite_rows or list(summary_rows)
    if not cohort:
        return []

    best = min(cohort, key=lambda row: _to_float(row.get(metric), default=math.inf))
    metric_values = [_to_float(row.get(metric), default=math.inf) for row in cohort]
    median_value = statistics.median(metric_values)
    median = min(
        cohort,
        key=lambda row: abs(_to_float(row.get(metric), default=math.inf) - median_value),
    )

    fall_rows = [row for row in summary_rows if _to_float(row.get("fall")) > 0.5]
    if fall_rows:
        worst = max(
            fall_rows,
            key=lambda row: (
                _to_float(row.get(metric), default=-math.inf),
                -_to_float(row.get("duration_s"), default=0.0),
            ),
        )
        worst_reason = "fall episode, then highest metric"
    else:
        worst = max(cohort, key=lambda row: _to_float(row.get(metric), default=-math.inf))
        worst_reason = "highest metric among successful episodes"

    return [
        {
            "label": "best",
            "episode_id": _to_int(best.get("episode_id")),
            "metric": metric,
            "metric_value": _to_float(best.get(metric), default=math.inf),
            "reason": "lowest metric among successful episodes",
        },
        {
            "label": "median",
            "episode_id": _to_int(median.get("episode_id")),
            "metric": metric,
            "metric_value": _to_float(median.get(metric), default=math.inf),
            "reason": "closest to median metric among successful episodes",
        },
        {
            "label": "worst",
            "episode_id": _to_int(worst.get("episode_id")),
            "metric": metric,
            "metric_value": _to_float(worst.get(metric), default=math.inf),
            "reason": worst_reason,
        },
    ]


def _format_number(value, digits=4):
    value = _to_float(value, default=math.nan)
    if not math.isfinite(value):
        return "n/a"
    return f"{value:.{digits}f}"


def _load_pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _configure_chinese_plot_font(plt):
    plt.rcParams["font.sans-serif"] = [
        "Noto Sans CJK JP",
        "DejaVu Sans",
        "Droid Sans Fallback",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def _time_axis(rows, policy_dt):
    return [
        _to_float(row.get("time_s"), _to_float(row.get("episode_step")) * policy_dt)
        for row in rows
    ]


def _add_time_column(rows, policy_dt):
    enriched = []
    for row in rows:
        time_s = _to_float(row.get("episode_step")) * policy_dt
        enriched.append({"time_s": time_s, **row})
    return enriched


def _plot_episode_curves(path, rows, metadata, title):
    plt = _load_pyplot()
    policy_dt = _to_float(metadata.get("policy_dt"), default=0.01)
    time_axis = _time_axis(rows, policy_dt)
    rotor_power_columns = _columns_with_suffix(rows, "_power", include="motor")
    joint_power_columns = _columns_with_suffix(rows, "_power", exclude="motor")

    fig, axes = plt.subplots(4, 1, figsize=(12, 11), sharex=True)
    fig.suptitle(title)

    velocity_rows = _dynamic_state_rows(rows)
    velocity_time_axis = _time_axis(velocity_rows, policy_dt)
    axes[0].plot(
        velocity_time_axis,
        _row_series(velocity_rows, "base_vel_x"),
        label="base_vel_x",
    )
    axes[0].plot(
        velocity_time_axis,
        _row_series(velocity_rows, "command_x"),
        label="command_x",
    )
    axes[0].set_ylabel("m/s")
    axes[0].legend(loc="best")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(time_axis, _row_series(rows, "post_step_root_pos_z"), label="root_z")
    axes[1].plot(time_axis, _row_series(rows, "left_contact_state"), label="left_contact")
    axes[1].plot(time_axis, _row_series(rows, "right_contact_state"), label="right_contact")
    axes[1].set_ylabel("height/contact")
    axes[1].legend(loc="best")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(
        time_axis,
        _cumulative(rows, "rotor_positive_energy_8"),
        label="rotor_E_pos",
    )
    axes[2].plot(
        time_axis,
        _cumulative(rows, "rotor_negative_energy_8"),
        label="rotor_E_neg",
    )
    axes[2].plot(
        time_axis,
        _cumulative(rows, "rotor_mixed_energy_8"),
        label="rotor_E_mix",
    )
    axes[2].set_ylabel("J")
    axes[2].legend(loc="best")
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(
        time_axis,
        _per_row_rms(rows, rotor_power_columns),
        label="rotor_power_rms",
    )
    axes[3].plot(
        time_axis,
        _per_row_rms(rows, joint_power_columns),
        label="joint_power_rms",
    )
    axes[3].set_xlabel("time (s)")
    axes[3].set_ylabel("W RMS")
    axes[3].legend(loc="best")
    axes[3].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_episode_velocity(path, rows, metadata, title):
    plt = _load_pyplot()
    policy_dt = _to_float(metadata.get("policy_dt"), default=0.01)
    rows = _dynamic_state_rows(rows)
    time_axis = _time_axis(rows, policy_dt)

    fig, axis = plt.subplots(figsize=(12, 4))
    axis.plot(time_axis, _row_series(rows, "base_vel_x"), label="base_vel_x")
    axis.plot(time_axis, _row_series(rows, "command_x"), label="command_x")
    axis.set_title(title)
    axis.set_xlabel("time (s)")
    axis.set_ylabel("m/s")
    axis.legend(loc="best")
    axis.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_gait_cycle_summary(path, rows, metadata, title):
    plt = _load_pyplot()
    rows = _steady_state_rows(rows)
    if not rows:
        return

    phase, velocity = _phase_binned_series(rows, "base_vel_x")
    _, command = _phase_binned_series(rows, "command_x")
    _, left_contact = _phase_binned_series(rows, "left_contact_state")
    _, right_contact = _phase_binned_series(rows, "right_contact_state")

    joint_names = list(metadata.get("joint_names") or [])
    if not joint_names:
        joint_names = [
            column[: -len("_power")]
            for column in _columns_with_suffix(rows, "_power", exclude="motor")
        ]
    joint_power = [
        _phase_binned_series(rows, f"{joint_name}_power")[1]
        for joint_name in joint_names
    ]

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(12, 10),
        sharex=True,
        gridspec_kw={"height_ratios": [1.2, 1.0, 2.4]},
    )
    fig.suptitle(title)

    axes[0].axvspan(0.0, 0.5, color="tab:blue", alpha=0.08)
    axes[0].axvspan(0.5, 1.0, color="tab:orange", alpha=0.08)
    axes[0].plot(phase, velocity, label="phase-mean base_vel_x", linewidth=2.0)
    axes[0].plot(phase, command, label="command_x", linewidth=1.8)
    axes[0].text(
        0.25,
        0.94,
        "left stance / right swing",
        ha="center",
        va="top",
        transform=axes[0].get_xaxis_transform(),
    )
    axes[0].text(
        0.75,
        0.94,
        "right stance / left swing",
        ha="center",
        va="top",
        transform=axes[0].get_xaxis_transform(),
    )
    axes[0].set_ylabel("velocity (m/s)")
    axes[0].legend(loc="lower right")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(
        phase,
        left_contact,
        label="left contact",
        linewidth=2.0,
        drawstyle="steps-mid",
    )
    axes[1].plot(
        phase,
        right_contact,
        label="right contact",
        linewidth=2.0,
        drawstyle="steps-mid",
    )
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].set_ylabel("contact probability")
    axes[1].legend(loc="center right")
    axes[1].grid(True, alpha=0.3)

    if joint_power:
        finite_power = [
            abs(value)
            for series in joint_power
            for value in series
            if _finite(value)
        ]
        power_limit = max(finite_power, default=1.0) or 1.0
        image = axes[2].imshow(
            joint_power,
            aspect="auto",
            interpolation="nearest",
            cmap="coolwarm",
            vmin=-power_limit,
            vmax=power_limit,
            extent=(0.0, 1.0, len(joint_names) - 0.5, -0.5),
        )
        axes[2].set_yticks(range(len(joint_names)))
        axes[2].set_yticklabels(joint_names)
        colorbar = fig.colorbar(image, ax=axes[2], pad=0.01)
        colorbar.set_label("joint power (W): red=drive, blue=brake")
    axes[2].set_xlabel("gait phase (one cycle = 0.55 s)")
    axes[2].set_ylabel("joint")

    for axis in axes:
        axis.axvline(0.5, color="black", linewidth=1.0, linestyle="--", alpha=0.6)
        axis.set_xlim(0.0, 1.0)
    axes[2].set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_gait_stage_joint_energy_table(path, table_rows, joint_names, title):
    if not table_rows or not joint_names:
        return

    plt = _load_pyplot()
    _configure_chinese_plot_font(plt)
    localized_joint_names = [
        JOINT_NAMES_ZH.get(joint_name, joint_name) for joint_name in joint_names
    ]
    columns = ["步态阶段", *localized_joint_names, "合计"]
    cell_text = []
    energy_values = []
    for row in table_rows:
        values = [row[joint_name] for joint_name in joint_names]
        energy_values.extend(values)
        cell_text.append(
            [
                f"{GAIT_STAGE_NAMES_ZH.get(row['stage'], row['stage'])}\n{row['phase']}",
                *[f"{value:.4f}" for value in values],
                f"{row['total']:.4f}",
            ]
        )

    fig, axis = plt.subplots(figsize=(22, 6.4))
    axis.axis("off")
    axis.set_title(title, pad=18, fontsize=14)
    column_widths = [0.21, *([0.064] * len(joint_names)), 0.075]
    table = axis.table(
        cellText=cell_text,
        colLabels=columns,
        cellLoc="center",
        colLoc="center",
        colWidths=column_widths,
        bbox=(0.0, 0.12, 1.0, 0.80),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)

    energy_limit = max(energy_values, default=1.0) or 1.0
    color_map = plt.get_cmap("YlOrRd")
    for row_index, row in enumerate(table_rows, start=1):
        table[(row_index, 0)].set_text_props(ha="left")
        for column_index, joint_name in enumerate(joint_names, start=1):
            normalized = row[joint_name] / energy_limit
            table[(row_index, column_index)].set_facecolor(
                color_map(0.08 + 0.82 * normalized)
            )
        table[(row_index, len(columns) - 1)].set_text_props(weight="bold")

    for column_index in range(len(columns)):
        table[(0, column_index)].set_facecolor("#d9e6f2")
        table[(0, column_index)].set_text_props(weight="bold")

    axis.text(
        0.0,
        0.02,
        "能量定义：绝对机械能＝正向机械能＋负向机械能。"
        "数值为按相位归一化的每完整步态周期平均能量，单位：焦耳/周期。"
        "颜色越深表示能量越高。",
        transform=axis.transAxes,
        fontsize=9,
        ha="left",
        va="bottom",
    )
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _joint_names_from_metadata(metadata, summary_rows):
    joint_names = metadata.get("joint_names") or []
    if joint_names:
        return list(joint_names)
    if not summary_rows:
        return []
    names = []
    for key in summary_rows[0].keys():
        match = re.match(r"(.+)_positive_energy$", key)
        if not match:
            continue
        name = match.group(1)
        if "motor" in name or name.startswith("rotor"):
            continue
        names.append(name)
    return names


def _plot_joint_energy(path, summary_rows, representatives, metadata):
    joint_names = _joint_names_from_metadata(metadata, summary_rows)
    if not joint_names:
        return

    plt = _load_pyplot()
    summary_by_episode = {
        _to_int(row.get("episode_id")): row for row in summary_rows
    }
    selected_rows = [
        (rep["label"], summary_by_episode.get(_to_int(rep["episode_id"])))
        for rep in representatives
    ]
    selected_rows = [(label, row) for label, row in selected_rows if row]
    if not selected_rows:
        return

    fig, axes = plt.subplots(
        len(selected_rows), 1, figsize=(12, 3.2 * len(selected_rows)), squeeze=False
    )
    x_values = list(range(len(joint_names)))
    width = 0.38
    for ax, (label, row) in zip(axes[:, 0], selected_rows):
        positive = [
            _to_float(row.get(f"{joint_name}_positive_energy")) for joint_name in joint_names
        ]
        negative = [
            _to_float(row.get(f"{joint_name}_negative_energy")) for joint_name in joint_names
        ]
        ax.bar([x - width / 2 for x in x_values], positive, width, label="positive")
        ax.bar([x + width / 2 for x in x_values], negative, width, label="negative")
        ax.set_title(f"{label} episode joint energy")
        ax.set_ylabel("J")
        ax.set_xticks(x_values)
        ax.set_xticklabels(joint_names, rotation=35, ha="right")
        ax.legend(loc="best")
        ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _write_episode_csvs(output_dir, grouped_steps, metadata):
    episode_dir = os.path.join(output_dir, "episodes")
    os.makedirs(episode_dir, exist_ok=True)
    policy_dt = _to_float(metadata.get("policy_dt"), default=0.01)
    paths = {}
    for episode_id, rows in sorted(grouped_steps.items()):
        enriched = _add_time_column(rows, policy_dt)
        path = os.path.join(episode_dir, f"episode_{episode_id:04d}.csv")
        _write_csv_dicts(path, enriched, preferred_fields=["episode_id", "episode_step", "time_s"])
        paths[episode_id] = path
    return paths


def _write_report(
    path,
    eval_dir,
    output_dir,
    metadata,
    summary_rows,
    representatives,
    plot_status,
):
    episodes = len(summary_rows)
    successes = sum(_to_float(row.get("success")) > 0.5 for row in summary_rows)
    success_rate = successes / episodes if episodes else 0.0
    metric_values = [_to_float(row.get(DEFAULT_METRIC), default=math.nan) for row in summary_rows]
    velocity_values = [_to_float(row.get("mean_velocity_x"), default=math.nan) for row in summary_rows]
    distance_values = [_to_float(row.get("distance_x"), default=math.nan) for row in summary_rows]
    tracking_values = [
        _to_float(row.get("velocity_error_rms"), default=math.nan) for row in summary_rows
    ]
    reducer_values = [
        _to_float(row.get("e_reducer_corrected_per_m"), default=math.nan)
        for row in summary_rows
    ]

    lines = [
        "# Evaluation Review",
        "",
        f"- eval_dir: `{eval_dir}`",
        f"- report_dir: `{output_dir}`",
        f"- task: `{metadata.get('task', 'unknown')}`",
        f"- load_run: `{metadata.get('load_run', 'unknown')}`",
        f"- checkpoint: `{metadata.get('checkpoint', 'unknown')}`",
        f"- command_x: `{metadata.get('command_x', 'unknown')}`",
        f"- 步态相位偏移: `{_to_float(metadata.get('gait_phase_offset'), 0.0):g}` 周期",
        f"- 镜像策略评估: `{'是' if metadata.get('mirrored_policy') else '否'}`",
        "",
        "## Summary",
        "",
        f"- episodes: {episodes}",
        f"- success_rate: {_format_number(success_rate * 100, 2)}%",
        f"- mean_velocity_x: {_format_number(_mean(velocity_values))} m/s",
        f"- mean_distance_x: {_format_number(_mean(distance_values))} m",
        f"- mean_E_mix_per_m: {_format_number(_mean(metric_values))} J/m",
        f"- mean_velocity_error_rms: {_format_number(_mean(tracking_values))} m/s",
        "",
        "## Representative Episodes",
        "",
        "| label | episode_id | metric | value | reason |",
        "| --- | ---: | --- | ---: | --- |",
    ]
    if any(_finite(value) for value in reducer_values):
        lines.insert(
            lines.index("## Representative Episodes") - 1,
            "- mean_E_reducer_corrected_per_m: "
            f"{_format_number(_mean([value for value in reducer_values if _finite(value)]))} J/m",
        )
    for rep in representatives:
        lines.append(
            "| {label} | {episode_id} | {metric} | {value} | {reason} |".format(
                label=rep["label"],
                episode_id=rep["episode_id"],
                metric=rep["metric"],
                value=_format_number(rep["metric_value"]),
                reason=rep["reason"],
            )
        )

    lines.extend(
        [
            "",
            "## Generated Files",
            "",
            "- `summary.csv`: per-episode metrics with energy per meter and stability proxies.",
            "- `episodes/episode_*.csv`: one time-series CSV per episode.",
            "- `representative_episodes.json`: machine-readable best/median/worst selection.",
            "- `gait_stage_joint_energy.csv`、`gait_stage_joint_energy_zh.csv`：按步态阶段统计的关节绝对机械能中文版，单位为焦耳/周期。",
        ]
    )
    if plot_status == "ok":
        lines.extend(
            [
                "- `episode_first_velocity.png`, `episode_last_velocity.png`: velocity tracking for the first and last episodes; invalid reset-boundary samples are excluded.",
                "- `gait_cycle_summary.png`: phase-averaged steady-state velocity, foot contacts, and ten-joint power over one gait cycle.",
                "- `gait_stage_joint_energy.png`、`gait_stage_joint_energy_zh.png`：按步态阶段展示关节绝对机械能的中文可视化表格。",
                "- `episode_best_curves.png`, `episode_median_curves.png`, `episode_worst_curves.png`: representative time-series plots.",
                "- `joint_energy_contribution.png`: positive/negative joint energy for representatives.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "## Plot Status",
                "",
                "PNG plots were skipped."
                if plot_status == "skipped"
                else "Plots were skipped because matplotlib was not available in this Python environment.",
                "",
            ]
        )

    reducer_rated_torque = metadata.get("reducer_rated_torque")
    reducer_rated_torque_line = []
    if reducer_rated_torque is not None:
        if isinstance(reducer_rated_torque, (list, tuple)):
            reducer_rated_torque = ",".join(
                format(float(value), ".17g") for value in reducer_rated_torque
            )
        reducer_rated_torque_line = [
            f"  --reducer_rated_torque={reducer_rated_torque} \\",
        ]

    lines.extend(
        [
            "## Video Replay",
            "",
            "After this report is generated, record the selected episodes with:",
            "",
            "```bash",
            "CUDA_VISIBLE_DEVICES=0 python -u -m bruce_gym.scripts.record_eval_video \\",
            f"  --task={metadata.get('task', '<task>')} \\",
            "  --headless \\",
            "  --resume \\",
            f"  --experiment_name={metadata.get('experiment_name', 'exp')} \\",
            f"  --load_run={metadata.get('load_run', '<load_run>')} \\",
            f"  --checkpoint={metadata.get('checkpoint', '<checkpoint>')} \\",
            f"  --energy_cost_mode={metadata.get('energy_cost_mode', '<mode>')} \\",
            *reducer_rated_torque_line,
            f"  --command_x={metadata.get('command_x', 0.1)} \\",
            f"  --command_y={metadata.get('command_y', 0.0)} \\",
            f"  --command_yaw={metadata.get('command_yaw', 0.0)} \\",
            f"  --review_dir={output_dir} \\",
            "  --sim_device=cuda:0 \\",
            "  --rl_device=cuda:0",
            "```",
            "",
        ]
    )

    with open(path, "w", encoding="utf-8") as report_file:
        report_file.write("\n".join(lines))


def generate_review(
    eval_dir,
    output_dir=None,
    metric=DEFAULT_METRIC,
    make_plots=True,
    write_episode_csvs=True,
):
    eval_dir = os.path.abspath(eval_dir)
    output_dir = os.path.abspath(output_dir or os.path.join(eval_dir, "eval_report"))
    os.makedirs(output_dir, exist_ok=True)

    metadata = _read_metadata(os.path.join(eval_dir, "metadata.json"))
    episode_rows = _read_csv_dicts(os.path.join(eval_dir, "episode_summary.csv"))
    step_rows = _read_csv_dicts(os.path.join(eval_dir, "step_timeseries.csv"))
    grouped_steps = _group_steps_by_episode(step_rows)

    summary_rows = build_summary_rows(episode_rows, step_rows, metadata)
    representatives = select_representatives(summary_rows, metric=metric)

    _write_csv_dicts(os.path.join(output_dir, "summary.csv"), summary_rows)
    if write_episode_csvs:
        _write_episode_csvs(output_dir, grouped_steps, metadata)

    first_episode_id = min(grouped_steps) if grouped_steps else None
    first_rows = (
        _add_time_column(
            grouped_steps[first_episode_id],
            _to_float(metadata.get("policy_dt"), default=0.01),
        )
        if first_episode_id is not None
        else []
    )
    gait_energy_rows, gait_joint_names = build_gait_stage_joint_energy(
        first_rows, metadata
    )
    if gait_energy_rows:
        localized_gait_energy_rows, localized_joint_fields = (
            _localized_gait_energy_rows(gait_energy_rows, gait_joint_names)
        )
        for filename in (
            "gait_stage_joint_energy.csv",
            "gait_stage_joint_energy_zh.csv",
        ):
            _write_csv_dicts(
                os.path.join(output_dir, filename),
                localized_gait_energy_rows,
                preferred_fields=[
                    "步态阶段",
                    "相位范围",
                    *localized_joint_fields,
                    "合计",
                ],
            )

    representative_payload = {
        "eval_dir": eval_dir,
        "report_dir": output_dir,
        "selection_metric": metric,
        "representatives": representatives,
    }
    with open(
        os.path.join(output_dir, "representative_episodes.json"),
        "w",
        encoding="utf-8",
    ) as jsonfile:
        json.dump(representative_payload, jsonfile, indent=2)

    plot_status = "skipped"
    if make_plots:
        try:
            if grouped_steps:
                boundary_episodes = (
                    ("first", min(grouped_steps)),
                    ("last", max(grouped_steps)),
                )
                for boundary_label, episode_id in boundary_episodes:
                    rows = _add_time_column(
                        grouped_steps[episode_id],
                        _to_float(metadata.get("policy_dt"), default=0.01),
                    )
                    _plot_episode_velocity(
                        os.path.join(
                            output_dir,
                            f"episode_{boundary_label}_velocity.png",
                        ),
                        rows,
                        metadata,
                        f"{boundary_label} episode {episode_id}",
                    )
                    if boundary_label == "first":
                        evaluation_variant = _evaluation_variant_description(metadata)
                        _plot_gait_cycle_summary(
                            os.path.join(output_dir, "gait_cycle_summary.png"),
                            rows,
                            metadata,
                            f"steady gait cycle — first episode {episode_id}",
                        )
                        for filename in (
                            "gait_stage_joint_energy.png",
                            "gait_stage_joint_energy_zh.png",
                        ):
                            _plot_gait_stage_joint_energy_table(
                                os.path.join(output_dir, filename),
                                gait_energy_rows,
                                gait_joint_names,
                                "按步态阶段统计的平均关节绝对机械能\n"
                                "代价模式：{}｜第一个回合：{}".format(
                                    ENERGY_COST_MODE_NAMES_ZH.get(
                                        metadata.get("energy_cost_mode"),
                                        "未知代价模式",
                                    ),
                                    episode_id,
                                )
                                + (
                                    f"\n实验设置：{evaluation_variant}"
                                    if evaluation_variant
                                    else ""
                                ),
                            )
            for rep in representatives:
                episode_id = _to_int(rep["episode_id"])
                rows = _add_time_column(
                    grouped_steps.get(episode_id, []),
                    _to_float(metadata.get("policy_dt"), default=0.01),
                )
                if rows:
                    _plot_episode_curves(
                        os.path.join(output_dir, f"episode_{rep['label']}_curves.png"),
                        rows,
                        metadata,
                        f"{rep['label']} episode {episode_id}",
                    )
            _plot_joint_energy(
                os.path.join(output_dir, "joint_energy_contribution.png"),
                summary_rows,
                representatives,
                metadata,
            )
            plot_status = "ok"
        except ImportError as exc:
            with open(
                os.path.join(output_dir, "plot_status.txt"),
                "w",
                encoding="utf-8",
            ) as status_file:
                status_file.write(f"matplotlib unavailable: {exc}\n")
            plot_status = "unavailable"

    _write_report(
        os.path.join(output_dir, "report.md"),
        eval_dir,
        output_dir,
        metadata,
        summary_rows,
        representatives,
        plot_status=plot_status,
    )
    return {
        "report_dir": output_dir,
        "summary_csv": os.path.join(output_dir, "summary.csv"),
        "report_md": os.path.join(output_dir, "report.md"),
        "representatives": representatives,
    }
