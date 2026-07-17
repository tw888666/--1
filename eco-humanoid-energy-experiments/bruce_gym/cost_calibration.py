# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2026 ECO Authors. All rights reserved.

"""Pure-Python statistics for training-distribution cost calibration."""

from __future__ import annotations

import statistics
from typing import Iterable, Mapping


def evenly_spaced_indices(total_count: int, sample_count: int) -> list[int]:
    """Return unique indices spanning a fixed environment cohort."""

    if total_count <= 0:
        raise ValueError("total_count must be positive.")
    if not 0 < sample_count <= total_count:
        raise ValueError(
            "sample_count must be positive and no greater than total_count."
        )
    if sample_count == 1:
        return [0]
    scale = (total_count - 1) / (sample_count - 1)
    return [round(index * scale) for index in range(sample_count)]


def _describe(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "population_std": None,
            "minimum": None,
            "maximum": None,
        }
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "population_std": statistics.pstdev(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def summarize_episode_costs(
    episode_rows: Iterable[Mapping[str, object]],
    limit_fraction: float = 0.95,
) -> dict[str, object]:
    """Summarize complete-episode costs and derive a provisional cost limit.

    The all-episode mean matches the population consumed by
    ``Train/mean_cost1``. A success-only summary is also reported so falls
    cannot silently make the calibrated cost appear better.
    """

    if not 0.0 < limit_fraction <= 1.0:
        raise ValueError("limit_fraction must be in the interval (0, 1].")

    rows = list(episode_rows)
    if not rows:
        raise ValueError("At least one complete episode is required.")

    all_costs = [float(row["cost1"]) for row in rows]
    success_costs = [
        float(row["cost1"])
        for row in rows
        if str(row.get("episode_outcome", "")) == "success"
    ]
    all_stats = _describe(all_costs)
    success_stats = _describe(success_costs)
    all_mean = all_stats["mean"]
    success_mean = success_stats["mean"]
    provisional_all_limit = (
        float(all_mean) * limit_fraction if all_mean is not None else None
    )
    has_successes = bool(success_costs)
    if not has_successes:
        recommendation_status = "invalid_no_successful_episodes"
    elif float(all_mean) <= 0.0 or float(success_mean) <= 0.0:
        recommendation_status = "invalid_non_positive_mean_cost"
    elif len(success_costs) < len(rows):
        recommendation_status = "review_required_falls_present"
    else:
        recommendation_status = "valid_all_episodes_successful"

    return {
        "episode_count": len(rows),
        "success_count": len(success_costs),
        "fall_count": len(rows) - len(success_costs),
        "success_rate": len(success_costs) / len(rows),
        "cost1_all_episodes": all_stats,
        "cost1_success_episodes": success_stats,
        "limit_fraction": limit_fraction,
        "provisional_cost_limit1_all_episodes": provisional_all_limit,
        "recommended_cost_limit1_all_episodes": (
            provisional_all_limit
            if has_successes and recommendation_status != "invalid_non_positive_mean_cost"
            else None
        ),
        "recommended_cost_limit1_success_episodes": (
            float(success_mean) * limit_fraction
            if success_mean is not None
            and recommendation_status != "invalid_non_positive_mean_cost"
            else None
        ),
        "recommendation_status": recommendation_status,
        "recommendation_semantics": (
            "The all-episode recommendation matches Train/mean_cost1. "
            "It is invalid when no successful episodes are observed, when "
            "the all-episode or success-only mean cost is non-positive, and "
            "requires review when falls are present. Repeat across calibration "
            "seeds before formal training."
        ),
    }
