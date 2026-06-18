# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2026 ECO Authors. All rights reserved.

"""Pure-Python statistics for training-distribution cost calibration."""

from __future__ import annotations

import statistics
from typing import Iterable, Mapping


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

    return {
        "episode_count": len(rows),
        "success_count": len(success_costs),
        "fall_count": len(rows) - len(success_costs),
        "success_rate": len(success_costs) / len(rows),
        "cost1_all_episodes": all_stats,
        "cost1_success_episodes": success_stats,
        "limit_fraction": limit_fraction,
        "recommended_cost_limit1_all_episodes": (
            float(all_mean) * limit_fraction if all_mean is not None else None
        ),
        "recommended_cost_limit1_success_episodes": (
            float(success_mean) * limit_fraction
            if success_mean is not None
            else None
        ),
        "recommendation_semantics": (
            "The all-episode recommendation matches Train/mean_cost1. "
            "Treat it as provisional until repeated across calibration seeds."
        ),
    }
