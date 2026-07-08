# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2026 ECO Authors. All rights reserved.

import argparse

from bruce_gym.eval_review import DEFAULT_METRIC, generate_review


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate an offline review report from evaluate_energy outputs."
    )
    parser.add_argument(
        "--eval_dir",
        required=True,
        help="Directory containing metadata.json, episode_summary.csv, and step_timeseries.csv.",
    )
    parser.add_argument(
        "--report_dir",
        help="Output directory for the review report. Defaults to <eval_dir>/eval_report.",
    )
    parser.add_argument(
        "--metric",
        default=DEFAULT_METRIC,
        help=f"Representative selection metric. Defaults to {DEFAULT_METRIC}.",
    )
    parser.add_argument(
        "--no_plots",
        action="store_true",
        default=False,
        help="Skip PNG plot generation.",
    )
    parser.add_argument(
        "--no_episode_csvs",
        action="store_true",
        default=False,
        help="Skip writing one CSV per episode.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    result = generate_review(
        eval_dir=args.eval_dir,
        output_dir=args.report_dir,
        metric=args.metric,
        make_plots=not args.no_plots,
        write_episode_csvs=not args.no_episode_csvs,
    )
    print(f"Wrote evaluation review report to: {result['report_dir']}")


if __name__ == "__main__":
    main()
