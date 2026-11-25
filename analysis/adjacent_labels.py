"""Compare nearby catalogue labels with held-out predictions and matched controls."""

from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Callable, Iterable

LABELS = ("Alpha", "Beta", "Beta-Gamma")
WINDOWS = (1, 2)


@dataclass(frozen=True)
class Observation:
    number: int
    observation_date: date
    true_label: str
    predicted_label: str
    correct: bool
    high_confidence_error: bool
    instrument: str


def parse_bool(value: str) -> bool:
    return value.strip().lower() == "true"


def read_observations(path: Path) -> list[Observation]:
    observations: list[Observation] = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            observations.append(
                Observation(
                    number=int(row["number"]),
                    observation_date=date.fromisoformat(row["date_only"]),
                    true_label=row["true_label"],
                    predicted_label=row["pred_label"],
                    correct=parse_bool(row["correct"]),
                    high_confidence_error=parse_bool(row["is_high_conf_error"]),
                    instrument=row["instrument"],
                )
            )
    return observations


def build_timelines(
    observations: Iterable[Observation],
) -> dict[int, list[tuple[date, str]]]:
    timelines: dict[int, list[tuple[date, str]]] = defaultdict(list)
    for observation in observations:
        timelines[observation.number].append((observation.observation_date, observation.true_label))
    return timelines


def temporal_coverage(
    observation: Observation,
    timeline: list[tuple[date, str]],
    window_days: int,
) -> int:
    count = sum(0 < abs((other_date - observation.observation_date).days) <= window_days for other_date, _ in timeline)
    return min(count, 2)


def candidate_occurs_nearby(
    observation: Observation,
    candidate_label: str,
    timeline: list[tuple[date, str]],
    window_days: int,
) -> bool:
    return any(
        other_label == candidate_label and 0 < abs((other_date - observation.observation_date).days) <= window_days
        for other_date, other_label in timeline
    )


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = quantile * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def bootstrap_region_clusters(
    region_values: list[tuple[int, int, float, float]],
    replicates: int,
    seed: int,
) -> dict[str, tuple[float, float]]:
    """Bootstrap observed and standardized expected matches by NOAA region."""
    rng = random.Random(seed)
    n_regions = len(region_values)
    observed_samples: list[float] = []
    control_samples: list[float] = []
    difference_samples: list[float] = []
    risk_ratio_samples: list[float] = []
    reassigned_samples: list[float] = []
    reassigned_difference_samples: list[float] = []
    reassigned_risk_ratio_samples: list[float] = []

    for _ in range(replicates):
        sampled = rng.choices(region_values, k=n_regions)
        n = sum(item[0] for item in sampled)
        matches = sum(item[1] for item in sampled)
        expected_control_matches = sum(item[2] for item in sampled)
        expected_reassigned_matches = sum(item[3] for item in sampled)
        observed = matches / n
        control = expected_control_matches / n
        reassigned = expected_reassigned_matches / n
        observed_samples.append(observed)
        control_samples.append(control)
        difference_samples.append(observed - control)
        risk_ratio_samples.append(observed / control)
        reassigned_samples.append(reassigned)
        reassigned_difference_samples.append(observed - reassigned)
        reassigned_risk_ratio_samples.append(observed / reassigned)

    samples = {
        "fraction": observed_samples,
        "control_fraction": control_samples,
        "difference": difference_samples,
        "risk_ratio": risk_ratio_samples,
        "reassigned_fraction": reassigned_samples,
        "reassigned_difference": reassigned_difference_samples,
        "reassigned_risk_ratio": reassigned_risk_ratio_samples,
    }
    return {key: (percentile(sample, 0.025), percentile(sample, 0.975)) for key, sample in samples.items()}


def subset_definitions() -> dict[str, Callable[[Observation], bool]]:
    return {
        "all_errors": lambda row: not row.correct,
        "high_confidence_errors": lambda row: row.high_confidence_error,
        "alpha_beta_errors": lambda row: not row.correct and {row.true_label, row.predicted_label} == {"Alpha", "Beta"},
        "beta_beta_gamma_errors": lambda row: (
            not row.correct and {row.true_label, row.predicted_label} == {"Beta", "Beta-Gamma"}
        ),
        "catalogue_beta_errors": lambda row: not row.correct and row.true_label == "Beta",
        "catalogue_beta_high_confidence_errors": lambda row: row.high_confidence_error and row.true_label == "Beta",
    }


def analyse_window(
    observations: list[Observation],
    timelines: dict[int, list[tuple[date, str]]],
    window_days: int,
    replicates: int,
    seed: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    coverage = {id(row): temporal_coverage(row, timelines[row.number], window_days) for row in observations}

    # Each correctly classified observation supplies a control outcome for
    # both possible alternative labels.
    control_counts: dict[tuple[str, str, str, int], list[int]] = defaultdict(lambda: [0, 0])
    for row in observations:
        if not row.correct:
            continue
        for candidate_label in LABELS:
            if candidate_label == row.true_label:
                continue
            key = (
                row.true_label,
                candidate_label,
                row.instrument,
                coverage[id(row)],
            )
            control_counts[key][1] += 1
            control_counts[key][0] += int(
                candidate_occurs_nearby(
                    row,
                    candidate_label,
                    timelines[row.number],
                    window_days,
                )
            )

    control_rates = {key: matches / total for key, (matches, total) in control_counts.items()}

    summary_rows: list[dict[str, object]] = []
    for subset_index, (subset_name, selector) in enumerate(subset_definitions().items()):
        selected = [row for row in observations if selector(row)]
        reassignment_counts: dict[tuple[str, str, int], dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for row in selected:
            reassignment_key = (
                row.true_label,
                row.instrument,
                coverage[id(row)],
            )
            reassignment_counts[reassignment_key][row.predicted_label] += 1

        per_region: dict[int, list[float]] = defaultdict(lambda: [0, 0, 0.0, 0.0])
        matched = 0
        expected_control_matches = 0.0
        expected_reassigned_matches = 0.0

        for row in selected:
            key = (
                row.true_label,
                row.predicted_label,
                row.instrument,
                coverage[id(row)],
            )
            if key not in control_rates:
                raise ValueError(f"No matched controls for stratum {key}")
            is_match = candidate_occurs_nearby(
                row,
                row.predicted_label,
                timelines[row.number],
                window_days,
            )
            matched += int(is_match)
            expected_control_matches += control_rates[key]
            reassignment_key = (
                row.true_label,
                row.instrument,
                coverage[id(row)],
            )
            candidate_counts = reassignment_counts[reassignment_key]
            candidate_total = sum(candidate_counts.values())
            reassigned_rate = sum(
                candidate_count
                / candidate_total
                * candidate_occurs_nearby(
                    row,
                    candidate_label,
                    timelines[row.number],
                    window_days,
                )
                for candidate_label, candidate_count in candidate_counts.items()
            )
            expected_reassigned_matches += reassigned_rate
            per_region[row.number][0] += 1
            per_region[row.number][1] += int(is_match)
            per_region[row.number][2] += control_rates[key]
            per_region[row.number][3] += reassigned_rate

        n = len(selected)
        fraction = matched / n
        control_fraction = expected_control_matches / n
        reassigned_fraction = expected_reassigned_matches / n
        intervals = bootstrap_region_clusters(
            [tuple(values) for values in per_region.values()],
            replicates=replicates,
            seed=seed + 100 * window_days + subset_index,
        )
        summary_rows.append(
            {
                "subset": subset_name,
                "window_days": window_days,
                "n_errors": n,
                "n_regions": len(per_region),
                "n_predicted_label_nearby": matched,
                "fraction": fraction,
                "fraction_ci_low": intervals["fraction"][0],
                "fraction_ci_high": intervals["fraction"][1],
                "standardized_control_fraction": control_fraction,
                "control_ci_low": intervals["control_fraction"][0],
                "control_ci_high": intervals["control_fraction"][1],
                "difference": fraction - control_fraction,
                "difference_ci_low": intervals["difference"][0],
                "difference_ci_high": intervals["difference"][1],
                "risk_ratio": fraction / control_fraction,
                "risk_ratio_ci_low": intervals["risk_ratio"][0],
                "risk_ratio_ci_high": intervals["risk_ratio"][1],
                "reassigned_prediction_fraction": reassigned_fraction,
                "reassigned_ci_low": intervals["reassigned_fraction"][0],
                "reassigned_ci_high": intervals["reassigned_fraction"][1],
                "difference_from_reassigned": fraction - reassigned_fraction,
                "difference_from_reassigned_ci_low": intervals["reassigned_difference"][0],
                "difference_from_reassigned_ci_high": intervals["reassigned_difference"][1],
                "risk_ratio_to_reassigned": fraction / reassigned_fraction,
                "risk_ratio_to_reassigned_ci_low": intervals["reassigned_risk_ratio"][0],
                "risk_ratio_to_reassigned_ci_high": intervals["reassigned_risk_ratio"][1],
                "bootstrap_replicates": replicates,
                "bootstrap_seed": seed,
            }
        )

    pair_rows: list[dict[str, object]] = []
    errors = [row for row in observations if not row.correct]
    for true_label in LABELS:
        for predicted_label in LABELS:
            selected = [
                row for row in errors if row.true_label == true_label and row.predicted_label == predicted_label
            ]
            if not selected:
                continue
            matched = 0
            expected_control_matches = 0.0
            for row in selected:
                key = (
                    row.true_label,
                    row.predicted_label,
                    row.instrument,
                    coverage[id(row)],
                )
                matched += int(
                    candidate_occurs_nearby(
                        row,
                        row.predicted_label,
                        timelines[row.number],
                        window_days,
                    )
                )
                expected_control_matches += control_rates[key]
            n = len(selected)
            pair_rows.append(
                {
                    "true_label": true_label,
                    "predicted_label": predicted_label,
                    "window_days": window_days,
                    "n_errors": n,
                    "n_predicted_label_nearby": matched,
                    "fraction": matched / n,
                    "standardized_control_fraction": expected_control_matches / n,
                    "difference": (matched - expected_control_matches) / n,
                    "risk_ratio": matched / expected_control_matches,
                }
            )
    return summary_rows, pair_rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="label_noise_diagnostics.csv written by label_noise_analysis.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("adjacent_label_summary.csv"),
    )
    parser.add_argument(
        "--pair-output",
        type=Path,
        default=Path("adjacent_label_pairs.csv"),
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    observations = read_observations(args.input)
    timelines = build_timelines(observations)
    if len(observations) != 26_015:
        raise ValueError(f"Expected 26,015 observations, found {len(observations):,}")
    if len(timelines) != 4_817:
        raise ValueError(f"Expected 4,817 NOAA regions, found {len(timelines):,}")

    summary_rows: list[dict[str, object]] = []
    pair_rows: list[dict[str, object]] = []
    for window_days in WINDOWS:
        window_summary, window_pairs = analyse_window(
            observations,
            timelines,
            window_days=window_days,
            replicates=args.bootstrap_replicates,
            seed=args.seed,
        )
        summary_rows.extend(window_summary)
        pair_rows.extend(window_pairs)

    write_csv(args.summary_output, summary_rows)
    write_csv(args.pair_output, pair_rows)

    for row in summary_rows:
        print(
            f"{row['subset']}, ±{row['window_days']} d: "
            f"{row['n_predicted_label_nearby']}/{row['n_errors']} "
            f"({100 * float(row['fraction']):.1f}%), "
            f"matched control {100 * float(row['standardized_control_fraction']):.1f}%, "
            f"difference {100 * float(row['difference']):+.1f} pp "
            f"[{100 * float(row['difference_ci_low']):+.1f}, "
            f"{100 * float(row['difference_ci_high']):+.1f}]; "
            f"reassigned-prediction expectation "
            f"{100 * float(row['reassigned_prediction_fraction']):.1f}%, "
            f"difference "
            f"{100 * float(row['difference_from_reassigned']):+.1f} pp"
        )


if __name__ == "__main__":
    main()
