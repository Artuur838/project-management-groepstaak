"""Evaluate ProPy Experiment 3 priority rules on all PSPLIB j30 instances."""

from __future__ import annotations

import csv
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Python_Modules.my_CPM_Scheduler import (  # noqa: E402
    calculate_critical_path,
    create_activity_network_dependencies,
)
from Python_Modules.my_Project_Data import (  # noqa: E402
    add_dummy_nodes_to_project_data,
    read_patterson_file,
)
from Python_Modules.my_RCP_Scheduler import (  # noqa: E402
    create_priority_list,
    parallel_sgs,
    serial_sgs,
)


PRIORITY_RULES = [
    "SPT",
    "LPT",
    "MIS",
    "MTS",
    "LNRJ",
    "GRPW",
    "EST",
    "EFT",
    "LST",
    "LFT",
    "MSLK",
    "GRWC",
    "GCRWC",
]
SCHEDULE_GENERATION_SCHEMES = {
    "Serial": serial_sgs,
    "Parallel": parallel_sgs,
}
EXPECTED_INSTANCE_COUNT = 480
DATA_DIRECTORY = PROJECT_ROOT / "Example_Files" / "j30_rcp"
OUTPUT_DIRECTORY = PROJECT_ROOT / "Summary_Files" / "Experiment_3_j30_study"


def _load_project(instance_path: Path):
    (
        _,
        activity_set,
        successors_list,
        activity_names,
        resource_requirements,
        resource_availability,
        _,
    ) = read_patterson_file(str(instance_path))
    add_dummy_nodes_to_project_data(
        activity_set,
        activity_names,
        sul=successors_list,
        rr=resource_requirements,
    )
    dependencies = create_activity_network_dependencies(
        {}, successors_list, activity_names, debugInfo=False
    )
    return (
        activity_set,
        activity_names,
        dependencies,
        resource_requirements,
        resource_availability,
    )


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _average_ranks(rows: list[dict]) -> dict[tuple[str, str], float]:
    makespans_by_instance = defaultdict(list)
    for row in rows:
        makespans_by_instance[row["instance"]].append(row)

    rank_sums = defaultdict(float)
    rank_counts = defaultdict(int)
    for instance_rows in makespans_by_instance.values():
        ordered = sorted(instance_rows, key=lambda row: row["makespan"])
        first_rank = 1
        index = 0
        while index < len(ordered):
            end = index + 1
            while (
                end < len(ordered)
                and ordered[end]["makespan"] == ordered[index]["makespan"]
            ):
                end += 1
            average_rank = (first_rank + first_rank + end - index - 1) / 2
            for row in ordered[index:end]:
                key = (row["priority_rule"], row["sgs"])
                rank_sums[key] += average_rank
                rank_counts[key] += 1
            first_rank += end - index
            index = end

    return {key: rank_sums[key] / rank_counts[key] for key in rank_sums}


def run_experiment() -> tuple[list[dict], list[dict]]:
    instance_paths = sorted(DATA_DIRECTORY.glob("j*.rcp"))
    if len(instance_paths) != EXPECTED_INSTANCE_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_INSTANCE_COUNT} PSPLIB j30 .rcp files in "
            f"{DATA_DIRECTORY}, found {len(instance_paths)}."
        )

    random.seed(0)
    results = []
    for index, instance_path in enumerate(instance_paths, start=1):
        (
            activity_set,
            activity_names,
            dependencies,
            resource_requirements,
            resource_availability,
        ) = _load_project(instance_path)
        _, cpm_duration, _, _ = calculate_critical_path(
            activity_set, dependencies, activity_names, False
        )
        if cpm_duration <= 0:
            raise ValueError(f"Invalid CPM duration for {instance_path.name}")

        for rule in PRIORITY_RULES:
            priority_list = create_priority_list(
                rule,
                activity_names,
                activity_set,
                dependencies,
                resource_requirements,
                debugInfo=False,
            )
            for sgs_name, scheduler in SCHEDULE_GENERATION_SCHEMES.items():
                schedule_result = scheduler(
                    activity_names,
                    activity_set,
                    dependencies,
                    resource_requirements,
                    resource_availability,
                    priority_list,
                    debugInfo=False,
                )
                makespan = schedule_result[0]
                results.append(
                    {
                        "instance": instance_path.stem,
                        "priority_rule": rule,
                        "sgs": sgs_name,
                        "makespan": makespan,
                        "cpm_lower_bound": cpm_duration,
                        "cpm_relative_gap_percent": (
                            (makespan - cpm_duration) / cpm_duration * 100
                        ),
                    }
                )

        if index % 40 == 0 or index == len(instance_paths):
            print(f"Processed {index}/{len(instance_paths)} instances")

    best_makespan_by_instance = {}
    for row in results:
        instance = row["instance"]
        best_makespan_by_instance[instance] = min(
            best_makespan_by_instance.get(instance, row["makespan"]),
            row["makespan"],
        )
    for row in results:
        row["best_among_tested"] = int(
            row["makespan"] == best_makespan_by_instance[row["instance"]]
        )

    ranks = _average_ranks(results)
    grouped_results = defaultdict(list)
    for row in results:
        grouped_results[(row["priority_rule"], row["sgs"])].append(row)

    summary = []
    for (rule, sgs), method_rows in grouped_results.items():
        makespans = [row["makespan"] for row in method_rows]
        gaps = [row["cpm_relative_gap_percent"] for row in method_rows]
        best_count = sum(row["best_among_tested"] for row in method_rows)
        summary.append(
            {
                "priority_rule": rule,
                "sgs": sgs,
                "mean_makespan": statistics.mean(makespans),
                "median_makespan": statistics.median(makespans),
                "stdev_makespan": statistics.stdev(makespans),
                "mean_cpm_relative_gap_percent": statistics.mean(gaps),
                "mean_rank": ranks[(rule, sgs)],
                "best_count": best_count,
                "best_share_percent": best_count / len(instance_paths) * 100,
            }
        )

    summary.sort(key=lambda row: (row["mean_rank"], row["mean_makespan"]))
    return results, summary


def main() -> None:
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    results, summary = run_experiment()

    _write_csv(
        OUTPUT_DIRECTORY / "j30_all_results.csv",
        [
            "instance",
            "priority_rule",
            "sgs",
            "makespan",
            "cpm_lower_bound",
            "cpm_relative_gap_percent",
            "best_among_tested",
        ],
        results,
    )
    _write_csv(
        OUTPUT_DIRECTORY / "j30_summary.csv",
        [
            "priority_rule",
            "sgs",
            "mean_makespan",
            "median_makespan",
            "stdev_makespan",
            "mean_cpm_relative_gap_percent",
            "mean_rank",
            "best_count",
            "best_share_percent",
        ],
        summary,
    )

    rules = PRIORITY_RULES
    x_positions = list(range(len(rules)))
    bar_width = 0.38
    fig, axis = plt.subplots(figsize=(12, 6))
    for offset, sgs in ((-bar_width / 2, "Serial"), (bar_width / 2, "Parallel")):
        means = {
            row["priority_rule"]: row["mean_cpm_relative_gap_percent"]
            for row in summary
            if row["sgs"] == sgs
        }
        axis.bar(
            [position + offset for position in x_positions],
            [means[rule] for rule in rules],
            width=bar_width,
            label=sgs,
        )

    axis.set_xticks(x_positions, rules)
    axis.set_xlabel("Priority rule")
    axis.set_ylabel("Mean gap above CPM lower bound (%)")
    axis.set_title("Priority rule performance across PSPLIB j30")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIRECTORY / "j30_mean_cpm_gap.png", dpi=200)
    plt.close(fig)

    print("\nBest methods by average rank (smaller is better):")
    for row in summary[:10]:
        print(
            f"{row['priority_rule']:5} {row['sgs']:8} "
            f"mean rank={row['mean_rank']:.2f}, "
            f"mean makespan={row['mean_makespan']:.2f}, "
            f"best on {row['best_count']}/{EXPECTED_INSTANCE_COUNT}"
        )
    print(f"\nSaved experiment outputs to: {OUTPUT_DIRECTORY}")


if __name__ == "__main__":
    main()
