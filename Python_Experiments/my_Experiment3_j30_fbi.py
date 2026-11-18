"""Part B of the J30 study: forward-backward improvement (FBI / justification).

For every PSPLIB J30 project:
  1. Build the 26 rule/SGS schedules of experiment 3 (13 rules x serial/parallel).
  2. S = the best of the 26 (first in fixed order on a tie).
  3. Improve S with forward-backward improvement in two variants:
       - serial   (main method): backward_serial_sgs + serial_sgs.
                  Each pass can never lengthen the schedule (Valls et al., 2005);
                  this is checked with an assertion on every pass.
       - parallel (control):     backward_parallel_sgs + parallel_sgs.
                  No such guarantee: a parallel pass always yields a non-delay
                  schedule and may lengthen it. A new schedule is only accepted
                  when it is shorter; worsening passes are counted.
  4. Extra variant: apply each FBI variant to all 26 schedules and keep the best.

Output: Summary_Files/Experiment_3_j30_study/fbi/j30_fbi_results.csv (one row per project).
Analysis: my_Experiment3_j30_fbi_analysis.py
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Python_Experiments.my_Experiment3_j30_study import (  # noqa: E402
    DATA_DIRECTORY,
    EXPECTED_INSTANCE_COUNT,
    PRIORITY_RULES,
    SCHEDULE_GENERATION_SCHEMES,
    _load_project,
)
from Python_Modules.my_CPM_Scheduler import calculate_critical_path  # noqa: E402
from Python_Modules.my_RCP_Scheduler import (  # noqa: E402
    backward_parallel_sgs,
    backward_serial_sgs,
    create_priority_list,
    parallel_sgs,
    serial_sgs,
)

OUTPUT_DIRECTORY = PROJECT_ROOT / "Summary_Files" / "Experiment_3_j30_study" / "fbi"
MAX_ITERATIONS = 50
BASELINE_METHOD = ("LFT", "Parallel")  # best single rule in part A
VARIANTS = ("serial", "parallel")


def _start_end(schedule):
    start = {a: v[0][3] for a, v in schedule.items()}
    end = {a: v[0][4] for a, v in schedule.items()}
    return start, end


def _backward(variant, project, end_times, makespan):
    """Right justification: activities on decreasing finish time in S.
    Returns (makespan, start times of the right-justified schedule)."""
    activity_set, _, deps, rr, ra = project
    priority = list(end_times.items())  # higher value = scheduled first
    if variant == "serial":
        back_ms, _, back_schedule = backward_serial_sgs(
            activity_set, deps, rr, ra, priority, makespan
        )
        back_start, _ = _start_end(back_schedule)
    else:
        back_ms, back_start, _, _ = backward_parallel_sgs(
            activity_set, deps, rr, ra, priority, makespan
        )
    return back_ms, back_start


def _forward(variant, project, start_times):
    """Left justification: activities on increasing start time.
    Returns (makespan, schedule)."""
    activity_set, names, deps, rr, ra = project
    priority = list(start_times.items())  # lower value = scheduled first
    scheduler = serial_sgs if variant == "serial" else parallel_sgs
    result = scheduler(names, activity_set, deps, rr, ra, priority)
    return result[0], result[2]


def forward_backward_improvement(variant, project, schedule, makespan):
    """Iterate backward + forward pass while the makespan decreases.

    Returns a dict with the final makespan, the number of improving iterations,
    the number of passes that produced a LONGER schedule than the current one,
    and the makespan after the first (unguarded) backward-forward iteration."""
    improving = 0
    worse_passes = 0
    first_iteration_makespan = None
    for _ in range(MAX_ITERATIONS):
        _, end = _start_end(schedule)
        back_ms, back_start = _backward(variant, project, end, makespan)
        new_ms, new_schedule = _forward(variant, project, back_start)

        worse_passes += (back_ms > makespan) + (new_ms > makespan)
        if first_iteration_makespan is None:
            first_iteration_makespan = new_ms
        if variant == "serial" and (back_ms > makespan or new_ms > makespan):
            raise AssertionError("Serial FBI pass lengthened the schedule")

        if new_ms < makespan:
            schedule, makespan = new_schedule, new_ms
            improving += 1
        else:
            break
    return {
        "makespan": makespan,
        "iterations": improving,
        "worse_passes": worse_passes,
        "first_iteration_makespan": first_iteration_makespan,
    }


def run_instance(instance_path: Path) -> dict:
    project = _load_project(instance_path)
    activity_set, names, deps, rr, ra = project
    _, cpm_duration, _, _ = calculate_critical_path(activity_set, deps, names, False)

    # Step 1: the 26 rule/SGS schedules (same order as the part A study)
    t0 = time.perf_counter()
    candidates = []
    for rule in PRIORITY_RULES:
        priority_list = create_priority_list(rule, names, activity_set, deps, rr, debugInfo=False)
        for sgs_name, scheduler in SCHEDULE_GENERATION_SCHEMES.items():
            result = scheduler(names, activity_set, deps, rr, ra, priority_list, debugInfo=False)
            candidates.append((result[0], rule, sgs_name, result[2]))
    time_26 = time.perf_counter() - t0

    # Step 2: S = best of 26 (min() keeps the first one on a tie)
    best_ms, best_rule, best_sgs, best_schedule = min(candidates, key=lambda c: c[0])
    baseline_ms = next(c[0] for c in candidates if (c[1], c[2]) == BASELINE_METHOD)

    row = {
        "instance": instance_path.stem,
        "cpm": cpm_duration,
        "baseline_makespan": baseline_ms,
        "best26_makespan": best_ms,
        "best26_method": f"{best_rule}-{best_sgs}",
        "time_26_s": time_26,
    }

    for variant in VARIANTS:
        # Step 3: FBI on S (best of 26)
        t0 = time.perf_counter()
        res = forward_backward_improvement(variant, project, best_schedule, best_ms)
        row[f"fbi_{variant}_makespan"] = res["makespan"]
        row[f"fbi_{variant}_iterations"] = res["iterations"]
        row[f"fbi_{variant}_worse_passes"] = res["worse_passes"]
        row[f"fbi_{variant}_first_iteration_makespan"] = res["first_iteration_makespan"]
        row[f"fbi_{variant}_time_s"] = time.perf_counter() - t0

        # Step 4: FBI on all 26 schedules, keep the best
        t0 = time.perf_counter()
        all_results = [
            forward_backward_improvement(variant, project, sched, ms)
            for ms, _, _, sched in candidates
        ]
        row[f"fbi_{variant}_all26_makespan"] = min(r["makespan"] for r in all_results)
        row[f"fbi_{variant}_all26_worse_passes"] = sum(r["worse_passes"] for r in all_results)
        row[f"fbi_{variant}_all26_time_s"] = time.perf_counter() - t0

    return row


def main() -> None:
    instance_paths = sorted(DATA_DIRECTORY.glob("j*.rcp"))
    if len(instance_paths) != EXPECTED_INSTANCE_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_INSTANCE_COUNT} .rcp files in {DATA_DIRECTORY}, "
            f"found {len(instance_paths)}."
        )
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    rows = []
    for index, path in enumerate(instance_paths, start=1):
        rows.append(run_instance(path))
        if index % 40 == 0 or index == len(instance_paths):
            print(f"Processed {index}/{len(instance_paths)} instances", flush=True)

    out_file = OUTPUT_DIRECTORY / "j30_fbi_results.csv"
    with out_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} rows to {out_file}")


if __name__ == "__main__":
    main()
