"""Computation time of the exact IP model of experiment 4 on a small J30 sample.

Sample: one project per RF x RS cell with RS < 1.0 (NC = 1.5, instance 1), 12 projects.
Model: resource_scheduler_optimizer (PuLP/CBC) with time windows and an LFT upper
bound, as in my_Experiment4_optimal.py. Each project runs in its own process with a
time limit, so one hard project cannot block the rest.

Output: Summary_Files/Experiment_3_j30_study/fbi/exp4_sample_timing.csv
"""

from __future__ import annotations

import csv
import multiprocessing as mp
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

TIME_LIMIT_S = 600
PARAMETER_NUMBERS = [1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15]  # NC 1.5, RS < 1.0
OUT_FILE = PROJECT_ROOT / "Summary_Files" / "Experiment_3_j30_study" / "fbi" / "exp4_sample_timing.csv"


def _solve(path: str, queue) -> None:
    import contextlib
    import io

    from Python_Experiments.my_Experiment3_j30_study import _load_project
    from Python_Modules.my_RCP_Scheduler import resource_scheduler_optimizer

    activity_set, names, deps, rr, ra = _load_project(Path(path))
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        makespan, _, _ = resource_scheduler_optimizer(
            names, activity_set, deps, ra, rr, priority_rule="LFT", time_window=True
        )
    queue.put((makespan, time.perf_counter() - t0))


def main() -> None:
    from Python_Experiments.my_Experiment3_j30_analysis import parameters, read_optima

    optima = read_optima()
    data_dir = PROJECT_ROOT / "Example_Files" / "j30_rcp"
    rows = []
    for p in PARAMETER_NUMBERS:
        instance = f"j30{p}_1"
        queue = mp.Queue()
        proc = mp.Process(target=_solve, args=(str(data_dir / f"{instance}.rcp"), queue))
        t0 = time.perf_counter()
        proc.start()
        proc.join(TIME_LIMIT_S)
        if proc.is_alive():
            proc.terminate()
            proc.join()
            makespan, seconds, status = "", TIME_LIMIT_S, "time limit"
        else:
            makespan, seconds = queue.get()
            makespan, status = round(makespan), "solved"
        nc, rf, rs = parameters(p)
        rows.append({"instance": instance, "nc": nc, "rf": rf, "rs": rs, "opt": optima[instance],
                     "ip_makespan": makespan, "ip_time_s": round(seconds, 2), "status": status})
        print(rows[-1], flush=True)

    with OUT_FILE.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved to {OUT_FILE}")


if __name__ == "__main__":
    main()
