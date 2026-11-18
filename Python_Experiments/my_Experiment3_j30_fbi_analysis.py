"""Part B analysis of the J30 study: forward-backward improvement vs. the optimum.

Input:
  Summary_Files/Experiment_3_j30_study/fbi/j30_fbi_results.csv  (my_Experiment3_j30_fbi.py)
  Example_Files/j30opt.sm                                        (PSPLIB optimal makespans)

Output (Summary_Files/Experiment_3_j30_study/fbi/):
  table5_methods.csv      - table 5 of the report (360 projects with RS < 1.0)
  fbi_gain_per_rs_rf.csv  - deviation and gain per RS and RF value (H5, H6)
  fbi_control.csv         - serial vs. parallel control counts
  summary.txt
  fig4_fbi_gain_rs_rf.png
"""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Python_Experiments.my_Experiment3_j30_analysis import (  # noqa: E402
    INK,
    INK2,
    RF_VALUES,
    RS_VALUES,
    SERIES,
    mean,
    parameters,
    read_optima,
    style,
)

FBI_DIR = PROJECT_ROOT / "Summary_Files" / "Experiment_3_j30_study" / "fbi"
RESULTS = FBI_DIR / "j30_fbi_results.csv"

# (column in the results file, label in the report)
METHODS = [
    ("baseline_makespan", "Best single rule (LFT-Parallel)"),
    ("best26_makespan", "Best of 26 combinations"),
    ("fbi_serial_makespan", "Best of 26 + FBI serial (our improvement)"),
    ("fbi_parallel_makespan", "Best of 26 + FBI parallel (control)"),
    ("fbi_serial_all26_makespan", "FBI serial on all 26, best kept"),
    ("fbi_parallel_all26_makespan", "FBI parallel on all 26, best kept"),
]


def load() -> list[dict]:
    optima = read_optima()
    rows = []
    with RESULTS.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            p = int(re.match(r"j30(\d+)_\d+", r["instance"]).group(1))
            nc, rf, rs = parameters(p)
            row = {"instance": r["instance"], "nc": nc, "rf": rf, "rs": rs,
                   "opt": optima[r["instance"]], "cpm": int(r["cpm"])}
            for key, value in r.items():
                if key in row or key == "instance":
                    continue
                if key.endswith("_s"):
                    row[key] = float(value)
                elif key == "best26_method":
                    row[key] = value
                else:
                    row[key] = int(value)
            for col, _ in METHODS:
                if row[col] < row["opt"]:
                    raise ValueError(f"{r['instance']}: {col} below the optimum")
            if row["fbi_serial_makespan"] > row["best26_makespan"]:
                raise ValueError(f"{r['instance']}: serial FBI worse than its start")
            rows.append(row)
    return rows


def dev(row, col):
    return (row[col] - row["opt"]) / row["opt"] * 100


def method_time_ms(row, col):
    """Average computation time per project of each method (cumulative)."""
    t26 = row["time_26_s"]
    times = {
        "baseline_makespan": t26 / 26,  # approx.: one of the 26 schedules
        "best26_makespan": t26,
        "fbi_serial_makespan": t26 + row["fbi_serial_time_s"],
        "fbi_parallel_makespan": t26 + row["fbi_parallel_time_s"],
        "fbi_serial_all26_makespan": t26 + row["fbi_serial_all26_time_s"],
        "fbi_parallel_all26_makespan": t26 + row["fbi_parallel_all26_time_s"],
    }
    return times[col] * 1000


def write_csv(name, rows):
    with (FBI_DIR / name).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    rows = load()
    hard = [r for r in rows if r["rs"] < 1.0]
    lines = [f"Projects: {len(rows)} (RS < 1.0: {len(hard)})"]

    # ---------- Table 5 ----------
    t5 = []
    for col, label in METHODS:
        t5.append({
            "method": label,
            "avg_dev_from_opt_pct": round(mean(dev(r, col) for r in hard), 2),
            "pct_projects_optimal": round(100 * mean(r[col] == r["opt"] for r in hard), 1),
            "avg_time_per_project_ms": round(mean(method_time_ms(r, col) for r in hard), 1),
            "avg_dev_all480_pct": round(mean(dev(r, col) for r in rows), 2),
        })
    write_csv("table5_methods.csv", t5)
    lines.append("\nTable 5 (RS < 1.0):")
    for t in t5:
        lines.append(f"  {t['method']:45} dev {t['avg_dev_from_opt_pct']:5.2f}%  "
                     f"opt {t['pct_projects_optimal']:5.1f}%  time {t['avg_time_per_project_ms']:7.1f} ms")

    # ---------- Gain per RS / RF (H5, H6) ----------
    gain_rows = []
    fig4 = {"RS": {}, "RF": {}}
    for factor, key, values, pool_all in (("RS", "rs", RS_VALUES, rows), ("RF", "rf", RF_VALUES, hard)):
        for v in values:
            pool = [r for r in pool_all if r[key] == v]
            d26 = mean(dev(r, "best26_makespan") for r in pool)
            ds = mean(dev(r, "fbi_serial_makespan") for r in pool)
            dp = mean(dev(r, "fbi_parallel_makespan") for r in pool)
            gain_rows.append({
                "factor": factor, "value": v, "projects": len(pool),
                "dev_best26": round(d26, 2), "dev_fbi_serial": round(ds, 2),
                "dev_fbi_parallel": round(dp, 2),
                "gain_serial_pp": round(d26 - ds, 2), "gain_parallel_pp": round(d26 - dp, 2),
                "pct_improved_serial": round(100 * mean(r["fbi_serial_makespan"] < r["best26_makespan"] for r in pool), 1),
                "pct_improved_parallel": round(100 * mean(r["fbi_parallel_makespan"] < r["best26_makespan"] for r in pool), 1),
                "pct_opt_fbi_serial": round(100 * mean(r["fbi_serial_makespan"] == r["opt"] for r in pool), 1),
            })
            fig4[factor][v] = (d26 - ds, d26 - dp)
    write_csv("fbi_gain_per_rs_rf.csv", gain_rows)

    # ---------- Control: serial vs. parallel ----------
    def count(cond, pool=hard):
        return sum(1 for r in pool if cond(r))

    control = [
        ("serial FBI improves best-of-26", count(lambda r: r["fbi_serial_makespan"] < r["best26_makespan"])),
        ("parallel FBI improves best-of-26", count(lambda r: r["fbi_parallel_makespan"] < r["best26_makespan"])),
        ("serial better than parallel", count(lambda r: r["fbi_serial_makespan"] < r["fbi_parallel_makespan"])),
        ("parallel better than serial", count(lambda r: r["fbi_parallel_makespan"] < r["fbi_serial_makespan"])),
        ("serial = parallel", count(lambda r: r["fbi_serial_makespan"] == r["fbi_parallel_makespan"])),
        ("parallel: first iteration (unguarded) longer than S",
         count(lambda r: r["fbi_parallel_first_iteration_makespan"] > r["best26_makespan"])),
        ("parallel: projects with >= 1 lengthening pass", count(lambda r: r["fbi_parallel_worse_passes"] > 0)),
        ("serial: projects with >= 1 lengthening pass (must be 0)",
         count(lambda r: r["fbi_serial_worse_passes"] > 0, rows)),
        ("serial all-26: lengthening passes (must be 0)", sum(r["fbi_serial_all26_worse_passes"] for r in rows)),
        ("parallel all-26: lengthening passes", sum(r["fbi_parallel_all26_worse_passes"] for r in rows)),
        ("serial: max improving iterations", max(r["fbi_serial_iterations"] for r in rows)),
        ("parallel: max improving iterations", max(r["fbi_parallel_iterations"] for r in rows)),
    ]
    write_csv("fbi_control.csv", [{"check": c, "value": v,
                                   "pool": "all 480" if "must be 0" in c or "max" in c else "RS<1 (360)"}
                                  for c, v in control])
    lines.append("\nControl serial vs. parallel:")
    lines += [f"  {c}: {v}" for c, v in control]
    lines.append("\nGain per RS / RF (pp deviation, best-of-26 -> FBI):")
    lines += [f"  {g['factor']}={g['value']}: serial {g['gain_serial_pp']:.2f}, "
              f"parallel {g['gain_parallel_pp']:.2f}, improved {g['pct_improved_serial']}% / "
              f"{g['pct_improved_parallel']}%" for g in gain_rows]
    (FBI_DIR / "summary.txt").write_text("\n".join(lines), encoding="utf-8")

    # ---------- Figure 4 ----------
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True,
                             gridspec_kw={"width_ratios": [4, 4]})
    width = 0.36
    for ax, (factor, values) in zip(axes, (("RS", RS_VALUES), ("RF", RF_VALUES))):
        style(ax)
        x = list(range(len(values)))
        for k, (color, label) in enumerate(zip(SERIES[:2], ("FBI serial (our improvement)",
                                                           "FBI parallel (control)"))):
            heights = [fig4[factor][v][k] for v in values]
            xs = [xi + (k - 0.5) * (width + 0.03) for xi in x]
            ax.bar(xs, heights, width=width, color=color, edgecolor="#fcfcfb", linewidth=2, label=label)
            for xi, h in zip(xs, heights):
                ax.text(xi, h + 0.03, f"{h:.2f}", ha="center", va="bottom", color=INK2, fontsize=8)
        ax.set_xticks(x, [str(v) for v in values])
        sub = "all 480 projects" if factor == "RS" else "projects with RS < 1.0"
        ax.set_xlabel(f"{factor}  ({sub})", color=INK2, fontsize=9)
        ax.set_title(factor, color=INK, fontsize=11, loc="left")
    axes[0].set_ylabel("Reduction of avg. deviation from optimum\n(percentage points vs. best of 26)",
                       color=INK2, fontsize=9)
    axes[0].set_ylim(bottom=0)
    handles, labs = axes[0].get_legend_handles_labels()
    fig.legend(handles, labs, loc="lower center", ncol=2, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(FBI_DIR / "fig4_fbi_gain_rs_rf.png", dpi=200, facecolor="#fcfcfb")
    plt.close(fig)

    print("\n".join(lines))
    print(f"\nSaved to {FBI_DIR}")


if __name__ == "__main__":
    main()
