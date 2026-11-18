"""Part A analysis of the J30 study: deviation from the optimum per NC/RF/RS.

Input:
  Summary_Files/Experiment_3_j30_study/j30_all_results.csv  (my_Experiment3_j30_study.py)
  Example_Files/j30opt.sm                                     (PSPLIB optimal makespans)

Output (Summary_Files/Experiment_3_j30_study/analysis/):
  rules_vs_optimum.csv, best_rule_per_rs.csv, factor_effects.csv,
  heatmap_rf_rs.csv, decomposition_rs.csv, summary.txt,
  fig1_deviation_per_parameter.png, fig2_heatmap_rf_rs.png,
  fig3_decomposition_rs.png
"""

from __future__ import annotations

import csv
import re
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "Summary_Files" / "Experiment_3_j30_study" / "j30_all_results.csv"
OPT_FILE = ROOT / "Example_Files" / "j30opt.sm"
OUT = ROOT / "Summary_Files" / "Experiment_3_j30_study" / "analysis"

NC_VALUES = [1.5, 1.8, 2.1]
RF_VALUES = [0.25, 0.50, 0.75, 1.00]
RS_VALUES = [0.2, 0.5, 0.7, 1.0]

INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]


def parameters(p: int) -> tuple[float, float, float]:
    """PSPLIB J30 parameter number p (1..48) -> (NC, RF, RS)."""
    i = p - 1
    return NC_VALUES[i // 16], RF_VALUES[(i // 4) % 4], RS_VALUES[i % 4]


def read_optima() -> dict[str, int]:
    optima = {}
    for line in OPT_FILE.read_text().splitlines():
        parts = line.split()
        if len(parts) == 4 and parts[0].isdigit() and parts[1].isdigit():
            optima[f"j30{parts[0]}_{parts[1]}"] = int(parts[2])
    return optima


def load():
    optima = read_optima()
    rows = []
    with RESULTS.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            p = int(re.match(r"j30(\d+)_\d+", r["instance"]).group(1))
            nc, rf, rs = parameters(p)
            opt = optima[r["instance"]]
            ms = int(r["makespan"])
            cpm = int(r["cpm_lower_bound"])
            if ms < opt or opt < cpm:
                raise ValueError(f"Inconsistent data for {r['instance']}")
            rows.append(
                {
                    "instance": r["instance"],
                    "method": f"{r['priority_rule']}-{r['sgs']}",
                    "rule": r["priority_rule"],
                    "sgs": r["sgs"],
                    "nc": nc, "rf": rf, "rs": rs,
                    "makespan": ms, "opt": opt, "cpm": cpm,
                    "dev": (ms - opt) / opt * 100,
                }
            )
    return rows


def mean(xs):
    xs = list(xs)
    return statistics.mean(xs) if xs else float("nan")


def eta_squared(values_by_group: dict) -> float:
    allv = [v for vs in values_by_group.values() for v in vs]
    grand = mean(allv)
    ss_tot = sum((v - grand) ** 2 for v in allv)
    ss_b = sum(len(vs) * (mean(vs) - grand) ** 2 for vs in values_by_group.values())
    return ss_b / ss_tot if ss_tot else 0.0


def write_csv(name, rows):
    with (OUT / name).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = load()
    by_inst = defaultdict(list)
    for r in rows:
        by_inst[r["instance"]].append(r)
    inst_info = {i: rs[0] for i, rs in by_inst.items()}
    best_of = {i: min(r["makespan"] for r in rs) for i, rs in by_inst.items()}
    methods = sorted({r["method"] for r in rows})
    lines = []

    # ---------- Table 3: rules vs optimum (RS < 1.0) + avg rank ----------
    hard = {i for i, r in inst_info.items() if r["rs"] < 1.0}
    ranks = defaultdict(list)
    for i in hard:
        ordered = sorted(by_inst[i], key=lambda r: r["makespan"])
        pos = 0
        while pos < len(ordered):
            end = pos
            while end < len(ordered) and ordered[end]["makespan"] == ordered[pos]["makespan"]:
                end += 1
            avg_rank = (pos + 1 + end) / 2
            for r in ordered[pos:end]:
                ranks[r["method"]].append(avg_rank)
            pos = end

    table3 = []
    for m in methods:
        mr = [r for r in rows if r["method"] == m and r["instance"] in hard]
        table3.append(
            {
                "rule": mr[0]["rule"], "sgs": mr[0]["sgs"],
                "avg_dev_opt_pct": round(mean(r["dev"] for r in mr), 2),
                "max_dev_opt_pct": round(max(r["dev"] for r in mr), 1),
                "pct_optimal": round(100 * mean(r["dev"] == 0 for r in mr), 1),
                "avg_rank": round(mean(ranks[m]), 2),
                "avg_dev_opt_all480_pct": round(
                    mean(r["dev"] for r in rows if r["method"] == m), 2),
            }
        )
    table3.sort(key=lambda r: r["avg_dev_opt_pct"])
    write_csv("rules_vs_optimum.csv", table3)
    best_method = f"{table3[0]['rule']}-{table3[0]['sgs']}"

    bo_dev = {i: (best_of[i] - inst_info[i]["opt"]) / inst_info[i]["opt"] * 100 for i in by_inst}
    lines.append(f"Best single method (RS<1): {best_method}")
    lines.append(
        f"Best-of-26 (RS<1): avg dev {mean(bo_dev[i] for i in hard):.2f}%, "
        f"optimal {100*mean(bo_dev[i]==0 for i in hard):.1f}%")
    lines.append(
        f"Best-of-26 (all 480): avg dev {mean(bo_dev.values()):.2f}%, "
        f"optimal {100*mean(v==0 for v in bo_dev.values()):.1f}%")
    ser = mean(r["avg_dev_opt_pct"] for r in table3 if r["sgs"] == "Serial")
    par = mean(r["avg_dev_opt_pct"] for r in table3 if r["sgs"] == "Parallel")
    lines.append(f"Mean over 13 rules, RS<1: serial {ser:.2f}%, parallel {par:.2f}%")
    par_better = sum(
        1 for rule in {r["rule"] for r in table3}
        if [t for t in table3 if t["rule"] == rule and t["sgs"] == "Parallel"][0]["avg_dev_opt_pct"]
        < [t for t in table3 if t["rule"] == rule and t["sgs"] == "Serial"][0]["avg_dev_opt_pct"])
    lines.append(f"Rules where parallel beats serial (RS<1): {par_better}/13")

    # ---------- H4: RS = 1.0 ----------
    rs1 = [r for r in rows if r["rs"] == 1.0]
    lines.append(
        f"RS=1.0: share of (project, method) pairs at optimum = "
        f"{100*mean(r['dev']==0 for r in rs1):.2f}%, "
        f"opt==CPM in {100*mean(inst_info[i]['opt']==inst_info[i]['cpm'] for i in inst_info if inst_info[i]['rs']==1.0):.1f}% of projects, "
        f"max dev {max(r['dev'] for r in rs1):.2f}%")

    # ---------- Factor effects ----------
    def dev_of(method, i):
        return [r["dev"] for r in by_inst[i] if r["method"] == method][0]

    def avg_all(i):
        return mean(r["dev"] for r in by_inst[i])

    measures = {
        "best_single": lambda i: dev_of(best_method, i),
        "avg_26": avg_all,
        "best_of_26": lambda i: bo_dev[i],
    }
    factor_rows = []
    fig1_data = {}
    for factor, values, pool in (
        ("RS", RS_VALUES, list(inst_info)),
        ("RF", RF_VALUES, sorted(hard)),
        ("NC", NC_VALUES, sorted(hard)),
    ):
        key = factor.lower()
        fig1_data[factor] = {}
        for mname, fn in measures.items():
            groups = {v: [fn(i) for i in pool if inst_info[i][key] == v] for v in values}
            fig1_data[factor][mname] = [mean(groups[v]) for v in values]
            row = {"factor": factor, "measure": mname,
                   "projects": "all 480" if factor == "RS" else "RS<1 (360)"}
            for v in values:
                row[f"{factor}={v}"] = round(mean(groups[v]), 2)
            row["eta_squared"] = round(eta_squared(groups), 3)
            factor_rows.append(row)
        # eta^2 within RS<1 for RS too (fair comparison)
        if factor == "RS":
            for mname, fn in measures.items():
                groups = {v: [fn(i) for i in hard if inst_info[i]["rs"] == v] for v in RS_VALUES[:3]}
                lines.append(f"eta^2 RS (RS<1 only), {mname}: {eta_squared(groups):.3f}")
    with (OUT / "factor_effects.csv").open("w", newline="", encoding="utf-8") as f:
        keys = []
        for r in factor_rows:
            keys += [k for k in r if k not in keys]
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(factor_rows)

    # ---------- Heatmap RF x RS (best single method) ----------
    heat = []
    for rf in RF_VALUES:
        heat.append([
            mean(dev_of(best_method, i) for i in inst_info
                 if inst_info[i]["rf"] == rf and inst_info[i]["rs"] == rs)
            for rs in RS_VALUES])
    write_csv("heatmap_rf_rs.csv", [
        {"RF": rf, **{f"RS={rs}": round(heat[a][b], 2) for b, rs in enumerate(RS_VALUES)}}
        for a, rf in enumerate(RF_VALUES)])

    # ---------- Table 4: best method per RS ----------
    t4 = []
    for rs in RS_VALUES:
        pool = [i for i in inst_info if inst_info[i]["rs"] == rs]
        scores = {m: mean(dev_of(m, i) for i in pool) for m in methods}
        ordered = sorted(scores.items(), key=lambda kv: kv[1])
        ser_mean = mean(v for m, v in scores.items() if m.endswith("Serial"))
        par_mean = mean(v for m, v in scores.items() if m.endswith("Parallel"))
        ties = [m for m, v in ordered if abs(v - ordered[0][1]) < 1e-9]
        t4.append({
            "RS": rs, "best": " / ".join(ties), "best_dev": round(ordered[0][1], 2),
            "second": ordered[len(ties)][0] if len(ties) < len(ordered) else "",
            "second_dev": round(ordered[len(ties)][1], 2) if len(ties) < len(ordered) else "",
            f"{best_method}_dev": round(scores[best_method], 2),
            "serial_mean13": round(ser_mean, 2), "parallel_mean13": round(par_mean, 2),
            "best_of_26_dev": round(mean(bo_dev[i] for i in pool), 2),
            "best_of_26_pct_opt": round(100 * mean(bo_dev[i] == 0 for i in pool), 1),
        })
    write_csv("best_rule_per_rs.csv", t4)

    # best method per (RF, RS) cell, to see if the winner changes
    for rf in RF_VALUES:
        for rs in RS_VALUES[:3]:
            pool = [i for i in inst_info if inst_info[i]["rf"] == rf and inst_info[i]["rs"] == rs]
            scores = sorted(((mean(dev_of(m, i) for i in pool), m) for m in methods))
            lines.append(f"cell RF={rf} RS={rs}: best {scores[0][1]} {scores[0][0]:.2f}%, "
                         f"2nd {scores[1][1]} {scores[1][0]:.2f}%")
    for nc in NC_VALUES:
        pool = [i for i in hard if inst_info[i]["nc"] == nc]
        scores = sorted(((mean(dev_of(m, i) for i in pool), m) for m in methods))
        lines.append(f"NC={nc} (RS<1): best {scores[0][1]} {scores[0][0]:.2f}%")

    # ---------- Decomposition ----------
    dec = []
    for rs in RS_VALUES:
        pool = [i for i in inst_info if inst_info[i]["rs"] == rs]
        scar = mean((inst_info[i]["opt"] - inst_info[i]["cpm"]) / inst_info[i]["cpm"] * 100 for i in pool)
        rule_best = mean((min(r["makespan"] for r in by_inst[i] if r["method"] == best_method)
                          - inst_info[i]["opt"]) / inst_info[i]["cpm"] * 100 for i in pool)
        rule_avg = mean(mean(r["makespan"] - inst_info[i]["opt"] for r in by_inst[i])
                        / inst_info[i]["cpm"] * 100 for i in pool)
        dec.append({"RS": rs, "scarcity_cpm_to_opt_pct_of_cpm": round(scar, 2),
                    f"rule_opt_to_{best_method}_pct_of_cpm": round(rule_best, 2),
                    "rule_opt_to_avg26_pct_of_cpm": round(rule_avg, 2),
                    "share_rule_in_total_best_pct": round(100 * rule_best / (scar + rule_best), 1)
                    if scar + rule_best else 0.0})
    pool = sorted(hard)
    scar = mean((inst_info[i]["opt"] - inst_info[i]["cpm"]) / inst_info[i]["cpm"] * 100 for i in pool)
    rule_best = mean((dev_of(best_method, i) * inst_info[i]["opt"] / 100) / inst_info[i]["cpm"] * 100 for i in pool)
    lines.append(f"Decomposition RS<1: scarcity {scar:.2f}% of CPM, rule ({best_method}) "
                 f"{rule_best:.2f}% -> rule share {100*rule_best/(scar+rule_best):.1f}%")
    write_csv("decomposition_rs.csv", dec)

    # hardest cell
    worst = max(((r["dev"], r["instance"], r["method"]) for r in rows))
    lines.append(f"Worst single result: {worst}")
    (OUT / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ---------- Figure 1 ----------
    labels = {"best_single": f"Best single rule ({best_method.replace('-', ', ')})",
              "avg_26": "Average of the 26 rule/SGS combinations",
              "best_of_26": "Best of the 26 combinations per project"}
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), sharey=True)
    for ax, (factor, values) in zip(axes, (("RS", RS_VALUES), ("RF", RF_VALUES), ("NC", NC_VALUES))):
        style(ax)
        x = list(range(len(values)))
        for color, mname in zip(SERIES, ("avg_26", "best_single", "best_of_26")):
            ax.plot(x, fig1_data[factor][mname], color=color, linewidth=2,
                    marker="o", markersize=6, markeredgecolor="#fcfcfb", markeredgewidth=1.5,
                    label=labels[mname])
        ax.set_xticks(x, [str(v) for v in values])
        sub = "all 480 projects" if factor == "RS" else "projects with RS < 1.0"
        ax.set_xlabel(f"{factor}  ({sub})", color=INK2, fontsize=9)
        ax.set_title(factor, color=INK, fontsize=11, loc="left")
    axes[0].set_ylabel("Avg. deviation from optimum (%)", color=INK2, fontsize=9)
    axes[0].set_ylim(bottom=0)
    handles, labs = axes[0].get_legend_handles_labels()
    fig.legend(handles, labs, loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(OUT / "fig1_deviation_per_parameter.png", dpi=200, facecolor="#fcfcfb")
    plt.close(fig)

    # ---------- Figure 2 ----------
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list(
        "blue", ["#f0efec", "#9ec5f4", "#3987e5", "#1c5cab", "#0d366b"])
    fig, ax = plt.subplots(figsize=(6, 4.6))
    vmax = max(max(r) for r in heat)
    im = ax.imshow(heat, cmap=cmap, vmin=0, vmax=vmax, origin="lower", aspect="auto")
    ax.set_xticks(range(4), [str(v) for v in RS_VALUES])
    ax.set_yticks(range(4), [f"{v:.2f}" for v in RF_VALUES])
    ax.set_xlabel("RS (resource strength, low = scarce)", color=INK2, fontsize=9)
    ax.set_ylabel("RF (resource factor)", color=INK2, fontsize=9)
    ax.set_title(f"Avg. deviation from optimum (%) - {best_method.replace('-', ', ')}",
                 color=INK, fontsize=10, loc="left")
    for a in range(4):
        for b in range(4):
            v = heat[a][b]
            ax.text(b, a, f"{v:.1f}", ha="center", va="center", fontsize=10,
                    color="#fcfcfb" if v > 0.55 * vmax else INK)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(colors=INK2, length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=INK2, labelsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_heatmap_rf_rs.png", dpi=200, facecolor="#fcfcfb")
    plt.close(fig)

    # ---------- Figure 3 ----------
    fig, ax = plt.subplots(figsize=(7, 4.4))
    style(ax)
    x = list(range(len(RS_VALUES)))
    scar_v = [d["scarcity_cpm_to_opt_pct_of_cpm"] for d in dec]
    rule_v = [d[f"rule_opt_to_{best_method}_pct_of_cpm"] for d in dec]
    ax.bar(x, scar_v, width=0.55, color=SERIES[0], edgecolor="#fcfcfb", linewidth=2,
           label="CPM -> optimum (resource scarcity, unavoidable)")
    ax.bar(x, rule_v, width=0.55, bottom=scar_v, color=SERIES[1], edgecolor="#fcfcfb",
           linewidth=2, label=f"Optimum -> {best_method.replace('-', ', ')} (rule, improvable)")
    for xi, s, r in zip(x, scar_v, rule_v):
        ax.text(xi, s + r + 0.6, f"{s + r:.1f}", ha="center", color=INK, fontsize=9)
    ax.set_xticks(x, [str(v) for v in RS_VALUES])
    ax.set_xlabel("RS (resource strength)", color=INK2, fontsize=9)
    ax.set_ylabel("Makespan above CPM duration (% of CPM)", color=INK2, fontsize=9)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "fig3_decomposition_rs.png", dpi=200, facecolor="#fcfcfb")
    plt.close(fig)

    print("\n".join(lines))
    print(f"\nSaved to {OUT}")


if __name__ == "__main__":
    main()
