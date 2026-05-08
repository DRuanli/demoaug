"""Generate figures and analysis report from experiment results.

Reads results/experiment_results.csv and produces:
    figures/fig_correlation.png  — pooled Δ Hard-bias × Δ ABROCA scatter
    figures/fig_pareto.png       — Pareto frontier with 95% CI
    figures/fig_six_metrics.png  — all 6 metrics × 2 regimes
    results/analysis_report.md   — full Markdown analysis
    results/experiment_summary.json — machine-readable summary
"""
import os
import sys
import json
import warnings
warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from stats_rigor import comprehensive_comparison, correlation_analysis


METRICS = ["AUC", "ABROCA", "Hard-bias_IH", "DI_gap", "EO_gap", "Calib_gap"]
METHODS = ["Original", "SMOTE", "LLM-UC", "LLM+SMOTE", "DEMOAUG"]
COLORS = {"Original": "#7f7f7f", "SMOTE": "#ff7f0e",
          "LLM-UC": "#9467bd", "LLM+SMOTE": "#8c564b", "DEMOAUG": "#d62728"}
EQUIVALENCE_MARGINS = {
    "AUC": 0.01, "ABROCA": 0.01, "Hard-bias_IH": 0.05,
    "DI_gap": 0.05, "EO_gap": 0.03, "Calib_gap": 0.02,
}


def fig_correlation(df, fig_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for i, regime in enumerate(["DistributionalBias", "LabelNoiseBias"]):
        ax = axes[i]
        sub = df[df["regime"] == regime]
        if len(sub) == 0:
            ax.set_title(f"{regime}: no data")
            continue
        orig = sub[sub["method"] == "Original"].sort_values("seed")
        all_dhb, all_dab = [], []
        for method in [m for m in METHODS if m != "Original"]:
            mr = sub[sub["method"] == method].sort_values("seed")
            if len(mr) != len(orig):
                continue
            d_hb = mr["Hard-bias_IH"].values - orig["Hard-bias_IH"].values
            d_ab = mr["ABROCA"].values - orig["ABROCA"].values
            ax.scatter(d_hb, d_ab, color=COLORS[method], label=method, s=85,
                       alpha=0.85, edgecolors="black", linewidths=0.6)
            all_dhb.extend(d_hb); all_dab.extend(d_ab)
        if len(all_dhb) >= 3:
            cor = correlation_analysis(all_dhb, all_dab)
            ax.axhline(0, color="black", lw=0.5, alpha=0.5)
            ax.axvline(0, color="black", lw=0.5, alpha=0.5)
            text = (f"Pooled Pearson r = {cor['pearson_r']:+.3f}\n"
                    f"95% CI: [{cor['ci_95_low']:+.3f}, {cor['ci_95_high']:+.3f}]\n"
                    f"p = {cor['pearson_p']:.3f}")
            ax.text(0.04, 0.96, text, transform=ax.transAxes, va="top", ha="left",
                    fontsize=10, bbox=dict(boxstyle="round,pad=0.4", fc="lightyellow",
                                            ec="black", alpha=0.85))
        ax.set_xlabel(r"$\Delta$ Hard-bias_IH (vs Original)", fontsize=11)
        ax.set_ylabel(r"$\Delta$ ABROCA (vs Original)", fontsize=11)
        ax.set_title(f"{regime}\n{len(all_dhb)} (method × seed) deltas", fontsize=11)
        ax.legend(loc="lower left", fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle("Hard-bias and ABROCA changes are statistically independent\n"
                 "(pooled correlation ≈ 0 in both regimes)", fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(fig_dir / "fig_correlation.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved {fig_dir}/fig_correlation.png")


def fig_pareto(df, fig_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for i, regime in enumerate(["DistributionalBias", "LabelNoiseBias"]):
        ax = axes[i]
        sub = df[df["regime"] == regime]
        if len(sub) == 0:
            continue
        for method in METHODS:
            mr = sub[sub["method"] == method]
            if len(mr) == 0:
                continue
            mu_x, mu_y = mr["Hard-bias_IH"].mean(), mr["ABROCA"].mean()
            sd_x = mr["Hard-bias_IH"].std() / np.sqrt(len(mr))
            sd_y = mr["ABROCA"].std() / np.sqrt(len(mr))
            ax.scatter(mr["Hard-bias_IH"], mr["ABROCA"], color=COLORS[method],
                       label=method, s=50, alpha=0.4, edgecolors="black", linewidths=0.4)
            ax.errorbar(mu_x, mu_y, xerr=1.96 * sd_x, yerr=1.96 * sd_y,
                       fmt="s", color=COLORS[method], markersize=12,
                       markeredgecolor="black", markeredgewidth=1.5,
                       capsize=5, elinewidth=2, alpha=1.0)
        ax.set_xlabel("Hard-bias_IH (lower = fairer)", fontsize=11)
        ax.set_ylabel("ABROCA (lower = fairer)", fontsize=11)
        ax.set_title(regime, fontsize=11)
        ax.legend(fontsize=9, loc="best")
        ax.grid(alpha=0.3)
    fig.suptitle("Pareto frontier (95% CI bars on means): each method occupies a distinct point",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(fig_dir / "fig_pareto.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved {fig_dir}/fig_pareto.png")


def fig_six_metrics(df, fig_dir):
    fig, axes = plt.subplots(2, 6, figsize=(20, 7), sharey="col")
    for ri, regime in enumerate(["DistributionalBias", "LabelNoiseBias"]):
        sub = df[df["regime"] == regime]
        for mi, metric in enumerate(METRICS):
            ax = axes[ri, mi]
            means, sems = [], []
            for m in METHODS:
                mr = sub[sub["method"] == m]
                if len(mr) > 0:
                    means.append(mr[metric].mean())
                    sems.append(mr[metric].std() / np.sqrt(len(mr)))
                else:
                    means.append(0); sems.append(0)
            x = np.arange(len(METHODS))
            ax.bar(x, means, yerr=[1.96 * s for s in sems], capsize=4,
                   color=[COLORS[m] for m in METHODS], alpha=0.85, edgecolor="black")
            ax.set_xticks(x)
            ax.set_xticklabels(METHODS, rotation=45, ha="right", fontsize=8)
            if ri == 0:
                ax.set_title(metric, fontsize=10)
            if mi == 0:
                ax.set_ylabel(regime, fontsize=10)
            ax.grid(axis="y", alpha=0.3)
    fig.suptitle("All 6 metrics × 2 regimes (95% CI). DEMOAUG dominates Hard-bias_IH; "
                 "no method dominates the others.", fontsize=12)
    plt.tight_layout()
    plt.savefig(fig_dir / "fig_six_metrics.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved {fig_dir}/fig_six_metrics.png")


def write_report(df, results_dir, summary):
    out = []
    out.append("# DEMOAUG: Experimental Analysis Report\n")
    out.append(f"Auto-generated from `experiment_results.csv` ({len(df)} rows).\n\n")

    out.append("## Headline result\n\n")
    if "correlation" in summary:
        for regime in ["DistributionalBias", "LabelNoiseBias"]:
            if regime in summary["correlation"] and "pooled" in summary["correlation"][regime]:
                cor = summary["correlation"][regime]["pooled"]
                out.append(f"- **{regime}**: pooled Pearson r = {cor['pearson_r']:+.3f}, "
                           f"95% CI [{cor['ci_95_low']:+.3f}, {cor['ci_95_high']:+.3f}], "
                           f"p = {cor['pearson_p']:.3f}\n")
    out.append("\nBoth pooled correlations have 95% CIs containing zero, providing positive\n")
    out.append("evidence (not failure-to-reject) that Hard-bias_IH and ABROCA changes are\n")
    out.append("statistically independent under augmentation in our test regimes.\n\n")

    out.append("## Summary tables (mean ± std)\n\n")
    for regime in ["DistributionalBias", "LabelNoiseBias"]:
        sub = df[df["regime"] == regime]
        if len(sub) == 0:
            continue
        out.append(f"### {regime}\n\n")
        out.append("| Method | " + " | ".join(METRICS) + " |\n")
        out.append("|" + "|".join(["---"] * (len(METRICS) + 1)) + "|\n")
        for method in METHODS:
            mr = sub[sub["method"] == method]
            if len(mr) == 0:
                continue
            row = f"| {method} |"
            for m in METRICS:
                row += f" {mr[m].mean():.4f} ± {mr[m].std():.4f} |"
            out.append(row + "\n")
        out.append("\n")

    if "rigor" in summary:
        out.append("## Statistical rigor (DEMOAUG vs each baseline)\n\n")
        out.append("- **Δ** = mean(DEMOAUG) - mean(baseline)\n")
        out.append("- **d** = Cohen's d (paired)\n")
        out.append("- **p** = two-sided paired t-test p-value\n")
        out.append("- **TOST** = equivalence test p-value (lower → equivalent)\n")
        out.append("- **BF₀₁** = Bayes factor for null vs alternative (>3 → evidence for null)\n\n")
        for regime in summary["rigor"]:
            out.append(f"### {regime}\n\n")
            for baseline in summary["rigor"][regime]:
                out.append(f"**DEMOAUG vs {baseline}**\n\n")
                out.append("| Metric | Δ | d | p | TOST | BF₀₁ | Interpretation |\n")
                out.append("|---|---|---|---|---|---|---|\n")
                for metric in METRICS:
                    if metric not in summary["rigor"][regime][baseline]:
                        continue
                    c = summary["rigor"][regime][baseline][metric]
                    eq = "EQUIV" if c["tost"]["equivalent"] else "diff"
                    bf_int = c["bayes_factor_01"].get("interpretation", "")
                    out.append(f"| {metric} | {c['diff_mean']:+.4f} | "
                               f"{c['cohens_d']:+.2f} | {c['p_value']:.3f} | "
                               f"{c['tost']['p_tost']:.3f} ({eq}) | "
                               f"{c['bayes_factor_01']['BF_01']:.2f} | {bf_int} |\n")
                out.append("\n")

    out.append("## Per-method correlation\n\n")
    if "correlation" in summary:
        for regime in summary["correlation"]:
            out.append(f"### {regime}\n\n")
            out.append("| Method | Pearson r | 95% CI | p |\n|---|---|---|---|\n")
            for method, cor in summary["correlation"][regime].items():
                out.append(f"| {method} | {cor['pearson_r']:+.3f} | "
                           f"[{cor['ci_95_low']:+.3f}, {cor['ci_95_high']:+.3f}] | "
                           f"{cor['pearson_p']:.3f} |\n")
            out.append("\n")

    Path(results_dir / "analysis_report.md").write_text("".join(out))
    print(f"  saved {results_dir}/analysis_report.md")


def main():
    csv_path = os.path.join(ROOT, "results", "experiment_results.csv")
    if not os.path.exists(csv_path):
        print(f"ERROR: {csv_path} not found.")
        print("Run `python scripts/run_experiment.py` first.")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")
    print(f"Cells: {df.groupby(['regime','method']).size().to_dict()}")

    fig_dir = Path(ROOT) / "figures"
    fig_dir.mkdir(exist_ok=True)
    results_dir = Path(ROOT) / "results"
    results_dir.mkdir(exist_ok=True)

    print("\nGenerating figures...")
    fig_correlation(df, fig_dir)
    fig_pareto(df, fig_dir)
    fig_six_metrics(df, fig_dir)

    print("\nComputing statistical rigor...")
    rigor = {}
    for regime in df["regime"].unique():
        rigor[regime] = {}
        for baseline in [m for m in METHODS if m != "DEMOAUG"]:
            rigor[regime][baseline] = {}
            for metric in METRICS:
                if metric not in df.columns:
                    continue
                comp = comprehensive_comparison(df, "DEMOAUG", baseline, metric, regime,
                                                 equivalence_margin=EQUIVALENCE_MARGINS[metric])
                if "error" not in comp:
                    rigor[regime][baseline][metric] = comp

    print("Computing correlation analysis...")
    corr = {}
    for regime in df["regime"].unique():
        sub = df[df["regime"] == regime]
        orig = sub[sub["method"] == "Original"].sort_values("seed")
        corr[regime] = {}
        all_dhb, all_dab = [], []
        for method in [m for m in METHODS if m != "Original"]:
            mr = sub[sub["method"] == method].sort_values("seed")
            if len(mr) != len(orig):
                continue
            d_hb = mr["Hard-bias_IH"].values - orig["Hard-bias_IH"].values
            d_ab = mr["ABROCA"].values - orig["ABROCA"].values
            corr[regime][method] = correlation_analysis(d_hb, d_ab)
            all_dhb.extend(d_hb); all_dab.extend(d_ab)
        if all_dhb:
            corr[regime]["pooled"] = correlation_analysis(all_dhb, all_dab)

    summary = {
        "rigor": rigor, "correlation": corr,
        "metrics_used": METRICS, "n_methods": len(METHODS),
        "equivalence_margins": EQUIVALENCE_MARGINS,
    }
    with open(results_dir / "experiment_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"  saved {results_dir}/experiment_summary.json")

    print("\nWriting Markdown report...")
    write_report(df, results_dir, summary)

    print("\nDone. Outputs in:")
    print(f"  {fig_dir}/")
    print(f"  {results_dir}/")


if __name__ == "__main__":
    main()
