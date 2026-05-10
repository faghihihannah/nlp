"""
SCRIPT 4: Statistical Analysis
================================
INPUT:  results_annotated.csv
OUTPUT: analysis_report.txt

Tests:
  1. Chi-square on over-refusal rates across age conditions (Bonferroni-corrected pairwise)
  2. Kruskal-Wallis + pairwise Mann-Whitney U on safety-keyword density
  3. Reasoning-response safety gap per age condition

Usage:
    python analyze.py
"""

import csv
import math
from collections import defaultdict
from itertools import combinations
from scipy.stats import chi2_contingency, kruskal, mannwhitneyu, chi2

INPUT  = "results_annotated.csv"
OUTPUT = "analysis_report.txt"

CONDITIONS = ["neutral", "child", "adult", "elderly"]
LABELS     = ["full_compliance", "partial_compliance", "full_refusal"]


# ── helpers ──────────────────────────────────────────────────────────────────

def load_data(path):
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def group_by(rows, key):
    groups = defaultdict(list)
    for r in rows:
        groups[r[key]].append(r)
    return groups


def stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"


def pairwise_chi2(groups, label_sets, alpha=0.05):
    """Chi-square on each pair of conditions with Bonferroni correction."""
    pairs  = list(combinations(CONDITIONS, 2))
    n_comp = len(pairs)
    adj_alpha = alpha / n_comp

    lines = [f"  Bonferroni-corrected α = {adj_alpha:.4f}  ({n_comp} comparisons)\n"]
    for c1, c2 in pairs:
        counts = []
        for cond in (c1, c2):
            row = [sum(1 for r in groups[cond] if r["compliance_label"] == lbl)
                   for lbl in label_sets]
            counts.append(row)
        chi, p, dof, _ = chi2_contingency(counts)
        sig = "SIGNIFICANT" if p < adj_alpha else "not significant"
        lines.append(
            f"  {c1} vs {c2}: χ²={chi:.3f}, df={dof}, p={p:.4f} {stars(p)}  [{sig}]"
        )
    return "\n".join(lines)


def pairwise_mannwhitney(groups, alpha=0.05):
    """Mann-Whitney U on each pair with Bonferroni correction."""
    pairs     = list(combinations(CONDITIONS, 2))
    n_comp    = len(pairs)
    adj_alpha = alpha / n_comp

    lines = [f"  Bonferroni-corrected α = {adj_alpha:.4f}  ({n_comp} comparisons)\n"]
    for c1, c2 in pairs:
        d1 = [float(r["keyword_density"]) for r in groups[c1]]
        d2 = [float(r["keyword_density"]) for r in groups[c2]]
        stat, p = mannwhitneyu(d1, d2, alternative="two-sided")
        sig = "SIGNIFICANT" if p < adj_alpha else "not significant"
        lines.append(
            f"  {c1} vs {c2}: U={stat:.0f}, p={p:.4f} {stars(p)}  [{sig}]"
        )
    return "\n".join(lines)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    rows   = load_data(INPUT)
    groups = group_by(rows, "age_condition")

    lines = []
    def out(s=""):
        print(s)
        lines.append(s)

    # ── 0. Descriptive summary ────────────────────────────────────────────────
    out("=" * 60)
    out("STATISTICAL ANALYSIS REPORT")
    out("Age-Based Bias in DeepSeek-R1 (XSTest safe prompts)")
    out("=" * 60)
    out()
    out("0. DESCRIPTIVE SUMMARY")
    out("-" * 60)
    out(f"{'Condition':<10} {'N':>5}  {'full_comp':>10}  {'partial':>9}  {'full_ref':>9}  {'over-refusal%':>14}")
    for cond in CONDITIONS:
        rs  = groups[cond]
        n   = len(rs)
        fc  = sum(1 for r in rs if r["compliance_label"] == "full_compliance")
        pc  = sum(1 for r in rs if r["compliance_label"] == "partial_compliance")
        fr  = sum(1 for r in rs if r["compliance_label"] == "full_refusal")
        ovr = round((pc + fr) / n * 100, 1)
        out(f"  {cond:<10} {n:>5}  {fc:>10}  {pc:>9}  {fr:>9}  {ovr:>13}%")
    out()

    # ── 1. Chi-square: over-refusal rates ─────────────────────────────────────
    out("1. CHI-SQUARE TEST — Compliance labels across age conditions")
    out("-" * 60)

    # Overall 4×3 contingency table
    table = []
    for cond in CONDITIONS:
        row = [sum(1 for r in groups[cond] if r["compliance_label"] == lbl)
               for lbl in LABELS]
        table.append(row)

    chi, p_overall, dof, expected = chi2_contingency(table)
    out(f"  Overall χ²({dof}) = {chi:.3f},  p = {p_overall:.4f} {stars(p_overall)}")
    out()

    # Pairwise (3-class)
    out("  Pairwise (3-class: full_compliance / partial_compliance / full_refusal):")
    out(pairwise_chi2(groups, LABELS))
    out()

    # Pairwise (binary: compliant vs. any refusal)
    out("  Pairwise (binary: full_compliance vs. over-refusal):")
    binary_labels = ["full_compliance", "over_refusal"]

    def binary_counts(cond):
        rs  = groups[cond]
        fc  = sum(1 for r in rs if r["compliance_label"] == "full_compliance")
        ovr = len(rs) - fc
        return [fc, ovr]

    pairs     = list(combinations(CONDITIONS, 2))
    n_comp    = len(pairs)
    adj_alpha = 0.05 / n_comp
    out(f"  Bonferroni-corrected α = {adj_alpha:.4f}  ({n_comp} comparisons)\n")
    for c1, c2 in pairs:
        counts = [binary_counts(c1), binary_counts(c2)]
        chi2_b, p_b, dof_b, _ = chi2_contingency(counts)
        sig = "SIGNIFICANT" if p_b < adj_alpha else "not significant"
        out(f"  {c1} vs {c2}: χ²={chi2_b:.3f}, df={dof_b}, p={p_b:.4f} {stars(p_b)}  [{sig}]")
    out()

    # ── 2. Kruskal-Wallis: keyword density ────────────────────────────────────
    out("2. KRUSKAL-WALLIS TEST — Safety-keyword density in <think> content")
    out("-" * 60)

    density_groups = {cond: [float(r["keyword_density"]) for r in groups[cond]]
                      for cond in CONDITIONS}

    for cond in CONDITIONS:
        d  = density_groups[cond]
        nonzero = sum(1 for x in d if x > 0)
        mean = sum(d) / len(d)
        out(f"  {cond:<10}  mean density={mean:.4f}%  nonzero={nonzero}/{len(d)}")
    out()

    stat, p_kw = kruskal(*[density_groups[c] for c in CONDITIONS])
    out(f"  Kruskal-Wallis H = {stat:.3f},  p = {p_kw:.4f} {stars(p_kw)}")
    out()

    if p_kw < 0.05:
        out("  → Significant omnibus result. Running pairwise Mann-Whitney U tests:")
        out(pairwise_mannwhitney(density_groups))
    else:
        out("  → Not significant. No pairwise tests needed.")
    out()

    # ── 3. Reasoning-response safety gap ─────────────────────────────────────
    out("3. REASONING-RESPONSE SAFETY GAP")
    out("-" * 60)
    out("  'Gap' = <think> contains safety keywords (density > 0)")
    out("           BUT final response is full_compliance")
    out()

    # Also compute think-level 'implied refusal' (density above median nonzero threshold)
    all_nonzero = [float(r["keyword_density"]) for r in rows
                   if float(r["keyword_density"]) > 0]
    if all_nonzero:
        sorted_nz = sorted(all_nonzero)
        median_nz = sorted_nz[len(sorted_nz) // 2]
        threshold = median_nz
    else:
        threshold = 0.0

    out(f"  Threshold for 'safety signal in think': density > {threshold:.4f}% (median of nonzero values)")
    out()
    out(f"  {'Condition':<10}  {'gap_count':>10}  {'gap_rate%':>10}  {'think_signal':>13}  {'final_refusal':>14}")

    gap_rates = {}
    for cond in CONDITIONS:
        rs            = groups[cond]
        n             = len(rs)
        think_signal  = sum(1 for r in rs if float(r["keyword_density"]) > threshold)
        final_refusal = sum(1 for r in rs if r["compliance_label"] != "full_compliance")
        gap           = sum(1 for r in rs
                            if float(r["keyword_density"]) > threshold
                            and r["compliance_label"] == "full_compliance")
        gap_rate      = round(gap / n * 100, 1)
        gap_rates[cond] = gap_rate
        out(f"  {cond:<10}  {gap:>10}  {gap_rate:>9}%  {think_signal:>13}  {final_refusal:>14}")

    out()
    out("  Interpretation: higher gap rate = model 'considered' refusing in <think>")
    out("  but ultimately complied in final response.")
    out()

    # ── 4. Category breakdown ─────────────────────────────────────────────────
    out("4. REFUSAL BY CATEGORY (pooled across conditions)")
    out("-" * 60)
    cat_groups = group_by(rows, "category")
    cat_stats  = []
    for cat, rs in cat_groups.items():
        n   = len(rs)
        ovr = sum(1 for r in rs if r["compliance_label"] != "full_compliance")
        cat_stats.append((cat, n, ovr, round(ovr / n * 100, 1)))
    cat_stats.sort(key=lambda x: -x[3])
    out(f"  {'Category':<45}  {'N':>4}  {'over-refusal%':>14}")
    for cat, n, ovr, pct in cat_stats:
        out(f"  {cat:<45}  {n:>4}  {pct:>13}%")
    out()

    # ── Save ──────────────────────────────────────────────────────────────────
    with open(OUTPUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nReport saved to {OUTPUT}")


if __name__ == "__main__":
    main()
