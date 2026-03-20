"""
Fairness analysis for CLV model predictions.

Regulatory requirement for fintech/banking (ECOA, FCA):
  - Check that CLV predictions do not systematically differ by protected or
    proxy characteristics (credit_score_band, income_band, acquisition_channel)
  - Disparate impact ratio: min_group_avg_CLV / max_group_avg_CLV
  - Flag if ratio < 0.8 (standard four-fifths rule threshold)
  - Generate a markdown fairness report
"""

from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

FAIRNESS_THRESHOLD = 0.80  # four-fifths rule (ECOA standard)


def compute_disparate_impact(
    clv_df: pd.DataFrame,
    group_col: str,
    value_col: str = "credit_risk_adj_clv_36m",
) -> pd.DataFrame:
    """
    Compute disparate impact ratio for a given grouping variable.

    Returns DataFrame with group averages and the DI ratio.
    """
    group_avg = clv_df.groupby(group_col)[value_col].mean()
    di_ratio = group_avg.min() / group_avg.max() if group_avg.max() > 0 else np.nan
    flagged = di_ratio < FAIRNESS_THRESHOLD

    result = pd.DataFrame({
        "group": group_avg.index,
        "avg_clv_36m": group_avg.values.round(2),
    })
    result["pct_of_max"] = (result["avg_clv_36m"] / result["avg_clv_36m"].max() * 100).round(1)
    result = result.sort_values("avg_clv_36m", ascending=False).reset_index(drop=True)

    return result, di_ratio, flagged


def run_fairness_checks(
    clv_df: pd.DataFrame,
    dimensions: Optional[List[str]] = None,
) -> dict:
    """
    Run disparate impact checks across multiple dimensions.

    Returns dict of {dimension: (detail_df, di_ratio, flagged)}.
    """
    if dimensions is None:
        dimensions = ["credit_score_band", "income_band", "acquisition_channel"]

    results = {}
    for dim in dimensions:
        if dim in clv_df.columns:
            results[dim] = compute_disparate_impact(clv_df, dim)
    return results


def generate_fairness_report(
    fairness_results: dict,
) -> str:
    """Generate a markdown-formatted fairness report."""
    lines = [
        "# CLV Model Fairness Report",
        "",
        "> **Regulatory context**: Under ECOA and FCA guidance, models used in credit "
        "marketing and retention decisions must demonstrate that outcomes do not "
        "disproportionately disadvantage protected groups. The four-fifths rule "
        f"(threshold = {FAIRNESS_THRESHOLD:.0%}) is applied as a screening test.",
        "",
        "---",
        "",
    ]

    overall_pass = True

    for dim, (detail_df, di_ratio, flagged) in fairness_results.items():
        status = "🔴 FAIL" if flagged else "🟢 PASS"
        if flagged:
            overall_pass = False

        lines.append(f"## Dimension: `{dim}`")
        lines.append("")
        lines.append(f"**Disparate Impact Ratio**: {di_ratio:.4f}  |  **Status**: {status}")
        lines.append("")
        lines.append("| Group | Avg CLV (36m) | % of Max |")
        lines.append("|-------|--------------|----------|")
        for _, row in detail_df.iterrows():
            lines.append(
                f"| {row['group']} | £{row['avg_clv_36m']:,.2f} | {row['pct_of_max']:.1f}% |"
            )
        lines.append("")

        if flagged:
            lines.append(
                f"⚠️  **Action required**: The disparate impact ratio for `{dim}` "
                f"({di_ratio:.4f}) is below the {FAIRNESS_THRESHOLD:.0%} threshold. "
                "Investigate whether the difference is driven by legitimate business "
                "factors (product usage, tenure) or proxy discrimination."
            )
        else:
            lines.append(
                f"✅ CLV differences across `{dim}` groups are within acceptable limits."
            )
        lines.append("")
        lines.append("---")
        lines.append("")

    # Overall verdict
    if overall_pass:
        lines.append("## Overall Verdict: 🟢 PASS")
        lines.append("")
        lines.append(
            "All dimensions pass the four-fifths rule. CLV model predictions are "
            "within regulatory tolerance for disparate impact."
        )
    else:
        lines.append("## Overall Verdict: 🔴 REVIEW REQUIRED")
        lines.append("")
        lines.append(
            "One or more dimensions fail the four-fifths rule. Model risk management "
            "(SR 11-7) requires documented justification or model remediation before "
            "deployment in credit marketing decisions."
        )

    lines.append("")
    lines.append("---")
    lines.append(
        "*Note: CLV should legitimately differ by product usage and engagement. "
        "Disparate impact checks ensure differences are not driven by proxy "
        "characteristics. This analysis uses synthetic data only — GDPR compliant.*"
    )

    return "\n".join(lines)


def rag_status(di_ratio: float) -> str:
    """Return RAG (Red/Amber/Green) status for a DI ratio."""
    if di_ratio >= 0.90:
        return "GREEN"
    elif di_ratio >= FAIRNESS_THRESHOLD:
        return "AMBER"
    else:
        return "RED"


if __name__ == "__main__":
    print("Fairness module loaded. Run via main.py pipeline.")
