"""
Cohort analysis for banking customer acquisition and retention.

Produces:
  - Monthly acquisition cohort retention heatmap
  - Average CLV_12m by acquisition channel
  - Product depth evolution (% with 2+ products at 12 months)
  - Cross-sell propensity scoring
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

OUTPUT_DIR = "outputs"


def build_cohort_table(
    customers: pd.DataFrame, transactions: pd.DataFrame
) -> pd.DataFrame:
    """
    Build a monthly acquisition cohort table with retention rates.

    Returns a pivot: rows = cohort month, columns = months since acquisition,
    values = retention rate (% of cohort still transacting).
    """
    cust = customers.copy()
    txns = transactions.copy()

    cust["account_open_date"] = pd.to_datetime(cust["account_open_date"])
    txns["transaction_date"] = pd.to_datetime(txns["transaction_date"])

    cust["cohort_month"] = cust["account_open_date"].dt.to_period("M")
    txns = txns.merge(
        cust[["customer_id", "cohort_month"]], on="customer_id", how="left"
    )
    txns["txn_month"] = txns["transaction_date"].dt.to_period("M")

    # Months since acquisition
    txns["months_since"] = (
        txns["txn_month"].astype("int64") - txns["cohort_month"].astype("int64")
    )
    txns = txns[txns["months_since"] >= 0]

    # Active customers per cohort per month-since
    active = (
        txns.groupby(["cohort_month", "months_since"])["customer_id"]
        .nunique()
        .reset_index(name="active_customers")
    )

    # Cohort sizes
    cohort_sizes = cust.groupby("cohort_month")["customer_id"].nunique().rename("cohort_size")
    active = active.merge(cohort_sizes, on="cohort_month")

    active["retention_rate"] = (
        active["active_customers"] / active["cohort_size"] * 100
    ).round(1)

    # Pivot for heatmap
    pivot = active.pivot_table(
        index="cohort_month",
        columns="months_since",
        values="retention_rate",
    )
    # Keep only first 24 months
    pivot = pivot[[c for c in pivot.columns if c <= 24]]

    return pivot


def plot_retention_heatmap(
    cohort_pivot: pd.DataFrame,
    save_path: str = None,
) -> plt.Figure:
    """Plot a 24-month retention rate heatmap by cohort."""
    fig, ax = plt.subplots(figsize=(16, max(8, len(cohort_pivot) * 0.4)))
    sns.heatmap(
        cohort_pivot,
        annot=True,
        fmt=".0f",
        cmap="YlOrRd_r",
        ax=ax,
        cbar_kws={"label": "Retention %"},
    )
    ax.set_title("Monthly Cohort Retention Rates (%)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Months Since Acquisition")
    ax.set_ylabel("Cohort (Account Open Month)")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved retention heatmap → {save_path}")

    return fig


def clv_by_channel(
    clv_df: pd.DataFrame,
) -> pd.DataFrame:
    """Average CLV_12m and CLV_36m by acquisition channel."""
    if "acquisition_channel" not in clv_df.columns:
        raise ValueError("clv_df must contain 'acquisition_channel'")

    channel_stats = (
        clv_df.groupby("acquisition_channel")
        .agg(
            customer_count=("clv_12m", "count"),
            avg_clv_12m=("clv_12m", "mean"),
            avg_clv_36m=("credit_risk_adj_clv_36m", "mean"),
            avg_p_alive=("p_alive", "mean"),
        )
        .round(2)
        .sort_values("avg_clv_36m", ascending=False)
    )

    # Acquisition cost estimates (industry benchmarks)
    acq_costs = {
        "branch": 250,
        "digital": 80,
        "referral": 120,
        "broker": 350,
        "employer_scheme": 60,
    }
    channel_stats["est_acquisition_cost"] = channel_stats.index.map(acq_costs)
    channel_stats["roi_36m"] = (
        (channel_stats["avg_clv_36m"] - channel_stats["est_acquisition_cost"])
        / channel_stats["est_acquisition_cost"]
        * 100
    ).round(1)

    return channel_stats


def product_depth_evolution(
    customers: pd.DataFrame,
    transactions: pd.DataFrame,
) -> pd.DataFrame:
    """
    For each cohort, what % holds 2+ products at 12 months?
    (Proxy: customers with 2+ products whose tenure ≥ 12 months.)
    """
    cust = customers.copy()
    cust["account_open_date"] = pd.to_datetime(cust["account_open_date"])
    cust["cohort_month"] = cust["account_open_date"].dt.to_period("M")

    latest = pd.to_datetime(transactions["transaction_date"]).max()
    cust["tenure_months"] = (latest - cust["account_open_date"]).dt.days / 30.44

    # Only consider customers with ≥12 months tenure
    mature = cust[cust["tenure_months"] >= 12]
    if mature.empty:
        return pd.DataFrame()

    cohort_depth = (
        mature.groupby("cohort_month")
        .agg(
            total_customers=("customer_id", "count"),
            multi_product=("product_count", lambda x: (x >= 2).sum()),
        )
    )
    cohort_depth["pct_2plus_products"] = (
        cohort_depth["multi_product"] / cohort_depth["total_customers"] * 100
    ).round(1)

    return cohort_depth


def cross_sell_propensity(
    customers: pd.DataFrame,
    clv_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Score customers for cross-sell propensity (mortgage next product).

    High propensity = high CLV + primary bank + 1-2 products + good credit + mid-high income.
    """
    cust = customers.set_index("customer_id")
    score = pd.DataFrame(index=cust.index)

    # Base score from CLV tier
    tier_scores = {"Platinum": 5, "Gold": 4, "Silver": 3, "Bronze": 2, "At Risk": 1}
    if "tier" in clv_df.columns:
        score["tier_score"] = clv_df["tier"].map(tier_scores).fillna(2)
    else:
        score["tier_score"] = 3

    # Primary bank bonus
    score["primary_score"] = cust["is_primary_bank"] * 2

    # Product gap (room to cross-sell)
    score["product_gap_score"] = (
        (cust["product_count"].between(1, 2)).astype(int) * 3
    )

    # Credit quality
    credit_scores = {"excellent": 4, "good": 3, "fair": 2, "poor": 1}
    score["credit_score"] = cust["credit_score_band"].map(credit_scores)

    # Income capacity
    income_scores = {"high": 3, "mid": 2, "low": 1}
    score["income_score"] = cust["income_band"].map(income_scores)

    score["propensity_score"] = (
        score["tier_score"]
        + score["primary_score"]
        + score["product_gap_score"]
        + score["credit_score"]
        + score["income_score"]
    )

    # Normalize to 0-100
    s_min, s_max = score["propensity_score"].min(), score["propensity_score"].max()
    if s_max > s_min:
        score["propensity_score_normalized"] = (
            (score["propensity_score"] - s_min) / (s_max - s_min) * 100
        ).round(1)
    else:
        score["propensity_score_normalized"] = 50.0

    return score[["propensity_score", "propensity_score_normalized"]].sort_values(
        "propensity_score_normalized", ascending=False
    )


if __name__ == "__main__":
    print("Cohort module loaded. Run via main.py pipeline.")
