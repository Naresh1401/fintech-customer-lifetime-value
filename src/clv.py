"""
Customer Lifetime Value calculation.

Combines BG/NBD (engagement) and Gamma-Gamma (monetary) predictions to produce:
  - CLV_12m = predicted_active_months × expected_monthly_revenue
  - CLV_36m = discounted at 8% annual (bank's hurdle rate)
  - Credit-risk adjusted CLV
  - Portfolio tier segmentation (Platinum / Gold / Silver / Bronze / At Risk)
  - Recommended retention actions per tier
"""

import numpy as np
import pandas as pd


# Bank hurdle rate (annual)
ANNUAL_DISCOUNT_RATE = 0.08
MONTHLY_DISCOUNT_RATE = (1 + ANNUAL_DISCOUNT_RATE) ** (1 / 12) - 1

# Estimated default probabilities by credit score band (annualised)
DEFAULT_PROB = {
    "excellent": 0.005,
    "good": 0.015,
    "fair": 0.045,
    "poor": 0.10,
}

# CLV tier thresholds (GBP)
TIER_THRESHOLDS = {
    "Platinum": 5_000,
    "Gold": 2_000,
    "Silver": 500,
    "Bronze": 100,
}

# Retention action recommendations
TIER_ACTIONS = {
    "Platinum": "Assign dedicated RM; proactive cross-sell premium products; annual loyalty review",
    "Gold": "Cross-sell mortgage/investment; offer fee waivers on savings; quarterly engagement call",
    "Silver": "Nudge digital engagement; promote 2nd product; automated retention offers",
    "Bronze": "Low-cost digital engagement; monitor for upgrade signals; consider cost-to-serve review",
    "At Risk": "Immediate outbound retention call; fee rebate offer; exit interview if churning",
}


def compute_clv(
    engagement: pd.DataFrame,
    expected_monetary: pd.Series,
    customers: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute 12m and 36m CLV for every customer.

    Parameters
    ----------
    engagement : DataFrame with 'predicted_active_months_next_12m' and 'p_alive'
    expected_monetary : Series of expected monthly revenue per customer
    customers : customer master table (for credit_score_band, account_type)

    Returns
    -------
    DataFrame with CLV metrics and tier assignments.
    """
    cust = customers.set_index("customer_id")

    clv = pd.DataFrame(index=engagement.index)
    clv["predicted_active_months_12m"] = engagement["predicted_active_months_next_12m"]
    clv["p_alive"] = engagement["p_alive"]
    clv["expected_monthly_revenue"] = expected_monetary

    # --- CLV_12m (undiscounted) ---
    clv["clv_12m"] = clv["predicted_active_months_12m"] * clv["expected_monthly_revenue"]

    # --- CLV_36m (discounted) ---
    # Extrapolate 36m active months: scale by ratio (capped at 36)
    clv["predicted_active_months_36m"] = (
        clv["predicted_active_months_12m"] * 2.8
    ).clip(upper=36)

    # Apply monthly discounting
    clv["clv_36m"] = _discount_clv(
        clv["predicted_active_months_36m"],
        clv["expected_monthly_revenue"],
        months=36,
    )

    # --- Credit risk adjustment ---
    credit_bands = clv.index.map(cust["credit_score_band"].to_dict())
    account_types = clv.index.map(cust["account_type"].to_dict())

    default_probs_36m = np.array([
        1 - (1 - DEFAULT_PROB.get(band, 0.03)) ** 3  # 3-year cumulative PD
        for band in credit_bands
    ])
    # Only apply default adjustment to credit products
    is_credit_product = np.array([
        1.0 if at in ("credit_card", "loan") else 0.3
        for at in account_types
    ])
    risk_factor = 1 - default_probs_36m * is_credit_product
    clv["credit_risk_adj_clv_36m"] = clv["clv_36m"] * risk_factor

    # --- Tier assignment ---
    clv["tier"] = _assign_tier(clv)

    # --- Retention action ---
    clv["retention_action"] = clv["tier"].map(TIER_ACTIONS)

    # --- Additional context ---
    for col in ["acquisition_channel", "product_count", "account_type",
                 "credit_score_band", "income_band"]:
        if col in cust.columns:
            clv[col] = clv.index.map(cust[col].to_dict())

    clv = clv.round(2)
    return clv


def _discount_clv(
    active_months: pd.Series,
    monthly_revenue: pd.Series,
    months: int = 36,
) -> pd.Series:
    """Apply monthly discounting to CLV over the projection horizon."""
    discount_factors = np.array([
        1 / (1 + MONTHLY_DISCOUNT_RATE) ** m for m in range(1, months + 1)
    ])
    total_discount = discount_factors.sum()

    # Scale by proportion of months the customer is expected to be active
    fraction_active = active_months / months
    return monthly_revenue * fraction_active * total_discount


def _assign_tier(clv: pd.DataFrame) -> pd.Series:
    """Assign portfolio tier based on CLV_36m and p_alive."""
    tier = pd.Series("Bronze", index=clv.index)

    # At Risk overrides everything if p_alive is low
    tier.loc[clv["p_alive"] < 0.3] = "At Risk"

    # Tier by CLV_36m
    adj = clv["credit_risk_adj_clv_36m"]
    tier.loc[adj >= TIER_THRESHOLDS["Platinum"]] = "Platinum"
    tier.loc[(adj >= TIER_THRESHOLDS["Gold"]) & (adj < TIER_THRESHOLDS["Platinum"])] = "Gold"
    tier.loc[(adj >= TIER_THRESHOLDS["Silver"]) & (adj < TIER_THRESHOLDS["Gold"])] = "Silver"
    tier.loc[(adj >= TIER_THRESHOLDS["Bronze"]) & (adj < TIER_THRESHOLDS["Silver"])] = "Bronze"

    # At Risk keeps its label even if CLV is high
    tier.loc[clv["p_alive"] < 0.3] = "At Risk"

    return tier


def portfolio_summary(clv: pd.DataFrame) -> pd.DataFrame:
    """Aggregate portfolio-level CLV summary by tier."""
    summary = (
        clv.groupby("tier")
        .agg(
            customer_count=("clv_12m", "count"),
            avg_clv_12m=("clv_12m", "mean"),
            avg_clv_36m=("credit_risk_adj_clv_36m", "mean"),
            total_clv_36m=("credit_risk_adj_clv_36m", "sum"),
            avg_p_alive=("p_alive", "mean"),
        )
        .round(2)
    )
    summary["pct_of_portfolio_value"] = (
        summary["total_clv_36m"] / summary["total_clv_36m"].sum() * 100
    ).round(1)
    return summary.sort_values("avg_clv_36m", ascending=False)


if __name__ == "__main__":
    # Quick standalone test
    print("CLV module loaded. Run via main.py pipeline.")
