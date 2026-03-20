"""
RFM analysis adapted for retail banking.

Standard RFM is extended with banking-specific dimensions:
  - Depth: product cross-sell count
  - Primary bank score: weighted engagement metric
  - Banking-specific segment labels
"""

import numpy as np
import pandas as pd


def compute_rfm(
    customers: pd.DataFrame, transactions: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute per-customer RFM+ metrics.

    Returns a DataFrame indexed by customer_id with columns:
        recency, frequency, monetary, T, depth, primary_bank_score, segment
    """
    transactions = transactions.copy()
    transactions["transaction_date"] = pd.to_datetime(transactions["transaction_date"])

    analysis_date = transactions["transaction_date"].max() + pd.Timedelta(days=1)

    # --- Recency: days since last transaction ---
    last_txn = (
        transactions.groupby("customer_id")["transaction_date"]
        .max()
        .rename("last_txn_date")
    )
    recency = (analysis_date - last_txn).dt.days.rename("recency")

    # --- Frequency: distinct transaction months (repeat engagement) ---
    transactions["txn_month"] = transactions["transaction_date"].dt.to_period("M")
    frequency = (
        transactions.groupby("customer_id")["txn_month"]
        .nunique()
        .rename("frequency")
    )

    # --- Monetary: avg NIM contribution + avg fee revenue (bank profit) ---
    monetary = (
        customers.set_index("customer_id")[
            ["net_interest_margin_contribution", "fee_revenue"]
        ]
        .sum(axis=1)
        .rename("monetary")
    )

    # --- T: account tenure in months ---
    customers = customers.copy()
    customers["account_open_date"] = pd.to_datetime(customers["account_open_date"])
    tenure = (
        (analysis_date - customers.set_index("customer_id")["account_open_date"])
        .dt.days
        / 30.44
    ).rename("T")

    # --- Depth: product count ---
    depth = customers.set_index("customer_id")["product_count"].rename("depth")

    # --- Primary bank score ---
    cust_idx = customers.set_index("customer_id")
    primary_bank_score = (
        cust_idx["is_primary_bank"] * 2 + cust_idx["product_count"]
    ).rename("primary_bank_score")

    # Combine
    rfm = pd.concat(
        [recency, frequency, monetary, tenure, depth, primary_bank_score], axis=1
    )

    # Fill any customers with no transactions
    rfm["recency"] = rfm["recency"].fillna(rfm["recency"].max())
    rfm["frequency"] = rfm["frequency"].fillna(0)

    # --- Segment labels ---
    rfm["segment"] = _assign_segments(rfm)

    return rfm


def _assign_segments(rfm: pd.DataFrame) -> pd.Series:
    """
    Assign banking-specific segment labels using rule-based logic:
      - High Value: top monetary quartile AND frequency ≥ median
      - Deep Relationship: depth ≥ 3 AND primary_bank_score ≥ 4
      - Transactor: high frequency, mid monetary
      - Single Product Risk: depth == 1 AND tenure > 6 months
      - Dormant: recency > 180 days OR frequency ≤ 1
    """
    seg = pd.Series("Transactor", index=rfm.index, name="segment")

    monetary_q75 = rfm["monetary"].quantile(0.75)
    freq_median = rfm["frequency"].median()

    # Dormant (check first so others can override)
    seg.loc[(rfm["recency"] > 180) | (rfm["frequency"] <= 1)] = "Dormant"

    # Single Product Risk
    seg.loc[(rfm["depth"] == 1) & (rfm["T"] > 6)] = "Single Product Risk"

    # Deep Relationship
    seg.loc[(rfm["depth"] >= 3) & (rfm["primary_bank_score"] >= 4)] = (
        "Deep Relationship"
    )

    # High Value (highest priority override)
    seg.loc[
        (rfm["monetary"] >= monetary_q75) & (rfm["frequency"] >= freq_median)
    ] = "High Value"

    return seg


if __name__ == "__main__":
    cust = pd.read_csv("data/customers.csv")
    txns = pd.read_csv("data/transactions.csv")
    rfm_df = compute_rfm(cust, txns)
    print(rfm_df.head(10))
    print("\nSegment distribution:")
    print(rfm_df["segment"].value_counts())
