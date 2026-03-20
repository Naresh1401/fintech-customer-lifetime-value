"""
BG/NBD frequency model for banking customer engagement.

Predicts:
  - predicted_active_months_next_12m per customer
  - p_alive: probability customer is still active (not churned)
  - Cohort comparison: digital vs branch retention at 24 months
"""

import warnings
import numpy as np
import pandas as pd
from lifetimes import BetaGeoFitter
from lifetimes.utils import summary_data_from_transaction_data

warnings.filterwarnings("ignore", category=RuntimeWarning)


def prepare_frequency_data(
    customers: pd.DataFrame, transactions: pd.DataFrame
) -> pd.DataFrame:
    """
    Build the RFM summary required by the BG/NBD model from transaction data.

    Returns DataFrame with columns: frequency, recency, T  (all in ~monthly units).
    Uses daily frequency from lifetimes, then converts to months (÷30.44).
    """
    txns = transactions.copy()
    txns["transaction_date"] = pd.to_datetime(txns["transaction_date"])

    # Use 'D' (days) to avoid pandas timedelta issue with 'M'
    summary = summary_data_from_transaction_data(
        txns,
        customer_id_col="customer_id",
        datetime_col="transaction_date",
        freq="D",
    )

    # Convert from days to months
    summary["recency"] = summary["recency"] / 30.44
    summary["T"] = summary["T"] / 30.44

    # Merge account open date to get proper T (tenure)
    customers = customers.copy()
    customers["account_open_date"] = pd.to_datetime(customers["account_open_date"])
    obs_end = txns["transaction_date"].max()

    cust_tenure = customers.set_index("customer_id")["account_open_date"]
    tenure_months = (obs_end - cust_tenure).dt.days / 30.44

    # Use the larger of computed T or actual tenure
    summary["T"] = summary["T"].clip(lower=0.5)
    for cid in summary.index:
        if cid in tenure_months.index:
            summary.loc[cid, "T"] = max(summary.loc[cid, "T"], tenure_months[cid])

    # BG/NBD requires frequency > 0 for meaningful fit; keep all for predictions
    summary["T"] = summary["T"].clip(lower=0.5)
    summary["frequency"] = summary["frequency"].clip(lower=0)

    return summary


def fit_bgnbd(summary: pd.DataFrame, penalizer: float = 0.01) -> BetaGeoFitter:
    """Fit BG/NBD model on the frequency/recency/T summary."""
    bgf = BetaGeoFitter(penalizer_coef=penalizer)
    bgf.fit(summary["frequency"], summary["recency"], summary["T"])
    print("BG/NBD model fitted.")
    print(f"  Parameters: {bgf.summary}")
    return bgf


def predict_engagement(
    bgf: BetaGeoFitter, summary: pd.DataFrame
) -> pd.DataFrame:
    """
    Predict per-customer:
      - predicted_active_months_next_12m
      - p_alive
    """
    result = summary.copy()
    result["predicted_active_months_next_12m"] = bgf.conditional_expected_number_of_purchases_up_to_time(
        12,  # 12 months ahead
        summary["frequency"],
        summary["recency"],
        summary["T"],
    )
    result["p_alive"] = bgf.conditional_probability_alive(
        summary["frequency"], summary["recency"], summary["T"]
    )
    return result


def cohort_comparison(
    predictions: pd.DataFrame, customers: pd.DataFrame
) -> pd.DataFrame:
    """
    Compare digital vs branch customers: who is more likely still active at 24 months?
    """
    cust_ch = customers.set_index("customer_id")["acquisition_channel"]
    pred = predictions.copy()
    pred["acquisition_channel"] = pred.index.map(cust_ch)

    cohort_stats = (
        pred.groupby("acquisition_channel")
        .agg(
            avg_p_alive=("p_alive", "mean"),
            avg_predicted_months=("predicted_active_months_next_12m", "mean"),
            customer_count=("p_alive", "count"),
        )
        .round(4)
    )
    return cohort_stats


if __name__ == "__main__":
    cust = pd.read_csv("data/customers.csv")
    txns = pd.read_csv("data/transactions.csv")

    summary = prepare_frequency_data(cust, txns)
    bgf = fit_bgnbd(summary)
    preds = predict_engagement(bgf, summary)

    print("\nSample predictions:")
    print(preds[["frequency", "recency", "T", "predicted_active_months_next_12m", "p_alive"]].head(10))

    print("\nCohort comparison (digital vs branch vs referral):")
    print(cohort_comparison(preds, cust))
