"""
Gamma-Gamma monetary value model for banking revenue prediction.

Predicts expected monthly revenue contribution per customer, with
separate models for:
  - net_interest_margin_contribution (NIM)
  - fee_revenue
"""

import warnings
import numpy as np
import pandas as pd
from lifetimes import GammaGammaFitter

warnings.filterwarnings("ignore", category=RuntimeWarning)


def prepare_monetary_data(
    customers: pd.DataFrame,
    transactions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build per-customer monetary summary for the Gamma-Gamma model.

    Gamma-Gamma requires:
      - frequency (>0, repeat purchasers only for fitting)
      - monetary_value (avg spend / revenue per transaction period)

    We compute two monetary columns:
      - monetary_nim: avg NIM contribution
      - monetary_fee: avg fee revenue
      - monetary_value: total monthly bank revenue (NIM + fees)
    """
    txns = transactions.copy()
    txns["transaction_date"] = pd.to_datetime(txns["transaction_date"])

    # Count distinct transaction months per customer (frequency)
    txns["txn_month"] = txns["transaction_date"].dt.to_period("M")
    freq = txns.groupby("customer_id")["txn_month"].nunique().rename("frequency")

    cust = customers.set_index("customer_id")

    monetary = pd.DataFrame(index=cust.index)
    monetary["frequency"] = freq
    monetary["frequency"] = monetary["frequency"].fillna(0).astype(int)
    monetary["monetary_nim"] = cust["net_interest_margin_contribution"]
    monetary["monetary_fee"] = cust["fee_revenue"]
    monetary["monetary_value"] = monetary["monetary_nim"] + monetary["monetary_fee"]

    return monetary


def fit_gamma_gamma(
    monetary_data: pd.DataFrame,
    penalizer: float = 0.01,
    col: str = "monetary_value",
) -> GammaGammaFitter:
    """
    Fit the Gamma-Gamma model on repeat customers.

    Parameters
    ----------
    monetary_data : DataFrame with 'frequency' and monetary column
    col : which monetary column to model
    """
    # Gamma-Gamma requires frequency > 0
    repeat = monetary_data[monetary_data["frequency"] > 0].copy()

    ggf = GammaGammaFitter(penalizer_coef=penalizer)
    ggf.fit(repeat["frequency"], repeat[col])
    print(f"Gamma-Gamma model fitted on '{col}'.")
    print(f"  Parameters: {ggf.summary}")
    return ggf


def predict_monetary(
    ggf: GammaGammaFitter,
    monetary_data: pd.DataFrame,
    col: str = "monetary_value",
) -> pd.Series:
    """
    Predict expected average monetary value per customer.

    For customers with frequency == 0, use the population mean.
    """
    repeat = monetary_data[monetary_data["frequency"] > 0]
    expected = ggf.conditional_expected_average_profit(
        repeat["frequency"], repeat[col]
    )

    # Assign population mean for zero-frequency customers
    all_expected = pd.Series(
        ggf.conditional_expected_average_profit(
            monetary_data["frequency"].clip(lower=1),
            monetary_data[col].clip(lower=0.01),
        ),
        index=monetary_data.index,
    )
    # Override zero-freq with global mean
    pop_mean = expected.mean()
    all_expected.loc[monetary_data["frequency"] == 0] = pop_mean

    return all_expected.rename(f"expected_{col}")


def validate_predictions(
    actual: pd.Series, predicted: pd.Series
) -> dict:
    """Compare predicted vs actual monetary values."""
    common = actual.index.intersection(predicted.index)
    a = actual.loc[common]
    p = predicted.loc[common]

    mape = (np.abs(a - p) / a.clip(lower=0.01)).mean()
    corr = a.corr(p)
    rmse = np.sqrt(((a - p) ** 2).mean())

    metrics = {"MAPE": round(mape, 4), "Correlation": round(corr, 4), "RMSE": round(rmse, 2)}
    print(f"Validation: {metrics}")
    return metrics


if __name__ == "__main__":
    cust = pd.read_csv("data/customers.csv")
    txns = pd.read_csv("data/transactions.csv")

    mon_data = prepare_monetary_data(cust, txns)

    # Fit combined model
    ggf = fit_gamma_gamma(mon_data, col="monetary_value")
    expected = predict_monetary(ggf, mon_data, col="monetary_value")
    print("\nSample expected monetary values:")
    print(expected.head(10))

    # Validate
    validate_predictions(mon_data["monetary_value"], expected)
