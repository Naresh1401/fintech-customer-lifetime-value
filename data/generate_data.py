"""
Synthetic fintech dataset generator.

Generates ~80,000 transactions across ~12,000 customers for a retail bank,
embedding realistic behavioural patterns:
  - Product cross-sell boosts CLV (3+ products → 4× CLV)
  - Digital acquisition = lower cost, similar CLV to branch
  - Referral = highest CLV customers
  - Credit card customers: higher transaction frequency, fee revenue
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

SEED = 42
N_CUSTOMERS = 12_000
TARGET_TRANSACTIONS = 80_000

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def _make_customers(rng: np.random.Generator) -> pd.DataFrame:
    """Create the customer-level master table."""
    cids = [f"C{str(i).zfill(6)}" for i in range(1, N_CUSTOMERS + 1)]

    # Acquisition channels with realistic bank distribution
    channels = rng.choice(
        ["branch", "digital", "referral", "broker", "employer_scheme"],
        size=N_CUSTOMERS,
        p=[0.25, 0.35, 0.15, 0.15, 0.10],
    )

    # Account types — credit_card slightly less common
    account_types = rng.choice(
        ["current", "savings", "credit_card", "loan"],
        size=N_CUSTOMERS,
        p=[0.35, 0.25, 0.25, 0.15],
    )

    # Account open dates: spread across 2.5 years (Jul 2023 → Dec 2025)
    start = datetime(2023, 7, 1)
    open_offsets = rng.integers(0, 900, size=N_CUSTOMERS)  # days
    open_dates = pd.to_datetime([start + timedelta(days=int(d)) for d in open_offsets])

    # Credit score band — referral customers skew higher
    base_scores = rng.choice(
        ["excellent", "good", "fair", "poor"],
        size=N_CUSTOMERS,
        p=[0.20, 0.40, 0.25, 0.15],
    )
    # Boost referral customers
    for i, ch in enumerate(channels):
        if ch == "referral" and rng.random() < 0.4:
            base_scores[i] = "excellent"

    income_bands = rng.choice(
        ["low", "mid", "high"],
        size=N_CUSTOMERS,
        p=[0.30, 0.45, 0.25],
    )
    # Referral customers skew higher income
    for i, ch in enumerate(channels):
        if ch == "referral" and rng.random() < 0.35:
            income_bands[i] = "high"

    # Product count — key driver: referral & high-income get more products
    product_counts = np.ones(N_CUSTOMERS, dtype=int)
    for i in range(N_CUSTOMERS):
        base = 1
        if channels[i] == "referral":
            base += rng.choice([1, 2, 3], p=[0.3, 0.4, 0.3])
        elif channels[i] in ("branch", "employer_scheme"):
            base += rng.choice([0, 1, 2], p=[0.4, 0.4, 0.2])
        elif channels[i] == "digital":
            base += rng.choice([0, 1, 2], p=[0.5, 0.35, 0.15])
        else:
            base += rng.choice([0, 1], p=[0.6, 0.4])
        if income_bands[i] == "high":
            base += rng.choice([0, 1], p=[0.5, 0.5])
        product_counts[i] = min(base, 6)

    is_primary = (rng.random(N_CUSTOMERS) < 0.55).astype(int)
    # Multi-product customers more likely primary
    for i in range(N_CUSTOMERS):
        if product_counts[i] >= 3:
            is_primary[i] = 1 if rng.random() < 0.85 else 0

    # Avg monthly balance — driven by income band & product count
    avg_balance = np.zeros(N_CUSTOMERS)
    for i in range(N_CUSTOMERS):
        base_bal = {"low": 800, "mid": 2500, "high": 6000}[income_bands[i]]
        base_bal *= (1 + 0.15 * product_counts[i])
        avg_balance[i] = max(100, rng.normal(base_bal, base_bal * 0.3))

    # Credit utilization (only meaningful for credit_card)
    credit_util = np.zeros(N_CUSTOMERS)
    for i in range(N_CUSTOMERS):
        if account_types[i] == "credit_card":
            mu = {"excellent": 0.20, "good": 0.35, "fair": 0.55, "poor": 0.75}[
                base_scores[i]
            ]
            credit_util[i] = np.clip(rng.normal(mu, 0.12), 0.01, 0.99)

    # Overdraft events — inversely related to credit score
    overdraft_counts = np.zeros(N_CUSTOMERS, dtype=int)
    for i in range(N_CUSTOMERS):
        rate = {"excellent": 0.1, "good": 0.4, "fair": 1.2, "poor": 2.5}[
            base_scores[i]
        ]
        overdraft_counts[i] = rng.poisson(rate)

    # Net interest margin contribution — key revenue metric
    nim = np.zeros(N_CUSTOMERS)
    for i in range(N_CUSTOMERS):
        base_nim = avg_balance[i] * 0.025 / 12  # ~2.5% NIM annualised → monthly
        # Multi-product boost: 3+ products → ~4× revenue (cross-sell effect)
        if product_counts[i] >= 4:
            base_nim *= 4.0
        elif product_counts[i] == 3:
            base_nim *= 3.0
        elif product_counts[i] == 2:
            base_nim *= 1.8
        # Credit card additional margin
        if account_types[i] == "credit_card":
            base_nim *= 1.5
        nim[i] = max(0.5, rng.normal(base_nim, base_nim * 0.2))

    # Fee revenue — monthly fees, overdraft fees, FX fees
    fee_rev = np.zeros(N_CUSTOMERS)
    for i in range(N_CUSTOMERS):
        base_fee = 5 + product_counts[i] * 3
        base_fee += overdraft_counts[i] * 8
        if account_types[i] == "credit_card":
            base_fee += 12  # annual fee / 12 + interest
        fee_rev[i] = max(0, rng.normal(base_fee, base_fee * 0.25))

    # Churn flag — dormant or closed
    churn_prob = np.full(N_CUSTOMERS, 0.12)
    for i in range(N_CUSTOMERS):
        if product_counts[i] >= 3:
            churn_prob[i] *= 0.3  # deep relationships rarely churn
        if is_primary[i]:
            churn_prob[i] *= 0.5
        if base_scores[i] == "poor":
            churn_prob[i] *= 1.5
        if channels[i] == "referral":
            churn_prob[i] *= 0.6
    churn_flag = (rng.random(N_CUSTOMERS) < churn_prob).astype(int)

    return pd.DataFrame(
        {
            "customer_id": cids,
            "account_open_date": open_dates,
            "account_type": account_types,
            "acquisition_channel": channels,
            "credit_score_band": base_scores,
            "income_band": income_bands,
            "product_count": product_counts,
            "is_primary_bank": is_primary,
            "avg_monthly_balance": np.round(avg_balance, 2),
            "credit_utilization_ratio": np.round(credit_util, 4),
            "overdraft_events_count": overdraft_counts,
            "net_interest_margin_contribution": np.round(nim, 2),
            "fee_revenue": np.round(fee_rev, 2),
            "churn_flag": churn_flag,
        }
    )


def _make_transactions(
    customers: pd.DataFrame, rng: np.random.Generator
) -> pd.DataFrame:
    """Generate transaction-level data (~80k rows)."""
    end_date = datetime(2025, 12, 31)
    rows = []

    avg_txns_per_cust = TARGET_TRANSACTIONS / N_CUSTOMERS  # ~6.67

    mcc_map = {
        "current": ["5411", "5812", "5541", "4829", "6012", "5999"],
        "savings": ["6012", "6211"],
        "credit_card": ["5411", "5812", "5912", "5541", "5311", "5999", "4511"],
        "loan": ["6012", "6300"],
    }

    txn_types_map = {
        "current": (["debit", "credit", "transfer", "fee"], [0.40, 0.30, 0.20, 0.10]),
        "savings": (["credit", "transfer", "interest"], [0.40, 0.35, 0.25]),
        "credit_card": (
            ["debit", "credit", "fee", "interest"],
            [0.45, 0.25, 0.15, 0.15],
        ),
        "loan": (["debit", "credit", "fee", "interest"], [0.30, 0.35, 0.15, 0.20]),
    }

    for _, cust in customers.iterrows():
        open_dt = cust["account_open_date"]
        if isinstance(open_dt, str):
            open_dt = pd.Timestamp(open_dt)

        # Credit card & primary bank → more transactions
        multiplier = 1.0
        if cust["account_type"] == "credit_card":
            multiplier *= 1.8
        if cust["is_primary_bank"]:
            multiplier *= 1.4
        if cust["product_count"] >= 3:
            multiplier *= 1.3
        if cust["churn_flag"]:
            multiplier *= 0.4  # Churned customers have fewer transactions

        n_txns = max(1, int(rng.poisson(avg_txns_per_cust * multiplier)))

        # Spread transactions between open date and end date
        days_active = (end_date - open_dt.to_pydatetime()).days
        if days_active <= 0:
            days_active = 1

        txn_offsets = rng.integers(0, days_active, size=n_txns)
        acct = cust["account_type"]
        types, probs = txn_types_map[acct]
        mccs = mcc_map[acct]

        for offset in txn_offsets:
            txn_date = open_dt + pd.Timedelta(days=int(offset))
            txn_type = rng.choice(types, p=probs)

            # Amount varies by transaction type and balance
            base_amt = cust["avg_monthly_balance"] * 0.05
            if txn_type == "fee":
                amt = rng.uniform(1, 35)
            elif txn_type == "interest":
                amt = rng.uniform(0.5, max(1.0, cust["avg_monthly_balance"] * 0.005 + 5))
            elif txn_type == "transfer":
                amt = rng.uniform(10, max(50, base_amt * 3))
            else:
                amt = max(1, rng.normal(max(5, base_amt), max(1, base_amt * 0.5)))

            rows.append(
                {
                    "customer_id": cust["customer_id"],
                    "transaction_date": txn_date,
                    "transaction_type": txn_type,
                    "transaction_amount": round(abs(amt), 2),
                    "merchant_category_code": rng.choice(mccs),
                }
            )

    return pd.DataFrame(rows)


def generate_dataset(seed: int = SEED):
    """Generate both customers and transactions tables, save to data/."""
    rng = np.random.default_rng(seed)
    print("Generating customer master table …")
    customers = _make_customers(rng)

    print(f"Generating ~{TARGET_TRANSACTIONS:,} transactions …")
    transactions = _make_transactions(customers, rng)
    print(f"  → {len(transactions):,} transactions generated.")

    os.makedirs(DATA_DIR, exist_ok=True)
    cust_path = os.path.join(DATA_DIR, "customers.csv")
    txn_path = os.path.join(DATA_DIR, "transactions.csv")

    customers.to_csv(cust_path, index=False)
    transactions.to_csv(txn_path, index=False)
    print(f"Saved: {cust_path}")
    print(f"Saved: {txn_path}")

    return customers, transactions


if __name__ == "__main__":
    generate_dataset()
