"""
Main pipeline: generates data, fits models, computes CLV, runs all analyses.

Usage:
    python main.py
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

# Ensure project root is on path
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from data.generate_data import generate_dataset
from src.rfm import compute_rfm
from src.frequency_model import prepare_frequency_data, fit_bgnbd, predict_engagement, cohort_comparison
from src.monetary_model import prepare_monetary_data, fit_gamma_gamma, predict_monetary, validate_predictions
from src.clv import compute_clv, portfolio_summary
from src.fairness import run_fairness_checks, generate_fairness_report
from src.cohort import (
    build_cohort_table,
    plot_retention_heatmap,
    clv_by_channel,
    product_depth_evolution,
    cross_sell_propensity,
)

OUTPUT_DIR = os.path.join(ROOT, "outputs")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Step 1: Generate synthetic data ──────────────────────────
    print("=" * 60)
    print("STEP 1 — Generating synthetic fintech dataset")
    print("=" * 60)
    customers, transactions = generate_dataset()
    print(f"  Customers: {len(customers):,}")
    print(f"  Transactions: {len(transactions):,}")

    # ── Step 2: RFM Analysis ─────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 2 — RFM Analysis (banking-adapted)")
    print("=" * 60)
    rfm_df = compute_rfm(customers, transactions)
    rfm_df.to_csv(os.path.join(OUTPUT_DIR, "rfm_table.csv"))
    print(f"  Segments:\n{rfm_df['segment'].value_counts().to_string()}")

    # ── Step 3: BG/NBD Frequency Model ───────────────────────────
    print("\n" + "=" * 60)
    print("STEP 3 — BG/NBD Frequency Model")
    print("=" * 60)
    freq_summary = prepare_frequency_data(customers, transactions)
    bgf = fit_bgnbd(freq_summary)
    engagement = predict_engagement(bgf, freq_summary)

    print("\nCohort comparison (acquisition channels):")
    cohort_comp = cohort_comparison(engagement, customers)
    print(cohort_comp.to_string())
    cohort_comp.to_csv(os.path.join(OUTPUT_DIR, "channel_engagement_comparison.csv"))

    # ── Step 4: Gamma-Gamma Monetary Model ───────────────────────
    print("\n" + "=" * 60)
    print("STEP 4 — Gamma-Gamma Monetary Model")
    print("=" * 60)
    monetary_data = prepare_monetary_data(customers, transactions)
    ggf = fit_gamma_gamma(monetary_data, col="monetary_value")
    expected_monetary = predict_monetary(ggf, monetary_data, col="monetary_value")

    print("\nValidation:")
    validate_predictions(monetary_data["monetary_value"], expected_monetary)

    # ── Step 5: CLV Calculation ──────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 5 — CLV Calculation & Tier Segmentation")
    print("=" * 60)
    clv_df = compute_clv(engagement, expected_monetary, customers)
    clv_df.to_csv(os.path.join(OUTPUT_DIR, "clv_table.csv"))

    summary = portfolio_summary(clv_df)
    summary.to_csv(os.path.join(OUTPUT_DIR, "portfolio_summary.csv"))
    print("\nPortfolio Summary:")
    print(summary.to_string())

    # ── Step 6: Fairness Checks ──────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 6 — Fairness Analysis (Regulatory)")
    print("=" * 60)
    fairness_results = run_fairness_checks(clv_df)
    report = generate_fairness_report(fairness_results)

    report_path = os.path.join(OUTPUT_DIR, "fairness_report.md")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"  Fairness report saved → {report_path}")

    for dim, (detail_df, di_ratio, flagged) in fairness_results.items():
        status = "FAIL ⚠️" if flagged else "PASS ✅"
        print(f"  {dim}: DI ratio = {di_ratio:.4f} [{status}]")

    # ── Step 7: Cohort Analysis ──────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 7 — Cohort Analysis")
    print("=" * 60)

    # Retention heatmap
    cohort_pivot = build_cohort_table(customers, transactions)
    heatmap_path = os.path.join(OUTPUT_DIR, "cohort_retention_heatmap.png")
    plot_retention_heatmap(cohort_pivot, save_path=heatmap_path)

    # CLV by channel
    channel_clv = clv_by_channel(clv_df)
    channel_clv.to_csv(os.path.join(OUTPUT_DIR, "clv_by_channel.csv"))
    print("\nCLV by Acquisition Channel:")
    print(channel_clv.to_string())

    # Product depth evolution
    depth_evo = product_depth_evolution(customers, transactions)
    if not depth_evo.empty:
        depth_evo.to_csv(os.path.join(OUTPUT_DIR, "product_depth_evolution.csv"))
        print(f"\nProduct depth evolution: {len(depth_evo)} cohorts analysed")

    # Cross-sell propensity
    propensity = cross_sell_propensity(customers, clv_df)
    propensity.to_csv(os.path.join(OUTPUT_DIR, "cross_sell_propensity.csv"))
    top_prospects = propensity.head(20)
    print(f"\nTop 20 cross-sell prospects (mortgage):")
    print(top_prospects.to_string())

    # ── Summary ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"\nOutputs saved to: {OUTPUT_DIR}/")
    for f_name in sorted(os.listdir(OUTPUT_DIR)):
        fpath = os.path.join(OUTPUT_DIR, f_name)
        size = os.path.getsize(fpath)
        print(f"  {f_name:45s} {size:>10,} bytes")

    print(f"\n📊 To launch the dashboard:")
    print(f"   streamlit run dashboard/app.py")


if __name__ == "__main__":
    main()
