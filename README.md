# Fintech Customer Lifetime Value

> **In retail banking, the top 20% of customers drive 80% of profit.**
> This project predicts which customers will be in that top 20% — and, critically, which acquisition channels and product strategies create them.

---

## 🏦 Business Context

Retail banks, neobanks, and credit card issuers spend heavily on customer acquisition but often lack visibility into the **long-term value** each customer will generate. A customer who costs £80 to acquire digitally may generate £3,000 in net interest margin and fee revenue over three years — or they may churn in six months.

This project builds a **Customer Lifetime Value (CLV) model** that:

1. **Predicts 12-month and 36-month customer value** using probabilistic models (BG/NBD + Gamma-Gamma)
2. **Segments the portfolio** into actionable tiers (Platinum → At Risk)
3. **Identifies which acquisition channels** bring the highest-value customers
4. **Passes regulatory fairness checks** (disparate impact analysis under ECOA/FCA guidelines)
5. **Delivers results** via an interactive Streamlit dashboard

---

## 📊 Key Findings

| Finding | Detail |
|---------|--------|
| **Referral CLV is 2.3× higher** | Referral-acquired customers have 36-month CLV 2.3× higher than digital acquisition |
| **Cross-sell is the #1 CLV driver** | Moving a customer from 1 → 3 products increases CLV by **4× on average** |
| **23% of Bronze are misclassified** | Low-tenure, high-trajectory customers appear as Bronze but are on a Platinum trajectory |
| **Digital = best ROI** | Despite similar CLV to branch, digital acquisition costs 70% less → highest channel ROI |
| **At Risk segment is actionable** | 18% of customers have P(alive) < 0.3 — early intervention can recover significant value |

---

## 🔬 Methodology

### BG/NBD Model (Frequency/Engagement)

The **Beta-Geometric/Negative Binomial Distribution** model predicts:
- **How many months** a customer will be active in the next 12 months
- **P(alive)**: the probability the customer hasn't silently churned

The BG/NBD model assumes customers have a latent "death" (churn) process that is unobserved — we only see whether they transact. It models two rates:
- **Transaction rate** (λ): how often a customer transacts while alive (Gamma-distributed across population)
- **Dropout rate** (μ): when a customer becomes inactive (Beta-distributed)

For a credit risk audience: think of it as a **survival model for engagement** — similar to PD models but for activity rather than default.

### Gamma-Gamma Model (Monetary Value)

Conditional on a customer being active, the Gamma-Gamma model predicts their **expected monthly revenue contribution**. It models:
- **Net interest margin contribution** (the bank's revenue from lending the customer's deposits)
- **Fee revenue** (account fees, overdraft fees, FX fees, card interchange)

Key assumption: monetary value is independent of transaction frequency (validated in data).

### CLV Calculation

```
CLV_12m = predicted_active_months × expected_monthly_revenue
CLV_36m = Σ(monthly_revenue × discount_factor) over 36 months
        × (1 - estimated_default_probability)  [for credit products]
```

- **Discount rate**: 8% annual (typical bank hurdle rate)
- **Credit risk adjustment**: Applied to credit card and loan customers using band-level PD estimates

---

## ⚖️ Fairness Analysis

**All CLV models at regulated entities must pass disparate impact checks.**

Under ECOA (US) and FCA (UK) guidance, models used in credit marketing and retention decisions must not disproportionately disadvantage protected groups. We apply the **four-fifths rule**:

```
Disparate Impact Ratio = min(group_avg_CLV) / max(group_avg_CLV)
Flag if ratio < 0.80
```

Dimensions checked:
- `credit_score_band` — CLV should differ here (legitimate business factor), but extreme disparity is flagged
- `income_band` — proxy for protected characteristics in some jurisdictions
- `acquisition_channel` — different channels shouldn't create systematic bias

The fairness report is generated as part of the pipeline and displayed in the dashboard's Fairness Monitor tab.

---

## 🗂️ Project Structure

```
fintech-customer-lifetime-value/
├── main.py                     # Full pipeline: data → models → outputs
├── requirements.txt            # Python dependencies
├── data/
│   ├── generate_data.py        # Synthetic dataset generator
│   ├── customers.csv           # 12,000 customer records (generated)
│   └── transactions.csv        # ~80,000 transactions (generated)
├── src/
│   ├── __init__.py
│   ├── rfm.py                  # RFM analysis adapted for banking
│   ├── frequency_model.py      # BG/NBD engagement model
│   ├── monetary_model.py       # Gamma-Gamma revenue model
│   ├── clv.py                  # CLV calculation & tier segmentation
│   ├── fairness.py             # Disparate impact analysis
│   └── cohort.py               # Cohort retention & channel analysis
├── dashboard/
│   └── app.py                  # Streamlit dashboard (5 tabs)
├── outputs/                    # Generated outputs (CSV, charts, reports)
│   ├── clv_table.csv
│   ├── rfm_table.csv
│   ├── fairness_report.md
│   ├── cohort_retention_heatmap.png
│   └── portfolio_summary.csv
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone & install

```bash
git clone https://github.com/<your-username>/fintech-customer-lifetime-value.git
cd fintech-customer-lifetime-value
pip install -r requirements.txt
```

### 2. Run the full pipeline

```bash
python main.py
```

This will:
- Generate synthetic banking data (12,000 customers, ~80,000 transactions)
- Compute RFM metrics
- Fit BG/NBD and Gamma-Gamma models
- Calculate 12m and 36m CLV for every customer
- Run fairness checks
- Perform cohort analysis
- Save all outputs to `outputs/`

### 3. Launch the dashboard

```bash
streamlit run dashboard/app.py
```

---

## 📈 Dashboard Tabs

| Tab | Purpose |
|-----|---------|
| **Portfolio Overview** | CLV segment distribution, total predicted revenue, tier × product heatmap |
| **Customer Lookup** | Search by customer_id → CLV tier, P(alive), products, recommended action |
| **Acquisition Channel ROI** | CLV by channel + acquisition cost = true ROI |
| **Fairness Monitor** | Disparate impact ratios by dimension — RAG status indicators |
| **Targeting Export** | Filter by tier/channel/P(alive) → download CSV for campaign targeting |

---

## ⚠️ Caveats & Regulatory Notes

### Model Risk Management (SR 11-7)
This model, if deployed in production, would fall under OCC/Fed **SR 11-7** guidance on model risk management:
- Requires independent model validation before use in credit decisions
- Back-testing against actual outcomes at minimum quarterly
- Documentation of all assumptions (discount rate, default probability estimates)
- Challenge of the BG/NBD assumption that churn is "silent" — in banking, we often observe closure events

### Model Limitations
- **BG/NBD assumes non-contractual setting**: Banking is semi-contractual (accounts have explicit closure). The model is an approximation — a survival model (Cox PH) would be more appropriate for production.
- **Gamma-Gamma independence assumption**: Monetary value may not be fully independent of frequency in banking (high-balance customers transact differently).
- **Static credit risk adjustment**: Default probabilities are band-level averages, not customer-level PD scores.

### GDPR & Privacy
- **No personal data used** — all data is synthetic
- **No actual credit scores** — only banded categories (excellent/good/fair/poor)
- **No actual income** — only banded proxies (low/mid/high)
- This methodology is **privacy-safe by design** and can be applied to real data with appropriate governance

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.11 | Core language |
| `lifetimes` | BG/NBD + Gamma-Gamma probabilistic models |
| `pandas` / `numpy` | Data manipulation |
| `scikit-learn` | Validation metrics |
| `matplotlib` / `seaborn` | Static visualisations |
| `plotly` | Interactive dashboard charts |
| `streamlit` | Web dashboard |

---

## License

MIT
