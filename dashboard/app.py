"""
Streamlit dashboard for Customer Lifetime Value analysis.

Tabs:
  1. Portfolio Overview — CLV segment distribution, total predicted revenue
  2. Customer Lookup — Search by customer_id → full CLV profile
  3. Acquisition Channel ROI — CLV by channel + costs = ROI
  4. Fairness Monitor — Disparate impact ratios, RAG status
  5. Targeting Export — Filter & export CSV for campaign targeting
"""

import os
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.fairness import run_fairness_checks, rag_status

# ───────────────────────── Page config ─────────────────────────
st.set_page_config(
    page_title="Fintech CLV Dashboard",
    page_icon="🏦",
    layout="wide",
)

# ───────────────────────── Data loading ────────────────────────
@st.cache_data
def load_data():
    base = os.path.join(os.path.dirname(__file__), "..")
    customers = pd.read_csv(os.path.join(base, "data", "customers.csv"))
    clv = pd.read_csv(os.path.join(base, "outputs", "clv_table.csv"), index_col=0)
    rfm = pd.read_csv(os.path.join(base, "outputs", "rfm_table.csv"), index_col=0)
    return customers, clv, rfm


try:
    customers, clv_df, rfm_df = load_data()
except FileNotFoundError:
    st.error(
        "Data files not found. Run `python main.py` first to generate the "
        "pipeline outputs (data/ and outputs/)."
    )
    st.stop()

# ───────────────────────── Sidebar ─────────────────────────────
st.sidebar.title("🏦 Fintech CLV")
st.sidebar.markdown("**Customer Lifetime Value** modelling for retail banking.")
st.sidebar.markdown("---")
st.sidebar.metric("Total Customers", f"{len(clv_df):,}")
st.sidebar.metric(
    "Portfolio CLV (36m)",
    f"£{clv_df['credit_risk_adj_clv_36m'].sum():,.0f}",
)
st.sidebar.metric(
    "Avg P(Alive)",
    f"{clv_df['p_alive'].mean():.1%}",
)

# ───────────────────────── Tabs ────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Portfolio Overview",
    "🔍 Customer Lookup",
    "📈 Acquisition Channel ROI",
    "⚖️ Fairness Monitor",
    "🎯 Targeting Export",
])

# ======================== TAB 1: Portfolio Overview =========================
with tab1:
    st.header("Portfolio Overview")

    col1, col2, col3, col4 = st.columns(4)
    tier_counts = clv_df["tier"].value_counts()
    total_rev = clv_df["credit_risk_adj_clv_36m"].sum()

    for i, (tier, count) in enumerate(tier_counts.items()):
        tier_rev = clv_df.loc[clv_df["tier"] == tier, "credit_risk_adj_clv_36m"].sum()
        [col1, col2, col3, col4][i % 4].metric(
            f"{tier}", f"{count:,} customers", f"£{tier_rev:,.0f}"
        )

    st.markdown("---")

    # CLV distribution by tier
    col_a, col_b = st.columns(2)

    with col_a:
        fig_tier = px.pie(
            clv_df,
            names="tier",
            title="Customer Count by Tier",
            color="tier",
            color_discrete_map={
                "Platinum": "#E5E4E2",
                "Gold": "#FFD700",
                "Silver": "#C0C0C0",
                "Bronze": "#CD7F32",
                "At Risk": "#FF4444",
            },
        )
        st.plotly_chart(fig_tier, use_container_width=True)

    with col_b:
        tier_summary = (
            clv_df.groupby("tier")["credit_risk_adj_clv_36m"]
            .sum()
            .reset_index(name="total_clv_36m")
        )
        fig_rev = px.bar(
            tier_summary.sort_values("total_clv_36m", ascending=True),
            x="total_clv_36m",
            y="tier",
            orientation="h",
            title="Total Portfolio Revenue by Tier (36m, £)",
            labels={"total_clv_36m": "CLV 36m (£)", "tier": "Tier"},
        )
        st.plotly_chart(fig_rev, use_container_width=True)

    # CLV distribution histogram
    fig_hist = px.histogram(
        clv_df,
        x="credit_risk_adj_clv_36m",
        nbins=80,
        color="tier",
        title="CLV (36m) Distribution by Tier",
        labels={"credit_risk_adj_clv_36m": "Risk-Adjusted CLV 36m (£)"},
        color_discrete_map={
            "Platinum": "#E5E4E2",
            "Gold": "#FFD700",
            "Silver": "#C0C0C0",
            "Bronze": "#CD7F32",
            "At Risk": "#FF4444",
        },
    )
    fig_hist.update_layout(xaxis_range=[0, clv_df["credit_risk_adj_clv_36m"].quantile(0.98)])
    st.plotly_chart(fig_hist, use_container_width=True)

    # Segment x Product count heatmap
    st.subheader("Average CLV by Tier × Product Count")
    cross = clv_df.pivot_table(
        index="tier",
        columns="product_count",
        values="credit_risk_adj_clv_36m",
        aggfunc="mean",
    ).round(0)
    st.dataframe(cross.style.format("£{:,.0f}"), use_container_width=True)


# ======================== TAB 2: Customer Lookup ===========================
with tab2:
    st.header("Customer Lookup")

    search_id = st.text_input(
        "Enter Customer ID (e.g. C000001)",
        max_chars=10,
    ).strip().upper()

    if search_id:
        if search_id in clv_df.index:
            c = clv_df.loc[search_id]
            st.success(f"Customer **{search_id}** found — Tier: **{c['tier']}**")

            col1, col2, col3 = st.columns(3)
            col1.metric("CLV 12m", f"£{c['clv_12m']:,.2f}")
            col2.metric("CLV 36m (risk-adj)", f"£{c['credit_risk_adj_clv_36m']:,.2f}")
            col3.metric("P(Alive)", f"{c['p_alive']:.1%}")

            col4, col5, col6 = st.columns(3)
            col4.metric("Products Held", int(c.get("product_count", 0)))
            col5.metric("Account Type", c.get("account_type", "N/A"))
            col6.metric("Acquisition Channel", c.get("acquisition_channel", "N/A"))

            st.markdown("---")

            st.subheader("Retention & Next Action")
            st.info(f"**Recommended Action**: {c.get('retention_action', 'N/A')}")

            # RFM details
            if search_id in rfm_df.index:
                r = rfm_df.loc[search_id]
                st.subheader("RFM Profile")
                rcol1, rcol2, rcol3, rcol4 = st.columns(4)
                rcol1.metric("Recency (days)", int(r["recency"]))
                rcol2.metric("Frequency (months)", int(r["frequency"]))
                rcol3.metric("Monetary (£/mo)", f"£{r['monetary']:.2f}")
                rcol4.metric("Segment", r["segment"])

            # Full record
            with st.expander("Full Customer Record"):
                st.json(c.to_dict())
        else:
            st.warning(f"Customer ID '{search_id}' not found.")


# ======================== TAB 3: Acquisition Channel ROI ===================
with tab3:
    st.header("Acquisition Channel ROI")

    acq_costs = {
        "branch": 250,
        "digital": 80,
        "referral": 120,
        "broker": 350,
        "employer_scheme": 60,
    }

    if "acquisition_channel" in clv_df.columns:
        ch_stats = (
            clv_df.groupby("acquisition_channel")
            .agg(
                customer_count=("clv_12m", "count"),
                avg_clv_12m=("clv_12m", "mean"),
                avg_clv_36m=("credit_risk_adj_clv_36m", "mean"),
                avg_p_alive=("p_alive", "mean"),
            )
            .round(2)
        )
        ch_stats["est_acq_cost"] = ch_stats.index.map(acq_costs)
        ch_stats["net_value_36m"] = ch_stats["avg_clv_36m"] - ch_stats["est_acq_cost"]
        ch_stats["roi_pct"] = (ch_stats["net_value_36m"] / ch_stats["est_acq_cost"] * 100).round(1)

        st.dataframe(
            ch_stats.style.format({
                "avg_clv_12m": "£{:,.2f}",
                "avg_clv_36m": "£{:,.2f}",
                "est_acq_cost": "£{:,.0f}",
                "net_value_36m": "£{:,.2f}",
                "roi_pct": "{:.1f}%",
                "avg_p_alive": "{:.1%}",
            }),
            use_container_width=True,
        )

        col_a, col_b = st.columns(2)

        with col_a:
            fig_ch = px.bar(
                ch_stats.reset_index(),
                x="acquisition_channel",
                y=["avg_clv_36m", "est_acq_cost"],
                barmode="group",
                title="CLV vs Acquisition Cost by Channel",
            )
            st.plotly_chart(fig_ch, use_container_width=True)

        with col_b:
            fig_roi = px.bar(
                ch_stats.reset_index().sort_values("roi_pct", ascending=True),
                x="roi_pct",
                y="acquisition_channel",
                orientation="h",
                title="36-Month ROI by Channel (%)",
                labels={"roi_pct": "ROI %", "acquisition_channel": "Channel"},
                color="roi_pct",
                color_continuous_scale="RdYlGn",
            )
            st.plotly_chart(fig_roi, use_container_width=True)
    else:
        st.warning("Acquisition channel data not available.")


# ======================== TAB 4: Fairness Monitor ==========================
with tab4:
    st.header("Fairness Monitor")
    st.markdown(
        "Regulatory check: CLV predictions should not systematically differ "
        "by credit score band, income band, or acquisition channel beyond "
        "the four-fifths rule threshold (0.80)."
    )

    fairness_results = run_fairness_checks(clv_df)

    for dim, (detail_df, di_ratio, flagged) in fairness_results.items():
        status_text = rag_status(di_ratio)
        color_map = {"GREEN": "🟢", "AMBER": "🟡", "RED": "🔴"}
        icon = color_map.get(status_text, "⚪")

        st.subheader(f"{icon} {dim}")
        st.metric("Disparate Impact Ratio", f"{di_ratio:.4f}", delta=None)

        st.dataframe(
            detail_df.style.format({"avg_clv_36m": "£{:,.2f}", "pct_of_max": "{:.1f}%"}),
            use_container_width=True,
        )

        if flagged:
            st.error(
                f"⚠️ Ratio ({di_ratio:.4f}) below 0.80 threshold. "
                "Investigate for potential proxy discrimination."
            )
        else:
            st.success("Within regulatory tolerance.")

        st.markdown("---")


# ======================== TAB 5: Targeting Export ==========================
with tab5:
    st.header("Campaign Targeting Export")
    st.markdown("Filter customers by tier and channel, then export for targeting.")

    col_f1, col_f2 = st.columns(2)

    with col_f1:
        selected_tiers = st.multiselect(
            "Select Tiers",
            options=sorted(clv_df["tier"].unique()),
            default=sorted(clv_df["tier"].unique()),
        )

    with col_f2:
        if "acquisition_channel" in clv_df.columns:
            selected_channels = st.multiselect(
                "Select Channels",
                options=sorted(clv_df["acquisition_channel"].unique()),
                default=sorted(clv_df["acquisition_channel"].unique()),
            )
        else:
            selected_channels = []

    # Filter
    mask = clv_df["tier"].isin(selected_tiers)
    if selected_channels and "acquisition_channel" in clv_df.columns:
        mask &= clv_df["acquisition_channel"].isin(selected_channels)

    filtered = clv_df[mask]

    st.write(f"**{len(filtered):,}** customers match the filter.")

    # P(alive) range filter
    p_range = st.slider("P(Alive) Range", 0.0, 1.0, (0.0, 1.0), step=0.05)
    filtered = filtered[
        (filtered["p_alive"] >= p_range[0]) & (filtered["p_alive"] <= p_range[1])
    ]
    st.write(f"After P(Alive) filter: **{len(filtered):,}** customers.")

    # Preview
    st.dataframe(filtered.head(100), use_container_width=True)

    # Export
    csv_data = filtered.to_csv()
    st.download_button(
        label="📥 Download Targeting CSV",
        data=csv_data,
        file_name="clv_targeting_export.csv",
        mime="text/csv",
    )
