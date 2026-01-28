import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# =========================
# PAGE CONFIG (LIGHT THEME)
# =========================
st.set_page_config(page_title="Supply Chain Decision Intelligence", layout="wide")

st.title("🌍 Supply Chain Decision Intelligence")
st.caption("Scenario Optimization • Network Flow • Executive KPIs")

# =========================
# SESSION STATE
# =========================
if "results" not in st.session_state:
    st.session_state.results = []
if "baseline" not in st.session_state:
    st.session_state.baseline = None

# =========================
# MASTER DATA (REALISTIC)
# =========================
SUPPLIERS = ["China", "India", "Taiwan"]

supplier_cost = {"China": 180, "India": 210, "Taiwan": 260}
supplier_leadtime = {"China": 38, "India": 26, "Taiwan": 14}
supplier_risk = {"China": 75, "India": 40, "Taiwan": 20}

weekly_demand = 12000  # EU weekly demand

# =========================
# SIDEBAR CONTROLS
# =========================
with st.sidebar:
    st.header("Optimization Controls")

    scenario = st.selectbox(
        "Optimization Objective",
        ["Baseline", "Min Cost", "Speed to Market", "Max Resilience"]
    )

    demand_volatility = st.slider("Demand Volatility (%)", -10, 30, 0)
    logistics_inflation = st.slider("Logistics Inflation (%)", 0, 25, 0)

    run_button = st.button("▶ Run Optimization")

# =========================
# OPTIMIZATION (SAFE HEURISTIC)
# =========================
def run_optimization(scenario, demand_volatility, logistics_inflation):

    demand = weekly_demand * (1 + demand_volatility / 100)

    if scenario == "Baseline":
        mix = {"China": 0.45, "India": 0.35, "Taiwan": 0.20}
        service = 96.5
    elif scenario == "Min Cost":
        mix = {"China": 0.60, "India": 0.25, "Taiwan": 0.15}
        service = 95.2
    elif scenario == "Speed to Market":
        mix = {"China": 0.20, "India": 0.30, "Taiwan": 0.50}
        service = 98.6
    else:  # Max Resilience
        mix = {"China": 0.25, "India": 0.40, "Taiwan": 0.35}
        service = 99.1

    volumes = {s: mix[s] * demand for s in SUPPLIERS}

    total_cost = sum(
        volumes[s] * supplier_cost[s] * (1 + logistics_inflation / 100)
        for s in SUPPLIERS
    )

    risk_exposure = sum(volumes[s] * supplier_risk[s] for s in SUPPLIERS) / demand

    return {
        "Scenario": scenario,
        "Total Cost": round(total_cost / 1e6, 2),
        "Service Level": round(service, 1),
        "Risk Exposure": round(risk_exposure, 0),
        "Volumes": volumes
    }

# =========================
# RUN BUTTON LOGIC
# =========================
if run_button:
    result = run_optimization(scenario, demand_volatility, logistics_inflation)

    if st.session_state.baseline is None:
        st.session_state.baseline = result

    st.session_state.results.append(result)

# =========================
# DELTA FUNCTION
# =========================
def delta(curr, base, invert=False):
    diff = curr - base
    arrow = "↑" if diff > 0 else "↓"
    if invert:
        arrow = "↓" if diff > 0 else "↑"
    return f"{arrow} {abs(diff):.1f}"

# =========================
# EXECUTIVE OVERVIEW
# =========================
if st.session_state.results:
    latest = st.session_state.results[-1]
    base = st.session_state.baseline

    st.subheader("🟦 Executive KPIs")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Total Supply Chain Cost ($M)",
        latest["Total Cost"],
        delta(latest["Total Cost"], base["Total Cost"], invert=True)
    )

    c2.metric(
        "Service Level (%)",
        latest["Service Level"],
        delta(latest["Service Level"], base["Service Level"])
    )

    c3.metric(
        "Risk Exposure (Index)",
        latest["Risk Exposure"],
        delta(latest["Risk Exposure"], base["Risk Exposure"], invert=True)
    )

# =========================
# COST STRUCTURE (PIE)
# =========================
if st.session_state.results:
    st.subheader("🟨 Cost Structure Analysis")

    cost_df = pd.DataFrame({
        "Component": ["Procurement", "Logistics", "Risk Premium"],
        "Cost": [
            latest["Total Cost"] * 0.6,
            latest["Total Cost"] * 0.25,
            latest["Total Cost"] * 0.15
        ]
    })

    fig_pie = px.pie(
        cost_df,
        names="Component",
        values="Cost",
        title="Cost Breakdown (%)"
    )
    st.plotly_chart(fig_pie, use_container_width=True)

# =========================
# SCENARIO HISTORY (TWO BAR CHARTS vs BASELINE)
# =========================
if len(st.session_state.results) > 1:
    st.subheader("🟩 Scenario History vs Baseline")

    hist_df = pd.DataFrame(st.session_state.results)
    base_row = hist_df.iloc[0]

    compare_df = hist_df.copy()
    compare_df["Cost Delta"] = compare_df["Total Cost"] - base_row["Total Cost"]
    compare_df["Service Delta"] = compare_df["Service Level"] - base_row["Service Level"]
    compare_df["Risk Delta"] = compare_df["Risk Exposure"] - base_row["Risk Exposure"]

    col1, col2 = st.columns(2)

    with col1:
        fig_cost = px.bar(
            compare_df,
            x="Scenario",
            y="Cost Delta",
            title="Cost Impact vs Baseline ($M)",
            text_auto=".2f"
        )
        st.plotly_chart(fig_cost, use_container_width=True)

    with col2:
        fig_sr = px.bar(
            compare_df,
            x="Scenario",
            y=["Service Delta", "Risk Delta"],
            barmode="group",
            title="Service & Risk Impact vs Baseline",
            text_auto=".2f"
        )
        st.plotly_chart(fig_sr, use_container_width=True)

# =========================
# SUPPLY CHAIN NETWORK FLOW (CLEAR SANKEY)
# =========================
st.subheader("🟧 Supply Chain Network Flow")

if st.session_state.results:
    v = latest["Volumes"]

    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(
            pad=20,
            thickness=25,
            line=dict(color="black", width=0.5),
            label=[
                "China Supplier",
                "India Supplier",
                "Taiwan Supplier",
                "Assembly Plant (India)",
                "EU Distribution Center",
                "Retailers"
            ]
        ),
        link=dict(
            source=[0, 1, 2, 3, 4],
            target=[3, 3, 3, 4, 5],
            value=[
                v["China"],
                v["India"],
                v["Taiwan"],
                sum(v.values()),
                sum(v.values())
            ]
        )
    ))

    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

# =========================
# FOOTER
# =========================
st.subheader("🟥 Resilience Summary")
st.write(f"Demand volatility applied: **{demand_volatility}%**")
st.write(f"Logistics inflation applied: **{logistics_inflation}%**")
st.success("Optimization executed successfully.")
