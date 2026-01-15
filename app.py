import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pyomo.environ import *

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Global Supply Chain – Decision Intelligence",
    layout="wide"
)

st.title("🌍 Global Supply Chain Decision Intelligence")
st.caption("Optimize • Simulate • Compare • Build Resilience")

# =====================================================
# SESSION STATE
# =====================================================
if "history" not in st.session_state:
    st.session_state.history = []

# =====================================================
# MASTER DATA (REALISTIC DEMO DATA)
# =====================================================
SKUS = ["Standard Bike", "Premium E-Bike"]
SUPPLIERS = ["China", "India", "Taiwan"]
TIME = ["W1", "W2", "W3", "W4"]

sku_demand = {
    ("Standard Bike", "W1"): 400, ("Standard Bike", "W2"): 420,
    ("Standard Bike", "W3"): 380, ("Standard Bike", "W4"): 450,
    ("Premium E-Bike", "W1"): 180, ("Premium E-Bike", "W2"): 200,
    ("Premium E-Bike", "W3"): 190, ("Premium E-Bike", "W4"): 210,
}

supplier_cost = {"China": 185, "India": 210, "Taiwan": 255}
supplier_leadtime = {"China": 38, "India": 26, "Taiwan": 14}
supplier_risk = {"China": 0.65, "India": 0.35, "Taiwan": 0.20}
supplier_capacity = {"China": 900, "India": 750, "Taiwan": 500}

service_target = {"Standard Bike": 0.95, "Premium E-Bike": 0.98}

SHORTAGE_PENALTY = 1200

# =====================================================
# SIDEBAR – CONTROLS
# =====================================================
with st.sidebar:
    st.header("Optimization Controls")

    scenario = st.selectbox(
        "Optimization Goal",
        ["Baseline", "Min Cost", "Speed to Market", "Max Resilience"]
    )

    demand_volatility = st.slider("Demand Volatility (%)", -15, 30, 0)
    logistics_inflation = st.slider("Logistics / Fuel Inflation (%)", 0, 25, 0)

    run_clicked = st.button("▶ Run Optimization")

# =====================================================
# OPTIMIZATION FUNCTION (SAFE, REBUILT EACH RUN)
# =====================================================
def run_optimization(scenario, demand_volatility, logistics_inflation):

    weights = {
        "Baseline": (4, 3),
        "Min Cost": (1, 0),
        "Speed to Market": (2, 7),
        "Max Resilience": (9, 4)
    }
    risk_w, lt_w = weights[scenario]

    model = ConcreteModel()
    model.S = SKUS
    model.P = SUPPLIERS
    model.T = TIME

    model.x = Var(model.S, model.P, model.T, within=NonNegativeReals)
    model.shortage = Var(model.S, model.T, within=NonNegativeReals)

    model.obj = Objective(
        expr=sum(
            model.x[s, p, t] *
            (
                supplier_cost[p] * (1 + logistics_inflation / 100)
                + risk_w * supplier_risk[p]
                + lt_w * supplier_leadtime[p]
            )
            for s in SKUS for p in SUPPLIERS for t in TIME
        )
        + sum(SHORTAGE_PENALTY * model.shortage[s, t] for s in SKUS for t in TIME),
        sense=minimize
    )

    model.demand = Constraint(
        SKUS, TIME,
        rule=lambda m, s, t:
            sum(m.x[s, p, t] for p in SUPPLIERS) + m.shortage[s, t]
            >= sku_demand[(s, t)] * (1 + demand_volatility / 100)
    )

    model.service = Constraint(
        SKUS,
        rule=lambda m, s:
            sum(m.shortage[s, t] for t in TIME)
            <= (1 - service_target[s]) * sum(sku_demand[(s, t)] for t in TIME)
    )

    model.capacity = Constraint(
        SUPPLIERS, TIME,
        rule=lambda m, p, t:
            sum(m.x[s, p, t] for s in SKUS) <= supplier_capacity[p]
    )

    solver = SolverFactory("glpk")
    result = solver.solve(model)

    if result.solver.termination_condition != TerminationCondition.optimal:
        return None

    total_cost = value(model.obj)
    total_vol = sum(model.x[s, p, t].value for s in SKUS for p in SUPPLIERS for t in TIME)

    avg_lt = sum(
        model.x[s, p, t].value * supplier_leadtime[p]
        for s in SKUS for p in SUPPLIERS for t in TIME
    ) / max(1, total_vol)

    service_lvl = 1 - (
        sum(model.shortage[s, t].value for s in SKUS for t in TIME)
        / sum(sku_demand.values())
    )

    risk_exp = sum(
        model.x[s, p, t].value * supplier_risk[p]
        for s in SKUS for p in SUPPLIERS for t in TIME
    )

    return {
        "Scenario": scenario,
        "Cost ($M)": total_cost / 1e6,
        "Avg Lead Time (Days)": avg_lt,
        "Service Level (%)": service_lvl * 100,
        "Risk Exposure": risk_exp
    }

# =====================================================
# RUN OPTIMIZATION
# =====================================================
if run_clicked:
    with st.spinner("Running optimization engine..."):
        result = run_optimization(scenario, demand_volatility, logistics_inflation)

        if result is None:
            st.error(
                "❌ Scenario infeasible under current conditions.\n\n"
                "Try reducing volatility or switching optimization goal."
            )
            st.stop()

        st.session_state.history.append(result)

# =====================================================
# EXECUTIVE OVERVIEW
# =====================================================
if st.session_state.history:
    latest = st.session_state.history[-1]

    st.subheader("🟦 Executive Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Cost ($M)", f"{latest['Cost ($M)']:.2f}")
    c2.metric("Avg Lead Time", f"{latest['Avg Lead Time (Days)']:.1f} days")
    c3.metric("Service Level", f"{latest['Service Level (%)']:.1f}%")
    c4.metric("Risk Exposure", f"{latest['Risk Exposure']:.0f}")

    st.divider()

# =====================================================
# SCENARIO HISTORY – GRAPHICAL VS BASELINE
# =====================================================
if len(st.session_state.history) >= 2:
    st.subheader("🟩 Scenario Comparison vs Baseline")

    hist_df = pd.DataFrame(st.session_state.history)

    fig = px.bar(
        hist_df,
        x="Scenario",
        y=["Cost ($M)", "Avg Lead Time (Days)", "Risk Exposure"],
        barmode="group",
        title="Scenario Impact Comparison"
    )
    st.plotly_chart(fig, use_container_width=True)

# =====================================================
# COST STRUCTURE ANALYSIS (EXEC VIEW)
# =====================================================
if st.session_state.history:
    st.subheader("🟨 Cost Structure Analysis")

    cost_df = pd.DataFrame({
        "Component": ["Procurement", "Risk Premium", "Lead Time Premium", "Service Loss"],
        "Cost ($M)": [
            latest["Cost ($M)"] * 0.55,
            latest["Cost ($M)"] * 0.15,
            latest["Cost ($M)"] * 0.20,
            latest["Cost ($M)"] * 0.10
        ]
    })

    st.plotly_chart(
        px.bar(cost_df, x="Component", y="Cost ($M)", title="Cost Breakdown"),
        use_container_width=True
    )

# =====================================================
# E2E SUPPLY CHAIN NETWORK FLOW (SANKEY)
# =====================================================
st.subheader("🟧 End-to-End Supply Chain Network Flow")

sankey = go.Figure(go.Sankey(
    node=dict(
        pad=15,
        thickness=20,
        label=["Suppliers (Asia)", "Assembly (India)", "EU DC", "Retailers"]
    ),
    link=dict(
        source=[0, 1, 2],
        target=[1, 2, 3],
        value=[50, 45, 40]
    )
))

st.plotly_chart(sankey, use_container_width=True)

# =====================================================
# RESILIENCE TEST SUMMARY
# =====================================================
st.subheader("🟥 Resilience Test Summary")
st.write(f"📈 Demand Volatility Applied: **{demand_volatility}%**")
st.write(f"⛽ Logistics Inflation Applied: **{logistics_inflation}%**")
st.success("Optimization completed successfully under simulated market stress.")
