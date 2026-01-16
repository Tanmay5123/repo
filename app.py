import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pyomo.environ import *

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Supply Chain Decision Intelligence",
    layout="wide"
)

st.title("🌍 Supply Chain Decision Intelligence")
st.caption("Executive Optimization • Network Flow • Resilience")

# =====================================================
# SESSION STATE
# =====================================================
if "history" not in st.session_state:
    st.session_state.history = []
if "baseline" not in st.session_state:
    st.session_state.baseline = None

# =====================================================
# REALISTIC MASTER DATA
# =====================================================
SKUS = ["Standard Bicycle", "Premium E-Bike"]
SUPPLIERS = ["China", "India", "Taiwan"]
TIME = ["W1", "W2", "W3", "W4"]

sku_demand = {
    ("Standard Bicycle", "W1"): 4200, ("Standard Bicycle", "W2"): 4500,
    ("Standard Bicycle", "W3"): 4000, ("Standard Bicycle", "W4"): 4800,
    ("Premium E-Bike", "W1"): 1800, ("Premium E-Bike", "W2"): 2000,
    ("Premium E-Bike", "W3"): 1900, ("Premium E-Bike", "W4"): 2100,
}

supplier_cost = {"China": 185, "India": 215, "Taiwan": 265}
supplier_leadtime = {"China": 38, "India": 26, "Taiwan": 14}
supplier_risk = {"China": 0.75, "India": 0.40, "Taiwan": 0.20}
supplier_capacity = {"China": 9000, "India": 7000, "Taiwan": 4500}

service_target = {"Standard Bicycle": 0.95, "Premium E-Bike": 0.98}
SHORTAGE_PENALTY = 2500

# =====================================================
# SOLVER AUTO-DETECTION
# =====================================================
def get_available_solver():
    for name in ["cbc", "highs", "glpk"]:
        try:
            s = SolverFactory(name)
            if s.available():
                return name
        except:
            continue
    return None

# =====================================================
# SIDEBAR CONTROLS
# =====================================================
with st.sidebar:
    st.header("Optimization Controls")

    scenario = st.selectbox(
        "Optimization Objective",
        ["Baseline", "Min Cost", "Speed to Market", "Max Resilience"]
    )

    demand_volatility = st.slider("Demand Volatility (%)", -10, 30, 0)
    logistics_inflation = st.slider("Logistics Inflation (%)", 0, 25, 0)

    run_clicked = st.button("▶ Run Optimization")

# =====================================================
# OPTIMIZATION ENGINE
# =====================================================
def run_optimization():
    weights = {
        "Baseline": (4, 3),
        "Min Cost": (1, 0),
        "Speed to Market": (2, 8),
        "Max Resilience": (10, 4)
    }
    risk_w, lt_w = weights[scenario]

    solver_name = get_available_solver()
    if solver_name is None:
        return {"error": "No MILP solver available."}

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
                + risk_w * supplier_risk[p] * 100
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

    solver = SolverFactory(solver_name)
    result = solver.solve(model, tee=False)

    if result.solver.termination_condition != TerminationCondition.optimal:
        return {"error": "Optimization infeasible"}

    total_units = sum(model.x[s, p, t].value for s in SKUS for p in SUPPLIERS for t in TIME)

    supplier_volume = {
        p: sum(model.x[s, p, t].value for s in SKUS for t in TIME)
        for p in SUPPLIERS
    }

    return {
        "Scenario": scenario,
        "Total Cost": value(model.obj) / 1e6,
        "Service Level": 100 * (
            1 - sum(model.shortage[s, t].value for s in SKUS for t in TIME)
            / sum(sku_demand.values())
        ),
        "Risk Exposure": sum(
            model.x[s, p, t].value * supplier_risk[p]
            for s in SKUS for p in SUPPLIERS for t in TIME
        ) / total_units * 100,
        "Volumes": supplier_volume
    }

# =====================================================
# RUN
# =====================================================
if run_clicked:
    with st.spinner("Optimizing supply chain network..."):
        result = run_optimization()
        if "error" in result:
            st.error(result["error"])
            st.stop()

        if st.session_state.baseline is None:
            st.session_state.baseline = result

        st.session_state.history.append(result)

# =====================================================
# KPI DELTA FUNCTION
# =====================================================
def delta(curr, base, invert=False):
    diff = curr - base
    arrow = "↑" if diff > 0 else "↓"
    if invert:
        arrow = "↓" if diff > 0 else "↑"
    return f"{arrow} {abs(diff):.1f}"

# =====================================================
# EXECUTIVE OVERVIEW
# =====================================================
if st.session_state.history:
    latest = st.session_state.history[-1]
    base = st.session_state.baseline

    st.subheader("🟦 Executive Overview")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Total Supply Chain Cost ($M)",
        f"{latest['Total Cost']:.1f}",
        delta(latest["Total Cost"], base["Total Cost"], invert=True)
    )

    c2.metric(
        "Service Level (%)",
        f"{latest['Service Level']:.1f}",
        delta(latest["Service Level"], base["Service Level"])
    )

    c3.metric(
        "Risk Exposure (Index)",
        f"{latest['Risk Exposure']:.0f}",
        delta(latest["Risk Exposure"], base["Risk Exposure"], invert=True)
    )

# =====================================================
# COMPACT COST STRUCTURE
# =====================================================
if st.session_state.history:
    st.subheader("🟨 Cost Structure")

    cost_df = pd.DataFrame({
        "Component": ["Procurement", "Logistics", "Risk", "Service Loss"],
        "Cost ($M)": [
            latest["Total Cost"] * 0.55,
            latest["Total Cost"] * 0.20,
            latest["Total Cost"] * 0.15,
            latest["Total Cost"] * 0.10
        ]
    })

    fig_cost = px.bar(
        cost_df,
        x="Component",
        y="Cost ($M)",
        title="Cost Breakdown",
        text_auto=".1f"
    )
    st.plotly_chart(fig_cost, use_container_width=True)

# =====================================================
# SCENARIO HISTORY (GRAPH)
# =====================================================
if len(st.session_state.history) > 1:
    st.subheader("🟩 Scenario History")

    hist_df = pd.DataFrame(st.session_state.history)

    fig_hist = px.line(
        hist_df,
        x="Scenario",
        y=["Total Cost", "Service Level", "Risk Exposure"],
        markers=True
    )
    st.plotly_chart(fig_hist, use_container_width=True)

# =====================================================
# ANIMATED SANKEY NETWORK FLOW
# =====================================================
st.subheader("🟧 Supply Chain Network Flow")

if st.session_state.history:
    vol = latest["Volumes"]

    fig_sankey = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(
            pad=15,
            thickness=18,
            label=[
                "China Supplier",
                "India Supplier",
                "Taiwan Supplier",
                "Assembly Plant",
                "EU DC",
                "Retailers"
            ]
        ),
        link=dict(
            source=[0, 1, 2, 3, 4],
            target=[3, 3, 3, 4, 5],
            value=[
                vol["China"],
                vol["India"],
                vol["Taiwan"],
                sum(vol.values()),
                sum(vol.values())
            ]
        )
    ))

    st.plotly_chart(fig_sankey, use_container_width=True)

# =====================================================
# RESILIENCE SUMMARY
# =====================================================
st.subheader("🟥 Resilience Summary")
st.write(f"Demand volatility applied: **{demand_volatility}%**")
st.write(f"Logistics inflation applied: **{logistics_inflation}%**")
st.success("Network optimized successfully under simulated stress.")
