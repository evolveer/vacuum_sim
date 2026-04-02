"""
app.py
======
Main Streamlit application for the Vacuum Chamber Evacuation Simulation.

Layout:
  - Sidebar: Input sliders and preset selector
  - Main area:
      Tab 1 — Dashboard (KPIs + Chart A + Chart D)
      Tab 2 — Flow Regimes (Chart B + Chart E + regime detail)
      Tab 3 — Conductance (Chart C)
      Tab 4 — Formulas & Equations
      Tab 5 — Physical Interpretation
      Tab 6 — Export
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import math
import json
import io
import streamlit as st
import pandas as pd
import numpy as np

from physics import (
    conductance,
    effective_pumping_speed,
    mean_free_path,
    knudsen_number,
    reynolds_number,
    gas_throughput,
    ultimate_pressure,
    pressure_at_kn,
    GAS_PROPERTIES,
)
from flow_regimes import classify_flow_regime, regime_color
from simulation import run_simulation
from utils import (
    validate_inputs,
    fmt_pressure, fmt_time, fmt_speed, fmt_throughput, fmt_mfp, fmt_kn, fmt_re,
    generate_interpretation,
    PRESETS,
    LEAK_COMPONENTS,
    LEAK_CATEGORIES,
    total_leak_rate,
)
from plots import (
    plot_pressure_vs_time,
    plot_regime_vs_pressure,
    plot_conductance_vs_pressure,
    plot_throughput_vs_time,
    plot_kn_vs_time,
)

# ─────────────────────────────────────────────
# Page configuration
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Vacuum Chamber Evacuation Simulation",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🔬 Vacuum Chamber Evacuation Simulation")
st.caption(
    "Interactive physical simulation of vacuum chamber pump-down. "
    "Adjust parameters in the sidebar and observe the results in real time."
)

# ─────────────────────────────────────────────
# Sidebar — Inputs
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Parameters")

    # ── Preset selector ─────────────────────────────────────────────────────
    preset_names = ["Custom"] + list(PRESETS.keys())
    selected_preset = st.selectbox("Load Preset", preset_names, index=0)

    if selected_preset != "Custom":
        preset = PRESETS[selected_preset]
    else:
        preset = None

    def pv(key, default):
        """Return preset value if a preset is selected, else default."""
        return preset[key] if preset else default

    st.divider()

    # ── Mode toggle ─────────────────────────────────────────────────────────
    expert_mode = st.toggle("Expert Mode", value=False)

    st.divider()

    # ── Chamber ─────────────────────────────────────────────────────────────
    st.subheader("Chamber")
    V = st.slider(
        "Chamber Volume V (L)",
        min_value=1.0, max_value=5000.0,
        value=float(pv("V", 50.0)),
        step=1.0,
        help="Internal volume of the vacuum chamber in liters.",
    )

    # ── Pump ────────────────────────────────────────────────────────────────
    st.subheader("Pump")
    S_pump = st.slider(
        "Nominal Pump Speed S_pump (L/s)",
        min_value=0.1, max_value=1000.0,
        value=float(pv("S_pump", 20.0)),
        step=0.1,
        help="Nominal pumping speed of the vacuum pump in L/s.",
    )

    # ── Pipe ────────────────────────────────────────────────────────────────
    st.subheader("Pipe / Connection")
    d_mm = st.slider(
        "Pipe Diameter d (mm)",
        min_value=1.0, max_value=300.0,
        value=float(pv("d_mm", 40.0)),
        step=1.0,
        help="Inner diameter of the connecting pipe in mm.",
    )
    l_m = st.slider(
        "Pipe Length l (m)",
        min_value=0.05, max_value=10.0,
        value=float(pv("l_m", 0.5)),
        step=0.05,
        help="Length of the connecting pipe in m.",
    )

    # ── Pressure ────────────────────────────────────────────────────────────
    st.subheader("Pressure")
    p_0_exp = st.slider(
        "Starting Pressure p₀ (mbar, exponent)",
        min_value=-3, max_value=3,
        value=3,
        step=1,
        format="10^%d",
        help="Starting pressure as a power of 10 in mbar.",
    )
    p_0 = 10.0 ** p_0_exp
    st.caption(f"p₀ = **{fmt_pressure(p_0)}**")

    p_target_exp = st.slider(
        "Target Pressure p_target (mbar, exponent)",
        min_value=-10, max_value=2,
        value=-4,
        step=1,
        format="10^%d",
        help="Desired target pressure as a power of 10 in mbar.",
    )
    p_target = 10.0 ** p_target_exp
    st.caption(f"p_target = **{fmt_pressure(p_target)}**")

    # ── Gas load — component-based leak rate builder ─────────────────────────
    st.subheader("Gas Load — Leak Sources")

    # Initialise session state for the component list
    if "leak_components" not in st.session_state:
        st.session_state.leak_components = []   # list of dicts

    ignore_leak = st.checkbox("Ignore All Leaks", value=False)

    if not ignore_leak:
        # ── Add a new component ───────────────────────────────────────────────
        with st.expander("➕ Add Leak Source", expanded=len(st.session_state.leak_components) == 0):
            # Category filter
            cat_filter = st.selectbox(
                "Category",
                options=["(All)"] + LEAK_CATEGORIES,
                key="lc_cat",
            )
            # Filtered component names
            if cat_filter == "(All)":
                comp_names = list(LEAK_COMPONENTS.keys())
            else:
                comp_names = [
                    k for k, v in LEAK_COMPONENTS.items()
                    if v["category"] == cat_filter
                ]

            selected_comp = st.selectbox(
                "Component",
                options=comp_names,
                key="lc_name",
            )
            comp_info = LEAK_COMPONENTS[selected_comp]

            # Show note
            st.caption(f"_{comp_info['note']}_")
            st.caption(f"Typical: **{comp_info['Q']:.1e} mbar·L/s** {comp_info['unit']}")

            col_cnt, col_q = st.columns([1, 2])
            with col_cnt:
                count = st.number_input(
                    "Count", min_value=1, max_value=100, value=1,
                    step=1, key="lc_count",
                )
            with col_q:
                # Allow user to override the typical value
                q_exp_default = int(round(math.log10(comp_info["Q"])))
                q_exp = st.slider(
                    "Q per unit (exponent)",
                    min_value=-13, max_value=-2,
                    value=q_exp_default,
                    step=1,
                    format="10^%d",
                    key="lc_q_exp",
                )
                q_each = 10.0 ** q_exp
                st.caption(f"Q per unit = **{q_each:.2e} mbar·L/s**")

            if st.button("Add to List", type="primary"):
                st.session_state.leak_components.append({
                    "name":   selected_comp,
                    "count":  int(count),
                    "Q_each": q_each,
                    "unit":   comp_info["unit"],
                    "category": comp_info["category"],
                })
                st.rerun()

        # ── Current component list ────────────────────────────────────────────
        if st.session_state.leak_components:
            st.markdown("**Defined Leak Sources:**")
            for i, comp in enumerate(st.session_state.leak_components):
                c1, c2, c3 = st.columns([3, 2, 1])
                c1.markdown(
                    f"**{comp['count']}×** {comp['name']}"
                )
                c2.markdown(
                    f"`{comp['count'] * comp['Q_each']:.2e}` mbar·L/s"
                )
                if c3.button("🗑", key=f"del_{i}", help="Remove this component"):
                    st.session_state.leak_components.pop(i)
                    st.rerun()

            if st.button("🗑 Clear All"):
                st.session_state.leak_components.clear()
                st.rerun()

            Q_leak = total_leak_rate(st.session_state.leak_components)
            st.success(f"**Total Q_L = {fmt_throughput(Q_leak)}**")
        else:
            st.info("No leak sources defined. Add components above or ignore all leaks.")
            Q_leak = 0.0
    else:
        Q_leak = 0.0

    # ── Expert: Outgassing, Temperature, Gas type ────────────────────────────
    if expert_mode:
        st.subheader("Advanced")
        consider_outgassing = st.checkbox("Consider Outgassing", value=False)
        Q_outgassing_exp = st.slider(
            "Outgassing Rate Q_out (mbar·L/s, exponent)",
            min_value=-12, max_value=-2,
            value=-8,
            step=1,
            format="10^%d",
            disabled=not consider_outgassing,
        )
        Q_outgassing = 10.0 ** Q_outgassing_exp if consider_outgassing else 0.0
        if consider_outgassing:
            st.caption(f"Q_out = **{fmt_throughput(Q_outgassing)}**")

        T = st.slider(
            "Temperature T (K)",
            min_value=77, max_value=600,
            value=int(pv("T", 293)),
            step=1,
            help="Gas temperature in Kelvin.",
        )
        gas = st.selectbox(
            "Gas Type",
            options=list(GAS_PROPERTIES.keys()),
            index=list(GAS_PROPERTIES.keys()).index(pv("gas", "Air")),
        )
    else:
        Q_outgassing = 0.0
        T = float(pv("T", 293.15))
        gas = pv("gas", "Air")

    st.divider()
    st.caption("Units: mbar · L · L/s · s · mm · m · K")

# ─────────────────────────────────────────────
# Input validation
# ─────────────────────────────────────────────
validation = validate_inputs(V, S_pump, d_mm, l_m, p_0, p_target,
                              Q_leak, Q_outgassing, T)

if not validation.valid:
    for err in validation.errors:
        st.error(f"❌ {err}")
    st.stop()

for warn in validation.warnings:
    st.warning(f"⚠️ {warn}")

# ─────────────────────────────────────────────
# Run simulation
# ─────────────────────────────────────────────
Q_total = Q_leak + Q_outgassing

with st.spinner("Running simulation…"):
    result = run_simulation(
        V=V, S_pump=S_pump, d_mm=d_mm, l_m=l_m,
        p_0=p_0, p_target=p_target,
        Q_leak=Q_leak, Q_outgassing=Q_outgassing,
        T=T, gas=gas,
    )

# ── Current-state quantities (at t=0 / starting pressure) ──────────────────
C_current    = conductance(d_mm, l_m, p_0, T, gas)
S_eff_current = effective_pumping_speed(S_pump, C_current)
kn_current   = knudsen_number(p_0, d_mm, T, gas)
re_current   = reynolds_number(p_0, S_eff_current, d_mm, T, gas)
lam_current  = mean_free_path(p_0, T, gas)
Q_current    = gas_throughput(p_0, S_eff_current)
p_end        = result.ultimate_pressure
regime_result = classify_flow_regime(p_0, S_eff_current, d_mm, T, gas)

# ── Threshold pressures ─────────────────────────────────────────────────────
p_visc_limit = pressure_at_kn(0.01, d_mm, T, gas)
p_mol_limit  = pressure_at_kn(1.0,  d_mm, T, gas)

# ─────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Dashboard",
    "🌊 Flow Regimes",
    "🔧 Conductance",
    "📐 Formulas",
    "💡 Interpretation",
    "📥 Export",
])

# ══════════════════════════════════════════════
# TAB 1 — Dashboard
# ══════════════════════════════════════════════
with tab1:
    st.subheader("Key Performance Indicators")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Evacuation Time", fmt_time(result.evacuation_time))
    col2.metric("Ultimate Pressure", fmt_pressure(p_end))
    col3.metric("Effective Speed S_eff", fmt_speed(S_eff_current))
    col4.metric("Conductance C", fmt_speed(C_current))

    col5, col6, col7, col8 = st.columns(4)
    col5.metric("Flow Regime", regime_result.regime)
    col6.metric("Knudsen Number Kn", fmt_kn(kn_current))
    col7.metric("Reynolds Number Re", fmt_re(re_current))
    col8.metric("Mean Free Path λ", fmt_mfp(lam_current))

    st.divider()

    col_a, col_b = st.columns([3, 2])
    with col_a:
        st.plotly_chart(
            plot_pressure_vs_time(result, p_target, p_end, result.evacuation_time),
            use_container_width=True,
        )
    with col_b:
        st.plotly_chart(
            plot_throughput_vs_time(result),
            use_container_width=True,
        )

    # ── Summary table ──────────────────────────────────────────────────────
    st.subheader("Parameter Summary")
    summary_data = {
        "Parameter": [
            "Chamber Volume V",
            "Nominal Pump Speed S_pump",
            "Pipe Diameter d",
            "Pipe Length l",
            "Starting Pressure p₀",
            "Target Pressure p_target",
            "Total Leak Rate Q_L",
            "Outgassing Q_out",
            "Temperature T",
            "Gas Type",
            "Total Gas Load Q_total",
        ],
        "Value": [
            f"{V:.1f} L",
            f"{S_pump:.2f} L/s",
            f"{d_mm:.1f} mm",
            f"{l_m:.2f} m",
            fmt_pressure(p_0),
            fmt_pressure(p_target),
            fmt_throughput(Q_leak),
            fmt_throughput(Q_outgassing),
            f"{T:.1f} K",
            gas,
            fmt_throughput(Q_total),
        ],
    }
    st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)

    # ── Individual leak source breakdown ───────────────────────────────────
    if st.session_state.get("leak_components"):
        st.subheader("Leak Source Breakdown")
        leak_rows = []
        for comp in st.session_state.leak_components:
            q_total_comp = comp["count"] * comp["Q_each"]
            pct = (q_total_comp / Q_leak * 100) if Q_leak > 0 else 0.0
            leak_rows.append({
                "Component":      comp["name"],
                "Category":       comp["category"],
                "Count":          comp["count"],
                "Q per unit (mbar·L/s)": f"{comp['Q_each']:.2e}",
                "Q total (mbar·L/s)":   f"{q_total_comp:.2e}",
                "Share (%)": f"{pct:.1f}",
            })
        st.dataframe(pd.DataFrame(leak_rows), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════
# TAB 2 — Flow Regimes
# ══════════════════════════════════════════════
with tab2:
    st.subheader("Flow Regime Overview")

    st.info(
        f"**Current regime at p₀ = {fmt_pressure(p_0)}:** "
        f"**{regime_result.regime}**  |  "
        f"Kn = {fmt_kn(kn_current)}  |  "
        f"Re = {fmt_re(re_current)}"
    )

    col_b1, col_b2 = st.columns([2, 1])
    with col_b1:
        st.plotly_chart(
            plot_regime_vs_pressure(d_mm, T, gas, p_min=1e-9, p_max=max(p_0 * 2, 1200)),
            use_container_width=True,
        )
    with col_b2:
        st.plotly_chart(
            plot_kn_vs_time(result),
            use_container_width=True,
        )

    st.subheader("Regime Threshold Pressures")
    col_t1, col_t2 = st.columns(2)
    col_t1.metric(
        "Viscous → Transitional (Kn = 0.01)",
        fmt_pressure(p_visc_limit),
        help="Below this pressure the system leaves the viscous continuum regime.",
    )
    col_t2.metric(
        "Transitional → Molecular (Kn = 1)",
        fmt_pressure(p_mol_limit),
        help="Below this pressure the system enters the molecular flow regime.",
    )

    st.subheader("Regime Description")
    st.markdown(f"> {regime_result.description}")
    st.markdown(f"**Active conductance model:** `{regime_result.conductance_model}`")

    # ── Regime timeline table ───────────────────────────────────────────────
    st.subheader("Regime Timeline")
    regime_changes = []
    prev = None
    for t_val, p_val, reg in zip(result.time, result.pressure, result.regimes):
        if reg != prev:
            regime_changes.append({
                "Time (s)": f"{t_val:.2f}",
                "Pressure (mbar)": f"{p_val:.3e}",
                "Regime": reg,
            })
            prev = reg
    if regime_changes:
        st.dataframe(pd.DataFrame(regime_changes), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════
# TAB 3 — Conductance
# ══════════════════════════════════════════════
with tab3:
    st.subheader("Conductance & Effective Pumping Speed vs. Pressure")
    st.plotly_chart(
        plot_conductance_vs_pressure(
            S_pump, d_mm, l_m, T, gas,
            p_min=1e-9, p_max=max(p_0 * 2, 1200),
        ),
        use_container_width=True,
    )

    st.subheader("Conductance at Key Pressures")
    key_pressures = [p_0, 100.0, 1.0, 1e-3, 1e-6, p_target]
    key_pressures = sorted(set([p for p in key_pressures if 1e-12 < p < 1e6]))
    rows = []
    for p_key in key_pressures:
        C_key    = conductance(d_mm, l_m, p_key, T, gas)
        Seff_key = effective_pumping_speed(S_pump, C_key)
        kn_key   = knudsen_number(p_key, d_mm, T, gas)
        rows.append({
            "Pressure (mbar)": f"{p_key:.2e}",
            "Conductance C (L/s)": fmt_speed(C_key),
            "S_eff (L/s)": fmt_speed(Seff_key),
            "Knudsen Kn": fmt_kn(kn_key),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════
# TAB 4 — Formulas
# ══════════════════════════════════════════════
with tab4:
    st.subheader("Applied Formulas & Equations")

    st.markdown(f"**Currently active model:** `{regime_result.conductance_model}`")
    st.divider()

    with st.expander("1. Pressure Decay (Ideal, No Gas Load)", expanded=True):
        st.latex(r"p(t) = p_0 \cdot e^{-\frac{S_\text{eff}}{V} \, t}")
        st.markdown(
            "The pressure decays exponentially with the time constant τ = V / S_eff. "
            "This formula applies in the ideal case without any gas load."
        )

    with st.expander("2. Evacuation Time Between Two Pressures"):
        st.latex(r"t = \frac{V}{S_\text{eff}} \ln\!\left(\frac{p_0}{p_1}\right)")
        st.markdown(
            "The time required to pump from pressure p₀ to p₁ depends linearly on "
            "the volume-to-speed ratio V/S_eff."
        )

    with st.expander("3. Effective Pumping Speed"):
        st.latex(
            r"\frac{1}{S_\text{eff}} = \frac{1}{S_\text{pump}} + \frac{1}{C}"
        )
        st.markdown(
            "The effective pumping speed at the chamber is always less than both "
            "the pump speed and the pipe conductance. "
            f"Current values: S_pump = {fmt_speed(S_pump)}, "
            f"C = {fmt_speed(C_current)}, "
            f"S_eff = {fmt_speed(S_eff_current)}."
        )

    with st.expander("4. Gas Throughput"):
        st.latex(r"Q = p \cdot S_\text{eff}")
        st.markdown(
            f"At the starting pressure: Q = {fmt_throughput(Q_current)}."
        )

    with st.expander("5. Ultimate Pressure"):
        st.latex(
            r"p_\text{end} = \frac{Q_\text{total}}{S_\text{eff}}"
            r"\quad \text{with} \quad Q_\text{total} = Q_L + Q_\text{outgassing}"
        )
        st.markdown(
            f"Total gas load Q_total = {fmt_throughput(Q_total)}. "
            f"Ultimate pressure p_end = {fmt_pressure(p_end)}."
        )

    with st.expander("6. Mean Free Path"):
        st.latex(
            r"\lambda = \frac{k_B \, T}{\sqrt{2}\,\pi\, d_m^2 \, p}"
        )
        st.markdown(
            f"At starting pressure p₀: λ = {fmt_mfp(lam_current)}. "
            "Note: pressure must be in Pa for this formula."
        )

    with st.expander("7. Knudsen Number"):
        st.latex(r"Kn = \frac{\lambda}{D}")
        st.markdown(
            "Kn < 0.01: viscous flow | 0.01 ≤ Kn ≤ 1: transitional | Kn > 1: molecular flow. "
            f"Current Kn = {fmt_kn(kn_current)}."
        )

    with st.expander("8. Reynolds Number"):
        st.latex(r"Re = \frac{\rho \, v \, D}{\eta}")
        st.markdown(
            "Re < 2300: laminar | 2300–4000: transitional | Re > 4000: turbulent. "
            f"Current Re = {fmt_re(re_current)}."
        )

    with st.expander("9. Viscous Pipe Conductance (Hagen-Poiseuille)"):
        st.latex(r"C_\text{visc} = \frac{\pi \, d^4 \, \bar{p}}{128 \, \eta \, l}")
        st.markdown(
            "Valid in the viscous continuum regime (Kn < 0.01). "
            "Conductance scales with the **fourth power** of the diameter."
        )

    with st.expander("10. Molecular Pipe Conductance (Knudsen)"):
        st.latex(
            r"C_\text{mol} = \frac{\pi}{12} \cdot \frac{d^3}{l} "
            r"\sqrt{\frac{8 R T}{\pi M}}"
        )
        st.markdown(
            "Valid in the molecular flow regime (Kn > 1). "
            "Conductance is pressure-independent and scales with d³."
        )


# ══════════════════════════════════════════════
# TAB 5 — Physical Interpretation
# ══════════════════════════════════════════════
with tab5:
    st.subheader("Physical Interpretation")

    messages = generate_interpretation(
        S_pump=S_pump,
        S_eff=S_eff_current,
        C=C_current,
        p_end=p_end,
        p_target=p_target,
        Q_total=Q_total,
        regime=regime_result.regime,
        kn=kn_current,
        re=re_current,
        d_mm=d_mm,
        V=V,
        evacuation_time=result.evacuation_time,
    )

    for i, msg in enumerate(messages, 1):
        st.info(f"**{i}.** {msg}")

    st.divider()
    st.subheader("Sensitivity Hints")

    # Pipe vs pump sensitivity
    if C_current > 0 and S_pump > 0:
        ratio = C_current / S_pump
        if ratio < 0.3:
            st.warning(
                "**Pipe-limited system.** The pipe conductance is much lower than the pump speed. "
                "Increasing the pipe diameter by 25% would approximately double the conductance "
                "(C ∝ d⁴ in laminar regime)."
            )
        elif ratio > 3.0:
            st.success(
                "**Pump-limited system.** The pipe conductance is well above the pump speed. "
                "The pump is the primary bottleneck."
            )
        else:
            st.info(
                "**Balanced system.** Pipe and pump contribute roughly equally. "
                "Improvements to either will have a meaningful effect."
            )

    # Leak rate check
    if math.isfinite(p_end) and p_end > p_target:
        st.error(
            f"**Target not reachable.** The ultimate pressure ({fmt_pressure(p_end)}) "
            f"is above the target ({fmt_pressure(p_target)}). "
            "Reduce the total gas load or increase the effective pumping speed."
        )

    # Temperature effect
    if expert_mode:
        st.markdown(
            f"At T = {T:.0f} K, the mean thermal velocity of {gas} molecules "
            f"affects the molecular conductance. "
            "Lower temperatures reduce outgassing but also slightly reduce molecular conductance."
        )


# ══════════════════════════════════════════════
# TAB 6 — Export
# ══════════════════════════════════════════════
with tab6:
    st.subheader("Export Simulation Data")

    # ── Build DataFrame ─────────────────────────────────────────────────────
    df_export = pd.DataFrame({
        "time_s":           result.time,
        "pressure_mbar":    result.pressure,
        "S_eff_ls":         result.S_eff,
        "conductance_ls":   result.conductance_C,
        "throughput_mbarls": result.throughput_Q,
        "knudsen_number":   result.kn,
        "reynolds_number":  result.re,
        "mean_free_path_m": result.mean_free_path,
        "flow_regime":      result.regimes,
    })

    # ── CSV export ──────────────────────────────────────────────────────────
    csv_buffer = io.StringIO()
    df_export.to_csv(csv_buffer, index=False)
    st.download_button(
        label="⬇️ Download Time-Series Data (CSV)",
        data=csv_buffer.getvalue(),
        file_name="vacuum_simulation_timeseries.csv",
        mime="text/csv",
    )

    # ── JSON parameter export ───────────────────────────────────────────────────────
    leak_components_export = [
        {
            "name":   c["name"],
            "category": c["category"],
            "count":  c["count"],
            "Q_each_mbarls": c["Q_each"],
            "Q_total_mbarls": c["count"] * c["Q_each"],
        }
        for c in st.session_state.get("leak_components", [])
    ]
    params_dict = {
        "V_L":              V,
        "S_pump_ls":        S_pump,
        "d_mm":             d_mm,
        "l_m":              l_m,
        "p_0_mbar":         p_0,
        "p_target_mbar":    p_target,
        "Q_leak_total_mbarls": Q_leak,
        "leak_components":  leak_components_export,
        "Q_outgassing_mbarls": Q_outgassing,
        "T_K":              T,
        "gas":              gas,
        "results": {
            "evacuation_time_s":  result.evacuation_time if math.isfinite(result.evacuation_time) else None,
            "ultimate_pressure_mbar": p_end if math.isfinite(p_end) else None,
            "S_eff_at_p0_ls":     S_eff_current,
            "conductance_at_p0_ls": C_current,
            "kn_at_p0":           kn_current,
            "re_at_p0":           re_current,
            "flow_regime_at_p0":  regime_result.regime,
        },
    }
    st.download_button(
        label="⬇️ Download Parameters & Results (JSON)",
        data=json.dumps(params_dict, indent=2),
        file_name="vacuum_simulation_params.json",
        mime="application/json",
    )

    # ── Preview ──────────────────────────────────────────────────────────────
    st.subheader("Data Preview (first 20 rows)")
    st.dataframe(df_export.head(20), use_container_width=True)
