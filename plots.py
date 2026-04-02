"""
plots.py
========
Chart generation functions for the vacuum chamber evacuation simulation.
All charts use Plotly for interactive display in Streamlit.

Charts provided:
  A. Pressure over Time          (log y-axis)
  B. Flow Regime over Pressure   (color-coded bands)
  C. Conductance & S_eff over Pressure
  D. Gas Throughput over Time
"""

from __future__ import annotations
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from simulation import SimulationResult
from flow_regimes import regime_color
from physics import (
    conductance,
    effective_pumping_speed,
    pressure_at_kn,
)
from flow_regimes import KN_VISCOUS_LIMIT, KN_MOLECULAR_LIMIT

# ─────────────────────────────────────────────
# Color palette
# ─────────────────────────────────────────────
COLOR_PRESSURE   = "#1565C0"
COLOR_TARGET     = "#E53935"
COLOR_ULTIMATE   = "#FF6F00"
COLOR_CONDUCTANCE = "#2E7D32"
COLOR_SEFF       = "#6A1B9A"
COLOR_THROUGHPUT = "#00838F"


# ─────────────────────────────────────────────
# Chart A: Pressure over Time
# ─────────────────────────────────────────────
def plot_pressure_vs_time(
    result: SimulationResult,
    p_target: float,
    p_end: float,
    evacuation_time: float,
) -> go.Figure:
    """
    Chart A: Pressure p(t) over time with logarithmic y-axis.
    Marks the target pressure and ultimate pressure as horizontal lines.
    Marks the evacuation time as a vertical line.
    """
    fig = go.Figure()

    # Pressure curve, colored by flow regime
    regime_labels = sorted(set(result.regimes))
    for label in regime_labels:
        mask = np.array([r == label for r in result.regimes])
        # Add a trace segment for each regime (with gaps between segments)
        indices = np.where(mask)[0]
        if len(indices) == 0:
            continue
        # Build segments of consecutive indices
        segments = []
        seg = [indices[0]]
        for i in range(1, len(indices)):
            if indices[i] == indices[i-1] + 1:
                seg.append(indices[i])
            else:
                segments.append(seg)
                seg = [indices[i]]
        segments.append(seg)

        for seg in segments:
            fig.add_trace(go.Scatter(
                x=result.time[seg],
                y=result.pressure[seg],
                mode="lines",
                name=label,
                line=dict(color=regime_color(label), width=2.5),
                showlegend=(seg == segments[0]),
                legendgroup=label,
            ))

    # Target pressure line
    fig.add_hline(
        y=p_target,
        line_dash="dash",
        line_color=COLOR_TARGET,
        annotation_text=f"Target: {p_target:.2e} mbar",
        annotation_position="bottom right",
    )

    # Ultimate pressure line
    if np.isfinite(p_end) and p_end > 0:
        fig.add_hline(
            y=p_end,
            line_dash="dot",
            line_color=COLOR_ULTIMATE,
            annotation_text=f"Ultimate: {p_end:.2e} mbar",
            annotation_position="top right",
        )

    # Evacuation time marker
    if np.isfinite(evacuation_time) and evacuation_time > 0:
        fig.add_vline(
            x=evacuation_time,
            line_dash="dashdot",
            line_color=COLOR_TARGET,
            annotation_text=f"t_evac = {evacuation_time:.1f} s",
            annotation_position="top left",
        )

    fig.update_layout(
        title="Chart A — Pressure over Time",
        xaxis_title="Time (s)",
        yaxis_title="Pressure (mbar)",
        yaxis_type="log",
        legend_title="Flow Regime",
        height=420,
        template="plotly_white",
        margin=dict(l=60, r=20, t=50, b=50),
    )
    return fig


# ─────────────────────────────────────────────
# Chart B: Flow Regime over Pressure
# ─────────────────────────────────────────────
def plot_regime_vs_pressure(
    d_mm: float,
    T: float,
    gas: str,
    p_min: float = 1e-9,
    p_max: float = 1100.0,
) -> go.Figure:
    """
    Chart B: Color-coded horizontal bands showing which flow regime
    is active at each pressure.
    Vertical markers at the viscous→transitional and transitional→molecular
    threshold pressures.
    """
    p_visc_limit = pressure_at_kn(KN_VISCOUS_LIMIT, d_mm, T, gas)
    p_mol_limit  = pressure_at_kn(KN_MOLECULAR_LIMIT, d_mm, T, gas)

    fig = go.Figure()

    # Colored bands
    band_data = [
        ("Molecular Flow",              p_min,         p_mol_limit,  "#4CAF50"),
        ("Transitional Flow (Knudsen)", p_mol_limit,   p_visc_limit, "#9C27B0"),
        ("Viscous Flow",                p_visc_limit,  p_max,        "#2196F3"),
    ]

    for label, x0, x1, color in band_data:
        if x0 >= x1:
            continue
        fig.add_shape(
            type="rect",
            x0=x0, x1=x1, y0=0, y1=1,
            xref="x", yref="paper",
            fillcolor=color,
            opacity=0.25,
            line_width=0,
        )
        x_mid = 10 ** ((np.log10(max(x0, 1e-12)) + np.log10(max(x1, 1e-12))) / 2)
        fig.add_annotation(
            x=x_mid, y=0.5,
            xref="x", yref="paper",
            text=label,
            showarrow=False,
            font=dict(size=11, color=color),
            textangle=-90,
        )

    # Threshold vertical lines
    fig.add_vline(
        x=p_visc_limit,
        line_dash="dash",
        line_color="#9C27B0",
        annotation_text=f"Kn=0.01<br>{p_visc_limit:.2e} mbar",
        annotation_position="top right",
        annotation_font_size=10,
    )
    fig.add_vline(
        x=p_mol_limit,
        line_dash="dash",
        line_color="#4CAF50",
        annotation_text=f"Kn=1<br>{p_mol_limit:.2e} mbar",
        annotation_position="top left",
        annotation_font_size=10,
    )

    # Dummy traces for legend
    for label, _, _, color in band_data:
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode="markers",
            marker=dict(size=12, color=color, symbol="square"),
            name=label,
        ))

    fig.update_layout(
        title="Chart B — Flow Regime over Pressure",
        xaxis_title="Pressure (mbar)",
        xaxis_type="log",
        xaxis_range=[np.log10(max(p_min, 1e-12)), np.log10(p_max)],
        yaxis=dict(visible=False),
        height=300,
        template="plotly_white",
        legend_title="Regime",
        margin=dict(l=60, r=20, t=50, b=50),
    )
    return fig


# ─────────────────────────────────────────────
# Chart C: Conductance and S_eff over Pressure
# ─────────────────────────────────────────────
def plot_conductance_vs_pressure(
    S_pump: float,
    d_mm: float,
    l_m: float,
    T: float,
    gas: str,
    p_min: float = 1e-9,
    p_max: float = 1100.0,
    n_points: int = 300,
) -> go.Figure:
    """
    Chart C: Pipe conductance C(p) and effective pumping speed S_eff(p)
    as a function of pressure on a log-log scale.
    """
    pressures = np.logspace(np.log10(max(p_min, 1e-12)), np.log10(p_max), n_points)
    C_vals    = np.array([conductance(p, d_mm, l_m, T, gas) for p in pressures])
    Seff_vals = np.array([effective_pumping_speed(S_pump, c) for c in C_vals])

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=pressures, y=C_vals,
        mode="lines",
        name="Conductance C(p)",
        line=dict(color=COLOR_CONDUCTANCE, width=2.5),
    ))

    fig.add_trace(go.Scatter(
        x=pressures, y=Seff_vals,
        mode="lines",
        name="Effective Speed S_eff(p)",
        line=dict(color=COLOR_SEFF, width=2.5, dash="dash"),
    ))

    fig.add_hline(
        y=S_pump,
        line_dash="dot",
        line_color="#B71C1C",
        annotation_text=f"S_pump = {S_pump:.1f} L/s",
        annotation_position="bottom right",
    )

    fig.update_layout(
        title="Chart C — Conductance & Effective Pumping Speed over Pressure",
        xaxis_title="Pressure (mbar)",
        yaxis_title="Speed / Conductance (L/s)",
        xaxis_type="log",
        yaxis_type="log",
        height=400,
        template="plotly_white",
        legend_title="Quantity",
        margin=dict(l=60, r=20, t=50, b=50),
    )
    return fig


# ─────────────────────────────────────────────
# Chart D: Gas Throughput over Time
# ─────────────────────────────────────────────
def plot_throughput_vs_time(result: SimulationResult) -> go.Figure:
    """
    Chart D: Gas throughput Q(t) over time on a log y-axis.
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=result.time,
        y=result.throughput_Q,
        mode="lines",
        name="Gas Throughput Q(t)",
        line=dict(color=COLOR_THROUGHPUT, width=2.5),
        fill="tozeroy",
        fillcolor="rgba(0,131,143,0.1)",
    ))

    fig.update_layout(
        title="Chart D — Gas Throughput over Time",
        xaxis_title="Time (s)",
        yaxis_title="Throughput Q (mbar·L/s)",
        yaxis_type="log",
        height=350,
        template="plotly_white",
        margin=dict(l=60, r=20, t=50, b=50),
    )
    return fig


# ─────────────────────────────────────────────
# Chart E: Knudsen number over Time
# ─────────────────────────────────────────────
def plot_kn_vs_time(result: SimulationResult) -> go.Figure:
    """
    Chart E: Knudsen number Kn(t) over time with regime threshold lines.
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=result.time,
        y=result.kn,
        mode="lines",
        name="Knudsen Number Kn(t)",
        line=dict(color="#795548", width=2.5),
    ))

    fig.add_hline(
        y=KN_VISCOUS_LIMIT,
        line_dash="dash",
        line_color="#9C27B0",
        annotation_text="Kn = 0.01 (viscous limit)",
        annotation_position="top right",
    )
    fig.add_hline(
        y=KN_MOLECULAR_LIMIT,
        line_dash="dash",
        line_color="#4CAF50",
        annotation_text="Kn = 1 (molecular limit)",
        annotation_position="top right",
    )

    fig.update_layout(
        title="Chart E — Knudsen Number over Time",
        xaxis_title="Time (s)",
        yaxis_title="Knudsen Number Kn",
        yaxis_type="log",
        height=350,
        template="plotly_white",
        margin=dict(l=60, r=20, t=50, b=50),
    )
    return fig
