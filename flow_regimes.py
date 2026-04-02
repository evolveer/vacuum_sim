"""
flow_regimes.py
===============
Classification of gas flow regimes in vacuum systems.

Regime logic:
  - Kn < 0.01  → Viscous regime
      - Re < 2300  → Viscous-Laminar
      - 2300 ≤ Re ≤ 4000 → Viscous-Transitional (laminar→turbulent)
      - Re > 4000  → Viscous-Turbulent
  - 0.01 ≤ Kn ≤ 1 → Transitional (Knudsen) flow
  - Kn > 1    → Molecular flow
"""

from dataclasses import dataclass
from physics import knudsen_number, reynolds_number

# ─────────────────────────────────────────────
# Regime constants
# ─────────────────────────────────────────────
KN_VISCOUS_LIMIT      = 0.01   # Kn below this → viscous continuum
KN_MOLECULAR_LIMIT    = 1.0    # Kn above this → molecular flow
RE_LAMINAR_LIMIT      = 2300   # Re below this → laminar
RE_TURBULENT_LIMIT    = 4000   # Re above this → turbulent


@dataclass
class FlowRegimeResult:
    """Container for flow regime classification results."""
    regime: str           # Human-readable regime label
    kn: float             # Knudsen number
    re: float             # Reynolds number (0 if not in viscous regime)
    is_viscous: bool
    is_laminar: bool
    is_turbulent: bool
    is_transitional_kn: bool   # Knudsen transitional (between viscous and molecular)
    is_molecular: bool
    conductance_model: str     # Which conductance model is currently active
    description: str           # Short human-readable description


def classify_flow_regime(
    p_mbar: float,
    S_eff_ls: float,
    d_mm: float,
    T: float = 293.15,
    gas: str = "Air",
) -> FlowRegimeResult:
    """
    Classify the current gas flow regime based on Knudsen and Reynolds numbers.

    Parameters
    ----------
    p_mbar : float
        Current pressure in mbar.
    S_eff_ls : float
        Effective pumping speed in L/s.
    d_mm : float
        Pipe inner diameter in mm.
    T : float
        Temperature in K.
    gas : str
        Gas type key.

    Returns
    -------
    FlowRegimeResult
        Dataclass with regime classification and descriptive strings.
    """
    kn = knudsen_number(p_mbar, d_mm, T, gas)
    re = 0.0

    # ── Molecular flow ──────────────────────────────────────────────────────
    if kn > KN_MOLECULAR_LIMIT:
        return FlowRegimeResult(
            regime="Molecular Flow",
            kn=kn,
            re=0.0,
            is_viscous=False,
            is_laminar=False,
            is_turbulent=False,
            is_transitional_kn=False,
            is_molecular=True,
            conductance_model="Molecular conductance  (C ∝ d³/l)",
            description=(
                "The mean free path is much larger than the pipe diameter. "
                "Gas molecules travel independently without intermolecular collisions. "
                "Molecular conductance model is applied."
            ),
        )

    # ── Knudsen transitional flow ───────────────────────────────────────────
    if KN_VISCOUS_LIMIT <= kn <= KN_MOLECULAR_LIMIT:
        return FlowRegimeResult(
            regime="Transitional Flow (Knudsen)",
            kn=kn,
            re=0.0,
            is_viscous=False,
            is_laminar=False,
            is_turbulent=False,
            is_transitional_kn=True,
            is_molecular=False,
            conductance_model="Interpolated transitional model",
            description=(
                "The system is in the Knudsen transitional regime. "
                "Continuum assumptions are becoming inaccurate. "
                "An interpolated conductance between viscous and molecular models is used."
            ),
        )

    # ── Viscous regime (Kn < 0.01) ─────────────────────────────────────────
    re = reynolds_number(p_mbar, S_eff_ls, d_mm, T, gas)

    if re < RE_LAMINAR_LIMIT:
        regime = "Viscous-Laminar"
        is_laminar    = True
        is_turbulent  = False
        conductance_model = "Viscous pipe model  (C ∝ d⁴/l, Hagen-Poiseuille)"
        description = (
            "The flow is in the viscous continuum regime and is laminar. "
            "The Hagen-Poiseuille conductance model applies. "
            "Pressure-dependent conductance dominates system behaviour."
        )
    elif re > RE_TURBULENT_LIMIT:
        regime = "Viscous-Turbulent"
        is_laminar    = False
        is_turbulent  = True
        conductance_model = "Viscous pipe model  (turbulent correction applicable)"
        description = (
            "The flow is viscous but turbulent (Re > 4000). "
            "Turbulent mixing increases resistance. "
            "The Hagen-Poiseuille model is an approximation; turbulent corrections apply."
        )
    else:
        regime = "Viscous-Transitional (Laminar→Turbulent)"
        is_laminar    = False
        is_turbulent  = False
        conductance_model = "Viscous pipe model  (transitional Re range)"
        description = (
            "The Reynolds number is in the transitional range (2300–4000). "
            "The flow may switch between laminar and turbulent. "
            "Results carry increased uncertainty in this range."
        )

    return FlowRegimeResult(
        regime=regime,
        kn=kn,
        re=re,
        is_viscous=True,
        is_laminar=is_laminar,
        is_turbulent=is_turbulent,
        is_transitional_kn=False,
        is_molecular=False,
        conductance_model=conductance_model,
        description=description,
    )


def regime_color(regime: str) -> str:
    """Return a color string for plotting the given regime label."""
    mapping = {
        "Viscous-Laminar":                        "#2196F3",   # blue
        "Viscous-Transitional (Laminar→Turbulent)": "#FF9800", # orange
        "Viscous-Turbulent":                      "#F44336",   # red
        "Transitional Flow (Knudsen)":            "#9C27B0",   # purple
        "Molecular Flow":                         "#4CAF50",   # green
    }
    return mapping.get(regime, "#607D8B")
