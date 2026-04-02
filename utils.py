"""
utils.py
========
Utility functions for the vacuum chamber simulation:
  - Input validation
  - Value formatting with appropriate units
  - Preset configurations
  - Physical interpretation text generation
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional


# ─────────────────────────────────────────────
# Input validation
# ─────────────────────────────────────────────
@dataclass
class ValidationResult:
    valid: bool
    warnings: list[str]
    errors: list[str]


def validate_inputs(
    V: float,
    S_pump: float,
    d_mm: float,
    l_m: float,
    p_0: float,
    p_target: float,
    Q_leak: float,
    Q_outgassing: float,
    T: float,
) -> ValidationResult:
    """
    Validate all simulation input parameters.

    Returns a ValidationResult with lists of warnings and errors.
    An error means the simulation cannot proceed; a warning is advisory.
    """
    errors   = []
    warnings = []

    if V <= 0:
        errors.append("Chamber volume V must be greater than 0.")
    if S_pump <= 0:
        errors.append("Pump speed S_pump must be greater than 0.")
    if d_mm <= 0:
        errors.append("Pipe diameter d must be greater than 0.")
    if l_m <= 0:
        errors.append("Pipe length l must be greater than 0.")
    if p_0 <= 0:
        errors.append("Starting pressure p_0 must be greater than 0.")
    if p_target <= 0:
        errors.append("Target pressure p_target must be greater than 0.")
    if Q_leak < 0:
        errors.append("Leak rate Q_leak cannot be negative.")
    if Q_outgassing < 0:
        errors.append("Outgassing rate Q_outgassing cannot be negative.")
    if T <= 0:
        errors.append("Temperature T must be greater than 0 K.")

    if not errors:
        if p_target >= p_0:
            warnings.append(
                "Target pressure is equal to or higher than starting pressure. "
                "No evacuation will occur."
            )
        if d_mm < 5:
            warnings.append(
                "Pipe diameter is very small (< 5 mm). "
                "Conductance will be very low and may dominate the system."
            )
        if l_m > 5:
            warnings.append(
                "Pipe length is large (> 5 m). "
                "Conductance losses may significantly limit effective pumping speed."
            )
        if T < 77:
            warnings.append("Temperature is very low (< 77 K). Cryogenic regime.")
        if T > 600:
            warnings.append("Temperature is very high (> 600 K). Check gas property validity.")

    return ValidationResult(valid=len(errors) == 0, warnings=warnings, errors=errors)


# ─────────────────────────────────────────────
# Number formatting
# ─────────────────────────────────────────────
def fmt_pressure(p: float) -> str:
    """Format a pressure value in mbar with appropriate precision."""
    if not math.isfinite(p):
        return "∞"
    if p >= 100:
        return f"{p:.1f} mbar"
    if p >= 1:
        return f"{p:.3f} mbar"
    if p >= 1e-3:
        return f"{p:.2e} mbar"
    return f"{p:.2e} mbar"


def fmt_time(t: float) -> str:
    """Format a time value in seconds, minutes, or hours."""
    if not math.isfinite(t):
        return "Not reachable"
    if t < 60:
        return f"{t:.1f} s"
    if t < 3600:
        return f"{t/60:.1f} min  ({t:.0f} s)"
    return f"{t/3600:.2f} h  ({t:.0f} s)"


def fmt_speed(s: float) -> str:
    """Format a pumping speed / conductance in L/s."""
    if not math.isfinite(s):
        return "∞ L/s"
    if s >= 1:
        return f"{s:.2f} L/s"
    return f"{s:.3e} L/s"


def fmt_throughput(q: float) -> str:
    """Format a gas throughput in mbar·L/s."""
    if not math.isfinite(q):
        return "∞ mbar·L/s"
    return f"{q:.3e} mbar·L/s"


def fmt_mfp(lam: float) -> str:
    """Format mean free path in appropriate units."""
    if not math.isfinite(lam):
        return "∞"
    if lam >= 1:
        return f"{lam:.2f} m"
    if lam >= 1e-3:
        return f"{lam*1e3:.2f} mm"
    if lam >= 1e-6:
        return f"{lam*1e6:.2f} µm"
    return f"{lam:.2e} m"


def fmt_kn(kn: float) -> str:
    """Format Knudsen number."""
    if not math.isfinite(kn):
        return "∞"
    return f"{kn:.3e}"


def fmt_re(re: float) -> str:
    """Format Reynolds number."""
    if not math.isfinite(re):
        return "∞"
    return f"{re:.1f}"


# ─────────────────────────────────────────────
# Preset configurations
# ─────────────────────────────────────────────
PRESETS = {
    "Small Laboratory Chamber": {
        "V":            10.0,     # L
        "S_pump":        5.0,     # L/s
        "d_mm":         25.0,     # mm
        "l_m":           0.5,     # m
        "p_0":        1013.0,     # mbar
        "p_target":    1e-4,      # mbar
        "Q_leak":      1e-7,      # mbar·L/s
        "Q_outgassing": 1e-7,     # mbar·L/s
        "T":           293.15,    # K
        "gas":         "Air",
    },
    "Medium Process Chamber": {
        "V":           200.0,
        "S_pump":      100.0,
        "d_mm":         63.0,
        "l_m":           1.0,
        "p_0":        1013.0,
        "p_target":    1e-5,
        "Q_leak":      1e-6,
        "Q_outgassing": 1e-6,
        "T":           293.15,
        "gas":         "Air",
    },
    "Large Industrial Plant": {
        "V":          2000.0,
        "S_pump":     1000.0,
        "d_mm":        200.0,
        "l_m":           3.0,
        "p_0":        1013.0,
        "p_target":    1e-4,
        "Q_leak":      1e-4,
        "Q_outgassing": 1e-4,
        "T":           293.15,
        "gas":         "Air",
    },
}


# ─────────────────────────────────────────────
# Leak component library
# ─────────────────────────────────────────────

# Each entry: (display_name, typical_Q_mbarls_per_unit, description)
# Q values are per single component / per meter of weld, etc.
LEAK_COMPONENTS: dict[str, dict] = {
    # ── Elastomer seals ────────────────────────────────────────────────────
    "O-Ring (Viton, DN16)":          {"Q": 1e-7,  "unit": "per seal",   "category": "Elastomer Seal",
                                       "note": "Viton (FKM) O-ring, DN16 flange. Typical for roughing/medium vacuum."},
    "O-Ring (Viton, DN40)":          {"Q": 2e-7,  "unit": "per seal",   "category": "Elastomer Seal",
                                       "note": "Viton (FKM) O-ring, DN40 flange."},
    "O-Ring (Viton, DN63)":          {"Q": 4e-7,  "unit": "per seal",   "category": "Elastomer Seal",
                                       "note": "Viton (FKM) O-ring, DN63 flange."},
    "O-Ring (Viton, DN100)":         {"Q": 6e-7,  "unit": "per seal",   "category": "Elastomer Seal",
                                       "note": "Viton (FKM) O-ring, DN100 flange."},
    "O-Ring (Viton, DN160)":         {"Q": 1e-6,  "unit": "per seal",   "category": "Elastomer Seal",
                                       "note": "Viton (FKM) O-ring, DN160 flange."},
    "O-Ring (NBR/Buna-N, DN40)":     {"Q": 5e-7,  "unit": "per seal",   "category": "Elastomer Seal",
                                       "note": "NBR O-ring. Higher permeation than Viton; not recommended for HV."},
    "O-Ring (EPDM, DN40)":           {"Q": 8e-7,  "unit": "per seal",   "category": "Elastomer Seal",
                                       "note": "EPDM O-ring. High gas permeation; rough vacuum only."},
    "O-Ring (Silicone, DN40)":       {"Q": 2e-6,  "unit": "per seal",   "category": "Elastomer Seal",
                                       "note": "Silicone O-ring. Very high permeation; rough vacuum only."},
    "O-Ring (PTFE, DN40)":           {"Q": 3e-8,  "unit": "per seal",   "category": "Elastomer Seal",
                                       "note": "PTFE O-ring. Lower permeation but poor elasticity."},
    # ── Metal seals ────────────────────────────────────────────────────────
    "CF Flange Copper Gasket (DN16)": {"Q": 1e-11, "unit": "per seal",  "category": "Metal Seal",
                                        "note": "ConFlat (CF) copper gasket. Standard for UHV applications."},
    "CF Flange Copper Gasket (DN40)": {"Q": 2e-11, "unit": "per seal",  "category": "Metal Seal",
                                        "note": "ConFlat (CF) copper gasket, DN40."},
    "CF Flange Copper Gasket (DN63)": {"Q": 3e-11, "unit": "per seal",  "category": "Metal Seal",
                                        "note": "ConFlat (CF) copper gasket, DN63."},
    "CF Flange Copper Gasket (DN100)":{"Q": 5e-11, "unit": "per seal",  "category": "Metal Seal",
                                        "note": "ConFlat (CF) copper gasket, DN100."},
    "CF Flange Aluminium Gasket (DN40)":{"Q": 5e-11,"unit": "per seal", "category": "Metal Seal",
                                          "note": "CF aluminium gasket. Slightly higher leak rate than copper."},
    "ISO-K Claw Clamp (DN40)":        {"Q": 1e-8,  "unit": "per seal",  "category": "Metal Seal",
                                        "note": "ISO-K claw clamp with centering ring. Medium vacuum."},
    "ISO-K Claw Clamp (DN100)":       {"Q": 3e-8,  "unit": "per seal",  "category": "Metal Seal",
                                        "note": "ISO-K claw clamp, DN100."},
    # ── Valves ─────────────────────────────────────────────────────────────
    "Gate Valve (DN40, Viton)": {"Q": 1e-7,  "unit": "per valve", "category": "Valve",
                                  "note": "Viton-sealed gate valve. Typical for HV systems."},
    "Gate Valve (DN63, Viton)": {"Q": 2e-7,  "unit": "per valve", "category": "Valve",
                                  "note": "Viton-sealed gate valve, DN63."},
    "Gate Valve (DN40, Metal)": {"Q": 1e-10, "unit": "per valve", "category": "Valve",
                                  "note": "All-metal gate valve for UHV."},
    "Angle Valve (DN16, Viton)":{"Q": 5e-8,  "unit": "per valve", "category": "Valve",
                                  "note": "Viton-sealed angle valve."},
    "Butterfly Valve (DN63)":   {"Q": 5e-6,  "unit": "per valve", "category": "Valve",
                                  "note": "Butterfly valve. Suitable for rough vacuum only."},
    "Needle Valve (small)":     {"Q": 1e-8,  "unit": "per valve", "category": "Valve",
                                  "note": "Small needle valve with metal tip."},
    # ── Feedthroughs ───────────────────────────────────────────────────────
    "Electrical Feedthrough (CF, 1 pin)":  {"Q": 1e-10, "unit": "per feedthrough", "category": "Feedthrough",
                                             "note": "Single-pin CF electrical feedthrough."},
    "Electrical Feedthrough (CF, 10 pin)": {"Q": 5e-10, "unit": "per feedthrough", "category": "Feedthrough",
                                             "note": "10-pin CF electrical feedthrough."},
    "Rotary Feedthrough (Viton)": {"Q": 1e-6, "unit": "per feedthrough", "category": "Feedthrough",
                                   "note": "Viton-sealed rotary feedthrough."},
    "Rotary Feedthrough (Ferrofluid)": {"Q": 1e-9, "unit": "per feedthrough", "category": "Feedthrough",
                                        "note": "Ferrofluidic rotary feedthrough. Very low leak rate."},
    "Viewport (CF, glass-metal)": {"Q": 1e-11, "unit": "per viewport", "category": "Feedthrough",
                                   "note": "CF viewport with glass-to-metal seal."},
    # ── Welds / Joints ─────────────────────────────────────────────────────
    "TIG Weld (stainless steel)": {"Q": 1e-11, "unit": "per meter", "category": "Weld",
                                   "note": "High-quality TIG weld on stainless steel. UHV-compatible."},
    "Brazed Joint (copper)": {"Q": 1e-10, "unit": "per joint", "category": "Weld",
                              "note": "Brazed copper joint."},
    # ── Custom ─────────────────────────────────────────────────────────────
    "Custom Component": {"Q": 1e-8, "unit": "per component", "category": "Custom",
                         "note": "User-defined component. Enter your own leak rate."},
}

# Sorted list of categories for grouping in the UI
LEAK_CATEGORIES = [
    "Elastomer Seal",
    "Metal Seal",
    "Valve",
    "Feedthrough",
    "Weld",
    "Custom",
]


def total_leak_rate(components: list[dict]) -> float:
    """
    Compute the total leak rate from a list of component dicts.

    Each dict must have keys:
      - 'name'  : str  — component name (key in LEAK_COMPONENTS)
      - 'count' : int  — number of identical components
      - 'Q_each': float — leak rate per component in mbar·L/s

    Returns
    -------
    float
        Total leak rate in mbar·L/s.
    """
    return sum(c["count"] * c["Q_each"] for c in components)


# ─────────────────────────────────────────────
# Physical interpretation text generator
# ─────────────────────────────────────────────
def generate_interpretation(
    S_pump: float,
    S_eff: float,
    C: float,
    p_end: float,
    p_target: float,
    Q_total: float,
    regime: str,
    kn: float,
    re: float,
    d_mm: float,
    V: float,
    evacuation_time: float,
) -> list[str]:
    """
    Generate a list of human-readable interpretation statements
    based on the current simulation state.

    Returns
    -------
    list[str]
        List of interpretation strings.
    """
    messages = []

    # ── Conductance vs. pump limitation ────────────────────────────────────
    if C > 0 and S_pump > 0:
        ratio = C / S_pump
        if ratio < 0.5:
            messages.append(
                f"The pipe conductance (C = {fmt_speed(C)}) is significantly lower than "
                f"the pump speed (S_pump = {fmt_speed(S_pump)}). "
                "The pipe is the dominant bottleneck — increasing the pump size will have "
                "little effect without also increasing the pipe diameter or reducing its length."
            )
        elif ratio < 1.5:
            messages.append(
                f"The pipe conductance (C = {fmt_speed(C)}) and pump speed "
                f"(S_pump = {fmt_speed(S_pump)}) are of similar magnitude. "
                "Both the pipe and the pump contribute equally to the effective pumping speed."
            )
        else:
            messages.append(
                f"The pump speed (S_pump = {fmt_speed(S_pump)}) is the dominant limitation. "
                f"The pipe conductance (C = {fmt_speed(C)}) is sufficiently high. "
                "Upgrading the pump will directly improve evacuation performance."
            )

    # ── Leak rate / gas load limitation ────────────────────────────────────
    if math.isfinite(p_end) and p_end > p_target * 0.9:
        messages.append(
            f"The total gas load (leak + outgassing = {fmt_throughput(Q_total)}) "
            f"limits the achievable ultimate pressure to {fmt_pressure(p_end)}. "
            "The desired target pressure cannot be reached under current conditions. "
            "Reduce the leak rate or outgassing to improve the ultimate pressure."
        )
    elif math.isfinite(p_end) and p_end > 1e-10:
        messages.append(
            f"The gas load sets an ultimate pressure of {fmt_pressure(p_end)}, "
            f"which is below the target pressure {fmt_pressure(p_target)}. "
            "The target pressure is reachable."
        )

    # ── Flow regime interpretation ──────────────────────────────────────────
    if "Molecular" in regime:
        messages.append(
            "The system is operating in the molecular flow regime. "
            "Gas molecules travel independently; conductance is pressure-independent "
            "and proportional to d³/l."
        )
    elif "Transitional" in regime and "Kn" not in regime and "Laminar" not in regime:
        messages.append(
            "The Reynolds number is in the transitional range (2300–4000). "
            "The flow may switch between laminar and turbulent behaviour unpredictably."
        )
    elif "Transitional" in regime and "Knudsen" in regime:
        messages.append(
            "The system is in the Knudsen transitional regime (0.01 ≤ Kn ≤ 1). "
            "Continuum fluid assumptions are becoming inaccurate. "
            "Both viscous and molecular effects are present."
        )
    elif "Laminar" in regime:
        messages.append(
            "The flow is viscous-laminar. "
            "Conductance is proportional to d⁴/l (Hagen-Poiseuille). "
            "Doubling the pipe diameter increases conductance by a factor of 16."
        )
    elif "Turbulent" in regime:
        messages.append(
            "The flow is viscous-turbulent (Re > 4000). "
            "Turbulent mixing increases flow resistance. "
            "Reducing the gas velocity (e.g., by using a larger pipe) will help."
        )

    # ── Pipe diameter sensitivity hint ─────────────────────────────────────
    if "Laminar" in regime and C < S_pump * 0.8:
        messages.append(
            "In the laminar regime, conductance scales with d⁴. "
            "A modest increase in pipe diameter will have a much stronger effect "
            "on evacuation time than increasing the pump speed."
        )

    # ── Evacuation time ─────────────────────────────────────────────────────
    if math.isfinite(evacuation_time):
        messages.append(
            f"Estimated evacuation time to reach target pressure: {fmt_time(evacuation_time)}."
        )
    else:
        messages.append(
            "The target pressure cannot be reached with the current configuration."
        )

    return messages
