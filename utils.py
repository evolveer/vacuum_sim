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
