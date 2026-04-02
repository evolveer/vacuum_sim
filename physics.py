"""
physics.py
==========
Physical models and formulas for vacuum chamber evacuation simulation.

All pressures are handled in mbar for the user interface.
Internally, SI units (Pa, m, m^3, m^3/s) are used where required by formulas.
"""

import math

# ─────────────────────────────────────────────
# Physical constants
# ─────────────────────────────────────────────
BOLTZMANN_CONSTANT = 1.380649e-23   # J/K  (k_B)
AVOGADRO_NUMBER    = 6.02214076e23  # 1/mol

# ─────────────────────────────────────────────
# Gas properties (molecular diameter d_m in m, molar mass M in kg/mol,
#                 dynamic viscosity eta in Pa·s at ~293 K)
# ─────────────────────────────────────────────
GAS_PROPERTIES = {
    "Air":      {"d_m": 3.7e-10, "M": 0.02897, "eta": 1.81e-5},
    "Nitrogen": {"d_m": 3.7e-10, "M": 0.02802, "eta": 1.76e-5},
    "Helium":   {"d_m": 2.6e-10, "M": 0.00400, "eta": 1.96e-5},
    "Argon":    {"d_m": 3.4e-10, "M": 0.03995, "eta": 2.27e-5},
}


# ─────────────────────────────────────────────
# Unit conversions
# ─────────────────────────────────────────────
def mbar_to_pa(p_mbar: float) -> float:
    """Convert pressure from mbar to Pa."""
    return p_mbar * 100.0


def pa_to_mbar(p_pa: float) -> float:
    """Convert pressure from Pa to mbar."""
    return p_pa / 100.0


def liters_to_m3(v_liters: float) -> float:
    """Convert volume from liters to m^3."""
    return v_liters * 1e-3


def m3_to_liters(v_m3: float) -> float:
    """Convert volume from m^3 to liters."""
    return v_m3 * 1e3


def ls_to_m3s(s_ls: float) -> float:
    """Convert pumping speed / conductance from L/s to m^3/s."""
    return s_ls * 1e-3


def m3s_to_ls(s_m3s: float) -> float:
    """Convert pumping speed / conductance from m^3/s to L/s."""
    return s_m3s * 1e3


def mm_to_m(d_mm: float) -> float:
    """Convert diameter from mm to m."""
    return d_mm * 1e-3


# ─────────────────────────────────────────────
# Effective pumping speed
# ─────────────────────────────────────────────
def effective_pumping_speed(S_pump: float, C: float) -> float:
    """
    Calculate the effective pumping speed at the chamber (L/s).

    Formula: 1/S_eff = 1/S_pump + 1/C

    Parameters
    ----------
    S_pump : float
        Nominal pump speed in L/s.
    C : float
        Pipe conductance in L/s.

    Returns
    -------
    float
        Effective pumping speed S_eff in L/s.
    """
    if S_pump <= 0 or C <= 0:
        return 0.0
    return 1.0 / (1.0 / S_pump + 1.0 / C)


# ─────────────────────────────────────────────
# Pipe conductance models
# ─────────────────────────────────────────────
def conductance_viscous(d_mm: float, l_m: float, p_avg_mbar: float) -> float:
    """
    Viscous (laminar) pipe conductance using the Hagen-Poiseuille approximation
    for air at room temperature (SI-based, returned in L/s).

    Formula (SI): C_visc = (pi * d^4 * p_avg) / (128 * eta * l)
    Converted to L/s.

    Parameters
    ----------
    d_mm : float
        Pipe inner diameter in mm.
    l_m : float
        Pipe length in m.
    p_avg_mbar : float
        Average pressure in mbar (arithmetic mean of inlet and outlet).

    Returns
    -------
    float
        Viscous conductance in L/s.
    """
    if d_mm <= 0 or l_m <= 0 or p_avg_mbar <= 0:
        return 0.0
    d = mm_to_m(d_mm)
    p_avg_pa = mbar_to_pa(p_avg_mbar)
    eta = 1.81e-5  # Pa·s, air at ~293 K (approximate for all gases here)
    C_m3s = (math.pi * d**4 * p_avg_pa) / (128.0 * eta * l_m)
    return m3s_to_ls(C_m3s)


def conductance_molecular(d_mm: float, l_m: float, T: float = 293.15,
                           gas: str = "Air") -> float:
    """
    Molecular flow pipe conductance using the Knudsen formula for a long tube.

    Formula: C_mol = (pi/12) * (d^3 / l) * sqrt(8 * R * T / (pi * M))
    where R = k_B * N_A is the universal gas constant.

    Parameters
    ----------
    d_mm : float
        Pipe inner diameter in mm.
    l_m : float
        Pipe length in m.
    T : float
        Temperature in K.
    gas : str
        Gas type key (from GAS_PROPERTIES).

    Returns
    -------
    float
        Molecular conductance in L/s.
    """
    if d_mm <= 0 or l_m <= 0:
        return 0.0
    d = mm_to_m(d_mm)
    M = GAS_PROPERTIES.get(gas, GAS_PROPERTIES["Air"])["M"]
    R = BOLTZMANN_CONSTANT * AVOGADRO_NUMBER  # 8.314 J/(mol·K)
    v_mean = math.sqrt(8.0 * R * T / (math.pi * M))  # mean thermal speed m/s
    C_m3s = (math.pi / 12.0) * (d**3 / l_m) * v_mean
    return m3s_to_ls(C_m3s)


def conductance_transitional(d_mm: float, l_m: float, p_avg_mbar: float,
                              T: float = 293.15, gas: str = "Air") -> float:
    """
    Transitional regime conductance using a smooth interpolation between
    viscous and molecular conductance based on the Knudsen number.

    A weighted blend: C = w * C_mol + (1-w) * C_visc
    where w = Kn / (Kn + 0.01) transitions smoothly from viscous to molecular.

    Parameters
    ----------
    d_mm : float
        Pipe inner diameter in mm.
    l_m : float
        Pipe length in m.
    p_avg_mbar : float
        Average pressure in mbar.
    T : float
        Temperature in K.
    gas : str
        Gas type key.

    Returns
    -------
    float
        Interpolated conductance in L/s.
    """
    kn = knudsen_number(p_avg_mbar, d_mm, T, gas)
    C_mol  = conductance_molecular(d_mm, l_m, T, gas)
    C_visc = conductance_viscous(d_mm, l_m, p_avg_mbar)
    # Smooth weight: 0 at Kn=0 (pure viscous), 1 at Kn>>1 (pure molecular)
    w = kn / (kn + 0.05) if kn > 0 else 0.0
    return w * C_mol + (1.0 - w) * C_visc


def conductance(d_mm: float, l_m: float, p_mbar: float,
                T: float = 293.15, gas: str = "Air") -> float:
    """
    Select and compute the appropriate conductance model based on the
    Knudsen number at the given pressure.

    Parameters
    ----------
    d_mm : float
        Pipe inner diameter in mm.
    l_m : float
        Pipe length in m.
    p_mbar : float
        Current pressure in mbar.
    T : float
        Temperature in K.
    gas : str
        Gas type key.

    Returns
    -------
    float
        Conductance C in L/s.
    """
    kn = knudsen_number(p_mbar, d_mm, T, gas)
    if kn < 0.01:
        return conductance_viscous(d_mm, l_m, p_mbar)
    elif kn > 1.0:
        return conductance_molecular(d_mm, l_m, T, gas)
    else:
        return conductance_transitional(d_mm, l_m, p_mbar, T, gas)


# ─────────────────────────────────────────────
# Mean free path
# ─────────────────────────────────────────────
def mean_free_path(p_mbar: float, T: float = 293.15, gas: str = "Air") -> float:
    """
    Calculate the mean free path of gas molecules (in m).

    Formula: lambda = k_B * T / (sqrt(2) * pi * d_m^2 * p)

    Parameters
    ----------
    p_mbar : float
        Pressure in mbar.
    T : float
        Temperature in K.
    gas : str
        Gas type key.

    Returns
    -------
    float
        Mean free path in m.
    """
    if p_mbar <= 0:
        return float("inf")
    p_pa = mbar_to_pa(p_mbar)
    d_m = GAS_PROPERTIES.get(gas, GAS_PROPERTIES["Air"])["d_m"]
    lam = BOLTZMANN_CONSTANT * T / (math.sqrt(2.0) * math.pi * d_m**2 * p_pa)
    return lam


# ─────────────────────────────────────────────
# Knudsen number
# ─────────────────────────────────────────────
def knudsen_number(p_mbar: float, d_mm: float, T: float = 293.15,
                   gas: str = "Air") -> float:
    """
    Calculate the Knudsen number Kn = lambda / D.

    Parameters
    ----------
    p_mbar : float
        Pressure in mbar.
    d_mm : float
        Pipe diameter in mm.
    T : float
        Temperature in K.
    gas : str
        Gas type key.

    Returns
    -------
    float
        Dimensionless Knudsen number.
    """
    if d_mm <= 0:
        return float("inf")
    lam = mean_free_path(p_mbar, T, gas)
    D = mm_to_m(d_mm)
    return lam / D


# ─────────────────────────────────────────────
# Reynolds number
# ─────────────────────────────────────────────
def reynolds_number(p_mbar: float, S_eff_ls: float, d_mm: float,
                    T: float = 293.15, gas: str = "Air") -> float:
    """
    Estimate the Reynolds number for viscous pipe flow.

    Re = (rho * v * D) / eta

    The mean flow velocity v is estimated from the volumetric flow rate Q_vol = S_eff
    and the pipe cross-section: v = Q_vol / A.
    The gas density rho is calculated from the ideal gas law.

    Parameters
    ----------
    p_mbar : float
        Pressure in mbar.
    S_eff_ls : float
        Effective pumping speed (volumetric flow rate) in L/s.
    d_mm : float
        Pipe diameter in mm.
    T : float
        Temperature in K.
    gas : str
        Gas type key.

    Returns
    -------
    float
        Reynolds number (dimensionless).
    """
    if d_mm <= 0 or S_eff_ls <= 0 or p_mbar <= 0:
        return 0.0
    p_pa  = mbar_to_pa(p_mbar)
    D     = mm_to_m(d_mm)
    M     = GAS_PROPERTIES.get(gas, GAS_PROPERTIES["Air"])["M"]
    eta   = GAS_PROPERTIES.get(gas, GAS_PROPERTIES["Air"])["eta"]
    R     = BOLTZMANN_CONSTANT * AVOGADRO_NUMBER
    # Ideal gas density: rho = p * M / (R * T)
    rho   = p_pa * M / (R * T)
    # Cross-section area
    A     = math.pi * (D / 2.0) ** 2
    # Volumetric flow rate in m^3/s
    Q_vol = ls_to_m3s(S_eff_ls)
    # Mean velocity
    v     = Q_vol / A if A > 0 else 0.0
    Re    = (rho * v * D) / eta if eta > 0 else 0.0
    return Re


# ─────────────────────────────────────────────
# Gas throughput
# ─────────────────────────────────────────────
def gas_throughput(p_mbar: float, S_ls: float) -> float:
    """
    Calculate gas throughput Q = p * S.

    Parameters
    ----------
    p_mbar : float
        Pressure in mbar.
    S_ls : float
        Pumping speed in L/s.

    Returns
    -------
    float
        Gas throughput Q in mbar·L/s.
    """
    return p_mbar * S_ls


# ─────────────────────────────────────────────
# Ultimate pressure
# ─────────────────────────────────────────────
def ultimate_pressure(Q_total_mbarls: float, S_eff_ls: float) -> float:
    """
    Calculate the ultimate (base) pressure limited by total gas load.

    p_end = Q_total / S_eff

    Parameters
    ----------
    Q_total_mbarls : float
        Total gas load (leak + outgassing) in mbar·L/s.
    S_eff_ls : float
        Effective pumping speed in L/s.

    Returns
    -------
    float
        Ultimate pressure p_end in mbar.
    """
    if S_eff_ls <= 0:
        return float("inf")
    return Q_total_mbarls / S_eff_ls


# ─────────────────────────────────────────────
# Pressure decay (ideal, no gas load)
# ─────────────────────────────────────────────
def pressure_at_time(p_0: float, S_eff: float, V: float, t: float) -> float:
    """
    Pressure at time t for ideal exponential decay without gas load.

    p(t) = p_0 * exp(- S_eff / V * t)

    Parameters
    ----------
    p_0 : float
        Starting pressure in mbar.
    S_eff : float
        Effective pumping speed in L/s.
    V : float
        Chamber volume in L.
    t : float
        Time in s.

    Returns
    -------
    float
        Pressure in mbar.
    """
    if V <= 0 or S_eff <= 0:
        return p_0
    return p_0 * math.exp(-S_eff / V * t)


# ─────────────────────────────────────────────
# Evacuation time
# ─────────────────────────────────────────────
def evacuation_time(p_0: float, p_1: float, S_eff: float, V: float) -> float:
    """
    Time required to pump from p_0 to p_1 (ideal, no gas load).

    t = (V / S_eff) * ln(p_0 / p_1)

    Parameters
    ----------
    p_0 : float
        Starting pressure in mbar.
    p_1 : float
        Target pressure in mbar.
    S_eff : float
        Effective pumping speed in L/s.
    V : float
        Chamber volume in L.

    Returns
    -------
    float
        Evacuation time in seconds.
    """
    if S_eff <= 0 or V <= 0 or p_1 <= 0 or p_0 <= p_1:
        return float("inf")
    return (V / S_eff) * math.log(p_0 / p_1)


# ─────────────────────────────────────────────
# Threshold pressures for regime transitions
# ─────────────────────────────────────────────
def pressure_at_kn(kn_target: float, d_mm: float, T: float = 293.15,
                   gas: str = "Air") -> float:
    """
    Calculate the pressure at which the Knudsen number equals kn_target.

    From Kn = lambda / D and lambda = k_B*T / (sqrt(2)*pi*d_m^2*p):
    p = k_B*T / (sqrt(2)*pi*d_m^2 * kn_target * D)

    Returns pressure in mbar.
    """
    if d_mm <= 0 or kn_target <= 0:
        return 0.0
    D   = mm_to_m(d_mm)
    d_m = GAS_PROPERTIES.get(gas, GAS_PROPERTIES["Air"])["d_m"]
    p_pa = BOLTZMANN_CONSTANT * T / (math.sqrt(2.0) * math.pi * d_m**2 * kn_target * D)
    return pa_to_mbar(p_pa)
