"""
simulation.py
=============
Time integration of the vacuum chamber evacuation process.

The governing ODE is:
    dp/dt = -(S_eff(p) / V) * p + Q_total / V

where:
    p         = chamber pressure (mbar)
    S_eff(p)  = effective pumping speed at pressure p (L/s)
    V         = chamber volume (L)
    Q_total   = total gas load (leak rate + outgassing) in mbar·L/s

Integration is performed with a simple adaptive step-size RK4 scheme
that keeps accuracy across many decades of pressure.
"""

from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass, field
from typing import List

from physics import (
    conductance,
    effective_pumping_speed,
    knudsen_number,
    reynolds_number,
    mean_free_path,
    gas_throughput,
)
from flow_regimes import classify_flow_regime, FlowRegimeResult


# ─────────────────────────────────────────────
# Simulation result container
# ─────────────────────────────────────────────
@dataclass
class SimulationResult:
    """Stores the full time-series output of the evacuation simulation."""
    time:            np.ndarray          # s
    pressure:        np.ndarray          # mbar
    S_eff:           np.ndarray          # L/s
    conductance_C:   np.ndarray          # L/s
    throughput_Q:    np.ndarray          # mbar·L/s
    kn:              np.ndarray          # dimensionless
    re:              np.ndarray          # dimensionless
    mean_free_path:  np.ndarray          # m
    regimes:         List[str]           # flow regime label at each time step
    evacuation_time: float               # s to reach p_target (inf if not reached)
    ultimate_pressure: float             # mbar


# ─────────────────────────────────────────────
# Core ODE right-hand side
# ─────────────────────────────────────────────
def _dpdt(p: float, V: float, S_pump: float, d_mm: float, l_m: float,
          Q_total: float, T: float, gas: str) -> tuple[float, float, float]:
    """
    Compute dp/dt and return (dp/dt, S_eff, C) at the current pressure p.

    Returns
    -------
    tuple
        (dp_dt in mbar/s, S_eff in L/s, C in L/s)
    """
    p_safe = max(p, 1e-15)
    C      = conductance(d_mm, l_m, p_safe, T, gas)
    S_eff  = effective_pumping_speed(S_pump, C)
    dp_dt  = -(S_eff / V) * p_safe + Q_total / V
    return dp_dt, S_eff, C


# ─────────────────────────────────────────────
# Adaptive RK4 integrator
# ─────────────────────────────────────────────
def run_simulation(
    V: float,
    S_pump: float,
    d_mm: float,
    l_m: float,
    p_0: float,
    p_target: float,
    Q_leak: float,
    Q_outgassing: float,
    T: float = 293.15,
    gas: str = "Air",
    n_points: int = 500,
    t_max_factor: float = 5.0,
) -> SimulationResult:
    """
    Run the vacuum chamber evacuation simulation.

    Parameters
    ----------
    V : float
        Chamber volume in L.
    S_pump : float
        Nominal pump speed in L/s.
    d_mm : float
        Pipe inner diameter in mm.
    l_m : float
        Pipe length in m.
    p_0 : float
        Starting pressure in mbar.
    p_target : float
        Target pressure in mbar.
    Q_leak : float
        Leak rate in mbar·L/s.
    Q_outgassing : float
        Outgassing rate in mbar·L/s.
    T : float
        Temperature in K.
    gas : str
        Gas type key.
    n_points : int
        Number of output time points.
    t_max_factor : float
        Multiplier on estimated evacuation time to set simulation end time.

    Returns
    -------
    SimulationResult
    """
    Q_total = Q_leak + Q_outgassing

    # Estimate ultimate pressure
    C_mol_est  = conductance(d_mm, l_m, 1e-6, T, gas)
    S_eff_est  = effective_pumping_speed(S_pump, C_mol_est)
    p_end      = Q_total / S_eff_est if S_eff_est > 0 else float("inf")

    # Effective target: cannot go below ultimate pressure
    p_sim_target = max(p_target, p_end * 1.01, 1e-12)

    # Estimate total simulation time
    C_visc_est = conductance(d_mm, l_m, p_0, T, gas)
    S_eff_init = effective_pumping_speed(S_pump, C_visc_est)
    if S_eff_init > 0 and p_0 > p_sim_target:
        t_evac_est = (V / S_eff_init) * math.log(p_0 / p_sim_target)
    else:
        t_evac_est = V / max(S_eff_init, 1e-12)

    t_max = max(t_evac_est * t_max_factor, 1.0)

    # ── RK4 integration with adaptive step ─────────────────────────────────
    dt_init = t_max / (n_points * 10)
    dt_min  = 1e-6
    dt_max  = t_max / 20.0

    t_list   = [0.0]
    p_list   = [p_0]
    Seff_list = []
    C_list   = []

    t   = 0.0
    p   = p_0
    dt  = dt_init
    evacuation_time_result = float("inf")
    target_reached = False

    # Pre-compute initial S_eff and C
    dp1, S_eff_cur, C_cur = _dpdt(p, V, S_pump, d_mm, l_m, Q_total, T, gas)
    Seff_list.append(S_eff_cur)
    C_list.append(C_cur)

    while t < t_max and len(t_list) < 20000:
        # Adaptive step: shrink dt when pressure changes rapidly
        if abs(dp1) > 0 and p > 0:
            dt_adaptive = 0.05 * p / abs(dp1)
            dt = min(max(dt_adaptive, dt_min), dt_max)

        # RK4
        k1, _, _  = _dpdt(p,          V, S_pump, d_mm, l_m, Q_total, T, gas)
        k2, _, _  = _dpdt(p + dt/2*k1, V, S_pump, d_mm, l_m, Q_total, T, gas)
        k3, _, _  = _dpdt(p + dt/2*k2, V, S_pump, d_mm, l_m, Q_total, T, gas)
        k4, _, _  = _dpdt(p + dt*k3,   V, S_pump, d_mm, l_m, Q_total, T, gas)

        p_new = p + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        p_new = max(p_new, p_end * 0.999, 1e-15)

        t_new = t + dt

        # Record evacuation time when target is first crossed
        if not target_reached and p_new <= p_sim_target:
            # Linear interpolation for precise crossing time
            if p > p_sim_target:
                frac = (p - p_sim_target) / (p - p_new) if (p - p_new) != 0 else 0.5
                evacuation_time_result = t + frac * dt
            target_reached = True

        t_list.append(t_new)
        p_list.append(p_new)

        dp1, S_eff_cur, C_cur = _dpdt(p_new, V, S_pump, d_mm, l_m, Q_total, T, gas)
        Seff_list.append(S_eff_cur)
        C_list.append(C_cur)

        t = t_new
        p = p_new

        # Stop early if pressure is essentially at ultimate pressure
        if p <= p_end * 1.001 and p_end < float("inf"):
            break

    # ── Downsample to n_points ──────────────────────────────────────────────
    t_arr    = np.array(t_list)
    p_arr    = np.array(p_list)
    Seff_arr = np.array(Seff_list)
    C_arr    = np.array(C_list)

    if len(t_arr) > n_points:
        idx   = np.unique(np.linspace(0, len(t_arr) - 1, n_points, dtype=int))
        t_arr    = t_arr[idx]
        p_arr    = p_arr[idx]
        Seff_arr = Seff_arr[idx]
        C_arr    = C_arr[idx]

    # ── Derived arrays ──────────────────────────────────────────────────────
    Q_arr   = np.array([gas_throughput(pp, ss) for pp, ss in zip(p_arr, Seff_arr)])
    kn_arr  = np.array([knudsen_number(pp, d_mm, T, gas) for pp in p_arr])
    re_arr  = np.array([reynolds_number(pp, ss, d_mm, T, gas)
                        for pp, ss in zip(p_arr, Seff_arr)])
    lam_arr = np.array([mean_free_path(pp, T, gas) for pp in p_arr])

    regimes = []
    for pp, ss in zip(p_arr, Seff_arr):
        r = classify_flow_regime(pp, ss, d_mm, T, gas)
        regimes.append(r.regime)

    return SimulationResult(
        time=t_arr,
        pressure=p_arr,
        S_eff=Seff_arr,
        conductance_C=C_arr,
        throughput_Q=Q_arr,
        kn=kn_arr,
        re=re_arr,
        mean_free_path=lam_arr,
        regimes=regimes,
        evacuation_time=evacuation_time_result,
        ultimate_pressure=p_end,
    )
