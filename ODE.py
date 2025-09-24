# Author: Julia Baumgarten

"""
Problem Description:
  Driven–damped harmonic oscillator:
    m x'' + c x' + k x = F_{0} cos(Omega t)

  We implement explicit Euler and RK4, validate against analytic results, and check key physical properties (energy conservation, resonance, phase).
 
Physics Set-up:
  - m : mass
  - k : spring constant, omega_{0} = sqrt(k/m)  (natural frequency)
  - c : damping coefficient, γ = c/m      (damping rate)
  - F_{0} cos(Omega t) : sinusoidal drive of amplitude F_{0} and angular frequency Omega

  Special cases and analytics:
    - Free SHO (no damping, no forcing):  c = 0, F0 = 0
      m x'' + k x = 0
      Exact solution: x(t) = x_{0} cos(omega_{0} t) + (v_{0}/omega_{0}) sin(omega_{0} t)
      Energy E(t) = 1/2 m v^2 + 1/2 k x^2 is conserved
    - Steady-state driven, damped oscillator:
      x_ss(t) = X(Omega) cos(Omega t − phi)
      X(Omega) = (F_{0}/m) / sqrt( (ω_{0}^2 − Omega^2)^2 + (gamma Omega)^2 )
      tan phi = (gamma Omega) / (omega_{0}^2 − Omega^2)
      Resonance peak (underdamped): Omega_peak ≈ sqrt( omega_{0}^2 − gamma^2/2 )
      Phase at resonance: phi ≈ pi/2 (90 degree).


Features (short version):
  1) Euler and RK4 integrators
  2) Phase portrait (x vs v) — ellipse when c = F0 = 0
  3) Energy drift comparison (Euler vs RK4) at same step size h
  4) Global error vs step size h (free SHO)
  5) Frequency response — amplitude & phase vs Omega with analytic
  6) Physical checks summary:
       - Energy conservation when c = F_{0} = 0
       - Resonance peak near sqrt(omega_{0}^2 − gamma^2/2)
       - Phase approx 90 degree at resonance
  5) phase_portrait     -> plot x–v curve (ellipse if c=F_{0}=0)
     energy_drift       -> print/plot energy vs time for Euler and RK4
     convergence        -> log–log global error vs h (free SHO analytic solution)
     frequency_response -> sweep Omega and compare amplitude & phase to analytic
     physical_checks    -> PASS/FAIL for required physical properties
     (Command Line Modes)

Usage examples:
  # Phase portrait (ellipse when c=F_{0}=0)
  python ODE.py phase_portrait --m 1 --k 1 --c 0 --F0 0 --x0 1 --v0 0 --h 0.02 --tmax 20 --method rk4

  # Energy drift (free SHO validation: force c=F_{0}=0)
  python ODE.py energy_drift --m 1 --k 1 --force_validation --h 0.05 --tmax 200

  # Convergence: global error vs step size for free SHO
  python ODE.py convergence --m 1 --k 1 --x0 1 --v0 0 --tmax 10 --h_list 0.4 0.2 0.1 0.05 0.025

  # Frequency response (amplitude & phase vs Omega) with analytic
  python ODE.py frequency_response --m 1 --k 1 --c 0.1 --F0 1 --h 0.01 \ --Omega_min 0.2 --Omega_max 2.0 --n_Omega 60 --method rk4

  # Physical checks (energy conservation, resonance, phase approx 90 degree)
  python ODE.py physical_checks --m 1 --k 1 --c 0.1 --F0 1 --h 0.01 --tmax 60
"""

from __future__ import annotations
# ----------------------------- Import ----------------------------- #
import argparse
import math
import sys
from dataclasses import dataclass
from typing import Callable, List, Tuple
import numpy as np

try:
    import matplotlib.pyplot as plt
    PLOT_OK = True
except Exception:
    PLOT_OK = False
# ----------------------------- Physics model & Parameters ----------------------------- 
@dataclass
class SHOParams:
    m: float = 1.0      # mass
    c: float = 0.0      # damping coefficient
    k: float = 1.0      # spring constant
    F0: float = 0.0     # driving amplitude
    Omega: float = 1.0  # driving angular frequency

def sho_rhs(t: float, y: np.ndarray, p: SHOParams) -> np.ndarray:
    """y = [x, v]; returns [x', v'] for m x'' + c x' + k x = F_{0} cos(Omega t)"""
    x, v = y
    drive = p.F0 * math.cos(p.Omega * t)
    a = (-p.c * v - p.k * x + drive) / p.m
    return np.array([v, a], dtype=float)

# ------------------------------------ Euler & RK4 ------------------------------------












