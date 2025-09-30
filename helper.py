# Author: Julia Baumgarten

# Does Euler and RK4 integrators for ODE module
# Does Riemann, Trapezoid, Simpson's for integral module

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


# ------------------------------------ Euler & RK4 ------------------------------------
def euler_step(f: Callable, t: float, y: np.ndarray, h: float, p: SHOParams) -> np.ndarray:
    return y + h * f(t, y, p)

def rk4_step(f: Callable, t: float, y: np.ndarray, h: float, p: SHOParams) -> np.ndarray:
    k1 = f(t,           y,            p)
    k2 = f(t + 0.5*h,   y + 0.5*h*k1, p)
    k3 = f(t + 0.5*h,   y + 0.5*h*k2, p)
    k4 = f(t + h,       y + h*k3,     p)
    return y + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)


# ----------------------------- Quadrature rules ----------------------------- #

def riemann_midpoint(f: Callable[[np.ndarray], np.ndarray], a: float, b: float, N: int) -> float:
    """
    Midpoint Riemann sum on [a, b] with N uniform subintervals.
    Order O(h^2) for smooth f, where h = (b - a) / N
    """
    if N <= 0:
        raise ValueError("N must be positive.")
    h = (b - a) / N
    # midpoints of each subinterval
    m = (np.arange(N, dtype=float) + 0.5) * h + a
    return float(h * np.sum(f(m)))


def trapezoid(f: Callable[[np.ndarray], np.ndarray], a: float, b: float, N: int) -> float:
    """
    Composite trapezoidal rule on [a, b] with N panels (N+1 nodes).
    Global error O(h^2) for smooth f.
    """
    if N <= 0:
        raise ValueError("N must be positive.")
    h = (b - a) / N
    x = np.linspace(a, b, N + 1)
    y = f(x)
    return float(h * (0.5 * y[0] + np.sum(y[1:-1]) + 0.5 * y[-1]))


def simpson(f: Callable[[np.ndarray], np.ndarray], a: float, b: float, N: int) -> float:
    """
    Composite Simpson's rule on [a, b] with N panels (requires N even).
    Global error O(h^4) for smooth f.
    """
    if N <= 0:
        raise ValueError("N must be positive.")
    if N % 2 != 0:
        raise ValueError("Simpson's rule requires N to be even.")
    h = (b - a) / N
    x = np.linspace(a, b, N + 1)
    y = f(x)
    # weights: 1, 4, 2, 4, ..., 2, 4, 1
    S = y[0] + y[-1] + 4.0 * np.sum(y[1:-1:2]) + 2.0 * np.sum(y[2:-2:2])
    return float(h * S / 3.0)


