# Author: Julia Baumgarten

# Does Euler and RK4 integrators for ODE module

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

