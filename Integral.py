# Author: Julia Baumgarten

"""
Problem Description:
  Coulomb's law in 1D geometry: Electric field on the perpendicular bisector of a uniformly charged finite line. 
  We integrate the smooth kernel and compare our numerical result against the analytic result and against SciPy.

Physics Set-up:
  Line of half-length L along x-axis with linear charge density λ.
  We evaluate E at the point (0, a) by symmetry, only the y-component survives:

    E_y(a) = (1 / (4π ε0)) * ∫_{-L}^{L} [ λ * a / (x^2 + a^2)^{3/2} ] dx

  We let K = λ / (4π ε0), so

    Ey(a, L) = K * ∫_{-L}^{L} a / (x^2 + a^2)^{3/2} dx
             = K * ( 2 L / (a * sqrt(a^2 + L^2)) )   

  This is our analytic closed form.


Features (short version):
  1) Riemann Midpoint
  2) Trapezoid
  3) Simpson
  4) E_{y} numeric and analytic
  5) error_vs_N     -> loglog error study vs N for each rule
     field_profile  -> E_{y} vs a curve, numeric vs analytic
     compare_scipy  -> numeric value vs SciPy quad/trapezoid/simpson
     (Command Line Modes)

Usage examples:
  python main_integral.py error_vs_N --L 1.0 --a 0.5 --N_list 20 40 80 160 --rules riemann trapezoid simpson --plot
  python main_integral.py field_profile --L 1.0 --a_min 0.1 --a_max 4.0 --n_a 50 --rule simpson --N 400 --plot
  python main_integral.py compare_scipy --L 1.0 --a 0.5 --rule simpson --N 400
"""
from __future__ import annotations
# ----------------------------- Import ----------------------------- #
import argparse
import math
import sys
from dataclasses import dataclass
from typing import Callable, Iterable, List, Tuple
import numpy as np

try:
    import scipy.integrate as spint  # quad, simpson, trapezoid
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False
# ----------------------------- Physics model ----------------------------- #
@dataclass
class CoulombLineParams:
    L: float  # half-length of the charged segment
    a: float  # distance from the line along the perpendicular bisector
    K: float = 1.0  # constant λ/(4π ε0). We keep as 1 by default to focus on numerics


def ey_kernel(x: np.ndarray | float, a: float) -> np.ndarray | float:
    """
    Kernel for the y-component of E on the perp. bisector:
      f(x; a) = a / (x^2 + a^2)^(3/2)
    This is smooth on [-L, L] for a > 0
    """
    return a / np.power(x * x + a * a, 1.5)


def ey_analytic(p: CoulombLineParams) -> float:
    """
    Closed-form result:
      Ey(a, L) = K * 2L / (a * sqrt(a^2 + L^2))
    """
    return p.K * (2.0 * p.L) / (p.a * math.sqrt(p.a * p.a + p.L * p.L))

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

# ----------------------------- Problem ---------------------------------- #

def ey_numeric(p: CoulombLineParams, rule: str, N: int) -> float:
    """
    We wrap the quadrature rules to compute Ey(a, L) for the Coulomb line.
    """
    f = lambda x: ey_kernel(x, p.a)
    a, b = -p.L, p.L
    if rule == "riemann":
        val = riemann_midpoint(f, a, b, N)
    elif rule == "trapezoid":
        val = trapezoid(f, a, b, N)
    elif rule == "simpson":
        val = simpson(f, a, b, N)
    else:
        raise ValueError(f"Unknown rule '{rule}'. Choose from: riemann, trapezoid, simpson.")
    return p.K * val


def rel_error(x: float, ref: float) -> float:
    """Relative error with safe handling near zero."""
    denom = max(1.0, abs(ref))
    return abs(x - ref) / denom


def print_note(msg: str) -> None:
    print(f"[note] {msg}")


# ----------------------------- Command Line Interface Modes ----------------------------- #

def mode_error_vs_N(args: argparse.Namespace) -> None:
    p = CoulombLineParams(L=args.L, a=args.a, K=args.K)
    exact = ey_analytic(p)
    print(f"Exact Ey(a={p.a}, L={p.L}) = {exact:.12g}  (K={p.K})\n")

    Ns: List[int] = args.N_list
    rules: List[str] = args.rules

    rows = []
    for rule in rules:
        for N in Ns:
            # enforce even N for Simpson
            N_eff = N + (N % 2) if rule == "simpson" else N
            val = ey_numeric(p, rule, N_eff)
            err = rel_error(val, exact)
            rows.append((rule, N_eff, val, err))

    # Table
    header = f"{'rule':<10} {'N':>8} {'Ey_numeric':>20} {'rel_error':>14}"
    print(header)
    print("-" * len(header))
    for rule, N, val, err in rows:
        print(f"{rule:<10} {N:>8d} {val:>20.12g} {err:>14.6e}")

    # Plot
    if args.plot:
        try:
            import matplotlib.pyplot as plt
            for rule in rules:
                Ns_rule = []
                errs_rule = []
                for N in Ns:
                    N_eff = N + (N % 2) if rule == "simpson" else N
                    Ns_rule.append(N_eff)
                    errs_rule.append(next(r[3] for r in rows if r[0] == rule and r[1] == N_eff))
                plt.loglog(Ns_rule, errs_rule, marker="o", label=rule)
            plt.xlabel("N (number of panels)")
            plt.ylabel("Relative error")
            plt.title(f"Error vs N (a={p.a}, L={p.L})")
            plt.grid(True, which="both")
            plt.legend()
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print_note(f"Plotting failed: {e}")


def mode_field_profile(args: argparse.Namespace) -> None:
    p = CoulombLineParams(L=args.L, a=1.0, K=args.K)  # a will be swept
    a_vals = np.linspace(args.a_min, args.a_max, args.n_a)
    Ey_num = []
    Ey_ex  = []

    for a in a_vals:
        p.a = float(a)
        N_eff = args.N + (args.N % 2) if args.rule == "simpson" else args.N
        Ey_num.append(ey_numeric(p, args.rule, N_eff))
        Ey_ex.append(ey_analytic(p))

    # Few sample rows
    print(f"{'a':>10} {'Ey_numeric':>20} {'Ey_analytic':>20} {'rel_error':>14}")
    print("-" * 70)
    for i in np.linspace(0, len(a_vals) - 1, num=min(8, len(a_vals)), dtype=int):
        err = rel_error(Ey_num[i], Ey_ex[i])
        print(f"{a_vals[i]:>10.4f} {Ey_num[i]:>20.12g} {Ey_ex[i]:>20.12g} {err:>14.6e}")

    if args.plot:
        try:
            import matplotlib.pyplot as plt
            plt.plot(a_vals, Ey_num, label=f"numeric ({args.rule}, N={args.N})")
            plt.plot(a_vals, Ey_ex,  label="analytic", linestyle="--")
            plt.xlabel("a")
            plt.ylabel("Ey(a)")
            plt.title(f"Field profile vs a  (L={args.L}, K={args.K})")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print_note(f"Plotting failed: {e}")


def mode_compare_scipy(args: argparse.Namespace) -> None:
    p = CoulombLineParams(L=args.L, a=args.a, K=args.K)
    exact = ey_analytic(p)

    N_eff = args.N + (args.N % 2) if args.rule == "simpson" else args.N
    val_my = ey_numeric(p, args.rule, N_eff)

    print(f"Parameters: L={p.L}, a={p.a}, K={p.K}")
    print(f"My {args.rule:<9} (N={N_eff:>5d}): {val_my:.12g}")
    print(f"Analytic                    : {exact:.12g}")
    print(f"Rel. error (numeric vs exact)  : {rel_error(val_my, exact):.6e}\n")

    if not SCIPY_OK:
        print_note("SciPy not found. Install with `pip install scipy` to enable comparisons.")
        return

    # Compare to SciPy
    f = lambda x: ey_kernel(x, p.a)
    quad_val, quad_err = spint.quad(f, -p.L, p.L, epsabs=1e-12, epsrel=1e-12)
    quad_val *= p.K

    # Compare to SciPy composite rules on a uniform grid
    x = np.linspace(-p.L, p.L, N_eff + 1)
    y = f(x)
    trap_sci = p.K * spint.trapezoid(y, x)
    simp_sci = None
    if N_eff % 2 == 0:
        simp_sci = p.K * spint.simpson(y, x)

    print("SciPy comparisons:")
    print(f"  quad (adaptive)        : {quad_val:.12g}   (reported abs err ~ {quad_err:.1e})")
    print(f"  trapezoid (SciPy)      : {trap_sci:.12g}   (same grid)")
    if simp_sci is not None:
        print(f"  simpson  (SciPy)       : {simp_sci:.12g}   (same grid)")

    print("\nRelative errors vs analytic:")
    print(f"  quad       : {rel_error(quad_val, exact):.6e}")
    print(f"  trapezoid  : {rel_error(trap_sci, exact):.6e}")
    if simp_sci is not None:
        print(f"  simpson    : {rel_error(simp_sci, exact):.6e}")


# ---------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Definite integral of Coulomb finite line: Riemann/Trapezoid/Simpson vs analytic/SciPy",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = p.add_subparsers(dest="mode", required=True)

    # error_vs_N
    q = sub.add_parser("error_vs_N", help="Error vs N")
    q.add_argument("--L", type=float, default=1.0, help="half-length of the line")
    q.add_argument("--a", type=float, default=0.5, help="distance along perpendicular bisector")
    q.add_argument("--K", type=float, default=1.0, help="constant λ/(4π ε0)")
    q.add_argument("--N_list", type=int, nargs="+", default=[20, 40, 80, 160, 320], help="list of N to test")
    q.add_argument(
        "--rules",
        type=str,
        nargs="+",
        choices=["riemann", "trapezoid", "simpson"],
        default=["riemann", "trapezoid", "simpson"],
        help="which rules to include",
    )
    q.add_argument("--plot", action="store_true", help="make a log–log error plot")
    q.set_defaults(func=mode_error_vs_N)

    # field_profile
    r = sub.add_parser("field_profile", help="Field profile Ey(a) vs a")
    r.add_argument("--L", type=float, default=1.0)
    r.add_argument("--K", type=float, default=1.0)
    r.add_argument("--a_min", type=float, default=0.1)
    r.add_argument("--a_max", type=float, default=4.0)
    r.add_argument("--n_a", type=int, default=60)
    r.add_argument("--rule", type=str, choices=["riemann", "trapezoid", "simpson"], default="simpson")
    r.add_argument("--N", type=int, default=400)
    r.add_argument("--plot", action="store_true")
    r.set_defaults(func=mode_field_profile)

    # compare_scipy
    s = sub.add_parser("compare_scipy", help="Compare to SciPy quad/trapezoid/simpson")
    s.add_argument("--L", type=float, default=1.0)
    s.add_argument("--a", type=float, default=0.5)
    s.add_argument("--K", type=float, default=1.0)
    s.add_argument("--rule", type=str, choices=["riemann", "trapezoid", "simpson"], default="simpson")
    s.add_argument("--N", type=int, default=400)
    s.set_defaults(func=mode_compare_scipy)

    return p


def main(argv: List[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


