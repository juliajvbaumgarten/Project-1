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
def euler_step(f: Callable, t: float, y: np.ndarray, h: float, p: SHOParams) -> np.ndarray:
    return y + h * f(t, y, p)

def rk4_step(f: Callable, t: float, y: np.ndarray, h: float, p: SHOParams) -> np.ndarray:
    k1 = f(t,           y,            p)
    k2 = f(t + 0.5*h,   y + 0.5*h*k1, p)
    k3 = f(t + 0.5*h,   y + 0.5*h*k2, p)
    k4 = f(t + h,       y + h*k3,     p)
    return y + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)

# ------------------------------------- Integrator loop ------------------------------------
def integrate(f: Callable, stepper: Callable, t0: float, y0, h: float, nsteps: int, p: SHOParams):
    y0 = np.array(y0, dtype=float)
    t = np.empty(nsteps + 1)
    Y = np.empty((nsteps + 1, 2))
    t[0], Y[0] = t0, y0
    ti, yi = t0, y0
    for i in range(1, nsteps + 1):
        yi = stepper(f, ti, yi, h, p)
        ti = t0 + i*h
        t[i], Y[i] = ti, yi
    return t, Y

# ---------------------------- Analytics & Helper Functions ----------------------------

def omega0(p: SHOParams) -> float:
    return math.sqrt(p.k / p.m)

def gamma(p: SHOParams) -> float:
    return p.c / p.m

def energy(x: np.ndarray, v: np.ndarray, p: SHOParams) -> np.ndarray:
    return 0.5 * p.m * v*v + 0.5 * p.k * x*x

def exact_free_solution(t: np.ndarray, x0: float, v0: float, p: SHOParams) -> np.ndarray:
    om0 = omega0(p)
    return x0 * np.cos(om0*t) + (v0/om0) * np.sin(om0*t)

def steady_state_amp_phase_analytic(p: SHOParams) -> Tuple[float, float]:
    """Amplitude X and phase phi for steady-state x = X cos(Omega t - phi)."""
    om0 = omega0(p)
    gam  = gamma(p)
    X = (p.F0 / p.m) / math.sqrt((om0**2 - p.Omega**2)**2 + (gam*p.Omega)**2)
    phi = math.atan2(gam*p.Omega, om0**2 - p.Omega**2)
    return X, phi

def fit_amp_phase_cos_sin(t: np.ndarray, x: np.ndarray, Omega: float) -> Tuple[float, float, float]:
    """
    Least-squares fit x(t) approx A cos(Omega t) + B sin(Omega t) + C
    Returns (amplitude R, phase phi, offset C) with R cos(Omega t - phi) form.
    """
    C = np.column_stack([np.cos(Omega*t), np.sin(Omega*t), np.ones_like(t)])
    coeff, *_ = np.linalg.lstsq(C, x, rcond=None)
    A, B, C0 = coeff
    R = float(np.hypot(A, B))
    phi = float(np.arctan2(B, A))
    return R, phi, float(C0)

def resonance_peak_theory(p: SHOParams) -> float:
    """For underdamped case, amplitude peak near sqrt(omega_{0}^2 - gamma^2/2)."""
    om0 = omega0(p); gam = gamma(p)
    if gam >= math.sqrt(2)*om0:
        return float('nan')
    return math.sqrt(max(om0**2 - 0.5*gam**2, 0.0))

def pick_stepper(name: str) -> Callable:
    return {"euler": euler_step, "rk4": rk4_step}[name]

# ---------------------------- Command Line Integration Modes ----------------------------

def mode_phase_portrait(a: argparse.Namespace) -> None:
    p = SHOParams(m=a.m, c=a.c, k=a.k, F0=a.F0, Omega=a.Omega)
    stepper = pick_stepper(a.method)
    N = int(a.tmax / a.h)
    t, Y = integrate(sho_rhs, stepper, 0.0, [a.x0, a.v0], a.h, N, p)

    if PLOT_OK:
        plt.figure()
        plt.plot(Y[:,0], Y[:,1])
        plt.xlabel("x")
        plt.ylabel("v")
        plt.title(f"Phase portrait ({a.method}), m={p.m}, k={p.k}, c={p.c}, F0={p.F0}, Ω={p.Omega}")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    else:
        print("[note] matplotlib not available; no plot produced.")
    print(f"Final state: x={Y[-1,0]:.6f}, v={Y[-1,1]:.6f}")


def mode_energy_drift(a: argparse.Namespace) -> None:
    # Force the validation case c=0, F_{0}=0 unless user overrides
    p = SHOParams(m=a.m, c=0.0 if a.force_validation else a.c, k=a.k,
                  F0=0.0 if a.force_validation else a.F0, Omega=a.Omega)

    N = int(a.tmax / a.h)
    tE, YE = integrate(sho_rhs, euler_step, 0.0, [a.x0, a.v0], a.h, N, p)
    tR, YR = integrate(sho_rhs, rk4_step,   0.0, [a.x0, a.v0], a.h, N, p)
    EE = energy(YE[:,0], YE[:,1], p)
    ER = energy(YR[:,0], YR[:,1], p)

    print(f"h={a.h}, steps={N}  (forced c=F0=0: {a.force_validation})")
    print(f"Euler: E in [{EE.min():.6f}, {EE.max():.6f}], drift ΔE={EE.max()-EE.min():.6e}")
    print(f"RK4  : E in [{ER.min():.6f}, {ER.max():.6f}], drift ΔE={ER.max()-ER.min():.6e}")

    if PLOT_OK:
        plt.figure()
        plt.plot(tE, EE, label="Euler")
        plt.plot(tR, ER, label="RK4")
        plt.xlabel("t")
        plt.ylabel("Energy")
        plt.title("Energy drift (free SHO if c=F0=0)")
        plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()


def mode_convergence(a: argparse.Namespace) -> None:
    """Global error vs step size h for the free SHO (c=F_{0}=0)."""
    # Free oscillator: no damping/drive
    p = SHOParams(m=a.m, c=0.0, k=a.k, F0=0.0)
    errs_eu, errs_rk, hs = [], [], []

    for h in a.h_list:
        N = int(a.tmax / h)
        tE, YE = integrate(sho_rhs, euler_step, 0.0, [a.x0, a.v0], h, N, p)
        tR, YR = integrate(sho_rhs, rk4_step,   0.0, [a.x0, a.v0], h, N, p)

        # exact free solution for comparison
        x_exact_E = exact_free_solution(tE, a.x0, a.v0, p)
        x_exact_R = exact_free_solution(tR, a.x0, a.v0, p)

        errE = np.linalg.norm(YE[:,0] - x_exact_E, ord=np.inf)
        errR = np.linalg.norm(YR[:,0] - x_exact_R, ord=np.inf)

        hs.append(h); errs_eu.append(errE); errs_rk.append(errR)
        print(f"h={h:<8g}  err_Euler={errE:.6e}  err_RK4={errR:.6e}")

    if PLOT_OK:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.loglog(hs, errs_eu, "o-", label="Euler (O(h))")
        plt.loglog(hs, errs_rk, "s-", label="RK4 (O(h^4))")
        plt.gca().invert_xaxis()
        plt.xlabel("step size h")
        plt.ylabel("global max-norm error in x(t)")
        plt.title("Global error vs step size (free SHO)")
        plt.grid(True, which="both"); plt.legend(); plt.tight_layout(); plt.show()


def mode_frequency_response(a: argparse.Namespace) -> None:
    """Sweep Omega and measure steady-state amplitude & phase and compare to analytic."""
    p_base = SHOParams(m=a.m, c=a.c, k=a.k, F0=a.F0, Omega=0.0)
    stepper = pick_stepper(a.method)
    Om_list = np.linspace(a.Omega_min, a.Omega_max, a.n_Omega)
    amps_num, phases_num, amps_ana, phases_ana = [], [], [], []

    # Choose a simulation length Decay time ~ m/c also needs multiple periods
    gam = gamma(p_base)
    for Om in Om_list:
        p = SHOParams(m=p_base.m, c=p_base.c, k=p_base.k, F0=p_base.F0, Omega=Om)
        Tdrive = 2*np.pi/Om
        t_trans = max(10.0/gam if gam>0 else 0.0, 10*Tdrive)
        t_meas  = max(5*Tdrive, 2.0/gam if gam>0 else 5*Tdrive)
        h = a.h
        N = int((t_trans + t_meas)/h)
        t, Y = integrate(sho_rhs, stepper, 0.0, [a.x0, a.v0], h, N, p)

        # Tail for measurement
        mask = t >= (t[-1] - t_meas)
        tt, xx = t[mask], Y[mask,0]
        R, phi, _ = fit_amp_phase_cos_sin(tt, xx, Om)
        amps_num.append(R); phases_num.append(phi)

        X, ph = steady_state_amp_phase_analytic(p)
        amps_ana.append(X); phases_ana.append(ph)

    # Table around resonance
    om_peak = resonance_peak_theory(p_base)
    print(f"Theory resonance peak (underdamped): Omega_peak ≈ {om_peak:.6g} rad/s")
    print(f"{'Ω':>8} {'Amp_num':>12} {'Amp_th':>12} {'RelErr':>10} {'phi_num(rad)':>14} {'phi_th(rad)':>12}")
    for Om, An, Aa, phn, pha in zip(Om_list, amps_num, amps_ana, phases_num, phases_ana):
        rel = abs(An - Aa) / max(1.0, abs(Aa))
        print(f"{Om:>8.3f} {An:>12.6g} {Aa:>12.6g} {rel:>10.3e} {phn:>14.3f} {pha:>12.3f}")

    if PLOT_OK:
        fig, ax = plt.subplots(2, 1, figsize=(6, 7))
        ax[0].plot(Om_list, amps_num, "o-", label=f"numeric ({a.method})")
        ax[0].plot(Om_list, amps_ana, "--", label="analytic")
        if not np.isnan(om_peak):
            ax[0].axvline(om_peak, color="k", linestyle=":", label="Omega_peak theory")
        ax[0].set_xlabel("Omega"); ax[0].set_ylabel("Amplitude")
        ax[0].set_title("Frequency response — amplitude")
        ax[0].grid(True); ax[0].legend()

        ax[1].plot(Om_list, phases_num, "o-", label=f"numeric ({a.method})")
        ax[1].plot(Om_list, phases_ana, "--", label="analytic")
        if not np.isnan(om_peak):
            ax[1].axvline(om_peak, color="k", linestyle=":", label="Omega_peak theory")
        ax[1].set_xlabel("Omega"); ax[1].set_ylabel("Phase Phi (rad)")
        ax[1].set_title("Frequency response — phase")
        ax[1].grid(True); ax[1].legend()
        plt.tight_layout(); plt.show()


def mode_physical_checks(a: argparse.Namespace) -> None:
    """
    Compact PASS/FAIL summary of required physical properties:
      - Energy conservation when c=F_{0}=0
      - Resonance peak near sqrt(omega_{0}^2 - gamma^2/2)
      - Phase approx 90 degree at resonance
    """
    # 1) Energy conservation
    p_free = SHOParams(m=a.m, c=0.0, k=a.k, F0=0.0, Omega=a.Omega)
    h, T = a.h, a.tmax
    N = int(T/h)
    tE, YE = integrate(sho_rhs, euler_step, 0.0, [a.x0, a.v0], h, N, p_free)
    tR, YR = integrate(sho_rhs, rk4_step,   0.0, [a.x0, a.v0], h, N, p_free)
    driftE = energy(YE[:,0], YE[:,1], p_free); driftR = energy(YR[:,0], YR[:,1], p_free)
    dE = driftE.max() - driftE.min()
    dR = driftR.max() - driftR.min()
    print("Energy conservation (c=F_{0}=0):")
    print(f"  Euler ΔE = {dE:.3e}  |  RK4 ΔE = {dR:.3e}  -> PASS if RK4 << Euler and both down as h down")

    # 2) Resonance & phase at resonance
    p_drive = SHOParams(m=a.m, c=a.c, k=a.k, F0=a.F0, Omega=0.0)
    om_peak = resonance_peak_theory(p_drive)
    if math.isnan(om_peak):
        print("Resonance check: system not underdamped (gamma le sqrt(2) omega_{0}). Skipping.")
        return

    # Evaluate at a grid around the peak
    Om_list = np.linspace(0.6*om_peak, 1.4*om_peak, 15)
    stepper = rk4_step
    gam = gamma(p_drive)
    records = []
    for Om in Om_list:
        p = SHOParams(m=p_drive.m, c=p_drive.c, k=p_drive.k, F0=p_drive.F0, Omega=Om)
        Tdrive = 2*np.pi/Om
        t_trans = max(10.0/gam if gam>0 else 0.0, 10*Tdrive)
        t_meas  = max(5*Tdrive, 2.0/gam if gam>0 else 5*Tdrive)
        N = int((t_trans + t_meas)/h)
        t, Y = integrate(sho_rhs, stepper, 0.0, [0.0, 0.0], h, N, p)
        mask = t >= (t[-1] - t_meas)
        R, phi, _ = fit_amp_phase_cos_sin(t[mask], Y[mask,0], Om)
        records.append((Om, R, phi))
    # Find numeric peak
    Om_num, R_num, phi_num = max(records, key=lambda r: r[1])
    print(f"Resonance peak:")
    print(f"  theory Omega_peak ≈ {om_peak:.6f} | numeric Ω_peak ≈ {Om_num:.6f} (|Δ| = {abs(Om_num-om_peak):.3e})")
    print(f"Phase near peak (should be ~ π/2):  phi_num = {phi_num:.3f} rad  ({np.degrees(phi_num):.1f}degree)")
    print("  -> PASS if |Omega_num - Omega_theory| is small and Phi approx 1.57 rad (90 degree).")


# -------------------------------------------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Driven–damped harmonic oscillator: Euler/RK4 with analytics & physical checks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = p.add_subparsers(dest="mode", required=True)

    # Common defaults
    def add_common(sp):
        sp.add_argument("--m", type=float, default=1.0)
        sp.add_argument("--k", type=float, default=1.0)
        sp.add_argument("--c", type=float, default=0.0)
        sp.add_argument("--F0", type=float, default=0.0)
        sp.add_argument("--Omega", type=float, default=1.0)
        sp.add_argument("--x0", type=float, default=1.0)
        sp.add_argument("--v0", type=float, default=0.0)
        sp.add_argument("--h",  type=float, default=0.02)
        sp.add_argument("--tmax", type=float, default=40.0)

    # phase_portrait
    q = sub.add_parser("phase_portrait", help="Plot x–v phase portrait")
    add_common(q)
    q.add_argument("--method", choices=["euler","rk4"], default="rk4")
    q.set_defaults(func=mode_phase_portrait)

    # energy_drift
    r = sub.add_parser("energy_drift", help="Energy drift (Euler vs RK4)")
    add_common(r)
    r.add_argument("--force_validation", action="store_true",
                   help="Force c=0,F0=0 (free SHO) regardless of flags")
    r.set_defaults(func=mode_energy_drift)

    # convergence
    s = sub.add_parser("convergence", help="Global error vs step size h (free SHO)")
    s.add_argument("--m", type=float, default=1.0)
    s.add_argument("--k", type=float, default=1.0)
    s.add_argument("--x0", type=float, default=1.0)
    s.add_argument("--v0", type=float, default=0.0)
    s.add_argument("--tmax", type=float, default=20.0)
    s.add_argument("--h_list", type=float, nargs="+", default=[0.4,0.2,0.1,0.05,0.025])
    s.set_defaults(func=mode_convergence)

    # frequency_response
    t = sub.add_parser("frequency_response", help="Sweep Omega and amplitude/phase vs Omega with analytic overlay")
    add_common(t)
    t.add_argument("--Omega_min", type=float, default=0.2)
    t.add_argument("--Omega_max", type=float, default=2.0)
    t.add_argument("--n_Omega",   type=int,   default=50)
    t.add_argument("--method", choices=["euler","rk4"], default="rk4")
    t.set_defaults(func=mode_frequency_response)

    # physical_checks
    u = sub.add_parser("physical_checks", help="PASS/FAIL checks for project rubric")
    add_common(u)
    u.set_defaults(func=mode_physical_checks)

    return p


def main(argv: List[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())










