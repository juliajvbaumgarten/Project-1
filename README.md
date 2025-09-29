Author: Julia Baumgarten
# Project 1: Numerical Integration in Physics

This project implements Python modules to solve two different types of physics problems using numerical integration:

 **Definite integral problem**  
   Coulomb’s law in 1D geometry
   Electric field on the perpendicular bisector of a finite uniformly charged line.

 **ODE problem**  
   Driven–damped harmonic oscillator:  
   $\(m x'' + c x' + k x = F_0 \cos(\Omega t)\)$

Both problems are solved using different numerical integration methods:
1. Euler and RK4
2. Riemann sum, Trapezoid, Simpson
These are compared to analytic results and to SciPy implementations.

Furthermore, physical validation checks are included.


  # Content
1. "integral.py" is the CLI for the Coulomb line integral problem  
2. "ODE.py" is the CLI for the driven–damped oscillator problem  
3. "README.md" gives usage instructions and an overview


## Definite Integral Problem: Coulomb’s Law in 1D

### Physics Setup

Line of half-length $\(L\)$ on the x-axis, uniform charge density $\(\lambda\)$.  
We evaluate the electric field at point $\(0,a)\$. By symmetry, only the y-component survives:

$$
E_y(a) = \frac{1}{4 \pi \varepsilon_0} \int_{-L}^{L} 
         \frac{\lambda a}{(x^2 + a^2)^{3/2}} \, dx
$$

Analytic result:

$$
E_y(a, L) = \frac{\lambda}{4 \pi \varepsilon_0} \cdot 
            \frac{2L}{a \sqrt{a^2 + L^2}}.
$$

---

### Features

1. Riemann midpoint, Trapezoid, Simpson rules
2. Numeric vs analytic field comparison
3. Error scaling with number of intervals \(N\)
4. Comparison against SciPy’s quad, trapezoid, simpson

---

### CLI Modes

The script is run via:  
```bash
python main_integral.py <mode> [options]
```


### Usage 

1. "error_vs_N" is an error vs N convergence analysis tool.
2. "field_profile" gives the y component of the electric field as a function of a vs the distance a.
3. "compare_scipy" compares to SciPy implementations.

***error_vs_N***
```bash
python integral.py error_vs_N --L 1.0 --a 0.5 \
  --N_list 20 40 80 160 --rules riemann trapezoid simpson --plot
```

This specific command runs the integral.py file using "error vs. N" mode, which checks how the numerical error decreases with N.  
"--L 1.0" sets the half length of the charged line so it goes from -1.0 to +1.0  
"--a 0.5" sets the perpendicular distance from the line to the point from which we observe. This is $E_y$ at point $\(0, 0.5)\$  
"--N_list 20 40 80 160" runs the integration with N = 20, 40, 80, 160. Each N is step size h = (2L)/N  
"--rules riemann trapezoid simpson" makes use of all three integration rules to run  
"--plot" generates the figure

***field_profile***
```bash
python integral.py field_profile --L 1.0 --a_min 0.1 --a_max 4.0 \
  --n_a 50 --rule simpson --N 400 --plot
```

This specific command runs the integral.py file using "field profile" mode, which sweeps over distances a to build a profile of the electric field $E_y(a)$.  
"--L 1.0" sets the half length of the charged line so it goes from -1.0 to +1.0  
"--a_min 0.1 --a_max 4.0" sets the range of perpendicular distances from the line where the field is evaluated. It starts at $a = 0.1$ and ends at $a = 4.0$  
"--n_a 50" sets the number of points being swept, so this computes 50 evenly spaced values between a_min and a_max  
"--rule simpson" use of Simpson's rule for integration  
"--N 400" is the number of integration subintervals  
"--plot" generates the figure  

***compare_scipy***
```bash
python integral.py compare_scipy --L 1.0 --a 0.5 --rule trapezoid --N 200
```
This specific command runs the integral.py file using "compare scipy" mode, which compares the coded integration routines against the integrated SciPy option.  
"--L 1.0" sets the half length of the charged line so it goes from -1.0 to +1.0  
"--a 0.5" sets the perpendicular distance from the line to the point from which we observe. This is $E_y$ at point $\(0, 0.5)\$  
"--rule trapezoid" use of trapezoid rule for integration  
"--N 200" is the number of integration subintervals  


## ODE Problem: Driven–damped harmonic oscillator

### Physics Setup

We consider the equation:

$$
m \ddot{x} + c \dot{x} + k x = F_0 \cos(\Omega t),
$$

where  
$\(m\)$ : mass  
$\(k\)$ : spring constant (\(\omega_0 = \sqrt{k/m}\))  
$\(c\)$ : damping coefficient (\(\gamma = c/m\))  
$\(F_0\)$ : drive amplitude  
$\(\Omega\)$ : drive frequency  

Special cases:  
1. Free oscillator $\(c=F_0=0\)$: exact solution exists, energy is conserved.  
2. Driven damped oscillator: steady-state amplitude and phase given by analytic formulae.

---

## Features

1. Euler and RK4 fixed-step integrators
2. Phase portraits (x vs v)
3. Energy drift comparison: Euler vs RK4
4. Global error vs step size $\(h\)$ (log–log)
5. Frequency response (amplitude & phase vs $\(\Omega\)$
6. Physical checks summary:
  - Energy conservation in free SHO  
  - Resonance peak near $\(\Omega \approx \sqrt{\omega_0^2 - \gamma^2/2}\)$    
  - Phase $\(\approx 90^\circ\)$ at resonance

---

### CLI Modes

The script is run via:

```bash
python ODE.py <mode> [options]
```

### Usage

***phase_portrait***
```bash
python ODE.py phase_portrait --m 1 --k 1 --c 0 --F0 0 --x0 1 --v0 0 --h 0.02 --tmax 20 --method rk4
```

This specific command runs the ODE.py file using "phase portrait" mode, which plots the phase trajectory $v(t)$ vs $x(t)$ to visualize the oscillator's dynamics.  
"--m 1" sets the mass to 1  
"--k 1" sets the spring constant to 1  
"--c 0" sets the damping coefficient to 0  
"--F0 0" sets the driving force amplitude to 0  
"--x0 1 --v0 0" sets the initial conditions to $x(0) = 1$ and $v(0) = 0$   
"--h 0.02" sets the time step at 0.02  
"--tmax 20" simulates until $t = 20$. We note $\frac{20}{2\pi}$ is about 3 oscillations.  
"--method rk4" runs Runge-Kutta 4th order method to solve the ODE  


***energy_drift***
```bash
python ODE.py energy_drift --m 1 --k 1 --force_validation --h 0.05 --tmax 200
```



***convergence***
```bash
python ODE.py convergence --m 1 --k 1 --x0 1 --v0 0 --tmax 10 --h_list 0.4 0.2 0.1 0.05 0.025
```



***frequency_response***
```bash
python ODE.py frequency_response --m 1 --k 1 --c 0.1 --F0 1 --h 0.01 \
  --Omega_min 0.2 --Omega_max 2.0 --n_Omega 60 --method rk4
```



***physical_checks***
```bash
python ODE.py physical_checks --m 1 --k 1 --c 0.1 --F0 1 --h 0.01 --tmax 60
```








