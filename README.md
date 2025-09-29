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
"--a 0.5" sets the perpendicular distance from the line to the point from which we observe. This is E_y at point (0, 0.5)  
"--N_list 20 40 80 160" runs the integration with N = 20, 40, 80, 160. Each N is step size h = (2L)/N  
"--rules riemann trapezoid simpson" makes use of all three integration rules to run  
"--plot" generates the figure

***field_profile***
```bash
python integral.py field_profile --L 1.0 --a_min 0.1 --a_max 4.0 \
  --n_a 50 --rule simpson --N 400 --plot
```

***compare_scipy***
```bash
python integral.py compare_scipy --L 1.0 --a 0.5 --rule trapezoid --N 200
```







