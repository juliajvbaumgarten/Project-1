Author: Julia Baumgarten
# Project 1: Numerical Integration in Physics

This project implements Python modules to solve two different types of physics problems using numerical integration:

 **Definite integral problem**  
   Coulomb’s law in 1D geometry
   Electric field on the perpendicular bisector of a finite uniformly charged line.

 **ODE problem**  
   Driven–damped harmonic oscillator:
   \(m x'' + c x' + k x = F_0 \cos(\Omega t)\)

Both problems are solved using different numerical integration methods:
1. Euler and RK4
2. Riemann sum, Trapezoid, Simpson
These are compared to analytic results and to SciPy implementations.

Furthermore, physical validation checks are included.


  # Content
1. 'integral.py' is the CLI for the Coulomb line integral problem  
2. "ODE.py" is the CLI for the driven–damped oscillator problem  
3. "README.md" gives usage instructions and an overview
