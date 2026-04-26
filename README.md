# CFD Project: 1D Hydrodynamics

This repository contains a first implementation of a one-dimensional hydrodynamics solver for a reversed Sod shock tube problem.

The code is written in Python using PyTorch, NumPy, and Matplotlib.

## What this project does

The project solves a 1D hydrodynamical system with density, velocity, pressure, and internal energy. The main test case is a reversed Sod shock tube problem. An additional smooth Gaussian advection test is also included.

## Numerical method

The code uses:

- centered finite differences for spatial derivatives
- explicit Euler time integration
- CFL-based time stepping
- fixed boundary conditions
- comparison with an exact Riemann solution

## Output figures

### Reversed Sod shock tube

![Sod shock tube](figures/sod_shock_tube_numerical_vs_exact.png)

### Gaussian advection test

![Gaussian advection](figures/gaussian_advection_test.png)

## Limitations

This is a first implementation and uses a centered finite-difference method. Therefore, it can show the main qualitative wave structure, but it is not a fully conservative shock-capturing finite-volume method.

For more accurate shock simulations, a Rusanov, HLL, or HLLC finite-volume solver would be more appropriate.

## How to run

```bash
python hydro_1d.py
