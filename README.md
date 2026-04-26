# CFD Project: 1D Hydrodynamics

This repository contains a first implementation of a one-dimensional hydrodynamics solver for a reversed Sod shock tube problem.

The code is written in Python using PyTorch, NumPy, and Matplotlib.

## Overview

The project solves a 1D hydrodynamical system with density, velocity, pressure, and internal energy. The main test case is a reversed Sod shock tube problem. A smooth Gaussian advection test is also included.

## Numerical method

The code uses centered finite differences, explicit Euler time integration, CFL-based time stepping, and fixed boundary conditions. The numerical shock tube result is compared with an exact Riemann solution.

## Output figures

### Reversed Sod shock tube

![Sod shock tube](figures/sod_shock_tube_numerical_vs_exact.png)

### Gaussian advection test

![Gaussian advection](figures/gaussian_advection_test.png)

## Limitations

This is a first implementation. It uses a centered finite-difference method and an internal-energy formulation, so it is not a fully conservative shock-capturing finite-volume method. Oscillations may occur near shocks and contact discontinuities.

A more robust version would use conservative variables and a Rusanov, HLL, or HLLC finite-volume flux.

## How to run

```bash
python hydro_1d.py
