#Imports and global settings

import torch
import numpy as np
import matplotlib.pyplot as plt
import math

from pathlib import Path


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float64
torch.set_default_dtype(dtype)

# Folder for saving figures
fig_dir = Path(__file__).resolve().parent / "AST5110_part1_figures"
fig_dir.mkdir(exist_ok=True)

def save_figure(fig, filename):
    path = fig_dir / filename
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved figure: {path}")

#Build your own code
# 1D hydrodynamics
# We solve continuity, momentum, and energy equations


# Parameters and left/right states
gamma = 5.0 / 3.0
x_0 = 1.0

rho_L = 0.125
u_L   = 0.0
p_L   = 0.125 / gamma
e_L   = p_L / (gamma - 1.0)

rho_R = 1.0
u_R   = 0.0
p_R   = 1.0 / gamma
e_R   = p_R / (gamma - 1.0)


#Initial condition for the reversed Sod shock tube
def initialcondition(x):
    x = torch.as_tensor(x, device=device, dtype=dtype)

    left_mask = x < x_0

    rho = torch.where(
        left_mask,
        torch.full_like(x, rho_L),
        torch.full_like(x, rho_R)
    )

    u = torch.where(
        left_mask,
        torch.full_like(x, u_L),
        torch.full_like(x, u_R)
    )

    p = torch.where(
        left_mask,
        torch.full_like(x, p_L),
        torch.full_like(x, p_R)
    )

    e = p / (gamma - 1.0)

    return rho, u, e


# Verify the  initial condition


rho_test, u_test, e_test = initialcondition(0.5)
print(f"x=0.5: rho = {rho_test.item()}, u = {u_test.item()}, e = {e_test.item()}")

rho_test, u_test, e_test = initialcondition(2.0)
print(f"x=2.0: rho = {rho_test.item()}, u = {u_test.item()}, e = {e_test.item()}")


# Spatial grid


x_min = -5.0
x_max = 5.0
N = 1000
x = torch.linspace(x_min, x_max, N, device=device, dtype=dtype) #The mesh grid etc

# Spatial derivative 
#The derivatives


def ddx(f,dx): #
    dfdx = torch.zeros_like(f)
    dfdx[1:-1]=(f[2:]-f[:-2])/(2*dx)

    #intial conditions
    dfdx[0] = (f[1] - f[0]) / dx
    dfdx[-1] = (f[-1] - f[-2]) / dx
    return dfdx


# Right-hand side of the hydrodynamical equations

def compute_rhs(rho,u,e,dx,gamma):
     p = (gamma - 1.0) * e
     m = rho * u 
     
     #Find the derivatives 
     rhs_rho = -ddx(m,dx)
     rhs_m = -ddx(m*u +p,dx)
     rhs_e = -ddx(e *u,dx)-p*ddx(u,dx)

     return rhs_rho,rhs_m,rhs_e 


#Next step is define the CFL condtions

# CFL time step

def compute_dt(rho, u, e, dx, gamma, cfl=0.2): #The CFl condtitions 
    rho_safe = torch.clamp(rho, min=1e-10)
    e_safe   = torch.clamp(e,   min=1e-10)
    p = (gamma - 1.0) * e_safe
    c_s = torch.sqrt(gamma * p / rho_safe)
    max_speed = torch.max(torch.abs(u) + c_s)
    dt = cfl * dx / max_speed
    return dt

#intial data

rho_value,u_values,e_values =initialcondition(x)

#the dx
dx = x[1] - x[0]


#Choose and justify boundary conditions.
#I will use the fixed boundary conditions , (The Dirichlet BC)

#The left we have rho_L = 0.125,u_L = 0, p_L =0.125/gamma
#The righy side is rho_r=1.0, u_R = 0 p_L =1.0/gamma

#So we have p= (gamma - 1) e
#Then we can fixed the BC
#e_L 0 p_L /(gamma-1)
#So, after every step, we fixed , pho(x_min) = 0.125, u(x_min)=0, e(x_min)=e_l

# Fixed boundary conditions
def apply_bc_fixed(rho, u, e):
    rho[0] = rho_L
    u[0]   = u_L
    e[0]   = e_L

    rho[-1] = rho_R
    u[-1]   = u_R
    e[-1]   = e_R
    return rho, u, e


# Time integration for the shock tube problem


t = 0.0
t_end = 0.1
n_steps = 0

rho = rho_value.clone()
u = u_values.clone()
e = e_values.clone()

rho, u, e = apply_bc_fixed(rho, u, e)

while t < t_end:
    rho, u, e = apply_bc_fixed(rho, u, e)

    dt = compute_dt(rho, u, e, dx, gamma, cfl=0.01)
    if t + float(dt) > t_end:
        dt = torch.tensor(t_end - t, device=device, dtype=dtype)

    rhs_rho, rhs_m, rhs_e = compute_rhs(rho, u, e, dx, gamma)

    m = rho * u
    rho = rho + dt * rhs_rho
    m   = m   + dt * rhs_m
    e   = e   + dt * rhs_e

    rho = torch.clamp(rho, min=1e-10)
    e   = torch.clamp(e,   min=1e-10)

    u   = m / rho
    u   = torch.nan_to_num(u, nan=0.0, posinf=0.0, neginf=0.0)

    rho, u, e = apply_bc_fixed(rho, u, e)

    t += float(dt)
    n_steps += 1

print("Final time =", t)
print("Number of time steps =", n_steps)
#I have set up and run a first 1D hydrodynamical shock tube simulation.
#The code includes the initial conditions, spatial grid, equation of state, CFL-based time stepping,
#and time evolution of density, momentum, and energy.
#The next step is to analyze the numerical results, choose boundary conditions explicitly, and compare with the analytical Sod solution.



#Compare with the analytical solution 
#Since we know the initial value

# Basic numerical checks

x_np   = x.detach().cpu().numpy()
rho_np = rho.detach().cpu().numpy()
u_np   = u.detach().cpu().numpy()
p_np   = ((gamma - 1.0) * e).detach().cpu().numpy()
print("any NaN in rho?", np.isnan(rho_np).any())
print("any NaN in u?",   np.isnan(u_np).any())
print("any NaN in p?",   np.isnan(p_np).any())

print("any inf in rho?", np.isinf(rho_np).any())
print("any inf in u?",   np.isinf(u_np).any())
print("any inf in p?",   np.isinf(p_np).any())

print("rho min, max =", np.nanmin(rho_np), np.nanmax(rho_np))
print("u min, max   =", np.nanmin(u_np), np.nanmax(u_np))
print("p min, max   =", np.nanmin(p_np), np.nanmax(p_np))



#Exact Riemann solver for Euler/Sod-type problem 


def pressure_function(p, rho_k, u_k, p_k, gamma):
    a_k = math.sqrt(gamma * p_k / rho_k)
    A_k = 2.0 / ((gamma + 1.0) * rho_k)
    B_k = (gamma - 1.0) / (gamma + 1.0) * p_k

    if p > p_k:   # shock
        f = (p - p_k) * math.sqrt(A_k / (p + B_k))
        fd = math.sqrt(A_k / (p + B_k)) * (1.0 - 0.5 * (p - p_k) / (p + B_k))
    else:         # rarefaction
        pr = p / p_k
        f = (2.0 * a_k / (gamma - 1.0)) * (pr**((gamma - 1.0) / (2.0 * gamma)) - 1.0)
        fd = (1.0 / (rho_k * a_k)) * pr**(-(gamma + 1.0) / (2.0 * gamma))

    return f, fd


def star_pressure_velocity(rho_L, u_L, p_L, rho_R, u_R, p_R, gamma,
                           tol=1e-10, max_iter=100):

    a_L = math.sqrt(gamma * p_L / rho_L)
    a_R = math.sqrt(gamma * p_R / rho_R)

    # initial guess
    p_guess = 0.5 * (p_L + p_R) - 0.125 * (u_R - u_L) * (rho_L + rho_R) * (a_L + a_R)
    p = max(1e-8, p_guess)

    for _ in range(max_iter):
        fL, fdL = pressure_function(p, rho_L, u_L, p_L, gamma)
        fR, fdR = pressure_function(p, rho_R, u_R, p_R, gamma)

        p_new = p - (fL + fR + u_R - u_L) / (fdL + fdR)
        p_new = max(1e-8, p_new)

        if abs(p_new - p) / (0.5 * (p_new + p)) < tol:
            p = p_new
            break

        p = p_new

    fL, _ = pressure_function(p, rho_L, u_L, p_L, gamma)
    fR, _ = pressure_function(p, rho_R, u_R, p_R, gamma)

    u_star = 0.5 * (u_L + u_R + fR - fL)

    return p, u_star


def exact_solution(x, t, x_0, rho_L, u_L, p_L, rho_R, u_R, p_R, gamma, p_star, u_star):
    rho_ex = torch.zeros_like(x)
    u_ex   = torch.zeros_like(x)
    p_ex   = torch.zeros_like(x)

    a_L = math.sqrt(gamma * p_L / rho_L)
    a_R = math.sqrt(gamma * p_R / rho_R)

    # left shock star state
    rho_star_L = rho_L * ((p_star / p_L + (gamma - 1.0)/(gamma + 1.0)) /
                          (((gamma - 1.0)/(gamma + 1.0)) * (p_star / p_L) + 1.0))

    # right rarefaction star state
    rho_star_R = rho_R * (p_star / p_R)**(1.0 / gamma)

    xi = (x - x_0) / t

    S_L = u_L - a_L * math.sqrt((gamma + 1.0)/(2.0*gamma) * (p_star/p_L)
                                + (gamma - 1.0)/(2.0*gamma))

    S_contact = u_star

    a_star_R = a_R * (p_star / p_R)**((gamma - 1.0)/(2.0*gamma))
    S_tail_R = u_star + a_star_R
    S_head_R = u_R + a_R

    # 1. left constant state
    mask1 = xi <= S_L
    rho_ex[mask1] = rho_L
    u_ex[mask1]   = u_L
    p_ex[mask1]   = p_L

    # 2. left star state
    mask2 = (xi > S_L) & (xi <= S_contact)
    rho_ex[mask2] = rho_star_L
    u_ex[mask2]   = u_star
    p_ex[mask2]   = p_star

    # 3. right star state
    mask3 = (xi > S_contact) & (xi <= S_tail_R)
    rho_ex[mask3] = rho_star_R
    u_ex[mask3]   = u_star
    p_ex[mask3]   = p_star

    # 4. right rarefaction fan
    mask4 = (xi > S_tail_R) & (xi < S_head_R)
    xi_fan = xi[mask4]

    u_fan = 2.0 / (gamma + 1.0) * (-a_R + 0.5*(gamma - 1.0)*u_R + xi_fan)
    a_fan = 2.0 / (gamma + 1.0) * (a_R - 0.5*(gamma - 1.0)*(u_R - xi_fan))
    rho_fan = rho_R * (a_fan / a_R)**(2.0 / (gamma - 1.0))
    p_fan   = p_R   * (a_fan / a_R)**(2.0 * gamma / (gamma - 1.0))

    rho_ex[mask4] = rho_fan
    u_ex[mask4]   = u_fan
    p_ex[mask4]   = p_fan

    # 5. right constant state
    mask5 = xi >= S_head_R
    rho_ex[mask5] = rho_R
    u_ex[mask5]   = u_R
    p_ex[mask5]   = p_R

    return rho_ex, u_ex, p_ex, S_L, S_contact, S_tail_R, S_head_R

# Compute exact solution
p_num = (gamma - 1.0) * e


# compute p_star and u_star
p_star, u_star = star_pressure_velocity(rho_L, u_L, p_L, rho_R, u_R, p_R, gamma)

# exact solution
rho_ex, u_ex, p_ex, S_L, S_contact, S_tail_R, S_head_R = exact_solution(
    x, t_end, x_0, rho_L, u_L, p_L, rho_R, u_R, p_R, gamma, p_star, u_star
)

print("p_star =", p_star)
print("u_star =", u_star)

# wave positions
x_shock   = x_0 + t_end * S_L
x_contact = x_0 + t_end * S_contact
x_rt      = x_0 + t_end * S_tail_R
x_rh      = x_0 + t_end * S_head_R

print("x_shock   =", x_shock)
print("x_contact =", x_contact)
print("x_rt      =", x_rt)
print("x_rh      =", x_rh)

# Plot numerical solution against exact solution
x_np      = x.detach().cpu().numpy()
rho_np    = rho.detach().cpu().numpy()
u_np      = u.detach().cpu().numpy()
p_np      = p_num.detach().cpu().numpy()
rho_ex_np = rho_ex.detach().cpu().numpy()
u_ex_np   = u_ex.detach().cpu().numpy()
p_ex_np   = p_ex.detach().cpu().numpy()

fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)

# Density
axes[0].plot(x_np, rho_np, label="Numerical density")
axes[0].plot(x_np, rho_ex_np, "--", label="Exact density")
axes[0].axvline(x_shock, linestyle=":", label="Shock")
axes[0].axvline(x_contact, linestyle=":", label="Contact")
axes[0].axvline(x_rt, linestyle=":", label="Rarefaction tail")
axes[0].axvline(x_rh, linestyle=":", label="Rarefaction head")
axes[0].set_ylabel("Density")
axes[0].set_title("Reversed Sod shock tube: numerical vs exact")
axes[0].legend()
axes[0].grid()

# Velocity
axes[1].plot(x_np, u_np, label="Numerical velocity")
axes[1].plot(x_np, u_ex_np, "--", label="Exact velocity")
axes[1].axvline(x_shock, linestyle=":")
axes[1].axvline(x_contact, linestyle=":")
axes[1].axvline(x_rt, linestyle=":")
axes[1].axvline(x_rh, linestyle=":")
axes[1].set_ylabel("Velocity")
axes[1].legend()
axes[1].grid()

# Pressure
axes[2].plot(x_np, p_np, label="Numerical pressure")
axes[2].plot(x_np, p_ex_np, "--", label="Exact pressure")
axes[2].axvline(x_shock, linestyle=":")
axes[2].axvline(x_contact, linestyle=":")
axes[2].axvline(x_rt, linestyle=":")
axes[2].axvline(x_rh, linestyle=":")
axes[2].set_xlabel("x")
axes[2].set_ylabel("Pressure")
axes[2].legend()
axes[2].grid()

save_figure(fig, "sod_shock_tube_numerical_vs_exact.png")
plt.show()

# Error analysis


L1_rho = (torch.sum(torch.abs(rho - rho_ex)) * dx).item()
Linf_rho = torch.max(torch.abs(rho - rho_ex)).item()
L1_u   = (torch.sum(torch.abs(u - u_ex)) * dx).item()
Linf_u   = torch.max(torch.abs(u - u_ex)).item()
L1_p   = (torch.sum(torch.abs(p_num - p_ex)) * dx).item()
Linf_p   = torch.max(torch.abs(p_num - p_ex)).item()
print("L1 error in density   =", L1_rho)
print("Linf error in density =", Linf_rho)
print("L1 error in velocity   =", L1_u)
print("Linf error in velocity =", Linf_u)
print("L1 error in pressure   =", L1_p)
print("Linf error in pressure =", Linf_p)
#Discuss the code’s ability to capture shocks, contact discontinuities, and rarefaction wave


# Check Rankine-Hugoniot conditions. And

# Additional smooth Gaussian advection test
#The u is u_0
#The pressure p is p_0
#So the density \rho is not constant.
#In this part we consider the Gaussian bump

#The gaussiam move to right u_0t
#The shape is same
#velocity and pressure is constant.
#This test is usefull
#If Gaussian is note move to right, then this advection , RHS, time loop is wrong
#If the Gaussian the speed is wrong then this time step and derivation is wrong.
#If the Gaussian is flat, then 


#We consider the initial u_0, p_0
rho0  = 1.0
u0    = 1.0
p0    = 1.0 / gamma

A     = 0.2
xc    = -2.0
sigma = 0.4

#Definite the initial function
def initial_condition_gaussian(x):
    x = torch.as_tensor(x, device=device, dtype=dtype)
    #The Gaussian bump
    rho = rho0 + A * torch.exp(-((x - xc) / sigma)**2)
    u = torch.full_like(x,u0)
    p = torch.full_like(x,p0)
    e = p/(gamma-1.0)

    return rho,u,e 

def apply_bc_gaussian(rho,u,e):
    rho[0]  = rho0
    u[0]    = u0
    e[0]    = p0 / (gamma - 1.0)

    rho[-1] = rho0
    u[-1]   = u0
    e[-1]   = p0 / (gamma - 1.0)

    return rho, u, e

N = 1000
#Recall the mesh, interval
x = torch.linspace(x_min, x_max, N, device=device, dtype=dtype)

#The dx
dx = x[1] - x[0]


#Initial data. 
rho,u,e= initial_condition_gaussian(x)

t = 0.0
t_end =1.0

rho,u,e = apply_bc_gaussian(rho,u,e)

while t < t_end:
    rho, u, e = apply_bc_gaussian(rho, u, e)

    dt = compute_dt(rho, u, e, dx, gamma, cfl=0.01).item()

    if t + dt > t_end:
        dt = t_end - t

    rhs_rho, rhs_m, rhs_e = compute_rhs(rho, u, e, dx, gamma)

    m = rho * u

    rho = rho + dt * rhs_rho
    m   = m   + dt * rhs_m
    e   = e   + dt * rhs_e

    rho = torch.clamp(rho, min=1e-10)
    e   = torch.clamp(e,   min=1e-10)

    u = m / rho
    u = torch.nan_to_num(u, nan=0.0, posinf=0.0, neginf=0.0)

    rho, u, e = apply_bc_gaussian(rho, u, e)

    t += dt

p = (gamma - 1.0) * e
rho_ex = rho0 + A * torch.exp(-((x - (xc + u0 * t_end)) / sigma)**2)
u_ex   = torch.full_like(x, u0)
p_ex   = torch.full_like(x, p0)

#Plot
x_np      = x.detach().cpu().numpy()
rho_np    = rho.detach().cpu().numpy()
u_np      = u.detach().cpu().numpy()
p_np      = p.detach().cpu().numpy()

rho_ex_np = rho_ex.detach().cpu().numpy()
u_ex_np   = u_ex.detach().cpu().numpy()
p_ex_np   = p_ex.detach().cpu().numpy()

fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)

# Density
axes[0].plot(x_np, rho_np, label="Numerical density")
axes[0].plot(x_np, rho_ex_np, "--", label="Exact density")
axes[0].set_ylabel("Density")
axes[0].set_title("Gaussian advection test: numerical vs exact")
axes[0].legend()
axes[0].grid()

# Velocity
axes[1].plot(x_np, u_np, label="Numerical velocity")
axes[1].plot(x_np, u_ex_np, "--", label="Exact velocity")
axes[1].set_ylabel("Velocity")
axes[1].legend()
axes[1].grid()

# Pressure
axes[2].plot(x_np, p_np, label="Numerical pressure")
axes[2].plot(x_np, p_ex_np, "--", label="Exact pressure")
axes[2].set_xlabel("x")
axes[2].set_ylabel("Pressure")
axes[2].legend()
axes[2].grid()

save_figure(fig, "gaussian_advection_test.png")
plt.show()

#The error analysis
L1_rho   = torch.mean(torch.abs(rho - rho_ex)).item()
Linf_rho = torch.max(torch.abs(rho - rho_ex)).item()

L1_u     = torch.mean(torch.abs(u - u_ex)).item()
Linf_u   = torch.max(torch.abs(u - u_ex)).item()

L1_p     = torch.mean(torch.abs(p - p_ex)).item()
Linf_p   = torch.max(torch.abs(p - p_ex)).item()

print(" Additional L1 error in density   =", L1_rho)
print("Linf error in density =", Linf_rho)

print("L1 error in velocity   =", L1_u)
print("Linf error in velocity =", Linf_u)

print("L1 error in pressure   =", L1_p)
print("Linf error in pressure =", Linf_p)



#This implementation uses a centered finite-difference approximation and explicit Euler time integration. 
#The energy equation is written in internal-energy form. Therefore, the code can reproduce the main qualitative wave structure of the reversed Sod shock tube,
# But it is not a fully conservative shock-capturing finite-volume method. 
#Some numerical diffusion or oscillations near shocks and contact discontinuities may occur.


# Reference:
# https://www.math.utah.edu/~gustafso/s2014/3150/slides/reaction-advection-dispersion-equation-chapter2.pdf