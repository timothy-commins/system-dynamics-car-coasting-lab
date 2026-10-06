"""
Car Coasting Lab - Non-linear model

While coasting, the car is slowed by rolling resistance (constant), air drag
(proportional to v^2) and the road slope (constant, sign depends on direction):

    m dv/dt = -F_roll - (1/2) rho Cd A v^2 -/+ m g sin(theta)

Dividing by m gives the model fit to each run:

    dv/dt = -a - c v^2
        a = F_roll/m -/+ g sin(theta)   [m/s^2]
        c = rho Cd A / (2 m)            [1/m]

The slope speeds the car up in one direction and slows it in the other, so
averaging a from the North and South runs leaves only rolling resistance.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit

RUNS = {"North": "north_data.csv", "South": "south_data.csv"}
G = 9.81          # m/s^2
RHO_AIR = 1.2     # kg/m^3
MASS_KG = None    # fill in to also get forces and Cd*A


def load_run(path):
    """Return time [s] and speed [m/s], starting when the speed first changes.

    The logger repeats its first reading for the first ~0.7 s, so those
    samples are not real coasting data.
    """
    t, v = np.loadtxt(path, delimiter=",", skiprows=1, unpack=True)
    t, first = np.unique(t, return_index=True)   # drop duplicate time stamps
    v = v[first]
    start = np.argmax(v != v[0])
    return t[start:] - t[start], v[start:]


def model(t, v0, a, c):
    """Speed at times t from solving dv/dt = -a - c v^2 with v(0) = v0."""
    sol = solve_ivp(lambda _, v: -a - c * v**2, (0, t[-1]), [v0],
                    t_eval=t, rtol=1e-8)
    return sol.y[0]


def main():
    fits = {}
    fig, axes = plt.subplots(2, len(RUNS), figsize=(12, 7), sharex="col",
                             gridspec_kw={"height_ratios": [3, 1]})

    print(f"{'Run':<7}{'v0 [m/s]':>10}{'a [m/s^2]':>12}{'c [1/m]':>12}"
          f"{'RMSE [m/s]':>12}")
    for col, (name, path) in enumerate(RUNS.items()):
        t, v = load_run(path)
        (v0, a, c), _ = curve_fit(model, t, v, p0=[v[0], 0.1, 1e-4])
        v_fit = model(t, v0, a, c)
        rmse = np.sqrt(np.mean((v - v_fit) ** 2))
        fits[name] = (a, c)
        print(f"{name:<7}{v0:>10.2f}{a:>12.4f}{c:>12.2e}{rmse:>12.3f}")

        top, bottom = axes[0, col], axes[1, col]
        top.plot(t, v, ".", ms=2, color="0.6", label="measured")
        top.plot(t, v_fit, color="C3", lw=2,
                 label=f"fit: a = {a:.4f}, c = {c:.2e}")
        top.set(title=f"{name} run", ylabel="speed [m/s]")
        top.legend()
        bottom.plot(t, v - v_fit, ".", ms=2)
        bottom.axhline(0, color="k", lw=0.8)
        bottom.set(xlabel="time [s]", ylabel="residual [m/s]")

    (a_n, c_n), (a_s, c_s) = fits["North"], fits["South"]
    a_roll = (a_n + a_s) / 2
    g_sin = abs(a_n - a_s) / 2
    c_avg = (c_n + c_s) / 2
    print(f"\nRolling resistance a_roll = {a_roll:.4f} m/s^2"
          f"  (C_rr = {a_roll / G:.4f})")
    print(f"Road slope g*sin(theta)   = {g_sin:.4f} m/s^2"
          f"  ({np.degrees(np.arcsin(g_sin / G)):.2f} deg)")
    print(f"Drag coefficient c (avg)  = {c_avg:.2e} 1/m")
    if MASS_KG:
        print(f"F_roll = {MASS_KG * a_roll:.0f} N,"
              f"  Cd*A = {2 * MASS_KG * c_avg / RHO_AIR:.3f} m^2")

    fig.tight_layout()
    fig.savefig("nonlinear_fit.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
