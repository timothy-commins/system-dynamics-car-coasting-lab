"""
Car Coasting Lab - Non-linear model
===================================

Model
-----
While coasting (no engine force) the car is slowed by rolling resistance
(roughly constant), aerodynamic drag (proportional to v^2), and gravity if
the road has a slope:

    m dv/dt = -F_roll - (1/2) rho Cd A v^2 -/+ m g sin(theta)

Dividing by the mass gives the non-linear first-order ODE that is fit here:

    dv/dt = -a - c v^2

    a = F_roll/m +/- g sin(theta)   [m/s^2]   constant deceleration
    c = rho Cd A / (2 m)            [1/m]     quadratic (drag) coefficient

For a > 0 and c > 0 this ODE has the closed-form solution

    v(t) = sqrt(a/c) * tan( arctan(v0 sqrt(c/a)) - sqrt(a c) t )

but the fit below integrates the ODE numerically, so it still works if a
comes out negative (a downhill run where gravity beats rolling resistance).

North/South runs
----------------
The road grade helps the car in one direction and slows it in the other.
Averaging the "a" values from the two directions cancels the grade term,
leaving the true rolling-resistance deceleration. Half of the difference
gives an estimate of g*sin(theta).

Usage
-----
    python nonlinear_model.py

Set MASS_KG below (and optionally FRONTAL_AREA_M2) to also get the rolling
resistance force, the drag constant, and an estimate of Cd.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit

# ---------------------------------------------------------------- settings --
DATA_FILES = {"North": "north_data.csv", "South": "south_data.csv"}
MASS_KG = None          # vehicle mass in kg (fill in from the lab handout)
FRONTAL_AREA_M2 = None  # frontal area in m^2 (optional, used to estimate Cd)
RHO_AIR = 1.2           # air density in kg/m^3
G = 9.81                # m/s^2


# ------------------------------------------------------------ data loading --
def load_data(path):
    """Load time [s] and speed [m/s]; drop duplicate time stamps."""
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    t, v = data[:, 0], data[:, 1]
    t, idx = np.unique(t, return_index=True)
    return t, v[idx]


# ------------------------------------------------------------------- model --
def nonlinear_ode(t, v, a, c):
    """dv/dt = -a - c v^2"""
    return -a - c * v**2


def simulate(t, v0, a, c):
    """Integrate the non-linear ODE and return v at the times t."""
    sol = solve_ivp(nonlinear_ode, (t[0], t[-1]), [v0], t_eval=t,
                    args=(a, c), rtol=1e-8, atol=1e-10)
    if not sol.success or sol.y.shape[1] != len(t):
        return np.full_like(t, np.nan)
    return sol.y[0]


def closed_form(t, v0, a, c):
    """Analytical solution (valid for a > 0, c > 0) - used as a check."""
    k = np.sqrt(a / c)
    return k * np.tan(np.arctan(v0 / k) - np.sqrt(a * c) * t)


def fit_run(t, v):
    """Least-squares fit of v0, a, c to one coasting run."""
    t = t - t[0]
    # initial guess from a quadratic fit of the estimated deceleration
    dvdt = np.gradient(v, t)
    good = np.isfinite(dvdt)
    c_guess, _, a_guess = np.polyfit(v[good], -dvdt[good], 2)
    p0 = [v[0], a_guess, max(c_guess, 1e-5)]

    def model(tt, v0, a, c):
        return simulate(tt, v0, a, c)

    popt, pcov = curve_fit(model, t, v, p0=p0, maxfev=20000)
    perr = np.sqrt(np.diag(pcov))
    v_fit = model(t, *popt)
    resid = v - v_fit
    rmse = np.sqrt(np.mean(resid**2))
    r2 = 1 - np.sum(resid**2) / np.sum((v - v.mean())**2)
    return dict(t=t, v=v, v_fit=v_fit, resid=resid, popt=popt, perr=perr,
                rmse=rmse, r2=r2)


# -------------------------------------------------------------------- main --
def main():
    results = {}
    for name, path in DATA_FILES.items():
        t, v = load_data(path)
        results[name] = fit_run(t, v)

    print("Non-linear coasting model:  dv/dt = -a - c v^2\n")
    print(f"{'Run':<7}{'v0 [m/s]':>12}{'a [m/s^2]':>20}{'c [1/m]':>24}"
          f"{'RMSE [m/s]':>13}{'R^2':>9}")
    for name, r in results.items():
        (v0, a, c), (_, ea, ec) = r["popt"], r["perr"]
        print(f"{name:<7}{v0:>12.3f}{a:>12.5f} ± {ea:<7.5f}"
              f"{c:>14.3e} ± {ec:<9.2e}{r['rmse']:>11.3f}{r['r2']:>10.5f}")

    a_n, c_n = results["North"]["popt"][1:]
    a_s, c_s = results["South"]["popt"][1:]
    a_roll = (a_n + a_s) / 2          # grade cancels out
    g_sin = abs(a_n - a_s) / 2        # grade contribution
    c_avg = (c_n + c_s) / 2

    print("\nCombining both directions (cancels road grade):")
    print(f"  rolling resistance decel  a_roll = {a_roll:.5f} m/s^2")
    print(f"  equivalent C_rr = a_roll/g       = {a_roll / G:.5f}")
    print(f"  grade term g*sin(theta)          = {g_sin:.5f} m/s^2"
          f"  (~{np.degrees(np.arcsin(min(g_sin / G, 1))):.3f} deg)")
    print(f"  drag coefficient c (average)     = {c_avg:.3e} 1/m")

    if MASS_KG:
        print(f"\nWith m = {MASS_KG} kg:")
        print(f"  rolling resistance force F_roll  = {MASS_KG * a_roll:.1f} N")
        print(f"  drag constant (1/2 rho Cd A)     = {MASS_KG * c_avg:.4f} kg/m")
        cda = 2 * MASS_KG * c_avg / RHO_AIR
        print(f"  Cd*A                             = {cda:.3f} m^2")
        if FRONTAL_AREA_M2:
            print(f"  Cd (A = {FRONTAL_AREA_M2} m^2)              "
                  f"= {cda / FRONTAL_AREA_M2:.3f}")

    # sanity check: closed form vs numerical solution (when a > 0)
    for name, r in results.items():
        v0, a, c = r["popt"]
        if a > 0 and c > 0:
            diff = np.max(np.abs(closed_form(r["t"], v0, a, c) - r["v_fit"]))
            print(f"\n{name}: max |closed form - numerical| = {diff:.2e} m/s")

    # ---------------------------------------------------------------- plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex="col",
                             gridspec_kw={"height_ratios": [3, 1]})
    for col, (name, r) in enumerate(results.items()):
        v0, a, c = r["popt"]
        ax = axes[0, col]
        ax.plot(r["t"], r["v"], ".", ms=2, color="0.6", label="measured")
        ax.plot(r["t"], r["v_fit"], "-", lw=2, color="C3",
                label=f"fit: a={a:.4f}, c={c:.2e}")
        ax.set_title(f"{name} run – non-linear model")
        ax.set_ylabel("speed [m/s]")
        ax.grid(alpha=0.3)
        ax.legend()
        axr = axes[1, col]
        axr.plot(r["t"], r["resid"], ".", ms=2, color="C0")
        axr.axhline(0, color="k", lw=0.8)
        axr.set_xlabel("time [s]")
        axr.set_ylabel("residual [m/s]")
        axr.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("nonlinear_fit.png", dpi=150)
    print("\nSaved plot to nonlinear_fit.png")
    plt.show()


if __name__ == "__main__":
    main()
