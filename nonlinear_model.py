"""
Car Coasting Lab - Non-linear model  (MeEn 335)

Equation of motion (x positive in the direction of travel):

    m dv/dt = -s m g sin(theta)  -  mu_k m g cos(theta)  -  1/2 Cd A rho v_rel|v_rel|

    s     = +1 going North (uphill), -1 going South (downhill)
    v_rel = v + s v_wind   (wind blows from the North: headwind going North,
                            tailwind going South)

Every term is a force in N. Gravity's sign flips with direction, friction and
drag always oppose the motion, and drag uses the speed relative to the air.

The unknowns Cd and mu_k (and each run's starting speed v0, since the first
GPS reading is noisy) are fit by least squares, simulating with solve_ivp.

Run this file for the fits and Figures 1-2; run uncertainty_power.py for the
power predictions and uncertainty analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

# ---- given in the lab handout ----
MASS = 1760 + 150             # kg, car + passengers
THETA_DEG = 1.03              # incline angle, degrees
AREA = 2.15                   # m^2, projected frontal area
RHO = 1.06                    # kg/m^3, air density
V_WIND = 7 * 0.44704          # m/s, 7 mph wind from the North
G = 9.81                      # m/s^2

RUNS = {"North": ("north_data.csv", +1), "South": ("south_data.csv", -1)}


def load_run(path):
    """Time [s] and speed [m/s], starting when the speed first changes.

    The logger repeats its first reading for ~0.7 s, so those samples are
    dropped. Duplicate time stamps are removed.
    """
    t, v = np.loadtxt(path, delimiter=",", skiprows=1, unpack=True)
    t, first = np.unique(t, return_index=True)
    v = v[first]
    start = np.argmax(v != v[0])
    return t[start:] - t[start], v[start:]


def forces(v, Cd, mu, s, theta_deg=THETA_DEG, v_wind=V_WIND, m=MASS):
    """Gravity, friction and drag forces [N] acting along the direction of travel."""
    th = np.radians(theta_deg)
    v_rel = v + s * v_wind
    f_grav = -s * m * G * np.sin(th)
    f_fric = -mu * m * G * np.cos(th) * np.ones_like(v)
    f_drag = -0.5 * Cd * AREA * RHO * v_rel * np.abs(v_rel)
    return f_grav, f_fric, f_drag


def simulate(t, v0, Cd, mu, s, **conditions):
    """Speed at times t from integrating the equation of motion."""
    m = conditions.get("m", MASS)
    dvdt = lambda _, v: sum(forces(v, Cd, mu, s, **conditions)) / m
    return solve_ivp(dvdt, (0, t[-1]), [v0], t_eval=t, rtol=1e-8).y[0]


def fit_run(t, v, s, **conditions):
    """Least-squares fit of Cd, mu_k and v0 to one run. Returns (Cd, mu, v0), rmse."""
    resid = lambda p: v - simulate(t, p[2], p[0], p[1], s, **conditions)
    sol = least_squares(resid, [0.3, 0.01, v[0]])
    return sol.x, np.sqrt(np.mean(sol.fun ** 2))


def fit_joint_free_theta(data):
    """One Cd and mu_k for BOTH runs, with the incline angle also fit.

    Returns (Cd, mu, theta_deg, v0_north, v0_south), rmse.
    """
    def resid(p):
        Cd, mu, th, *v0s = p
        return np.concatenate([
            v - simulate(t, v0, Cd, mu, s, theta_deg=th)
            for (t, v, s), v0 in zip(data.values(), v0s)])
    p0 = [0.3, 0.01, THETA_DEG] + [v[0] for _, v, _ in data.values()]
    sol = least_squares(resid, p0)
    return sol.x, np.sqrt(np.mean(sol.fun ** 2))


def load_all():
    return {name: (*load_run(path), s) for name, (path, s) in RUNS.items()}


def main():
    data = load_all()
    fits = {name: fit_run(t, v, s) for name, (t, v, s) in data.items()}
    joint, joint_rmse = fit_joint_free_theta(data)

    print(f"Fits with theta = {THETA_DEG} deg (from handout):")
    print(f"{'Run':<7}{'Cd':>8}{'mu_k':>9}{'v0 [m/s]':>10}{'RMSE [m/s]':>12}")
    for name, ((Cd, mu, v0), rmse) in fits.items():
        print(f"{name:<7}{Cd:>8.3f}{mu:>9.4f}{v0:>10.2f}{rmse:>12.3f}")
    Cd_avg = np.mean([f[0][0] for f in fits.values()])
    mu_avg = np.mean([f[0][1] for f in fits.values()])
    print(f"{'Mean':<7}{Cd_avg:>8.3f}{mu_avg:>9.4f}")
    print(f"\nCheck - one Cd, mu_k for both runs, theta also fit:")
    print(f"  Cd = {joint[0]:.3f}, mu_k = {joint[1]:.4f}, "
          f"theta = {joint[2]:.2f} deg, RMSE = {joint_rmse:.3f} m/s")

    # ---- Figure 1: model vs data ----
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex="col",
                             gridspec_kw={"height_ratios": [3, 1]})
    for col, (name, (t, v, s)) in enumerate(data.items()):
        (Cd, mu, v0), rmse = fits[name]
        v_fit = simulate(t, v0, Cd, mu, s)
        v_joint = simulate(t, joint[3 + col], joint[0], joint[1], s,
                           theta_deg=joint[2])
        top, bottom = axes[:, col]
        top.plot(t, v, ".", ms=2, color="0.65", label="GPS data")
        top.plot(t, v_fit, color="C3", lw=2,
                 label=f"fit (θ={THETA_DEG}°): Cd={Cd:.3f}, μk={mu:.4f}")
        top.plot(t, v_joint, "--", color="C0", lw=1.5,
                 label=f"joint fit (θ={joint[2]:.2f}°): "
                       f"Cd={joint[0]:.3f}, μk={joint[1]:.4f}")
        top.set(title=f"Traveling {name}", ylabel="speed [m/s]")
        top.legend(fontsize=8)
        bottom.plot(t, v - v_fit, ".", ms=2, color="C3")
        bottom.axhline(0, color="k", lw=0.8)
        bottom.set(xlabel="time [s]", ylabel="residual [m/s]")
    fig.suptitle("Non-linear model vs. measured coast-down")
    fig.tight_layout()
    fig.savefig("fig1_nonlinear_fit.png", dpi=150)

    # ---- Figure 2: what each term does ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    t, v, s = data["North"]
    (Cd, mu, v0), _ = fits["North"]
    base = simulate(t, v0, Cd, mu, s)
    for k, (label, args) in enumerate({
            "Cd +20%": (1.2 * Cd, mu), "μk +20%": (Cd, 1.2 * mu)}.items()):
        ax1.plot(t, simulate(t, v0, *args, s) - base, color=f"C{k}", label=label)
    ax1.set(xlabel="time [s]", ylabel="change in speed [m/s]",
            title="Effect of each parameter (North run)")
    ax1.axhline(0, color="k", lw=0.8)
    ax1.legend()
    # drag vs friction on level ground with no wind, using the mean values
    speeds = np.linspace(0, 45, 100)
    _, f_fric, f_drag = forces(speeds, Cd_avg, mu_avg, s=0, theta_deg=0)
    ax2.plot(speeds, -f_drag, label="aerodynamic drag  (∝ Cd v²)")
    ax2.plot(speeds, -f_fric, label="rolling friction  (∝ μk, constant)")
    for k, (name, (t, v, s)) in enumerate(data.items()):
        ax2.axvspan(v.min(), v.max(), color=f"C{k + 2}", alpha=0.12,
                    label=f"speeds covered going {name}")
    ax2.set(xlabel="speed [m/s]", ylabel="retarding force [N]",
            title=f"Level ground, no wind (Cd={Cd_avg:.3f}, μk={mu_avg:.4f})")
    ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("fig2_term_effects.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
