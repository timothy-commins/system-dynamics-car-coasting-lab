"""
Car Coasting Lab - Power prediction and uncertainty analysis  (MeEn 335)

Uses the non-linear model in nonlinear_model.py.

Power on level ground with no wind (steady speed, so engine force = losses):

    P = v * ( mu_k m g  +  1/2 Cd A rho v^2 )

Uncertainty in Cd and mu_k comes from three sources, combined in quadrature:
  1. GPS noise          - least-squares covariance of each fit
  2. Repeating the test - the North and South runs give different values
  3. Given inputs       - refit with the incline, wind and mass changed

Figure 3 shows which (Cd, mu_k) combinations fit each run acceptably;
Figure 4 shows the predicted power with its uncertainty band.
"""

import numpy as np
import matplotlib.pyplot as plt

from nonlinear_model import (MASS, AREA, RHO, G, THETA_DEG, V_WIND,
                             load_all, simulate, fit_run, fit_joint_free_theta)

MPH = 0.44704      # m/s per mph
HP = 745.7         # W per hp
ENGINE_HP = 311    # 2016 Lexus GS350 rated power
SPEEDS_MPH = [55, 100]

# How far each given input is changed to test sensitivity (our judgement):
INPUT_CHANGES = {
    "incline θ ± 0.1°":   [dict(theta_deg=THETA_DEG + d) for d in (-0.1, 0.1)],
    "wind ± 2 mph":       [dict(v_wind=V_WIND + d * MPH) for d in (-2, 2)],
    "mass ± 50 kg":       [dict(m=MASS + d) for d in (-50, 50)],
}


def power_hp(v, Cd, mu):
    """Power [hp] to hold speed v [m/s] on level ground with no wind."""
    return v * (mu * MASS * G + 0.5 * Cd * AREA * RHO * v ** 2) / HP


def gps_uncertainty(t, v, s, params):
    """2x2 covariance matrix of (Cd, mu) from the least-squares fit.

    The logger records each GPS speed many times, so the number of
    independent measurements is the number of speed updates, not samples.
    """
    Cd, mu, v0 = params
    base = simulate(t, v0, Cd, mu, s)
    resid = v - base
    # sensitivity of the simulated speed to each parameter (finite differences)
    J = np.column_stack([
        (simulate(t, v0, Cd + 1e-3, mu, s) - base) / 1e-3,
        (simulate(t, v0, Cd, mu + 1e-5, s) - base) / 1e-5,
        (simulate(t, v0 + 1e-3, Cd, mu, s) - base) / 1e-3])
    n_independent = np.count_nonzero(np.diff(v)) + 1
    noise_var = np.sum(resid ** 2) / (len(v) - 3)
    cov = np.linalg.inv(J.T @ J) * noise_var * len(v) / n_independent
    return cov[:2, :2]


def main():
    data = load_all()
    fits = {name: fit_run(t, v, s)[0] for name, (t, v, s) in data.items()}
    Cd = np.mean([p[0] for p in fits.values()])
    mu = np.mean([p[1] for p in fits.values()])

    # 1. GPS noise (averaging two runs divides the variance by 2)
    gps = {name: gps_uncertainty(*data[name], p) for name, p in fits.items()}
    cov_gps = sum(gps.values()) / 4
    sd_Cd_gps, sd_mu_gps = np.sqrt(np.diag(cov_gps))

    # 2. repeating the test: half the North-South difference
    sd_Cd_rep = abs(fits["North"][0] - fits["South"][0]) / 2
    sd_mu_rep = abs(fits["North"][1] - fits["South"][1]) / 2

    # 3. given inputs: refit both runs with each input changed
    scenarios = {"North run only": fits["North"][:2],
                 "South run only": fits["South"][:2]}
    joint, _ = fit_joint_free_theta(data)
    scenarios[f"joint fit, θ fit = {joint[2]:.2f}°"] = joint[:2]
    input_rows = []
    for label, cases in INPUT_CHANGES.items():
        results = []
        for cond in cases:
            p = [fit_run(t, v, s, **cond)[0] for t, v, s in data.values()]
            results.append(np.mean(p, axis=0)[:2])
        dCd = max(abs(r[0] - Cd) for r in results)
        dmu = max(abs(r[1] - mu) for r in results)
        input_rows.append((label, dCd, dmu))
        for sign, r in zip("-+", results):
            scenarios[f"{label.split(' ±')[0]} {sign}"] = r
    sd_Cd_in = np.sqrt(sum(r[1] ** 2 for r in input_rows))
    sd_mu_in = np.sqrt(sum(r[2] ** 2 for r in input_rows))

    sd_Cd = np.sqrt(sd_Cd_gps ** 2 + sd_Cd_rep ** 2 + sd_Cd_in ** 2)
    sd_mu = np.sqrt(sd_mu_gps ** 2 + sd_mu_rep ** 2 + sd_mu_in ** 2)

    print(f"Best estimate (mean of North and South fits, θ = {THETA_DEG}°)")
    print(f"  Cd   = {Cd:.3f} ± {sd_Cd:.3f}")
    print(f"  mu_k = {mu:.4f} ± {sd_mu:.4f}\n")
    print(f"{'Uncertainty source':<26}{'± Cd':>8}{'± mu_k':>10}")
    print(f"{'GPS noise (fit)':<26}{sd_Cd_gps:>8.3f}{sd_mu_gps:>10.4f}")
    print(f"{'Repeat test (N vs S)':<26}{sd_Cd_rep:>8.3f}{sd_mu_rep:>10.4f}")
    for label, dCd, dmu in input_rows:
        print(f"{label:<26}{dCd:>8.3f}{dmu:>10.4f}")
    print(f"{'Combined':<26}{sd_Cd:>8.3f}{sd_mu:>10.4f}")
    for name, c in gps.items():
        r = c[0, 1] / np.sqrt(c[0, 0] * c[1, 1])
        print(f"  ({name} fit: Cd and mu_k correlation = {r:+.2f})")

    # power for the best estimate and for every alternative fit
    print(f"\n{'Power [hp]':<26}" + "".join(f"{s:>4} mph" for s in SPEEDS_MPH))
    print(f"{'Best estimate':<26}"
          + "".join(f"{power_hp(s * MPH, Cd, mu):>8.1f}" for s in SPEEDS_MPH))
    spread = {s: [] for s in SPEEDS_MPH}
    for label, (c, m) in scenarios.items():
        row = [power_hp(s * MPH, c, m) for s in SPEEDS_MPH]
        for s, p in zip(SPEEDS_MPH, row):
            spread[s].append(p)
        print(f"  {label:<24}" + "".join(f"{p:>8.1f}" for p in row))
    for s in SPEEDS_MPH:
        best = power_hp(s * MPH, Cd, mu)
        # alternative fits, plus GPS noise propagated through P(Cd, mu)
        err_fits = max(abs(np.array(spread[s]) - best))
        grad = np.array([0.5 * AREA * RHO * (s * MPH) ** 3, MASS * G * s * MPH]) / HP
        err = np.sqrt(err_fits ** 2 + grad @ cov_gps @ grad)
        print(f"{s} mph: {best:.1f} ± {err:.1f} hp "
              f"({100 * best / ENGINE_HP:.0f}% of the {ENGINE_HP} hp engine)")

    # ---- Figure 3: which (Cd, mu_k) pairs fit each run? ----
    # RMSE of each run over a grid of (Cd, mu_k); left: handout incline,
    # right: incline from the joint fit. v0 is held at each run's best fit.
    Cd_grid = np.linspace(0.15, 0.45, 61)
    mu_grid = np.linspace(0.0, 0.035, 61)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, theta in zip(axes, [THETA_DEG, joint[2]]):
        for k, (name, (t, v, s)) in enumerate(data.items()):
            t_sub, v_sub = t[::10], v[::10]           # thinned for speed
            v0 = fits[name][2]
            rmse = np.array([[np.sqrt(np.mean(
                (v_sub - simulate(t_sub, v0, c, m, s, theta_deg=theta)) ** 2))
                for c in Cd_grid] for m in mu_grid])
            cs = ax.contour(Cd_grid, mu_grid, rmse, levels=[0.2, 0.3, 0.5],
                            colors=f"C{k}", linestyles=["-", "--", ":"])
            ax.clabel(cs, fmt=lambda x: f"{x:.1f}", fontsize=7)
            ax.plot([], [], color=f"C{k}", label=f"{name} run, RMSE = 0.2/0.3/0.5 m/s")
        ax.set(xlabel="drag coefficient Cd", title=f"incline θ = {theta:.2f}°")
    axes[0].plot(*fits["North"][:2], "o", color="C0")
    axes[0].plot(*fits["South"][:2], "o", color="C1")
    axes[0].errorbar(Cd, mu, xerr=sd_Cd, yerr=sd_mu, fmt="k*", ms=12,
                     capsize=4, label="best estimate ± uncertainty")
    axes[1].plot(*joint[:2], "k*", ms=12, label="joint fit (both runs)")
    axes[0].set_ylabel("rolling friction μk")
    for ax in axes:
        ax.legend(fontsize=8, loc="upper right")
    fig.suptitle("Combinations of Cd and μk that fit each run")
    fig.tight_layout()
    fig.savefig("fig3_parameter_tradeoff.png", dpi=150)

    # ---- Figure 4: power vs speed ----
    mph = np.linspace(0, 110, 200)
    curves = np.array([power_hp(mph * MPH, c, m) for c, m in scenarios.values()])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.fill_between(mph, curves.min(0), curves.max(0), alpha=0.25,
                    label="range over alternative fits")
    ax.plot(mph, power_hp(mph * MPH, Cd, mu), "k", label="best estimate")
    for s in SPEEDS_MPH:
        p = power_hp(s * MPH, Cd, mu)
        ax.plot(s, p, "o", color="C3")
        ax.annotate(f"{p:.0f} hp", (s, p), textcoords="offset points",
                    xytext=(-35, 8))
    all_speeds = np.concatenate([v for _, v, _ in data.values()]) / MPH
    ax.axvspan(all_speeds.min(), all_speeds.max(), color="0.5", alpha=0.1,
               label="speed range of the test data")
    ax.set(xlabel="speed [mph]", ylabel="power required [hp]",
           title="Power to drive on level ground (no wind)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("fig4_power.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
