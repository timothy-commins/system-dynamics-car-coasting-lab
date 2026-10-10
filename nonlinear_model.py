# Car Coasting Lab - Nonlinear model (MeEn 335)
#
# Run the whole file (F5 in Spyder), or one cell at a time (Ctrl+Enter).
# The data files north_data.csv and south_data.csv must be in the same
# folder as this file.

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

# Look for the data files in the same folder as this script
if "__file__" in globals():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))


# %% 1. Values given in the lab handout

m = 1760 + 150                    # mass of car + passengers [kg]
g = 9.81                          # gravity [m/s^2]
theta = np.radians(1.03)          # road incline [rad]
A = 2.15                          # frontal area [m^2]
rho = 1.06                        # air density [kg/m^3]
v_wind = 7 * 0.44704              # 7 mph wind from the North [m/s]


# %% 2. Load the GPS data

def load_data(filename):
    data = np.loadtxt(filename, delimiter=",", skiprows=1)
    # Some time stamps appear twice; keep one of each (solve_ivp needs
    # strictly increasing times)
    t, keep = np.unique(data[:, 0], return_index=True)
    v = data[keep, 1]
    # The logger repeats its first reading for ~0.7 s before it starts
    # updating, so start the data where the speed first changes.
    start = np.argmax(v != v[0])
    return t[start:] - t[start], v[start:]

t_north, v_north = load_data("north_data.csv")   # driving uphill, into the wind
t_south, v_south = load_data("south_data.csv")   # driving downhill, wind behind


# %% 3. Equation of motion
#
#   m dv/dt = - (gravity) - (rolling friction) - (aerodynamic drag)
#
#   gravity  = +m g sin(theta) going North (uphill), -m g sin(theta) going South
#   friction = mu_k * N = mu_k m g cos(theta)          (always slows the car)
#   drag     = 1/2 Cd A rho v_rel|v_rel|, where v_rel = speed relative to the air:
#              v + v_wind going North (headwind), v - v_wind going South

def dv_dt(t, v, Cd, mu_k, direction):
    if direction == "North":
        gravity = m * g * np.sin(theta)
        v_rel = v + v_wind
    else:
        gravity = -m * g * np.sin(theta)
        v_rel = v - v_wind
    friction = mu_k * m * g * np.cos(theta)
    # v_rel * |v_rel| is v_rel^2 with the sign kept, so drag always opposes
    # the car's motion relative to the air (same as v_rel^2 when v_rel > 0)
    drag = 0.5 * Cd * A * rho * v_rel * abs(v_rel)
    return -(gravity + friction + drag) / m


def simulate(t, v0, Cd, mu_k, direction):
    """Solve the equation of motion with solve_ivp; return speed at times t."""
    sol = solve_ivp(dv_dt, [0, t[-1]], [v0], t_eval=t,
                    args=(Cd, mu_k, direction), rtol=1e-8)
    return sol.y[0]


# %% 4. Find the Cd and mu_k that best match each run
#
# least_squares adjusts [Cd, mu_k, v0] until the simulated speed matches the
# GPS speed as closely as possible. v0 (starting speed) is also adjusted
# because the first GPS reading is noisy.

def fit(t, v, direction):
    def error(p):
        Cd, mu_k, v0 = p
        return v - simulate(t, v0, Cd, mu_k, direction)
    # bounds keep Cd, mu_k and v0 from going negative (not physical)
    result = least_squares(error, [0.3, 0.01, v[0]],
                           bounds=([0, 0, 0], [2, 1, np.inf]))
    Cd, mu_k, v0 = result.x
    rms_error = np.sqrt(np.mean(result.fun**2))
    return Cd, mu_k, v0, rms_error

Cd_N, mu_N, v0_N, err_N = fit(t_north, v_north, "North")
Cd_S, mu_S, v0_S, err_S = fit(t_south, v_south, "South")

# Final answer: average of the two runs. Any error in the incline angle makes
# mu_k too high in one direction and too low in the other, so it cancels.
Cd = (Cd_N + Cd_S) / 2
mu_k = (mu_N + mu_S) / 2

print("Run      Cd      mu_k     RMS error [m/s]")
print(f"North  {Cd_N:.3f}   {mu_N:.4f}    {err_N:.3f}")
print(f"South  {Cd_S:.3f}   {mu_S:.4f}    {err_S:.3f}")
print(f"Mean   {Cd:.3f}   {mu_k:.4f}")


# %% 5. Plot: model vs. data (both runs on one graph)
#
# GPS data = open circles (every 60th point so they don't blur together)
# Model    = thick solid lines

every = 60
plt.figure(figsize=(8, 5))
plt.plot(t_north[::every], v_north[::every], "o", markerfacecolor="none",
         markeredgecolor="cornflowerblue", markersize=4, label="North GPS data")
plt.plot(t_north, simulate(t_north, v0_N, Cd_N, mu_N, "North"), color="navy",
         linewidth=2.5, label=f"North model: Cd = {Cd_N:.3f}, μk = {mu_N:.4f}")
plt.plot(t_south[::every], v_south[::every], "s", markerfacecolor="none",
         markeredgecolor="orange", markersize=4, label="South GPS data")
plt.plot(t_south, simulate(t_south, v0_S, Cd_S, mu_S, "South"), color="darkred",
         linewidth=2.5, label=f"South model: Cd = {Cd_S:.3f}, μk = {mu_S:.4f}")
plt.title("Nonlinear model vs. GPS data")
plt.xlabel("time [s]")
plt.ylabel("speed [m/s]")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("fig_nonlinear_fit.png", dpi=150)


# %% 6. What does each term do?
#
# Raise Cd or mu_k by 20% and see how the North run's speed changes.
# Drag matters most at high speed (early in the run); friction slows the car
# by the same amount at every speed, so its effect grows steadily with time.

base = simulate(t_north, v0_N, Cd_N, mu_N, "North")
more_drag = simulate(t_north, v0_N, 1.2 * Cd_N, mu_N, "North")
more_friction = simulate(t_north, v0_N, Cd_N, 1.2 * mu_N, "North")

plt.figure(figsize=(6, 4))
plt.plot(t_north, more_drag - base, label="Cd + 20%")
plt.plot(t_north, more_friction - base, label="μk + 20%")
plt.title("Effect of each parameter (North run)")
plt.xlabel("time [s]")
plt.ylabel("change in speed [m/s]")
plt.legend()
plt.tight_layout()
plt.savefig("fig_term_effects.png", dpi=150)


# %% 7. Power needed on level ground (no wind)
#
# On level ground (theta = 0) with no wind, gravity drops out. At a steady
# speed the engine force equals rolling friction + drag, and power = F * v:
#
#   P = v * (mu_k m g  +  1/2 Cd A rho v^2)
#
# Computed in metric units (W, then kW) and reported in horsepower
# (1 hp = 745.7 W) so it can be compared with the engine's maximum.

ENGINE_HP = 311                   # 2016 Lexus GS350 rated power [hp]
MPH = 0.44704                     # 1 mph in m/s


def power_hp(v, Cd, mu_k):
    """Power [hp] to hold speed v [m/s] on level ground with no wind."""
    force = mu_k * m * g + 0.5 * Cd * A * rho * v**2
    return force * v / 745.7


v55 = 55 * MPH
v100 = 100 * MPH

print("\nPower on level ground, no wind (Cd and mu_k from section 4)")
print("Speed     Friction  Drag     Total    Power    Power    % of")
print("          [N]       [N]      [N]      [kW]     [hp]     engine")
for mph in [55, 100]:
    v = mph * MPH
    F_friction = mu_k * m * g
    F_drag = 0.5 * Cd * A * rho * v**2
    F_total = F_friction + F_drag
    P_kW = F_total * v / 1000
    P_hp = F_total * v / 745.7
    print(f"{mph:>3} mph   {F_friction:6.1f}   {F_drag:6.1f}   {F_total:6.1f}   "
          f"{P_kW:6.2f}   {P_hp:6.1f}   {100 * P_hp / ENGINE_HP:4.1f}%")
print(f"(The 2016 GS350 engine is rated at {ENGINE_HP} hp.)")

# Plot: power needed vs. speed, split into friction and drag
mph_range = np.linspace(0, 110, 200)
v_range = mph_range * MPH
P_friction = mu_k * m * g * v_range / 745.7
P_drag = 0.5 * Cd * A * rho * v_range**3 / 745.7

plt.figure(figsize=(7, 4.5))
plt.plot(mph_range, P_friction + P_drag, "k", linewidth=2.5, label="total")
plt.plot(mph_range, P_friction, "--", color="C1", label="rolling friction (∝ v)")
plt.plot(mph_range, P_drag, "--", color="C0", label="aerodynamic drag (∝ v³)")
for mph in [55, 100]:
    P = power_hp(mph * MPH, Cd, mu_k)
    plt.plot(mph, P, "o", color="C3")
    plt.annotate(f"{P:.1f} hp ({100 * P / ENGINE_HP:.0f}% of engine)", (mph, P),
                 textcoords="offset points", xytext=(-120, 8))
plt.title("Power needed to drive on level ground (no wind)")
plt.xlabel("speed [mph]")
plt.ylabel("power [hp]")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("fig_power.png", dpi=150)


# %% 8. Uncertainty
#
# (a) Repeating the test: the North and South runs are two separate tests of
#     the same car, so the difference between them shows how much Cd and
#     mu_k could vary.  Best estimate = average of the two runs,
#     uncertainty = half the difference between them.
#
#     - mu_k has the larger relative uncertainty: friction is a small force
#       compared with gravity on the slope, so small errors in the slope or
#       wind change mu_k a lot.
#     - The two uncertainties are linked: a higher Cd with a lower mu_k fits
#       almost as well (see the contour plot below), so the power estimates
#       are more certain than the separate ± values suggest.
#     - Changing the slope, wind or mass by reasonable amounts moves the
#       averages much less than the North/South difference, so the
#       run-to-run spread is used as the uncertainty.

dCd = abs(Cd_N - Cd_S) / 2
dmu = abs(mu_N - mu_S) / 2
print(f"\nCd   = {Cd:.3f} ± {dCd:.3f}   (± {100 * dCd / Cd:.0f}%)")
print(f"mu_k = {mu_k:.4f} ± {dmu:.4f}  (± {100 * dmu / mu_k:.0f}%)")

# Effect on horsepower: compare the power from each run's own values
for speed, name in [(v55, " 55 mph"), (v100, "100 mph")]:
    p_N = power_hp(speed, Cd_N, mu_N)
    p_S = power_hp(speed, Cd_S, mu_S)
    print(f"{name}: {power_hp(speed, Cd, mu_k):.1f} ± {abs(p_N - p_S) / 2:.1f} hp")

# (b) Effect of the values given in the handout: change the incline, wind
#     and mass by a reasonable amount, refit both runs, and see how much the
#     averaged Cd and mu_k move. These are combined with (a) "in quadrature"
#     (square root of the sum of squares) to get the total uncertainty.

def refit_with(theta_deg=1.03, wind_mph=7, mass=1910):
    global theta, v_wind, m
    saved = theta, v_wind, m
    theta, v_wind, m = np.radians(theta_deg), wind_mph * 0.44704, mass
    Cd_n, mu_n, _, _ = fit(t_north, v_north, "North")
    Cd_s, mu_s, _, _ = fit(t_south, v_south, "South")
    theta, v_wind, m = saved
    return (Cd_n + Cd_s) / 2, (mu_n + mu_s) / 2

changes = {"incline ± 0.1 deg": [dict(theta_deg=0.93), dict(theta_deg=1.13)],
           "wind ± 2 mph":      [dict(wind_mph=5), dict(wind_mph=9)],
           "mass ± 50 kg":      [dict(mass=1860), dict(mass=1960)]}

print("\nUncertainty budget          ± Cd     ± mu_k")
print(f"North vs. South runs       {dCd:.4f}   {dmu:.5f}")
total_Cd2, total_mu2 = dCd**2, dmu**2
for label, cases in changes.items():
    results = [refit_with(**c) for c in cases]
    change_Cd = max(abs(r[0] - Cd) for r in results)
    change_mu = max(abs(r[1] - mu_k) for r in results)
    total_Cd2 += change_Cd**2
    total_mu2 += change_mu**2
    print(f"{label:<26} {change_Cd:.4f}   {change_mu:.5f}")
print(f"Total (quadrature)         {np.sqrt(total_Cd2):.4f}   {np.sqrt(total_mu2):.5f}")

# (c) Different combinations that fit equally well: compute the RMS error
#     over a grid of Cd and mu_k values. A long, thin valley means a higher Cd
#     with a lower mu_k fits almost as well, so the two are hard to separate.
#     Note: v0 is held at each run's best fit here (refitting it at every
#     grid point is too slow), so the valleys look a bit narrower than they
#     really are. E.g. forcing the North Cd to +/-20% gives RMS 0.25 m/s
#     with v0 fixed, but only 0.195 m/s when v0 is also refit.

Cd_values = np.linspace(0.15, 0.45, 60)
mu_values = np.linspace(0.0, 0.035, 60)
plt.figure(figsize=(7, 5))
for t, v, v0, direction, color in [(t_north, v_north, v0_N, "North", "C0"),
                                   (t_south, v_south, v0_S, "South", "C1")]:
    rms = np.zeros((len(mu_values), len(Cd_values)))
    for i, mu_try in enumerate(mu_values):
        for j, Cd_try in enumerate(Cd_values):
            v_model = simulate(t[::10], v0, Cd_try, mu_try, direction)
            rms[i, j] = np.sqrt(np.mean((v[::10] - v_model)**2))
    plt.contour(Cd_values, mu_values, rms, levels=[0.2, 0.3, 0.5], colors=color)
    plt.plot([], [], color=color, label=f"{direction} run (RMS error 0.2, 0.3, 0.5 m/s)")
plt.errorbar(Cd, mu_k, xerr=dCd, yerr=dmu, fmt="k*", markersize=12,
             capsize=4, label="final answer ± uncertainty")
plt.xlabel("drag coefficient Cd")
plt.ylabel("rolling friction μk")
plt.title("Combinations of Cd and μk that fit each run")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig("fig_uncertainty.png", dpi=150)
plt.show()
