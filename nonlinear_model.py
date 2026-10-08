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
#   drag     = 1/2 Cd A rho v_rel^2, where v_rel = speed relative to the air:
#              v + v_wind going North (headwind), v - v_wind going South

def dv_dt(t, v, Cd, mu_k, direction):
    if direction == "North":
        gravity = m * g * np.sin(theta)
        v_rel = v + v_wind
    else:
        gravity = -m * g * np.sin(theta)
        v_rel = v - v_wind
    friction = mu_k * m * g * np.cos(theta)
    drag = 0.5 * Cd * A * rho * v_rel**2
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
    result = least_squares(error, [0.3, 0.01, v[0]])
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
# At a steady speed the engine force equals friction + drag, and P = F * v.

def power_hp(v, Cd, mu_k):
    force = mu_k * m * g + 0.5 * Cd * A * rho * v**2
    return force * v / 745.7          # 745.7 W = 1 hp

v55 = 55 * 0.44704
v100 = 100 * 0.44704
print(f"\nPower at  55 mph: {power_hp(v55, Cd, mu_k):.1f} hp")
print(f"Power at 100 mph: {power_hp(v100, Cd, mu_k):.1f} hp")
print("(The 2016 GS350 engine is rated at 311 hp.)")


# %% 8. Uncertainty
#
# (a) Repeating the test: the North and South runs are two separate tests,
#     so the difference between them shows how much Cd and mu_k could vary.

dCd = abs(Cd_N - Cd_S) / 2
dmu = abs(mu_N - mu_S) / 2
print(f"\nCd   = {Cd:.3f} ± {dCd:.3f}")
print(f"mu_k = {mu_k:.4f} ± {dmu:.4f}")

# Effect on horsepower: compare the power from each run's own values
for speed, name in [(v55, " 55 mph"), (v100, "100 mph")]:
    p_N = power_hp(speed, Cd_N, mu_N)
    p_S = power_hp(speed, Cd_S, mu_S)
    print(f"{name}: {power_hp(speed, Cd, mu_k):.1f} ± {abs(p_N - p_S) / 2:.1f} hp")

# (b) Different combinations that fit equally well: compute the RMS error
#     over a grid of Cd and mu_k values. A long, thin valley means a higher Cd
#     with a lower mu_k fits almost as well, so the two are hard to separate.

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
