# Nonlinear Model – Write-up (MeEn 335 Car Coast Down)

All numbers below come from `nonlinear_model.py` (run it to reproduce them and the figures).

*Note: these are uncertainty estimates from repeat tests and sensitivity checks, not a formal statistical confidence interval.*

---

## Final nonlinear equation of motion

Taking the positive direction as the direction the car is traveling:

$$
m\frac{dv}{dt} = -\,s\,m g\sin\theta \;-\; \mu_k\, m g\cos\theta \;-\; \tfrac{1}{2} C_d A \rho\, v_{rel}\,|v_{rel}|
$$

where *s* = +1 when traveling North (uphill) and *s* = −1 when traveling South (downhill), and v_rel is the speed of the car relative to the air: v_rel = v + v_wind going North (into the 7 mph headwind) and v_rel = v − v_wind going South (with the tailwind). The given values are m = 1910 kg (1760 kg car + 150 kg passengers), θ = 1.03°, A = 2.15 m², ρ = 1.06 kg/m³ and v_wind = 3.13 m/s.

**What each term means**

| Term | Meaning |
|---|---|
| m dv/dt | Mass times acceleration of the car. It is negative here because the car is slowing down. |
| −s·m·g·sinθ | Component of gravity along the road. It slows the car when going uphill (North) and speeds it up when going downhill (South). |
| −μk·m·g·cosθ | Rolling resistance modeled as Coulomb friction, F = μk·N, with normal force N = m·g·cosθ on the incline. It is the same size at every speed. |
| −½·Cd·A·ρ·v_rel·\|v_rel\| | Aerodynamic drag. It grows with the square of the car's speed relative to the air, so it depends on the wind. Writing v_rel·\|v_rel\| instead of v_rel² keeps the sign so drag always opposes motion relative to the air; in this test v_rel > 0 the whole time (the slowest speed, 9.5 m/s, is well above the 3.13 m/s wind), so the two are identical here. |

**Why the signs are correct.** Because positive is the direction of travel, any force that resists the motion must be negative. Friction and drag always resist motion, so both are always negative (for drag, relative to the air). Gravity is the only term whose sign depends on direction: it opposes the car going uphill and helps it going downhill. This matches the data: the North (uphill) run loses about 27 m/s in 70 s while the South (downhill) run loses only about 14 m/s in 82 s.

**Why the units are correct.** Every term is a force in newtons:

- m·g·sinθ → (kg)(m/s²) = N
- μk·m·g·cosθ → μk is dimensionless, so (kg)(m/s²) = N
- ½·Cd·A·ρ·v² → (m²)(kg/m³)(m²/s²) = kg·m/s² = N (Cd is dimensionless)

Dividing by m gives m/s², an acceleration, which is what dv/dt must be.

**Limiting-case checks.** With Cd = μk = 0 the equation reduces to dv/dt = −g·sinθ going uphill, the result for a block on a frictionless ramp. Without drag the deceleration would be constant and v(t) would be a straight line; the measured v(t) curves (it decelerates faster at high speed), which shows the v² drag term is needed.

---

## Power required on level ground

On level ground (θ = 0) with no wind, gravity drops out of the equation of motion. At a steady speed the car is not accelerating, so the engine must supply a force equal to rolling friction plus drag, and the power is that force times the speed:

$$
P = v\left(\mu_k m g + \tfrac{1}{2} C_d A \rho\, v^2\right)
$$

Power was computed in watts/kW using Cd = 0.291 and μk = 0.0154, then converted to horsepower (1 hp = 745.7 W) to compare with the GS350's 311 hp engine.

**Worked example, 55 mph** (v = 55 × 0.44704 = 24.59 m/s):

- Friction force: μk·m·g = 0.0154 × 1910 × 9.81 = 287.6 N
- Drag force: ½·Cd·A·ρ·v² = 0.5 × 0.291 × 2.15 × 1.06 × 24.59² = 200.1 N
- Power: (287.6 + 200.1) N × 24.59 m/s = 11.99 kW = **16.1 hp**

| Speed | Friction force | Drag force | Total force | Power | Power | % of 311 hp engine |
|---|---|---|---|---|---|---|
| 55 mph (24.59 m/s) | 287.6 N | 200.1 N | 487.8 N | 11.99 kW | **16.1 hp** | 5.2% |
| 100 mph (44.70 m/s) | 287.6 N | 661.6 N | 949.3 N | 42.44 kW | **56.9 hp** | 18.3% |

**Figure:** `fig_power.png` (section 7 of the script) plots the power needed versus speed, split into friction and drag.

- **Cruising uses a small fraction of the engine.** Even at 100 mph only about 18% of the rated 311 hp is needed; the rest is available for accelerating and climbing.
- **Power grows much faster than speed.** Going from 55 to 100 mph is 1.8× the speed but about 3.5× the power, because drag power grows with v³ while friction power grows only with v.
- **The dominant loss changes with speed.** Friction is 59% of the resistance at 55 mph, but drag is 70% of it at 100 mph. The two are equal at about 66 mph.
- The uncertainty in these values (16.1 ± 3.3 hp and 56.9 ± 0.5 hp) is discussed at the end of the uncertainty section.

---

## Uncertainty estimates: justification and procedure

| Parameter | Best estimate | Uncertainty |
|---|---|---|
| Drag coefficient, Cd | 0.29 | ± 0.06 (± 20%) |
| Kinetic friction coefficient, μk | 0.015 | ± 0.008 (± 49%) |
| Power at 55 mph | 16.1 hp | ± 3.3 hp |
| Power at 100 mph | 56.9 hp | ± 0.5 hp |

We used three procedures to quantify the uncertainty: (1) treating the two runs as repeated tests, (2) testing how sensitive the results are to the values given in the handout, and (3) mapping which combinations of Cd and μk fit the data acceptably.

### 1. Repeating the test (North vs. South runs)

The North and South runs are two independent coast-down tests of the same car, so the difference between them is a direct measurement of how much Cd and μk change if the test is repeated. The equation of motion was solved with `solve_ivp` and Cd, μk and the starting speed v0 were fit to each run separately with `least_squares` (bounded so none can go negative):

| Run | Cd | μk | RMS error |
|---|---|---|---|
| North (uphill) | 0.348 | 0.0079 | 0.136 m/s |
| South (downhill) | 0.233 | 0.0228 | 0.108 m/s |

Both fits match their data to within about the GPS speed resolution (~0.1 m/s), so each run on its own is well described by the model. The best estimate is the average of the two runs, and the uncertainty is half the difference:

- Cd = (0.348 + 0.233) / 2 = 0.291, uncertainty = |0.348 − 0.233| / 2 = **± 0.058**
- μk = (0.0079 + 0.0228) / 2 = 0.0154, uncertainty = |0.0079 − 0.0228| / 2 = **± 0.0075**

Averaging the two directions is also what removes the largest systematic error. An error in the incline angle makes the fitted μk too high in one direction and too low in the other by the same amount, so it cancels in the average (see section 2).

### 2. Sensitivity to the values given in the handout

Each given value was changed by a reasonable amount, both runs were refit, and the change in the averaged Cd and μk was recorded:

| Change | Change in Cd | Change in μk |
|---|---|---|
| North vs. South runs (from section 1) | 0.058 | 0.0075 |
| Incline θ ± 0.1° | 0.000 | 0.0000 |
| Wind ± 2 mph | 0.003 | 0.00004 |
| Mass ± 50 kg | 0.008 | 0.0000 |
| **Total (root-sum-square)** | **0.058** | **0.0075** |

- **Incline.** Changing θ by ±0.1° shifts each run's μk by about ±0.0017 (North 0.0096 / 0.0061, South 0.0211 / 0.0246), but in opposite directions, so the average does not change at all.
- **Wind.** A stronger wind makes the North (headwind) Cd smaller and the South (tailwind) Cd larger, so the effect mostly cancels as well, leaving ±0.003.
- **Mass.** Gravity and friction scale with m, but drag does not, so the mass only affects Cd (±0.008).

Combining all sources in quadrature gives essentially the same totals as the North/South comparison alone. The run-to-run difference dominates the uncertainty, so it is the value we report.

### 3. Which combinations of Cd and μk fit the data?

The RMS error between the model and the data was computed over a grid of Cd (0.15–0.45) and μk (0–0.035) for each run, holding each run's starting speed v0 at its best-fit value. **Figure:** `fig_uncertainty.png`, saved when `nonlinear_model.py` is run (section 8c). It shows the RMS = 0.2, 0.3 and 0.5 m/s contours for each run and the final estimate with its error bars.

- **Each parameter alone is tightly constrained.** Holding μk at its best value, Cd can only change by about ±2.5% before the North RMS error rises 50% (to 0.20 m/s). Holding Cd fixed, μk can only change by about ±6%. *(Separate check; not part of the script.)*
- **Together they can trade off.** The acceptable-fit regions are long, thin diagonal valleys, so a higher Cd with a lower μk fits almost as well. For example, forcing the North run's Cd to 0.279 or 0.418 (±20%) and refitting μk gives μk = 0.0105 or 0.0053, and the RMS error only rises from 0.136 to 0.195 m/s *(from the Cd vs. μk effects script)*. The least-squares fits give a correlation between Cd and μk of about −0.97 (North) and −0.99 (South) *(separate check)*.
- **The plot slightly understates the trade-off.** Because v0 is held fixed in the map, the valleys look narrower than they are: forcing the North Cd to ±20% gives an RMS of 0.25 m/s with v0 fixed but only 0.195 m/s when v0 is refit too.
- **The two valleys do not overlap.** No single (Cd, μk) pair fits both runs at θ = 1.03°, which is why the North and South fits differ. The most likely cause is that the road grade is not exactly 1.03° over the whole stretch (the elevation profile in the handout is steeper near the bottom), along with error in the reported wind. The final estimate lies between the two valleys, and its error bars reach both runs' best fits.

This trade-off is why Cd and μk are hard to separate. Drag dominates at high speed and friction at low speed, so a run with a wide speed range (the North run, 37 → 10 m/s) separates them better than a narrow one (the South run, 40 → 27 m/s).

### Effect on the horsepower estimates

Power on level ground with no wind is P = v·(μk·m·g + ½·Cd·A·ρ·v²). Using each run's own Cd and μk:

| Speed | North values | South values | Result |
|---|---|---|---|
| 55 mph | 12.8 hp | 19.4 hp | 16.1 ± 3.3 hp |
| 100 mph | 56.4 hp | 57.4 hp | 56.9 ± 0.5 hp |

The 100 mph estimate is much more certain than the separate ± values on Cd and μk would suggest. Because Cd and μk trade off against each other, the two fits give almost the same total resistance at high speed (near the top of the measured speed range), and they disagree more at 55 mph. Note that 100 mph is slightly above the fastest test speed (about 92 mph), so it is a small extrapolation.
