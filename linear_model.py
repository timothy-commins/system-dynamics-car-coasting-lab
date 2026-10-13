import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Given Values
m = 1760 + 150 # (kg)
g = 9.81 # (m/s^2)
theta = np.deg2rad(1.03) # convert degrees to radians

# f(t) uphill and downhill = force of gravity
fg_south = m * g * np.sin(theta)
fg_north = -m * g * np.sin(theta)

# Equation
def linear_deriv(t, v, b, fg):
    return (fg - b * v) / m

# Guess for damping coefficient b
b_guess = 20 

# Initial velocities for uphill(north) and downhill(south)
v0_south = 4.127999877999999967e+01
v0_north = 3.772999954000000145e+01 

# Time spans (0 to max)
t_span_south = (0, 8.226399999999999579e+01)
t_span_north = (0, 6.985800000000000409e+01)


# Simulation solve
sol_south = solve_ivp(linear_deriv, t_span_south, [v0_south], args=(b_guess, fg_south), t_eval=np.linspace(t_span_south[0], t_span_south[1], 100))
sol_north = solve_ivp(linear_deriv, t_span_north, [v0_north], args=(b_guess, fg_north), t_eval=np.linspace(t_span_north[0], t_span_north[1], 100))

# Plot to match given graph
plt.figure(figsize=(8, 5))
plt.plot(sol_south.t, sol_south.y[0], label='South (Downhill)', color='blue')
plt.plot(sol_north.t, sol_north.y[0], label='North (Uphill)', color='orange')
plt.title('Speed vs. Time: Linear Model')
plt.xlabel('Time (s)')
plt.ylabel('Speed (m/s)')
plt.grid(True)
plt.legend()
plt.show()
