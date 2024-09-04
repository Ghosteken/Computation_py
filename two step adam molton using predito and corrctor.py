import numpy as np
import matplotlib.pyplot as plt

def func(x, y):
    return (3 * x**2) * y

def two_step_adams_moulton(x0, y0, h, xn):
    x_values = [x0]
    y_values = [y0]

    while x_values[-1] < xn:
        x = x_values[-1]
        y = y_values[-1]

        # Predictor step using the Runge-Kutta method (4th order)
        k1 = h * func(x, y)
        k2 = h * func(x + h/2, y + k1/2)
        k3 = h * func(x + h/2, y + k2/2)
        k4 = h * func(x + h, y + k3)

        y_pred = y + (k1 + 2*k2 + 2*k3 + k4)/6

        # Corrector step using the Adams-Moulton formula
        x = x + h
        y_corr = y + (h/2) * (func(x, y_pred) + func(x, y))

        x_values.append(x)
        y_values.append(y_corr)

    return x_values, y_values

# Parameters
x0 = 0
y0 = 1
xn = 1

# Step sizes
h_values = [0.1, 0.01, 0.001]

# Plotting
for h in h_values:
    x_vals, y_vals = two_step_adams_moulton(x0, y0, h, xn)
    plt.plot(x_vals, y_vals, label=f'h = {h}')

plt.title("Two-Step Adams-Moulton Method for y' = (3x^2)y")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()
