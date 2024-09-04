import numpy as np
import pandas as pd

def func(x, y):
    return (3 * x**2) * y

def two_step_adams_moulton(x0, y0, h, xn):
    data = {'x': [x0], f'y (h={h})': [y0]}

    while data['x'][-1] < xn:
        x = data['x'][-1]
        y = data[f'y (h={h})'][-1]

        # Predictor step using the Runge-Kutta method (4th order)
        k1 = h * func(x, y)
        k2 = h * func(x + h/2, y + k1/2)
        k3 = h * func(x + h/2, y + k2/2)
        k4 = h * func(x + h, y + k3)

        y_pred = y + (k1 + 2*k2 + 2*k3 + k4)/6

        # Corrector step using the Adams-Moulton formula
        x = x + h
        y_corr = y + (h/2) * (func(x, y_pred) + func(x, y))

        data['x'].append(x)
        data[f'y (h={h})'].append(y_corr)

    return pd.DataFrame(data)

# Parameters
x0 = 0
y0 = 1
xn = 1

# Step sizes
h_values = [0.1, 0.01, 0.001]

# Display results in tabular form
for h in h_values:
    df = two_step_adams_moulton(x0, y0, h, xn)
    print(f"\nResults for h = {h}:\n")
    print(df.round(5).to_string(index=False))
