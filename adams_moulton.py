import numpy as np
import pandas as pd

def func(x, y):
    return x + y

def adams_moulton(x0, y0, h, xn):
    data = {'x': [x0], f'y (h={h})': [y0]}

    while data['x'][-1] < xn:
        x = data['x'][-1]
        y = data[f'y (h={h})'][-1]

        # Adams-Moulton formula (implicit, second-order)
        f1 = func(x, y)
        f2 = func(x - h, data[f'y (h={h})'][-1])

        # Solve the implicit equation using Newton's method
        def implicit_equation(y_next):
            return y - y_next - (h/2) * (f1 + func(x - h, y_next))

        y_next_guess = y + h * f1  # Initial guess using Adams-Bashforth
        y_next = y + h * f1  # Initialize y_next to start the iteration

        # Newton's method for solving the implicit equation
        while abs(y_next - y_next_guess) > 1e-8:
            y_next_guess = y_next
            y_next = y + (h/2) * (f1 + func(x, y_next))

        x = x + h

        data['x'].append(x)
        data[f'y (h={h})'].append(y_next)

    return pd.DataFrame(data)

# Parameters
x0 = 0
y0 = 1
xn = 1
h = 0.1

# Apply the Adams-Moulton method
df = adams_moulton(x0, y0, h, xn)

# Display results in tabular form
print(df.round(5).to_string(index=False))
