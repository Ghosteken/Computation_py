import numpy as np
import matplotlib.pyplot as plt

def func(x, y):
    return x + y

def adams_bashforth(x0, y0, h, xn):
    x_values = [x0]
    y_values = [y0]

    while x_values[-1] < xn:
        x = x_values[-1]
        y = y_values[-1]

        # Adams-Bashforth formula (fourth-order)
        f1 = func(x, y)
        f2 = func(x - h, y_values[-2] if len(y_values) > 1 else y0)
        f3 = func(x - 2*h, y_values[-3] if len(y_values) > 2 else y0)
        f4 = func(x - 3*h, y_values[-4] if len(y_values) > 3 else y0)

        y_next = y + (h/24) * (55*f1 - 59*f2 + 37*f3 - 9*f4)

        x = x + h

        x_values.append(x)
        y_values.append(y_next)

    return x_values, y_values

# Parameters
x0 = 0
y0 = 1
xn = 1
h = 0.1

# Apply the Adams-Bashforth method
x_vals, y_vals = adams_bashforth(x0, y0, h, xn)

# Plotting
plt.plot(x_vals, y_vals, label=f'Adams-Bashforth (h={h})')
plt.title("Adams-Bashforth Method for y' = x + y")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()
