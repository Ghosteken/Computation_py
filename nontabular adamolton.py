import numpy as np
import matplotlib.pyplot as plt

def func(x, y):
    return x + y

def adams_moulton(x0, y0, h, xn):
    x_values = [x0]
    y_values = [y0]

    while x_values[-1] < xn:
        x = x_values[-1]
        y = y_values[-1]

        # Adams-Moulton formula (implicit, second-order)
        f1 = func(x, y)
        f2 = func(x - h, y_values[-1])

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

        x_values.append(x)
        y_values.append(y_next)

    return x_values, y_values

# Parameters
x0 = 0
y0 = 1
xn = 1
h = 0.1

# Apply the Adams-Moulton method
x_vals, y_vals = adams_moulton(x0, y0, h, xn)

# Plotting
plt.plot(x_vals, y_vals, label=f'Adams-Moulton (h={h})')
plt.title("Adams-Moulton Method for y' = x + y")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()
