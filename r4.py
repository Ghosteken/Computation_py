import numpy as np

def rk4(f, x0, y0, h, tspan):
    """
    Solves an initial value problem using the 4th-order Runge-Kutta method.

    Parameters:
        f: The right-hand side of the differential equation.
        x0: The initial value of the independent variable.
        y0: The initial value of the dependent variable.
        h: The step size.
        tspan: The time span over which to solve the problem.

    Returns:
        x: A vector of the independent variable values.
        y: A vector of the dependent variable values.
    """

    x = np.linspace(tspan[0], tspan[1], int((tspan[1] - tspan[0]) / h + 1))
    y = np.zeros(len(x))
    y[0] = y0

    # Table header
    print("| Step | Time (x) | Dependent Variable (y) | k1 | k2 | k3 | k4 |")
    print("|---|---|---|---|---|---|---|")

    for i in range(1, len(x)):
        # Calculate k1, k2, k3, and k4
        k1 = f(x[i-1], y[i-1])
        k2 = f(x[i-1] + h/2, y[i-1] + h/2 * k1)
        k3 = f(x[i-1] + h/2, y[i-1] + h/2 * k2)
        k4 = f(x[i-1] + h, y[i-1] + h * k3)

        # Update the dependent variable
        y[i] = y[i-1] + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        # Table row
        print(f"| {i:3} | {x[i]:3.2f} | {y[i]:5.2f} | {k1:5.2f} | {k2:5.2f} | {k3:5.2f} | {k4:5.2f} |")

    return x, y


import numpy as np

def f(x, y):
    return x + y

# Define parameters
x0 = 0.0
y0 = 1.0
h = 0.1
tspan = [0.0, 1.0]

# Solve the initial value problem using the rk4 method
x, y = rk4(f, x0, y0, h, tspan)
