def runge_kutta(f, y0, t0, t_end, h):
    """
    Solves a first-order ordinary differential equation using the fourth-order Runge-Kutta method.

    Parameters:
    - f: The function representing the ODE dy/dt = f(t, y)
    - y0: Initial value of the dependent variable y at t0
    - t0: Initial value of the independent variable
    - t_end: End value of the independent variable
    - h: Step size

    Returns:
    - t_values: List of time values
    - y_values: List of corresponding y values
    """
    t_values = [t0]
    y_values = [y0]

    while t_values[-1] < t_end:
        t = t_values[-1]
        y = y_values[-1]

        k1 = h * f(t, y)
        k2 = h * f(t + 0.5 * h, y + 0.5 * k1)
        k3 = h * f(t + 0.5 * h, y + 0.5 * k2)
        k4 = h * f(t + h, y + k3)

        y_new = y + (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
        t_new = t + h

        t_values.append(t_new)
        y_values.append(y_new)

    return t_values, y_values

# Example usage:
def example_ode(t, y):
    return -0.1 * y  # Example ODE: dy/dt = -0.1 * y

t0 = 0
t_end = 5
y0 = 1
h = 0.1

t_values, y_values = runge_kutta(example_ode, y0, t0, t_end, h)

# Print the results
for t, y in zip(t_values, y_values):
    print(f"t: {t}, y: {y}")
