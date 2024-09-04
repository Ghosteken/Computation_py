def runge_kutta_second_order(f, y0, t0, t_end, h):
    t_values = [t0]
    y_values = [y0]

    while t_values[-1] < t_end:
        t = t_values[-1]
        y = y_values[-1]

        k1 = h * f(t, y)
        k2 = h * f(t + h, y + k1)

        y_new = y + 0.5 * (k1 + k2)
        t_new = t + h

        t_values.append(t_new)
        y_values.append(y_new)

    return t_values, y_values

# Example ODE function
def example_ode(t, y):
    return -0.1 * y

# Example usage:
t0 = 0
t_end = 5
y0 = 1
h = 0.1

t_values_rk2, y_values_rk2 = runge_kutta_second_order(example_ode, y0, t0, t_end, h)

# Print the results
for t, y in zip(t_values_rk2, y_values_rk2):
    print(f"t: {t}, y: {y}")
