from scipy.integrate import solve_ivp

# Example usage:
t_span = (t0, t_end)
y0 = [1]

result = solve_ivp(example_ode, t_span, y0, method='RK45', t_eval=t_values_rk4)

# Print the results
for t, y in zip(result.t, result.y[0]):
    print(f"t: {t}, y: {y}")
