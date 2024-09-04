import numpy as np

def f(x, y):
    return 3 * y * np.exp(3 * x)

def predictor_corrector(f, x0, y0, h, tspan):
    """
    Solves an initial value problem using the predictor-corrector pair for the two-step Adams method.

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

    # Predictor step
    y_pred = y[0] + h * f(x[0], y[0]) + h**2 / 2 * f(x[0] + h, y[0])

    for i in range(1, len(x)):
        # Corrector step
        y[i] = y[i-1] + h * f(x[i], y[i]) + h**2 / 2 * (f(x[i], y[i]) - f(x[i-1], y[i-1]))

        # Predictor step for the next iteration
        y_pred = y[i] + h * f(x[i], y[i]) + h**2 / 2 * f(x[i] + h, y[i])

    return x, y

def theoretical_solution(x):
    return np.exp(3 * x)

# Define parameters
x0 = 0.0
y0 = 1.0
h = 0.1
tspan = [0.0, 1.0]

# Solve the IVP using the predictor-corrector method
x, y = predictor_corrector(f, x0, y0, h, tspan)

# Calculate the theoretical solution
y_theo = theoretical_solution(x)

# Calculate the absolute error
error = np.abs(y - y_theo)

# Print the results
print("x | y_PC | y_theo | Error")
print("--------------------------")
for i in range(len(x)):
    print(f"{x[i]:3.2f} | {y[i]:5.2f} | {y_theo[i]:5.2f} | {error[i]:5.4e}")
