import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use Agg backend (non-interactive, doesn't require GUI)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
import io
import base64
import traceback

def gradient_descent(X, y, learning_rate=0.01, n_iterations=100, tolerance=1e-6, polynomial_degree=1,
                   record_steps=False, step_interval=10):
    """
    Implements gradient descent for polynomial regression with safeguards for high learning rates
    """
    # Add a column of ones to X for the intercept term
    X_b = np.c_[np.ones((X.shape[0], 1)), X]

    # Modified learning rate scaling - better balance for higher degrees
    if polynomial_degree <= 3:
        scaled_learning_rate = learning_rate
    elif polynomial_degree <= 5:
        scaled_learning_rate = learning_rate / (1 + 0.2 * (polynomial_degree - 3))
    else:
        scaled_learning_rate = learning_rate / (1 + 0.4 * (polynomial_degree - 5))

    print(f"Original learning rate: {learning_rate}, Scaled learning rate: {scaled_learning_rate}")

    # Initialize parameters
    theta = np.random.randn(X_b.shape[1]) * 0.01

    cost_history = []
    theta_history =[] if record_steps else None

    # Feature scaling for better convergence
    if X_b.shape[1] > 1:
        feature_means = np.mean(X_b[:, 1:], axis=0)
        feature_stds = np.std(X_b[:, 1:], axis=0)
        feature_stds = np.where(feature_stds == 0, 1, feature_stds)

        X_b_scaled = X_b.copy()
        X_b_scaled[:, 1:] = (X_b[:, 1:] - feature_means) / feature_stds

        y_mean = np.mean(y)
        y_std = np.std(y) if np.std(y) > 0 else 1
        y_scaled = (y - y_mean) / y_std
    else:
        X_b_scaled = X_b
        feature_means = np.array([])
        feature_stds = np.array([])
        y_scaled = y
        y_mean = 0
        y_std = 1

    actual_iterations = 0
    prev_cost = float('inf')

    for i in range(n_iterations):
        actual_iterations += 1
        y_pred = X_b_scaled.dot(theta)
        error = y_pred - y_scaled
        cost = np.mean(error**2)

        # Check for NaN or infinite values
        if np.isnan(cost) or np.isinf(cost) or cost > 1e10:
            print(f"Warning: Divergence detected at iteration {i}. Learning rate too high.")
            if i > 5 and len(cost_history) > 5:
                break
            else:
                return gradient_descent(X, y, learning_rate=learning_rate*0.1, n_iterations=n_iterations,
                                      tolerance=tolerance, polynomial_degree=polynomial_degree,
                                      record_steps=record_steps, step_interval=step_interval)

        # Check for divergence
        if i > 0 and cost > prev_cost * 1.5 and i > 5:
            print(f"Warning: Cost increasing at iteration {i}. Reducing learning rate.")
            return gradient_descent(X, y, learning_rate=scaled_learning_rate*0.1, n_iterations=n_iterations,
                                  tolerance=tolerance, polynomial_degree=polynomial_degree,
                                  record_steps=record_steps, step_interval=step_interval)

        cost_history.append(float(cost))
        prev_cost = cost

        if record_steps and i % step_interval == 0:
            if X_b.shape[1] > 1:
                intercept = theta[0] * y_std + y_mean
                coefficients = theta[1:] / feature_stds * y_std
                intercept = intercept - np.sum(coefficients * feature_means)
                orig_theta = np.concatenate([[intercept], coefficients])
            else:
                orig_theta = theta.copy() * y_std + y_mean

            theta_history.append(orig_theta)

        gradients = 2/X_b_scaled.shape[0] * X_b_scaled.T.dot(error)

        if polynomial_degree > 3:
            max_grad = 5.0 / polynomial_degree
            gradients = np.clip(gradients, -max_grad, max_grad)

        theta = theta - scaled_learning_rate * gradients

        if i > 0 and np.abs(cost_history[i] - cost_history[i-1]) < tolerance:
            print(f"Converged after {i+1} iterations")
            if record_steps and (i % step_interval != 0):
                if X_b.shape[1] > 1:
                    intercept = theta[0] * y_std + y_mean
                    coefficients = theta[1:] / feature_stds * y_std
                    intercept = intercept - np.sum(coefficients * feature_means)
                    orig_theta = np.concatenate([[intercept], coefficients])
                else:
                    orig_theta = theta.copy() * y_std + y_mean
                theta_history.append(orig_theta)
            break

    if X_b.shape[1] > 1:
        coefficients = theta[1:] / feature_stds * y_std
        intercept = theta[0] * y_std + y_mean - np.sum(coefficients * feature_means)
    else:
        coefficients = theta[1:] * y_std
        intercept = theta[0] * y_std + y_mean

    return coefficients, intercept, cost_history, actual_iterations, theta_history


def create_regression_plot(X, y, coefficients, intercept, degree):
    """
    Creates a static high-res visualization of the polynomial regression fit
    """
    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)

    # Plot data points
    ax.scatter(X, y, color='#3b82f6', alpha=0.8, edgecolor='k', label='Data Points')

    # Generate smooth curve for plotting
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_range = np.linspace(min(X), max(X), 100).reshape(-1, 1)
    X_poly = poly.fit_transform(X_range)
    y_pred = intercept + np.dot(X_poly, coefficients)

    # Plot regression curve
    ax.plot(X_range, y_pred, color='#ef4444', linewidth=3, label=f'Polynomial (Degree {degree})')

    ax.set_xlabel('Feature X', fontsize=12)
    ax.set_ylabel('Target Y', fontsize=12)
    ax.set_title(f'Final Regression Curve', fontsize=14)
    ax.legend(loc='best')
    ax.grid(alpha=0.3, linestyle='--')

    plt.tight_layout()
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight')
    plt.close(fig)
    buffer.seek(0)

    return base64.b64encode(buffer.read()).decode('utf-8')


def run_polynomial_regression(data, degree=1, alpha=0.01, iterations=100, random_state=42, record_steps=False, step_interval=10):
    """
    Runs polynomial regression of specified degree on the provided data
    """
    try:
        np.random.seed(random_state)

        try:
            X = np.array(data['X'], dtype=float).reshape(-1, 1)
            y = np.array(data['y'], dtype=float)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Error converting data to numeric format: {str(e)}.")

        if np.isnan(X).any() or np.isnan(y).any():
            raise ValueError("Data contains NaN values. Please check your input.")

        if len(X) != len(y) or len(X) < 2:
            raise ValueError("Invalid data lengths.")

        if np.all(X == X[0]):
            X = X + np.random.normal(0, 0.01, X.shape)

        poly = PolynomialFeatures(degree=degree, include_bias=False)
        X_poly = poly.fit_transform(X)

        record_steps = True
        effective_step_interval = max(1, iterations // 100) if iterations > 100 else 1

        coefficients, intercept, cost_history, actual_iterations, theta_history = gradient_descent(
            X_poly, y, learning_rate=alpha, n_iterations=iterations + 1, polynomial_degree=degree,
            record_steps=record_steps, step_interval=effective_step_interval
        )

        X_poly_pred = poly.transform(X)
        y_pred = intercept + np.dot(X_poly_pred, coefficients)
        y_pred = np.array(y_pred).flatten()

        mse = float(mean_squared_error(y, y_pred))
        r2 = float(r2_score(y, y_pred))

        iteration_history =[]
        if theta_history:
            for i, theta in enumerate(theta_history):
                iter_intercept = theta[0]
                iter_coeffs = theta[1:]
                iter_step = i * effective_step_interval
                iter_cost = cost_history[iter_step] if iter_step < len(cost_history) else cost_history[-1]

                iteration_history.append({
                    "iteration": iter_step,
                    "coefficients": iter_coeffs.tolist(),
                    "intercept": float(iter_intercept),
                    "cost": float(iter_cost),
                    "degree": int(degree)
                })

        # Structure Cost History for Recharts React Component
        cost_history_recharts =[{"iteration": i, "cost": cost} for i, cost in enumerate(cost_history)]

        # Generate Final Plot
        final_plot = create_regression_plot(X, y, coefficients, intercept, degree)

        results = {
            "coefficients": coefficients.tolist(),
            "intercept": float(intercept),
            "mse": mse,
            "r2": r2,
            "cost_history_data": cost_history_recharts,
            "degree": int(degree),
            "alpha": float(alpha),
            "iterations": int(actual_iterations),
            "iteration_history": iteration_history,
            "final_plot": final_plot
        }

        return results

    except Exception as e:
        error_msg = f"{str(e)}"
        if "learning rate" in str(e).lower() or "diverg" in str(e).lower():
            error_msg = f"Learning rate too high: {str(e)}"
        raise ValueError(f"Error in polynomial regression: {error_msg}")
