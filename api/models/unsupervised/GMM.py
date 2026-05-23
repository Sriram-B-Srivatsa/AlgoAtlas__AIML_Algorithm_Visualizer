import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import io
import base64
import traceback

def render_gmm_frame(model, X, iteration, is_final=False):
    """Renders a single frame of the GMM probability contours"""
    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=80)

    # Create meshgrid for contours
    x_min, x_max = -8, 8
    y_min, y_max = -8, 8
    X_grid, Y_grid = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    XX = np.array([X_grid.ravel(), Y_grid.ravel()]).T

    # Evaluate the probabilities on the grid
    Z = -model.score_samples(XX)
    Z = Z.reshape(X_grid.shape)

    # Plot contours (the ellipses)
    contour = ax.contourf(X_grid, Y_grid, Z, levels=np.linspace(Z.min(), np.percentile(Z, 95), 15),
                          cmap='viridis_r', alpha=0.6)

    # Plot data points
    ax.scatter(X[:, 0], X[:, 1], c='#ef4444', s=20, edgecolor='white', linewidth=0.5, alpha=0.8)

    # Plot the means (centers of the Gaussians)
    ax.scatter(model.means_[:, 0], model.means_[:, 1], marker='X', s=150, c='#fbbf24', edgecolor='black', zorder=10)

    ax.set_xlim(-8, 8)
    ax.set_ylim(-8, 8)
    ax.set_title(f"EM Algorithm Progress (Iteration {iteration})" if not is_final else "Final Converged Gaussians", fontsize=12)
    ax.grid(alpha=0.3, linestyle='--')

    plt.tight_layout()
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode('utf-8')

def run_gmm(data):
    try:
        X = np.array(data['X'], dtype=float)
        n_components = int(data.get('components', 3))
        max_iter = int(data.get('max_iterations', 100))

        if len(X) < n_components:
            return {'error': 'Need more data points than components.'}

        # 1. BUILD AIC/BIC CURVE (Test K from 1 to 8)
        aic_bic_history =[]
        max_k_test = min(8, len(X))
        for k in range(1, max_k_test + 1):
            temp_gmm = GaussianMixture(n_components=k, random_state=42, reg_covar=1e-5)
            temp_gmm.fit(X)
            aic_bic_history.append({
                "k": k,
                "AIC": float(temp_gmm.aic(X)),
                "BIC": float(temp_gmm.bic(X))
            })

        # 2. TRAIN MAIN MODEL ITERATIVELY FOR ANIMATION
        history =[]
        log_likelihood_history =[]

        # We use warm_start=True and train 1 iteration at a time to capture the EM process
        gmm = GaussianMixture(n_components=n_components, max_iter=1, warm_start=True, random_state=42, reg_covar=1e-5)

        actual_iterations = 0
        for i in range(1, max_iter + 1):
            gmm.fit(X)
            actual_iterations = i

            # Save Log-Likelihood for Recharts
            ll = gmm.lower_bound_
            log_likelihood_history.append({
                "iteration": i,
                "log_likelihood": float(ll)
            })

            # Save Image Frame (Save every step if < 20, else skip frames to save bandwidth)
            if i <= 20 or i % 5 == 0 or gmm.converged_:
                frame = render_gmm_frame(gmm, X, i, is_final=gmm.converged_)
                history.append({
                    "iteration": i,
                    "image": frame,
                    "log_likelihood": float(ll)
                })

            if gmm.converged_:
                break

        # If it didn't converge, render the last frame as final
        if not gmm.converged_:
            frame = render_gmm_frame(gmm, X, actual_iterations, is_final=True)
            history.append({
                "iteration": actual_iterations,
                "image": frame,
                "log_likelihood": float(gmm.lower_bound_)
            })

        return {
            'n_components': n_components,
            'iterations': actual_iterations,
            'converged': bool(gmm.converged_),
            'final_aic': float(gmm.aic(X)),
            'final_bic': float(gmm.bic(X)),
            'final_log_likelihood': float(gmm.lower_bound_),
            'history': history,
            'log_likelihood_history': log_likelihood_history,
            'aic_bic_history': aic_bic_history
        }

    except Exception as e:
        print(f"Error in GMM: {str(e)}")
        return {'error': str(e), 'traceback': traceback.format_exc()}

def generate_gmm_sample_data(data):
    """Generates sample data suited for GMMs"""
    try:
        from sklearn.datasets import make_blobs, make_moons

        count = int(data.get('count', 100))
        dataset_type = data.get('dataset_type', 'anisotropic')
        n_clusters = int(data.get('n_clusters', 3))
        variance = float(data.get('variance', 0.5))

        np.random.seed(42)

        if dataset_type == 'moons':
            X, _ = make_moons(n_samples=count, noise=variance*0.1, random_state=42)
            X = X * 4.5 - 2
        elif dataset_type == 'anisotropic':
            # Stretched/Elongated blobs that K-Means fails at but GMM excels at
            X, _ = make_blobs(n_samples=count, centers=n_clusters, random_state=42)
            transformation = [[0.6, -0.6], [-0.4, 0.8]]
            X = np.dot(X, transformation)
            X = X * 2 * (1 + variance)
        else:  # standard blobs
            X, _ = make_blobs(n_samples=count, centers=n_clusters, cluster_std=variance*1.5, random_state=42)
            # Clamp to prevent out of bounds
            max_val = np.max(np.abs(X))
            if max_val > 7.5:
                X = (X / max_val) * 7.5

        points = [{'x': float(p[0]), 'y': float(p[1])} for p in X]
        return {'points': points}

    except Exception as e:
        return {'error': str(e)}
