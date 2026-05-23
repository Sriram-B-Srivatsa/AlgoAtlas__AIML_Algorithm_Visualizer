import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import base64
import io
import traceback

class CustomStandardScaler:
    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit_transform(self, X):
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)
        return (X - self.mean_) / self.scale_

    def inverse_transform(self, X):
        return X * self.scale_ + self.mean_

class CustomPCA:
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.mean_ = None

    def fit_transform(self, X):
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_
        cov_matrix = np.cov(X_centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        self.components_ = eigenvectors.T[:self.n_components]
        self.explained_variance_ = eigenvalues[:self.n_components]
        self.explained_variance_ratio_ = (eigenvalues[:self.n_components] / np.sum(eigenvalues))

        return np.dot(X_centered, eigenvectors[:, :self.n_components])

def render_projection_frame(X_orig, X_current, pc1, mean, alpha, is_final=False):
    """
    Renders a single frame showing points sliding onto the Principal Component line.
    """
    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)

    # Plot the original points in light blue
    ax.scatter(X_orig[:, 0], X_orig[:, 1], c='#3b82f6', alpha=0.3, s=40, label='Original Data' if alpha == 0 else "")

    # Draw PC1 Line
    scale_factor = 10
    x_vals = np.array([mean[0] - pc1[0]*scale_factor, mean[0] + pc1[0]*scale_factor])
    y_vals = np.array([mean[1] - pc1[1]*scale_factor, mean[1] + pc1[1]*scale_factor])
    ax.plot(x_vals, y_vals, c='black', linewidth=2, label='Principal Component 1 (PC1)')

    # Draw dotted projection lines
    if alpha > 0:
        for i in range(len(X_orig)):
            ax.plot([X_orig[i, 0], X_current[i, 0]], [X_orig[i, 1], X_current[i, 1]],
                    c='gray', linestyle=':', alpha=0.5)

    # Plot current moving points
    color = '#ef4444' if is_final else '#8b5cf6'
    ax.scatter(X_current[:, 0], X_current[:, 1], c=color, alpha=0.9, edgecolor='white', s=50,
               label='Projected Data' if is_final else 'Projecting...')

    # Plot mean
    ax.scatter(mean[0], mean[1], c='#fbbf24', edgecolor='black', s=100, zorder=5, label='Mean Center')

    ax.set_xlim(-8, 8)
    ax.set_ylim(-8, 8)
    ax.set_xlabel('Feature X1', fontsize=12)
    ax.set_ylabel('Feature X2', fontsize=12)
    ax.set_title(f"Dimensionality Reduction Progress: {int(alpha*100)}%", fontsize=14)
    ax.grid(alpha=0.3, linestyle='--')
    ax.legend(loc='best')

    plt.tight_layout()
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight')
    plt.close(fig)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode('utf-8')

def run_pca(data):
    try:
        X = np.array(data['X'], dtype=float)

        if X.shape[1] != 2:
            raise ValueError("This PCA implementation only supports 2D data")

        # 1. Generate Correlation Heatmap
        df = pd.DataFrame(X, columns=['Feature 1', 'Feature 2'])
        corr_matrix = df.corr()

        fig, ax = plt.subplots(figsize=(6, 5), dpi=100)
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5, alpha=0.9, ax=ax, vmin=-1, vmax=1)
        ax.set_title('Feature Correlation Matrix', fontsize=14)

        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        corr_heatmap = base64.b64encode(buf.read()).decode('utf-8')

        # 2. Compute PCA Math
        original_mean = np.mean(X, axis=0)
        scaler = CustomStandardScaler()
        X_scaled = scaler.fit_transform(X)

        pca = CustomPCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)

        components = pca.components_
        explained_variance = pca.explained_variance_
        explained_variance_ratio = pca.explained_variance_ratio_
        mean = pca.mean_

        # 3. Calculate 1D Projections (Points landing on the PC1 line)
        X_reconstructed_1d_scaled = np.dot(X_pca[:, 0:1], components[0:1, :]) + mean
        X_reconstructed_1d = scaler.inverse_transform(X_reconstructed_1d_scaled)

        # 4. Build Animation Frames (Interpolating from original X to projected X)
        history =[]
        steps = 10
        for i in range(steps + 1):
            alpha = i / float(steps)
            # Linear interpolation formula
            X_current = X * (1 - alpha) + X_reconstructed_1d * alpha
            is_final = (i == steps)

            frame_img = render_projection_frame(X, X_current, components[0], original_mean, alpha, is_final)
            history.append({
                'step': i,
                'progress': int(alpha * 100),
                'image': frame_img
            })

        result = {
            'components': components.tolist(),
            'explained_variance': explained_variance.tolist(),
            'explained_variance_ratio': explained_variance_ratio.tolist(),
            'mean': mean.tolist(),
            'original_mean': original_mean.tolist(),
            'original': X.tolist(),
            'transformed': X_pca.tolist(),
            'reconstructed': X_reconstructed_1d.tolist(),
            'corr_heatmap': corr_heatmap,
            'history': history
        }

        return result

    except Exception as e:
        print(f"Error in run_pca: {str(e)}")
        print(traceback.format_exc())
        return {'error': str(e)}

def generate_pca_data(n_samples=50, noise=0.1, seed=42):
    np.random.seed(seed)
    mean =[0, 0]
    base_correlation = 0.8
    adjusted_correlation = max(0, min(1, base_correlation - noise))
    cov = [[1.0, adjusted_correlation],[adjusted_correlation, 1.0]]
    data = np.random.multivariate_normal(mean, cov, n_samples)
    return {'X': data.tolist()}
