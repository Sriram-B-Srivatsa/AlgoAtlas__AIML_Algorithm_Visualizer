import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
import traceback
from sklearn.preprocessing import StandardScaler

def render_ae_frame(X_real, X_recon, iteration, is_final=False):
    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=80)

    # Draw dotted lines connecting real to reconstructed
    for i in range(len(X_real)):
        ax.plot([X_real[i, 0], X_recon[i, 0]], [X_real[i, 1], X_recon[i, 1]],
                c='gray', linestyle=':', alpha=0.4)

    ax.scatter(X_real[:, 0], X_real[:, 1], c='#3b82f6', edgecolor='k', s=50, alpha=0.8, label='Original Data')
    ax.scatter(X_recon[:, 0], X_recon[:, 1], c='#ef4444', edgecolor='white', s=50, alpha=0.9, label='Reconstructed')

    ax.set_xlim(-8, 8)
    ax.set_ylim(-8, 8)
    ax.set_title(f"Autoencoder Training (Epoch {iteration})" if not is_final else "Final Reconstruction", fontsize=12)
    ax.grid(alpha=0.3, linestyle='--')
    ax.legend(loc='upper right', fontsize=8)

    plt.tight_layout()
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode('utf-8')

def run_autoencoder(data):
    try:
        points = data.get('points',[])
        epochs = int(data.get('parameters', {}).get('epochs', 150))
        lr = float(data.get('parameters', {}).get('learningRate', 0.05))

        X = np.array([[float(p['x']), float(p['y'])] for p in points])
        if len(X) < 3: return {'error': 'Need at least 3 points.'}

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        N = len(X)

        # Deep Autoencoder Architecture: 2 -> 8 -> 1 (Bottleneck) -> 8 -> 2
        np.random.seed(42)
        W1 = np.random.randn(2, 8) * 0.1; b1 = np.zeros(8)
        W2 = np.random.randn(8, 1) * 0.1; b2 = np.zeros(1)
        W3 = np.random.randn(1, 8) * 0.1; b3 = np.zeros(8)
        W4 = np.random.randn(8, 2) * 0.1; b4 = np.zeros(2)

        def relu(x): return np.maximum(0, x)
        def d_relu(x): return (x > 0).astype(float)

        history, loss_history = [],[]
        frames_to_render = np.unique(np.linspace(1, epochs, 15, dtype=int)).tolist()

        for epoch in range(1, epochs + 1):
            # Forward
            Z1 = np.dot(X_scaled, W1) + b1; A1 = np.tanh(Z1)
            Z2 = np.dot(A1, W2) + b2;       Latent = Z2  # Linear bottleneck
            Z3 = np.dot(Latent, W3) + b3;   A3 = np.tanh(Z3)
            X_hat_scaled = np.dot(A3, W4) + b4

            # Loss (MSE)
            error = X_hat_scaled - X_scaled
            loss = np.mean(error**2)
            loss_history.append({"epoch": epoch, "loss": float(loss)})

            # Backward
            d_out = 2 * error / N
            dW4 = np.dot(A3.T, d_out); db4 = np.sum(d_out, axis=0)
            dA3 = np.dot(d_out, W4.T); dZ3 = dA3 * (1 - A3**2)

            dW3 = np.dot(Latent.T, dZ3); db3 = np.sum(dZ3, axis=0)
            dLatent = np.dot(dZ3, W3.T); dZ2 = dLatent

            dW2 = np.dot(A1.T, dZ2); db2 = np.sum(dZ2, axis=0)
            dA1 = np.dot(dZ2, W2.T); dZ1 = dA1 * (1 - A1**2)

            dW1 = np.dot(X_scaled.T, dZ1); db1 = np.sum(dZ1, axis=0)

            # Update
            W1 -= lr * dW1; b1 -= lr * db1
            W2 -= lr * dW2; b2 -= lr * db2
            W3 -= lr * dW3; b3 -= lr * db3
            W4 -= lr * dW4; b4 -= lr * db4

            if epoch in frames_to_render or epoch == epochs:
                # Transform back to screen coordinates for rendering
                X_hat_real = scaler.inverse_transform(X_hat_scaled)
                history.append({
                    "epoch": epoch,
                    "image": render_ae_frame(X, X_hat_real, epoch, is_final=(epoch==epochs))
                })

        # Final Latent Data for 1D Projection Graph
        latent_data =[{"point_idx": i, "latent_val": float(Latent[i,0])} for i in range(N)]

        return {
            'epochs': epochs,
            'final_loss': float(loss_history[-1]['loss']),
            'history': history,
            'loss_history': loss_history,
            'latent_data': latent_data
        }
    except Exception as e: return {'error': str(e), 'traceback': traceback.format_exc()}

def generate_ae_sample_data(data):
    try:
        from sklearn.datasets import make_moons, make_circles, make_blobs
        dataset_type = data.get('dataset_type', 'moons')
        count = int(data.get('count', 60))
        variance = float(data.get('variance', 0.1))
        n_clusters = int(data.get('n_clusters', 3))

        np.random.seed(42)
        if dataset_type == 'moons':
            X, _ = make_moons(n_samples=count, noise=variance*0.5, random_state=42)
            X = X * 4.0 - 1.0
        elif dataset_type == 'circles':
            X, _ = make_circles(n_samples=count, noise=variance*0.5, factor=0.5, random_state=42)
            X = X * 5.0
        else: # blobs
            # Equidistant perfect circle
            centers = [[4.5 * np.cos(i * 2 * np.pi / n_clusters), 4.5 * np.sin(i * 2 * np.pi / n_clusters)] for i in range(n_clusters)]
            X, _ = make_blobs(n_samples=count, centers=centers, cluster_std=variance*1.5, random_state=42)

        return {'points': [{'x': float(X[i,0]), 'y': float(X[i,1])} for i in range(count)]}
    except Exception as e: return {'error': str(e)}
