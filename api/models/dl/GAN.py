import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
import traceback
from sklearn.preprocessing import StandardScaler

def render_gan_frame(X_real, X_fake, epoch, is_final=False):
    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=80)

    ax.scatter(X_real[:, 0], X_real[:, 1], c='#22c55e', edgecolor='k', s=50, alpha=0.8, label='Real Data')
    ax.scatter(X_fake[:, 0], X_fake[:, 1], c='#a855f7', edgecolor='white', s=50, alpha=0.9, label='Fake (Generated)')

    ax.set_xlim(-8, 8)
    ax.set_ylim(-8, 8)
    ax.set_title(f"GAN Training (Epoch {epoch})" if not is_final else "Final Generator Output", fontsize=12)
    ax.grid(alpha=0.3, linestyle='--')
    ax.legend(loc='upper right', fontsize=8)

    plt.tight_layout()
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode('utf-8')

def render_discriminator_heatmap(X_real, X_fake, D_W1, D_b1, D_W2, D_b2, scaler):
    """Draws a heatmap of where the Discriminator thinks 'Real' data lives"""
    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=80)

    h = 0.1
    xx, yy = np.meshgrid(np.arange(-8, 8.1, h), np.arange(-8, 8.1, h))
    grid = np.c_[xx.ravel(), yy.ravel()]
    grid_scaled = scaler.transform(grid)

    # Forward pass through Discriminator
    z1 = np.dot(grid_scaled, D_W1) + D_b1; a1 = np.tanh(z1)
    z2 = np.dot(a1, D_W2) + D_b2; prob = 1 / (1 + np.exp(-z2))

    Z = prob.reshape(xx.shape)

    # Contour plot (Green = Real, Purple = Fake)
    contour = ax.contourf(xx, yy, Z, 25, cmap='PRGn', alpha=0.5)

    ax.scatter(X_real[:, 0], X_real[:, 1], c='#22c55e', edgecolor='k', s=20, label='Real')
    ax.scatter(X_fake[:, 0], X_fake[:, 1], c='#a855f7', edgecolor='k', s=20, label='Fake')

    ax.set_xlim(-8, 8)
    ax.set_ylim(-8, 8)
    ax.set_title("Discriminator's Probability Map", fontsize=12)
    ax.legend(loc='best', fontsize=8)

    plt.tight_layout()
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode('utf-8')

def adam_update(param, grad, m, v, t, lr, beta1=0.9, beta2=0.999, eps=1e-8):
    """Pure Numpy Adam Optimizer for stable Deep Learning"""
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * (grad**2)
    m_hat = m / (1 - beta1**t)
    v_hat = v / (1 - beta2**t)
    param -= lr * m_hat / (np.sqrt(v_hat) + eps)
    return param, m, v

def run_gan(data):
    try:
        from sklearn.preprocessing import StandardScaler
        points = data.get('points',[])
        epochs = int(data.get('parameters', {}).get('epochs', 200))
        lr = float(data.get('parameters', {}).get('learningRate', 0.05))

        X = np.array([[float(p['x']), float(p['y'])] for p in points])
        if len(X) < 5: return {'error': 'Need at least 5 points for GAN.'}

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        N = len(X)

        # Generator: Noise(2D) -> 32 -> 2D
        np.random.seed(42)
        G_W1 = np.random.randn(2, 32) * np.sqrt(2./2)
        G_b1 = np.zeros(32)
        G_W2 = np.random.randn(32, 2) * np.sqrt(2./32)
        G_b2 = np.zeros(2)

        # Discriminator: 2D -> 32 -> 1D (Prob)
        D_W1 = np.random.randn(2, 32) * np.sqrt(2./2)
        D_b1 = np.zeros(32)
        D_W2 = np.random.randn(32, 1) * np.sqrt(2./32)
        D_b2 = np.zeros(1)

        # Adam Optimizer Memory
        m_GW1, v_GW1 = np.zeros_like(G_W1), np.zeros_like(G_W1)
        m_Gb1, v_Gb1 = np.zeros_like(G_b1), np.zeros_like(G_b1)
        m_GW2, v_GW2 = np.zeros_like(G_W2), np.zeros_like(G_W2)
        m_Gb2, v_Gb2 = np.zeros_like(G_b2), np.zeros_like(G_b2)

        m_DW1, v_DW1 = np.zeros_like(D_W1), np.zeros_like(D_W1)
        m_Db1, v_Db1 = np.zeros_like(D_b1), np.zeros_like(D_b1)
        m_DW2, v_DW2 = np.zeros_like(D_W2), np.zeros_like(D_W2)
        m_Db2, v_Db2 = np.zeros_like(D_b2), np.zeros_like(D_b2)

        def sigmoid(x): return 1 / (1 + np.exp(-np.clip(x, -250, 250)))
        def relu(x): return np.maximum(0, x)
        def d_relu(x): return (x > 0).astype(float)

        history, loss_history = [],[]
        frames_to_render = np.unique(np.linspace(1, epochs, 15, dtype=int)).tolist()

        t = 0
        for epoch in range(1, epochs + 1):
            t += 1

            # --- TRAIN DISCRIMINATOR ---
            noise = np.random.randn(N, 2)
            G_z1 = np.dot(noise, G_W1) + G_b1; G_a1 = relu(G_z1)
            X_fake = np.dot(G_a1, G_W2) + G_b2

            D_z1_r = np.dot(X_scaled, D_W1) + D_b1; D_a1_r = relu(D_z1_r)
            D_z2_r = np.dot(D_a1_r, D_W2) + D_b2; D_prob_r = sigmoid(D_z2_r)

            D_z1_f = np.dot(X_fake, D_W1) + D_b1; D_a1_f = relu(D_z1_f)
            D_z2_f = np.dot(D_a1_f, D_W2) + D_b2; D_prob_f = sigmoid(D_z2_f)

            D_loss = -np.mean(np.log(D_prob_r + 1e-8) + np.log(1 - D_prob_f + 1e-8))

            d_out_r = (D_prob_r - 1) / N
            d_out_f = (D_prob_f) / N

            dD_W2 = np.dot(D_a1_r.T, d_out_r) + np.dot(D_a1_f.T, d_out_f)
            dD_b2 = np.sum(d_out_r, axis=0) + np.sum(d_out_f, axis=0)

            dD_a1_r = np.dot(d_out_r, D_W2.T); dD_z1_r = dD_a1_r * d_relu(D_z1_r)
            dD_a1_f = np.dot(d_out_f, D_W2.T); dD_z1_f = dD_a1_f * d_relu(D_z1_f)

            dD_W1 = np.dot(X_scaled.T, dD_z1_r) + np.dot(X_fake.T, dD_z1_f)
            dD_b1 = np.sum(dD_z1_r, axis=0) + np.sum(dD_z1_f, axis=0)

            D_W1, m_DW1, v_DW1 = adam_update(D_W1, dD_W1, m_DW1, v_DW1, t, lr)
            D_b1, m_Db1, v_Db1 = adam_update(D_b1, dD_b1, m_Db1, v_Db1, t, lr)
            D_W2, m_DW2, v_DW2 = adam_update(D_W2, dD_W2, m_DW2, v_DW2, t, lr)
            D_b2, m_Db2, v_Db2 = adam_update(D_b2, dD_b2, m_Db2, v_Db2, t, lr)

            # --- TRAIN GENERATOR ---
            noise = np.random.randn(N, 2)
            G_z1 = np.dot(noise, G_W1) + G_b1; G_a1 = relu(G_z1)
            X_fake = np.dot(G_a1, G_W2) + G_b2

            D_z1_f = np.dot(X_fake, D_W1) + D_b1; D_a1_f = relu(D_z1_f)
            D_z2_f = np.dot(D_a1_f, D_W2) + D_b2; D_prob_f = sigmoid(D_z2_f)

            G_loss = -np.mean(np.log(D_prob_f + 1e-8))

            d_out_g = (D_prob_f - 1) / N
            dG_a1_f = np.dot(d_out_g, D_W2.T); dG_z1_f = dG_a1_f * d_relu(D_z1_f)
            dX_fake = np.dot(dG_z1_f, D_W1.T)

            dG_W2 = np.dot(G_a1.T, dX_fake); dG_b2 = np.sum(dX_fake, axis=0)
            dG_a1 = np.dot(dX_fake, G_W2.T); dG_z1 = dG_a1 * d_relu(G_z1)
            dG_W1 = np.dot(noise.T, dG_z1); dG_b1 = np.sum(dG_z1, axis=0)

            G_W1, m_GW1, v_GW1 = adam_update(G_W1, dG_W1, m_GW1, v_GW1, t, lr)
            G_b1, m_Gb1, v_Gb1 = adam_update(G_b1, dG_b1, m_Gb1, v_Gb1, t, lr)
            G_W2, m_GW2, v_GW2 = adam_update(G_W2, dG_W2, m_GW2, v_GW2, t, lr)
            G_b2, m_Gb2, v_Gb2 = adam_update(G_b2, dG_b2, m_Gb2, v_Gb2, t, lr)

            loss_history.append({"epoch": epoch, "d_loss": float(D_loss), "g_loss": float(G_loss)})

            if epoch in frames_to_render or epoch == epochs:
                X_fake_real = scaler.inverse_transform(X_fake)
                history.append({
                    "epoch": epoch,
                    "image": render_gan_frame(X, X_fake_real, epoch, is_final=(epoch==epochs))
                })

        disc_map = render_discriminator_heatmap(X, scaler.inverse_transform(X_fake), D_W1, D_b1, D_W2, D_b2, scaler)

        return {
            'epochs': epochs,
            'final_d_loss': float(loss_history[-1]['d_loss']),
            'final_g_loss': float(loss_history[-1]['g_loss']),
            'history': history,
            'loss_history': loss_history,
            'disc_map': disc_map
        }
    except Exception as e: return {'error': str(e), 'traceback': traceback.format_exc()}

def generate_gan_sample_data(data):
    try:
        from sklearn.datasets import make_moons, make_circles, make_blobs
        dataset_type = data.get('dataset_type', 'moons')
        count = int(data.get('count', 100))
        variance = float(data.get('variance', 0.1))
        n_clusters = int(data.get('n_clusters', 3))

        np.random.seed(42)
        if dataset_type == 'moons':
            # Less noisy, better scaled
            X, _ = make_moons(n_samples=count, noise=variance*0.5, random_state=42)
            X = X * 4.0 - 1.0
        elif dataset_type == 'circles':
            X, _ = make_circles(n_samples=count, noise=variance*0.5, factor=0.5, random_state=42)
            X = X * 5.0
        else: # blobs
            # Force blobs into an equidistant ring so they never touch!
            centers = [[4.5 * np.cos(i * 2 * np.pi / n_clusters), 4.5 * np.sin(i * 2 * np.pi / n_clusters)] for i in range(n_clusters)]
            X, _ = make_blobs(n_samples=count, centers=centers, cluster_std=variance*1.5, random_state=42)

        return {'points':[{'x': float(X[i,0]), 'y': float(X[i,1])} for i in range(count)]}
    except Exception as e: return {'error': str(e)}
