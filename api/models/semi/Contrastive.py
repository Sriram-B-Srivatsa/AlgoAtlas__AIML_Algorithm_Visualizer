import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
import traceback
from sklearn.preprocessing import StandardScaler

def render_latent_frame(Z, y_true, iteration, is_final=False):
    """Renders the evolving Latent/Embedding space"""
    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=80)

    # We color the dots using their true labels ONLY so the human can see
    # that the AI grouped them correctly. The AI did NOT use these labels!
    unique_classes = np.unique(y_true)
    colors =['#3B82F6', '#EF4444', '#22C55E', '#F59E0B', '#8B5CF6']

    for idx, cls in enumerate(unique_classes):
        mask = (y_true == cls)
        c = colors[idx % len(colors)]
        ax.scatter(Z[mask, 0], Z[mask, 1], c=c, edgecolor='white', s=50, alpha=0.8, linewidth=0.5)

    ax.set_title(f"Latent Space (Epoch {iteration})" if not is_final else "Final Separated Latent Space", fontsize=12)
    ax.grid(alpha=0.3, linestyle='--')

    # Dynamically scale axes to fit the exploding clusters
    max_val = np.max(np.abs(Z)) + 0.5
    ax.set_xlim(-max_val, max_val)
    ax.set_ylim(-max_val, max_val)
    ax.set_xlabel("Latent Dimension 1")
    ax.set_ylabel("Latent Dimension 2")

    plt.tight_layout()
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode('utf-8')

def render_original_space(X, y_true):
    """Renders the original X/Y inputs for comparison"""
    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=80)
    unique_classes = np.unique(y_true)
    colors =['#3B82F6', '#EF4444', '#22C55E', '#F59E0B', '#8B5CF6']

    for idx, cls in enumerate(unique_classes):
        mask = (y_true == cls)
        c = colors[idx % len(colors)]
        ax.scatter(X[mask, 0], X[mask, 1], c=c, edgecolor='k', s=40, alpha=0.8)

    ax.set_title("Original 2D Input Space", fontsize=12)
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_xlim(-8, 8)
    ax.set_ylim(-8, 8)

    plt.tight_layout()
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode('utf-8')

def run_contrastive(data):
    try:
        points = data.get('points',[])
        params = data.get('parameters', {})
        epochs = int(params.get('epochs', 100))
        learning_rate = float(params.get('learningRate', 0.05))
        margin = float(params.get('margin', 2.0))

        X = np.array([[float(p['x']), float(p['y'])] for p in points])
        y_true = np.array([int(p['class']) for p in points])

        if len(X) < 5:
            return {'error': 'Need at least 5 points to perform Contrastive Learning.'}

        # Scale X so distance calculations make sense
        X = StandardScaler().fit_transform(X)
        N = len(X)

        # Determine Positives: For every point, find its closest neighbor in original space
        positives = np.zeros(N, dtype=int)
        for i in range(N):
            dists = np.sum((X - X[i])**2, axis=1)
            dists[i] = np.inf # Ignore self
            positives[i] = np.argmin(dists)

        # Initialize Latent Space (Z) as a random, tightly packed cloud at the origin
        np.random.seed(42)
        Z = np.random.randn(N, 2) * 0.1

        history =[]
        loss_history =[]

        # Save exact frames to ensure fast API payload
        render_steps = np.unique(np.linspace(0, epochs-1, 15, dtype=int)).tolist()

        # Gradient Descent Loop (Spring Embedder / Margin Loss)
        for epoch in range(epochs):
            total_loss = 0
            dZ = np.zeros_like(Z)

            for i in range(N):
                # 1. ATTRACT POSITIVE
                pos_idx = positives[i]
                diff_pos = Z[i] - Z[pos_idx]
                dist_pos = np.linalg.norm(diff_pos) + 1e-8

                # Pull them together
                loss_pos = dist_pos**2
                total_loss += loss_pos
                dZ[i] += 2 * diff_pos
                dZ[pos_idx] -= 2 * diff_pos

                # 2. REPEL RANDOM NEGATIVES
                # Pick 3 random points that are NOT the positive match
                neg_indices = np.random.choice([j for j in range(N) if j != i and j != pos_idx], size=3, replace=False)
                for neg_idx in neg_indices:
                    diff_neg = Z[i] - Z[neg_idx]
                    dist_neg = np.linalg.norm(diff_neg) + 1e-8

                    # If they are closer than the margin, push them apart!
                    if dist_neg < margin:
                        loss_neg = (margin - dist_neg)**2
                        total_loss += loss_neg

                        # Gradient pushes them apart
                        push_vector = -2 * (margin - dist_neg) * (diff_neg / dist_neg)
                        dZ[i] += push_vector
                        dZ[neg_idx] -= push_vector

            # Apply Gradients
            Z -= learning_rate * (dZ / N)

            avg_loss = total_loss / N
            loss_history.append({"epoch": epoch + 1, "loss": float(avg_loss)})

            if epoch in render_steps:
                history.append({
                    "epoch": epoch + 1,
                    "image": render_latent_frame(Z, y_true, epoch + 1, is_final=(epoch == epochs-1))
                })

        # Add the original space image for comparison
        original_plot = render_original_space(np.array([[float(p['x']), float(p['y'])] for p in points]), y_true)

        return {
            'epochs': epochs,
            'margin': margin,
            'learning_rate': learning_rate,
            'final_loss': float(loss_history[-1]['loss']),
            'history': history,
            'loss_history': loss_history,
            'original_plot': original_plot
        }

    except Exception as e:
        return {'error': str(e), 'traceback': traceback.format_exc()}


def generate_cl_sample_data(data):
    """Generates grouped clusters for Contrastive Learning"""
    try:
        from sklearn.datasets import make_moons, make_blobs, make_circles

        dataset_type = data.get('dataset_type', 'blobs')
        count = int(data.get('count', 100))
        variance = float(data.get('variance', 0.5))
        n_clusters = int(data.get('n_clusters', 3))

        np.random.seed(42)
        if dataset_type == 'moons':
            X, y = make_moons(n_samples=count, noise=variance*0.2, random_state=42)
            X = X * 4.5 - 2
        elif dataset_type == 'circles':
            X, y = make_circles(n_samples=count, noise=variance*0.2, factor=0.5, random_state=42)
            X = X * 6.5
        else: # blobs
            centers =[]
            for i in range(n_clusters):
                angle = i * (2 * np.pi / n_clusters)
                centers.append([4 * np.cos(angle), 4 * np.sin(angle)])
            X, y = make_blobs(n_samples=count, centers=centers, cluster_std=variance*1.5, random_state=42)

        points = [{'x': float(X[i,0]), 'y': float(X[i,1]), 'class': int(y[i])} for i in range(count)]
        return {'points': points}

    except Exception as e:
        import traceback
        return {'error': str(e), 'traceback': traceback.format_exc()}
