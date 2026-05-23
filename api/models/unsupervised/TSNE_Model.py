import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import io
import base64
import traceback

def render_3d_original(X, y):
    fig = plt.figure(figsize=(6, 4.5), dpi=80)
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap='viridis', s=20, alpha=0.8)
    ax.set_title("Original 3D Data Space", fontsize=12)
    plt.tight_layout()
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode('utf-8')

def render_tsne_frame(Z, y, p, global_min, global_max):
    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=80)
    ax.scatter(Z[:, 0], Z[:, 1], c=y, cmap='viridis', s=20, alpha=0.8)
    ax.set_title(f"t-SNE 2D Projection (Perplexity = {p})", fontsize=12)

    # FIXED: Lock the axes and add grid!
    ax.set_xlim(global_min, global_max)
    ax.set_ylim(global_min, global_max)
    ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode('utf-8')

def run_tsne(data):
    try:
        from sklearn.datasets import make_swiss_roll, make_blobs, make_s_curve
        dataset_type = data.get('parameters', {}).get('dataset_type', 'swiss')
        n_samples = int(data.get('parameters', {}).get('count', 300))
        target_p = int(data.get('parameters', {}).get('perplexity', 30))

        np.random.seed(42)
        if dataset_type == 'swiss':
            X, y_raw = make_swiss_roll(n_samples=n_samples, noise=0.1, random_state=42)
            y = np.digitize(y_raw, bins=np.linspace(y_raw.min(), y_raw.max(), 5))
        elif dataset_type == 'scurve':
            # ADDED: S-Curve Dataset!
            X, y_raw = make_s_curve(n_samples=n_samples, noise=0.1, random_state=42)
            y = np.digitize(y_raw, bins=np.linspace(y_raw.min(), y_raw.max(), 5))
        else:
            X, y = make_blobs(n_samples=n_samples, n_features=3, centers=4, random_state=42)

        original_img = render_3d_original(X, y)

        history, loss_history, all_embeddings = [], [],[]

        perplexities = np.unique(np.linspace(5, target_p, 8, dtype=int)).tolist()
        if target_p not in perplexities: perplexities.append(target_p)

        final_model = None
        for p in perplexities:
            model = TSNE(n_components=2, perplexity=p, init='pca', random_state=42, n_iter=500)
            Z = model.fit_transform(X)
            kl_div = model.kl_divergence_

            all_embeddings.append((p, Z, kl_div))
            if p == target_p:
                final_model = model

        # Calculate Global Min/Max to prevent the canvas from expanding/contracting
        all_z_vals = np.concatenate([emb[1] for emb in all_embeddings])
        global_min = np.min(all_z_vals) - 5
        global_max = np.max(all_z_vals) + 5

        for p, Z, kl_div in all_embeddings:
            loss_history.append({"perplexity": int(p), "kl_divergence": float(kl_div)})

            # FIX: Whitespace fix. Enforce square ratio and tight layout padding.
            fig, ax = plt.subplots(figsize=(6, 6), dpi=80)
            ax.scatter(Z[:, 0], Z[:, 1], c=y, cmap='viridis', s=20, alpha=0.8)
            ax.set_title(f"t-SNE 2D Projection (Perplexity = {p})", fontsize=12)
            ax.set_xlim(global_min, global_max)
            ax.set_ylim(global_min, global_max)
            ax.set_aspect('equal', adjustable='box') # Prevents weird stretching
            ax.grid(True, linestyle='--', alpha=0.5)

            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight', pad_inches=0.1)
            plt.close(fig)
            buffer.seek(0)

            history.append({
                "perplexity": int(p),
                "image": base64.b64encode(buffer.read()).decode('utf-8')
            })

        sizes =[{"cluster": f"Group {i}", "size": int(np.sum(y==i))} for i in np.unique(y)]

        return {
            'target_perplexity': target_p, 'final_kl': float(final_model.kl_divergence_),
            'original_image': original_img, 'history': history, 'loss_history': loss_history, 'cluster_sizes': sizes
        }

    except Exception as e:
        return {'error': str(e), 'traceback': traceback.format_exc()}
