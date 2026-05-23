import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
import io
import base64
import traceback

def render_hierarchical_frame(X, labels, n_clusters, is_final=False):
    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=80)
    colors = plt.cm.tab10.colors

    for i in range(n_clusters):
        mask = (labels == i)
        ax.scatter(X[mask, 0], X[mask, 1], c=[colors[i % len(colors)]], edgecolor='white', s=50, alpha=0.8)

    ax.set_xlim(-8, 8); ax.set_ylim(-8, 8)
    ax.set_title(f"Agglomerative Merging ({n_clusters} Clusters Remaining)" if not is_final else f"Final Clusters (K={n_clusters})", fontsize=12)
    ax.grid(alpha=0.3, linestyle='--')

    plt.tight_layout()
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode('utf-8')

def render_dendrogram(Z, threshold):
    fig, ax = plt.subplots(figsize=(8, 4), dpi=100)
    dendrogram(Z, truncate_mode='level', p=5, ax=ax, color_threshold=threshold)
    ax.axhline(y=threshold, c='r', linestyle='--', label='Cutoff Distance')
    ax.set_title("Hierarchical Dendrogram", fontsize=14)
    ax.set_xlabel("Data Points (or Clusters)")
    ax.set_ylabel("Merge Distance")
    ax.legend()

    plt.tight_layout()
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode('utf-8')

def run_hierarchical(data):
    try:
        X = np.array([[float(p['x']), float(p['y'])] for p in data.get('points',[])])
        n_clusters = int(data.get('parameters', {}).get('clusters', 3))
        linkage_type = data.get('parameters', {}).get('linkage', 'ward')

        if len(X) < max(n_clusters, 2):
            return {'error': 'Need more data points than target clusters.'}

        Z = linkage(X, method=linkage_type)
        history = []
        cluster_history =[]

        start_k = min(len(X), 30)
        steps = np.unique(np.linspace(start_k, n_clusters, 15, dtype=int))[::-1]

        for k in steps:
            model = AgglomerativeClustering(n_clusters=k, linkage=linkage_type)
            labels = model.fit_predict(X)

            sil_score = silhouette_score(X, labels) if k > 1 else 0
            cluster_history.append({"clusters": int(k), "silhouette": float(sil_score)})

            history.append({
                "clusters": int(k),
                "image": render_hierarchical_frame(X, labels, k, is_final=(k==n_clusters))
            })

        final_model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_type)
        final_labels = final_model.fit_predict(X)
        final_sil = silhouette_score(X, final_labels) if n_clusters > 1 else 0

        threshold = Z[-(n_clusters-1), 2] if n_clusters > 1 else Z[-1, 2] + 1
        dendro_img = render_dendrogram(Z, threshold)
        sizes =[{"cluster": f"Cluster {i+1}", "size": int(np.sum(final_labels==i))} for i in range(n_clusters)]

        return {
            'target_clusters': n_clusters,
            'linkage': linkage_type,
            'final_silhouette': float(final_sil),
            'history': history,
            'cluster_history': cluster_history,
            'cluster_sizes': sizes,
            'dendrogram': dendro_img
        }
    except Exception as e:
        return {'error': str(e), 'traceback': traceback.format_exc()}

def generate_hierarchical_sample_data(data):
    try:
        from sklearn.datasets import make_blobs, make_moons, make_circles
        dataset_type = data.get('dataset_type', 'blobs')
        count = int(data.get('count', 60))
        n_clusters = int(data.get('n_clusters', 3))
        variance = float(data.get('variance', 0.5))

        np.random.seed(42)
        if dataset_type == 'moons':
            X, _ = make_moons(n_samples=count, noise=variance*0.1, random_state=42)
            X = X * 4.5 - 2
        elif dataset_type == 'circles':
            X, _ = make_circles(n_samples=count, noise=variance*0.1, factor=0.5, random_state=42)
            X = X * 6.5
        else:
            # FIX: Mathematically force the blobs to spawn in a tight circle within the -7 to 7 bounds!
            centers =[]
            for i in range(n_clusters):
                angle = i * (2 * np.pi / n_clusters)
                centers.append([4 * np.cos(angle), 4 * np.sin(angle)])
            X, _ = make_blobs(n_samples=count, centers=centers, cluster_std=variance*1.5, random_state=42)

            # Safety clamp: Ensure absolutely no point exceeds the canvas bounds
            max_val = np.max(np.abs(X))
            if max_val > 7.5:
                X = (X / max_val) * 7.5

        points =[{'x': float(p[0]), 'y': float(p[1])} for p in X]
        return {'points': points}
    except Exception as e:
        return {'error': str(e)}
