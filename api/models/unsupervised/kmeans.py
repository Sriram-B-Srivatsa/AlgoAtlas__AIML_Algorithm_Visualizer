import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.datasets import make_blobs, make_moons, make_circles
from sklearn.metrics import silhouette_score
import io
import base64
import traceback

def create_cluster_plot(X, labels, centroids, iteration, converged=False):
    """Generates a base64 image frame for the K-Means animation"""
    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)

    # Custom colormap for clusters
    colors =['#3b82f6', '#ef4444', '#22c55e', '#a855f7', '#f97316', '#ec4899', '#0ea5e9', '#eab308', '#06b6d4', '#8b5cf6']

    # If labels are all -1 (initial state), plot them gray
    if np.all(labels == -1):
        ax.scatter(X[:, 0], X[:, 1], c='#9ca3af', alpha=0.6, edgecolors='none', s=50)
    else:
        for i in range(len(centroids)):
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                ax.scatter(cluster_points[:, 0], cluster_points[:, 1],
                           c=colors[i % len(colors)], alpha=0.6, edgecolors='w', s=50, linewidth=0.5)

    # Plot centroids as large stars with black outlines
    for i, centroid in enumerate(centroids):
        ax.scatter(centroid[0], centroid[1], marker='*', s=350,
                   c=colors[i % len(colors)], edgecolors='black', linewidth=1.5, zorder=10)

    ax.set_xlim(-8, 8)
    ax.set_ylim(-8, 8)
    ax.set_xlabel('Feature X1', fontsize=12)
    ax.set_ylabel('Feature X2', fontsize=12)

    title = f'Final Converged Clusters (Iteration {iteration})' if converged else f'K-Means Iteration {iteration}'
    ax.set_title(title, fontsize=14)
    ax.grid(alpha=0.3, linestyle='--')

    plt.tight_layout()
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight')
    plt.close(fig)
    buffer.seek(0)

    return base64.b64encode(buffer.read()).decode('utf-8')

def run_kmeans(data, k=3, max_iterations=100):
    try:
        X = np.array(data['X'])
        if X.shape[1] != 2:
            return {'error': 'K-means visualization currently only supports 2D data'}

        history = []
        inertia_history =[]

        # 1. Initialize centroids randomly
        np.random.seed(42)
        indices = np.random.choice(X.shape[0], k, replace=False)
        centroids = X[indices]
        labels = np.full(X.shape[0], -1) # -1 means unassigned

        # Capture Initial State
        initial_plot = create_cluster_plot(X, labels, centroids, 0)
        history.append({
            'iteration': 0,
            'image': initial_plot
        })

        converged = False
        iteration = 0
        prev_centroids = None

        # K-Means Training Loop
        while not converged and iteration < max_iterations:
            iteration += 1

            # Step 1: Assign each point to the nearest centroid
            distances = np.sqrt(((X - centroids[:, np.newaxis])**2).sum(axis=2))
            labels = np.argmin(distances, axis=0)

            # Calculate Inertia for this step
            inertia = 0
            for i in range(k):
                cluster_points = X[labels == i]
                if len(cluster_points) > 0:
                    inertia += np.sum(np.square(cluster_points - centroids[i]))

            inertia_history.append({
                "iteration": iteration,
                "inertia": float(inertia)
            })

            # Step 2: Update centroids
            new_centroids = np.array([X[labels == i].mean(axis=0) if np.sum(labels == i) > 0
                                     else centroids[i] for i in range(k)])

            # Check for convergence
            if prev_centroids is not None and np.allclose(new_centroids, prev_centroids, rtol=1e-4):
                converged = True

            prev_centroids = new_centroids
            centroids = new_centroids

            # Capture Iteration Frame
            step_plot = create_cluster_plot(X, labels, centroids, iteration, converged)
            history.append({
                'iteration': iteration,
                'image': step_plot
            })

        # Calculate final silhouette score
        sil_score = None
        unique_labels = np.unique(labels)
        if len(unique_labels) > 1 and all(np.sum(labels == i) > 1 for i in unique_labels):
            try:
                sil_score = silhouette_score(X, labels)
            except Exception:
                sil_score = None

        # Calculate Cluster Sizes for the Bar Chart
        cluster_sizes =[]
        for i in range(k):
            count = int(np.sum(labels == i))
            cluster_sizes.append({
                "cluster": f"Cluster {i+1}",
                "size": count
            })

        return {
            'centroids': centroids.tolist(),
            'labels': labels.tolist(),
            'iterations': iteration,
            'converged': converged,
            'inertia': float(inertia_history[-1]['inertia']) if inertia_history else 0.0,
            'silhouette_score': float(sil_score) if sil_score else None,
            'history': history,                 # For the Animation Player
            'inertia_history': inertia_history, # For the Line Chart
            'cluster_sizes': cluster_sizes,     # For the Bar Chart
            'final_plot': history[-1]['image']
        }

    except Exception as e:
        print(f"Error in KMeans: {str(e)}")
        print(traceback.format_exc())
        return {'error': str(e), 'traceback': traceback.format_exc()}

def generate_clustering_data(n_samples=100, n_clusters=3, variance=0.5, dataset_type='blobs'):
    np.random.seed(42)
    if dataset_type == 'moons':
        X, _ = make_moons(n_samples=n_samples, noise=variance * 0.1)
        X = X * 4 - 1.5
        return {'X': X.tolist()}
    elif dataset_type == 'circles':
        X, _ = make_circles(n_samples=n_samples, noise=variance * 0.1, factor=0.5)
        X = X * 6
        return {'X': X.tolist()}
    else:
        centers = np.random.uniform(-7, 7, (n_clusters, 2))
        X, _ = make_blobs(n_samples=n_samples, centers=centers, cluster_std=variance * 2, random_state=42)
        return {'X': X.tolist()}
