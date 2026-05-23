import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.linear_model import LogisticRegression
import io
import base64
import traceback

def generate_boundary_frame(model, X_all, y_current, iteration):
    h = 0.05
    x_min, x_max = -8, 8
    y_min, y_max = -8, 8
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=80)

    # Draw boundary if model is trained
    if model is not None:
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        cmap_light = ListedColormap(['#93c5fd', '#fca5a5'])
        ax.contourf(xx, yy, Z, alpha=0.4, cmap=cmap_light)

        # Draw confidence margins
        Z_proba = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
        Z_proba = Z_proba.reshape(xx.shape)
        ax.contour(xx, yy, Z_proba, levels=[0.1, 0.5, 0.9], colors='k', linestyles=[':', '-', ':'], alpha=0.5, linewidths=1)

    # Plot points
    for i in range(len(X_all)):
        if y_current[i] == 0:
            ax.scatter(X_all[i, 0], X_all[i, 1], c='#3b82f6', edgecolor='k', s=40, zorder=3)
        elif y_current[i] == 1:
            ax.scatter(X_all[i, 0], X_all[i, 1], c='#ef4444', edgecolor='k', s=40, zorder=3)
        else:
            ax.scatter(X_all[i, 0], X_all[i, 1], c='#9ca3af', edgecolor='white', s=30, alpha=0.6, zorder=2)

    ax.set_xlim(-8, 8)
    ax.set_ylim(-8, 8)
    ax.set_title(f"Pseudo-Labeling (Iteration {iteration})", fontsize=12)
    ax.axis('off')

    plt.tight_layout()
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode('utf-8')

def run_pseudo_labeling(data):
    try:
        points = data.get('points',[])
        confidence_threshold = float(data.get('parameters', {}).get('confidence', 0.85))
        max_iter = int(data.get('parameters', {}).get('max_iter', 10))

        X_all = np.array([[float(p['x']), float(p['y'])] for p in points])
        y_true = np.array([int(p['class']) for p in points]) # -1 means unlabeled

        y_current = y_true.copy()
        labeled_indices = np.where(y_current != -1)[0]

        if len(set(y_current[labeled_indices])) < 2:
            return {'error': 'Need at least one labeled point of each class (0 and 1).'}

        history =[]
        confidence_history =[]

        history.append({
            'iteration': 0,
            'pseudo_labeled_count': 0,
            'image': generate_boundary_frame(None, X_all, y_current, 0)
        })

        model = LogisticRegression(random_state=42)
        total_pseudo_labeled = 0

        for iteration in range(1, max_iter + 1):
            labeled_idx = np.where(y_current != -1)[0]
            unlabeled_idx = np.where(y_current == -1)[0]

            model.fit(X_all[labeled_idx], y_current[labeled_idx])

            if len(unlabeled_idx) == 0:
                history.append({
                    'iteration': iteration,
                    'pseudo_labeled_count': int(total_pseudo_labeled),
                    'image': generate_boundary_frame(model, X_all, y_current, iteration)
                })
                break

            probas = model.predict_proba(X_all[unlabeled_idx])
            max_probas = np.max(probas, axis=1)
            predictions = np.argmax(probas, axis=1)

            confident_mask = max_probas >= confidence_threshold
            newly_labeled_count = int(np.sum(confident_mask)) # FIXED: Force Python int

            confidence_history.append({
                'iteration': iteration,
                'avg_confidence': float(np.mean(max_probas) * 100)
            })

            if newly_labeled_count == 0:
                history.append({
                    'iteration': iteration,
                    'pseudo_labeled_count': int(total_pseudo_labeled),
                    'image': generate_boundary_frame(model, X_all, y_current, iteration)
                })
                break

            confident_indices_in_unlabeled = np.where(confident_mask)[0]
            actual_indices = unlabeled_idx[confident_indices_in_unlabeled]

            y_current[actual_indices] = predictions[confident_indices_in_unlabeled]
            total_pseudo_labeled += newly_labeled_count

            history.append({
                'iteration': iteration,
                'pseudo_labeled_count': int(total_pseudo_labeled),
                'image': generate_boundary_frame(model, X_all, y_current, iteration)
            })

        return {
            'total_points': int(len(X_all)),
            'initial_labeled': int(len(labeled_indices)),
            'final_pseudo_labeled': int(total_pseudo_labeled),
            'remaining_unlabeled': int(np.sum(y_current == -1)), # FIXED: Force Python int
            'history': history,
            'confidence_history': confidence_history
        }

    except Exception as e:
        return {'error': str(e), 'traceback': traceback.format_exc()}


def generate_pl_sample_data(data):
    """Generates Unlabeled data with a tiny fraction of labeled targets"""
    try:
        from sklearn.datasets import make_moons, make_blobs, make_circles

        dataset_type = data.get('dataset_type', 'moons')
        count = int(data.get('count', 100))
        variance = float(data.get('variance', 0.15))
        n_clusters = int(data.get('n_clusters', 2))

        np.random.seed(42)
        if dataset_type == 'moons':
            X, y = make_moons(n_samples=count, noise=variance, random_state=42)
            X = X * 4.5 - 2
        elif dataset_type == 'circles':
            X, y = make_circles(n_samples=count, noise=variance, factor=0.5, random_state=42)
            X = X * 6.5
        else: # blobs
            centers =[]
            for i in range(n_clusters):
                angle = i * (2 * np.pi / n_clusters)
                centers.append([4 * np.cos(angle), 4 * np.sin(angle)])
            X, y = make_blobs(n_samples=count, centers=centers, cluster_std=variance*3.0, random_state=42)

        # Hide almost all labels (-1 = unlabeled)
        y_semi = np.full(count, -1)

        # Keep only 3 points labeled for Class 0, and 3 points for Class 1
        idx_0 = np.where(y == 0)[0][:3]
        idx_1 = np.where(y == 1)[0][:3]

        if len(idx_0) > 0: y_semi[idx_0] = 0
        if len(idx_1) > 0: y_semi[idx_1] = 1

        points = [{'x': float(X[i,0]), 'y': float(X[i,1]), 'class': int(y_semi[i])} for i in range(count)]
        return {'points': points}

    except Exception as e:
        import traceback
        return {'error': str(e), 'traceback': traceback.format_exc()}
