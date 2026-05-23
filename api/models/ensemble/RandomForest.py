import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
import base64
from io import BytesIO
import traceback

def generate_boundary_frame(model, X, y, mode, n_trees):
    h = 0.05
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    mesh_points = np.c_[xx.ravel(), yy.ravel()].astype(np.float64)
    Z = model.predict(mesh_points)

    if not np.issubdtype(Z.dtype, np.number):
        unique_z = np.unique(Z)
        z_map = {val: i for i, val in enumerate(unique_z)}
        Z = np.array([z_map[val] for val in Z], dtype=np.float64)

    Z = Z.reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=80)

    if mode == 'classification':
        unique_classes = np.unique(y)
        n_classes = len(unique_classes)
        colors =['#3B82F6', '#EF4444', '#22C55E', '#F59E0B']
        cmap_light = ListedColormap(colors[:n_classes]) if n_classes <= len(colors) else plt.cm.rainbow
        ax.contourf(xx, yy, Z.astype(float), alpha=0.3, cmap=cmap_light)

        for i, cls in enumerate(unique_classes):
            color = colors[i] if i < len(colors) else plt.cm.rainbow(i / n_classes)
            idx = np.where(y == cls)
            ax.scatter(X[idx, 0], X[idx, 1], c=color, edgecolor='k', s=30, alpha=0.8)
    else:
        contour = ax.contourf(xx, yy, Z, 20, cmap='viridis', alpha=0.8)
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolor='k', s=30)

    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())

    plt.tight_layout()
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=80, bbox_inches='tight', pad_inches=0.1, transparent=False)
    plt.close(fig)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode('utf-8')

def run_rf(data):
    try:
        X = np.array(data['X'], dtype=float)
        y = np.array(data['y'])
        mode = data.get('type', 'classification')

        trees = int(data.get('parameters', {}).get('trees', 15))
        max_depth = int(data.get('parameters', {}).get('max_depth', 5))

        history =[]
        final_model = None

        step_size = max(1, trees // 10)
        frames_to_render = list(range(1, trees, step_size))
        if trees not in frames_to_render:
            frames_to_render.append(trees)

        # 70/30 Validation Split
        np.random.seed(42)
        indices = np.random.permutation(len(X))
        split = int(0.7 * len(X))

        if split < 2 or len(X) - split < 1:
            X_tr, X_val = X, X
            y_tr, y_val = y, y
        else:
            X_tr, X_val = X[indices[:split]], X[indices[split:]]
            y_tr, y_val = y[indices[:split]], y[indices[split:]]

        for t in frames_to_render:
            if mode == 'classification':
                model = RandomForestClassifier(n_estimators=t, max_depth=max_depth, random_state=42)
                model.fit(X_tr, y_tr.astype(str))
                train_score = accuracy_score(y_tr.astype(str), model.predict(X_tr))
                val_score = accuracy_score(y_val.astype(str), model.predict(X_val))
            else:
                model = RandomForestRegressor(n_estimators=t, max_depth=max_depth, random_state=42)
                y_tr_float = y_tr.astype(float)
                y_val_float = y_val.astype(float)
                model.fit(X_tr, y_tr_float)
                train_score = mean_squared_error(y_tr_float, model.predict(X_tr))
                val_score = mean_squared_error(y_val_float, model.predict(X_val))

            full_model = RandomForestClassifier(n_estimators=t, max_depth=max_depth, random_state=42) if mode == 'classification' else RandomForestRegressor(n_estimators=t, max_depth=max_depth, random_state=42)
            full_model.fit(X, y.astype(float) if mode == 'regression' else y.astype(str))

            frame_img = generate_boundary_frame(full_model, X, y.astype(float) if mode=='regression' else y.astype(str), mode, t)

            history.append({
                'trees': t,
                'train_score': float(train_score),
                'val_score': float(val_score),
                'boundary': frame_img
            })
            if t == trees:
                final_model = full_model
                final_train_score = train_score
                final_val_score = val_score

        importances = final_model.feature_importances_
        feature_data =[
            {"feature": "Feature X1", "importance": float(importances[0]) * 100},
            {"feature": "Feature X2", "importance": float(importances[1]) * 100}
        ]

        if mode == 'regression':
            final_r2 = r2_score(y.astype(float), final_model.predict(X))
        else:
            final_r2 = None

        return {
            'mode': mode,
            'final_train_score': float(final_train_score),
            'final_val_score': float(final_val_score),
            'r2': float(final_r2) if final_r2 is not None else None,
            'feature_importance': feature_data,
            'history': history,
            'trees': trees,
            'max_depth': max_depth
        }

    except Exception as e:
        return {'error': str(e), 'traceback': traceback.format_exc()}


def predict_rf_points(data):
    """Generates predictions for new points"""
    try:
        X_train = np.array(data['trained_points']['X'], dtype=float)
        y_train = np.array(data['trained_points']['y'])
        X_predict = np.array(data['predict_points'], dtype=float)
        mode = data.get('type', 'classification')

        trees = int(data.get('parameters', {}).get('trees', 15))
        max_depth = int(data.get('parameters', {}).get('max_depth', 5))

        if mode == 'classification':
            model = RandomForestClassifier(n_estimators=trees, max_depth=max_depth, random_state=42)
            model.fit(X_train, y_train.astype(str))
            predictions = model.predict(X_predict).tolist()
        else:
            model = RandomForestRegressor(n_estimators=trees, max_depth=max_depth, random_state=42)
            model.fit(X_train, y_train.astype(float))
            predictions = [float(p) for p in model.predict(X_predict)]

        return {'predictions': predictions}
    except Exception as e:
        return {'error': str(e), 'traceback': traceback.format_exc()}


def generate_rf_sample_data(data):
    """Generates sample data perfectly formatted for the React frontend"""
    try:
        from sklearn.datasets import make_blobs, make_moons, make_circles

        data_type = data.get('type', 'classification')
        count = int(data.get('count', 40))
        dataset_type = data.get('dataset_type', 'blobs')
        n_clusters = int(data.get('n_clusters', 3))
        variance = float(data.get('variance', 0.5))
        sparsity = float(data.get('sparsity', 1.0))

        np.random.seed(42)

        if data_type == 'classification':
            if dataset_type == 'moons':
                X, y = make_moons(n_samples=count*2, noise=variance*0.1, random_state=42)
                X = X * 4.5 - 2
                y = [str(int(i)) for i in y]
            elif dataset_type == 'circles':
                X, y = make_circles(n_samples=count*2, noise=variance*0.1, factor=0.5, random_state=42)
                X = X * 6.5
                y =[str(int(i)) for i in y]
            else:
                centers =[]
                for i in range(n_clusters):
                    angle = i * (2 * np.pi / n_clusters)
                    centers.append([4 * np.cos(angle), 4 * np.sin(angle)])
                X, y_numeric = make_blobs(n_samples=count*n_clusters, centers=centers, cluster_std=variance*1.5, random_state=42)
                y =[str(int(i)) for i in y_numeric]
        else:
            if sparsity > 1.0:
                centers =[]
                num_clusters = min(5, int(sparsity * 2))
                for _ in range(num_clusters):
                    centers.append([np.random.uniform(-7, 7), np.random.uniform(-7, 7)])
                X =[]
                points_per_cluster = count // num_clusters
                remainder = count % num_clusters
                for i, center in enumerate(centers):
                    cluster_points = points_per_cluster + (1 if i < remainder else 0)
                    cluster_variance = 2.0 / sparsity
                    cluster_x = center[0] + np.random.normal(0, cluster_variance, cluster_points)
                    cluster_y = center[1] + np.random.normal(0, cluster_variance, cluster_points)
                    for j in range(cluster_points):
                        X.append([cluster_x[j], cluster_y[j]])
                X = np.array(X)
            else:
                range_scale = 16.0 * sparsity
                X = np.random.uniform(-range_scale/2, range_scale/2, (count, 2))

            if dataset_type == 'linear':
                y_raw = 2 + 0.5 * X[:, 0] + 0.3 * X[:, 1]
                y_raw += np.random.normal(0, variance * 0.5, count)
            else:
                # Better target scaling for regression RF
                y_raw = 2 + 0.5 * X[:, 0] + 0.3 * X[:, 1] + 0.2 * (X[:, 0]**2) - 0.5 * np.sin(X[:, 1])
                y_raw += np.random.normal(0, variance * 2.0, count)
            y =[float(val) for val in y_raw]

        # Format perfectly for frontend
        points = []

        if data_type == 'classification':
            for i in range(len(X)):
                points.append({
                    "x1": float(X[i][0]),
                    "x2": float(X[i][1]),
                    "y": str(y[i])
                })
        else:
            for i in range(len(X)):
                points.append({
                    "x1": float(X[i][0]),
                    "x2": float(X[i][1]),
                    "y": float(y_raw[i])
                })

        return {
            "points": points
        }

    except Exception as e:
        import traceback
        return {
            "error": str(e),
            "traceback": traceback.format_exc()
        }
