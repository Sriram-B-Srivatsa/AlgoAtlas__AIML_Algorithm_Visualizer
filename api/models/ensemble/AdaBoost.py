import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import base64
from io import BytesIO
import traceback

def generate_boundary_frame(model, X, y, mode, n_estimators):
    """Generates a single frame of the AdaBoost decision boundary animation"""
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

    # Return raw base64 string without prefix
    return base64.b64encode(buffer.read()).decode('utf-8')

def run_adaboost(data):
    try:
        X = np.array(data['X'], dtype=float)
        y = np.array(data['y'])
        mode = data.get('type', 'classification')

        estimators = int(data.get('parameters', {}).get('estimators', 50))
        learning_rate = float(data.get('parameters', {}).get('learningRate', 1.0))

        # 70/30 Train-Validation Split for genuine curve generation (fallback to 100% if tiny dataset)
        if len(X) >= 5:
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)
        else:
            X_train, X_val, y_train, y_val = X, X, y, y

        history =[]
        final_model = None

        # We step by max(1, estimators//10) to generate ~10 animation frames
        step_size = max(1, estimators // 10)
        frames_to_render = list(range(1, estimators, step_size))
        if estimators not in frames_to_render:
            frames_to_render.append(estimators)

        for t in frames_to_render:
            if mode == 'classification':
                # AdaBoost uses depth=1 (Stumps) by default
                base_estimator = DecisionTreeClassifier(max_depth=1, random_state=42)
                model = AdaBoostClassifier(estimator=base_estimator, n_estimators=t, learning_rate=learning_rate, random_state=42)

                model.fit(X_train, y_train.astype(str))
                train_score = accuracy_score(y_train.astype(str), model.predict(X_train))
                val_score = accuracy_score(y_val.astype(str), model.predict(X_val))

                frame_img = generate_boundary_frame(model, X, y.astype(str), mode, t)
            else:
                base_estimator = DecisionTreeRegressor(max_depth=3, random_state=42)
                model = AdaBoostRegressor(estimator=base_estimator, n_estimators=t, learning_rate=learning_rate, random_state=42)

                y_train_f, y_val_f = y_train.astype(float), y_val.astype(float)
                model.fit(X_train, y_train_f)
                train_score = mean_squared_error(y_train_f, model.predict(X_train))
                val_score = mean_squared_error(y_val_f, model.predict(X_val))

                frame_img = generate_boundary_frame(model, X, y.astype(float), mode, t)

            history.append({
                'estimators': t,
                'train_score': float(train_score),
                'val_score': float(val_score),
                'boundary': frame_img
            })
            if t == estimators:
                final_model = model

        # Feature Importance for Bar Chart
        importances = final_model.feature_importances_
        feature_data = [
            {"feature": "Feature X1", "importance": float(importances[0]) * 100},
            {"feature": "Feature X2", "importance": float(importances[1]) * 100}
        ]

        if mode == 'regression':
            final_r2 = r2_score(y.astype(float), final_model.predict(X))
        else:
            final_r2 = None

        return {
            'mode': mode,
            'final_train_score': float(history[-1]['train_score']),
            'final_val_score': float(history[-1]['val_score']),
            'r2': float(final_r2) if final_r2 is not None else None,
            'feature_importance': feature_data,
            'history': history,
            'estimators': estimators,
            'learningRate': learning_rate
        }

    except Exception as e:
        return {'error': str(e), 'traceback': traceback.format_exc()}


def predict_adaboost_points(data):
    try:
        X_train = np.array(data['trained_points']['X'], dtype=float)
        y_train = np.array(data['trained_points']['y'])
        X_pred = np.array(data['predict_points'], dtype=float)
        mode = data.get('type', 'classification')

        estimators = int(data.get('parameters', {}).get('estimators', 50))
        learning_rate = float(data.get('parameters', {}).get('learningRate', 1.0))

        if mode == 'classification':
            base_estimator = DecisionTreeClassifier(max_depth=1, random_state=42)
            model = AdaBoostClassifier(estimator=base_estimator, n_estimators=estimators, learning_rate=learning_rate, random_state=42)
            model.fit(X_train, y_train.astype(str))
            predictions = model.predict(X_pred).tolist()
        else:
            base_estimator = DecisionTreeRegressor(max_depth=3, random_state=42)
            model = AdaBoostRegressor(estimator=base_estimator, n_estimators=estimators, learning_rate=learning_rate, random_state=42)
            model.fit(X_train, y_train.astype(float))
            predictions = model.predict(X_pred).tolist()

        return {'predictions': predictions}
    except Exception as e:
        return {'error': str(e)}


def generate_adaboost_sample_data(data):
    """Isolated backend data generator for AdaBoost"""
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
                y =[str(int(i)) for i in y]
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
                y_raw += np.random.normal(0, variance * 0.5, count) # Reduced noise multiplier
            else:
                y_raw = 2 + 0.5 * X[:, 0] + 0.3 * X[:, 1] + 0.2 * (X[:, 0]**2) - 0.1 * (X[:, 0] * X[:, 1])
                y_raw += np.random.normal(0, variance * 2.0, count)
            y = [float(val) for val in y_raw]

        points =[]
        for i in range(len(X)):
            points.append({
                'x1': float(X[i][0]),
                'x2': float(X[i][1]),
                'y': y[i]
            })

        return {'points': points}

    except Exception as e:
        import traceback
        return {'error': str(e), 'traceback': traceback.format_exc()}
