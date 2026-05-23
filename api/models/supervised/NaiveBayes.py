import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import io
import base64
import traceback
from sklearn.model_selection import train_test_split

def render_nb_boundary(model, X, y, is_final=False):
    h = 0.05
    x_min, x_max = -8, 8
    y_min, y_max = -8, 8
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=80)

    Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1] if len(np.unique(y)) == 2 else model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    if len(np.unique(y)) == 2:
        ax.contourf(xx, yy, Z, 25, cmap='RdBu', alpha=0.5)
        ax.contour(xx, yy, Z, levels=[0.5], colors='k', linewidths=2)
    else:
        colors = ['#3B82F6', '#EF4444', '#22C55E', '#F59E0B']
        cmap_light = ListedColormap(colors[:len(np.unique(y))])
        ax.contourf(xx, yy, Z, alpha=0.3, cmap=cmap_light)

    for i, cls in enumerate(np.unique(y)):
        c = ['#3b82f6', '#ef4444', '#22c55e'][i % 3]
        ax.scatter(X[y==cls, 0], X[y==cls, 1], c=c, edgecolor='k', s=40)

    ax.set_xlim(-8, 8); ax.set_ylim(-8, 8)
    ax.set_title("Naive Bayes Probability Zones", fontsize=12)
    ax.axis('off')

    plt.tight_layout()
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode('utf-8')

def run_naive_bayes(data):
    try:
        points = data.get('points',[])
        X = np.array([[float(p['x']), float(p['y'])] for p in points])
        y = np.array([int(p['class']) for p in points])

        if len(np.unique(y)) < 2: return {'error': 'Need at least 2 classes.'}

        model = GaussianNB()
        model.fit(X, y)

        # Calculate Validation Accuracy (70/30 Split)
        if len(X) >= 5:
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)
            temp_model = GaussianNB()
            temp_model.fit(X_tr, y_tr)
            accuracy = accuracy_score(y_te, temp_model.predict(X_te))
        else:
            accuracy = accuracy_score(y, model.predict(X))
        boundary_img = render_nb_boundary(model, X, y)

        # Extract the Gaussian Bell Curve math for Recharts
        gaussian_data =[]
        x_range = np.linspace(-8, 8, 100)

        # P(x | y) = (1 / sqrt(2*pi*var)) * exp(-(x - mean)^2 / (2*var))
        for i, val in enumerate(x_range):
            point_data = {"x_val": float(val)}
            for c_idx, cls in enumerate(model.classes_):
                mean_x1 = model.theta_[c_idx, 0]
                var_x1 = model.var_[c_idx, 0]
                # Calculate bell curve height for Feature 1
                pdf = (1 / np.sqrt(2 * np.pi * var_x1)) * np.exp(-((val - mean_x1)**2) / (2 * var_x1))
                point_data[f"Class_{cls}_Dist"] = float(pdf)
            gaussian_data.append(point_data)

        # Priors
        priors = [{"class": f"Class {cls}", "prob": float(p)*100} for cls, p in zip(model.classes_, model.class_prior_)]

        return {
            'accuracy': float(accuracy),
            'boundary_image': boundary_img,
            'gaussian_data': gaussian_data,
            'priors': priors
        }
    except Exception as e:
        return {'error': str(e), 'traceback': traceback.format_exc()}

def predict_nb(data):
    try:
        X_train = np.array([[float(p['x']), float(p['y'])] for p in data['trained_points']])
        y_train = np.array([int(p['class']) for p in data['trained_points']])
        X_pred = np.array([[float(p['x']), float(p['y'])] for p in data['predict_points']])

        model = GaussianNB()
        model.fit(X_train, y_train)
        predictions = model.predict(X_pred).tolist()
        return {'predictions': predictions}
    except Exception as e: return {'error': str(e)}

def generate_nb_sample_data(data):
    try:
        from sklearn.datasets import make_blobs, make_moons, make_circles
        dataset_type = data.get('dataset_type', 'blobs')
        count = int(data.get('count', 60))
        variance = float(data.get('variance', 1.0))
        np.random.seed(42)
        if dataset_type == 'moons':
            X, y = make_moons(n_samples=count, noise=variance*0.1, random_state=42)
            X = X * 4.5 - 2
        elif dataset_type == 'circles':
            X, y = make_circles(n_samples=count, noise=variance*0.1, factor=0.5, random_state=42)
            X = X * 6.5
        else:
            X, y = make_blobs(n_samples=count, centers=[[-3,-3], [3,3]], cluster_std=variance*1.5, random_state=42)
        return {'points': [{'x': float(X[i,0]), 'y': float(X[i,1]), 'class': int(y[i])} for i in range(count)]}
    except Exception as e: return {'error': str(e)}
