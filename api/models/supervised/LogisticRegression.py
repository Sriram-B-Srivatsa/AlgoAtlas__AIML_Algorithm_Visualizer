import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, log_loss
import io
import base64
import traceback

def render_logreg_frame(model, X, y, iteration, is_final=False):
    h = 0.05
    x_min, x_max = -8, 8
    y_min, y_max = -8, 8
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=80)

    if model is not None and hasattr(model, 'coef_'):
        Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, 25, cmap='RdBu', alpha=0.6)
        ax.contour(xx, yy, Z, levels=[0.5], colors='k', linewidths=2)

    ax.scatter(X[y==0, 0], X[y==0, 1], c='#ef4444', edgecolor='k', s=40, label='Class 0')
    ax.scatter(X[y==1, 0], X[y==1, 1], c='#3b82f6', edgecolor='k', s=40, label='Class 1')

    ax.set_xlim(-8, 8); ax.set_ylim(-8, 8)
    ax.set_title(f"Decision Boundary (Epoch {iteration})" if not is_final else "Final Probability Heatmap", fontsize=12)
    ax.axis('off')

    plt.tight_layout()
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode('utf-8')

def run_logistic_regression(data):
    try:
        from sklearn.model_selection import train_test_split

        points = data.get('points',[])
        epochs = int(data.get('parameters', {}).get('epochs', 100))
        learning_rate = float(data.get('parameters', {}).get('learningRate', 0.01))

        X = np.array([[float(p['x']), float(p['y'])] for p in points])
        y = np.array([int(p['class']) for p in points])

        if len(np.unique(y)) < 2:
            return {'error': 'Logistic Regression requires at least one point from both Class 0 and Class 1.'}

        # 70/30 Split to calculate real accuracy
        if len(X) >= 5:
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)
        else:
            X_train, X_val, y_train, y_val = X, X, y, y

        history =[]
        loss_history =[]

        # Use SGDClassifier with log_loss to manually step through epochs
        model = SGDClassifier(loss='log_loss', learning_rate='constant', eta0=learning_rate, max_iter=1, warm_start=True, random_state=42)

        frames_to_render = np.unique(np.linspace(1, epochs, 10, dtype=int)).tolist()

        for epoch in range(1, epochs + 1):
            model.fit(X_train, y_train)

            # Calculate metrics
            y_pred_proba = model.predict_proba(X_train)
            current_loss = log_loss(y_train, y_pred_proba)

            train_acc = accuracy_score(y_train, model.predict(X_train))
            val_acc = accuracy_score(y_val, model.predict(X_val))

            loss_history.append({
                "epoch": epoch,
                "loss": float(current_loss),
                "train_accuracy": float(train_acc),
                "val_accuracy": float(val_acc)
            })

            # We pass the FULL dataset (X, y) to the renderer so the user sees all their dots on screen
            if epoch in frames_to_render or epoch == epochs:
                history.append({
                    "epoch": epoch,
                    "image": render_logreg_frame(model, X, y, epoch, is_final=(epoch==epochs))
                })

        final_accuracy = accuracy_score(y_val, model.predict(X_val))

        return {
            'epochs': epochs,
            'final_accuracy': float(final_accuracy),
            'final_loss': float(loss_history[-1]['loss']),
            'history': history,
            'loss_history': loss_history
        }

    except Exception as e:
        return {'error': str(e), 'traceback': traceback.format_exc()}

def predict_logreg(data):
    try:
        X_train = np.array([[float(p['x']), float(p['y'])] for p in data['trained_points']])
        y_train = np.array([int(p['class']) for p in data['trained_points']])
        X_pred = np.array([[float(p['x']), float(p['y'])] for p in data['predict_points']])

        model = SGDClassifier(loss='log_loss', max_iter=int(data['parameters']['epochs']), random_state=42)
        model.fit(X_train, y_train)
        predictions = model.predict(X_pred).tolist()

        return {'predictions': predictions}
    except Exception as e: return {'error': str(e)}

def generate_logreg_sample_data(data):
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
        else: # blobs
            X, y = make_blobs(n_samples=count, centers=[[-3,-3], [3,3]], cluster_std=variance*1.5, random_state=42)

        points = [{'x': float(X[i,0]), 'y': float(X[i,1]), 'class': int(y[i])} for i in range(count)]
        return {'points': points}

    except Exception as e:
        return {'error': str(e)}
