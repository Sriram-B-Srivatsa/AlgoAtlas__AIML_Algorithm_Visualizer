import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use Agg backend (non-interactive)
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.preprocessing import StandardScaler
import io
import base64
import traceback
from sklearn.model_selection import train_test_split

class KNeighborsClassifier:
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors
        self.X = None
        self.y = None

    def fit(self, X, y):
        self.X = np.array(X)
        self.y = np.array(y)
        return self

    def predict(self, X):
        X = np.array(X)
        predictions =[]
        for x in X:
            distances = np.sqrt(np.sum((self.X - x) ** 2, axis=1))
            k_nearest_indices = np.argsort(distances)[:self.n_neighbors]
            k_nearest_labels = self.y[k_nearest_indices]
            unique_labels, counts = np.unique(k_nearest_labels, return_counts=True)
            predictions.append(unique_labels[np.argmax(counts)])
        return np.array(predictions)

class KNeighborsRegressor:
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors
        self.X = None
        self.y = None

    def fit(self, X, y):
        self.X = np.array(X)
        self.y = np.array(y, dtype=float)
        return self

    def predict(self, X):
        X = np.array(X)
        predictions =[]
        for x in X:
            distances = np.sqrt(np.sum((self.X - x) ** 2, axis=1))
            k_nearest_indices = np.argsort(distances)[:self.n_neighbors]
            k_nearest_values = self.y[k_nearest_indices]
            predictions.append(np.mean(k_nearest_values))
        return np.array(predictions)

def predict_single_point(data, predict_point, n_neighbors=5):
    """Predict the class or value of a single point using KNN"""
    try:
        X = np.array(data['X'], dtype=float)
        y_orig = np.array(data['y'])
        predict_point = np.array(predict_point, dtype=float).reshape(1, -1)
        mode = data.get('mode', 'classification')

        if X.shape[0] < 1:
            return {"error": "Need at least 1 training point for prediction"}

        if mode == 'classification':
            y = np.array([str(val) for val in y_orig])
            model = KNeighborsClassifier(n_neighbors=min(n_neighbors, X.shape[0]))
            model.fit(X, y)
            predicted_class = str(model.predict(predict_point)[0])
            return {'predicted_class': predicted_class}
        else:
            y = np.array([float(val) for val in y_orig])
            model = KNeighborsRegressor(n_neighbors=min(n_neighbors, X.shape[0]))
            model.fit(X, y)
            predicted_value = model.predict(predict_point)[0]
            return {'predicted_class': str(round(predicted_value, 3))}
    except Exception as e:
        return {"error": f"Error making prediction: {str(e)}"}

def render_boundary_image(model, X, y, mode, k, h=0.1, is_final=False):
    """Helper function to render a single boundary frame"""
    x_min, x_max = -8, 8
    y_min, y_max = -8, 8
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    fig, ax = plt.subplots(figsize=(8, 8) if is_final else (6, 6), dpi=100)

    if mode == 'regression':
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        contour = ax.contourf(xx, yy, Z, 50, cmap='viridis', alpha=0.8)
        if is_final:
            fig.colorbar(contour, ax=ax, label='Predicted Value')
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolor='k', s=40)
    else:
        unique_classes = np.unique(y)
        class_to_num = {cls: i for i, cls in enumerate(unique_classes)}
        y_numeric = np.array([class_to_num[cls] for cls in y])

        Z_strings = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z_numeric = np.array([class_to_num[cls] for cls in Z_strings]).reshape(xx.shape)

        if len(unique_classes) <= 2: colors = ['#3b82f6', '#ef4444']
        elif len(unique_classes) == 3: colors =['#3b82f6', '#ef4444', '#22c55e']
        else: colors = plt.cm.tab10.colors

        cmap = mcolors.ListedColormap(colors[:len(unique_classes)])

        ax.contourf(xx, yy, Z_numeric, levels=len(unique_classes)-1, alpha=0.7, cmap=cmap)
        ax.contour(xx, yy, Z_numeric, levels=len(unique_classes)-1, colors='k', linewidths=0.5, alpha=0.5)

        for idx, cls in enumerate(unique_classes):
            mask = (y == cls)
            ax.scatter(X[mask, 0], X[mask, 1], c=[colors[idx]], edgecolor='k', label=f'Class {cls}', s=40)

        if is_final and len(unique_classes) > 0:
            ax.legend(loc='upper right')

    ax.grid(color='gray', linestyle=':', linewidth=0.5, alpha=0.3)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlim(-8, 8)
    ax.set_ylim(-8, 8)

    if is_final:
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.set_title(f'Final KNN Boundary (K={k})')
    else:
        ax.axis('off')

    plt.tight_layout()
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight')
    plt.close(fig)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode('utf-8')

def generate_decision_boundary(data, n_neighbors=None):
    """
    Main Training Function.
    Calculates Train/Val Error across K values, generates animation frames, and final stats.
    """
    try:
        X = np.array(data['X'], dtype=float)
        y_orig = np.array(data['y'])
        mode = data.get('mode', 'classification')
        if n_neighbors is None:
            n_neighbors = int(data.get('n_neighbors', 5))

        if X.shape[0] < 5:
            return {"error": "Need at least 5 training points to generate a decision boundary"}

        # 1. Prepare Data and 70/30 Split for Validation Curve
        # 1. Prepare Data and 70/30 Split for Validation Curve
        if mode == 'classification':
            y = np.array([str(val) for val in y_orig])
        else:
            y = np.array([float(val) for val in y_orig])

        # Use standard 70/30 split, but fallback to 100% if the user placed very few dots
        if len(X) >= 5:
            X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.3, random_state=42)
        else:
            X_tr, X_val, y_tr, y_val = X, X, y, y
        history =[]
        max_k_to_test = min(15, len(X_tr))

        # We only want to generate ~8 animation frames so it doesn't timeout
        frame_steps = np.unique(np.linspace(1, max_k_to_test, 8, dtype=int))
        if n_neighbors not in frame_steps and n_neighbors <= max_k_to_test:
            frame_steps = np.sort(np.append(frame_steps, n_neighbors))

        # 2. Loop through K values to build the Validation Curve
        for k in range(1, max_k_to_test + 1):
            if mode == 'classification':
                model = KNeighborsClassifier(n_neighbors=k)
                model.fit(X_tr, y_tr)

                # Accuracy
                train_score = np.mean(model.predict(X_tr) == y_tr)
                val_score = np.mean(model.predict(X_val) == y_val) if len(y_val) > 0 else 0

            else:
                model = KNeighborsRegressor(n_neighbors=k)
                model.fit(X_tr, y_tr)

                # Mean Squared Error
                train_score = np.mean((model.predict(X_tr) - y_tr)**2)
                val_score = np.mean((model.predict(X_val) - y_val)**2) if len(y_val) > 0 else 0

            # Save Image Frame if K is in our subset
            img_base64 = None
            if k in frame_steps:
                # Train full model for the image frame
                full_model = KNeighborsClassifier(n_neighbors=k) if mode == 'classification' else KNeighborsRegressor(n_neighbors=k)
                full_model.fit(X, y)
                # Lower resolution (h=0.2) for fast animation frames
                img_base64 = render_boundary_image(full_model, X, y, mode, k, h=0.2, is_final=False)

            history.append({
                'k': k,
                'train_score': float(train_score),
                'val_score': float(val_score),
                'image': img_base64
            })

        # 3. Train the Final Chosen Model
        n_neighbors = min(n_neighbors, len(X))
        final_model = KNeighborsClassifier(n_neighbors=n_neighbors) if mode == 'classification' else KNeighborsRegressor(n_neighbors=n_neighbors)
        final_model.fit(X, y)

        # High resolution (h=0.08) for final static image
        final_boundary = render_boundary_image(final_model, X, y, mode, n_neighbors, h=0.08, is_final=True)

        # Find stats for the chosen K
        chosen_stats = next((item for item in history if item['k'] == n_neighbors), history[-1])

        return {
            'history': history,
            'decision_boundary': final_boundary,
            'metrics': {
                'chosen_k': n_neighbors,
                'train_score': chosen_stats['train_score'],
                'val_score': chosen_stats['val_score'],
                'mode': mode
            }
        }

    except Exception as e:
        print(f"Error in generate_decision_boundary: {str(e)}")
        print(traceback.format_exc())
        return {"error": f"Error generating decision boundary: {str(e)}"}
