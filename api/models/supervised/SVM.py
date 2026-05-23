import numpy as np
import matplotlib
matplotlib.use('Agg') # Non-interactive backend
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import base64
from io import BytesIO
from sklearn.model_selection import train_test_split
import traceback

def run_svm(data):
    """
    Perform SVM on the input data and return results for decision boundary, support vectors,
    and the hyperparameter history animation.
    """
    try:
        # Extract data
        X = np.array([[float(point['x']), float(point['y'])] for point in data['X']])
        y = np.array(data['y'])

        # Extract parameters
        kernel = data.get('kernel', 'linear')
        gamma = data.get('gamma', 'scale')
        degree = int(data.get('degree', 3))
        coef0 = float(data.get('coef0', 0.0))
        user_margin_width = float(data.get('marginWidth', 1.0))

        history =[]

        # 1. ANIMATION HISTORY: Train across different margin widths (Soft to Hard)
        # Margin Width goes from 5.0 (Soft) to 0.1 (Hard)
        margin_steps = np.linspace(5.0, 0.1, 10)

        for m_width in margin_steps:
            # Map visual margin width to SVM's C parameter
            # Smaller margin width = Harder Margin = Higher C
            c_val = max(0.01, 1.0 / m_width)

            temp_model = SVC(
                C=c_val,
                kernel=kernel,
                gamma=gamma,
                degree=degree,
                coef0=coef0,
                probability=False,
                random_state=42
            )
            temp_model.fit(X, y)
            temp_acc = accuracy_score(y, temp_model.predict(X))
            temp_boundary = generate_decision_boundary_only(temp_model, X, y, kernel, m_width)

            history.append({
                'margin': float(m_width),
                'C': float(c_val),
                'accuracy': float(temp_acc),
                'n_support': int(sum(temp_model.n_support_)),
                'decision_boundary': temp_boundary
            })

        # 2. FINAL MODEL: Train with the user's specific hyperparameters
        final_c = max(0.01, 1.0 / user_margin_width)
        model = SVC(
            C=final_c,
            kernel=kernel,
            gamma=gamma,
            degree=degree,
            coef0=coef0,
            probability=True, # Need probability for future predictions if requested
            random_state=42
        )
        model.fit(X, y)

        # Get support vectors
        support_vectors = model.support_vectors_.tolist()
        support_indices = model.support_
        support_classes = [int(y[i]) for i in support_indices]
        n_support = model.n_support_.tolist()

        # Calculate Validation Accuracy (70/30 Split)
        if len(X) >= 5:
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)
            temp_model = SVC(C=final_c, kernel=kernel, gamma=gamma, degree=degree, coef0=coef0, random_state=42)
            temp_model.fit(X_tr, y_tr)
            accuracy = accuracy_score(y_te, temp_model.predict(X_te))
        else:
            accuracy = accuracy_score(y, model.predict(X))

        # Generate decision boundaries
        decision_boundary = generate_decision_boundary_only(model, X, y, kernel, user_margin_width)
        final_plot = generate_final_plot(model, X, y, kernel, user_margin_width)

        # Add the final user frame to the end of the animation history
        history.append({
            'margin': float(user_margin_width),
            'C': float(final_c),
            'accuracy': float(accuracy),
            'n_support': int(sum(model.n_support_)),
            'decision_boundary': decision_boundary
        })

        # Package model info
        model_info = {
            'kernel': kernel,
            'gamma': gamma if gamma != 'scale' else 'scale',
            'degree': degree,
            'coef0': coef0,
            'marginWidth': user_margin_width,
            'C': final_c,
            'n_support': n_support,
            'total_support_vectors': len(support_vectors),
            'classes': model.classes_.tolist(),
            'intercept': model.intercept_.tolist(),
            'supportVectorClasses': support_classes
        }

        if kernel == 'linear':
            model_info['weights'] = model.coef_.tolist()

        return {
            'decisionBoundary': decision_boundary,
            'final_plot': final_plot,
            'history': history,
            'supportVectors': support_vectors,
            'supportVectorClasses': support_classes,
            'accuracy': float(accuracy),
            'model_info': model_info
        }

    except Exception as e:
        print(f"Error in SVM: {str(e)}")
        print(traceback.format_exc())
        return {
            'error': str(e),
            'traceback': traceback.format_exc()
        }

def generate_decision_boundary_only(model, X, y, kernel='linear', margin_width=1.0):
    """Generates the transparent decision boundary overlay for the interactive canvas."""
    try:
        plt.figure(figsize=(6, 6), dpi=150)
        range_min, range_max = -8, 8
        h = 0.05
        xx, yy = np.meshgrid(np.arange(range_min, range_max, h), np.arange(range_min, range_max, h))

        Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

        cmap_light = colors.LinearSegmentedColormap.from_list('custom_cmap',[(0.3, 0.5, 0.95, 0.45), (0.95, 0.3, 0.3, 0.45)])
        plt.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.9)

        if hasattr(model, 'decision_function'):
            Z_decision = model.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
            plt.contour(xx, yy, Z_decision, colors='k', levels=[0], linewidths=1.2)
            margin_factor = 1.0 * margin_width
            plt.contour(xx, yy, Z_decision, colors='k', levels=[margin_factor], linestyles='dashed', linewidths=1.0, dashes=(5, 3))
            plt.contour(xx, yy, Z_decision, colors='k', levels=[-margin_factor], linestyles='dashed', linewidths=1.0, dashes=(5, 3))

        plt.xlim(range_min, range_max)
        plt.ylim(range_min, range_max)
        plt.axis('off')

        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=150, transparent=True, bbox_inches='tight', pad_inches=0)
        plt.close()
        buffer.seek(0)

        return base64.b64encode(buffer.read()).decode('utf-8')
    except Exception as e:
        return None

def generate_final_plot(model, X, y, kernel, margin_width):
    """Generates the high-res static Final Plot showing the actual Support Vectors circled in gold."""
    try:
        fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
        range_min, range_max = -8, 8
        h = 0.05
        xx, yy = np.meshgrid(np.arange(range_min, range_max, h), np.arange(range_min, range_max, h))

        Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

        cmap_light = colors.ListedColormap(['#a0c4ff', '#fca5a5'])
        ax.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.5)

        if hasattr(model, 'decision_function'):
            Z_decision = model.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
            ax.contour(xx, yy, Z_decision, colors='k', levels=[0], linewidths=2)
            margin_factor = 1.0 * margin_width
            ax.contour(xx, yy, Z_decision, colors='k', levels=[-margin_factor, margin_factor], linestyles='dashed', alpha=0.7)

        # Plot training data
        ax.scatter(X[y==0, 0], X[y==0, 1], c='#3B82F6', edgecolors='k', label='Class 0', zorder=2)
        ax.scatter(X[y==1, 0], X[y==1, 1], c='#EF4444', edgecolors='k', label='Class 1', zorder=2)

        # Highlight Support Vectors
        sv = model.support_vectors_
        ax.scatter(sv[:, 0], sv[:, 1], s=150, linewidth=2, facecolors='none', edgecolors='gold', label='Support Vectors', zorder=3)

        ax.set_xlim(range_min, range_max)
        ax.set_ylim(range_min, range_max)
        ax.set_xlabel('Feature X1', fontsize=12)
        ax.set_ylabel('Feature X2', fontsize=12)
        ax.set_title(f'Final Boundary (Kernel: {kernel.upper()})', fontsize=14)
        ax.legend(loc='best')
        ax.grid(alpha=0.3, linestyle='--')

        plt.tight_layout()
        buffer = BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight')
        plt.close()
        buffer.seek(0)

        return base64.b64encode(buffer.read()).decode('utf-8')
    except Exception as e:
        return None

def predict_new_points(model, points):
    try:
        X_new = np.array([[point['x'], point['y']] for point in points])
        predictions = model.predict(X_new).tolist()
        probabilities = None
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(X_new).tolist()

        result =[]
        for i, point in enumerate(points):
            result.append({
                'x': point['x'],
                'y': point['y'],
                'class': predictions[i],
                'probability': probabilities[i] if probabilities else None
            })

        return {'predictions': result, 'probabilities': probabilities}
    except Exception as e:
        return {'error': str(e)}

def generate_sample_data(dataset_type='blobs', n_samples=40, n_clusters=3, variance=0.5):
    import numpy as np
    np.random.seed(42)
    from sklearn.datasets import make_blobs, make_moons, make_circles

    if dataset_type == 'moons':
        X, y = make_moons(n_samples=n_samples*2, noise=variance*0.1, random_state=42)
        X = X * 4.5 - 2
        y = y.astype(int)
    elif dataset_type == 'circles':
        X, y = make_circles(n_samples=n_samples*2, noise=variance*0.1, factor=0.5, random_state=42)
        X = X * 6.5
        y = y.astype(int)
    else:
        centers =[]
        for i in range(n_clusters):
            angle = i * (2 * np.pi / n_clusters)
            centers.append([4 * np.cos(angle), 4 * np.sin(angle)])
        X, y_numeric = make_blobs(n_samples=n_samples*n_clusters, centers=centers, cluster_std=variance*1.5, random_state=42)
        y = (y_numeric % 2).astype(int)

    return {'X': X.tolist(), 'y': y.tolist()}
