import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import io
import base64
import traceback

def render_reg_frame(X_raw, y_raw, X_range, y_pred_standard, y_pred_reg, alpha, reg_type):
    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=80)
    ax.scatter(X_raw, y_raw, color='#9ca3af', edgecolor='k', s=40, alpha=0.5, label='Data')
    ax.plot(X_range, y_pred_standard, color='#ef4444', linewidth=2, linestyle='--', alpha=0.6, label='Overfitted (No Penalty)')
    ax.plot(X_range, y_pred_reg, color='#3b82f6', linewidth=3, label=f'{reg_type} (Alpha={alpha:.3f})')

    ax.set_ylim(-5, 5); ax.set_xlim(-8, 8)
    ax.set_title(f"{reg_type} Regularization Smoothing", fontsize=12)
    ax.legend(loc='best', fontsize=8)
    ax.grid(alpha=0.3, linestyle='--')

    plt.tight_layout()
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode('utf-8')

def run_regularization(data):
    try:
        points = data.get('points',[])
        reg_type = data.get('parameters', {}).get('type', 'Ridge')
        max_alpha = float(data.get('parameters', {}).get('alpha', 10.0))
        degree = int(data.get('parameters', {}).get('degree', 8))

        X = np.array([[float(p['x'])] for p in points])
        y = np.array([float(p['y']) for p in points])

        poly = PolynomialFeatures(degree=degree, include_bias=False)
        X_poly = poly.fit_transform(X)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_poly)

        std_model = LinearRegression()
        std_model.fit(X_scaled, y)

        X_range = np.linspace(-8, 8, 100).reshape(-1, 1)
        X_range_scaled = scaler.transform(poly.transform(X_range))
        y_pred_std = std_model.predict(X_range_scaled)

        history, coef_history = [],[]
        alphas = np.logspace(-3, np.log10(max_alpha), 15)

        for a in alphas:
            model = Ridge(alpha=a) if reg_type == 'Ridge' else Lasso(alpha=a, max_iter=10000)
            model.fit(X_scaled, y)
            mse = mean_squared_error(y, model.predict(X_scaled))

            coef_dict = {"alpha": float(a)}
            for i, c in enumerate(model.coef_): coef_dict[f"w{i+1}"] = float(c)
            coef_history.append(coef_dict)

            history.append({
                "alpha": float(a), "mse": float(mse),
                "image": render_reg_frame(X, y, X_range, y_pred_std, model.predict(X_range_scaled), a, reg_type)
            })

        final_model = Ridge(alpha=max_alpha) if reg_type == 'Ridge' else Lasso(alpha=max_alpha, max_iter=10000)
        final_model.fit(X_scaled, y)
        final_r2 = r2_score(y, final_model.predict(X_scaled))

        # Prepare final weights for BarChart
        final_weights =[{"feature": f"X^{i+1}", "weight": float(c)} for i, c in enumerate(final_model.coef_)]

        return {
            'reg_type': reg_type, 'max_alpha': max_alpha, 'degree': degree,
            'final_r2': float(final_r2), 'final_mse': float(history[-1]['mse']),
            'history': history, 'coef_history': coef_history, 'final_weights': final_weights
        }
    except Exception as e: return {'error': str(e)}

def predict_regularization(data):
    try:
        X_train = np.array([[float(p['x'])] for p in data['trained_points']])
        y_train = np.array([float(p['y']) for p in data['trained_points']])
        X_pred = np.array([[float(p['x'])] for p in data['predict_points']])

        reg_type = data['parameters']['type']
        alpha = float(data['parameters']['alpha'])
        degree = int(data['parameters']['degree'])

        poly = PolynomialFeatures(degree=degree, include_bias=False)
        scaler = StandardScaler()

        X_poly_train = scaler.fit_transform(poly.fit_transform(X_train))
        X_poly_pred = scaler.transform(poly.transform(X_pred))

        model = Ridge(alpha=alpha) if reg_type == 'Ridge' else Lasso(alpha=alpha, max_iter=10000)
        model.fit(X_poly_train, y_train)
        predictions = model.predict(X_poly_pred).tolist()

        return {'predictions': predictions}
    except Exception as e: return {'error': str(e)}

def generate_reg_sample_data(data):
    try:
        count = int(data.get('count', 20))
        dataset_type = data.get('dataset_type', 'sine')
        noise = float(data.get('noise', 0.5))
        np.random.seed(42)

        X = np.sort(np.random.uniform(-7, 7, count))

        if dataset_type == 'sine':
            y = np.sin(X) + np.random.normal(0, noise, count)
        elif dataset_type == 'linear':
            y = 0.5 * X + np.random.normal(0, noise, count)
        else: # nonlinear
            y = 0.1 * X**3 - 0.5 * X**2 + np.random.normal(0, noise, count)
            # Scale to fit canvas
            y = (y / np.max(np.abs(y))) * 4.0

        return {'points':[{'x': float(X[i]), 'y': float(y[i])} for i in range(count)]}
    except Exception as e:
        return {'error': str(e)}
