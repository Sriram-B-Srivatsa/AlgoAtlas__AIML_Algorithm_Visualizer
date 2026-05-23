import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
import traceback

def render_diffusion_frame(X, iteration, total_steps, title):
    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=80)

    # Beautiful purple-to-blue gradient for the dots
    ax.scatter(X[:, 0], X[:, 1], c='#8b5cf6', edgecolor='white', s=30, alpha=0.8, linewidths=0.5)

    ax.set_xlim(-8, 8)
    ax.set_ylim(-8, 8)
    ax.set_title(f"{title} (Step {iteration}/{total_steps})", fontsize=12)
    ax.grid(alpha=0.3, linestyle='--')

    plt.tight_layout()
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode('utf-8')

def run_diffusion(data):
    try:
        from sklearn.datasets import make_s_curve, make_moons, make_circles
        import traceback

        # Parse parameters from the frontend
        timesteps = int(data.get('parameters', {}).get('timesteps', 50))
        noise_amount = float(data.get('parameters', {}).get('noise', 1.0))
        dataset_shape = data.get('parameters', {}).get('dataset_type', 's_curve')

        # We mathematically generate the hidden target structure
        np.random.seed(42)
        if dataset_shape == 's_curve':
            X_target, _ = make_s_curve(n_samples=400, noise=0.0)
            X_target = X_target[:, [0, 2]]  # Take 2D projection of the 3D shape
            X_target = (X_target / np.max(np.abs(X_target))) * 6.0 # Scale to fit -8 to 8 canvas
        elif dataset_shape == 'moons':
            X_target, _ = make_moons(n_samples=400, noise=0.05)
            X_target = (X_target - np.mean(X_target, axis=0)) # Center it on the origin
            X_target = (X_target / np.max(np.abs(X_target))) * 6.0
        else: # circles
            X_target, _ = make_circles(n_samples=400, noise=0.02, factor=0.5)
            X_target = (X_target / np.max(np.abs(X_target))) * 6.0

        history = []
        noise_schedule =[]

        # Simulated Denoising Process (Reverse Process)
        # We start with pure static noise, and mathematically interpolate it towards the S-Curve target
        X_current = np.random.randn(400, 2) * 3.0 * noise_amount

        frames_to_render = np.unique(np.linspace(0, timesteps, 20, dtype=int)).tolist()

        for t in range(timesteps + 1):
            alpha = t / float(timesteps) # 0 to 1

            # The mathematical "score function" pushing noise back into structure
            X_frame = X_current * (1 - alpha) + X_target * alpha

            # Add a tiny bit of Langevin noise to simulate true diffusion (bouncing into place)
            if t < timesteps:
                X_frame += np.random.randn(400, 2) * 0.5 * (1 - alpha)

            # Log the Signal-to-Noise Ratio (SNR) for the Recharts graph
            signal = alpha * 100
            noise_pct = (1 - alpha) * 100
            noise_schedule.append({"step": t, "Signal": float(signal), "Noise": float(noise_pct)})

            if t in frames_to_render:
                history.append({
                    "step": t,
                    "image": render_diffusion_frame(X_frame, t, timesteps, "Denoising (Reverse Process)")
                })

        # Render original noise frame for comparison
        pure_noise_img = render_diffusion_frame(X_current, 0, timesteps, "Pure Static Noise")

        return {
            'timesteps': timesteps,
            'history': history,
            'noise_schedule': noise_schedule,
            'pure_noise_img': pure_noise_img
        }

    except Exception as e:
        return {'error': str(e), 'traceback': traceback.format_exc()}
