import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
import traceback
from sklearn.metrics import mean_squared_error

class PureNumpyRNN:
    """A mathematically pure Recurrent Neural Network built from scratch in NumPy"""
    def __init__(self, input_dim=1, hidden_dim=8, output_dim=1, lr=0.01):
        # Weights
        self.Wx = np.random.randn(hidden_dim, input_dim) * 0.1  # Input to Hidden
        self.Wh = np.random.randn(hidden_dim, hidden_dim) * 0.1 # Hidden to Hidden (Memory)
        self.Wy = np.random.randn(output_dim, hidden_dim) * 0.1 # Hidden to Output
        # Biases
        self.bh = np.zeros((hidden_dim, 1))
        self.by = np.zeros((output_dim, 1))
        self.lr = lr

    def forward(self, xs):
        """Processes a sequence of inputs one timestep at a time"""
        hs = {}
        hs[-1] = np.zeros((self.Wh.shape[0], 1)) # Initial memory is zero
        for t, x in enumerate(xs):
            x_vec = np.array([[x]])
            # Core RNN Math: Current Memory = tanh(Input*Wx + PastMemory*Wh + bias)
            hs[t] = np.tanh(np.dot(self.Wx, x_vec) + np.dot(self.Wh, hs[t-1]) + self.bh)

        # The output prediction is based on the FINAL memory state
        y_pred = np.dot(self.Wy, hs[len(xs)-1]) + self.by
        return y_pred[0,0], hs

    def backward(self, xs, hs, error):
        """Backpropagation Through Time (BPTT)"""
        dWx, dWh, dWy = np.zeros_like(self.Wx), np.zeros_like(self.Wh), np.zeros_like(self.Wy)
        dbh, dby = np.zeros_like(self.bh), np.zeros_like(self.by)
        dh_next = np.zeros_like(hs[0])

        dWy += error * hs[len(xs)-1].T
        dby += error

        # Unroll time backward
        for t in reversed(range(len(xs))):
            dh = np.dot(self.Wy.T, error) + dh_next
            # Derivative of tanh
            dtanh = (1 - hs[t] ** 2) * dh
            dbh += dtanh
            x_vec = np.array([[xs[t]]])
            dWx += np.dot(dtanh, x_vec.T)
            dWh += np.dot(dtanh, hs[t-1].T)
            dh_next = np.dot(self.Wh.T, dtanh)

        # Gradient Clipping (Prevents Exploding Gradients)
        for dparam in[dWx, dWh, dWy, dbh, dby]:
            np.clip(dparam, -1, 1, out=dparam)

        # Update weights
        self.Wx -= self.lr * dWx
        self.Wh -= self.lr * dWh
        self.Wy -= self.lr * dWy
        self.bh -= self.lr * dbh
        self.by -= self.lr * dby

def render_rnn_frame(X_all, y_true, y_pred, window_start, lookback, t, is_final=False):
    """Renders the sliding window animation over the time-series data"""
    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=80)

    # Plot true data (The past)
    ax.plot(X_all, y_true, c='#9ca3af', linewidth=2, label='True Sequence', alpha=0.5)
    ax.scatter(X_all, y_true, c='#9ca3af', s=20, alpha=0.5)

    # Plot predictions so far
    if len(y_pred) > 0:
        pred_x = X_all[lookback:lookback+len(y_pred)]
        ax.plot(pred_x, y_pred, c='#ef4444', linewidth=2, label='RNN Predictions', marker='o', markersize=4)

    if not is_final and window_start is not None:
        # Highlight the "Lookback Window" (what the RNN is currently reading)
        window_x = X_all[window_start:window_start+lookback]
        window_y = y_true[window_start:window_start+lookback]
        ax.plot(window_x, window_y, c='#3b82f6', linewidth=4, label='Current Lookback Window')

        # Highlight the target it's trying to predict
        if window_start + lookback < len(X_all):
            target_x = X_all[window_start+lookback]
            target_y = y_true[window_start+lookback]
            ax.scatter([target_x],[target_y], c='#10b981', s=100, zorder=5, label='Target Next Step')

    ax.set_ylim(0, 1) # Normalized coordinates
    ax.set_xlim(X_all[0], X_all[-1])
    ax.set_title(f"Sequence Prediction (Timestep {t})" if not is_final else "Final Sequence Prediction", fontsize=12)
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Value")
    ax.grid(alpha=0.3, linestyle='--')
    ax.legend(loc='lower left', fontsize=8)

    plt.tight_layout()
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode('utf-8')

def run_rnn(data):
    try:
        points = data.get('points',[])
        params = data.get('parameters', {})
        epochs = int(params.get('epochs', 50))
        lookback = int(params.get('lookback', 5))
        learning_rate = float(params.get('learningRate', 0.05))

        if len(points) <= lookback + 1:
            return {'error': f'Need at least {lookback + 2} points to use a lookback window of {lookback}.'}

        # Sort points by Time (X axis)
        points = sorted(points, key=lambda p: float(p['x']))

        # We normalize Y to 0-1 range to keep the math stable
        X_raw = np.array([float(p['x']) for p in points])
        y_raw = np.array([float(p['y']) for p in points])
        y_min, y_max = np.min(y_raw), np.max(y_raw)

        if y_max - y_min == 0:
            return {'error': 'Y values must vary. Cannot train on a perfectly flat line.'}

        y_norm = (y_raw - y_min) / (y_max - y_min)

        # Build sequences: X =[y_{t-L}, ..., y_{t-1}], Target = y_t
        inputs, targets = [],[]
        for i in range(len(y_norm) - lookback):
            inputs.append(y_norm[i:i+lookback])
            targets.append(y_norm[i+lookback])

        rnn = PureNumpyRNN(input_dim=1, hidden_dim=12, output_dim=1, lr=learning_rate)

        loss_history =[]

        # 1. Training Loop
        for epoch in range(epochs):
            epoch_loss = 0
            for i in range(len(inputs)):
                seq_in = inputs[i]
                target = targets[i]

                # Forward
                y_pred, hs = rnn.forward(seq_in)
                error = y_pred - target
                epoch_loss += error ** 2

                # Backward (BPTT)
                rnn.backward(seq_in, hs, error)

            loss_history.append({"epoch": epoch + 1, "loss": float(epoch_loss / len(inputs))})

        # 2. Testing / Animation Loop (Generate frames for the UI)
        history = []
        hidden_state_activity =[]
        final_predictions =[]

        # We take the trained model and run it across the sequence to build the video
        for i in range(len(inputs)):
            seq_in = inputs[i]
            y_pred, hs = rnn.forward(seq_in)
            final_predictions.append(y_pred)

            # Measure how "active" the memory cell was during this sequence
            avg_memory_activation = float(np.mean(np.abs(hs[lookback-1])))

            # Measure prediction error at this specific timestep
            step_error = float(abs(y_pred - targets[i]))

            hidden_state_activity.append({
                "timestep": i + lookback,
                "memory_activation": avg_memory_activation,
                "prediction_error": step_error
            })

            # Save frame for animation
            img_base64 = render_rnn_frame(X_raw, y_norm, final_predictions, window_start=i, lookback=lookback, t=i+lookback)
            history.append({
                "timestep": i + lookback,
                "image": img_base64
            })

        # Final static frame
        final_img = render_rnn_frame(X_raw, y_norm, final_predictions, window_start=None, lookback=lookback, t='Final', is_final=True)
        history.append({
            "timestep": "Final",
            "image": final_img
        })

        # Calculate final MSE using the imported function
        final_mse = mean_squared_error(targets, final_predictions)

        return {
            'epochs': epochs,
            'lookback': lookback,
            'final_loss': float(final_mse),
            'loss_history': loss_history,
            'history': history,
            'hidden_state_activity': hidden_state_activity
        }

    except Exception as e:
        return {'error': str(e), 'traceback': traceback.format_exc()}


def generate_rnn_sample_data(data):
    """Generates Time-Series data (Sine waves, Stock Trends)"""
    try:
        dataset_type = data.get('dataset_type', 'sine')
        count = int(data.get('count', 60))
        noise = float(data.get('noise', 0.1))

        np.random.seed(42)
        X = np.linspace(0, 10, count)

        if dataset_type == 'sine':
            y = np.sin(X) + np.random.normal(0, noise, count)
        elif dataset_type == 'stock':
            # Random walk with drift
            y = np.cumsum(np.random.normal(0.05, 0.2 + noise, count))
        else: # Triangle wave
            from scipy import signal
            y = signal.sawtooth(X * np.pi) + np.random.normal(0, noise, count)

        # Map to -8 to 8 scale roughly for the UI canvas
        X_mapped = np.linspace(-7, 7, count)
        y_mapped = (y / (np.max(np.abs(y)) + 1e-5)) * 6

        points =[{'x': float(X_mapped[i]), 'y': float(y_mapped[i])} for i in range(count)]
        return {'points': points}

    except Exception as e:
        import traceback
        return {'error': str(e), 'traceback': traceback.format_exc()}
