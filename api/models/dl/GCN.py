import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
from sklearn.preprocessing import StandardScaler
import traceback

def render_gcn_latent(H, labels, epoch, is_final=False):
    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=80)
    colors = ['#3b82f6', '#ef4444', '#22c55e', '#f59e0b', '#9ca3af']

    for i, h in enumerate(H):
        c = colors[labels[i]] if labels[i] != -1 else colors[4]
        ax.scatter(h[0], h[1], c=c, s=100, edgecolor='white', linewidth=1)
        ax.annotate(str(i), (h[0], h[1]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)

    ax.set_title(f"Node Embeddings (Epoch {epoch})" if not is_final else "Final Separated Node Embeddings", fontsize=12)
    ax.grid(alpha=0.3, linestyle='--')

    plt.tight_layout()
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode('utf-8')

def run_gcn(data):
    try:
        nodes = data.get('nodes',[])
        edges = data.get('edges',[])
        epochs = int(data.get('parameters', {}).get('epochs', 50))
        learning_rate = float(data.get('parameters', {}).get('learningRate', 0.1))

        N = len(nodes)
        if N < 3: return {'error': 'Need at least 3 nodes.'}
        if len(edges) == 0: return {'error': 'Nodes must be connected by edges.'}

        # Build Adjacency Matrix (A)
        A = np.zeros((N, N))
        for edge in edges:
            A[edge['source'], edge['target']] = 1
            A[edge['target'], edge['source']] = 1 # Undirected

        # Add Self-Loops (A_hat = A + I)
        A_hat = A + np.eye(N)

        # Degree Matrix (D_hat)
        D_hat = np.diag(np.sum(A_hat, axis=1))
        D_inv_sqrt = np.linalg.inv(np.sqrt(D_hat))

        # Normalized Adjacency: D^{-1/2} * A_hat * D^{-1/2}
        A_norm = np.dot(np.dot(D_inv_sqrt, A_hat), D_inv_sqrt)

        # Node Features (X): Map pixel coordinates (0-600) to (-1, 1) to prevent vanishing gradients
        X = np.array([[(n['x'] - 300)/300.0, (n['y'] - 300)/300.0] for n in nodes])
        labels = np.array([n['class'] for n in nodes])

        labeled_idx = np.where(labels != -1)[0]
        if len(set(labels[labeled_idx])) < 2:
            return {'error': 'Need at least one Class 0 and one Class 1 node.'}

        # GCN Weights - Increased hidden size and variance to fix flatlining loss
        np.random.seed(42)
        W0 = np.random.randn(2, 8) * 1.0
        W1 = np.random.randn(8, 2) * 1.0

        def relu(x): return np.maximum(0, x)
        def softmax(x):
            exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
            return exp_x / np.sum(exp_x, axis=1, keepdims=True)

        history = []
        loss_history =[]

        for epoch in range(epochs):
            # Forward Pass: H = A_norm * X * W
            H0 = relu(np.dot(np.dot(A_norm, X), W0))
            H1 = np.dot(np.dot(A_norm, H0), W1)
            probs = softmax(H1)

            # Cross-Entropy Loss (Only on Labeled Nodes!)
            loss = 0
            for i in labeled_idx:
                true_class = labels[i]
                loss -= np.log(probs[i, true_class] + 1e-8)
            loss /= len(labeled_idx)

            loss_history.append({"epoch": epoch+1, "loss": float(loss)})

            # Backward Pass (Simplified Gradient Descent on Labeled Nodes)
            d_out = probs.copy()
            for i in labeled_idx:
                d_out[i, labels[i]] -= 1
            d_out /= len(labeled_idx)

            # GCN Gradients
            dW1 = np.dot(H0.T, np.dot(A_norm, d_out))
            dH0 = np.dot(np.dot(A_norm, d_out), W1.T) * (H0 > 0)
            dW0 = np.dot(X.T, np.dot(A_norm, dH0))

            W1 -= learning_rate * dW1
            W0 -= learning_rate * dW0

            if epoch % max(1, epochs//10) == 0 or epoch == epochs-1:
                history.append({
                    "epoch": epoch+1,
                    "image": render_gcn_latent(H1, labels, epoch+1, epoch==epochs-1)
                })

        # Generate Final Predictions for Unlabeled Nodes
        final_probs = softmax(np.dot(np.dot(A_norm, relu(np.dot(np.dot(A_norm, X), W0))), W1))
        final_preds = np.argmax(final_probs, axis=1)

        return {
            'epochs': epochs,
            'final_loss': float(loss_history[-1]['loss']),
            'predictions': final_preds.tolist(),
            'history': history,
            'loss_history': loss_history
        }

    except Exception as e:
        return {'error': str(e), 'traceback': traceback.format_exc()}
