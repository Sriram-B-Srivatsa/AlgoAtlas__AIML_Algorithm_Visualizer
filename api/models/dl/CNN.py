import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
import traceback
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

def conv2d(image, kernel):
    """Pure Numpy 2D Convolution (Fast for 28x28 images)"""
    k_h, k_w = kernel.shape
    i_h, i_w = image.shape
    out = np.zeros((i_h - k_h + 1, i_w - k_w + 1))
    for y in range(out.shape[0]):
        for x in range(out.shape[1]):
            out[y, x] = np.sum(image[y:y+k_h, x:x+k_w] * kernel)
    return out

def max_pool_2d(mat, k=2):
    """Pure Numpy Max Pooling"""
    m, n = mat.shape
    # Pad if necessary
    pad_m = m % k
    pad_n = n % k
    if pad_m != 0 or pad_n != 0:
        mat = np.pad(mat, ((0, k - pad_m if pad_m != 0 else 0), (0, k - pad_n if pad_n != 0 else 0)), mode='constant')
        m, n = mat.shape
    return mat.reshape(m//k, k, n//k, k).max(axis=(1, 3))

def render_image_to_base64(img_array, cmap='gray', title=""):
    """Converts a 2D numpy array to a base64 PNG image"""
    fig, ax = plt.subplots(figsize=(3, 3), dpi=80)
    ax.imshow(img_array, cmap=cmap)
    ax.set_title(title, fontsize=12)
    ax.axis('off')
    plt.tight_layout()
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode('utf-8')

def run_cnn(data):
    try:
        # 1. Parse Image Data (28x28 flattened array from frontend)
        image_1d = np.array(data.get('image', np.zeros(784)))
        image_2d = image_1d.reshape((28, 28))

        # 2. Define standard CNN Edge Detection Kernels
        filters = {
            'Vertical Edge': np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]),
            'Horizontal Edge': np.array([[-1, -2, -1],[0, 0, 0], [1, 2, 1]]),
            'Outline': np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]),
            'Sharpen': np.array([[0, -1, 0], [-1, 5, -1],[0, -1, 0]])
        }

        feature_maps =[]

        # 3. Perform Convolution Layer -> ReLU -> MaxPooling
        for name, kernel in filters.items():
            # Convolution
            conv_out = conv2d(image_2d, kernel)
            # ReLU Activation
            relu_out = np.maximum(0, conv_out)
            # Max Pooling (2x2)
            pool_out = max_pool_2d(relu_out, k=2)

            feature_maps.append({
                'name': name,
                'conv_image': render_image_to_base64(conv_out, cmap='viridis', title=f"{name} (Conv2D)"),
                'pool_image': render_image_to_base64(pool_out, cmap='plasma', title=f"Max Pooling (2x2)")
            })

        original_img_base64 = render_image_to_base64(image_2d, cmap='gray', title="Original Input (28x28)")

        # 4. Train a Neural Network to classify the digit
        # To make it train in milliseconds, we use sklearn's 8x8 digits dataset
        digits = load_digits()
        X_data, y_data = digits.data, digits.target

        X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)

        # Train MLP (Neural Network)
        epochs = int(data.get('parameters', {}).get('epochs', 50))
        mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=epochs, alpha=1e-4,
                            solver='adam', random_state=42, learning_rate_init=0.01)

        mlp.fit(X_train, y_train)
        val_acc = mlp.score(X_test, y_test)

        # Format the loss curve for Recharts
        loss_history =[{"epoch": i+1, "loss": float(loss)} for i, loss in enumerate(mlp.loss_curve_)]

        # 5. Predict the User's Drawing!
        # sklearn's load_digits expects perfectly centered, padded 8x8 images.
        # We need to find the bounding box of your drawing, crop it, pad it, and downscale it properly.
        non_empty = np.where(image_2d > 0)
        if len(non_empty[0]) > 0:
            top, bottom = np.min(non_empty[0]), np.max(non_empty[0])
            left, right = np.min(non_empty[1]), np.max(non_empty[1])
            cropped = image_2d[top:bottom+1, left:right+1]

            # Make it a perfect square
            h, w = cropped.shape
            side = max(h, w)
            pad_y, pad_x = (side - h) // 2, (side - w) // 2
            square = np.pad(cropped, ((pad_y, side - h - pad_y), (pad_x, side - w - pad_x)), mode='constant')

            # Add a 20% border (sk-learn digits have thick black borders)
            pad_border = int(side * 0.2)
            square = np.pad(square, pad_border, mode='constant')

            # Downscale to 8x8 using spatial block averaging
            final_8x8 = np.zeros((8, 8))
            bin_size = square.shape[0] / 8.0
            for i in range(8):
                for j in range(8):
                    r_start, r_end = int(i * bin_size), max(int((i + 1) * bin_size), int(i * bin_size) + 1)
                    c_start, c_end = int(j * bin_size), max(int((j + 1) * bin_size), int(j * bin_size) + 1)
                    final_8x8[i, j] = np.mean(square[r_start:r_end, c_start:c_end])

            # Normalize to match sklearn's 0-16 color intensity scale
            final_8x8 = (final_8x8 / np.max(final_8x8)) * 16.0 if np.max(final_8x8) > 0 else final_8x8
        else:
            final_8x8 = np.zeros((8, 8))

        probabilities = mlp.predict_proba(final_8x8.reshape(1, -1))[0]
        prediction = int(np.argmax(probabilities))

        # Format probabilities for BarChart
        prob_chart_data =[{"digit": str(i), "probability": float(p) * 100} for i, p in enumerate(probabilities)]

        return {
            'prediction': prediction,
            'confidence': float(np.max(probabilities) * 100),
            'val_accuracy': float(val_acc * 100),
            'epochs': len(mlp.loss_curve_),
            'original_image': original_img_base64,
            'feature_maps': feature_maps,
            'loss_history': loss_history,
            'probabilities': prob_chart_data
        }

    except Exception as e:
        return {'error': str(e), 'traceback': traceback.format_exc()}
