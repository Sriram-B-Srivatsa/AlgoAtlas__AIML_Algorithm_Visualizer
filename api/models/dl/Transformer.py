import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import traceback

# A Toy Vocabulary with 2D Embeddings so we can visualize "Meaning"
# Animals (High X, High Y), Actions (Low X, Low Y), Descriptors (Negative X, High Y), Stop Words (Near Origin)
TOY_VOCAB = {
    "cat": [0.8, 0.9, "Animal"], "dog":[0.85, 0.8, "Animal"], "fox":[0.9, 0.7, "Animal"],
    "bear":[0.95, 0.85, "Animal"], "bird":[0.7, 0.95, "Animal"],
    "sat":[-0.8, -0.8, "Action"], "jumps":[-0.7, -0.9, "Action"], "runs":[-0.9, -0.7, "Action"],
    "walks":[-0.85, -0.6, "Action"], "sleeps": [-0.6, -0.8, "Action"],
    "quick":[-0.5, 0.8, "Descriptor"], "brown":[-0.6, 0.7, "Descriptor"], "lazy": [-0.7, 0.6, "Descriptor"],
    "fast":[-0.4, 0.9, "Descriptor"], "red":[-0.5, 0.6, "Descriptor"],
    "the":[0.1, 0.1, "Stop Word"], "a":[0.0, 0.1, "Stop Word"], "an":[0.1, 0.0, "Stop Word"],
    "on":[-0.1, -0.1, "Stop Word"], "over":[0.0, -0.1, "Stop Word"]
}

def get_word_embedding(word):
    word = word.lower().strip()
    if word in TOY_VOCAB:
        return np.array(TOY_VOCAB[word][:2])
    # If word is unknown, assign a random vector near the origin
    return np.random.randn(2) * 0.2

def positional_encoding(seq_len, d_model=2):
    """Generates sine/cosine positional encodings"""
    pe = np.zeros((seq_len, d_model))
    for pos in range(seq_len):
        for i in range(0, d_model, 2):
            pe[pos, i] = np.sin(pos / (10000 ** ((2 * i)/d_model)))
            if i + 1 < d_model:
                pe[pos, i + 1] = np.cos(pos / (10000 ** ((2 * i)/d_model)))
    return pe

def render_attention_heatmap(attention_weights, words):
    """Renders the N x N Self-Attention Matrix"""
    fig, ax = plt.subplots(figsize=(6, 5), dpi=90)

    # Draw heatmap
    sns.heatmap(attention_weights, xticklabels=words, yticklabels=words,
                cmap="Blues", annot=True, fmt=".2f", linewidths=0.5, ax=ax, cbar=True)

    ax.set_title("Self-Attention Matrix (Q × K^T)", fontsize=14, pad=15)
    ax.set_xlabel("Keys (Words being attended to)")
    ax.set_ylabel("Queries (Words paying attention)")

    # Rotate x labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    plt.tight_layout()
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight')
    plt.close(fig)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode('utf-8')

def run_transformer(data):
    try:
        prompt = data.get('prompt', 'the quick brown fox').strip()
        params = data.get('parameters', {})
        temperature = float(params.get('temperature', 1.0))
        apply_causal_mask = bool(params.get('causal_mask', True))

        if not prompt:
            return {'error': 'Prompt cannot be empty.'}

        words = prompt.split()

        # If the stream gets too long, keep only the last 20 words so the visual matrix fits on screen
        if len(words) > 20:
            words = words[-20:]

        seq_len = len(words)
        d_model = 2 # 2D embeddings for visualization

        # 1. Embedding Layer
        embeddings = np.array([get_word_embedding(w) for w in words])

        # 2. Positional Encoding
        pos_enc = positional_encoding(seq_len, d_model)
        X = embeddings + pos_enc

        # 3. Query, Key, Value Projections
        # For this toy model, we make Q and K similar to Identity so similar words attend to each other
        W_q = np.array([[1.2, 0.1],[0.1, 1.2]])
        W_k = np.array([[1.0, 0.0],[0.0, 1.0]])
        W_v = np.array([[1.0, -0.2],[0.2, 1.0]])

        Q = np.dot(X, W_q)
        K = np.dot(X, W_k)
        V = np.dot(X, W_v)

        # 4. Scaled Dot-Product Attention
        scores = np.dot(Q, K.T) / np.sqrt(d_model)

        # Apply Causal Mask (GPT-style: words can only look backwards)
        if apply_causal_mask:
            mask = np.triu(np.ones((seq_len, seq_len)), k=1)
            scores = np.where(mask == 1, -1e9, scores)

        # Softmax to get Attention Weights
        exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        attention_weights = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        # 5. Output Context Vector
        context = np.dot(attention_weights, V)

        # 6. Predict Next Word (Compare final context vector to all vocab embeddings)
        final_state = context[-1] # The context of the last word in the sequence

        vocab_words = list(TOY_VOCAB.keys())
        vocab_embeddings = np.array([TOY_VOCAB[w][:2] for w in vocab_words])

        # Dot product distance to all words in dictionary
        logits = np.dot(vocab_embeddings, final_state)

        # Apply Temperature
        # Temp > 1 makes distribution flatter (more random)
        # Temp < 1 makes distribution sharper (more confident)
        logits = logits / max(temperature, 0.01)

        # Softmax for probabilities
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)

        # TEMPERATURE SAMPLING: Actually roll the dice using the probabilities to pick the next word
        sampled_idx = np.random.choice(len(vocab_words), p=probs)
        sampled_word = vocab_words[sampled_idx]

        # Get Top 5 Predictions for the Bar Chart
        top_indices = np.argsort(probs)[::-1][:5]
        top_predictions =[
            {"word": vocab_words[i], "probability": float(probs[i] * 100)}
            for i in top_indices
        ]

        # 7. Prepare Embedding Scatter Plot Data
        embedding_data =[]
        for word, val in TOY_VOCAB.items():
            embedding_data.append({
                "word": word,
                "x": val[0],
                "y": val[1],
                "category": val[2]
            })

        return {
            'tokens': words,
            'attention_heatmap': render_attention_heatmap(attention_weights, words),
            'predictions': top_predictions,
            'sampled_word': sampled_word, # The mathematically selected next token
            'embedding_data': embedding_data,
            'temperature': temperature
        }

    except Exception as e:
        return {'error': str(e), 'traceback': traceback.format_exc()}
