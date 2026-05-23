import pandas as pd
import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import traceback

def run_apriori(data):
    try:
        transactions = data.get('transactions',[])
        min_support = float(data.get('parameters', {}).get('minSupport', 0.2))
        min_confidence = float(data.get('parameters', {}).get('minConfidence', 0.5))

        if len(transactions) < 3:
            return {'error': 'Need at least 3 transactions.'}

        # 1. Calculate Support (Frequency) of single items
        total_tx = len(transactions)
        item_counts = {}
        for tx in transactions:
            for item in set(tx):
                item_counts[item] = item_counts.get(item, 0) + 1

        # Filter by min_support
        valid_items = {k: v/total_tx for k, v in item_counts.items() if (v/total_tx) >= min_support}

        # 2. Calculate Pairs (Rules)
        pairs = {}
        for tx in transactions:
            items = list(set(tx))
            for i in range(len(items)):
                for j in range(i+1, len(items)):
                    if items[i] in valid_items and items[j] in valid_items:
                        pair = frozenset([items[i], items[j]])
                        pairs[pair] = pairs.get(pair, 0) + 1

        # 3. Generate Rules A -> B
        rules =[]
        for pair, count in pairs.items():
            pair_support = count / total_tx
            if pair_support >= min_support:
                item_a, item_b = list(pair)

                # A -> B
                conf_a_b = pair_support / valid_items[item_a]
                lift_a_b = conf_a_b / valid_items[item_b]
                if conf_a_b >= min_confidence:
                    rules.append({
                        "antecedent": item_a, "consequent": item_b,
                        "support": round(pair_support, 3), "confidence": round(conf_a_b, 3), "lift": round(lift_a_b, 3)
                    })

                # B -> A
                conf_b_a = pair_support / valid_items[item_b]
                lift_b_a = conf_b_a / valid_items[item_a]
                if conf_b_a >= min_confidence:
                    rules.append({
                        "antecedent": item_b, "consequent": item_a,
                        "support": round(pair_support, 3), "confidence": round(conf_b_a, 3), "lift": round(lift_b_a, 3)
                    })

        rules.sort(key=lambda x: x['lift'], reverse=True)

        # 4. Generate Support vs Confidence Scatter Plot
        fig, ax = plt.subplots(figsize=(6, 4.5), dpi=80)
        if len(rules) > 0:
            # FIX: Add "Jitter" (tiny random noise) so overlapping dots spread out and become visible!
            import numpy as np
            supps = [r['support'] + np.random.uniform(-0.015, 0.015) for r in rules]
            confs = [r['confidence'] + np.random.uniform(-0.015, 0.015) for r in rules]
            lifts = [r['lift'] for r in rules]

            # Dropped alpha to 0.6 so overlapping colors blend beautifully
            sc = ax.scatter(supps, confs, c=lifts, cmap='viridis', s=100, alpha=0.6, edgecolor='k')
            fig.colorbar(sc, ax=ax, label='Lift')

        ax.set_xlim(-0.05, 1.25)
        ax.set_ylim(-0.05, 1.25)
        ax.set_xlabel("Support (Frequency)")
        ax.set_ylabel("Confidence (Reliability)")
        ax.set_title("Association Rules Mapped")
        ax.grid(alpha=0.3, linestyle='--')

        plt.tight_layout()
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight')
        plt.close(fig)

        return {
            'rules': rules,
            'total_items': len(valid_items),
            'scatter_plot': base64.b64encode(buffer.getvalue()).decode('utf-8')
        }

    except Exception as e:
        return {'error': str(e), 'traceback': traceback.format_exc()}
