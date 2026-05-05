from sklearn.metrics import roc_auc_score

positive_scores = {
    'extraction_strength': [0.705],
    'forget_Q_A_Prob':     [0.880],
    'forget_Q_A_ROUGE':    [0.817],
    'forget_truth_ratio':  [0.476],
    'privleak':            [-99.3],
}

# All 8 methods now
negative_scores = {
    'extraction_strength': [0.033, 0.085, 0.093, 0.568, 0.098, 0.054, 0.665, 0.167],
    'forget_Q_A_Prob':     [0.0,   0.065, 0.072, 0.844, 0.123, 0.108, 0.880, 0.527],
    'forget_Q_A_ROUGE':    [0.000, 0.361, 0.281, 0.738, 0.082, 0.322, 0.796, 0.462],
    'forget_truth_ratio':  [0.0,   0.445, 0.580, 0.467, 0.552, 0.577, 0.466, 0.488],
    'privleak':            [16.5, -34.9,  22.2, -99.2, -18.2,  35.3, -99.4, -94.0],
}

print("=" * 55)
print(f"{'Metric':<30} {'AUC-ROC':>8}  Verdict")
print("=" * 55)

for metric in positive_scores:
    scores = positive_scores[metric] + negative_scores[metric]
    labels = [1] + [0] * 8
    if metric == 'privleak':
        scores = [-s for s in scores]
    try:
        auc = roc_auc_score(labels, scores)
        verdict = "✅ Faithful" if auc >= 0.8 else ("⚠️ Moderate" if auc >= 0.6 else "❌ Unreliable")
        print(f"{metric:<30} {auc:>8.3f}  {verdict}")
    except Exception as e:
        print(f"{metric:<30} {'ERROR':>8}  {e}")
print("=" * 55)