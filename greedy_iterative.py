"""
Greedy Iterative Shapley — "Winner Takes It All"

Feature selection method that iteratively selects the most important feature,
conditions on it (and all correlated features), then recalculates SHAP values
on the remaining features.

Based on: "When Shapley Values Break" by Alon Lanyado (2026)
https://towardsdatascience.com/when-shapley-values-break-a-guide-to-robust-model-explainability/
"""

import numpy as np
import shap


# Make function
def _is_tree_model(model):
    """Check if model is a tree-based model supported by TreeExplainer."""
    tree_types = (
        'LGBMRegressor', 'LGBMClassifier',
        'XGBRegressor', 'XGBClassifier',
        'GradientBoostingRegressor', 'GradientBoostingClassifier',
        'RandomForestRegressor', 'RandomForestClassifier',
        'DecisionTreeRegressor', 'DecisionTreeClassifier',
    )
    return type(model).__name__ in tree_types


def greedy_iterative_shapley(model, X_background, X_explain, feature_names=None,
                              n_steps=10, threshold=0.01, nsamples=200,
                              corr_threshold=0.8):
    """
    Greedy Iterative Shapley — "Winner Takes It All"

    At each step:
    1. Calculate SHAP values across X_explain, take mean |SHAP| per feature
    2. Select the feature with highest mean |SHAP|
    3. "Lock in" that feature AND all remaining features highly correlated
       with it (|corr| > corr_threshold) — this propagates the conditioning
       to the entire correlated group, so duplicates drop to ~0
    4. Repeat

    Auto-detects tree-based models and uses TreeExplainer (interventional mode)
    for speed and correctness. Falls back to KernelExplainer for other models.

    Parameters
    ----------
    model : fitted model
        Any model with a .predict() method. Tree-based models (LGBM, XGBoost,
        sklearn GradientBoosting, RandomForest) are auto-detected and use
        TreeExplainer for speed.
    X_background : np.ndarray
        Background data for SHAP marginalization. Use shap.sample(X, 50) for speed.
    X_explain : np.ndarray
        2D array of instances to explain. SHAP is computed across all rows,
        then aggregated via mean |SHAP| per feature.
    feature_names : list of str, optional
        Names for each feature. Defaults to F0, F1, ...
    n_steps : int, default=10
        Maximum number of features to select.
    threshold : float, default=0.01
        Stop when max mean |SHAP| across remaining features < threshold.
    nsamples : int, default=200
        Number of samples for KernelExplainer. Ignored for tree models.
    corr_threshold : float, default=0.8
        After selecting a winner, all remaining features with
        |correlation| > corr_threshold to the winner are also conditioned out.
        Set to 1.0 to disable (only condition the winner itself).

    Returns
    -------
    list of dict
        Each dict has keys: step, feature_idx, name, mean_abs_shap, n_conditioned.
    """
    n_feat = X_explain.shape[1]
    if feature_names is None:
        feature_names = [f'F{i}' for i in range(n_feat)]

    use_tree = _is_tree_model(model)
    if use_tree:
        print(f"  (Using TreeExplainer — interventional mode for {type(model).__name__})")
    print(f"  (Correlation conditioning threshold: {corr_threshold})")

    # Precompute correlation matrix on X_explain for conditioning propagation
    corr_matrix = np.corrcoef(X_explain.T)  # shape: (n_feat, n_feat)

    current_background = X_background.copy()
    selected = []
    remaining = set(range(n_feat))
    conditioned = set()

    for step in range(n_steps):
        if not remaining:
            break

        # Calculate SHAP values across all explain instances
        if use_tree:
            explainer = shap.TreeExplainer(model, data=current_background)
            sv = explainer.shap_values(X_explain)
        else:
            explainer = shap.KernelExplainer(model.predict, current_background)
            sv = explainer.shap_values(X_explain, nsamples=nsamples)

        sv = sv[0] if isinstance(sv, list) else sv  # shape: (n_explain, n_feat)

        # Mean |SHAP| per feature
        mean_abs_shap = np.mean(np.abs(sv), axis=0)

        # Find the remaining feature with highest mean |SHAP|
        best_idx = max(remaining, key=lambda i: mean_abs_shap[i])
        best_val = mean_abs_shap[best_idx]

        if best_val < threshold:
            print(f"  Step {step+1}: Stopping — max mean |SHAP| = {best_val:.4f} < threshold {threshold}")
            break

        # Find all remaining features correlated with the winner
        to_condition = {best_idx}
        for j in remaining:
            if j != best_idx and abs(corr_matrix[best_idx, j]) > corr_threshold:
                to_condition.add(j)

        # Condition ALL correlated features — set background to X_explain mean
        for idx in to_condition:
            current_background[:, idx] = X_explain[:, idx].mean()

        # Remove conditioned features from remaining
        remaining -= to_condition
        conditioned |= to_condition

        selected.append({
            'step': step + 1,
            'feature_idx': best_idx,
            'name': feature_names[best_idx],
            'mean_abs_shap': best_val,
            'n_conditioned': len(to_condition)
        })

        extra = f" (+ {len(to_condition)-1} correlated)" if len(to_condition) > 1 else ""
        print(f"  Step {step+1}: Selected {feature_names[best_idx]:>10s} | mean |SHAP| = {best_val:.4f}{extra}")

    return selected
