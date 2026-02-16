# When Shapley Values Break

Exploration of how SHAP values break down under multicollinearity, and methods to fix them.

Based on [Alon Lanyado's article](https://towardsdatascience.com/when-shapley-values-break-a-guide-to-robust-model-explainability/) (2026).

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then open `shapley_values_break.ipynb` in Jupyter or VSCode and select the `.venv` kernel.

## Why SHAP Values Break Under Multicollinearity

SHAP values follow the **Symmetry axiom**: if two features contribute equally to every coalition, they receive equal credit. When features are duplicated or highly correlated, this "fair" splitting **dilutes** the signal.

Example: Feature 0 (weight=10) is the dominant driver. After duplicating it 100 times:

```
Before duplication:  Feature 0 SHAP = 2.51  (clearly dominant)
After duplication:   Feature 0 SHAP ≈ 0.02  (diluted across 101 copies)
```

A stakeholder looking at the top features would never identify Feature 0 as the true driver.

## Effect on Linear Regression vs LGBM

A key finding: **the dilution pattern is opposite** for each model type depending on whether duplicates are exact or noisy.

| | Linear Regression | LGBM |
|---|---|---|
| **Perfect duplicates** (corr = 1.0) | Diluted — OLS spreads weights equally (~0.099 each) | **Not diluted** — deterministic tie-breaking always picks F0 |
| **Noisy duplicates** (corr ≈ 0.7-0.99) | **Not diluted** — OLS identifies the clean signal | Diluted — different trees split on different copies |

**Why?**
- **Linear Regression + perfect dups**: the design matrix is rank-deficient, OLS pseudo-inverse spreads weight equally across all copies
- **Linear Regression + noisy dups**: noise makes duplicates imperfect predictors, OLS assigns full weight to the original
- **LGBM + perfect dups**: all copies have identical information gain, tree picks the first one (F0) deterministically
- **LGBM + noisy dups**: copies have slightly different split quality, different trees in the ensemble pick different copies via feature subsampling

### Greedy Iterative Results Across 4 Combinations

```
Config                       Step 1 (Winner)    Step 2             Step 3
--------------------------------------------------------------------------------
LinearReg + Noisy Dup        F0         2.674   F1         1.139   F2         0.732
LinearReg + Perfect Dup      F1         0.881   F2         0.504   F3         0.160
LGBM + Noisy Dup             F1         1.139   F2         0.704   F0_dup41   0.313
LGBM + Perfect Dup           F0         2.674   F1         1.147   F2         0.727
```

## How to Use `greedy_iterative_shapley`

The function lives in [`greedy_iterative.py`](greedy_iterative.py).

```python
from greedy_iterative import greedy_iterative_shapley

results = greedy_iterative_shapley(
    model,                    # fitted model (LinearRegression, LGBMRegressor, etc.)
    X_background,             # background data for SHAP (e.g., shap.sample(X, 50))
    X_explain,                # instances to explain (e.g., 10% of data)
    feature_names=names,      # optional list of feature names
    n_steps=8,                # max features to select
    threshold=0.01,           # stop when max mean |SHAP| < threshold
    corr_threshold=0.8,       # auto-condition features with |corr| > this
    nsamples=200              # KernelExplainer samples (ignored for tree models)
)

# Results: list of dicts
for r in results:
    print(f"Step {r['step']}: {r['name']} | mean |SHAP| = {r['mean_abs_shap']:.4f} "
          f"({r['n_conditioned']} features conditioned)")
```

### Key Parameters

| Parameter | Description |
|-----------|-------------|
| `X_explain` | 2D array — SHAP is computed across all rows, aggregated via mean \|SHAP\| per feature |
| `corr_threshold` | After selecting a winner, all remaining features with \|correlation\| > this are also conditioned out. This is critical for linear models where features are additive (conditioning one feature has zero effect on another's SHAP). Set to `1.0` to disable. |
| `n_steps` | Maximum number of features to select. Each step recalculates SHAP on remaining features. |

### Auto-Detection

The function auto-detects tree-based models (LGBM, XGBoost, GradientBoosting, RandomForest) and uses `TreeExplainer` in interventional mode. For other models it falls back to `KernelExplainer`.

## Feature Selection Pipeline: Healthcare Example

Using greedy iterative as a feature selection method on a simulated healthcare dataset (21 features, LGBM classifier predicting hospital readmission):

### Step 1: Greedy Iterative selects 10 features from 21

```
Step 1: hba1c              | mean |SHAP| = 0.537 (+ 2 correlated: fasting_glucose, avg_glucose_3m)
Step 2: creatinine         | mean |SHAP| = 0.454 (+ 2 correlated: bun, egfr)
Step 3: num_prior_admissions | mean |SHAP| = 0.351 (+ 2 correlated)
Step 4: age                | mean |SHAP| = 0.276
...
```

### Step 2: Retrain model on selected features only

```
Original model AUC:   0.772  (21 features)
Retrained model AUC:  0.757  (10 features)
AUC difference:       -0.015
```

Nearly identical performance with less than half the features.

### Key Insight: `fasting_glucose`

In Standard SHAP, `fasting_glucose` ranks **#3** in feature importance. But in the retrained model, it's **completely absent**. Why? It's highly correlated with `hba1c` (corr ≈ 0.97). Greedy iterative selected `hba1c` first, conditioning out `fasting_glucose`. The retrained model without `fasting_glucose` loses almost no predictive power — it was redundant all along.

This is exactly the kind of insight that standard SHAP hides: a clinician seeing `fasting_glucose` at rank #3 might think it's independently important, when it's really just echoing the same signal as `hba1c`.

## File Structure

```
shapley_value_break/
├── greedy_iterative.py          # Standalone greedy iterative SHAP function
├── shapley_values_break.ipynb   # Main notebook with experiments
├── requirements.txt             # Pinned dependencies
├── pyproject.toml               # Ruff linting config
├── .gitignore
└── README.md
```

## Notebook Structure

| Part | Description |
|------|-------------|
| Part 1 | Toy model with 5 independent features — ground truth SHAP |
| Part 2 | Duplicated features (noisy + perfect) — demonstrates dilution on LR and LGBM |
| Part 3 | Solution 1: Grouped Shapley (sum within groups + direct group Shapley) |
| Part 4 | Solution 2: Greedy Iterative with correlation conditioning |
| Part 5 | Healthcare application — full pipeline with feature selection and retraining |
| Part 6 | Summary and key takeaways |

## References

- [When Shapley Values Break](https://towardsdatascience.com/when-shapley-values-break-a-guide-to-robust-model-explainability/) — Alon Lanyado, Towards Data Science (2026)
- [SHAP Library](https://github.com/shap/shap) — Lundberg & Lee
- [CMS AI Health Outcomes Challenge](https://www.closedloop.ai/cms-challenge/)
- [Interpretable ML Book — Shapley Values](https://christophm.github.io/interpretable-ml-book/shapley.html) — Christoph Molnar
