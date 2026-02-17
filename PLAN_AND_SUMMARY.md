# Shapley Value Break — Plan & Summary

## Source
[When Shapley Values Break: A Guide to Robust Model Explainability](https://towardsdatascience.com/when-shapley-values-break-a-guide-to-robust-model-explainability/) — Alon Lanyado, Towards Data Science (Jan 2026)

---

## Plan

### Context
Build a Jupyter notebook based on Alon Lanyado's article demonstrating how Shapley values become misleading with correlated/duplicated features and two fixes: **Grouped Shapley** and **Greedy Iterative Selection**. The author works at Medial EarlySign and competed in the CMS AI Health Outcomes Challenge.

### Key Research Finding
- The `medpython` package referenced in the article is an **internal package from Medial EarlySign** — NOT publicly available on PyPI. The notebook implements grouped Shapley and greedy iterative methods from scratch, faithful to the article's approach.
- For the CMS dataset, synthetic Medicare-like data was generated since the actual competition data is restricted.

### Notebook Structure

| Part | Topic | Type |
|------|-------|------|
| 1 | Introduction & Theory (Shapley formula, core problem) | Markdown |
| 2 | Toy Model — Independent Features (ground truth) | Code + Markdown |
| 3 | Breaking Shapley — Duplicated Features | Code + Markdown |
| 4 | Solution 1 — Direct Group Shapley (sum + exact methods) | Code + Markdown |
| 5 | Solution 2 — Greedy Iterative Selection ("Winner Takes It All") | Code + Markdown |
| 6 | Reusable `GroupedShapley` Class | Code + Markdown |
| 7 | Application to CMS Healthcare Data (XGBoost + full pipeline) | Code + Markdown |
| 8 | Summary & Key Takeaways | Markdown |

### Dependencies
```
pip install numpy pandas scikit-learn shap xgboost matplotlib seaborn
```

---

## Implementation Summary

### What Was Built

**File:** `shapley_values_break.ipynb` — 47 cells (13 markdown, 34 code)

### Part 1: Toy Model — Independent Features
- Generated 100 independent uniform random features with known weights (F0=10, F1=5, F2=3, F3=1, rest=0)
- Trained `LinearRegression` and verified SHAP values match theoretical ground truth: `phi_i = w_i * (x_i - E[x_i])`
- Scatter plot + bar chart comparison showing near-perfect match

### Part 2: Breaking Shapley — Duplicated Features
- Duplicated Feature 0 (the dominant feature, weight=10) 100 times → 200 total features
- Demonstrated that SHAP "fairly" splits credit across all 101 copies, diluting the true signal
- Visualization: Feature 0 no longer appears as dominant in top features
- This mirrors real-world patterns: rolling averages, lags, correlated lab tests

### Part 3: Direct Group Shapley
- **Method A (Sum within groups):** Sum individual SHAP values for features in the same group — simple, fast
- **Method B (Direct group-level Shapley):** From-scratch implementation treating each group as a single player in the coalition game, evaluating all 2^G coalitions with proper Shapley weighting
- Both methods correctly recover Feature 0's concept as the dominant driver
- Side-by-side comparison of Individual SHAP vs Summed Group vs Direct Group

### Part 4: Greedy Iterative Selection
- Implemented "Winner Takes It All" approach:
  1. Calculate SHAP values for all features
  2. Select the feature with highest |SHAP|
  3. Condition on it (replace its background values with instance value)
  4. Recalculate — duplicates now contribute ~0 (no new information)
  5. Repeat until contributions are negligible
- Works without predefined groups — automatically discovers redundancy

### Part 5: Reusable `GroupedShapley` Class
- Wraps all three methods in a single class:
  - `sum_group_shap()` — Method A
  - `direct_group_shapley()` — Method B
  - `greedy_iterative()` — feature-level or group-level greedy selection
- Standalone replacement for the proprietary `medpython` package

### Part 6: CMS Healthcare Data Application
- Generated synthetic Medicare-like data (2000 patients, 21 features) mimicking the CMS AI Health Outcomes Challenge:
  - **Blood Sugar Group:** hba1c, fasting_glucose, avg_glucose_3m, glucose_variability
  - **Kidney Function Group:** creatinine, bun, egfr, urine_albumin
  - **Blood Pressure Group:** systolic_bp, diastolic_bp, pulse_pressure, map_bp
  - **Healthcare Utilization Group:** num_prior_admissions, er_visits, outpatient_visits, total_cost_prior_year
  - **Others:** age, num_medications, height, weight, bmi
- Trained XGBoost classifier for hospital readmission prediction
- Showed correlation heatmap highlighting within-group correlations
- Compared Standard SHAP (diluted) vs Grouped SHAP (clear drivers) vs Greedy Iterative
- Individual patient explanation for a high-risk case

### Part 7: Summary Table

| Problem | Standard SHAP | Grouped SHAP | Greedy Iterative |
|---------|--------------|--------------|------------------|
| Duplicated features | Signal diluted evenly | Recovered by summing | Winner gets all credit |
| Correlated features | Credit split "fairly" | Grouped by domain knowledge | Auto-discovers redundancy |
| Computational cost | O(2^N) / approximated | O(2^G) direct, O(N) sum | O(K * SHAP) for K steps |
| Requires domain knowledge | No | Yes | No |

---

## References
- [When Shapley Values Break](https://towardsdatascience.com/when-shapley-values-break-a-guide-to-robust-model-explainability/) — Alon Lanyado (2026)
- [SHAP Library](https://github.com/shap/shap) — Lundberg & Lee
- [CMS AI Health Outcomes Challenge](https://www.closedloop.ai/cms-challenge/) — ClosedLoop.ai (winner, $1.6M)
- [Interpretable ML Book — Shapley Values](https://christophm.github.io/interpretable-ml-book/shapley.html) — Christoph Molnar
- Alon Lanyado — [Medial EarlySign](https://www.linkedin.com/in/lanyado/), CMS Challenge award winner
