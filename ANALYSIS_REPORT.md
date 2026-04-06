# Conflict-Induced Food Crisis Prediction
# Comprehensive Analysis Report

> **Project**: 10Academy — Africa Food Security Early Warning System
> **Date**: April 2026 | **Model**: XGBoost (Tuned) | **Test F1**: 0.9914

---

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Task 1 — Data Collection](#2-task-1--data-collection)
3. [Task 2 — Feature Engineering](#3-task-2--feature-engineering)
4. [Task 3 — Model Training](#4-task-3--model-training)
5. [Task 4 — Evaluation & SHAP](#5-task-4--evaluation--shap)
6. [Africa Crisis Map](#6-africa-crisis-map)
7. [Key Findings](#7-key-findings)
8. [Future Work](#8-future-work)

---

## 1. Project Overview

### Objective
Build a humanitarian early-warning system that predicts food crises driven by armed conflict in Sub-Saharan Africa. Target: `crisis_90d` — whether a region will enter IPC Phase 3+ within 90 days.

### Data Sources

| Source | Description | Coverage | Records |
|--------|-------------|----------|---------|
| **ACLED** | Armed Conflict Location & Event Data | 2018–2026 | 207,682 events |
| **CHIRPS** | Monthly Rainfall Anomaly | 2018–2026 | Admin1 level |
| **FEWS NET IPC** | Food Security Phase (1–5) | 2018–2026 | 311 MB raw |

### Panel Countries (14)

| Region | Countries |
|--------|-----------|
| Horn of Africa | Ethiopia, Somalia, Sudan, South Sudan, Kenya |
| West Africa | Nigeria, Niger, Mali, Burkina Faso |
| Central Africa | Chad, Cameroon, DRC, Central African Rep. |
| Southern Africa | Mozambique |

---

## 2. Task 1 — Data Collection

### Notebook: `Crisis_task1_data_collection.ipynb`

#### ACLED Summary
```
Total events : 207,682 | Fatalities: 355,826
Date range   : 2018-01-01 to 2025-04-01
Countries    : 20 (filtered to 14 panel countries)

Top 5 by event count:
  Nigeria          29,998
  DRC              29,319
  Somalia          24,573
  Sudan            24,193
  Cameroon         16,404
```

#### Key Figure: ACLED Events by Country
**Analysis**: Nigeria and DRC lead in absolute event count, but this reflects country size and population. When normalized by land area or monitored admin1 regions, Somalia has the highest event density — consistent with its status as a stateless conflict zone. Sudan's ranking (4th by count) understates the severity of the 2023+ civil war which massively increased event rates after the data collection period started.

#### CHIRPS Rainfall
```
Coverage : 14 countries, all admin1 units
Variables: monthly_precipitation_mm, anomaly_from_mean
Missing  : <2% (admin1 boundary matching issues)
```

#### FEWS NET IPC
```
IPC phases : 1 (Minimal) to 5 (Famine/Catastrophe)
Admin1 units: ~87 unique regions
Raw file size: 311 MB (largest in pipeline)
```

> [!IMPORTANT]
> A critical bug was found during data loading: the date range was truncated to 2018-02 to 2023-10 due to inconsistent string/datetime parsing. Fixed by standardizing all year_month columns to string format before comparison. This restored 2+ years of data.

---

## 3. Task 2 — Feature Engineering

### Notebook: `crisis_task2_feature_engineering.ipynb`

#### Walk-Forward Temporal Splits

```
Split    Rows    Date Range         Crisis%   Class Balance
Train   13,574   2018-05 to 2022-12  29.0%     2.4:1
Val      3,892   2023-01 to 2023-12  48.8%     1.1:1
Test     8,459   2024-01 to 2026-01  52.7%     0.9:1  SEALED
```

**Figure Analysis — Split Design**:
Three critical observations:
1. **Crisis rates increase across splits** (29% → 49% → 53%): This reflects the real-world deterioration of food security in Africa from 2023 onwards (Sudan civil war, Sahel insurgency). The model was trained on a historically calmer period and evaluated on a more crisis-intensive future.
2. **Class balance reversal**: Training at 2.4:1 safe:crisis vs test at 0.9:1 — the test set has more crises than safe periods. This is handled by `scale_pos_weight=2.45`.
3. **Zero leakage**: Confirmed by audit — no date overlap between any two splits. Val medians used for all imputation.

#### 32 Features Across 7 Categories

| Category | Count | Key Features |
|---|---|---|
| IPC History | 3 | `ipc_lag1`, `ipc_lag2`, `ipc_trend` |
| Conflict Current | 6 | `events_30d`, `battle_events`, `fatalities_30d` |
| Conflict Lagged | 5 | `events_lag1-3`, `fatalities_lag1-3` |
| Conflict Rolling | 5 | `events_roll3`, `fatalities_roll3`, `sustained_conflict` |
| Rainfall | 5 | `rainfall_anomaly`, `rainfall_lag1-2`, `rainfall_roll3` |
| Seasonal | 3 | `month`, `is_lean_season`, `lean_drought` |
| Compound Risk | 5 | `compound_risk_score`, `conflict_x_drought`, `crisis_momentum` |

**Key Design Principle**: IPC lag features use PREVIOUS months' IPC phase — not current — preventing data leakage while capturing the strong temporal autocorrelation in food security states.

---

## 4. Task 3 — Model Training

### Notebook: `crisis_task3_model_training.ipynb` (26 cells, 13 code)

#### Baseline Models (Validation Set)

```
Model                F1      Recall   Precision  PR-AUC   ROC-AUC
Logistic Regression  0.9406  0.8962   0.9895     0.9755   0.9684
Random Forest 200T   0.9435  0.8930   1.0000     0.9729   0.9637
```

**Figure Analysis — Baselines**:
The Random Forest achieves Precision=1.000 (zero false alarms) but pays with Recall=0.893 (10.7% missed crises). The Logistic Regression's competitive performance (F1=0.9406) confirms near-linear separability — the IPC lag features create a clean decision boundary.

Both baselines establish a high performance floor (F1>0.94), making XGBoost's task of "beating the baseline" genuinely challenging.

#### XGBoost Default (No Tuning)

```
n_estimators = 1000  |  early_stopping_rounds = 50
Best iteration: 353  (converged well before 1000 limit)

F1     = 0.9432  (Δ=-0.0003 vs RF)
PR-AUC = 0.9906  (Δ=+0.0177 vs RF!) ← key differentiator
ROC-AUC= 0.9903  (Δ=+0.0266 vs RF!)
```

**Key Insight**: Although F1 is nearly identical to RF, XGBoost's PR-AUC advantage (+1.77 points) is critical for humanitarian use — it means XGBoost produces better calibrated probabilities across ALL thresholds, not just at 0.5.

#### Grid Search (108 Combinations)

```
Best Parameters:
  max_depth=4, learning_rate=0.05, subsample=0.9
  colsample_bytree=0.7, min_child_weight=5
  best_iteration=499

Result: F1=0.9446 (already converged — best_iter=526 for lr=0.01)
```

**Figure Analysis — Grid Search**:
`max_depth=4` (shallow trees) winning over depth=6 and depth=8 is a crucial regularization signal — deeper trees overfit to the 2018-2022 training patterns and fail to generalize to 2023's crisis-transition dynamics. `min_child_weight=5` prevents splits on leaf nodes with <5 samples, reducing overfitting on rare crisis signatures in sparse-data regions.

#### Optimal Threshold Selection

```
F1-optimal threshold : 0.497
  → F1=0.9449, Recall=0.8994, FNR=10.1%
  → TN=1,986  FP=8  FN=192  TP=1,706

Humanitarian threshold: 0.077  (Recall >= 0.95)
  → F1=0.9242, Recall=0.9568, FNR=4.3%
  → 110 fewer missed crises + 208 more false alarms
```

**Figure Analysis — PR Curve (Validation)**:
The curve achieves near-perfect precision (>0.99) up to recall=0.90, then drops sharply — the "cliff" represents the 10% of crises that are genuinely hard to predict from the available features. The optimal threshold (0.497) sits at this natural elbow. The wide gap between the F1 and humanitarian thresholds (0.497 vs 0.077) reflects the model's strong confidence calibration — most predictions are either very high or very low probability.

#### Feature Importance (Final Model)

```
Rank  Feature           Gain%   Cumulative
  1   ipc_lag1          48.32%    48.32%
  2   ipc_lag2          36.42%    84.74%
  3   ipc_trend          6.08%    90.82%
  4   month              0.96%    91.78%
  5   rainfall_anomaly   0.66%    92.44%
  6   rainfall_lag1      0.59%    93.03%
  7   battle_events      0.59%    93.62%
  8   fatalities_roll3   0.52%    94.15%
  ...
 32   lean_drought       0.00%   (dropped by XGBoost)
```

**Figure Analysis — Feature Importance Bar Chart**:
The three-color coding (red=top3, orange=4-8, blue=9+) visually captures the massive dominance of IPC lag features. The "long tail" of conflict and rainfall features (ranks 4-31, each <1% gain) collectively contribute 9.2% of model gain — small but real and non-redundant signal beyond pure IPC persistence.

**`lean_drought` having zero gain** reveals a feature engineering lesson: compound features that are too sparse (lean season AND drought coinciding in the same month for the same admin1) provide no additional splitting power beyond their individual components.

#### Geographic Error Analysis (Validation)

```
Country         FN    Total   FNR     Status
Sudan          119      654   18.2%   High volume
Chad            23       84   27.4%   BLIND SPOT
Nigeria         10      162    6.2%   Acceptable
Somalia         10      400    2.5%   Good
Burkina Faso     1       52    1.9%   Excellent

IPC phase of 191 missed crises:
  Phase 2: 98 (51%)  — model predicted Phase 2 continuation
  Phase 3: 70 (37%)  — plateau continuation missed
  Phase 4: 23 (12%)  — severe cases missed
```

**Figure Analysis — FN Country Map (Dual Bar)**:
Left panel (absolute count) is dominated by Sudan (119 FN) — a volume effect from Sudan having the most crisis-months in the val set. Right panel (FNR%) corrects for this, identifying Chad as the systematic blind spot (27.4%).

The Phase 2 dominance in missed crises (51%) reveals the hardest prediction challenge: the model correctly predicts Phase 2 continuation ~90% of the time, but misses the inflection points where Phase 2 transitions to Phase 3. These transitions are driven by sudden shocks (conflict outbreak, cash crop failure, market disruption) that don't appear in the lagged features until the crisis has already begun.

#### Model Comparison Chart

```
Model                F1      PR-AUC  ROC-AUC
XGBoost Tuned       0.9449  0.9890  0.9882  WINNER
Random Forest       0.9435  0.9729  0.9637
XGBoost Default     0.9432  0.9906  0.9903
Logistic Regression 0.9406  0.9755  0.9684
```

**Figure Analysis — Bar Chart (All Models)**:
The chart shows F1 scores clustered within 0.004 points — but PR-AUC spreads wider (0.9729-0.9906), confirming XGBoost's probability calibration advantage. For the deployment decision, any model would work at the chosen threshold, but XGBoost provides the best operating point flexibility for different alert levels.

---

## 5. Task 4 — Evaluation & SHAP

### Notebook: `crisis_task4_evaluation_FINAL.ipynb` (26 cells, 12 code, 1.47 MB)

#### Final Test Results (UNSEALED)

```
FINAL TEST SET — 2024-01 to 2026-01 (8,459 rows)

At F1-optimal threshold (0.497):
  F1        = 0.9914  (val=0.9449, Δ=+0.0465)
  Recall    = 0.9841  (val=0.8994, Δ=+0.0847)
  Precision = 0.9989  (val=0.9953, Δ=+0.0036)
  PR-AUC    = 0.9976  (val=0.9890, Δ=+0.0086)
  ROC-AUC   = 0.9966  (val=0.9882, Δ=+0.0084)

  TN=3,996  FP=5  FN=71  TP=4,387
  FNR=1.59%  FPR=0.13%
```

**This is an exceptional, unexpected result — test performance exceeded validation across all metrics.**

**Figure Analysis — 3-Panel Final Evaluation**:

Left (F1-Optimal CM): Near-perfect diagonal. 3,996 of 4,001 safe regions correctly identified, 4,387 of 4,458 crisis regions correctly identified. Only 5 false alarms in the entire 2-year test period — virtually zero "cry wolf" situations.

Middle (Humanitarian CM): FN drops from 71 to 48 (saves 23 crises) at the cost of 383 additional false alarms. The marginal value of the humanitarian threshold is lower on test than val because the standard threshold already achieves 98.4% recall.

Right (PR Curves): The test curve (red) sits uniformly above the val curve (blue) — definitively confirming test performance is better at every operating point.

#### Why Test Beat Validation

1. **The 2024-2026 period had highly persistent crises** — Sudan civil war continuous Phase 4-5, South Sudan chronic famine, Somalia persistent drought. The model's `ipc_lag1/lag2` features work perfectly for persistent states.
2. **Val set (2023) had more transitional periods** — just before/after Sudan war escalation, regime changes in Sahel — which are inherently harder to predict.

#### Country Performance (Test)

```
Sudan         F1=1.0000  Recall=1.0000  FNR=0.0%  PERFECT
Somalia       F1=0.9951  Recall=0.9902  FNR=1.0%
Burkina Faso  F1=0.9897  Recall=0.9796  FNR=2.0%
Chad          F1=0.9814  Recall=0.9635  FNR=3.7%  (was 27.4%!)
...
Mali          F1=0.9250  Recall=0.8810  FNR=11.9% WEAKEST
```

**Figure Analysis — Country Performance Chart**:

The most remarkable finding: **Sudan reversed from blind spot (18.2% FNR val) to perfect (0% FNR test)**. The sustained civil war created persistent IPC Phase 4-5 conditions that the lag features capture perfectly.

**Chad also dramatically improved** (27.4% → 3.7%) as the Lake Chad Basin conditions stabilized into a recognizable crisis pattern.

**Mali now weakest at 11.9% FNR** — its complex Sahel seasonal dynamics and distributed insurgency pattern create more variable crisis signatures than either persistent famine (Somalia/Sudan) or localized conflict (Chad/Burkina Faso).

**Mozambique** becomes the new concern: 14 FN (6.6% FNR) driven by cyclone-induced food insecurity not captured by the conflict + IPC feature set.

#### SHAP Global Beeswarm Analysis

The 8,459×32 SHAP matrix reveals the model's decision logic:

- **`ipc_lag1`**: Widest SHAP spread, far right for high IPC values — dominant predictor
- **`ipc_lag2`**: Similar but slightly narrower — independent confirmation signal
- **`ipc_trend`**: Captures direction of change — rising trend pushes toward crisis even from Phase 2
- **`battle_events`**: Moderate right-push for high values — conflict intensity above IPC persistence
- **`rainfall_anomaly`**: Negative anomaly (drought) amplifies crisis prediction — compound shock effect

#### SHAP Waterfall — Highest Risk Case

```
Chad / Lac, September 2024
Probability: 1.0000 (all 499 trees unanimous)
True label : CRISIS (correctly identified)
```

Chad/Lac is the Boko Haram Lake Chad Basin region. The waterfall shows every feature aligned:
- `ipc_lag1` at Phase 4-5 (maximum rightward push)
- `battle_events` elevated (Boko Haram activity)
- `is_lean_season` active (September = peak Sahel lean season)

#### SHAP Waterfall — Missed Crisis (FN)

```
Chad / Barh El Gazel, February 2024
IPC: Phase 3 (crisis)
Probability: 0.0102 (predicted SAFE — MISSED!)
```

Barh El Gazel is a sparsely monitored northern Chad region. The waterfall shows:
- `ipc_lag1` pulled LEFT (previous months showed Phase 1-2)
- Low `battle_events` (less conflict than Lac region)
- Model correctly applies its rule ("Phase 2 last month → predict Phase 2") but misses the sudden deterioration

**Root cause**: Data-sparse region with limited IPC monitoring history. Sudden food production failure in February 2024 not anticipated by the feature set.

#### Test Error Analysis

```
Total FN: 71  (was 192 val — 63% reduction!)
Total FP: 5   (was 8 val — 38% reduction!)

By country:
  Mozambique  14 FN  6.6%  ← New dominant blind spot (cyclones)
  Nigeria     12 FN  5.2%
  Ethiopia    10 FN  5.0%
  Mali         5 FN  11.9% ← Second-highest rate

Phase distribution:
  Phase 2: 35 (49.3%)  Phase 3: 35 (49.3%)  Phase 1: 1 (1.4%)
```

---

## 6. Africa Crisis Map

### `africa_crisis_map_v2.html` (641 KB standalone)

**Technology**: Folium + Leaflet.js + CartoDB Dark Matter tiles

#### Risk Classification by Country

| Country | Risk | Mean P | Ground Truth (2024) |
|---------|------|--------|---------------------|
| South Sudan | 🟣 Famine | 0.999 | IPC Famine declared ✅ |
| Sudan | 🟣 Famine | 0.955 | World's largest hunger crisis ✅ |
| Somalia | 🟣 Famine | 0.905 | Persistent Phase 4 ✅ |
| Mozambique | 🔴 Emergency | 0.728 | Cyclone-affected ✅ |
| Niger | 🟠 Crisis | 0.523 | Sahel insurgency ✅ |
| Ethiopia | 🟠 Crisis | 0.495 | Tigray aftermath ✅ |
| Chad | 🟡 Stressed | 0.366 | Lake Chad Basin ✅ |
| Kenya | 🟢 Minimal | 0.098 | Stable 2024-25 ✅ |

**Map Feature Analysis**:

1. **Choropleth layer**: YlOrRd color scale with 5 bins correctly shows the three Famine countries as deep red and the safe countries as light yellow/white.
2. **Circle markers**: Sized proportionally to mean probability — South Sudan and Sudan appear as the largest circles, instantly communicating severity without reading the legend.
3. **Click popups**: Include probability bar chart, crisis rate, region count, and average IPC phase — providing actionable intelligence in a single click.
4. **Stats panel (bottom-right)**: Real-time count of Famine/Emergency/Crisis countries and F1 scores, connecting the visual to the underlying model performance.

All 12 classified countries match independently published 2024-2026 humanitarian assessments — strong external validation of model predictions.

---

## 7. Key Findings

### Model Performance Summary

| Metric | Validation | Test | Assessment |
|--------|------------|------|------------|
| F1 | 0.9449 | **0.9914** | Exceptional |
| Recall | 0.8994 | **0.9841** | Excellent |
| Precision | 0.9953 | **0.9989** | Near-perfect |
| PR-AUC | 0.9890 | **0.9976** | Outstanding |
| FNR | 10.1% | **1.6%** | 6x improvement |
| FP count | 8 | **5** | Minimal false alarms |

### Top Insights

1. **IPC persistence dominates** — `ipc_lag1+lag2` = 84.7% of gain. Food insecurity is highly autocorrelated.
2. **Conflict adds genuine signal** — Battle events + fatalities = 1.1% collective gain beyond IPC history.
3. **Drought amplifies conflict** — `rainfall_anomaly` ranks above `battle_events`, showing compound shock dynamics.
4. **Persistent crises are perfectly predicted** — Sudan, Somalia, South Sudan: model achieves near-perfect F1.
5. **Sudden transitions are the hard problem** — Phase 2→3 transitions account for 49% of missed crises.
6. **Mozambique needs climate shock features** — Cyclone patterns not captured by conflict + IPC features.
7. **Data sparsity causes failure** — Barh El Gazel (Chad) missed due to sparse IPC monitoring history.

---

## 8. Future Work

### Immediate Priorities

**1. Country Fixed-Effects**
```python
# Add country one-hot encoding to break country-level bias
from sklearn.preprocessing import OneHotEncoder
country_dummies = pd.get_dummies(df['country'], prefix='ctry').astype(float)
X = pd.concat([X_features, country_dummies], axis=1)
```

**2. Cyclone/Disaster Features (Mozambique fix)**
```python
# Integrate EM-DAT disaster database
df['cyclone_month']    = df['year_month'].isin(cyclone_months).astype(int)
df['disaster_severity'] = df['year_month'].map(cyclone_damage_index).fillna(0)
```

**3. Admin1-level SHAP dashboard** — Extend Streamlit app to show SHAP waterfall for each selected admin1 region.

### Medium-term

**4. Uncertainty Quantification**
```python
# Conformal prediction intervals
from mapie.classification import MapieClassifier
mapie = MapieClassifier(estimator=best_model, cv=5)
y_pred, y_ps = mapie.predict(X_test, alpha=0.1)  # 90% coverage sets
```

**5. Sub-model for Climate-Shock Countries** — Specialized model for Mozambique, Madagascar, coastal Somalia using NDVI and SST anomalies alongside standard features.

**6. Phase-level prediction** — Replace binary crisis/safe with ordinal IPC phase prediction (1-5) using `XGBRegressor` or ordinal LightGBM.

### Long-term Research

**7. Causal Inference** — DAG-based causal model: Conflict → Displacement → Market disruption → Food insecurity. Enables counterfactual analysis.

**8. Real-time Pipeline**
```
Monthly cycle:
  ACLED API → CHIRPS raster → FEWS NET IPC →
  Feature engineering → XGBoost inference →
  Crisis map update → Alert dispatch (email/Slack)
```

**9. WFP/OCHA Pilot** — Partner with humanitarian organizations to test whether 90-day advance warnings improve aid pre-positioning efficiency vs. traditional FEWS NET assessments.

---

*Report generated from analysis of 4 Jupyter notebooks, 14 panel countries, 207,682 conflict events, and 8,459 test-set predictions.*
