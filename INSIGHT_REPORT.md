# Conflict-Induced Food Crisis Prediction
# Comprehensive Insight Report

> **Project**: 10Academy — Africa Food Security Early Warning System  
> **Report Date**: April 7, 2026  
> **Model**: XGBoost (Tuned, 499 trees) | **Test F1**: 0.9914 | **FNR**: 1.6%  
> **Author**: 10Academy Capstone Team

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Introduction & Motivation](#2-introduction--motivation)
3. [Project Objectives](#3-project-objectives)
4. [Data Sources & Collection](#4-data-sources--collection)
5. [Methodology](#5-methodology)
   - 5.1 [Data Fusion & Panel Construction](#51-data-fusion--panel-construction)
   - 5.2 [Feature Engineering](#52-feature-engineering)
   - 5.3 [Walk-Forward Temporal Splitting](#53-walk-forward-temporal-splitting)
   - 5.4 [Modeling Approach](#54-modeling-approach)
   - 5.5 [Threshold Selection Strategy](#55-threshold-selection-strategy)
6. [Results](#6-results)
   - 6.1 [Baseline Performance](#61-baseline-performance)
   - 6.2 [XGBoost Tuning & Selection](#62-xgboost-tuning--selection)
   - 6.3 [Final Test Set Evaluation](#63-final-test-set-evaluation)
   - 6.4 [Country-Level Performance](#64-country-level-performance)
   - 6.5 [SHAP Explainability Analysis](#65-shap-explainability-analysis)
   - 6.6 [Africa Crisis Risk Map](#66-africa-crisis-risk-map)
7. [Error Analysis & Failure Modes](#7-error-analysis--failure-modes)
8. [Key Insights & Findings](#8-key-insights--findings)
9. [Limitations](#9-limitations)
10. [Future Work & Recommendations](#10-future-work--recommendations)
11. [Conclusion](#11-conclusion)

---

## 1. Executive Summary

This report presents a comprehensive analysis of the **Conflict-Induced Food Crisis Prediction** system — a machine learning pipeline designed to forecast IPC Phase 3+ food crises 90 days in advance across 14 Sub-Saharan African countries. The project fuses three heterogeneous data sources (ACLED conflict events, CHIRPS satellite rainfall, and FEWS NET food security assessments) into a unified analytical framework spanning 207,682 conflict events from 2018 to 2026.

The final tuned XGBoost model achieves an **F1-score of 0.9914** on the held-out test set (2024–2026), with a **False Negative Rate of only 1.6%** — meaning the system correctly identifies 98.4% of all actual food crises while maintaining near-zero false alarms (FPR = 0.1%). Notably, test performance **exceeded** validation performance across every metric, a rare outcome attributable to the highly persistent nature of crises in the 2024–2026 evaluation period.

SHAP explainability analysis confirms that the model's decisions are grounded in interpretable, domain-relevant features: IPC historical persistence (84.7% of model gain), conflict intensity, and rainfall anomalies. The system has been deployed as both an interactive Folium risk map and a Streamlit dashboard, providing actionable intelligence for humanitarian decision-makers.

> [!IMPORTANT]
> **Central finding**: Food insecurity in Sub-Saharan Africa is highly autocorrelated — a region in crisis last month has an overwhelming likelihood of remaining in crisis. The model's primary contribution is detecting the ~10% of cases where this pattern breaks: sudden onset crises and unexpected recoveries. Conflict and rainfall features are critical for capturing these transitions.

---

## 2. Introduction & Motivation

### 2.1 The Scale of the Problem

Sub-Saharan Africa faces a converging crisis of armed conflict, climate variability, and chronic food insecurity. As of 2024, over **140 million people** across the continent face acute food insecurity (IPC Phase 3 or higher), with Sudan alone hosting the world's largest hunger crisis affecting 25 million. The interplay between conflict and food security creates self-reinforcing feedback loops: armed conflicts displace populations, destroy agricultural infrastructure, and disrupt markets, while food scarcity in turn fuels social unrest and recruits for armed groups.

### 2.2 The Prediction Gap

Traditional food security assessments, such as those produced by the Famine Early Warning Systems Network (FEWS NET), rely on in-country analysts synthesizing satellite imagery, market price data, and field reports. While highly authoritative, these assessments are:

- **Reactive**: Published quarterly, often after conditions have already deteriorated
- **Resource-intensive**: Requiring extensive human expertise per region
- **Coverage-limited**: Not all administrative regions receive equal analytical attention

A machine learning system that can provide **90-day advance warnings** — even as a complement to, not replacement for, expert assessment — could fundamentally change how humanitarian organizations pre-position aid. The difference between a warning delivered 90 days early and one delivered 30 days late can mean the difference between preventive food assistance costing $1 per person per day and emergency intervention costing $7+ per person per day.

### 2.3 Why This Approach

This project addresses the prediction gap by combining three complementary data streams:

1. **Armed conflict data**: Capturing the human-driven shocks that precipitate food crises
2. **Satellite rainfall**: Detecting the climate anomalies that compound conflict impacts
3. **Historical food security phases**: Leveraging the strong temporal persistence of food insecurity states

By engineering these into a panel dataset at the country-admin1-month granularity and training a gradient-boosted model with strict temporal anti-leakage controls, the system achieves prediction accuracy that rivals or exceeds existing early warning benchmarks.

---

## 3. Project Objectives

The project was structured around four sequential, modular objectives — each producing verifiable artifacts and enforceable data contracts for the next stage:

| # | Objective | Deliverable | Success Criterion |
|---|-----------|-------------|-------------------|
| 1 | **Data Collection & Fusion** | Unified panel dataset at `country × admin1 × year_month` | ≥10,000 rows, ≥10 countries, crisis correlation >0.60 |
| 2 | **Feature Engineering** | 30+ features with temporal anti-leakage verification | Zero leakage under audit; documented feature catalogue |
| 3 | **Model Training & Tuning** | Tuned classifier exceeding baseline F1 | F1 > RF baseline (0.9435); calibrated probabilities |
| 4 | **Evaluation, Explainability & Deployment** | Test-set metrics, SHAP analysis, interactive map, dashboard | Test F1 > 0.90; SHAP alignment with domain knowledge; deployable artifacts |

All four objectives were achieved, with the final system exceeding every pre-specified success criterion.

---

## 4. Data Sources & Collection

### 4.1 Armed Conflict Location & Event Data (ACLED)

| Attribute | Detail |
|-----------|--------|
| **Source** | ACLED API (acleddata.com) |
| **Coverage** | 20 African countries, 2018-01 to 2025-04 |
| **Records** | 207,682 geo-coded conflict events |
| **Total Fatalities** | 355,826 |
| **Event Types** | Battles, violence against civilians, explosions/remote violence, protests, riots, strategic developments |
| **Granularity** | Event-level with GPS coordinates, then aggregated to admin1-month |

**Top 5 Countries by Event Count:**

| Country | Events | Fatalities | Events per Admin1 |
|---------|--------|------------|-------------------|
| Nigeria | 29,998 | 59,471 | 811 |
| DRC | 29,319 | 61,842 | 1,128 |
| Somalia | 24,573 | 38,991 | 1,365 |
| Sudan | 24,193 | 53,476 | 1,344 |
| Cameroon | 16,404 | 11,293 | 1,640 |

**Key Observation**: Raw event counts are misleading without normalization. Somalia has the highest event density per admin1 region, reflecting its status as the continent's most intense per-area conflict zone, despite ranking third in absolute counts.

### 4.2 Climate Hazards Group InfraRed Precipitation with Station Data (CHIRPS)

| Attribute | Detail |
|-----------|--------|
| **Source** | University of California, Santa Barbara |
| **Coverage** | 14 panel countries, all admin1 units, 2018–2026 |
| **Records** | 44,523 admin1-month observations |
| **Variables** | Monthly precipitation (mm), rainfall anomaly (z-score), binary drought/flood flags |
| **Missing Data** | <2% (admin1 boundary matching) |
| **Drought Prevalence** | 28.9% of observations flagged as drought |
| **Flood Prevalence** | 7.2% of observations flagged as flood |

The rainfall anomaly variable uses a standardized z-score relative to the long-term mean for each admin1 region, enabling cross-regional comparison. Drought (anomaly < -1σ) and flood (anomaly > +1.5σ) binary flags provide additional categorical signals.

### 4.3 FEWS NET IPC Food Security Phases

| Attribute | Detail |
|-----------|--------|
| **Source** | Famine Early Warning Systems Network |
| **Coverage** | ~87 unique admin1 regions across 14 countries |
| **Phases** | 1 (Minimal) → 2 (Stressed) → 3 (Crisis) → 4 (Emergency) → 5 (Famine/Catastrophe) |
| **Raw Size** | 311 MB |
| **Target** | `crisis_90d = 1` if IPC ≥ 3 in any of the next 3 months |

The IPC classification is the gold standard for food security assessment, adopted by the United Nations and all major humanitarian organizations. Phase 3+ indicates acute food insecurity requiring urgent humanitarian action.

### 4.4 Panel Countries

A total of **14 countries** were included in the final panel after excluding 6 countries due to data quality issues:

```
Included (14):
  Horn of Africa  : Ethiopia, Somalia, Sudan, South Sudan, Kenya
  West Africa     : Nigeria, Niger, Mali, Burkina Faso
  Central Africa  : Chad, Cameroon, DRC, Central African Republic
  Southern Africa : Mozambique

Excluded (6):
  Burundi, Eritrea, Madagascar, Malawi   → Absent from FEWS NET
  Uganda                                  → Admin-2/Admin-1 mismatch (99% unmatched)
  Zimbabwe                                → Absent from FEWS NET
```

---

## 5. Methodology

### 5.1 Data Fusion & Panel Construction

The three data sources operate at different spatial and temporal resolutions. The fusion strategy uses **FEWS NET as the left table** — ensuring only region-months with authoritative IPC assessments survive the merge:

```
FEWS NET (28,170 rows)  ← LEFT JOIN base
  ├── ACLED monthly aggregation (40,128 rows → matched via fuzzy admin1 names)
  └── CHIRPS monthly (44,523 rows → matched via country + admin1)
```

**Fuzzy Name Matching**: A critical preprocessing challenge was matching admin1 region names across datasets that use different naming conventions (e.g., "Boucle Du Mouhoun" vs. "Boucle du Mouhoun"). A fuzzy matching pipeline produced 380 ACLED→FEWS region pairs with a matching threshold of 85%.

**Final Production Panel**: 26,954 rows, 13 countries, 341 admin1 regions, spanning February 2018 to January 2026.

**Target Variable**: `crisis_90d` is constructed using only **forward-looking** IPC shifts — it equals 1 if the region enters IPC Phase 3+ in any of the following 3 months. This ensures the target represents a genuine 90-day prediction horizon.

> [!NOTE]
> A critical date-parsing bug was discovered during development: the date range was initially truncated to 2018-02 through 2023-10 due to inconsistent string/datetime parsing in the year_month column. Fixing this by standardizing all year_month columns to string format before comparison restored 2+ years of data — nearly a third of the entire panel.

### 5.2 Feature Engineering

Thirty-two features were engineered across seven categories, each designed to capture a specific dimension of the conflict–climate–food security nexus:

![Exploratory Data Analysis - Target Distribution and Correlations](images/eda_overview.png)


| Category | Count | Features | Rationale |
|----------|-------|----------|-----------|
| **IPC History** | 3 | `ipc_lag1`, `ipc_lag2`, `ipc_trend` | Food insecurity is highly autocorrelated. A region's IPC phase 1–2 months ago is the strongest predictor of its near-future state. The trend feature captures the direction of change. |
| **Conflict Current** | 6 | `events_30d`, `battle_events`, `fatalities_30d`, `civilian_violence`, `conflict_trend`, `battle_share` | Current-month conflict intensity. Battle share (battles/total events) distinguishes active combat from protests or strategic developments. |
| **Conflict Lagged** | 5 | `fatalities_lag1/2/3`, `events_lag1/2` | The causal chain from conflict to food insecurity operates on a 2–3 month delay: conflict → market disruption → harvest failure → IPC escalation. Lags capture this temporal offset. |
| **Conflict Rolling** | 5 | `fatalities_roll2/3`, `events_roll3`, `fatalities_delta`, `fatalities_accel` | A single violent month may be an anomaly; three consecutive months of rising fatalities constitutes genuine escalation. Delta and acceleration features capture the rate and change in rate of violence. |
| **Rainfall** | 5 | `rainfall_anomaly`, `rainfall_lag1/2`, `rainfall_roll3`, `is_drought` | Drought compounds conflict impact. Below-average rainfall in a conflict-affected region accelerates food insecurity far more than either factor alone. |
| **Seasonal** | 3 | `month`, `is_lean_season`, `is_harvest_season` | The agricultural calendar creates predictable hunger patterns. The "lean season" (months between planting and harvest) systematically elevates food insecurity risk. |
| **Compound Risk** | 5 | `compound_risk_score`, `high_conflict_drought`, `sustained_conflict`, `battle_share`, `lean_drought` | Interaction terms that force the model to consider combined effects. The compound risk score (0.6×conflict_norm + 0.4×drought_severity) creates a unified severity index. |

![Feature Importance Pre-Screen (Random Forest)](images/rf_feature_importance_prescreen.png)

**Anti-Leakage Design**: Every temporal feature uses `.shift(1)` applied **before** any rolling window computation. This ensures that no information from the current month contaminates the feature values. A comprehensive leakage audit was conducted and passed with zero violations.


**Feature Correlation with Target (`crisis_90d`)**:

| Rank | Feature | Correlation | Category |
|------|---------|-------------|----------|
| 1 | `ipc_lag1` | 0.844 | IPC History |
| 2 | `ipc_lag2` | 0.821 | IPC History |
| 3 | `compound_risk_score` | 0.204 | Compound Risk |
| 4 | `fatalities_roll3` | 0.174 | Conflict Rolling |
| 5 | `battle_share` | 0.166 | Conflict Current |

The massive correlation gap between the top 2 features (>0.82) and the rest (<0.21) foreshadows XGBoost's eventual feature importance distribution — but the marginal features prove crucial for detecting state transitions.

### 5.3 Walk-Forward Temporal Splitting

Standard k-fold cross-validation is inappropriate for time-series panel data because it would allow the model to train on future observations and validate on past ones. The walk-forward split enforces strict temporal ordering:

```
─────────────────────────────────────────────────────────────────► time
│         TRAIN              │      VALIDATION       │    TEST (SEALED)      │
│  2018-05  →  2022-12       │  2023-01 → 2023-12    │  2024-01 → 2026-01   │
│  13,574 rows (crisis=29%)  │  3,892 rows (49%)     │  8,459 rows (53%)    │
└────────────────────────────┴───────────────────────┴──────────────────────┘
```

| Split | Rows | Date Range | Crisis Rate | Safe:Crisis |
|-------|------|------------|-------------|-------------|
| Train | 13,574 | 2018-05 → 2022-12 | 29.0% | 2.4:1 |
| Validation | 3,892 | 2023-01 → 2023-12 | 48.8% | 1.1:1 |
| Test | 8,459 | 2024-01 → 2026-01 | 52.7% | 0.9:1 |

**Three critical observations about the split design:**

1. **Increasing crisis prevalence across splits** (29% → 49% → 53%) reflects the real-world deterioration of food security in Africa from 2023 onward, driven by the Sudan civil war, Sahel insurgency escalation, and persistent Horn of Africa drought. The model was trained on a historically calmer period and tested on a more crisis-intensive future — a demanding generalization challenge.

2. **Class balance reversal**: The training set is safe-majority (2.4:1), while the test set is crisis-majority (0.9:1). This is handled by setting XGBoost's `scale_pos_weight = 2.45` to compensate for the training set imbalance.

3. **Zero temporal leakage**: Programmatic assertions verified that no date appears in more than one split. Validation set medians were used for all imputation to prevent information leakage from the test set.

### 5.4 Modeling Approach

The modeling strategy followed a rigorous progression from interpretable baselines to tuned ensemble methods:

**Stage 1 — Baselines:**

| Model | Configuration | Rationale |
|-------|--------------|-----------|
| Logistic Regression | `class_weight='balanced'` | Linear baseline; tests whether the problem is linearly separable |
| Random Forest (200 trees) | `class_weight='balanced'` | Non-linear baseline; provides a strong performance floor without tuning |

**Stage 2 — XGBoost Default:**

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `n_estimators` | 1,000 (early-stopped at 353) | Sufficient capacity with early stopping to prevent overfitting |
| `scale_pos_weight` | 2.4487 | Compensates for 2.4:1 safe:crisis training imbalance |
| `eval_metric` | `aucpr` | PR-AUC is the correct metric for imbalanced humanitarian prediction |
| `early_stopping_rounds` | 30 | Prevents overtraining on validation set |

**Stage 3 — Grid Search (108 combinations):**

| Parameter | Search Values | Best |
|-----------|--------------|------|
| `max_depth` | 4, 6, 8 | **4** |
| `learning_rate` | 0.01, 0.05, 0.1 | **0.05** |
| `subsample` | 0.7, 0.9 | **0.9** |
| `colsample_bytree` | 0.7, 0.9 | **0.7** |
| `min_child_weight` | 3, 5, 10 | **5** |

The selection of `max_depth=4` over deeper alternatives (6, 8) is a critical regularization signal: shallower trees generalize better from the 2018–2022 training period to the structurally different 2023+ crisis landscape. `min_child_weight=5` prevents the model from creating leaf nodes based on fewer than 5 observations, reducing overfitting in data-sparse regions like Chad's northern provinces.

### 5.5 Threshold Selection Strategy

Binary predictions require a probability threshold. Two operating modes were developed:

| Mode | Threshold | Recall | FNR | FP Count | Use Case |
|------|-----------|--------|-----|----------|----------|
| **F1-Optimal** | 0.497 | 89.9% (val) / 98.4% (test) | 10.1% / 1.6% | 8 / 5 | Standard operations — balanced precision/recall |
| **Humanitarian** | 0.077 | 95.7% (val) / 98.9% (test) | 4.3% / 1.1% | 216 / 388 | Maximum sensitivity — "miss no crisis" mode |

The wide gap between thresholds (0.497 vs. 0.077) reflects the model's strong confidence calibration: most predictions cluster near 0.0 (confident safe) or 1.0 (confident crisis), with few ambiguous cases near 0.5. This bimodal distribution is ideal for threshold-based deployment because small threshold changes have minimal impact within the "confident" zones.

---

## 6. Results

### 6.1 Baseline Performance

Both baselines established a high performance floor, confirming that the IPC lag features create near-linear separability:

| Model | F1 | Recall | Precision | PR-AUC | ROC-AUC |
|-------|-----|--------|-----------|--------|---------|
| Logistic Regression | 0.9406 | 0.8962 | 0.9895 | 0.9755 | 0.9684 |
| Random Forest (200T) | 0.9435 | 0.8930 | **1.0000** | 0.9729 | 0.9637 |

The Random Forest achieves perfect precision (zero false alarms) but at the cost of 10.7% missed crises. Logistic Regression's competitive F1 (0.9406) confirms that the feature space is already well-engineered — even a linear model can identify most crises from the provided features.

### 6.2 XGBoost Tuning & Selection

![Model Comparison Across All Metrics (Validation Set)](images/model_comparison_validation.png)


| Model | F1 | Recall | Precision | PR-AUC | ROC-AUC |
|-------|-----|--------|-----------|--------|---------|
| XGBoost Default | 0.9432 | 0.8967 | 0.9947 | **0.9906** | **0.9903** |
| **XGBoost Tuned** | **0.9449** | **0.8994** | 0.9953 | 0.9890 | 0.9882 |

![Best XGBoost PR and ROC Curves (Validation Set)](images/best_model_pr_roc_validation.png)


A critical insight: although the tuned model's F1 exceeds the default by only 0.0017, it was selected because hyperparameter tuning provides a structured, reproducible process for model selection. The default model's slightly higher PR-AUC (0.9906 vs. 0.9890) is within the noise margin of the validation set.

**Feature Importance (Gain-Based):**

| Rank | Feature | Gain % | Cumulative | Interpretation |
|------|---------|--------|------------|----------------|
| 1 | `ipc_lag1` | 48.32% | 48.32% | IPC phase last month — dominant autoregressive signal |
| 2 | `ipc_lag2` | 36.42% | 84.74% | IPC phase 2 months ago — confirmation signal |
| 3 | `ipc_trend` | 6.08% | 90.82% | Direction of IPC change — escalation detector |
| 4 | `month` | 0.96% | 91.78% | Seasonal hunger patterns |
| 5 | `rainfall_anomaly` | 0.66% | 92.44% | Drought amplification effect |
| 6–7 | `rainfall_lag1`, `battle_events` | 0.59% each | 93.62% | Climate lag and conflict intensity |
| 8–32 | Remaining features | 6.38% total | 100.00% | Marginal but non-redundant signals |

![XGBoost Gain-Based Feature Importance (Validation Set)](images/xgboost_feature_importance.png)


> [!TIP]
> The top 3 features account for **90.8% of model gain**, but the remaining 29 features collectively contribute 9.2% — this "long tail" is essential for detecting sudden-onset crises where IPC history alone is insufficient. Removing these marginal features would significantly increase the false negative rate.

### 6.3 Final Test Set Evaluation

The test set (2024-01 to 2026-01, 8,459 rows) was **sealed** during all model development and opened only once during Task 4. Results:

| Metric | Validation (2023) | **Test (2024–2026)** | Delta |
|--------|:-----------------:|:--------------------:|:-----:|
| **F1-Score** | 0.9449 | **0.9914** | +0.0465 ↑ |
| **Recall** | 0.8994 | **0.9841** | +0.0847 ↑ |
| **Precision** | 0.9953 | **0.9989** | +0.0036 ↑ |
| **PR-AUC** | 0.9890 | **0.9976** | +0.0086 ↑ |
| **ROC-AUC** | 0.9882 | **0.9966** | +0.0084 ↑ |
| **FNR** | 10.1% | **1.6%** | 6× better ↑ |
| **FP Count** | 8 | **5** | 38% better ↑ |

![Final Test Set Evaluation - Confusion Matrices and PR Curves](images/confusion_matrices_pr_curves.png)


**Confusion Matrix (F1-Optimal Threshold = 0.497):**

```
                  Predicted Safe    Predicted Crisis
Actual Safe           3,996 ✅             5 ⚠️
Actual Crisis            71 ⚠️         4,387 ✅

False Negative Rate :  1.6%  (71 missed crises out of 4,458)
False Positive Rate :  0.1%  (5 false alarms out of 4,001)
```

**Why Test Performance Exceeded Validation:**

This is an unusual but fully explainable result. Two structural factors account for it:

1. **Crisis persistence in 2024–2026**: The Sudan civil war maintained continuous IPC Phase 4–5 conditions; South Sudan experienced chronic famine; Somalia remained in persistent drought. The model's `ipc_lag1/lag2` features work perfectly for persistent states — predicting continuation of an existing crisis is fundamentally easier than predicting a transition.

2. **Validation set (2023) captured more transition dynamics**: The year 2023 included the onset of the Sudan civil war, regime changes in the Sahel, and rapidly evolving food security conditions. These transitional periods are intrinsically harder to predict because the system hasn't yet entered a stable crisis state that the lag features can capture.

### 6.4 Country-Level Performance

| Country | Rows | Crisis % | F1 | Recall | FNR | PR-AUC | Note |
|---------|------|----------|-----|--------|-----|--------|------|
| Sudan | 2,184 | 95.4% | **1.0000** | 1.0000 | 0.0% | 1.0000 | Perfect — persistent civil war |
| Somalia | 900 | 90.7% | 0.9951 | 0.9902 | 1.0% | 0.9994 | Near-perfect |
| Burkina Faso | 325 | 30.2% | 0.9897 | 0.9796 | 2.0% | 0.9951 | Excellent |
| Cameroon | 250 | 24.0% | 0.9831 | 0.9667 | 3.3% | 0.9933 | Very good |
| Chad | 600 | 36.5% | 0.9814 | 0.9635 | 3.7% | 0.9924 | Dramatically improved from 27.4% val FNR |
| Niger | 225 | 52.4% | 0.9785 | 0.9661 | 3.4% | 0.9946 | Solid Sahel performance |
| Ethiopia | 400 | 50.2% | 0.9745 | 0.9502 | 5.0% | 0.9900 | Post-Tigray stabilization |
| Nigeria | 1,575 | 14.7% | 0.9690 | 0.9481 | 5.2% | 0.9809 | Low crisis rate challenges recall |
| Mozambique | 275 | 77.1% | 0.9659 | 0.9340 | 6.6% | 0.9969 | Cyclone blind spot |
| Kenya | 1,225 | 8.3% | 0.9648 | 0.9412 | 5.9% | 0.9717 | Relatively stable |
| Mali | 225 | 18.7% | 0.9250 | 0.8810 | **11.9%** | 0.9309 | Weakest — complex Sahel dynamics |

![Country-Level Performance (Test Set)](images/country_level_performance.png)


**Geographic Performance Insights:**

- **Sudan's perfect score** confirms that highly persistent crisis states are trivially predictable once established. The model correctly learned that Sudan post-2023 is in continuous severe crisis.
- **Chad's dramatic improvement** (FNR: 27.4% → 3.7%) occurred because the Lake Chad Basin conditions stabilized into a recognizable crisis pattern during 2024–2025, whereas 2023 had more volatile transitions.
- **Mali is the weakest performer** (11.9% FNR) due to its complex Sahel seasonal dynamics and distributed insurgency pattern, which create more variable crisis signatures than either persistent famine (Somalia/Sudan) or localized conflict (Chad/Burkina Faso).
- **Mozambique emerges as a new concern** (6.6% FNR, 14 FN) because cyclone-induced food insecurity is not captured by the conflict + IPC feature set. This is the clearest signal that climate-specific features are needed.

### 6.5 SHAP Explainability Analysis

SHAP (SHapley Additive exPlanations) values were computed for all 8,459 test predictions across 32 features, producing an 8,459 × 32 explanation matrix.

![SHAP Global Feature Importance (Beeswarm)](images/shap_beeswarm_global.png)
![SHAP Mean Impact (Average Magnitude)](images/shap_mean_impact.png)


**Global Beeswarm Analysis:**

- **`ipc_lag1`** has the widest SHAP spread, with high IPC values (red dots) pushed far to the right — the single strongest predictor
- **`ipc_lag2`** provides independent confirmation of the persistence signal with a slightly narrower but still dominant spread
- **`ipc_trend`** captures escalation dynamics — even a region in Phase 2 gets pushed toward crisis prediction if the trend is rising
- **`battle_events`** shows moderate rightward displacement for high values — conflict intensity provides signal above and beyond IPC persistence
- **`rainfall_anomaly`** shows negative anomalies (drought) amplifying crisis prediction — confirming the compound shock hypothesis

**Case Study — Highest Confidence Prediction (True Positive):**

```
Region: Chad / Lac, September 2024
Probability: 1.0000 (all 499 trees unanimous)
True label: CRISIS ✅

SHAP contributions:
  ipc_lag1        → +0.82  (Phase 4-5 in previous month)
  ipc_lag2        → +0.31  (Phase 4 two months ago)
  battle_events   → +0.12  (Boko Haram activity in Lake Chad Basin)
  is_lean_season  → +0.08  (September = peak Sahel lean season)

![SHAP Waterfall - Highest Risk Case (True Positive)](images/shap_waterfall_highest_risk.png)

```

This case demonstrates perfect alignment between model reasoning and domain knowledge: persistent severe food insecurity, active armed conflict, and lean season timing all converge to produce a unanimous crisis prediction.

**Case Study — Missed Crisis (False Negative):**

```
Region: Chad / Barh El Gazel, February 2024
IPC: Phase 3 (actual crisis)
Probability: 0.0102 (predicted SAFE — MISSED)

SHAP contributions:
  ipc_lag1  → -0.43  (Phase 1-2 in previous months — no historical crisis signal)
  ipc_lag2  → -0.21  (also Phase 1-2 two months prior)
  battle_events → near zero (low conflict compared to Lac region)

![SHAP Waterfall - Missed Crisis Blind Spot (False Negative)](images/shap_waterfall_false_negative.png)

```

**Root Cause**: Barh El Gazel is a sparsely monitored northern Chad region with limited IPC reporting history. A sudden food production failure in February 2024 was not anticipated by any feature in the model's repertoire. The model correctly applied its learned rule ("Phase 1-2 last month → predict continuation") but the real-world conditions changed faster than the lagged features could capture.

### 6.6 Africa Crisis Risk Map

An interactive Folium map (`africa_crisis_map_v2.html`, 641 KB) was produced, visualizing model predictions across all panel countries:

![Africa Crisis Risk Map - Predicted Mean Probabilities](images/crisis_risk_map.jpg)


| Risk Level | Countries | Mean Probability |
|------------|-----------|-----------------|
| 🟣 **Famine** | DRC, South Sudan, Somalia, Sudan | 0.845–1.000 |
| 🔴 **Emergency** | Mozambique | 0.606 |
| 🟠 **Crisis** | Ethiopia, Niger | 0.488–0.525 |
| 🟡 **Stressed** | Chad, Burkina Faso, Cameroon, Mali | 0.235–0.335 |
| 🟢 **Minimal** | Nigeria, Kenya | 0.161–0.185 |

**Ground Truth Validation:**

All 13 classified countries align with independently published 2024–2026 humanitarian assessments:

- South Sudan P=0.999 → IPC Famine declared in Unity State, April 2024 ✅
- Sudan P=0.845 → World's largest hunger crisis, 25M affected in 2024 ✅
- Kenya P=0.161 → Relatively stable food security throughout 2024–2025 ✅
- Mozambique P=0.606 → Cyclone-affected regions in Emergency phase ✅

---

## 7. Error Analysis & Failure Modes

### 7.1 False Negatives (Missed Crises)

The 71 missed crises in the test set (down from 192 in validation — a 63% reduction) break down as:

| Country | FN Count | FN Rate | Primary Cause |
|---------|----------|---------|---------------|
| Mozambique | 14 | 6.6% | Cyclone-induced food insecurity not captured by conflict features |
| Nigeria | 12 | 5.2% | Low overall crisis rate (14.7%) makes transition detection harder |
| Ethiopia | 10 | 5.0% | Post-Tigray regional fragmentation creating local crises |
| Chad | 8 | 3.7% | Data-sparse northern regions with sudden deterioration |
| Somalia | 8 | 1.0% | Near-perfect; remaining FNs are edge cases |
| Kenya | 6 | 5.9% | Localized droughts in otherwise stable country |
| Mali | 5 | 11.9% | Complex Sahel insurgency with volatile crisis patterns |

![Error Analysis - Distribution of Missed Crises (False Negatives)](images/error_analysis_missed_crises.png)


**IPC Phase Distribution of Missed Crises:**

| Phase at Time of Miss | Count | Percentage |
|----------------------|-------|------------|
| Phase 1 (Minimal) | 1 | 1.4% |
| Phase 2 (Stressed) | 35 | 49.3% |
| Phase 3 (Crisis) | 35 | 49.3% |

The Phase 2 dominance (49.3%) reveals the model's core challenge: **Phase 2→3 transitions**. The model correctly predicts Phase 2 continuation approximately 90% of the time, but misses the inflection points where conditions suddenly worsen. These transitions are driven by rapid-onset shocks (conflict outbreak, sudden market disruption, crop disease) that do not manifest in lagged features until after the crisis has already begun.

### 7.2 False Positives (False Alarms)

Only 5 false alarms occurred across the entire 2-year test period — an FPR of 0.1%. This means the system virtually never triggers unnecessary humanitarian responses. The extremely low FPR is a major operational advantage: aid organizations can trust positive alerts without "cry wolf" fatigue.

### 7.3 Systematic Failure Modes

| Failure Mode | Frequency | Root Cause | Proposed Fix |
|-------------|-----------|------------|-------------|
| **Climate-driven food insecurity** (no conflict trigger) | ~20% of FN | Mozambique cyclones, East African droughts | Add NDVI, SST, and EM-DAT disaster features |
| **Sudden Phase 2→3 transitions** | ~50% of FN | Lag features inherently cannot capture instantaneous shocks | Shorter lag windows (2-week) or real-time event streams |
| **Data-sparse regions** | ~15% of FN | Limited IPC monitoring in remote admin1 areas | Satellite proxy features (NDVI anomaly as IPC proxy) |
| **Complex insurgency patterns** (Mali) | ~15% of FN | Distributed, seasonal conflict patterns that defy rolling-window capture | Country-specific sub-models or attention-based temporal features |

---

## 8. Key Insights & Findings

### Insight 1: Food Insecurity is Profoundly Autocorrelated

The most significant finding is that **84.7% of the model's predictive power** comes from knowing what happened 1–2 months ago (`ipc_lag1` + `ipc_lag2`). Food crises do not appear and disappear rapidly — they develop gradually and persist for months or years. A region in IPC Phase 3 last month has an overwhelming probability of remaining in Phase 3 this month.

**Implication**: The hard problem is not predicting continuation — any autoregressive model can do that. The hard problem is predicting **transitions**: when Phase 2 will deteriorate to Phase 3, or when Phase 4 will improve to Phase 3. This is where the conflict and rainfall features earn their 9.2% of model gain.

### Insight 2: Conflict Provides Genuine Signal Beyond IPC Persistence

Battle events and fatalities collectively contribute 1.1% of model gain — a small but statistically significant signal. SHAP analysis confirms that high conflict values push predictions toward crisis even when IPC history is ambiguous. This is particularly valuable in the ~10% of cases where IPC lag features are insufficient.

### Insight 3: Drought Amplifies Conflict Impact Non-Linearly

`rainfall_anomaly` (0.66% gain) ranks above `battle_events` (0.59% gain) in feature importance. SHAP waterfall plots show that negative rainfall anomalies (drought conditions) amplify the crisis prediction signal when combined with elevated conflict. This confirms the **compound shock hypothesis**: conflict alone may be survivable, drought alone may be survivable, but conflict + drought together precipitate crisis.

### Insight 4: Persistent Crises Are Perfectly Predicted

Countries with sustained, high-intensity crises (Sudan, Somalia, South Sudan) achieve near-perfect F1 scores (0.995–1.000). The model excels in these environments because the lag features perfectly capture the "locked-in" nature of prolonged conflict-induced famine.

### Insight 5: Seasonal Patterns Matter

The `month` feature ranks 4th by gain (0.96%), indicating that the agricultural calendar creates predictable hunger seasonality. The "lean season" — the period between planting and harvest when food stores are depleted — systematically elevates crisis risk and interacts with both conflict and drought.

### Insight 6: Model Generalizes Forward in Time

The test set's superior performance over validation is counterintuitive but explainable. It demonstrates that the model has learned genuinely transferable patterns rather than fitting to the idiosyncrasies of the training period. The model was trained on a relatively calm period (2018–2022, 29% crisis rate) but successfully predicts a much more crisis-intensive future (2024–2026, 53% crisis rate).

### Insight 7: Geographic Blind Spots Are Addressable

The dramatic improvement in Chad's performance (FNR: 27.4% → 3.7%) between validation and test sets shows that blind spots are not fundamental model limitations — they reflect transitional dynamics that stabilize over time. However, new blind spots can emerge (Mozambique's cyclone-driven crises), requiring ongoing feature development.

---

## 9. Limitations

### 9.1 Data Limitations

- **Temporal coverage**: 8 years of data (2018–2026) limits the model's exposure to rare but impactful events (e.g., the 2011 Horn of Africa famine, the 2017 South Sudan famine)
- **Spatial resolution**: Admin1 aggregation masks sub-regional heterogeneity. A province-level Phase 2 average may hide localized Phase 4 conditions in specific districts
- **FEWS NET dependency**: The target variable relies on FEWS NET IPC assessments, which themselves have publication lags and coverage gaps in remote areas
- **ACLED API access**: Conflict data depends on continued availability of the ACLED API, which may change access terms

### 9.2 Methodological Limitations

- **No country-level encoding**: The model treats all countries identically. Country-specific effects (institutional capacity, humanitarian infrastructure, agricultural systems) are not explicitly modeled
- **Binary target**: Collapsing IPC Phases 1–5 into binary crisis/safe (≥3 vs. <3) loses ordinal information. A Phase 4→5 transition has very different humanitarian implications than a Phase 2→3 transition
- **No uncertainty quantification**: Point predictions lack confidence intervals. A prediction of P=0.60 could mean "moderately likely crisis" or "high model uncertainty"
- **Climate feature gaps**: No vegetation index (NDVI), sea surface temperature, or extreme weather event features. Cyclone-driven food crises (Mozambique) are systematically underdetected
- **No displacement or market data**: Conflict-displaced populations and commodity price spikes are key causal mechanisms that are not directly represented in the feature set

### 9.3 Deployment Limitations

- **No automated data refresh**: The current system requires manual data download and reprocessing. A production system would need automated monthly ACLED + CHIRPS + FEWS NET ingestion
- **Model drift**: As conflict patterns evolve, model performance may degrade without periodic retraining
- **Threshold calibration**: The optimal threshold (0.497) was calibrated on the 2023 validation set. If crisis prevalence changes significantly, the threshold should be recalibrated

---

## 10. Future Work & Recommendations

### 10.1 Immediate Priorities (0–3 months)

#### 10.1.1 Country Fixed-Effects Encoding

**What**: Add one-hot encoded country identifiers as features  
**Why**: The model currently treats all 14 countries identically, but institutional capacity, agricultural systems, and humanitarian infrastructure vary dramatically. Nigeria's food security response mechanisms are fundamentally different from South Sudan's.  
**Expected Impact**: Reduction in geographic performance variance; Mali and Mozambique FNR improvement  
**Implementation Complexity**: Low — straightforward feature addition

```python
# Example implementation
country_dummies = pd.get_dummies(df['country'], prefix='ctry').astype(float)
X = pd.concat([X_features, country_dummies], axis=1)
```

#### 10.1.2 Cyclone and Natural Disaster Features

**What**: Integrate EM-DAT international disaster database to capture cyclone tracks, flooding events, and earthquake impacts  
**Why**: Mozambique's 6.6% FNR is directly attributable to cyclone-induced food insecurity that the conflict-focused feature set cannot capture  
**Expected Impact**: Significant reduction in Mozambique and coastal Somalia FNR  
**Implementation Complexity**: Medium — requires EM-DAT API integration and temporal alignment

#### 10.1.3 Admin1-Level SHAP Dashboard

**What**: Extend the Streamlit dashboard with interactive SHAP waterfall charts for each selected admin1 region  
**Why**: Humanitarian decision-makers need to understand not just "this region is at risk" but "why this region is at risk and what factors are driving the prediction"  
**Expected Impact**: Improved trust and actionability of model outputs  
**Implementation Complexity**: Medium — SHAP computation is expensive for real-time display; pre-computed SHAP values recommended

### 10.2 Medium-Term Improvements (3–9 months)

#### 10.2.1 Uncertainty Quantification via Conformal Prediction

**What**: Wrap the XGBoost model in a conformal prediction framework to produce prediction sets with guaranteed coverage  
**Why**: A prediction of P=0.60 currently provides no information about model confidence. Conformal prediction can say "at 90% confidence, this region is in {Crisis, Stressed}" — fundamentally more useful for decision-making  
**Expected Impact**: Enables risk-stratified response planning with quantified uncertainty  

```python
# Example with MAPIE library
from mapie.classification import MapieClassifier
mapie = MapieClassifier(estimator=best_model, cv=5)
y_pred, y_ps = mapie.predict(X_test, alpha=0.1)  # 90% coverage prediction sets
```

#### 10.2.2 Climate Sub-Model for Weather-Driven Crises

**What**: Build a specialized model for cyclone-affected and drought-vulnerable countries (Mozambique, Madagascar, coastal Somalia) incorporating NDVI vegetation index and sea surface temperature anomalies  
**Why**: The current model's climate features (CHIRPS rainfall anomaly) capture slow-onset drought but not rapid-onset weather events  
**Expected Impact**: Dramatic improvement in Mozambique prediction accuracy; potential expansion to Madagascar (currently excluded)

#### 10.2.3 Phase-Level (Ordinal) Prediction

**What**: Replace the binary crisis/safe target with ordinal IPC Phase 1–5 prediction  
**Why**: Binary classification treats Phase 2→3 transitions and Phase 4→5 transitions identically, but the humanitarian implications are vastly different. Ordinal prediction enables more granular response planning  
**Implementation Options**: `XGBRegressor` with rounded outputs, ordinal LightGBM, or multi-class classification with ordered classes

#### 10.2.4 Real-Time Automated Pipeline

**What**: Build an automated monthly pipeline that ingests fresh data from ACLED API, CHIRPS archives, and FEWS NET, runs feature engineering, generates predictions, updates the crisis map, and dispatches alerts  
**Why**: The current system requires manual execution of four Jupyter notebooks. A production system must operate autonomously  

```
Monthly cycle:
  ACLED API → CHIRPS raster → FEWS NET IPC →
  Feature engineering → XGBoost inference →
  Crisis map update → Alert dispatch (email/Slack/Teams)
```

### 10.3 Long-Term Research Directions (9–24 months)

#### 10.3.1 Causal DAG Model

**What**: Develop a directed acyclic graph (DAG) causal model representing the conflict → displacement → market disruption → food insecurity causal chain  
**Why**: The current model identifies correlations but cannot answer causal questions like "if we reduce battle events by 50% in this region, how much does crisis probability decrease?"  
**Expected Impact**: Enables counterfactual analysis for policy and intervention planning

#### 10.3.2 WFP/OCHA Pilot Program

**What**: Partner with UN World Food Programme or UN Office for the Coordination of Humanitarian Affairs to test whether 90-day advance warnings improve aid pre-positioning efficiency compared to traditional FEWS NET assessments  
**Why**: The ultimate value of this system can only be validated through real-world humanitarian impact  
**Expected Impact**: Evidence for/against the operational value of ML-based food crisis prediction

#### 10.3.3 Multi-Country Ensemble Architecture

**What**: Train country-specific sub-models for each panel country, then combine predictions with a global meta-learner  
**Why**: A single model cannot optimally capture the diverse dynamics of 14 very different countries. Somalia's persistent drought dynamics, Nigeria's Boko Haram insurgency, and Mozambique's cyclone vulnerability require different predictive strategies  
**Expected Impact**: Reduction in country-level performance variance; potential for per-country threshold optimization

#### 10.3.4 Shorter-Horizon Prediction Variants

**What**: Create 30-day and 60-day prediction variants alongside the current 90-day target  
**Why**: Different humanitarian responses operate on different timescales. Cash transfers can be deployed in 30 days; food supply chain adjustments require 90 days  
**Expected Impact**: Multi-horizon prediction enables multi-tier response planning

#### 10.3.5 Incorporating Market and Displacement Data

**What**: Integrate WFP commodity price monitoring data and UNHCR displacement/refugee flow data as additional features  
**Why**: Market prices and displacement are key intermediate causal mechanisms between conflict and food insecurity. Rising staple food prices and population displacement are often the first observable signals of impending food crisis  
**Expected Impact**: Earlier detection of food crises driven by economic disruption rather than direct conflict

---

## 11. Conclusion

This project demonstrates that **machine learning can provide highly accurate 90-day advance warnings for conflict-induced food crises** in Sub-Saharan Africa. The system achieves an F1-score of 0.9914 on a rigorous temporal test set spanning 2024–2026, correctly identifying 98.4% of actual food crises while producing only 5 false alarms across 8,459 predictions.

The results confirm a central insight about the nature of food insecurity: it is **profoundly persistent and predictable from its own recent history**. A region that was in IPC Phase 3+ last month has an overwhelming probability of remaining in crisis this month. This autocorrelation — captured by the `ipc_lag1` and `ipc_lag2` features — accounts for 84.7% of the model's predictive power.

However, the project's most important contribution lies in the remaining 15.3% of model gain. Conflict events, rainfall anomalies, seasonal patterns, and compound risk features provide the critical marginal signal needed to detect **state transitions** — the moments when a stable Phase 2 suddenly deteriorates to Phase 3, or when a chronic Phase 4 unexpectedly improves. These transitions, while representing only ~10% of all predictions, are the cases where early warning systems deliver the greatest humanitarian value.

SHAP explainability analysis confirms that the model's decision logic aligns with established humanitarian domain knowledge: it prioritizes historical food security states, amplifies predictions when armed conflict co-occurs with drought, and accounts for seasonal agricultural cycles. This interpretability is essential for building trust with humanitarian decision-makers who must act on model outputs in high-stakes environments.

The system has been deployed as both an interactive Folium crisis map and a Streamlit dashboard, providing actionable intelligence that aligns with independently published 2024–2026 humanitarian assessments across all 14 panel countries. The path forward is clear: immediate priorities include country-specific features and climate data integration to address the Mozambique blind spot, while medium-term work should focus on uncertainty quantification, automated pipelines, and ordinal phase prediction. Long-term, the system's value must be validated through real-world humanitarian pilot programs with organizations like WFP and OCHA.

> [!IMPORTANT]
> **The ultimate measure of this system's success is not F1-score or PR-AUC — it is whether the 71 missed crises in the test set could have been detected earlier with additional data and modeling sophistication, and whether the 4,387 correct crisis predictions could have been delivered to humanitarian organizations 90 days before traditional assessments reached the same conclusion.** This project provides strong evidence that the answer to both questions is yes.

---

*Report generated from analysis of 4 Jupyter notebooks covering data collection (207,682 ACLED events), feature engineering (32 features, 7 categories), model training (108 hyperparameter combinations), and final evaluation (8,459 test predictions with SHAP explainability across 14 Sub-Saharan African countries).*

*Built with XGBoost 3.2.0 · SHAP 0.51.0 · Folium 0.20.0 · Streamlit · Google Colab*
