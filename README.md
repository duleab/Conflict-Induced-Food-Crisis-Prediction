# Conflict-Induced Food Crisis Early Warning System

> **Predicting IPC Phase 3+ food insecurity 90 days ahead at admin-1 level across 12 African countries using conflict, rainfall, and food security data.**

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)[![XGBoost](https://img.shields.io/badge/model-XGBoost-orange.svg)](https://xgboost.readthedocs.io/)[![AUC 0.9495](https://img.shields.io/badge/ROC--AUC-0.9495-green.svg)]()[![Recall 82.9%](https://img.shields.io/badge/Crisis%20Recall-82.9%25-brightgreen.svg)]()

---

## Overview

A machine learning early warning system that flags African admin-1 regions at risk of food crisis (IPC Phase 3+) within the next 90 days. Fuses ACLED conflict events, FEWS NET IPC food security phases, and CHIRPS rainfall data into a panel dataset, then trains XGBoost on 31 engineered features. Achieves ROC-AUC 0.9495 on a held-out 2023 test set with no target leakage.

**Countries:** Burkina Faso, Cameroon, Chad, Ethiopia, Kenya, Mali, Mozambique, Niger, Nigeria, Somalia, South Sudan, Sudan

**Period:** February 2018 to October 2023

---

## Results

| Metric | Value |
|---|---|
| ROC-AUC | 0.9495 |
| PR-AUC | 0.9367 |
| Crisis Recall | 82.9% — 806 of 972 crises caught |
| Crisis Precision | 92.5% — 65 false alarms |
| F1 Score (crisis) | 0.8747 |
| Val to Test AUC gap | −0.014 |

Top SHAP features: `ipc_lag1` (2.998) then `ipc_lag2` (0.725) then `rainfall_lag1` (0.314) then `month` (0.272) then `rainfall_roll3` (0.237)

---

## Data Sources

| Source | What it provides | Access |
|---|---|---|
| ACLED (acleddata.com) | Armed conflict events, fatalities | REST API, free registration |
| FEWS NET (fews.net/data) | IPC food security phases per admin unit | Free CSV download |
| CHIRPS (CHC UCSB) | Monthly rainfall totals | Direct download, free |

---

## Project Structure

```
crisis-early-warning/
├── notebooks/
│   ├── Crisis_task1_data_collection.ipynb
│   ├── crisis_task2_feature_engineering.ipynb
│   ├── crisis_task3_Baseline_Models.ipynb
│   └── crisis_task4_evaluation_FINAL.ipynb
├── crisis_outputs/
│   ├── panel_dataset.csv
│   ├── features_engineered.csv
│   ├── xgboost_best.pkl
│   ├── shap_values.npy
│   ├── test_results.json
│   ├── model_card.txt
│   └── app.py
├── requirements.txt
└── README.md
```

---

## Pipeline Summary

```
Task 1  ACLED + FEWS NET + CHIRPS fused into 14,697 region-month panel
Task 2  31 temporal features + walk-forward split (leakage audit: PASSED)
Task 3  XGBoost (Val AUC 0.9639) + Random Forest baseline
Task 4  SHAP explainability + error analysis + Streamlit dashboard
```

---

## Quickstart

```bash
pip install -r requirements.txt
streamlit run crisis_outputs/app.py
```

In Colab:

```python
!pip install streamlit pyngrok -q
!nohup streamlit run /content/crisis_outputs/app.py &
from pyngrok import ngrok
print(ngrok.connect(8501))
```

---

## Features (31 total)

| Category | Features | Key finding |
|---|---|---|
| IPC persistence | ipc_lag1, ipc_lag2, ipc_trend | Correlation 0.78 — dominant signal |
| Conflict current | fatalities_30d, events_30d, battle_events, civilian_violence, conflict_trend, battle_share | Current month state |
| Conflict lags | fatalities_lag1..3, events_lag1..2, fatalities_roll2/3, events_roll3, delta, accel | 2–3 month causal delay |
| Compound risk | high_conflict, high_conflict_drought, compound_risk_score, sustained_conflict | Synergistic hazard |
| Rainfall | rainfall_anomaly, is_drought, rainfall_lag1/2, rainfall_roll3 | Climate pathway |
| Seasonal | is_lean_season, is_harvest_season, month | Agricultural calendar |

Excluded from features to prevent leakage: `ipc_phase` (current month) and `population_in_crisis` (derived from ipc_phase).

---

## Walk-Forward Split

```
Train  9,372 rows  2018-05 to 2021-12  crisis rate 28.2%
Val    2,556 rows  2022-01 to 2022-12  crisis rate 41.4%
Test   2,130 rows  2023-01 to 2023-10  crisis rate 45.6%
```

Standard k-fold cross-validation is not used — it causes temporal leakage in panel data.

---

## Country Performance

| Country | Recall | AUC |
|---|---|---|
| South Sudan | 1.00 | — |
| Nigeria | 0.95 | 0.985 |
| Somalia | 0.92 | 0.749 |
| Cameroon | 0.89 | 0.918 |
| Ethiopia | 0.86 | 0.923 |
| Sudan | 0.85 | 0.768 |
| Kenya | 0.82 | 0.964 |
| Burkina Faso | 0.80 | 0.982 |
| Niger | 0.71 | 0.883 |
| Mali | 0.64 | 0.844 |
| Mozambique | 0.62 | 0.864 |
| Chad | 0.41 | 0.774 |

---

## Known Limitations

1. CHIRPS is a national broadcast — all regions in a country share identical rainfall values
2. Chad recall is 0.41 — admin name mismatch between ACLED and FEWS NET
3. Sudden-onset crises are missed when ipc_lag1 = 2 (Phase 2 Stressed)
4. Eight countries excluded due to FEWS NET coverage gaps or admin name mismatches
5. FEWS NET data ends October 2023

---

## Future Work

- Real per-admin-1 CHIRPS via Google Earth Engine (highest impact)
- Manual admin name mapping for Chad and DRC
- WFP market price anomaly features (maize, sorghum)
- UNHCR displacement counts and growth rates
- Probability calibration via Platt scaling
- Admin-2 resolution for Ethiopia, Somalia, Kenya, South Sudan
- LightGBM ensemble with XGBoost
- Automated monthly inference and annual retraining pipeline

---

## Model Card

```
Model         XGBoost n_estimators=400, max_depth=6, lr=0.05
Features      31 (no leakage)
Training data 14,058 region-months, 12 countries, 2018 to 2023
Target        crisis_90d = IPC Phase 3+ in next 90 days

TEST SET PERFORMANCE (2023)
  ROC-AUC   0.9495
  PR-AUC    0.9367
  Recall    82.9%  (806 crises caught, 166 missed)
  Precision 92.5%  (65 false alarms)
  Threshold 0.920  (F1-maximised on validation set)

SDG 2 Zero Hunger | SDG 16 Peace | SDG 13 Climate
Not for automated aid allocation without human review
```

---

## License

MIT. Data sources have their own terms: ACLED requires attribution, FEWS NET and CHIRPS are public domain.
