#  Africa Conflict-Induced Food Crisis Prediction — Full Pipeline Report

# Task 1: Data Collection & Raw Pipeline

**File:** `Crisis_task1_data_collection.ipynb`

---
# Conflict-Induced Food Crisis Prediction
## Task 1: Data Fusion & Panel Creation  v2.0

**Goal**: Fuse ACLED conflict, FEWS NET IPC, and CHIRPS rainfall into a single
`country × admin1 × year_month` panel ready for feature engineering.

| Step | Action | Output |
|------|--------|--------|
| 1 | Install & import | environment ready |
| 2 | Configuration | country lists, date range |
| 3 | Load ACLED | `acled_africa_raw.csv` → monthly aggregation |
| 4 | Load CHIRPS | `chirps_monthly.csv` → anomaly features |
| 5 | Load FEWS NET | IPC phases → `fewsnet_ipc.csv` |
| 6 | Three-way merge | `panel_dataset.csv` (14,697 rows) |
| 7 | Quality report | `data_quality_report.json` |
| 8 | Drive backup | all artifacts saved |

**Design rules**
- FEWS NET is the LEFT table → only labeled region-months survive the merge  
- `crisis_90d` = 1 if IPC Phase ≥ 3 in ANY of the next 3 months (forward-only)  
- No future leakage: features use current/past data only  
- Uganda excluded: ACLED at admin-2, FEWS at admin-1 → 99 % unmatched  


---
## Step 1 — Install & Import

#### 🏁 Output
```text
Data / output dir : /content/crisis_outputs
Pandas version    : 2.2.2
 All libraries loaded.

```

---
## Step 2 — Google Drive Mount

####  Output
```text
Drive already mounted at /content/drive; to attempt to forcibly remount, call drive.mount("/content/drive", force_remount=True).
 Files already present — skipping restore.

Existing CSVs (26):
  X_test.csv                                      1289.3 KB
  X_train.csv                                     2207.3 KB
  X_val.csv                                        621.3 KB
  acled_africa_raw.csv                           10334.1 KB
  acled_monthly.csv                               1766.5 KB
  chirps_monthly.csv                              2016.7 KB
  chirps_processed.csv                            2016.7 KB
  feature_importance.csv                             2.0 KB
  features_engineered.csv                         5791.2 KB
  features_manifest.csv                              1.4 KB
  fews_net_ipc_admin1 (1).csv                   311841.7 KB
  fews_net_ipc_admin1.csv                       311504.7 KB
  fewsnet_ipc.csv                                 1027.6 KB
  model_dataset.csv                               2056.2 KB
  panel_dataset.csv                               2363.9 KB
  panel_dataset_clean.csv                         3844.5 KB
  panel_dataset_clean_final.csv                   4197.6 KB
  panel_dataset_final.csv                         4276.6 KB
  panel_dataset_final_extended.csv                4644.1 KB
  panel_dataset_fuzzy.csv                       138218.8 KB
  panel_dataset_production.csv                    2072.8 KB
  panel_engineered.csv                            3218.4 KB
  panel_engineered_final.csv                      3555.1 KB
  y_test.csv                                        16.5 KB
  y_train.csv                                       26.5 KB
  y_val.csv                                          7.6 KB

```

---
## Step 3 — Configuration

####  Output
```text
  Using hardcoded credentials (set Colab Secrets for safety).

Date range      : 2018-01-01 → 2026-01-01
All countries   : 20
Excluded        : ['Burundi', 'Eritrea', 'Madagascar', 'Malawi', 'Uganda', 'Zimbabwe']
Panel countries : 14 → ['Ethiopia', 'Somalia', 'Sudan', 'South Sudan', 'Kenya', 'Nigeria', 'Niger', 'Mali', 'Burkina Faso', 'Chad', 'Cameroon', 'Mozambique', 'Democratic Republic of Congo', 'Central African Republic']

```

---
## Step 4 — Load & Aggregate ACLED Conflict Data

#### 🏁 Output
```text
Loading existing ACLED file: acled_africa_raw.csv

Total events  : 207,682
Countries     : 20
Date range    : 2018-01-01 → 2025-04-01
Fatalities    : 355,826

Events per country:
country
Nigeria                         29998
Democratic Republic of Congo    29319
Somalia                         24573
Sudan                           24193
Cameroon                        16404
Burkina Faso                    11406
Kenya                           10984
Ethiopia                        10890
Mali                            10864
South Sudan                     10311
Uganda                           4436
Niger                            4347
Burundi                          4296
Central African Republic         4213
Mozambique                       4167
Madagascar                       3166
Zimbabwe                         1434
Chad                             1378
Malawi                           1245
Eritrea                            58

```

---
## Step 5 — ACLED Monthly Aggregation

#### 🏁 Output
```text
 ACLED monthly aggregated:
  Rows      : 40,128
  Countries : 20
  Regions   : 450
  Columns   : ['country', 'admin1', 'year_month', 'fatalities_30d', 'events_30d', 'battle_events', 'civilian_violence', 'conflict_trend']
        country             admin1 year_month  fatalities_30d  events_30d  \
0  Burkina Faso  Boucle Du Mouhoun    2018-01           0.000       1.000   
1  Burkina Faso  Boucle Du Mouhoun    2018-02           0.000       0.000   
2  Burkina Faso  Boucle Du Mouhoun    2018-03           1.000       1.000   

   battle_events  civilian_violence  conflict_trend  
0          1.000              0.000               0  
1          0.000              0.000               0  
2          1.000              0.000               1  
```

---
## Step 6 — Process CHIRPS Rainfall (Zonal Stats → Anomalies)

#### 🏁 Output
```text
Loaded CHIRPS: 44,523 rows
Columns      : ['country', 'admin1', 'year_month', 'rainfall_mm', 'rainfall_anomaly', 'is_drought', 'is_flood']
Sample head  :
        country             admin1 year_month  rainfall_mm  rainfall_anomaly  is_drought  is_flood
0  burkina faso  boucle du mouhoun    2018-01       44.612            -1.406           1         0
1  burkina faso  boucle du mouhoun    2018-02       58.063             1.990           0         1
2  burkina faso  boucle du mouhoun    2018-03       63.212             1.041           0         0

After dedup: 44,523 rows (removed 0)

 CHIRPS processed:
  Rows          : 44,523
  Countries     : 20
  Admin regions : 451
  Drought ratio : 28.9%
  Flood ratio   : 7.2%
        country             admin1 year_month  rainfall_mm  rainfall_anomaly  \
0  burkina faso  boucle du mouhoun    2018-01       44.612            -1.406   
1  burkina faso  boucle du mouhoun    2018-02       58.063             1.990   
2  burkina faso  boucle du mouhoun    2018-03       63.212             1.041   

   is_drought  is_flood  
0           1         0  
1           0         1  
2           0         0  
```

---
## Step 7 — Load & Process FEWS NET IPC Data

#### 🏁 Output
```text
Loaded CHIRPS: 44,523 rows
Columns      : ['country', 'admin1', 'year_month', 'rainfall_mm', 'rainfall_anomaly', 'is_drought', 'is_flood']
Sample head  :
        country             admin1 year_month  rainfall_mm  rainfall_anomaly  is_drought  is_flood
0  burkina faso  boucle du mouhoun    2018-01       44.612            -1.406           1         0
1  burkina faso  boucle du mouhoun    2018-02       58.063             1.990           0         1
2  burkina faso  boucle du mouhoun    2018-03       63.212             1.041           0         0

After dedup: 44,523 rows (removed 0)

 CHIRPS processed:
  Rows          : 44,523
  Countries     : 20
  Admin regions : 451
  Drought ratio : 28.9%
  Flood ratio   : 7.2%
        country             admin1 year_month  rainfall_mm  rainfall_anomaly  \
0  burkina faso  boucle du mouhoun    2018-01       44.612            -1.406   
1  burkina faso  boucle du mouhoun    2018-02       58.063             1.990   
2  burkina faso  boucle du mouhoun    2018-03       63.212             1.041   

   is_drought  is_flood  
0           1         0  
1           0         1  
2           0         0  
```

---
## Step 8 — Three-Way Panel Merge

#### 🏁 Output
```text
ACLED  : 40,128 rows
FEWS   : 28,170 rows
CHIRPS : 44,523 rows

Fuzzy map size: 380 ACLED→FEWS region pairs

 Full panel: 28,170 rows, 14 countries
  Missing conflict : 0
  Missing CHIRPS   : 1304

```

---
## Step 9 — Production Panel (Reliable Countries Only)

#### 🏁 Output
```text
=== PRODUCTION PANEL ===
  Rows              : 26,954
  Countries         : 13
  Regions           : 341
  Date range        : 2018-02 → 2026-01
  Crisis rate       : 39.2%
  Class imbalance   : 1.5:1
  Duplicate rows    : 0 

Correlations with crisis_90d:
  ipc_lag1       : 0.8440  (target: > 0.60)
  conflict_trend : 0.0320
  is_drought     : -0.0006

Excluded countries:
  ✗ Burundi: absent from FEWS NET
  ✗ Eritrea: absent from FEWS NET
  ✗ Madagascar: absent from FEWS NET
  ✗ Malawi: absent from FEWS NET
  ✗ Uganda: admin-2/admin-1 mismatch
  ✗ Zimbabwe: absent from FEWS NET

 Saved → panel_dataset.csv
                         count     mean    std      min      25%      50%  \
ipc_phase            26954.000    2.251  0.987    1.000    1.000    2.000   
population_in_crisis 26954.000    0.372  0.483    0.000    0.000    0.000   
crisis_90d           26954.000    0.392  0.488    0.000    0.000    0.000   
ipc_lag1             26611.000    2.246  0.985    1.000    1.000    2.000   
ipc_lag2             26268.000    2.242  0.983    1.000    1.000    2.000   
fatalities_30d       26954.000   10.191 49.100    0.000    0.000    0.000   
events_30d           26954.000    5.268 15.459    0.000    0.000    0.000   
battle_events        26954.000    2.163 10.199    0.000    0.000    0.000   
civilian_violence    26954.000    1.436  4.377    0.000    0.000    0.000   
conflict_trend       26954.000   -0.126  0.580   -1.000    0.000    0.000   
rainfall_mm          26866.000   50.661  7.525   33.355   46.679   50.957   
rainfall_anomaly     26954.000   -0.005  0.785   -2.346   -0.362    0.000   
is_drought           26954.000    0.209  0.406    0.000    0.000    0.000   
is_flood             26954.000    0.051  0.220    0.000    0.000    0.000   
is_lean_season       26954.000    0.329  0.470    0.000    0.000    0.000   
is_harvest_season    26954.000    0.259  0.438    0.000    0.000    0.000   
year                 26954.000 2022.075  2.182 2018.000 2020.000 2022.000   

                          75%      max  
ipc_phase               3.000    5.000  
population_in_crisis    1.000    1.000  
crisis_90d              1.000    1.000  
ipc_lag1                3.000    5.000  
ipc_lag2                3.000    5.000  
fatalities_30d          2.000 2408.000  
events_30d              4.000  545.000  
battle_events           1.000  440.000  
civilian_violence       1.000  125.000  
conflict_trend          0.000    1.000  
rainfall_mm            55.535   68.446  
rainfall_anomaly        0.264    2.190  
is_drought              0.000    1.000  
is_flood                0.000    1.000  
is_lean_season          1.000    1.000  
is_harvest_season       1.000    1.000  
year                 2024.000 2026.000  
```

---
## Step 10 — Quality Report & Validation

#### 🏁 Output
```text
=== TASK 1 COMPLETE ===
{
  "task": "Task 1 \u2014 Data Fusion & Panel Creation",
  "panel_rows": 26954,
  "countries": 13,
  "regions": 341,
  "time_periods": 96,
  "date_start": "2018-02",
  "date_end": "2026-01",
  "crisis_rate": 0.3923,
  "class_imbalance": 1.55,
  "scale_pos_weight": 1.55,
  "missing_conflict_pct": 0.0,
  "missing_chirps_pct": 0.33,
  "drought_pct": 20.87,
  "ipc_lag1_corr": 0.844,
  "conflict_trend_corr": 0.032,
  "drought_corr": -0.0006,
  "feature_columns": [
    "ipc_lag1",
    "ipc_lag2",
    "fatalities_30d",
    "events_30d",
    "battle_events",
    "civilian_violence",
    "conflict_trend",
    "rainfall_mm",
    "rainfall_anomaly",
    "is_drought",
    "is_flood",
    "is_lean_season",
    "is_harvest_season",
    "year"
  ],
  "target_column": "crisis_90d",
  "excluded_countries": [
    "Burundi",
    "Eritrea",
    "Madagascar",
    "Malawi",
    "Uganda",
    "Zimbabwe"
  ],
  "fews_absent": [
    "Burundi",
    "Eritrea",
    "Madagascar",
    "Malawi",
    "Zimbabwe"
  ]
}

 data_quality_report.json saved
 Pass panel_dataset.csv → Task 2

```

---
## Step 11 — Backup to Google Drive



# Task 2: Feature Engineering & Splits

**File:** `crisis_task2_feature_engineering.ipynb`

---
# Conflict-Induced Food Crisis Prediction
## Task 2: Feature Engineering & Exploratory Data Analysis  v2.0

**Input** : `panel_dataset.csv` (from Task 1) + `data_quality_report.json`  
**Output** : `features_engineered.csv`, train/val/test splits, charts, `split_report.json`

---

### Notebook Structure

| Step | Section | What it does |
|------|---------|-------------|
| 1 | Setup | Install, import, auto-detect environment |
| 2 | Load Panel | Read Task 1 output, validate schema |
| 3 | Lag Features | Conflict & rainfall temporal lags (1–3 months) |
| 4 | Rolling Features | 2-month and 3-month rolling averages |
| 5 | IPC Trend | Slope of IPC phase over prior 3 months |
| 6 | Compound Risk | Conflict × drought interaction features |
| 7 | Feature Catalogue | Document every feature with source & correlation |
| 8 | EDA | 6-panel chart — crisis rate, country breakdown, distributions |
| 9 | RF Pre-screen | 100-tree importance ranking before Task 3 |
| 10 | Walk-Forward Split | Temporal train/val/test — strict no-leakage |
| 11 | Save Artifacts | All CSVs + split_report.json |
| 12 | Drive Backup | Google Drive (Colab) or local summary |

---

### Anti-leakage rules enforced in this notebook
- All lag/rolling features use `.shift(1)` before rolling → **no current-month leakage**
- `crisis_90d` uses only **forward** IPC shifts — confirmed in Task 1
- Temporal split is **strictly ordered** — no row from the future ever enters training
- Test set is **sealed** — not opened until Task 4


---
## Step 1 — Setup & Auto-Detect Environment

#### 🏁 Output
```text
Installing scikit-learn...
Environment : Google Colab
Data dir    : /content/crisis_outputs
Existing CSVs:

 Setup complete.

```

#### 🏁 Output
```text
Mounted at /content/drive
Restoring from /content/drive/MyDrive/crisis_outputs_backup ...
 Restore complete.

Existing CSVs (26):
  X_test.csv                                      1289.3 KB
  X_train.csv                                     2207.3 KB
  X_val.csv                                        621.3 KB
  acled_africa_raw.csv                           10334.1 KB
  acled_monthly.csv                               1766.5 KB
  chirps_monthly.csv                              2016.7 KB
  chirps_processed.csv                            2016.7 KB
  feature_importance.csv                             2.0 KB
  features_engineered.csv                         5791.2 KB
  features_manifest.csv                              1.4 KB
  fews_net_ipc_admin1 (1).csv                   311841.7 KB
  fews_net_ipc_admin1.csv                       311504.7 KB
  fewsnet_ipc.csv                                 1027.6 KB
  model_dataset.csv                               2056.2 KB
  panel_dataset.csv                               2363.9 KB
  panel_dataset_clean.csv                         3844.5 KB
  panel_dataset_clean_final.csv                   4197.6 KB
  panel_dataset_final.csv                         4276.6 KB
  panel_dataset_final_extended.csv                4644.1 KB
  panel_dataset_fuzzy.csv                       138218.8 KB
  panel_dataset_production.csv                    2072.8 KB
  panel_engineered.csv                            3218.4 KB
  panel_engineered_final.csv                      3555.1 KB
  y_test.csv                                        16.5 KB
  y_train.csv                                       26.5 KB
  y_val.csv                                          7.6 KB

```

---
## Step 2 — Load Panel & Validate Schema

#### 🏁 Output
```text
=== PANEL LOADED ===
  Rows           : 26,954
  Columns        : 20
  Countries      : 13
  Regions        : 341
  Date range     : 2018-02 → 2026-01
  Crisis rate    : 39.2%
  Class imbalance: 1.5:1
  scale_pos_weight (Task 1): 1.55

Nulls per column:
ipc_lag1       343
ipc_lag2       686
rainfall_mm     88

 Schema validated — all 19 required columns present.

```

---
## Step 3 — Temporal Lag Features

**Why lags matter**: Conflict in month T doesn't immediately cause food insecurity.  
The causal chain takes 2–3 months: conflict → market disruption → harvest failure → IPC escalation.

| Feature | Logic | Leakage? |
|---|---|---|
| `fatalities_lag1` | Fatalities from 1 month ago |  No — uses `.shift(1)` |
| `fatalities_lag2` | Fatalities from 2 months ago |  No |
| `fatalities_lag3` | Fatalities from 3 months ago |  No |
| `events_lag1/2` | Event count lags |  No |
| `rainfall_lag1/2` | Rainfall anomaly lags |  No |


#### 🏁 Output
```text
 Lag features created (8): ['fatalities_lag1', 'events_lag1', 'fatalities_lag2', 'events_lag2', 'fatalities_lag3', 'events_lag3', 'rainfall_lag1', 'rainfall_lag2']
Panel shape: (26954, 28)

```

---
## Step 4 — Rolling Window Features

**Why rolling averages**: A single violent month could be random.  
Three consecutive months of rising fatalities is a genuine escalation signal.  
All rolling features use `.shift(1)` before rolling — no current-month leakage.


#### 🏁 Output
```text
 Rolling features created (6): ['fatalities_roll2', 'fatalities_roll3', 'events_roll3', 'rainfall_roll3', 'fatalities_delta', 'fatalities_accel']

Leakage check — first 5 rows for Burkina Faso/Boucle Du Mouhoun:
year_month  fatalities_30d  fatalities_roll2  fatalities_roll3
   2021-06          0.0000               NaN               NaN
   2021-07          0.0000            0.0000            0.0000
   2021-08         15.0000            0.0000            0.0000
   2021-09          3.0000            7.5000            5.0000
   2021-10         21.0000            9.0000            6.0000

 Roll-2 at row 2 = mean of row 0 only (shift=1 confirmed no leakage)

```

---
## Step 5 — IPC Trend Feature

**IPC trend** = the slope (gradient) of the IPC phase over the prior 3 months.  
- Positive (↑): worsening food security — strong leading indicator  
- Zero: stable  
- Negative (↓): improving

This is computed only from **past data** (`.shift(1)` before the rolling window).


#### 🏁 Output
```text
 ipc_trend created.
  Mean ipc_trend (crisis=1) : 0.0377
  Mean ipc_trend (crisis=0) : -0.0120
  Correlation with crisis_90d: 0.1442

```

---
## Step 6 — Compound Risk & Interaction Features

**Why compound features**: High conflict AND drought simultaneously is far worse than either alone.  
Neither `fatalities_30d` nor `is_drought` alone captures their synergy — an explicit  
interaction term forces the model to consider the combined effect.

| Feature | Formula | Interpretation |
|---|---|---|
| `high_conflict` | fatalities > median | Binary: above-average violence |
| `high_conflict_drought` | high_conflict AND is_drought | Compound shock flag |
| `compound_risk_score` | 0.6×conflict_norm + 0.4×drought_severity | Weighted severity index 0–1 |
| `sustained_conflict` | lag1 > 0 AND lag2 > 0 | At least 2 consecutive months of violence |
| `battle_share` | battle_events / events_30d | % of events that are direct combat |
| `lean_drought` | is_lean_season AND is_drought | Drought in hunger gap month |


#### 🏁 Output
```text
 Compound & interaction features created:
  high_conflict                 crisis avg=0.381  safe avg=0.292  diff=+0.089
  high_conflict_drought         crisis avg=0.122  safe avg=0.082  diff=+0.040
  compound_risk_score           crisis avg=0.100  safe avg=0.037  diff=+0.063
  sustained_conflict            crisis avg=0.320  safe avg=0.197  diff=+0.123
  battle_share                  crisis avg=0.178  safe avg=0.092  diff=+0.086
  lean_drought                  crisis avg=0.078  safe avg=0.066  diff=+0.011

Panel shape: (26954, 44)

```

---
## Step 7 — Feature Catalogue & Correlation Ranking

Every feature is documented with its **data source**, **type**, and **correlation with `crisis_90d`**.  
This auditable record is required for humanitarian AI transparency and reproducibility.


#### 🏁 Output
```text
 Feature catalogue saved (32 features)

Top 15 features by |correlation| with crisis_90d:
            feature         category  corr_target
           ipc_lag1      IPC History       0.8440
           ipc_lag2      IPC History       0.8213
compound_risk_score    Compound Risk       0.2042
   fatalities_roll3 Conflict Rolling       0.1741
       battle_share Conflict Current       0.1663
   fatalities_roll2 Conflict Rolling       0.1637
       events_roll3 Conflict Rolling       0.1602
        events_lag2  Conflict Lagged       0.1537
        events_lag1  Conflict Lagged       0.1515
    fatalities_lag3  Conflict Lagged       0.1508
    fatalities_lag2  Conflict Lagged       0.1505
         events_30d Conflict Current       0.1492
          ipc_trend      IPC History       0.1442
  civilian_violence Conflict Current       0.1442
    fatalities_lag1  Conflict Lagged       0.1439

Features by category:
category
Compound Risk       5
Conflict Current    6
Conflict Lagged     5
Conflict Rolling    5
IPC History         3
Rainfall            5
Seasonal            3

```

---
## Step 8 — Exploratory Data Analysis (EDA)

Six-panel visualization covering every key dimension of the dataset:  
crisis trends over time, country breakdown, IPC distribution, conflict severity,  
compound risk, and feature correlation heatmap.


#### 🏁 Output
```text
Rows after dropna: 25,925 (from 26,954, dropped 1,029 lag-NaN rows)
Crisis rate: 39.7%
<Figure size 2000x1600 with 7 Axes> eda_charts.png saved.

```

---
## Step 9 — Random Forest Feature Importance Pre-Screen

A lightweight **100-tree Random Forest** on the training partition gives a  
model-based importance ranking *before* any hyperparameter tuning.

**Purpose**: Confirm that `ipc_lag1` dominates (validates data quality),  
and identify any features that contribute essentially zero importance (candidates for removal).


#### 🏁 Output
```text
Pre-screen train set: 9,722 rows  crisis rate: 27.3%
Training 100-tree Random Forest...

Rank Feature                       Importance  Cumulative
------------------------------------------------------------
1    ipc_lag1                          0.4213      0.4213
2    ipc_lag2                          0.2852      0.7065
3    ipc_trend                         0.1509      0.8574
4    fatalities_roll3                  0.0184      0.8758
5    fatalities_roll2                  0.0126      0.8883
6    compound_risk_score               0.0123      0.9006
7    battle_share                      0.0099      0.9105
8    battle_events                     0.0097      0.9203
9    fatalities_lag3                   0.0085      0.9288
10   events_roll3                      0.0075      0.9363
11   fatalities_30d                    0.0074      0.9437
12   rainfall_lag2                     0.0062      0.9499
13   rainfall_anomaly                  0.0057      0.9556
14   rainfall_roll3                    0.0056      0.9612
15   fatalities_lag1                   0.0055      0.9667
16   month                             0.0050      0.9717
17   rainfall_lag1                     0.0050      0.9767
18   fatalities_lag2                   0.0048      0.9816
19   events_lag1                       0.0028      0.9844
20   events_30d                        0.0027      0.9871
<Figure size 1100x800 with 1 Axes>
 feature_importance_prescreen.png saved.

Features with importance < 0.5% (15): ['fatalities_lag2', 'events_lag1', 'events_30d', 'events_lag2', 'fatalities_accel', 'civilian_violence', 'fatalities_delta', 'sustained_conflict', 'is_lean_season', 'is_harvest_season', 'conflict_trend', 'high_conflict', 'is_drought', 'high_conflict_drought', 'lean_drought']
These are candidates for removal in Task 3 if needed.

```

---
## Step 10 — Walk-Forward Temporal Split (No Leakage)

**Why temporal ordering is mandatory for time-series panel data:**  
Standard k-fold randomises rows — it would train on October 2022 while validating on  
March 2021, directly leaking future information. Walk-forward enforces strict ordering.

```
────────────────────────────────────────────────────────────────►  time
│         TRAIN          │      VALIDATE      │      TEST (SEALED)   │
│  2018-02  →  2021-12   │ 2022-01 → 2022-12  │  2023-01 → end       │
└────────────────────────┴────────────────────┴──────────────────────┘
```

Three assertions verify zero temporal overlap between any pair of splits.


#### 🏁 Output
```text
=== WALK-FORWARD SPLIT ===
Split        Rows                Date Range  Crisis Rate  Safe:Crisis
------------------------------------------------------------------------
Train      13,574         2018-05 → 2022-12        29.0%        2.4:1
Val         3,892         2023-01 → 2023-12        48.8%        1.1:1
Test        8,459         2024-01 → 2026-01        52.7%        0.9:1  ← SEALED

 Leakage audit PASSED — zero overlap between all three splits

scale_pos_weight for XGBoost Task 3: 2.4487
  (= safe count / crisis count = 2.45)

```

---
## Step 11 — Save All Artifacts

#### 🏁 Output
```text
=== TASK 2 COMPLETE ===

Feature set (32 features):
  Compound Risk             5 features
  Conflict Current          6 features
  Conflict Lagged           5 features
  Conflict Rolling          5 features
  IPC History               3 features
  Rainfall                  5 features
  Seasonal                  3 features

Walk-forward splits saved:
  train  : 13,574 rows  (2018-05 → 2022-12)  crisis: 29.0%
  val    : 3,892 rows  (2023-01 → 2023-12)  crisis: 48.8%
  test   : 8,459 rows  (2024-01 → 2026-01)  crisis: 52.7%

Files saved to /content/crisis_outputs:
  X_test.csv                                      1289.0 KB
  X_train.csv                                     2205.2 KB
  X_val.csv                                        621.0 KB
  acled_africa_raw.csv                           10334.1 KB
  acled_monthly.csv                               1766.5 KB
  chirps_monthly.csv                              2016.7 KB
  chirps_processed.csv                            2016.7 KB
  feature_importance.csv                             2.0 KB
  features_engineered.csv                         5788.4 KB
  features_manifest.csv                              1.4 KB
  fews_net_ipc_admin1 (1).csv                   311841.7 KB
  fews_net_ipc_admin1.csv                       311504.7 KB
  fewsnet_ipc.csv                                 1027.6 KB
  model_dataset.csv                               2056.2 KB
  panel_dataset.csv                               2363.9 KB
  panel_dataset_clean.csv                         3844.5 KB
  panel_dataset_clean_final.csv                   4197.6 KB
  panel_dataset_final.csv                         4276.6 KB
  panel_dataset_final_extended.csv                4644.1 KB
  panel_dataset_fuzzy.csv                       138218.8 KB
  panel_dataset_production.csv                    2072.8 KB
  panel_engineered.csv                            3218.4 KB
  panel_engineered_final.csv                      3555.1 KB
  y_test.csv                                        16.5 KB
  y_train.csv                                       26.5 KB
  y_val.csv                                          7.6 KB
  baseline_results.json                              0.3 KB
  best_model.json                                  764.3 KB
  data_quality_report.json                           1.0 KB
  data_quality_report_extended.json                  0.3 KB
  data_quality_report_final.json                     0.2 KB
  split_report.json                                  1.5 KB
  task3_results.json                                 2.7 KB
  test_results.json                                  3.0 KB
  country_performance.png                          133.3 KB
  data_quality_charts.png                          188.1 KB
  eda_charts.png                                   391.7 KB
  error_analysis.png                               238.8 KB
  error_analysis_countries.png                     126.9 KB
  feature_importance_final.png                     116.2 KB
  feature_importance_prescreen.png                  91.8 KB
  final_evaluation.png                             109.5 KB
  model_comparison.png                              81.8 KB
  pr_roc_curves.png                                 98.1 KB
  shap_beeswarm.png                                160.9 KB
  shap_summary.png                                 104.2 KB
  shap_waterfall_fn.png                            139.6 KB
  shap_waterfall_highrisk.png                      133.2 KB

 Pass to Task 3:
   X_train.csv, y_train.csv  →  model training
   X_val.csv,   y_val.csv    →  hyperparameter tuning
   X_test.csv,  y_test.csv   →  Task 4 evaluation only (SEALED)
   scale_pos_weight = 2.4487  →  use in XGBoostClassifier

```

---
## Step 12 — Backup to Google Drive (Colab) or Local Summary

---
## Task 2 Summary & Handoff to Task 3

### Features Engineered (32 total)

| Category | Features | Key insight |
|---|---|---|
| **IPC History** | `ipc_lag1`, `ipc_lag2`, `ipc_trend` | Correlation 0.77–0.81 — dominant signal |
| **Conflict Current** | `fatalities_30d`, `events_30d`, `battle_events`, `civilian_violence`, `conflict_trend`, `battle_share` | Immediate violence signal |
| **Conflict Lagged** | `fatalities_lag1/2/3`, `events_lag1/2` | 2–3 month causal delay |
| **Conflict Rolling** | `fatalities_roll2/3`, `events_roll3`, `delta`, `accel` | Sustained violence patterns |
| **Compound Risk** | `high_conflict_drought`, `compound_risk_score`, `sustained_conflict`, `lean_drought` | Non-linear shock synergy |
| **Rainfall** | `rainfall_anomaly`, `is_drought`, `lag1/2`, `roll3` | Drought compounds displacement |
| **Seasonal** | `is_lean_season`, `is_harvest_season`, `month` | Agricultural calendar effects |

### Key Numbers for Task 3

| Parameter | Value |
|---|---|
| `scale_pos_weight` | Printed above — use in XGBoostClassifier |
| Dominant feature | `ipc_lag1` (~38% RF importance) |
| Anti-leakage |  PASSED — strict temporal ordering enforced |
| Test set |  **SEALED** — do not open until Task 4 |

### Next → Task 3: Baseline Models, XGBoost & Hyperparameter Tuning




# Task 3: Baseline Models & XGBoost Tuning

**File:** `crisis_task3_model_training.ipynb`

---
# Conflict-Induced Food Crisis Prediction
## Task 3: Baseline Models, XGBoost & Hyperparameter Tuning  v2.0

**Input**  : `X_train/val/test.csv`, `y_train/val/test.csv`, `split_report.json`  
**Output** : `best_model.pkl`, `task3_results.json`, `model_comparison.png`, `feature_importance_final.png`

---

### Notebook Structure

| Step | Section | What it does |
|------|---------|-------------|
| 1 | Setup | Install, import, auto-detect environment |
| 2 | Load Splits | Read Task 2 outputs, validate shapes |
| 3 | Baseline Models | Logistic Regression + Random Forest benchmarks |
| 4 | XGBoost Default | XGBoost with `scale_pos_weight` — no tuning |
| 5 | Hyperparameter Tuning | Grid search on validation F1 (no test leakage) |
| 6 | Best Model Evaluation | Full metrics on validation set |
| 7 | Model Comparison Chart | All models side-by-side |
| 8 | Feature Importance | XGBoost gain-based importance |
| 9 | Save Artifacts | Pickle model + JSON results |
| 10 | Drive Backup | Google Drive (Colab) or local summary |

---

### Evaluation Metric Priority
`crisis_90d` prediction is a **humanitarian early-warning task** — missing a real crisis  
is far more costly than a false alarm. Therefore:  

| Metric | Priority | Reason |
|---|---|---|
| **Recall** (sensitivity) | 🥇 Primary | Minimize missed crises |
| **F1-score** | 🥈 Tune on this | Balance precision and recall |
| **PR-AUC** | 🥉 Report | Better than ROC-AUC for imbalanced data |
| ROC-AUC | Report only | Can be misleading for imbalanced classes |


---
## Step 1 — Setup & Auto-Detect Environment

#### 🏁 Output
```text
Installing scikit-learn...
Environment     : Google Colab
Data dir        : /content/crisis_outputs
XGBoost version : 3.2.0
sklearn version : 1.6.1

✅ Setup complete.

```

#### 🏁 Output
```text
Mounted at /content/drive

```

#### 🏁 Output
```text
 Google Drive already mounted.
Restoring from /content/drive/MyDrive/crisis_outputs_backup → /content/crisis_outputs ...
 Restored 59 files/directories.
   X_train.csv                                     2207.3 KB
   X_val.csv                                        621.3 KB
   X_test.csv                                      1289.3 KB
   y_train.csv                                       26.5 KB
   y_val.csv                                          7.6 KB
   y_test.csv                                        16.5 KB
   split_report.json                                  1.5 KB
   features_engineered.csv                         5791.2 KB

 All required files present — ready to run Task 3.

```

---
## Step 2 — Load Train / Val / Test Splits

#### 🏁 Output
```text
=== SPLITS LOADED ===
Split        Rows   Features   Crisis%  Safe:Crisis
----------------------------------------------------
Train      13,574         32     29.0%       2.4:1
Val         3,892         32     48.8%       1.1:1
Test        8,459         32     52.7%       0.9:1  [SEALED]

scale_pos_weight  : 2.4487
Feature count     : 32

 All splits validated — training medians used for imputation (no leakage).

```

---
## Step 3 — Baseline Models

Two interpretable baselines establish a performance floor:
- **Logistic Regression** — linear, fast, interpretable coefficients
- **Random Forest** — non-linear, strong benchmark, no hyperparameter tuning

Both use `class_weight='balanced'` to handle the 2.4:1 class imbalance.  
Evaluation is **on the validation set only** — test set remains sealed.


#### 🏁 Output
```text
Training Logistic Regression...
  F1=0.9406  Recall=0.8962  PR-AUC=0.9755  ROC-AUC=0.9684

Training Random Forest (200 trees)...
  F1=0.9435  Recall=0.8930  PR-AUC=0.9729  ROC-AUC=0.9637

 Baseline models trained.
   LR F1 = 0.9406 | RF F1 = 0.9435
   Target to beat with XGBoost: F1 > 0.9435

```

---
## Step 4 — XGBoost Default (No Tuning)

XGBoost with `scale_pos_weight` from Task 2.  
This is the **before-tuning** benchmark — shows how much tuning adds.

Key settings:
- `scale_pos_weight` = safe / crisis ratio from training set
- `eval_metric='aucpr'` — PR-AUC is the right metric for imbalanced data
- `early_stopping_rounds=30` — prevents overfitting on the val set during training


#### 🏁 Output
```text
Training XGBoost (default, scale_pos_weight=2.45)...
  Best iteration : 353
  F1        = 0.9432  (vs RF=0.9435, Δ=-0.0003)
  Recall    = 0.8967
  Precision = 0.9947
  PR-AUC    = 0.9906
  ROC-AUC   = 0.9903

```

---
## Step 5 — Hyperparameter Tuning (Validation Set Grid Search)

Grid search across key XGBoost hyperparameters.  
**All selection is done on the validation set F1** — test set is never touched.

### Search space
| Parameter | Values | Reason |
|---|---|---|
| `max_depth` | 4, 6, 8 | Controls over/underfitting |
| `learning_rate` | 0.01, 0.05, 0.1 | Controls step size |
| `subsample` | 0.7, 0.9 | Row sampling — regularisation |
| `colsample_bytree` | 0.7, 0.9 | Feature sampling per tree |
| `min_child_weight` | 3, 5, 10 | Min samples per leaf — prevents overfitting on rare crises |


#### 🏁 Output
```text
Grid size: 108 combinations — searching on val F1...
  [ 36/108] Best F1 so far: 0.9446 (depth=4, lr=0.05)
  [ 72/108] Best F1 so far: 0.9446 (depth=4, lr=0.05)
  [108/108] Best F1 so far: 0.9446 (depth=4, lr=0.05)

 Grid search complete.
   Best val F1   : 0.9446
   Best params   : {'max_depth': 4, 'learning_rate': 0.05, 'subsample': 0.9, 'colsample_bytree': 0.7, 'min_child_weight': 5}

Top 5 parameter combinations:
 max_depth  learning_rate  subsample  colsample_bytree  min_child_weight     f1  pr_auc  best_iter
         4         0.0100     0.9000            0.7000                 5 0.9446  0.9832        526
         4         0.0500     0.9000            0.7000                 5 0.9446  0.9890        499
         8         0.0100     0.7000            0.9000                 5 0.9446  0.9774         14
         6         0.0500     0.9000            0.9000                 3 0.9444  0.9897        259
         8         0.0500     0.7000            0.7000                10 0.9444  0.9882        250

```

---
## Step 6 — Best Model Full Evaluation (Validation Set)

Comprehensive evaluation of the tuned XGBoost on the validation set:
- Classification report (precision / recall / F1 per class)
- Confusion matrix
- Optimal probability threshold selection (maximise F1)
- PR curve and ROC curve


#### 🏁 Output
```text
=== BEST XGBoost — VAL SET EVALUATION ===
Best params: {'max_depth': 4, 'learning_rate': 0.05, 'subsample': 0.9, 'colsample_bytree': 0.7, 'min_child_weight': 5}
Best iteration: 499

Classification report (threshold=0.50):
              precision    recall  f1-score   support

    Safe (0)     0.9118    0.9960    0.9521      1994
  Crisis (1)     0.9953    0.8988    0.9446      1898

    accuracy                         0.9486      3892
   macro avg     0.9536    0.9474    0.9483      3892
weighted avg     0.9526    0.9486    0.9484      3892

Confusion matrix:
  TN=1,986  FP=8
  FN=192  TP=1,706
  False Negative Rate (missed crises): 10.1%  ← minimise this
  False Positive Rate (false alarms) : 0.4%

Optimal threshold: 0.497  (F1=0.9449)
              precision    recall  f1-score   support

    Safe (0)     0.9123    0.9960    0.9523      1994
  Crisis (1)     0.9953    0.8994    0.9449      1898

    accuracy                         0.9489      3892
   macro avg     0.9538    0.9477    0.9486      3892
weighted avg     0.9528    0.9489    0.9487      3892

<Figure size 1400x500 with 2 Axes>
 pr_roc_curves.png saved.

=== HUMANITARIAN THRESHOLD (Recall ≥ 0.95) ===
  Threshold  : 0.077  (vs F1-optimal=0.497)
  Recall     : 0.9568  ← catches 95.7% of crises
  Precision  : 0.8937
  F1         : 0.9242
  FN (missed): 82  (was 192 at F1-optimal threshold)
  FP (alarms): 216  (was 8 at F1-optimal threshold)

  Trade-off summary:
    Missed crises reduced : 110 fewer FN (192 → 82)
    Extra false alarms    : 208 more FP (8 → 216)

  Recommendation: use threshold=0.077 for deployment
  (lower threshold → more alerts → fewer missed crises)

```

---
## Step 6b — Error Analysis: Where Are the Missed Crises?

The 192 False Negatives (missed crisis warnings) are the most important diagnostic for
a humanitarian early-warning system. Missed warnings concentrated in specific countries
indicate a **geographic blind spot** — the model systematically underestimates crisis
risk in those regions.

**Why this matters:**
- DRC and South Sudan have chaotic conflict patterns that don't match historical IPC persistence
- If FNs cluster there, we need country-specific features or separate sub-models
- This directly informs Task 4 SHAP analysis: which feature values pushed the model to predict "safe" when the region was actually in crisis


#### 🏁 Output
```text
=== FALSE NEGATIVE ANALYSIS (Missed Crises by Country) ===
Total FN: 191  |  Total FP: 8

Country                                FN   Total   FN Rate   Avg Prob
----------------------------------------------------------------------
Sudan                                 119     654     18.2%      0.090
Chad                                   23      84     27.4%      0.188  ← HIGH RISK BLINDSPOT
Nigeria                                10     162      6.2%      0.095
Somalia                                10     400      2.5%      0.134
Kenya                                   8     170      4.7%      0.268
Ethiopia                                6      96      6.2%      0.402
Mozambique                              6      46     13.0%      0.285
Niger                                   4      44      9.1%      0.058
Cameroon                                2      34      5.9%      0.062
Mali                                    2      24      8.3%      0.183
Burkina Faso                            1      52      1.9%      0.301

IPC phase distribution of missed crises:
ipc_phase
2    98
3    70
4    23

Most commonly missed at phase: 2 (Phase 2→3 transitions, model predicted Phase 2 continuation)
<Figure size 1600x600 with 2 Axes>
 error_analysis_countries.png saved.

=== BLIND SPOT RECOMMENDATIONS ===
Countries with FNR > 20% (geographic blind spots):
    Chad: 27.4% FNR

Recommended actions:
  1. Add country fixed-effects or country-specific features in Task 4
  2. Check SHAP waterfall for these countries specifically
  3. Consider separate sub-models or higher recall threshold for blind spots

```

---
## Step 7 — Model Comparison (All Models vs Validation Set)

#### 🏁 Output
```text
=== MODEL COMPARISON (Validation Set) ===
              model     f1  recall  precision  pr_auc  roc_auc
Logistic Regression 0.9406  0.8962     0.9895  0.9755   0.9684
      Random Forest 0.9435  0.8930     1.0000  0.9729   0.9637
    XGBoost Default 0.9432  0.8967     0.9947  0.9906   0.9903
      XGBoost Tuned 0.9449  0.8994     0.9953  0.9890   0.9882

 Best model: XGBoost Tuned (F1=0.9449)
<Figure size 1400x600 with 1 Axes>
 model_comparison.png saved.

```

---
## Step 8 — XGBoost Feature Importance

Gain-based importance from the tuned XGBoost model.  
**Gain** = average reduction in loss when a feature is used for splitting — more directly  
interpretable than frequency-based (cover) importance.

Expected outcome: `ipc_lag1`, `ipc_lag2`, `ipc_trend` dominate with 80%+ cumulative gain.  
This confirms the model uses the correct signals — not spurious correlations.


#### 🏁 Output
```text
Feature Importance (gain-based, normalised):
 Rank Feature                        Gain%  Cumulative
----------------------------------------------------------
  1   ipc_lag1                      48.32%     0.4832
  2   ipc_lag2                      36.42%     0.8474
  3   ipc_trend                      6.08%     0.9082
  4   month                          0.96%     0.9178
  5   rainfall_anomaly               0.66%     0.9245
  6   rainfall_lag1                  0.59%     0.9304
  7   battle_events                  0.59%     0.9363
  8   fatalities_roll3               0.52%     0.9415
  9   rainfall_lag2                  0.52%     0.9467
  10  is_lean_season                 0.50%     0.9517
  11  rainfall_roll3                 0.41%     0.9558
  12  events_roll3                   0.37%     0.9595
  13  battle_share                   0.36%     0.9631
  14  sustained_conflict             0.33%     0.9665
  15  compound_risk_score            0.26%     0.9690
  16  events_30d                     0.24%     0.9714
  17  conflict_trend                 0.24%     0.9739
  18  fatalities_30d                 0.23%     0.9761
  19  fatalities_lag3                0.22%     0.9784
  20  events_lag1                    0.22%     0.9806

Features unused by XGBoost (zero gain): ['lean_drought']
<Figure size 1100x800 with 1 Axes>
 feature_importance_final.png saved.

```

---
## Step 9 — Save Best Model & Results

#### 🏁 Output
```text
 Best model saved: best_model.pkl  (693.3 KB)
 XGBoost native model saved: best_model.json  (764.3 KB)

=== TASK 3 COMPLETE ===

Best model      : XGBoost (tuned)
Best parameters : {'max_depth': 4, 'learning_rate': 0.05, 'subsample': 0.9, 'colsample_bytree': 0.7, 'min_child_weight': 5}
Best iteration  : 499

Validation set metrics (threshold=0.497):
  f1          : 0.9449
  recall      : 0.8994
  precision   : 0.9953
  pr_auc      : 0.9890
  roc_auc     : 0.9882

Top 5 features by gain:
  1. ipc_lag1                     48.32%
  2. ipc_lag2                     36.42%
  3. ipc_trend                    6.08%
  4. month                        0.96%
  5. rainfall_anomaly             0.66%

Files saved:
  best_model.pkl                                   693.3 KB
  best_model.json                                  764.3 KB
  task3_results.json                                 2.7 KB
  model_comparison.png                              81.8 KB
  feature_importance_final.png                     116.2 KB
  pr_roc_curves.png                                 98.1 KB
  feature_importance.csv                             2.0 KB

 Pass best_model.pkl + task3_results.json → Task 4 (Evaluation & SHAP)
   Optimal decision threshold = 0.497

```

---
## Step 10 — Backup to Google Drive

#### 🏁 Output
```text
 Backup complete → /content/drive/MyDrive/crisis_outputs_backup

Full task3_results.json:
{
  "task": "Task 3 \u2014 Model Training & Hyperparameter Tuning v2.0",
  "best_model": "XGBoost Tuned",
  "best_params": {
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.9,
    "colsample_bytree": 0.7,
    "min_child_weight": 5
  },
  "best_iteration": 499,
  "scale_pos_weight": 2.4487,
  "opt_threshold": 0.497,
  "val_metrics": {
    "f1": 0.9449,
    "recall": 0.8994,
    "precision": 0.9953,
    "pr_auc": 0.989,
    "roc_auc": 0.9882
  },
  "all_models": [
    {
      "model": "Logistic Regression",
      "threshold": 0.5,
      "f1": 0.9406,
      "recall": 0.8962,
      "precision": 0.9895,
      "roc_auc": 0.9684,
      "pr_auc": 0.9755
    },
    {
      "model": "Random Forest",
      "threshold": 0.5,
      "f1": 0.9435,
      "recall": 0.893,
      "precision": 1.0,
      "roc_auc": 0.9637,
      "pr_auc": 0.9729
    },
    {
      "model": "XGBoost Default",
      "threshold": 0.5,
      "f1": 0.9432,
      "recall": 0.8967,
      "precision": 0.9947,
      "roc_auc": 0.9903,
      "pr_auc": 0.9906,
      "best_iteration": 353
    },
    {
      "model": "XGBoost Tuned",
      "threshold": 0.4969618022441864,
      "f1": 0.9449,
      "recall": 0.8994,
      "precision": 0.9953,
      "roc_auc": 0.9882,
      "pr_auc": 0.989,
      "best_iteration": 499,
      "best_params": {
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.9,
        "colsample_bytree": 0.7,
        "min_child_weight": 5
      },
      "opt_threshold": 0.497,
      "hr_threshold": 0.0768,
      "hr_recall": 0.9568,
      "hr_fn_missed_crises": 82
    }
  ],
  "top5_features": [
    "ipc_lag1",
    "ipc_lag2",
    "ipc_trend",
    "month",
    "rainfall_anomaly"
  ],
  "feature_importance": [
    {
      "feature": "ipc_lag1",
      "gain_pct": 48.32
    },
    {
      "feature": "ipc_lag2",
      "gain_pct": 36.42
    },
    {
      "feature": "ipc_trend",
      "gain_pct": 6.08
    },
    {
      "feature": "month",
      "gain_pct": 0.96
    },
    {
      "feature": "rainfall_anomaly",
      "gain_pct": 0.66
    },
    {
      "feature": "rainfall_lag1",
      "gain_pct": 0.59
    },
    {
      "feature": "battle_events",
      "gain_pct": 0.59
    },
    {
      "feature": "fatalities_roll3",
      "gain_pct": 0.52
    },
    {
      "feature": "rainfall_lag2",
      "gain_pct": 0.52
    },
    {
      "feature": "is_lean_season",
      "gain_pct": 0.5
    },
    {
      "feature": "rainfall_roll3",
      "gain_pct": 0.41
    },
    {
      "feature": "events_roll3",
      "gain_pct": 0.37
    },
    {
      "feature": "battle_share",
      "gain_pct": 0.36
    },
    {
      "feature": "sustained_conflict",
      "gain_pct": 0.33
    },
    {
      "feature": "compound_risk_score",
      "gain_pct": 0.26
    }
  ]
}

```

---
## Task 3 Summary & Handoff to Task 4

### Model Performance (Validation Set)

All metrics printed above. Key takeaways:
- **XGBoost with tuned hyperparameters** is the best model
- **`ipc_lag1` and `ipc_lag2`** dominate feature importance (80%+ cumulative gain)  
  → confirms the model is learning real food security persistence, not noise
- **Optimal threshold** is typically < 0.5 for humanitarian tasks  
  → accepting more false alarms to avoid missing real crises

### Files for Task 4

| File | Use |
|---|---|
| `best_model.pkl` | Load with `joblib.load()` for SHAP analysis |
| `best_model.json` | XGBoost native format for deployment |
| `X_test.csv` + `y_test.csv` | **SEALED** — open in Task 4 only |
| `task3_results.json` | Optimal threshold, best params, val metrics |
| `feature_importance.csv` | Features ranked by gain |

### Next → Task 4: Evaluation, SHAP Explainability & Error Analysis




# Task 4: Final Evaluation & SHAP Analysis

**File:** `crisis_task4_evaluation_FINAL.ipynb`

---
# Conflict-Induced Food Crisis Prediction
## Task 4: Final Evaluation, SHAP Explainability & Africa Risk Map  v1.0

**Input**  : `best_model.pkl`, `X_test.csv`, `y_test.csv`, `task3_results.json`
**Output** : `final_evaluation.png`, `shap_*.png`, `country_performance.png`,
             `test_results.json`, `africa_crisis_map.html`

---
### Notebook Structure

| Step | Section |
|---|---|
| 1  | Setup (install shap, folium) |
| 1b | Restore from Google Drive |
| 2  | Load model + unseal test set |
| 3  | Final test-set evaluation |
| 4  | Confusion matrices + PR curves |
| 5  | Country-level performance |
| 6  | SHAP global beeswarm |
| 7  | SHAP waterfall — high-risk case |
| 8  | SHAP waterfall — blind-spot FN (Sudan/Chad) |
| 9  | Test set error analysis |
| 10 | Africa Crisis Map (Folium interactive HTML) |
| 11 | Save + Drive backup |


---
## Step 1 — Setup

#### 🏁 Output
```text
Installing scikit-learn...
Environment : Google Colab
Data dir    : /content/crisis_outputs
SHAP        : 0.51.0
Folium      : 0.20.0

 Setup complete.

```

---
## Step 1b — Restore from Google Drive

#### 🏁 Output
```text
Mounted at /content/drive
 Drive mounted.
Restored 66 files.

   best_model.pkl                      693.3 KB
   X_test.csv                          1289.3 KB
   y_test.csv                          16.5 KB
   task3_results.json                  2.7 KB
   X_val.csv                           621.3 KB
   y_val.csv                           7.6 KB
   features_engineered.csv             5791.2 KB

```

---
## Step 2 — Load Model & Unseal Test Set

#### 🏁 Output
```text
UNSEALING TEST SET (2024-01 to 2026-01)
Test  : 8,459 rows  crisis=52.7%
Val   : 3,892 rows  crisis=48.8%  (reference)
Model : best_iteration=499
Params: {'max_depth': 4, 'learning_rate': 0.05, 'subsample': 0.9, 'colsample_bytree': 0.7, 'min_child_weight': 5}
F1-threshold=0.497  |  HR-threshold=0.077

 Ready for evaluation.

```

---
## Step 3 — Final Test Set Evaluation (Unsealed)

#### 🏁 Output
```text
============================================================
FINAL TEST SET EVALUATION (2024-01 to 2026-01)
============================================================

--- F1-Optimal Threshold (0.497) ---
              precision    recall  f1-score   support

    Safe (0)     0.9825    0.9988    0.9906      4001
  Crisis (1)     0.9989    0.9841    0.9914      4458

    accuracy                         0.9910      8459
   macro avg     0.9907    0.9914    0.9910      8459
weighted avg     0.9911    0.9910    0.9910      8459

TN=3,996  FP=5  FN=71  TP=4,387
FNR=1.6%  FPR=0.1%

--- Humanitarian Threshold (0.077) ---
              precision    recall  f1-score   support

    Safe (0)     0.9869    0.9030    0.9431      4001
  Crisis (1)     0.9191    0.9892    0.9529      4458

    accuracy                         0.9485      8459
   macro avg     0.9530    0.9461    0.9480      8459
weighted avg     0.9512    0.9485    0.9483      8459

TN=3,613  FP=388  FN=48  TP=4,410
FNR=1.1%

ROC-AUC = 0.9966  (val=0.9882)
PR-AUC  = 0.9976  (val=0.9890)

=== GENERALIZATION (Val 2023 vs Test 2024-2026) ===
Metric              Val       Test    Delta
------------------------------------------
F1               0.9449     0.9914  +0.0465
Recall           0.8994     0.9841  +0.0847
PR-AUC           0.9890     0.9976  +0.0086
ROC-AUC          0.9882     0.9966  +0.0084

```

---
## Step 4 — Confusion Matrices + PR Curves

#### 🏁 Output
```text
<Figure size 1800x500 with 3 Axes> final_evaluation.png saved.

```

---
## Step 5 — Country-Level Performance

#### 🏁 Output
```text
Country Performance (Test 2024-2026):
Country                                  F1   Recall     FNR   PR-AUC
----------------------------------------------------------------------
Sudan                                1.0000   1.0000    0.0%   1.0000
Somalia                              0.9951   0.9902    1.0%   0.9994
Burkina Faso                         0.9897   0.9796    2.0%   0.9951
Cameroon                             0.9831   0.9667    3.3%   0.9933
Chad                                 0.9814   0.9635    3.7%   0.9924
Niger                                0.9785   0.9661    3.4%   0.9946
Ethiopia                             0.9745   0.9502    5.0%   0.9900
Nigeria                              0.9690   0.9481    5.2%   0.9809
Mozambique                           0.9659   0.9340    6.6%   0.9969
Kenya                                0.9648   0.9412    5.9%   0.9717
Mali                                 0.9250   0.8810   11.9%   0.9309
<Figure size 1600x600 with 2 Axes> country_performance.png saved.

```

---
## Step 6 — SHAP Global Beeswarm

SHAP values explain **why** the model makes each prediction.
- Right side of 0 = pushed prediction toward **crisis**
- Red dots = high feature value, Blue dots = low feature value


#### 🏁 Output
```text
Computing SHAP values...
SHAP matrix: (8459, 32)
<Figure size 800x950 with 2 Axes> shap_beeswarm.png saved.
<Figure size 800x950 with 1 Axes> shap_summary.png saved.

```

---
## Step 7 — SHAP Waterfall: Highest-Risk Case

#### 🏁 Output
```text
Highest-risk case:
  Country   : Chad / Lac
  Date      : 2024-09
  Prob      : 1.0000
  True label: CRISIS
<Figure size 800x900 with 3 Axes> shap_waterfall_highrisk.png saved.

```

---
## Step 8 — SHAP Waterfall: Blind-Spot Missed Crisis (FN)

#### 🏁 Output
```text
Missed crisis (FN):
  Country : Chad / Barh El Gazel
  Date    : 2024-02
  IPC     : Phase 3
  Prob    : 0.0102  (below threshold -> predicted SAFE)
  True    : CRISIS <- MISSED!
<Figure size 800x900 with 3 Axes> shap_waterfall_fn.png saved.

```

---
## Step 9 — Test Set Error Analysis

#### 🏁 Output
```text
Test FN: 71  FP: 5

Country                                FN   Total      FNR
-------------------------------------------------------
Mozambique                             14     212     6.6%
Nigeria                                12     231     5.2%
Ethiopia                               10     201     5.0%
Chad                                    8     219     3.7%
Somalia                                 8     816     1.0%
Kenya                                   6     102     5.9%
Mali                                    5      42    11.9%
Niger                                   4     118     3.4%
Cameroon                                2      60     3.3%
Burkina Faso                            2      98     2.0%

IPC phase of missed crises:
  Phase 1: 1 (1.4%)
  Phase 2: 35 (49.3%)
  Phase 3: 35 (49.3%)

Overall FNR: 1.6%  (val=10.1% reference)

```

---
## Step 10 — Africa Food Crisis Early Warning Map

#### 🏁 Output
```text
Map saved: africa_crisis_map.html  (31.5 KB)
<IPython.lib.display.IFrame at 0x7d9ed9d51820>
Country risk levels:
  South Sudan                         Famine       P=0.999
  Sudan                               Famine       P=0.955
  Somalia                             Famine       P=0.905
  Mozambique                          Emergency    P=0.728
  Niger                               Crisis       P=0.523
  Ethiopia                            Crisis       P=0.495
  Chad                                Stressed     P=0.366
  Burkina Faso                        Stressed     P=0.312
  Cameroon                            Stressed     P=0.262
  Mali                                Stressed     P=0.217
  Nigeria                             Minimal      P=0.166
  Kenya                               Minimal      P=0.098

```

#### 🏁 Output
```text
Rows loaded : 12,351

Full country risk table:
Country                                Risk          P(mean)  P(max)  Regions  Crisis%
─────────────────────────────────────────────────────────────────────────────────────
Democratic Republic Of Congo           Famine          1.000   1.000        1   100.0%
South Sudan                            Famine          0.999   1.000       10   100.0%
Somalia                                Famine          0.920   1.000       36    91.5%
Sudan                                  Famine          0.845   1.000       92    83.9%
Mozambique                             Emergency       0.606   1.000       11    58.5%
Ethiopia                               Crisis          0.525   1.000       16    50.5%
Niger                                  Crisis          0.488   1.000        9    47.1%
Chad                                   Stressed        0.335   1.000       24    30.6%
Burkina Faso                           Stressed        0.324   1.000       13    30.6%
Cameroon                               Stressed        0.277   1.000       10    24.3%
Mali                                   Stressed        0.235   1.000        9    18.9%
Nigeria                                Minimal         0.185   1.000       63    16.0%
Kenya                                  Minimal         0.161   1.000       49    14.3%

Map saved: africa_crisis_map_v2.html  (626.8 KB)

Risk summary:
  Democratic Republic Of Congo           Famine       P=1.000  ███████████████████
  South Sudan                            Famine       P=0.999  ███████████████████
  Somalia                                Famine       P=0.920  ██████████████████
  Sudan                                  Famine       P=0.845  ████████████████
  Mozambique                             Emergency    P=0.606  ████████████
  Ethiopia                               Crisis       P=0.525  ██████████
  Niger                                  Crisis       P=0.488  █████████
  Chad                                   Stressed     P=0.335  ██████
  Burkina Faso                           Stressed     P=0.324  ██████
  Cameroon                               Stressed     P=0.277  █████
  Mali                                   Stressed     P=0.235  ████
  Nigeria                                Minimal      P=0.185  ███
  Kenya                                  Minimal      P=0.161  ███
<folium.folium.Map at 0x7d9ed17cb200>
```

---
## Step 11 — Save Final Results & Drive Backup

#### 🏁 Output
```text
=== TASK 4 COMPLETE ===

Test (2024-2026) — F1-Optimal Threshold 0.497:
  f1            : 0.9914
  recall        : 0.9841
  precision     : 0.9989
  pr_auc        : 0.9976
  roc_auc       : 0.9966
  fnr           : 1.59

Files saved:
  final_evaluation.png                              109.5 KB
  shap_beeswarm.png                                 160.9 KB
  shap_summary.png                                  104.2 KB
  shap_waterfall_highrisk.png                       133.2 KB
  shap_waterfall_fn.png                             139.6 KB
  country_performance.png                           133.3 KB
  test_results.json                                   3.0 KB
  africa_crisis_map.html                             31.5 KB

 Drive backup complete -> /content/drive/MyDrive/crisis_outputs_backup

```

---
## Task 4 Summary & Project Handoff

### Deliverables Produced

| File | Purpose |
|---|---|
| `final_evaluation.png` | Confusion matrices + PR curves (test vs val) |
| `shap_beeswarm.png` | Global SHAP importance — all test predictions |
| `shap_summary.png` | SHAP mean absolute values bar chart |
| `shap_waterfall_highrisk.png` | Why the model predicted the highest-risk case |
| `shap_waterfall_fn.png` | Why Sudan/Chad crisis was missed |
| `country_performance.png` | F1 and FNR by country |
| `africa_crisis_map.html` | **Interactive Folium map** — open in any browser |
| `test_results.json` | All final metrics in structured JSON |

### Deployment
- **Streamlit app**: `app.py` — run with `streamlit run app.py`
- **Interactive map**: `africa_crisis_map.html` — open directly in browser
- **Model**: `best_model.pkl` + `best_model.json` ready for API integration




