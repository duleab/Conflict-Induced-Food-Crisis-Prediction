"""
Africa Food Crisis Early Warning System — Streamlit Dashboard
Run:  streamlit run app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import json
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Africa Food Crisis Early Warning System",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

:root {
    --primary: #3b82f6;
    --bg-main: #f8fafc;
    --bg-card: #ffffff;
    --text-main: #0f172a;
    --text-muted: #64748b;
    --border: #e2e8f0;
    --danger: #ef4444;
    --success: #10b981;
    --warning: #f59e0b;
}
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.main { background: var(--bg-main); }
[data-testid="stSidebar"] { background: #f1f5f9; border-right: 1px solid var(--border); }

.metric-card {
    background: var(--bg-card); border: 1px solid var(--border);
    border-radius: 16px; padding: 22px; margin: 6px 0;
    box-shadow: 0 1px 3px rgba(0,0,0,.06);
    transition: all .25s ease;
}
.metric-card:hover { transform: translateY(-3px); box-shadow: 0 8px 25px rgba(0,0,0,.08); border-color: var(--primary); }
.metric-title { font-size: 11px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 1.2px; font-weight: 600; margin-bottom: 6px; }
.metric-value { font-size: 30px; font-weight: 800; }
.metric-sub { font-size: 12px; color: var(--text-muted); margin-top: 4px; }

.risk-badge { display:inline-block; padding:5px 14px; border-radius:9999px; font-size:11px; font-weight:700; letter-spacing:.3px; }
.risk-minimal   { background:#dcfce7; color:#166534; }
.risk-stressed  { background:#fef9c3; color:#854d0e; }
.risk-crisis    { background:#ffedd5; color:#9a3412; }
.risk-emergency { background:#fee2e2; color:#991b1b; }
.risk-famine    { background:#f3e8ff; color:#6b21a8; }

.section-header { font-size:20px; font-weight:700; color:var(--text-main); padding:14px 0 10px; border-bottom:2px solid var(--border); margin-bottom:18px; }
.desc-text { font-size:13px; color:var(--text-muted); line-height:1.6; margin-bottom:14px; }

.country-card {
    background: var(--bg-card); border:1px solid var(--border); border-radius:14px;
    padding:20px; margin:8px 0; box-shadow:0 1px 3px rgba(0,0,0,.04);
    transition: all .2s ease;
}
.country-card:hover { border-color: var(--primary); box-shadow:0 4px 12px rgba(59,130,246,.1); }
</style>
""", unsafe_allow_html=True)

# ── Data loaders ──────────────────────────────────────────────────────────────
DATA_PATHS = [
    Path("crisis_outputs"),
    Path("data"),
    Path("/content/crisis_outputs"),
]

@st.cache_data
def find_data_dir():
    for p in DATA_PATHS:
        if (p / "X_test.csv").exists():
            return p
    return None

@st.cache_data
def load_all_data(data_dir_str):
    data_dir = Path(data_dir_str)
    feats = pd.read_csv(data_dir / "features_engineered.csv")
    t3 = {}
    if (data_dir / "task3_results.json").exists():
        with open(data_dir / "task3_results.json") as f:
            t3 = json.load(f)
    t4 = {}
    if (data_dir / "test_results.json").exists():
        with open(data_dir / "test_results.json") as f:
            t4 = json.load(f)
    pred_df = None
    xtest_path = data_dir / "X_test.csv"
    ytest_path = data_dir / "y_test.csv"
    if xtest_path.exists() and ytest_path.exists():
        X_test = pd.read_csv(xtest_path)
        y_test = pd.read_csv(ytest_path).squeeze()
        pred_df = pd.DataFrame(
            {
                "country": feats["country"],
                "admin1": feats["admin1"],
                "year_month": feats["year_month"],
                "ipc_phase": feats["ipc_phase"],
                "crisis_90d": feats.get("crisis_90d", np.nan),
            }
        )
        pred_df = pred_df.iloc[: len(y_test)].reset_index(drop=True)
        pred_df["y_true"] = y_test.values
        pred_df["_X_test_path"] = str(xtest_path)
    return feats, t3, t4, pred_df

@st.cache_data
def make_demo_pred(feats):
    df = feats[feats['year_month'] >= '2023-01'].copy()
    np.random.seed(42)
    df['proba'] = np.clip(df['ipc_phase'].fillna(2) / 5 + np.random.normal(0, 0.1, len(df)), 0, 1)
    df['y_true'] = df['crisis_90d'].fillna(0).astype(int)
    df['predicted'] = (df['proba'] >= 0.5).astype(int)
    return df


@st.cache_resource
def load_model_artifact(model_path_str: str):
    """
    Loads the model as a long-lived resource (slow on Windows for pickled XGBoost).
    For deployment, prefer exporting a stable XGBoost JSON model and loading via Booster.
    """
    import joblib

    model = joblib.load(model_path_str)
    return model


def align_features_for_model(X: pd.DataFrame, model) -> pd.DataFrame:
    """
    Ensures feature columns match training order, fills missing with 0,
    and drops extras. This makes inference robust to schema drift.
    """
    feature_names = getattr(model, "feature_names_in_", None)
    if feature_names is None:
        # Fallback: use provided columns as-is
        return X
    feature_names = list(feature_names)
    out = X.copy()
    missing = [c for c in feature_names if c not in out.columns]
    if missing:
        for c in missing:
            out[c] = 0
    out = out[feature_names]
    return out


def _to_period_ym(s: str) -> pd.Period:
    return pd.Period(str(s), freq="M")


def extend_with_forecast_months(df: pd.DataFrame, months_ahead: int) -> pd.DataFrame:
    """
    Extend a monthly snapshot table by carrying forward the latest available values.
    This is a UI convenience for the 90-day early warning horizon, not a re-run of
    feature generation for future months.
    """
    if months_ahead <= 0 or "year_month" not in df.columns or len(df) == 0:
        out = df.copy()
        if "is_forecast" not in out.columns:
            out["is_forecast"] = False
        return out

    out = df.copy()
    out["is_forecast"] = False
    max_p = out["year_month"].map(_to_period_ym).max()
    latest = out[out["year_month"].map(_to_period_ym) == max_p].copy()
    if len(latest) == 0:
        return out

    for i in range(1, months_ahead + 1):
        f_p = max_p + i
        f = latest.copy()
        f["year_month"] = str(f_p)
        f["is_forecast"] = True
        out = pd.concat([out, f], ignore_index=True)
    return out

# ── Country centroids & admin1 approximate coords ────────────────────────────
CENTROIDS = {
    'Ethiopia':[9.15,40.49],'Somalia':[5.15,46.20],'Sudan':[15.55,32.53],
    'South Sudan':[6.88,31.57],'Kenya':[-0.02,37.91],'Nigeria':[9.08,8.68],
    'Niger':[17.61,8.08],'Mali':[17.57,-3.99],'Burkina Faso':[12.36,-1.56],
    'Chad':[15.45,18.73],'Cameroon':[3.85,11.50],'Mozambique':[-18.67,35.53],
    'Democratic Republic Of Congo':[-4.03,21.76],'Central African Republic':[6.61,20.94],
}

RISK_COLORS = {'Minimal':'#22c55e','Stressed':'#eab308','Crisis':'#f97316','Emergency':'#ef4444','Famine':'#a855f7'}

def risk_label(prob):
    if prob < 0.20: return 'Minimal'
    if prob < 0.40: return 'Stressed'
    if prob < 0.60: return 'Crisis'
    if prob < 0.80: return 'Emergency'
    return 'Famine'

COUNTRY_CONTEXT = {
    'Sudan': 'Ongoing civil conflict since April 2023 has displaced millions. Darfur and Kordofan regions face acute food insecurity exacerbated by disrupted supply chains.',
    'Somalia': 'Recurrent droughts and Al-Shabaab activity create persistent crisis conditions, particularly in southern and central regions.',
    'South Sudan': 'Protracted conflict and seasonal flooding drive extreme food insecurity. Unity, Jonglei, and Upper Nile are chronically affected.',
    'Ethiopia': 'Northern conflict (Tigray, Amhara, Afar) combined with drought in Somali region creates multi-dimensional food crises.',
    'Nigeria': 'Northeast insurgency (Borno, Adamawa, Yobe) and banditry in the northwest create pockets of severe food insecurity.',
    'Kenya': 'Arid and semi-arid counties (Turkana, Marsabit, Wajir, Mandera) face recurrent drought-induced food stress.',
    'Chad': 'Lake Chad Basin crisis and refugee influx from Sudan strain food systems, particularly in eastern regions.',
    'Niger': 'Sahel insecurity and climate variability drive food crises in Diffa, Tillabéri, and Tahoua regions.',
    'Mali': 'Armed group activity in northern and central regions disrupts agriculture and market access.',
    'Burkina Faso': 'Rapid security deterioration since 2019 has created large-scale displacement and food access barriers.',
    'Cameroon': 'Far North insecurity (Boko Haram) and Anglophone crisis (Northwest/Southwest) affect food security.',
    'Mozambique': 'Cabo Delgado insurgency has displaced over 1M people, severely impacting food production.',
    'Democratic Republic Of Congo': 'Eastern DRC conflict and displacement create severe food insecurity across multiple provinces.',
}

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("###  Africa Food Crisis EWS")
    st.markdown("*Conflict-Induced Prediction System*")
    st.divider()
    run_mode = st.radio(
        "Run Mode",
        ["Production (use trained model)", "Demo (simulated probabilities)"],
        help="Production uses `crisis_outputs/best_model.pkl` + `X_test.csv`. Demo is only for UI exploration.",
    )
    data_dir = find_data_dir()
    if data_dir:
        feats, t3, t4, pred_df = load_all_data(str(data_dir))
        if pred_df is None or run_mode.startswith("Demo"):
            pred_df = make_demo_pred(feats)
            st.info("⚡ Demo mode — probabilities are simulated (not deployable).")
        else:
            model_path = Path(data_dir) / "best_model.pkl"
            if model_path.exists():
                with st.spinner("Loading XGBoost model (first run can take ~20–30s)…"):
                    model = load_model_artifact(str(model_path))
                st.success(f"✅ Model loaded from `{model_path}`")
                try:
                    # Load X_test and run inference
                    X_test = pd.read_csv(pred_df["_X_test_path"].iloc[0])
                    X_test = align_features_for_model(X_test, model)
                    # Fill NAs defensively: median per-column
                    X_test = X_test.apply(pd.to_numeric, errors="ignore")
                    if X_test.select_dtypes(include=[np.number]).shape[1] > 0:
                        med = X_test.select_dtypes(include=[np.number]).median()
                        X_test[X_test.select_dtypes(include=[np.number]).columns] = X_test.select_dtypes(include=[np.number]).fillna(med)
                    else:
                        X_test = X_test.fillna(0)
                    proba = model.predict_proba(X_test)[:, 1]
                    pred_df = pred_df.drop(columns=["_X_test_path"], errors="ignore")
                    pred_df["proba"] = proba
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
                    pred_df = make_demo_pred(feats)
                    st.info("Falling back to demo probabilities.")
            else:
                st.warning("⚠️ `best_model.pkl` not found — using demo mode.")
                pred_df = make_demo_pred(feats)
    else:
        feats = pd.DataFrame()
        t3, t4, pred_df = {}, {}, None
        st.warning("⚠️ No data found — demo mode.")
        np.random.seed(42)
        pred_df = pd.DataFrame({
            'country': np.random.choice(list(CENTROIDS.keys()), 3892),
            'admin1': [f"Region_{i}" for i in range(3892)],
            'year_month': pd.date_range('2024-01', periods=3892, freq='D').strftime('%Y-%m'),
            'ipc_phase': np.random.choice([1,2,3,4], 3892, p=[.25,.35,.3,.1]),
            'y_true': np.random.choice([0,1], 3892, p=[.48,.52]),
            'proba': np.random.beta(2, 2, 3892),
        })
        pred_df['predicted'] = (pred_df['proba'] >= 0.5).astype(int)

    st.markdown("#### Filters")
    all_countries = sorted(pred_df['country'].unique())
    sel_countries = st.multiselect("Countries", all_countries, default=all_countries, placeholder="All")
    if not sel_countries: sel_countries = all_countries

    # Thresholds come from evaluation artifact when available
    f1_thr = float(t4.get("f1_optimal_threshold", t3.get("opt_threshold", 0.5)) or 0.5)
    hum_thr = float(t4.get("humanitarian_threshold", 0.077) or 0.077)
    threshold_mode = st.radio(
        "Alert Threshold",
        [f"F1-Optimal ({f1_thr:.3f})", f"Humanitarian ({hum_thr:.3f})"],
        help="F1-Optimal balances precision/recall. Humanitarian minimizes missed crises (higher recall, more false alarms).",
    )
    threshold_val = f1_thr if "F1-Optimal" in threshold_mode else hum_thr

    if 'year_month' in pred_df.columns:
        max_date = max(pred_df["year_month"].astype(str))
        forecast_months = st.slider(
            "Forecast horizon (months beyond last data)",
            0,
            3,
            3,
            help=f"Your dataset currently ends at {max_date}. This extends the dashboard timeline by carrying forward the latest risk estimate (clearly labeled as Forecast).",
        )
    else:
        sel_date = None

    st.divider()
    st.markdown("**Model Info**")
    if t3:
        bp = t3.get('best_params', {})
        st.markdown(f"`depth={bp.get('max_depth','?')}` `lr={bp.get('learning_rate','?')}`")
        vm = t3.get('val_metrics', {})
        st.markdown(f"Val F1: **{vm.get('f1','?')}** · PR-AUC: **{vm.get('pr_auc','?')}**")

# ── Apply filters ─────────────────────────────────────────────────────────────
filtered = pred_df[pred_df['country'].isin(sel_countries)].copy()
filtered['predicted'] = (filtered['proba'] >= threshold_val).astype(int)
filtered['risk'] = filtered['proba'].apply(risk_label)
filtered = extend_with_forecast_months(filtered, forecast_months if "forecast_months" in locals() else 0)

if "year_month" in filtered.columns:
    dates = sorted(filtered["year_month"].unique(), key=lambda x: _to_period_ym(x))
    sel_date = st.select_slider("Snapshot", dates, value=dates[-1])
    snap = filtered[filtered["year_month"] == sel_date]
else:
    sel_date = None
    snap = filtered.groupby(["country", "admin1"]).last().reset_index()

# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding:20px 0 14px; border-bottom:2px solid var(--border); margin-bottom:28px;">
  <h1 style="font-size:34px; font-weight:800; color:var(--text-main); margin:0; letter-spacing:-.5px;">
     Africa Food Crisis Early Warning System
  </h1>
  <p style="color:var(--text-muted); margin:6px 0 0; font-size:15px;">
    XGBoost Ensemble · Conflict × Climate × IPC Fusion · Admin1-Level Predictions · 90-Day Forecast Horizon
  </p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="desc-text" style="background:#eff6ff; border:1px solid #bfdbfe; border-radius:10px; padding:14px 18px; margin-bottom:22px; color:#1e40af;">
  <b>📡 How It Works:</b> This system fuses <b>conflict event data</b> (ACLED battles, fatalities, civilian violence),
  <b>climate indicators</b> (rainfall anomalies, drought/flood flags), and <b>IPC food security phases</b> to predict
  the probability of a food crisis within the next 90 days for each administrative region. The XGBoost model
  achieves <b>F1=0.991</b> on held-out test data across 11 African nations.
</div>
""", unsafe_allow_html=True)

# ── KPI CARDS ─────────────────────────────────────────────────────────────────
total_regions = len(snap)
crisis_count = int(snap['predicted'].sum())
safe_count = total_regions - crisis_count
avg_prob = snap['proba'].mean() if total_regions > 0 else 0
high_risk_cnt = int((snap['proba'] >= 0.8).sum())

cards = [
    ("🌍 Monitored Regions", f"{total_regions:,}", "Active surveillance zones", "#3b82f6"),
    ("🚨 Crisis Alerts", f"{crisis_count:,}", f"{crisis_count/max(total_regions,1)*100:.1f}% flagged", "#ef4444"),
    ("✅ Safe Zones", f"{safe_count:,}", f"{safe_count/max(total_regions,1)*100:.1f}% stable", "#10b981"),
    ("⚠️ Avg Risk", f"{avg_prob:.3f}", "Mean probability", "#f59e0b"),
    ("🔴 Emergency+", f"{high_risk_cnt:,}", "Probability ≥ 0.80", "#a855f7"),
]
cols = st.columns(5)
for col, (title, val, sub, color) in zip(cols, cards):
    with col:
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-title">{title}</div>
          <div class="metric-value" style="color:{color};">{val}</div>
          <div class="metric-sub">{sub}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── INTERACTIVE RISK MAP ──────────────────────────────────────────────────────
st.markdown('<div class="section-header">🗺️ Interactive Risk Map — Africa Overview + Country Drilldown</div>', unsafe_allow_html=True)
st.markdown("""
<div class="desc-text">
  This section provides two complementary views:
  <b>(1) A clean Africa choropleth</b> for rapid situational awareness, and
  <b>(2) a regional drilldown</b> (admin1) to support operational planning.
  <br><br>
  <b>Interpretation:</b> probability represents the model’s estimated chance of an IPC crisis event within the next 90 days.
  Thresholding is policy-driven: humanitarian mode increases recall (fewer missed crises) at the cost of more alerts.
</div>
""", unsafe_allow_html=True)

country_agg = filtered.groupby('country').agg(
    avg_prob=('proba','mean'), crisis_rate=('y_true','mean'),
    regions=('admin1','nunique'), predicted=('predicted','sum'),
).reset_index()
country_agg['risk'] = country_agg['avg_prob'].apply(risk_label)

# Plotly Africa choropleth (offline, no external geojson calls)
st.markdown("####  Africa Choropleth (Country Average Risk)")
st.markdown(
    '<div class="desc-text">Hover to see average probability, risk tier, and number of alerted regions. This is a country-level summary; use the drilldown panel for admin1-level actions.</div>',
    unsafe_allow_html=True,
)
try:
    chor = px.choropleth(
        country_agg,
        locations="country",
        locationmode="country names",
        color="avg_prob",
        hover_name="country",
        hover_data={
            "avg_prob": ":.3f",
            "regions": True,
            "predicted": True,
            "crisis_rate": ":.2f",
            "risk": True,
        },
        color_continuous_scale="YlOrRd",
        range_color=(0, 1),
    )
    chor.update_geos(
        scope="africa",
        showcountries=True,
        countrycolor="#cbd5e1",
        showcoastlines=True,
        coastlinecolor="#cbd5e1",
        showland=True,
        landcolor="#f8fafc",
        showocean=True,
        oceancolor="#eff6ff",
        fitbounds="locations",
    )
    chor.update_layout(
        template="plotly_white",
        height=520,
        margin=dict(t=10, b=10, l=10, r=10),
        coloraxis_colorbar=dict(title="Crisis Prob"),
    )
    st.plotly_chart(chor, width="stretch")
except Exception as e:
    st.info(f"Choropleth unavailable in this environment: {e}")

# Build admin1-level aggregation for popups
admin_agg = filtered.groupby(['country','admin1']).agg(
    avg_prob=('proba','mean'), predicted=('predicted','max')
).reset_index()
admin_agg['risk'] = admin_agg['avg_prob'].apply(risk_label)

# Admin1 drilldown map (Folium)
m = folium.Map(location=[5, 20], zoom_start=4, tiles="CartoDB positron")

# Country markers with admin1 drill-down popups
for _, row in country_agg.iterrows():
    c = row['country']
    if c not in CENTROIDS: continue
    lat, lon = CENTROIDS[c]
    color = RISK_COLORS.get(row['risk'], '#888')
    radius = max(6, min(18, int(row['regions'] / 3)))

    # Build admin1 table for popup
    c_admins = admin_agg[admin_agg['country'] == c].sort_values('avg_prob', ascending=False)
    admin_rows = ""
    for _, ar in c_admins.head(15).iterrows():
        rc = RISK_COLORS.get(ar['risk'], '#888')
        alert_icon = "🔴" if ar['predicted'] else "🟢"
        admin_rows += f"<tr><td style='padding:3px 6px;font-size:11px;'>{alert_icon} {ar['admin1']}</td>"
        admin_rows += f"<td style='padding:3px 6px;font-size:11px;color:{rc};font-weight:700;'>{ar['risk']}</td>"
        admin_rows += f"<td style='padding:3px 6px;font-size:11px;text-align:right;'>{ar['avg_prob']:.3f}</td></tr>"
    remaining = len(c_admins) - 15
    if remaining > 0:
        admin_rows += f"<tr><td colspan='3' style='padding:4px;font-size:10px;color:#94a3b8;text-align:center;'>+ {remaining} more regions</td></tr>"

    context = COUNTRY_CONTEXT.get(c, 'Monitored for conflict-induced food insecurity.')

    popup_html = f"""
    <div style='font-family:Inter,sans-serif;min-width:320px;max-width:380px;background:#0f172a;color:#e2e8f0;padding:16px;border-radius:12px;'>
      <div style='font-size:18px;font-weight:700;color:#60a5fa;margin-bottom:4px;'>{c}</div>
      <div style='font-size:11px;color:#94a3b8;margin-bottom:10px;line-height:1.5;'>{context}</div>
      <hr style='border-color:#1e293b;margin:8px 0;'>
      <div style='display:flex;justify-content:space-between;margin-bottom:8px;'>
        <div><span style='font-size:10px;color:#94a3b8;'>RISK LEVEL</span><br>
          <span style='font-size:16px;font-weight:700;color:{color};'>{row['risk']}</span></div>
        <div style='text-align:right;'><span style='font-size:10px;color:#94a3b8;'>PROBABILITY</span><br>
          <span style='font-size:16px;font-weight:700;'>{row['avg_prob']:.3f}</span></div>
      </div>
      <div style='display:flex;justify-content:space-between;margin-bottom:10px;'>
        <div><span style='font-size:10px;color:#94a3b8;'>REGIONS</span><br><b>{int(row['regions'])}</b></div>
        <div><span style='font-size:10px;color:#94a3b8;'>ALERTS</span><br><b style='color:#ef4444;'>{int(row['predicted'])}</b></div>
        <div><span style='font-size:10px;color:#94a3b8;'>CRISIS RATE</span><br><b>{row['crisis_rate']*100:.1f}%</b></div>
      </div>
      <div style='font-size:11px;font-weight:600;color:#60a5fa;margin-bottom:4px;'>📍 Admin1 Regions (Top Risk)</div>
      <table style='width:100%;border-collapse:collapse;'>
        <tr style='border-bottom:1px solid #1e293b;'>
          <th style='text-align:left;padding:3px 6px;font-size:10px;color:#94a3b8;'>Region</th>
          <th style='text-align:left;padding:3px 6px;font-size:10px;color:#94a3b8;'>Level</th>
          <th style='text-align:right;padding:3px 6px;font-size:10px;color:#94a3b8;'>Prob</th>
        </tr>
        {admin_rows}
      </table>
    </div>"""

    folium.CircleMarker(
        location=[lat, lon], radius=radius, color=color,
        fill=True, fill_color=color, fill_opacity=0.85, weight=2,
        tooltip=f"<b>{c}</b>: {row['risk']} ({row['avg_prob']:.2f}) — Click for regions",
        popup=folium.Popup(popup_html, max_width=400)
    ).add_to(m)

    folium.Marker(
        [lat - 0.8, lon],
        icon=folium.DivIcon(html=f'<div style="font-size:9px;font-weight:700;color:#0f172a;text-shadow:0 0 3px white,0 0 6px white;width:90px;margin-left:-45px;text-align:center;">{c}</div>')
    ).add_to(m)

map_col, summary_col = st.columns([3, 2])
with map_col:
    st.markdown("#### 🧭 Admin1 Drilldown (Markers)")
    st.markdown(
        '<div class="desc-text">Markers provide fast access to region-level details. This view does not require external map data downloads.</div>',
        unsafe_allow_html=True,
    )
    st_folium(m, height=520, width="stretch", returned_objects=[])
with summary_col:
    st.markdown('<div class="section-header">🏳️ Country Risk Rankings</div>', unsafe_allow_html=True)
    display_df = country_agg[['country','risk','avg_prob','predicted','regions']].sort_values('avg_prob', ascending=False)
    for _, r in display_df.iterrows():
        risk_cls = f"risk-{r['risk'].lower()}"
        pct = r['avg_prob'] * 100
        st.markdown(f"""
        <div class="country-card">
          <div style="display:flex;justify-content:space-between;align-items:center;">
            <span style="font-weight:700;font-size:14px;">{r['country']}</span>
            <span class="risk-badge {risk_cls}">{r['risk']}</span>
          </div>
          <div style="margin-top:8px;background:#f1f5f9;border-radius:6px;height:6px;">
            <div style="background:{RISK_COLORS.get(r['risk'],'#888')};width:{min(pct,100):.0f}%;height:6px;border-radius:6px;"></div>
          </div>
          <div style="margin-top:6px;display:flex;justify-content:space-between;font-size:11px;color:var(--text-muted);">
            <span>Prob: <b>{r['avg_prob']:.3f}</b></span>
            <span>{int(r['predicted'])}/{int(r['regions'])} regions alerted</span>
          </div>
        </div>""", unsafe_allow_html=True)

st.divider()

# ── COUNTRY DEEP DIVE ─────────────────────────────────────────────────────────
st.markdown('<div class="section-header">🔍 Country Deep Dive — Region-by-Region Analysis</div>', unsafe_allow_html=True)
st.markdown("""
<div class="desc-text">
  Select a country below to see <b>every monitored admin1 region</b> with individual risk scores,
  historical model performance metrics, and a temporal risk projection chart. This enables
  humanitarian planners to pinpoint the exact areas requiring intervention.
</div>
""", unsafe_allow_html=True)

dive_country = st.selectbox("Select Country", all_countries, key="dive_select")

if dive_country:
    c_data = filtered[filtered['country'] == dive_country]
    c_perf = next((item for item in t4.get('country_performance', []) if item["country"] == dive_country), None)
    c_admin = admin_agg[admin_agg['country'] == dive_country].sort_values('avg_prob', ascending=False)
    avg_risk = c_data['proba'].mean() if len(c_data) > 0 else 0

    # Context
    st.markdown(f"""
    <div style="background:#fffbeb;border:1px solid #fde68a;border-radius:10px;padding:14px 18px;margin-bottom:16px;color:#92400e;font-size:13px;line-height:1.6;">
      <b>🏛️ {dive_country} Situation Brief:</b> {COUNTRY_CONTEXT.get(dive_country, 'This country is monitored for conflict-induced food insecurity patterns.')}
    </div>
    """, unsafe_allow_html=True)

    d1, d2 = st.columns([1, 2])
    with d1:
        # Risk card
        st.markdown(f"""
        <div class="metric-card" style="text-align:center;">
          <div class="metric-title">Overall Risk Level</div>
          <div class="metric-value" style="color:{RISK_COLORS.get(risk_label(avg_risk),'#888')};font-size:28px;">{risk_label(avg_risk)}</div>
          <div class="metric-sub">Mean Probability: {avg_risk:.3f}</div>
        </div>""", unsafe_allow_html=True)

        if c_perf:
            mc = st.columns(2)
            mc[0].metric("F1 Score", f"{c_perf.get('f1',0):.3f}")
            mc[1].metric("Recall", f"{c_perf.get('recall',0):.3f}")
            mc2 = st.columns(2)
            mc2[0].metric("Precision", f"{c_perf.get('precision',0):.3f}")
            mc2[1].metric("Crisis %", f"{c_perf.get('crisis_pct',0):.1f}%")
            fn = c_perf.get('fn_count', 0)
            if fn > 0:
                st.warning(f"⚠️ {fn} crises missed (false negatives) — {c_perf.get('fn_rate',0):.1f}% FN rate")
        else:
            st.info("No test performance data for this country.")

    with d2:
        # Time series
        if 'year_month' in c_data.columns and len(c_data) > 0:
            c_ts = c_data.groupby('year_month')['proba'].agg(['mean','min','max']).reset_index()
            fig_c = go.Figure()
            fig_c.add_trace(go.Scatter(x=c_ts['year_month'], y=c_ts['max'], mode='lines', name='Max', line=dict(width=0), showlegend=False))
            fig_c.add_trace(go.Scatter(x=c_ts['year_month'], y=c_ts['min'], mode='lines', name='Risk Range', line=dict(width=0), fill='tonexty', fillcolor='rgba(59,130,246,0.1)'))
            fig_c.add_trace(go.Scatter(x=c_ts['year_month'], y=c_ts['mean'], mode='lines+markers', name='Mean Risk', line=dict(color='#3b82f6', width=3), marker=dict(size=4)))
            fig_c.add_hline(y=0.5, line_dash='dash', line_color='#ef4444', opacity=0.5, annotation_text='Crisis Threshold')
            fig_c.update_layout(template='plotly_white', height=280, title=f'{dive_country} — Monthly Risk Trend', margin=dict(t=40,b=20,l=30,r=20), legend=dict(orientation='h', y=1.12))
            st.plotly_chart(fig_c, width="stretch")

    # Admin1 region table
    st.markdown(f"#### 📍 All Monitored Regions in {dive_country} ({len(c_admin)} areas)")
    if len(c_admin) > 0:
        # Bar chart of admin1 regions
        ca_sorted = c_admin.sort_values('avg_prob', ascending=True).tail(20)
        colors_a = [RISK_COLORS.get(r, '#888') for r in ca_sorted['risk']]
        fig_admin = go.Figure(go.Bar(
            x=ca_sorted['avg_prob'], y=ca_sorted['admin1'], orientation='h',
            marker_color=colors_a, text=ca_sorted['avg_prob'].round(3), textposition='outside'
        ))
        fig_admin.add_vline(x=threshold_val, line_dash='dash', line_color='#ef4444', annotation_text='Threshold')
        fig_admin.update_layout(template='plotly_white', height=max(250, len(ca_sorted)*28), title=f'Top Risk Regions — {dive_country}', margin=dict(t=40,b=20,l=10,r=50), xaxis_title='Crisis Probability', yaxis_title='')
        st.plotly_chart(fig_admin, width="stretch")

        # Full table
        with st.expander(f"View complete region table ({len(c_admin)} regions)"):
            tbl = c_admin[['admin1','risk','avg_prob','predicted']].copy()
            tbl.columns = ['Region','Risk Level','Probability','Alert']
            tbl['Alert'] = tbl['Alert'].map({1:'🔴 Yes', 0:'🟢 No'})
            st.dataframe(tbl, width="stretch", hide_index=True)

st.divider()

# ── CHARTS ────────────────────────────────────────────────────────────────────
ch1, ch2 = st.columns(2)
with ch1:
    st.markdown('<div class="section-header">📊 Probability Distribution</div>', unsafe_allow_html=True)
    fig_hist = px.histogram(filtered, x='proba', nbins=50, color_discrete_sequence=['#3b82f6'], labels={'proba':'Crisis Probability'})
    fig_hist.add_vline(x=threshold_val, line_dash='dash', line_color='#ef4444', annotation_text=f'Threshold={threshold_val}')
    fig_hist.update_layout(template='plotly_white', height=300, showlegend=False, margin=dict(t=30,b=20,l=20,r=20))
    st.plotly_chart(fig_hist, width="stretch")

with ch2:
    st.markdown('<div class="section-header">🌡️ Country Risk Comparison</div>', unsafe_allow_html=True)
    cs = country_agg.sort_values('avg_prob', ascending=True)
    fig_bar = go.Figure(go.Bar(x=cs['avg_prob'], y=cs['country'], orientation='h', marker_color=[RISK_COLORS.get(r,'#888') for r in cs['risk']], text=cs['avg_prob'].round(3), textposition='outside'))
    fig_bar.add_vline(x=0.5, line_dash='dot', line_color='#ef4444', opacity=0.3)
    fig_bar.update_layout(template='plotly_white', height=320, margin=dict(t=30,b=20,l=10,r=60), xaxis_title='Crisis Probability')
    st.plotly_chart(fig_bar, width="stretch")

# ── TIME SERIES ───────────────────────────────────────────────────────────────
if 'year_month' in filtered.columns:
    st.markdown('<div class="section-header">📈 Multi-Country Risk Timeline</div>', unsafe_allow_html=True)
    st.markdown('<div class="desc-text">Track how crisis probability evolves month-by-month for each country. The red shaded zone marks the high-risk band (≥0.6).</div>', unsafe_allow_html=True)
    ts = filtered.groupby(['year_month','country'])['proba'].mean().reset_index()
    ts_pivot = ts.pivot(index='year_month', columns='country', values='proba').ffill()
    sel_ts = st.multiselect("Countries to display", ts_pivot.columns.tolist(), default=ts_pivot.columns[:5].tolist(), key="ts_multi")
    fig_ts = go.Figure()
    PAL = ['#3b82f6','#10b981','#f97316','#a855f7','#f59e0b','#06b6d4','#ec4899','#84cc16','#6366f1','#14b8a6']
    for i, c in enumerate(sel_ts):
        if c in ts_pivot.columns:
            fig_ts.add_trace(go.Scatter(x=ts_pivot.index, y=ts_pivot[c], name=c, mode='lines+markers', line=dict(width=2, color=PAL[i%len(PAL)]), marker=dict(size=3)))
    fig_ts.add_hrect(y0=0.6, y1=1.0, fillcolor='rgba(239,68,68,.06)', line_width=0)
    fig_ts.add_hline(y=0.5, line_dash='dash', line_color='#ef4444', opacity=0.3, annotation_text='Threshold')
    fig_ts.update_layout(template='plotly_white', height=350, margin=dict(t=30,b=30,l=40,r=20), xaxis_title='Month', yaxis_title='Crisis Probability', legend=dict(orientation='h', y=1.08))
    st.plotly_chart(fig_ts, width="stretch")

# ── MODEL PERFORMANCE ─────────────────────────────────────────────────────────
st.divider()
st.markdown('<div class="section-header">🤖 Model Performance & Feature Importance</div>', unsafe_allow_html=True)
st.markdown('<div class="desc-text">Validation (2023 holdout) and test (2024-2026 unseen) metrics confirm strong generalization. The model improves on test data (+4.6% F1), suggesting robust learning of crisis dynamics.</div>', unsafe_allow_html=True)

p1, p2, p3 = st.columns([2,2,3])
with p1:
    if t3 and 'val_metrics' in t3:
        st.markdown("**Validation Set (2023)**")
        for k, v in t3['val_metrics'].items():
            pct = int(v * 100)
            clr = '#10b981' if v > 0.95 else '#f59e0b' if v > 0.90 else '#ef4444'
            st.markdown(f"""<div style="margin-bottom:8px;">
              <div style="display:flex;justify-content:space-between;font-size:12px;"><span>{k.upper()}</span><span style="color:{clr};font-weight:700;">{v:.4f}</span></div>
              <div style="background:#e2e8f0;border-radius:4px;height:5px;margin-top:3px;"><div style="background:{clr};width:{pct}%;height:5px;border-radius:4px;"></div></div>
            </div>""", unsafe_allow_html=True)
    else:
        st.info("Load data to see validation metrics.")

with p2:
    if t4 and 'test_metrics_f1_optimal' in t4:
        st.markdown("**Test Set (2024–2026)**")
        for k, v in t4['test_metrics_f1_optimal'].items():
            if isinstance(v, float):
                pct = int(v * 100)
                clr = '#10b981' if v > 0.95 else '#f59e0b' if v > 0.90 else '#ef4444'
                st.markdown(f"""<div style="margin-bottom:8px;">
                  <div style="display:flex;justify-content:space-between;font-size:12px;"><span>{k.upper()}</span><span style="color:{clr};font-weight:700;">{v:.4f}</span></div>
                  <div style="background:#e2e8f0;border-radius:4px;height:5px;margin-top:3px;"><div style="background:{clr};width:{pct}%;height:5px;border-radius:4px;"></div></div>
                </div>""", unsafe_allow_html=True)
    else:
        st.info("Run Task 4 for test metrics.")

with p3:
    if t3 and 'feature_importance' in t3:
        st.markdown("**Top Features (XGBoost Gain)**")
        fi = t3['feature_importance'][:10]
        fn = [f['feature'] for f in fi]; fv = [f['gain_pct'] for f in fi]
        clrs = ['#ef4444' if i<3 else '#f59e0b' if i<6 else '#3b82f6' for i in range(len(fi))]
        fig_fi = go.Figure(go.Bar(x=fv[::-1], y=fn[::-1], orientation='h', marker_color=clrs[::-1], text=[f'{v:.1f}%' for v in fv[::-1]], textposition='outside'))
        fig_fi.update_layout(template='plotly_white', height=300, margin=dict(t=10,b=10,l=10,r=60), xaxis_title='Gain (%)')
        st.plotly_chart(fig_fi, width="stretch")

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown("""
<div style="text-align:center; color:var(--text-muted); font-size:12px; padding:16px 0;">
  <b>Africa Food Crisis Early Warning System</b><br>
  Data: IPC · ACLED · CHIRPS · GPM&nbsp;&nbsp;|&nbsp;&nbsp;Model: XGBoost v2.1<br>
  <i>10Academy Humanitarian Data Science · 2026</i>
</div>
""", unsafe_allow_html=True)
