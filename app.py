"""
Africa Food Crisis Early Warning System — Streamlit Dashboard
Run:  streamlit run app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import json
import folium
from streamlit_folium import st_folium
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Africa Food Crisis Early Warning System",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.main { background: #fdfdfd; color: #1a1f26; }
[data-testid="stSidebar"] { background: #f1f3f5; border-right: 1px solid #dee2e6; }

.metric-card {
    background: #ffffff;
    border: 1px solid #e9ecef;
    border-radius: 12px;
    padding: 18px 22px;
    margin: 6px 0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    transition: transform 0.2s, box-shadow 0.2s;
}
.metric-card:hover { transform: translateY(-2px); box-shadow: 0 4px 12px rgba(0,0,0,0.1); border-color: #58a6ff; }
.metric-title { font-size: 11px; color: #6c757d; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 6px; }
.metric-value { font-size: 28px; font-weight: 700; color: #1a1f26; }
.metric-sub   { font-size: 12px; color: #adb5bd; margin-top: 4px; }

.risk-badge {
    display: inline-block; padding: 4px 12px; border-radius: 20px;
    font-size: 11px; font-weight: 600; letter-spacing: 0.5px;
}
.risk-minimal   { background:#e8f5e9; color:#2e7d32; }
.risk-stressed  { background:#fff8e1; color:#f57f17; }
.risk-crisis    { background:#fff3e0; color:#ef6c00; }
.risk-emergency { background:#ffebee; color:#c62828; }
.risk-famine    { background:#f3e5f5; color:#7b1fa2; }

.section-header {
    font-size: 18px; font-weight: 600; color: #1a1f26;
    padding: 12px 0 8px; border-bottom: 1px solid #dee2e6; margin-bottom: 16px;
}
.alert-box {
    background: #fff5f5; border: 1px solid #feb2b2;
    border-radius: 8px; padding: 12px 16px; margin: 8px 0; color: #c53030;
}
.info-box {
    background: #ebf8ff; border: 1px solid #90cdf4;
    border-radius: 8px; padding: 12px 16px; margin: 8px 0; color: #2b6cb0;
}
</style>
""", unsafe_allow_html=True)

# ── Data loaders ──────────────────────────────────────────────────────────────
DATA_PATHS = [
    Path("data"),
    Path("/content/crisis_outputs"),
    Path(r"D:\Project\10academy\Conflict-Induced Food Crisis Prediction\data"),
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

    # X_test predictions if available
    pred_df = None
    xtest_path = data_dir / "X_test.csv"
    ytest_path = data_dir / "y_test.csv"
    if xtest_path.exists() and ytest_path.exists():
        X_test = pd.read_csv(xtest_path)
        y_test = pd.read_csv(ytest_path).squeeze()
        # Try to load model
        try:
            import joblib
            model = joblib.load(data_dir / "best_model.pkl")
            val_med = pd.read_csv(data_dir / "X_val.csv").median()
            X_test = X_test.fillna(val_med)
            proba  = model.predict_proba(X_test)[:, 1]
            # Merge with meta
            meta = feats[feats['year_month'] >= '2024-01'][
                ['country','admin1','year_month','ipc_phase','crisis_90d']
            ].reset_index(drop=True).iloc[:len(y_test)]
            pred_df = meta.copy()
            pred_df['y_true']    = y_test.values
            pred_df['proba']     = proba
            pred_df['predicted'] = (proba >= t3.get('opt_threshold', 0.5)).astype(int)
        except Exception:
            pass

    return feats, t3, t4, pred_df

@st.cache_data
def make_demo_pred(feats):
    """Synthetic predictions when model not available."""
    df = feats[feats['year_month'] >= '2023-01'].copy()
    np.random.seed(42)
    df['proba'] = np.clip(
        df['ipc_phase'].fillna(2) / 5 + np.random.normal(0, 0.1, len(df)), 0, 1
    )
    df['y_true']    = df['crisis_90d'].fillna(0).astype(int)
    df['predicted'] = (df['proba'] >= 0.5).astype(int)
    return df

# ── Country centroids ─────────────────────────────────────────────────────────
CENTROIDS = {
    'Ethiopia'                    : [9.15, 40.49],
    'Somalia'                     : [5.15, 46.20],
    'Sudan'                       : [15.55, 32.53],
    'South Sudan'                 : [6.88, 31.57],
    'Kenya'                       : [-0.02, 37.91],
    'Nigeria'                     : [9.08, 8.68],
    'Niger'                       : [17.61, 8.08],
    'Mali'                        : [17.57, -3.99],
    'Burkina Faso'                : [12.36, -1.56],
    'Chad'                        : [15.45, 18.73],
    'Cameroon'                    : [3.85, 11.50],
    'Mozambique'                  : [-18.67, 35.53],
    'Democratic Republic of Congo': [-4.03, 21.76],
    'Central African Republic'    : [6.61, 20.94],
}

RISK_COLORS = {
    'Minimal'  : '#27ae60',
    'Stressed' : '#f39c12',
    'Crisis'   : '#e67e22',
    'Emergency': '#e74c3c',
    'Famine'   : '#9b59b6',
}

def risk_label(prob):
    if prob < 0.20: return 'Minimal'
    if prob < 0.40: return 'Stressed'
    if prob < 0.60: return 'Crisis'
    if prob < 0.80: return 'Emergency'
    return 'Famine'

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🌍 Africa Food Crisis EWS")
    st.markdown("*Conflict-Induced Food Crisis Prediction*")
    st.divider()

    # Load data
    data_dir = find_data_dir()
    if data_dir:
        feats, t3, t4, pred_df = load_all_data(str(data_dir))
        if pred_df is None:
            pred_df = make_demo_pred(feats)
            st.info("⚡ Demo mode — model predictions simulated")
        else:
            st.success(f"✅ Model loaded\n`{data_dir}`")
    else:
        feats = pd.DataFrame()
        t3, t4, pred_df = {}, {}, None
        st.warning("⚠️ No data found. Running in demo mode.")
        from sklearn.datasets import make_classification
        np.random.seed(42)
        pred_df = pd.DataFrame({
            'country'   : np.random.choice(list(CENTROIDS.keys()), 3892),
            'admin1'    : [f"Region_{i}" for i in range(3892)],
            'year_month': pd.date_range('2024-01', periods=3892, freq='D').strftime('%Y-%m'),
            'ipc_phase' : np.random.choice([1,2,3,4], 3892, p=[0.25,0.35,0.3,0.1]),
            'y_true'    : np.random.choice([0,1], 3892, p=[0.48, 0.52]),
            'proba'     : np.random.beta(2, 2, 3892),
        })
        pred_df['predicted'] = (pred_df['proba'] >= 0.5).astype(int)

    st.markdown("#### Filters")
    all_countries = sorted(pred_df['country'].unique())
    sel_countries = st.multiselect("Countries", all_countries, default=all_countries,
                                   placeholder="All countries")
    if not sel_countries: sel_countries = all_countries

    threshold_mode = st.radio(
        "Alert Threshold",
        ["F1-Optimal (0.497)", "Humanitarian (0.077)"],
        help="F1-Optimal balances precision and recall.\nHumanitarian catches more crises at cost of more false alarms."
    )
    threshold_val = 0.497 if "F1" in threshold_mode else 0.077

    if 'year_month' in pred_df.columns:
        dates = sorted(pred_df['year_month'].unique())
        sel_date = st.select_slider("Snapshot Date", dates, value=dates[-1])
    else:
        sel_date = None

    st.divider()
    st.markdown("**Model Info**")
    if t3:
        bp = t3.get('best_params', {})
        st.markdown(f"`max_depth={bp.get('max_depth','?')}` `lr={bp.get('learning_rate','?')}`")
        st.markdown(f"Val F1: **{t3.get('val_metrics',{}).get('f1','?')}**")
        st.markdown(f"Val PR-AUC: **{t3.get('val_metrics',{}).get('pr_auc','?')}**")

# ── Apply filters ─────────────────────────────────────────────────────────────
filtered = pred_df[pred_df['country'].isin(sel_countries)].copy()
filtered['predicted'] = (filtered['proba'] >= threshold_val).astype(int)
filtered['risk']     = filtered['proba'].apply(risk_label)

# Snapshot (single date)
if sel_date and 'year_month' in filtered.columns:
    snap = filtered[filtered['year_month'] == sel_date]
else:
    snap = filtered.groupby(['country','admin1']).last().reset_index()

# ── MAIN HEADER ───────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding: 20px 0 10px; border-bottom: 1px solid #dee2e6; margin-bottom: 24px;">
  <h1 style="font-size:28px; font-weight:700; color:#1a1f26; margin:0;">
    🌍 Africa Food Crisis Early Warning System
  </h1>
  <p style="color:#6c757d; margin:6px 0 0; font-size:14px;">
    XGBoost Prediction Model · Conflict-Induced Food Crisis · 2024–2026 Forecast
  </p>
</div>
""", unsafe_allow_html=True)

# ── TOP KPI CARDS ─────────────────────────────────────────────────────────────
col1, col2, col3, col4, col5 = st.columns(5)

total_regions  = len(snap)
crisis_count   = int(snap['predicted'].sum())
safe_count     = total_regions - crisis_count
avg_prob       = snap['proba'].mean()
high_risk_cnt  = int((snap['proba'] >= 0.8).sum())

cards = [
    ("🌍 Regions Monitored", f"{total_regions:,}", "Active tracking", "#58a6ff"),
    ("🚨 Crisis Alerts", f"{crisis_count:,}", f"{crisis_count/total_regions*100:.1f}% of regions", "#e74c3c"),
    ("✅ Safe Regions", f"{safe_count:,}", f"{safe_count/total_regions*100:.1f}% of regions", "#2ecc71"),
    ("⚠️ Avg Risk Score", f"{avg_prob:.3f}", "Mean crisis probability", "#f39c12"),
    ("🔴 Emergency+ Regions", f"{high_risk_cnt:,}", "Probability ≥ 0.80", "#9b59b6"),
]

for col, (title, value, sub, color) in zip([col1,col2,col3,col4,col5], cards):
    with col:
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-title">{title}</div>
          <div class="metric-value" style="color:{color};">{value}</div>
          <div class="metric-sub">{sub}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── MAP + COUNTRY TABLE ───────────────────────────────────────────────────────
map_col, table_col = st.columns([3, 2])

with map_col:
    st.markdown('<div class="section-header">🗺️ Interactive Risk Map</div>', unsafe_allow_html=True)

    country_agg = filtered.groupby('country').agg(
        avg_prob=('proba','mean'),
        crisis_rate=('y_true','mean'),
        regions=('admin1','nunique'),
        predicted=('predicted','sum'),
    ).reset_index()
    country_agg['risk'] = country_agg['avg_prob'].apply(risk_label)

    # ── Map Settings ────────────────────────────────────────────────────────
    m = folium.Map(location=[5, 20], zoom_start=4,
                   tiles='CartoDB positron', width='100%')

    # ── 1. Choropleth (Country Fill) ────────────────────────────────────────
    GEO_JSON = "https://raw.githubusercontent.com/python-visualization/folium/main/examples/data/world-countries.json"
    
    # Align names with GeoJSON
    NAME_ALIGN = {
        'Democratic Republic of Congo': 'Democratic Republic of the Congo',
        'Congo': 'Republic of the Congo',
        'South Sudan': 'South Sudan',
        'Central African Republic': 'Central African Republic'
    }
    country_agg['geo_name'] = country_agg['country'].replace(NAME_ALIGN)

    folium.Choropleth(
        geo_data=GEO_JSON,
        data=country_agg,
        columns=['geo_name', 'avg_prob'],
        key_on='feature.properties.name',
        fill_color='YlOrRd',
        fill_opacity=0.7,
        line_opacity=0.4,
        legend_name='Mean Crisis Probability',
        nan_fill_color='#1a1f2c',
        highlight=True
    ).add_to(m)

    # ── 2. Fixed Radius Markers ──────────────────────────────────────────────
    for _, row in country_agg.iterrows():
        c = row['country']
        if c not in CENTROIDS: continue
        lat, lon = CENTROIDS[c]
        color = RISK_COLORS.get(row['risk'], '#888')
        
        radius = 8 # Fixed small radius for markers

        popup_html = f"""
        <div style='font-family:Inter;min-width:200px;background:#161b22;color:white;padding:10px;border-radius:8px;'>
          <b style='font-size:14px;color:#58a6ff;'>{c}</b><br><hr style='margin:5px 0;border-color:#30363d;'>
          <b>Risk Level:</b> <span style='color:{color};font-weight:700;'>{row['risk']}</span><br>
          <b>Alert Probability:</b> {row['avg_prob']:.3f}<br>
          <b>Affected Regions:</b> {int(row['regions'])}<br>
          <b>Historic Crisis Rate:</b> {row['crisis_rate']*100:.1f}%
        </div>"""

        folium.CircleMarker(
            location=[lat, lon],
            radius=radius,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.9,
            weight=2,
            tooltip=f"{c}: {row['risk']} ({row['avg_prob']:.2f})",
            popup=folium.Popup(popup_html, max_width=250)
        ).add_to(m)

        # Country label
        folium.Marker(
            [lat - 0.8, lon],
            icon=folium.DivIcon(
                html=f'<div style="font-size:9px;font-weight:600;color:white;text-shadow:0 0 4px black;width:80px;margin-left:-40px;text-align:center;">{c}</div>'
            )
        ).add_to(m)

    st_folium(m, height=500, use_container_width=True, returned_objects=[])

with table_col:
    st.markdown('<div class="section-header">🏳️ Country Risk Summary</div>', unsafe_allow_html=True)
    display_df = country_agg[['country','risk','avg_prob','predicted','regions']].copy()
    display_df.columns = ['Country','Risk Level','Avg Prob','Alerts','Regions']
    display_df = display_df.sort_values('Avg Prob', ascending=False)

    for _, r in display_df.iterrows():
        risk_cls = f"risk-{r['Risk Level'].lower()}"
        st.markdown(f"""
        <div class="metric-card" style="padding:12px 16px; margin:4px 0; background:#f8f9fa;">
          <div style="display:flex; justify-content:space-between; align-items:center;">
            <span style="font-weight:600; font-size:13px; color:#1a1f26;">{r['Country']}</span>
            <span class="risk-badge {risk_cls}">{r['Risk Level']}</span>
          </div>
          <div style="margin-top:6px; font-size:12px; color:#6c757d;">
            P={r['Avg Prob']:.3f} &nbsp;|&nbsp; {r['Alerts']}/{r['Regions']} regions alerted
          </div>
        </div>
        """, unsafe_allow_html=True)

st.divider()

# ── CHARTS ROW ────────────────────────────────────────────────────────────────
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.markdown('<div class="section-header">📊 Crisis Probability Distribution</div>', unsafe_allow_html=True)
    fig_hist = px.histogram(
        filtered, x='proba', nbins=50, color_discrete_sequence=['#58a6ff'],
        labels={'proba':'Crisis Probability'},
        title='Distribution of Predicted Crisis Probabilities'
    )
    fig_hist.add_vline(x=threshold_val, line_dash='dash', line_color='#e74c3c',
                       annotation_text=f'Threshold={threshold_val}')
    fig_hist.update_layout(
        template='plotly_white', paper_bgcolor='#ffffff', plot_bgcolor='#ffffff',
        height=300, showlegend=False, margin=dict(t=40,b=20,l=20,r=20)
    )
    st.plotly_chart(fig_hist, use_container_width=True)

with chart_col2:
    st.markdown('<div class="section-header">🌡️ Risk Level by Country</div>', unsafe_allow_html=True)
    country_sorted = country_agg.sort_values('avg_prob', ascending=True)
    colors_bar = [RISK_COLORS.get(r,'#888') for r in country_sorted['risk']]
    fig_bar = go.Figure(go.Bar(
        x=country_sorted['avg_prob'], y=country_sorted['country'],
        orientation='h', marker_color=colors_bar,
        text=country_sorted['avg_prob'].round(3), textposition='outside',
    ))
    fig_bar.add_vline(x=0.5, line_dash='dot', line_color='white', opacity=0.4)
    fig_bar.update_layout(
        template='plotly_white', paper_bgcolor='#ffffff', plot_bgcolor='#ffffff',
        height=320, title='Avg Crisis Probability by Country',
        xaxis_title='Crisis Probability', yaxis_title='',
        margin=dict(t=40, b=20, l=20, r=60)
    )
    st.plotly_chart(fig_bar, use_container_width=True)

# ── TIME SERIES ───────────────────────────────────────────────────────────────
if 'year_month' in filtered.columns:
    st.markdown('<div class="section-header">📈 Crisis Rate Over Time</div>', unsafe_allow_html=True)
    ts = filtered.groupby(['year_month','country'])['proba'].mean().reset_index()
    ts_pivot = ts.pivot(index='year_month', columns='country', values='proba').fillna(method='ffill')

    sel_ts_countries = st.multiselect(
        "Select countries for time series",
        options=ts_pivot.columns.tolist(),
        default=ts_pivot.columns[:5].tolist()
    )

    fig_ts = go.Figure()
    PALETTE = ['#58a6ff','#3fb950','#f78166','#d2a8ff','#ffa657',
               '#79c0ff','#56d364','#ff7b72','#bc8cff','#ffb86c']
    for i, c in enumerate(sel_ts_countries):
        if c in ts_pivot.columns:
            fig_ts.add_trace(go.Scatter(
                x=ts_pivot.index, y=ts_pivot[c],
                name=c, mode='lines+markers',
                line=dict(width=2, color=PALETTE[i % len(PALETTE)]),
                marker=dict(size=4)
            ))
    fig_ts.add_hrect(y0=0.6, y1=1.0, fillcolor='rgba(231,76,60,0.08)', line_width=0)
    fig_ts.add_hline(y=0.5, line_dash='dash', line_color='rgba(255,255,255,0.3)',
                     annotation_text='Crisis Threshold (0.5)')
    fig_ts.update_layout(
        template='plotly_white', paper_bgcolor='#ffffff', plot_bgcolor='#ffffff',
        height=340, title='Monthly Average Crisis Probability by Country',
        xaxis_title='Month', yaxis_title='Crisis Probability',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        margin=dict(t=50, b=30, l=40, r=20)
    )
    st.plotly_chart(fig_ts, use_container_width=True)

# ── MODEL PERFORMANCE ─────────────────────────────────────────────────────────
st.divider()
st.markdown('<div class="section-header">🤖 Model Performance Summary</div>', unsafe_allow_html=True)

perf_col1, perf_col2, perf_col3 = st.columns([2,2,3])

with perf_col1:
    if t3 and 'val_metrics' in t3:
        st.markdown("**Validation Set (2023)**")
        for k, v in t3['val_metrics'].items():
            bar_pct = int(v * 100)
            color = '#2ecc71' if v > 0.95 else '#f39c12' if v > 0.90 else '#e74c3c'
            st.markdown(f"""
            <div style="margin-bottom:10px;">
              <div style="display:flex;justify-content:space-between;font-size:12px;">
                <span>{k.upper()}</span><span style="color:{color};font-weight:700;">{v:.4f}</span>
              </div>
              <div style="background:#21262d;border-radius:4px;height:6px;margin-top:4px;">
                <div style="background:{color};width:{bar_pct}%;height:6px;border-radius:4px;"></div>
              </div>
            </div>""", unsafe_allow_html=True)
    else:
        st.info("Load data to see validation metrics")

with perf_col2:
    if t4 and 'test_metrics_f1_optimal' in t4:
        st.markdown("**Test Set (2024–2026) — Final**")
        for k, v in t4['test_metrics_f1_optimal'].items():
            if isinstance(v, float):
                bar_pct = int(v * 100)
                color = '#2ecc71' if v > 0.95 else '#f39c12' if v > 0.90 else '#e74c3c'
                st.markdown(f"""
                <div style="margin-bottom:10px;">
                  <div style="display:flex;justify-content:space-between;font-size:12px;">
                    <span>{k.upper()}</span><span style="color:{color};font-weight:700;">{v:.4f}</span>
                  </div>
                  <div style="background:#21262d;border-radius:4px;height:6px;margin-top:4px;">
                    <div style="background:{color};width:{bar_pct}%;height:6px;border-radius:4px;"></div>
                  </div>
                </div>""", unsafe_allow_html=True)
    else:
        st.info("Run Task 4 to see test metrics here")

with perf_col3:
    if t3 and 'feature_importance' in t3:
        st.markdown("**Top Features (XGBoost Gain)**")
        fi = t3['feature_importance'][:10]
        feat_names = [f['feature'] for f in fi]
        feat_vals  = [f['gain_pct'] for f in fi]
        colors_fi  = ['#d62728' if i < 3 else '#ff7f0e' if i < 6 else '#58a6ff'
                      for i in range(len(fi))]
        fig_fi = go.Figure(go.Bar(
            x=feat_vals[::-1], y=feat_names[::-1],
            orientation='h', marker_color=colors_fi[::-1],
            text=[f'{v:.1f}%' for v in feat_vals[::-1]], textposition='outside'
        ))
        fig_fi.update_layout(
            template='plotly_white', paper_bgcolor='#ffffff', plot_bgcolor='#ffffff',
            height=300, margin=dict(t=10,b=10,l=10,r=60),
            xaxis_title='Gain (%)', yaxis_title=''
        )
        st.plotly_chart(fig_fi, use_container_width=True)

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown("""
<div style="text-align:center; color:#484f58; font-size:12px; padding:10px 0;">
  Africa Food Crisis Early Warning System · XGBoost Model (Task 3) ·
  Conflict-Induced Food Security Prediction · 10Academy Project
</div>
""", unsafe_allow_html=True)
