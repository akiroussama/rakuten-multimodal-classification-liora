"""
Page 7 — Qualite Logicielle & Tests.

Test pyramid visualization, code coverage, ML quality gates,
security testing (OWASP), and pytest commands.
"""
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import APP_CONFIG, ASSETS_DIR
from utils.ui_utils import load_css

st.set_page_config(
    page_title=f"Qualite - {APP_CONFIG['title']}",
    page_icon="🧪",
    layout=APP_CONFIG["layout"],
)

load_css(ASSETS_DIR / "style.css")

# Header
st.title("Qualite Logicielle & Tests")

# Metriques cles
st.divider()
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Tests", "210+")
col2.metric("Couverture", "85%")
col3.metric("Securite", "50+")
col4.metric("Tests ML", "40+")
col5.metric("Execution", "< 2 min")

# ==========================================
# PYRAMIDE DES TESTS
# ==========================================
st.divider()
st.header("Pyramide des Tests")

# Pyramide interactive avec Plotly
fig_pyramid = go.Figure()

# Base -> sommet
layers = [
    ("Unitaires", 90, "#C8E6C9", "Fonctions isolees, edge cases, formatage"),
    ("Integration", 30, "#B3E5FC", "Pages Streamlit, imports, pipeline complet"),
    ("ML / Performance", 40, "#D1C4E9", "Quality gates, non-regression, golden predictions"),
    ("Securite / OWASP", 50, "#FFCDD2", "XSS, injection, path traversal, pickle bombs"),
]

for i, (name, count, color, desc) in enumerate(layers):
    fig_pyramid.add_trace(go.Funnel(
        y=[name],
        x=[count],
        textinfo="value+text",
        text=[f" tests — {desc}"],
        marker=dict(color=color),
    ))

fig_pyramid.update_layout(
    height=350,
    showlegend=False,
    margin=dict(l=10, r=10, t=10, b=10),
    funnelmode="stack",
)
st.plotly_chart(fig_pyramid, width="stretch")

# ==========================================
# COUVERTURE DE CODE
# ==========================================
st.divider()
st.header("Couverture de Code")

coverage_data = pd.DataFrame([
    {"Module": "category_mapping", "Couverture": 100, "Tests": 15},
    {"Module": "config", "Couverture": 95, "Tests": 8},
    {"Module": "real_classifier", "Couverture": 92, "Tests": 25},
    {"Module": "preprocessing", "Couverture": 88, "Tests": 20},
    {"Module": "model_downloader", "Couverture": 85, "Tests": 12},
    {"Module": "data_loader", "Couverture": 82, "Tests": 18},
    {"Module": "predict_model", "Couverture": 80, "Tests": 22},
    {"Module": "ui_utils", "Couverture": 75, "Tests": 10},
])

fig_cov = px.bar(
    coverage_data.sort_values("Couverture"),
    x='Couverture', y='Module', orientation='h',
    color='Couverture',
    color_continuous_scale=['#FFCDD2', '#C8E6C9'],
    range_color=[60, 100],
    text='Couverture',
)
fig_cov.update_layout(height=350, coloraxis_showscale=False)
fig_cov.update_traces(texttemplate='%{text}%', textposition='outside')
st.plotly_chart(fig_cov, width="stretch")

# ==========================================
# QUALITY GATES ML
# ==========================================
st.divider()
st.header("Quality Gates Machine Learning")

st.markdown("""
Chaque deploiement doit passer ces **gates automatiques** avant mise en production.
""")

# Gauge charts pour quality gates
gauge_cols = st.columns(3)
with gauge_cols[0]:
    fig_g1 = go.Figure(go.Indicator(mode="gauge+number", value=79.28, title={"text": "Accuracy"},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#BF0000"},
               "threshold": {"line": {"color": "green", "width": 4}, "thickness": 0.75, "value": 75}}))
    fig_g1.update_layout(height=200, margin=dict(l=20, r=20, t=40, b=10))
    st.plotly_chart(fig_g1, width="stretch")

with gauge_cols[1]:
    fig_g2 = go.Figure(go.Indicator(mode="gauge+number", value=78.5, title={"text": "F1 Macro"},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#BF0000"},
               "threshold": {"line": {"color": "green", "width": 4}, "thickness": 0.75, "value": 75}}))
    fig_g2.update_layout(height=200, margin=dict(l=20, r=20, t=40, b=10))
    st.plotly_chart(fig_g2, width="stretch")

with gauge_cols[2]:
    fig_g3 = go.Figure(go.Indicator(mode="gauge+number+delta", value=80, title={"text": "Latence (ms)"},
        delta={"reference": 100, "decreasing": {"color": "green"}},
        gauge={"axis": {"range": [0, 200]}, "bar": {"color": "#BF0000"},
               "threshold": {"line": {"color": "green", "width": 4}, "thickness": 0.75, "value": 100}}))
    fig_g3.update_layout(height=200, margin=dict(l=20, r=20, t=40, b=10))
    st.plotly_chart(fig_g3, width="stretch")

gates_data = pd.DataFrame({
    "Gate": [
        "Accuracy >= 75%",
        "F1 Macro >= 75%",
        "Latence < 100ms (GPU)",
        "ECE < 0.05 (Calibration)",
        "Pas de regression > 2%",
        "27 classes predites",
    ],
    "Seuil": ["75%", "75%", "100ms", "0.05", "2%", "27"],
    "Actuel": ["79.28%", "~78.5%", "~80ms", "0.03", "0%", "27"],
    "Statut": ["PASS", "PASS", "PASS", "PASS", "PASS", "PASS"],
})

st.dataframe(
    gates_data.style.map(
        lambda v: "background-color: #d4edda; color: black; font-weight: bold;"
        if v == "PASS" else "",
        subset=["Statut"]
    ),
    width="stretch",
    hide_index=True,
)

# ==========================================
# SECURITE OWASP
# ==========================================
st.divider()
st.header("Tests de Securite (OWASP)")

col1, col2 = st.columns(2, gap="large")

with col1:
    security_data = pd.DataFrame({
        "Vulnerabilite": [
            "XSS (Script Injection)",
            "XSS (Event Handlers)",
            "HTML Injection",
            "Path Traversal",
            "SQL-like Injection",
            "Pickle Bomb / DoS",
        ],
        "Payloads": ["15+", "10+", "8+", "6+", "5+", "4+"],
        "Statut": ["Bloque", "Bloque", "Bloque", "Gere", "Bloque", "Gere"],
    })

    st.dataframe(
        security_data.style.map(
            lambda v: "background-color: #d4edda; color: black;"
            if v in ("Bloque", "Gere") else "",
            subset=["Statut"]
        ),
        width="stretch",
        hide_index=True,
    )

with col2:
    st.markdown("""
    **Principes de securite :**

    - Sanitization de **toutes** les entrees utilisateur
    - Validation des chemins de fichiers (pas de `../`)
    - Limitation taille uploads (10 MB max)
    - Pas de `pickle.load()` sur fichiers utilisateur
    - Headers CSP configures via `.streamlit/config.toml`
    - Pas de secrets en dur dans le code
    """)

# Stress test rotation image
img_rotation = str(ASSETS_DIR / "stress_test_rotation.png")
if os.path.exists(img_rotation):
    st.markdown("---")
    st.subheader("Test de Robustesse : Rotation 360")
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.image(img_rotation, width="stretch")
        st.caption("Confiance de chaque modele lors de la rotation 0-360. "
                   "DINOv3 (violet) est le plus stable. XGBoost (vert) est fragile — poids 1/7.")

# ==========================================
# TESTS NON-REGRESSION ML
# ==========================================
st.divider()
st.header("Non-Regression ML")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Golden Predictions")
    st.markdown("""
    10 produits de reference dont la prediction est **fixee** :
    - Si le modele change sa prediction → **ALERTE**
    - Tolerance : 2% de variation sur la confiance
    - Baseline stocke en JSON versionne
    """)

with col2:
    st.subheader("Robustesse")
    st.markdown("""
    Tests automatiques de resistance :
    - **Texte vide** → prediction gracieuse (pas de crash)
    - **Image corrompue** → fallback sur texte seul
    - **Caracteres speciaux** → sanitization avant prediction
    - **Classes inconnues** → "Produit non classifie"
    """)

# ==========================================
# COMMANDES
# ==========================================
st.divider()
st.header("Commandes pytest")

st.code("""
# Tous les tests
pytest -v

# Par type (markers)
pytest -m unit          # 90 tests unitaires
pytest -m integration   # 30 tests integration
pytest -m ml            # 40 tests ML
pytest -m security      # 50 tests securite

# Couverture HTML
pytest --cov=utils --cov-report=html

# Tests rapides (sans ML lourd)
pytest -m "not slow" -x
""", language="bash")

# Sidebar
with st.sidebar:
    st.markdown("### Qualite")
    st.divider()
    st.metric("Tests", "210+")
    st.metric("Couverture", "85%")
    st.metric("Quality Gates", "6/6 PASS")
    st.divider()
    st.code("pytest -v", language="bash")
