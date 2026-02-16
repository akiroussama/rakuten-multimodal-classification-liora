"""
Page 6 — Conclusions & Perspectives.

Summarizes results (F1~0.85 fusion), business impact (5min -> <1s per product),
known limitations (minority classes, temporal drift), and future improvements
(CamemBERT, CLIP, OCR on images, CI/CD monitoring pipeline).
"""
import streamlit as st
import plotly.graph_objects as go
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import APP_CONFIG, ASSETS_DIR
from utils.ui_utils import load_css

st.set_page_config(
    page_title=f"Conclusions - {APP_CONFIG['title']}",
    page_icon="💡",
    layout=APP_CONFIG["layout"],
)

load_css(ASSETS_DIR / "style.css")

# Header
st.title("Conclusions & Perspectives")

# ==========================================
# RESULTATS
# ==========================================
st.divider()
st.header("Resultats Cles")

col1, col2, col3, col4 = st.columns(4)
col1.metric("F1 Fusion", "~0.85", "+0.02 vs texte seul")
col2.metric("Categories", "27", "Toutes couvertes")
col3.metric("Modeles testes", "40+", "DL + ML + NLP")
col4.metric("Champion Texte", "LinearSVC", "F1 = 83%")

st.success("Classification automatique de **84 916 produits** en 27 categories, "
           "avec un systeme de fusion texte+image atteignant un F1-score d'environ 0.85.")

# Image recapitulative : accuracy par modele
img_accuracy = str(ASSETS_DIR / "model_accuracy_comparison.png")
if os.path.exists(img_accuracy):
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.image(img_accuracy, width="stretch")
        st.caption("Accuracy par modele (scores corriges). XGBoost (85.32%) est le champion individuel, Voting (79.28%) apporte la robustesse.")

# Graphique funnel : progression des resultats
st.subheader("Progression des Performances")

fig_funnel = go.Figure(go.Funnel(
    y=["Baseline (RandomForest)", "Image (Voting 3 modeles)", "Texte (LinearSVC)",
       "Fusion (Texte + Image)"],
    x=[72, 79, 83, 85],
    textinfo="value+percent initial",
    marker=dict(color=["#FFCDD2", "#EF9A9A", "#E57373", "#BF0000"]),
))
fig_funnel.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10))
st.plotly_chart(fig_funnel, width="stretch")
st.caption("De la baseline a la fusion : chaque etape apporte un gain mesurable.")

# ==========================================
# IMPACT BUSINESS
# ==========================================
st.divider()
st.header("Impact Business")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Avant (Manuel)")
    st.markdown("""
    - Temps : **~5 min/produit**
    - Erreur : **10-15%**
    - Scalabilite : Limitee (equipe humaine)
    - Cout : Eleve (main d'oeuvre)
    """)

with col2:
    st.subheader("Apres (IA)")
    st.markdown("""
    - Temps : **< 1 sec/produit**
    - Erreur : **~15%** (fusion)
    - Scalabilite : **100K+/jour** (1 serveur)
    - Taux automatisation : **~70%** (seuil 80%)
    """)

# Podium validation
img_podium = str(ASSETS_DIR / "podium_final.png")
if os.path.exists(img_podium):
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.image(img_podium, width="stretch")
        st.caption("Produits valides avec >80% de confiance : Voting 42/60 (~70%), XGBoost seul 6/60 (10%).")

# Radar recapitulatif
img_radar = str(ASSETS_DIR / "radar_models.png")
if os.path.exists(img_radar):
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.image(img_radar, width="stretch")
        st.caption("Profils des modeles. XGBoost domine en accuracy (85.32%), le Voting apporte diversite et robustesse.")

# ROI chart
st.subheader("Retour sur Investissement")

fig_roi = go.Figure()
fig_roi.add_trace(go.Bar(
    x=["Temps/produit", "Taux erreur", "Volume/jour", "Automatisation"],
    y=[300, 12.5, 1, 0],
    name="Avant (Manuel)",
    marker_color="#FFCDD2",
    text=["5 min", "12.5%", "~300/jour", "0%"],
    textposition="auto",
))
fig_roi.add_trace(go.Bar(
    x=["Temps/produit", "Taux erreur", "Volume/jour", "Automatisation"],
    y=[1, 15, 100, 70],
    name="Apres (IA)",
    marker_color="#BF0000",
    text=["< 1 sec", "~15%", "100K+/jour", "~70%"],
    textposition="auto",
))
fig_roi.update_layout(
    barmode='group', height=350,
    margin=dict(l=10, r=10, t=30, b=10),
    yaxis_title="Score (normalise)",
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
)
st.plotly_chart(fig_roi, width="stretch")

# ==========================================
# LIMITES
# ==========================================
st.divider()
st.header("Limites Identifiees")

st.markdown("""
| Limite | Impact | Severite | Mitigation |
|--------|--------|----------|------------|
| Classes minoritaires (1180, 60) | F1 plus faible (~60%) | Moyenne | Class weighting, augmentation |
| Confusion 1280/1281 | ~15% d'erreur mutuelle | Moyenne | Fusion texte+image aide |
| Drift temporel | Nouveaux produits non reconnus | Haute | Reentrainement periodique |
| Qualite texte variable | Descriptions manquantes 35% | Moyenne | Fallback sur designation seule |
| XGBoost fragile seul | 34% confiance sur cas difficiles | Basse | Poids 1/7 dans le voting |
""")

# ==========================================
# PERSPECTIVES
# ==========================================
st.divider()
st.header("Perspectives d'Evolution")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Court terme")
    st.markdown("""
    - Augmentation classes minoritaires
    - OCR sur images (texte dans images)
    - Seuil de confiance adaptatif
    - Enrichir les donnees d'entrainement
    """)

with col2:
    st.subheader("Moyen terme")
    st.markdown("""
    - Evaluer **CamemBERT** (Transformers)
    - Modele **CLIP** (vision-language)
    - Active learning (cas ambigus)
    - Cross-validation robuste
    """)

with col3:
    st.subheader("MLOps")
    st.markdown("""
    - Pipeline **CI/CD** automatise
    - **Monitoring** drift (Evidently)
    - **A/B testing** en production
    - Reentrainement periodique
    """)

# ==========================================
# CONCLUSION
# ==========================================
st.divider()
st.header("Conclusion")

st.info("""
**Mission accomplie** : Classification multimodale de produits e-commerce
avec un **F1-score d'environ 0.85** (fusion texte+image). Solution deployee sur
Hugging Face Spaces, scalable a 100K+ produits/jour, avec
interpretabilite complete (SHAP, Grad-CAM) et conformite AI Act.
""")

# Sidebar
with st.sidebar:
    st.markdown("### Conclusions")
    st.divider()
    st.success("F1 Fusion: ~0.85")
    st.success("27 categories")
    st.success("40+ modeles testes")
    st.success("~70% automatisation")
    st.divider()
    st.markdown("**Remerciements**")
    st.markdown("DataScientest, Mentors, Equipe")
