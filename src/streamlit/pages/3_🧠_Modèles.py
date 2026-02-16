"""
Page 3 — Model Architecture.

Tab 1 (Vision): Graphviz voting pipeline + calibration before/after + complementarity matrix + radar.
Tab 2 (Text): NLP pipeline, benchmark table (LinearSVC 83%), CamemBERT in perspectives.
"""
import streamlit as st
import pandas as pd
import graphviz
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import APP_CONFIG, ASSETS_DIR
from utils.ui_utils import load_css

st.set_page_config(
    page_title=f"Modeles - {APP_CONFIG['title']}",
    page_icon="🧠",
    layout=APP_CONFIG["layout"],
)

load_css(ASSETS_DIR / "style.css")

st.title("Architecture & Modeles")
st.markdown("---")

tabs = st.tabs(["🖼️ Vision (Image)", "📝 Semantique (Texte)"])

# ==========================================
# ONGLET 1 : VISION (IMAGE)
# ==========================================
with tabs[0]:
    # --- ZONE 1 : LE PIPELINE ---
    st.markdown("#### 1. Pipeline de Decision : Le 'Voting'")


    try:
        graph = graphviz.Digraph()
        graph.attr(rankdir='LR', size='12,3', ratio='compress', margin='0')

        graph.node('I', 'Input', shape='oval', style='filled', fillcolor='#e0e0e0', fontsize='10')
        graph.node('P', 'Preproc', shape='box', style='rounded', fontsize='10')
        graph.node('D', 'DINOv3', style='filled', fillcolor='#d1c4e9', fontsize='10')
        graph.node('E', 'EffNet', style='filled', fillcolor='#b3e5fc', fontsize='10')
        graph.node('X', 'XGBoost', style='filled', fillcolor='#c8e6c9', fontsize='10')
        graph.node('V', 'VOTING', shape='Mdiamond', style='filled', fillcolor='#ff8a80', fontsize='12')
        graph.node('O', 'Sortie', shape='oval', style='filled', fillcolor='gold', fontsize='10')

        graph.edge('I', 'P')
        graph.edge('P', 'D')
        graph.edge('P', 'E')
        graph.edge('D', 'X', style='dashed')
        graph.edge('E', 'X', style='dashed')
        graph.edge('D', 'V', label='x4', color='#7e57c2', penwidth='2')
        graph.edge('E', 'V', label='x2', color='#039be5', penwidth='2')
        graph.edge('X', 'V', label='x1', color='#2e7d32', penwidth='1')
        graph.edge('V', 'O')

        c_g1, c_g2, c_g3 = st.columns([1, 4, 1])
        with c_g2:
            st.graphviz_chart(graph, width="stretch")

    except Exception as e:
        st.warning(f"Impossible d'afficher le graphe du pipeline (Graphviz non installe ou 'dot' non trouve).")
        st.info("Pipeline : Input -> Preproc -> [DINOv3 (x4), EfficientNet (x2), XGBoost (x1)] -> Voting -> Sortie")

    st.markdown("---")

    # --- ZONE 2 : RADAR + COMPLEMENTARITE ---
    st.markdown("#### 2. Profils & Complementarite des Modeles")

    c1, c2 = st.columns(2, gap="medium")

    with c1:
        st.markdown("##### Synthese (Radar)")
        img_radar = str(ASSETS_DIR / "radar_models.png")
        if os.path.exists(img_radar):
            st.image(img_radar, width="stretch")
            st.caption("Profils des modeles. XGBoost (85.32%) domine en accuracy individuelle, "
                       "mais le Voting apporte robustesse et diversite architecturale.")

    with c2:
        st.markdown("##### Matrice de Complementarite")
        img_compl = str(ASSETS_DIR / "complementarity_matrix.png")
        if os.path.exists(img_compl):
            st.image(img_compl, width="stretch")
            st.caption("XGBoost est 'Independant' (Alien) de tous les autres modeles. "
                       "C'est precisement cette independance qui securise le vote.")

    st.markdown("---")

    # --- ZONE 3 : CALIBRATION AVANT/APRES ---
    st.markdown("#### 3. Calibration : Probleme et Solution")

    st.markdown("""
    **Probleme** : XGBoost produit des scores de confiance groupes autour de 25% (pic vert),
    ce qui dilue le vote. **Solution** : Le 'Sharpening' (elevation au cube : p^3)
    force XGBoost a trancher nettement.
    """)

    c1, c2 = st.columns(2, gap="medium")

    with c1:
        st.markdown("##### AVANT (Probleme)")
        img_before = str(ASSETS_DIR / "calibration_before.png")
        if os.path.exists(img_before):
            st.image(img_before, width="stretch")
            st.caption("XGBoost (vert) concentre ses scores a ~25%. Le seuil de 80% (ligne rouge) "
                       "est rarement atteint — le modele hesite trop.")

    with c2:
        st.markdown("##### APRES (Sharpening)")
        img_after = str(ASSETS_DIR / "calibration_after.png")
        if os.path.exists(img_after):
            st.image(img_after, width="stretch")
            st.caption("Apres sharpening, XGBoost (vert) depasse le seuil de 80%. "
                       "Le VOTING (rouge) est mieux calibre — decisions plus nettes.")

    st.markdown("---")

    # --- ZONE 4 : TABLEAU RECAPITULATIF ---
    st.markdown("#### 4. Recapitulatif des Modeles Image")

    st.markdown("""
    | Modele | Architecture | Accuracy | Poids | Role |
    |--------|-------------|----------|-------|------|
    | **DINOv3** | ViT-B/14 (Transformer) | 79.43% seul | **4/7** | Vision globale, Le Patron |
    | **EfficientNet** | CNN B0 | 66.63% seul | **2/7** | Details fins, L'Expert |
    | **XGBoost** | Gradient Boosting | **85.32%** seul | **1/7** | Champion accuracy, Le Statisticien |
    | **VOTING** | Ensemble pondere | **79.28%** | - | Robustesse, diversite architecturale |
    """)

# ==========================================
# ONGLET 2 : SEMANTIQUE (TEXTE)
# ==========================================
with tabs[1]:
    st.subheader("Traitement du Langage Naturel (NLP)")

    col_txt1, col_txt2 = st.columns([1, 1], gap="large")

    with col_txt1:
        st.markdown("#### Pipeline Technique")
        st.code("""
        1. Clean : HTML, Lowercase, Specials
        2. Langue : Detection + Traduction FR
        3. Vector : TF-IDF FeatureUnion
           - Word n-grams (1,2)
           - Char n-grams (3,5)
        4. Model : LinearSVC (SVM lineaire)
        """, language="python")

        st.info("< 10ms par produit. Ideal temps reel.")

    with col_txt2:
        st.markdown("#### Benchmark Texte")

        df_text = pd.DataFrame({
            "Modele": ["LinearSVC", "Random Forest", "LogReg"],
            "F1-Score": ["83.0%", "72.0%", "69.5%"],
            "Vitesse": ["< 10ms", "~50ms", "< 10ms"],
            "Interpretable": ["Oui", "Partiel", "Oui"],
        })

        st.dataframe(
            df_text.style.highlight_max(
                axis=0,
                subset=["F1-Score"],
                props="color: black; background-color: #d4edda; font-weight: bold;"
            ),
            width="stretch",
            hide_index=True
        )

    st.markdown("---")

    st.markdown("""
    **Pourquoi pas CamemBERT ?**

    CamemBERT (Transformer) n'a pas ete retenu pour ce projet. Sur un corpus de descriptions
    produit courtes (moyenne 8 mots), multilingues, avec des codes/references, la litterature
    montre que les approches TF-IDF + SVM rivalisent avec les Transformers :

    | Critere | LinearSVC | CamemBERT (estime) |
    |---------|-----------|-----------|
    | F1-Score | **83.0%** | ~80-82% |
    | Vitesse inference | **< 10ms** | ~200ms |
    | Interpretabilite | **Coefficients directs** | Boite noire |
    | Ressources GPU | **Aucune** | 4 GB VRAM |
    | Deploiement | **Simple (joblib)** | Complexe (PyTorch) |

    Le gain potentiel ne justifie pas la complexite d'infrastructure (GPU, temps d'entrainement x100).
    CamemBERT reste une **perspective d'amelioration** pour un prochain cycle.
    """)

# Sidebar
with st.sidebar:
    st.markdown("### Modeles")
    st.divider()
    st.metric("Image (Voting)", "F1 ~ 0.79")
    st.metric("Texte (LinearSVC)", "F1 = 0.83")
    st.metric("Fusion", "F1 ~ 0.85")
    st.divider()
    st.markdown("**Voting Weights**")
    st.markdown("DINOv3: 4/7 | EffNet: 2/7 | XGB: 1/7")
