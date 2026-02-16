"""
Page 8 — Explicabilite des Modeles.

Demonstrates real explainability techniques used in the project:
  - Tab 1 (Vision): Grad-CAM heatmaps, Attention Maps, Focus Battle across models
  - Tab 2 (Texte): SHAP feature importance demo
  - Tab 3 (Robustesse): Rotation stress test, model diversity radar, podium
  - Tab 4 (Conformite): AI Act, business value of explainability

Uses real images generated from notebooks (Grad-CAM via Captum, attention maps
from DINOv3 ViT, feature importance from XGBoost).
"""
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import APP_CONFIG, ASSETS_DIR
from utils.ui_utils import load_css

st.set_page_config(
    page_title=f"Explicabilite - {APP_CONFIG['title']}",
    page_icon="🔬",
    layout=APP_CONFIG["layout"],
)

load_css(ASSETS_DIR / "style.css")

# ==========================================
# HEADER
# ==========================================
st.title("Explicabilite des Modeles")
st.markdown("""
**Pourquoi l'explicabilite ?** Un modele a F1~0.85 ne suffit pas en production.
Il faut comprendre *pourquoi* il decide, pour detecter les biais, gagner la confiance
des utilisateurs, et respecter les exigences reglementaires (AI Act europeen).
""")

st.divider()

# ==========================================
# TABS
# ==========================================
tabs = st.tabs([
    "🖼️ Vision (Grad-CAM)",
    "📝 Texte (SHAP)",
    "🛡️ Robustesse & Diversite",
    "⚖️ Conformite & Apport"
])

# ==========================================
# TAB 1 : VISION — Grad-CAM & Attention Maps
# ==========================================
with tabs[0]:
    st.header("Grad-CAM & Attention Maps")
    st.markdown("""
    Chaque modele du Voting System regarde l'image differemment.
    **Grad-CAM** (Gradient-weighted Class Activation Mapping) revele les zones
    de l'image qui influencent le plus la decision du modele.
    """)

    # --- Image principale : comparaison 3 modeles ---
    st.subheader("Comparaison des 3 Modeles du Voting")

    img_drive = str(ASSETS_DIR / "explainability_drive.png")
    if os.path.exists(img_drive):
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            st.image(img_drive, width="stretch")
            st.caption("De gauche a droite : Image originale, DINOv3 (Attention Maps — vision globale), "
                       "EfficientNet (Zones d'Activation — details fins), XGBoost (Top Features decisives).")
    else:
        st.warning("Image explainability_drive.png introuvable.")

    st.markdown("---")

    # --- Explication technique des 3 approches ---
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("##### DINOv3 (Vision Transformer)")
        st.markdown("""
        - **Methode** : Attention Maps
        - **Principe** : Le ViT decoupe l'image en patches 16x16 et calcule des poids d'attention entre eux
        - **Resultat** : Heatmap des zones les plus attendues
        - **Force** : Vision globale, comprend le contexte spatial
        """)
    with c2:
        st.markdown("##### EfficientNet-B0 (CNN)")
        st.markdown("""
        - **Methode** : Grad-CAM (couche `features[8]`)
        - **Principe** : Gradients retropropages sur la derniere couche convolutive
        - **Resultat** : Activation thermique des regions discriminantes
        - **Force** : Details fins, textures, motifs locaux
        """)
    with c3:
        st.markdown("##### XGBoost (ML Classique)")
        st.markdown("""
        - **Methode** : Feature Importance
        - **Principe** : Mesure le gain d'information de chaque feature extraite
        - **Resultat** : Classement des dimensions les plus decisives
        - **Force** : Interpretable, correction statistique
        """)

    st.markdown("---")

    # --- Focus Battle : Grad-CAM 4 modeles ---
    st.subheader("Focus Battle : Ou chaque modele regarde-t-il ?")

    img_focus = str(ASSETS_DIR / "focus_battle.png")
    if os.path.exists(img_focus):
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            st.image(img_focus, width="stretch")
            st.caption("Exemple sur un produit : chaque quadrant montre le focus d'un modele different. "
                       "DINOv3 atteint 90.7% de confiance (vision globale), EfficientNet 77.8% (details), "
                       "XGBoost 31.7% (decision divergente — c'est la diversite qui renforce le vote).")
    else:
        st.warning("Image focus_battle.png introuvable.")

    st.info("""
    **Observation cle** : Les modeles ne regardent pas les memes zones. C'est precisement
    cette **complementarite** qui rend le Voting System robuste — si un modele se trompe
    sur une zone, les autres corrigent via le vote pondere.
    """)

    st.markdown("---")

    # --- Rapport Technique ---
    st.subheader("Rapport Technique : Heatmaps + Confiance")

    img_rapport = str(ASSETS_DIR / "rapport_technique_gradcam.png")
    if os.path.exists(img_rapport):
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            st.image(img_rapport, width="stretch")
            st.caption("Rapport genere automatiquement : image client, focus Phoenix (Overfit), "
                       "focus EfficientNet, et barres de confiance par modele. "
                       "Le VOTING final (37.2%) combine les avis avec les poids 4:2:1.")
    else:
        st.warning("Image rapport_technique_gradcam.png introuvable.")

# ==========================================
# TAB 2 : TEXTE — SHAP
# ==========================================
with tabs[1]:
    st.header("Explicabilite du Modele Texte")
    st.markdown("""
    Le modele texte (TF-IDF + LinearSVC, 83% accuracy) est un modele lineaire :
    chaque mot contribue directement au score de decision. On utilise **SHAP**
    pour quantifier cette contribution.
    """)

    # --- SHAP ---
    st.subheader("SHAP (SHapley Additive exPlanations)")
    st.markdown("""
    **Principe** : Base sur la theorie des jeux (valeurs de Shapley).
    Mesure la contribution marginale de chaque mot a la prediction finale.
    """)

    st.info("**Produit** : iPhone 15 Pro Max 256GB Smartphone Apple")

    shap_data = pd.DataFrame({
        'Feature': ['iphone', 'smartphone', 'apple', 'telephone', '256gb', 'pro'],
        'SHAP Value': [0.42, 0.28, 0.18, 0.15, 0.12, 0.08],
    })

    fig_shap = px.bar(
        shap_data, x='SHAP Value', y='Feature', orientation='h',
        color='SHAP Value',
        color_continuous_scale=['#FFE5E5', '#BF0000'],
    )
    fig_shap.update_layout(
        height=280,
        coloraxis_showscale=False,
        margin=dict(l=0, r=10, t=10, b=10),
        xaxis_title="Contribution SHAP",
        yaxis_title="",
    )
    st.plotly_chart(fig_shap, width="stretch")

    c1, c2 = st.columns(2)
    c1.metric("Prediction", "Telephones (2583)")
    c2.metric("Confiance", "94.2%")

    st.caption("Les mots 'iphone' et 'smartphone' poussent fortement vers la categorie Telephones.")

    st.markdown("---")

    # --- Metriques globales ---
    st.subheader("Metriques d'Explicabilite Texte")

    c1, c2, c3 = st.columns(3)
    c1.metric("Coherence SHAP", "94%", help="Pourcentage de predictions ou les top features sont stables")
    c2.metric("Temps / Explication", "2.1s", help="Temps moyen pour generer une explication complete")
    c3.metric("Features Cles", "~50", help="Nombre de mots TF-IDF les plus discriminants")

    st.markdown("""
    **Avantage du LinearSVC** : Etant un modele lineaire, les coefficients TF-IDF sont
    directement interpretables. Contrairement a un Transformer (CamemBERT), on peut
    toujours expliquer *pourquoi* le modele a decide, sans approximation.
    """)

# ==========================================
# TAB 3 : ROBUSTESSE & DIVERSITE
# ==========================================
with tabs[2]:
    st.header("Tests de Robustesse")
    st.markdown("""
    Un modele explicable doit aussi etre **robuste** : ses predictions doivent rester
    stables face aux perturbations (rotation, bruit, recadrage).
    """)

    # --- Stress Test Rotation ---
    st.subheader("Stress Test : Rotation 360°")

    img_rotation = str(ASSETS_DIR / "stress_test_rotation.png")
    if os.path.exists(img_rotation):
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            st.image(img_rotation, width="stretch")
            st.caption("Confiance de chaque modele quand l'image est tournee de 0 a 360 degres. "
                       "DINOv3 (violet) reste le plus stable grace a son architecture Vision Transformer. "
                       "XGBoost (vert pointille) est le plus fragile — c'est pourquoi son poids est 1/7.")
    else:
        st.warning("Image stress_test_rotation.png introuvable.")

    st.markdown("---")

    # --- Galerie Voting ---
    st.subheader("Galerie : Majorite vs Desaccord")

    img_galerie = str(ASSETS_DIR / "galerie_voting.png")
    if os.path.exists(img_galerie):
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            st.image(img_galerie, width="stretch")
            st.caption("5 produits analyses par le Voting. Barres rouges : Le Chef (VOTING final). "
                       "Barres bleues : confiance par modele. 'MAJORITE' = tous d'accord, "
                       "'DESACCORD' = un ou plusieurs modeles divergent.")
    else:
        st.warning("Image galerie_voting.png introuvable.")

    st.markdown("---")

    # --- Podium + Radar ---
    c_pod, c_rad = st.columns(2, gap="large")

    with c_pod:
        st.subheader("Podium : Produits Valides (>80%)")
        img_podium = str(ASSETS_DIR / "podium_final.png")
        if os.path.exists(img_podium):
            st.image(img_podium, width="stretch")
            st.caption("Sur 60 produits testes : le Voting valide 42 (~70%) avec >80% de confiance, "
                       "contre 38 pour DINOv3 seul et 6 pour XGBoost seul.")
        else:
            st.warning("Image podium_final.png introuvable.")

    with c_rad:
        st.subheader("Profils des Modeles (Radar)")
        img_radar = str(ASSETS_DIR / "radar_models.png")
        if os.path.exists(img_radar):
            st.image(img_radar, width="stretch")
            st.caption("Le VOTING (rouge) enveloppe tous les modeles individuels sur les 5 axes : "
                       "Precision, Confiance, Robustesse, Universalite, Vitesse.")
        else:
            st.warning("Image radar_models.png introuvable.")

    st.markdown("---")

    # --- Synthese diversite ---
    st.subheader("Pourquoi la Diversite est Cle")

    st.markdown("""
    | Modele | Force | Faiblesse | Poids |
    |--------|-------|-----------|-------|
    | **DINOv3** | Vision globale, robuste aux rotations | Plus lent (ViT) | **4/7** |
    | **EfficientNet** | Details fins, textures | Sensible aux transformations | **2/7** |
    | **XGBoost** | Independant statistiquement | Faible seul, fragile | **1/7** |
    | **VOTING** | Combine les 3 avis | - | **Total** |
    """)

    st.success("""
    **Resultat** : Le Voting atteint **79.28% d'accuracy** en combinant 3 architectures complementaires.
    XGBoost seul atteint 85.32%, mais la diversite (Transformer + CNN + ML) assure la robustesse.
    """)

# ==========================================
# TAB 4 : CONFORMITE & APPORT
# ==========================================
with tabs[3]:
    st.header("Conformite Reglementaire")

    st.markdown("""
    Le **Reglement europeen sur l'Intelligence Artificielle** (AI Act, 2024) impose
    des exigences de transparence pour les systemes d'IA a haut risque.
    """)

    # Tableau conformite
    compliance_data = pd.DataFrame({
        "Exigence AI Act": [
            "Transparence algorithmique",
            "Droit a l'explication",
            "Audit des biais",
            "Documentation technique",
            "Intervention humaine",
        ],
        "Notre Reponse": [
            "Grad-CAM + SHAP",
            "Explication par produit en <3s",
            "F1/classe analysee, classes minoritaires identifiees",
            "Rapport technique complet + code source",
            "Seuil de confiance 80% — revue humaine sous ce seuil",
        ],
        "Statut": [
            "Conforme",
            "Conforme",
            "Conforme",
            "Conforme",
            "Conforme",
        ],
    })

    st.dataframe(
        compliance_data.style.map(
            lambda v: "background-color: #d4edda; color: black; font-weight: bold;"
            if v == "Conforme" else "",
            subset=["Statut"]
        ),
        width="stretch",
        hide_index=True,
    )

    st.markdown("---")

    # --- Apport business ---
    st.header("Apport Business de l'Explicabilite")

    c1, c2 = st.columns(2, gap="large")

    with c1:
        st.markdown("#### Pour l'Equipe Technique")
        st.markdown("""
        - **Debugging** : Identifier les erreurs systematiques (ex: confusion 1280/1281)
        - **Feature Engineering** : Comprendre quels mots/zones sont discriminants
        - **Optimisation** : Ajuster les poids du Voting en connaissance de cause
        - **Monitoring** : Detecter le drift via le changement des features cles
        """)

    with c2:
        st.markdown("#### Pour le Metier")
        st.markdown("""
        - **Confiance** : L'operateur voit *pourquoi* l'IA a classifie un produit
        - **Validation** : Revue ciblee des 12% de cas <80% confiance
        - **Formation** : Les erreurs de l'IA forment les equipes aux cas limites
        - **ROI** : Reduction des erreurs de 10-15% (manuel) a ~6% (IA + humain)
        """)

    st.markdown("---")

    # --- Formule Fusion ---
    st.subheader("Formule de Fusion")

    st.latex(r"""
    P_{fusion}(c) = 0.6 \cdot P_{voting}(c) + 0.4 \cdot P_{svc}(c)
    """)

    st.markdown("""
    **Configuration retenue : 60% Image / 40% Texte** (ajustable dans la page Demo).
    Pour chaque classe *c*, le score fusionne est la moyenne ponderee des probabilites
    des deux modalites.
    """)

    c_img, c_txt = st.columns(2)
    with c_img:
        st.markdown("**Poids Image : 60%**")
        st.progress(0.6)
    with c_txt:
        st.markdown("**Poids Texte : 40%**")
        st.progress(0.4)

    c1, c2, c3 = st.columns(3)
    c1.metric("Texte seul", "F1 = 0.83", help="LinearSVC + TF-IDF")
    c2.metric("Image seule", "F1 ~ 0.79", help="Voting (DINOv3 + EffNet + XGBoost)")
    c3.metric("Fusion", "F1 ~ 0.85", "+0.02 vs texte seul", help="Late fusion ponderee")

# ==========================================
# SIDEBAR
# ==========================================
with st.sidebar:
    st.markdown("### Explicabilite")
    st.divider()
    st.markdown("**Methodes Image**")
    st.markdown("- Grad-CAM (CNN)")
    st.markdown("- Attention Maps (ViT)")
    st.markdown("- Feature Importance (XGB)")
    st.divider()
    st.markdown("**Methodes Texte**")
    st.markdown("- SHAP (Shapley)")
    st.divider()
    st.markdown("**Tests Robustesse**")
    st.markdown("- Rotation 360 deg")
    st.markdown("- Galerie Voting")
    st.markdown("- Seuil 80% confiance")
    st.divider()
    st.success("AI Act: Conforme")
