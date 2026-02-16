"""
Page 2 — Preprocessing Pipeline.

Visualizes text pipeline (cleaning -> language detection -> TF-IDF ~280K features)
and image pipeline (resize 224px -> normalization -> feature extraction).
Includes real images from notebooks showing the pipeline in action.
"""
import streamlit as st
import re
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import APP_CONFIG, ASSETS_DIR
from utils.ui_utils import load_css

st.set_page_config(
    page_title=f"Preprocessing - {APP_CONFIG['title']}",
    page_icon="⚙️",
    layout=APP_CONFIG["layout"],
)

load_css(ASSETS_DIR / "style.css")

# Header
st.title("Pipeline de Preprocessing")

# Metriques
col1, col2, col3, col4 = st.columns(4)
col1.metric("Produits", "84 916")
col2.metric("Vocabulaire TF-IDF", "~280K")
col3.metric("Features Image", "1 024")
col4.metric("Langues", "5")

# ==========================================
# PIPELINE IMAGE (avec images reelles)
# ==========================================
st.divider()
st.header("Pipeline Image")

# Pipeline visuel complet
img_pipeline = str(ASSETS_DIR / "image_pipeline.png")
if os.path.exists(img_pipeline):
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.image(img_pipeline, width="stretch")
        st.caption("Pipeline complet : 1. Image originale (500x500) → 2. Representation matricielle (Canal R) → "
                   "3. Data Augmentation (rotations) → 4. Extraction de features (vecteur 2048D).")

st.markdown("---")

col1, col2, col3 = st.columns(3)
with col1:
    st.subheader("1. Resize")
    st.markdown("""
    - 500x500 → **224 x 224** pixels
    - Format attendu par DINOv3/EfficientNet
    - Interpolation bilineaire
    """)

with col2:
    st.subheader("2. Normalisation")
    st.markdown("""
    - Mean/std ImageNet
    - `[0.485, 0.456, 0.406]`
    - Mise a l'echelle [0,1]
    """)

with col3:
    st.subheader("3. Extraction Features")
    st.markdown("""
    - DINOv3 pre-entraine (ViT-B/14)
    - **1 024 features** par image
    - Self-supervised learning
    """)

# Exemples resize
img_resize = str(ASSETS_DIR / "image_resize.png")
if os.path.exists(img_resize):
    st.markdown("---")
    st.subheader("Avant / Apres Resize")
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.image(img_resize, width="stretch")
        st.caption("Ligne du haut : images originales (500x500). Ligne du bas : apres resize (224x224). "
                   "Les produits restent reconnaissables malgre la reduction.")

# Data augmentation
img_augmentation = str(ASSETS_DIR / "data_augmentation.png")
if os.path.exists(img_augmentation):
    st.markdown("---")
    st.subheader("Data Augmentation")
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.image(img_augmentation, width="stretch")
        st.caption("Exemple d'augmentation sur un produit (classe 1180 - Figurines). "
                   "5 variantes generees par rotation, flip, changement de luminosite.")

# ==========================================
# PIPELINE TEXTE
# ==========================================
st.divider()
st.header("Pipeline Texte")

col1, col2, col3 = st.columns(3)
with col1:
    st.subheader("1. Nettoyage")
    st.markdown("""
    - Suppression balises HTML
    - Caracteres speciaux
    - Normalisation espaces
    - Lowercase
    """)

with col2:
    st.subheader("2. Detection Langue")
    st.markdown("""
    - Detection via langid
    - Traduction → FR
    - ~85% deja en francais
    - 5 langues detectees
    """)

with col3:
    st.subheader("3. Vectorisation TF-IDF")
    st.markdown("""
    - FeatureUnion (word 1-2 + char 3-5)
    - **~280K dimensions**
    - designation + description
    - Sparse matrix
    """)

# Demo interactive
st.markdown("---")
st.subheader("Demo : Nettoyage en Direct")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Input (texte brut)**")
    demo_text = st.text_area(
        "Texte brut",
        value='<p>iPhone 15 Pro Max</p> - Smartphone Apple, ecran OLED 6.7"',
        height=100
    )

with col2:
    st.markdown("**Output (nettoye)**")
    if demo_text:
        cleaned = re.sub(r'<[^>]+>', '', demo_text)
        cleaned = re.sub(r'[^\w\s\-]', ' ', cleaned)
        cleaned = ' '.join(cleaned.lower().split())
        st.code(cleaned)
        st.caption(f"{len(cleaned)} caracteres, {len(cleaned.split())} mots")

# ==========================================
# GESTION DU DESEQUILIBRE
# ==========================================
st.divider()
st.header("Strategie de Gestion du Desequilibre")

st.markdown("""
Le dataset Rakuten presente un **desequilibre de 13.4x** entre la classe majoritaire
(Telephones: 10K) et la plus petite (Figurines: ~800). Nous avons mis en place
une strategie de class weighting pour compenser.
""")

c1, c2 = st.columns(2, gap="large")

with c1:
    img_efficacy = str(ASSETS_DIR / "strategy_efficacy.png")
    if os.path.exists(img_efficacy):
        st.image(img_efficacy, width="stretch")
        st.caption("Preuve d'efficacite : le taux de detection des classes minoritaires (Figurines) "
                   "passe de 60.7% (modele naif) a 83.8% (avec strategie).")

with c2:
    img_impact = str(ASSETS_DIR / "imbalance_impact.png")
    if os.path.exists(img_impact):
        st.image(img_impact, width="stretch")
        st.caption("Impact sur les courbes d'entrainement : le modele naif (rouge) converge plus vite "
                   "mais le modele avec strategie (vert) apprend mieux les classes rares.")

# ==========================================
# CHOIX TECHNIQUES
# ==========================================
st.divider()
st.header("Justification des Choix Techniques")

tab_text, tab_image = st.tabs(["Texte", "Image"])

with tab_text:
    st.markdown("""
    | Choix | Alternative | Justification |
    |-------|-------------|---------------|
    | **TF-IDF** | Word2Vec, FastText | Performance equivalente, meilleure interpretabilite, pas de pre-entrainement |
    | **Traduction FR** | Multilingue | 85% FR, vocabulaire unifie simplifie le modele |
    | **Pas de lemmatisation** | Spacy lemma | Preserver marques (iPhone, PlayStation) — mots discriminants |
    | **FeatureUnion** | Concatenation | Word n-grams + Char n-grams capturent morphologie et semantique |
    """)

with tab_image:
    st.markdown("""
    | Choix | Alternative | Justification |
    |-------|-------------|---------------|
    | **DINOv3** | ResNet50, VGG16 | Self-supervised, meilleures features sans labels (SOTA 2023) |
    | **224x224** | 500x500 | Standard ViT, bon compromis resolution/vitesse |
    | **Voting 3 modeles** | Single model | Robustesse par diversite (ViT + CNN + ML classique) |
    | **Poids 4:2:1** | Poids egaux | DINOv3 dominant car le plus precis et robuste |
    """)

# Sidebar
with st.sidebar:
    st.markdown("### Preprocessing")
    st.divider()
    st.markdown("**Texte**")
    st.markdown("Nettoyage → Langue → TF-IDF")
    st.divider()
    st.markdown("**Image**")
    st.markdown("Resize → Norm → DINOv3")
    st.divider()
    st.markdown("**Desequilibre**")
    st.markdown("Class weighting")
    st.markdown("60.7% → 83.8% rappel")
