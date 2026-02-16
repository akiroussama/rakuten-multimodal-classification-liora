"""
Home page — Rakuten Product Classifier.

Entry point for the Streamlit multi-page app. Displays project overview,
key metrics (84K products, 27 categories, F1~0.85 fusion),
pipeline summary, and navigation buttons to Demo and Data pages.
"""
import streamlit as st
from config import APP_CONFIG, MODEL_CONFIG, ASSETS_DIR
from utils.ui_utils import load_css
from utils.category_mapping import get_all_categories

# Configuration
st.set_page_config(
    page_title=APP_CONFIG["title"],
    page_icon=APP_CONFIG["icon"],
    layout=APP_CONFIG["layout"],
    initial_sidebar_state=APP_CONFIG["initial_sidebar_state"],
)

load_css(ASSETS_DIR / "style.css")

# Session state
if "classifier" not in st.session_state:
    from utils.mock_classifier import DemoClassifier
    st.session_state.classifier = DemoClassifier()
if "use_mock" not in st.session_state:
    st.session_state.use_mock = MODEL_CONFIG["use_mock"]

# Header
st.title("Rakuten Product Classifier")
st.markdown("Classification automatique de produits e-commerce en 27 catégories.")

st.divider()

# Métriques clés
col1, col2, col3, col4 = st.columns(4)
col1.metric("Produits", "84 916")
col2.metric("Catégories", "27")
col3.metric("Modalités", "Texte + Image")
col4.metric("F1 Fusion", "0.85")

st.divider()

# Contexte
st.header("Le Projet")
st.markdown("""
Rakuten France traite des millions de produits chaque année.
Ce projet automatise la classification par Deep Learning multimodal,
combinant analyse de texte (désignation, description) et d'image.
""")

# Pipeline
st.header("Pipeline")
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Texte")
    st.markdown("Nettoyage, détection de langue, TF-IDF")

with col2:
    st.subheader("Image")
    st.markdown("Features DINOv3 pré-entraîné")

with col3:
    st.subheader("Fusion")
    st.markdown("Combinaison multimodale → 27 classes")

st.divider()

# Catégories (grille compacte)
st.header("27 Catégories")
categories = get_all_categories()
cat_list = list(categories.items())

for i in range(0, 27, 9):
    cols = st.columns(9)
    for j, col in enumerate(cols):
        if i + j < 27:
            code, (name, full_name, emoji) = cat_list[i + j]
            col.markdown(f"{emoji} **{name}**")

st.divider()

# Navigation
st.header("Tester")
col1, col2 = st.columns(2)
with col1:
    if st.button("Classifier un produit", width="stretch", type="primary"):
        st.switch_page("pages/5_🔍_Démo.py")
with col2:
    if st.button("Explorer les données", width="stretch", type="primary"):
        st.switch_page("pages/1_📊_Données.py")

# Footer
st.divider()
st.caption("Projet DataScientest — Formation BMLE Octobre 2025")

# Sidebar
with st.sidebar:
    st.markdown("### Rakuten")
    st.markdown("Product Classifier")
    st.divider()
    if st.session_state.use_mock:
        st.warning("Mode Démo")
    else:
        st.success("Production")
