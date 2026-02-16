"""
Page 4 — Interactive Demo (PoC) with inline Explainability.

Three tabs for live classification using real trained models:
  - Text:   Enter a product description -> LinearSVC prediction + TF-IDF feature importance
  - Image:  Upload a product photo -> Voting System + per-model breakdown
  - Fusion: Both inputs + adjustable weight slider + full explainability

Business context: automates product classification from ~5min/product (manual)
to <1s (AI) with ~70% full automation rate at 80% confidence threshold.
"""
import streamlit as st
import time
import os
import sys
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import APP_CONFIG, ASSETS_DIR, MODELS_DIR
from utils.ui_utils import load_css
from utils.real_classifier import MultimodalClassifier
from utils.model_downloader import ensure_models

st.set_page_config(
    page_title=f"Démo - {APP_CONFIG['title']}",
    page_icon="🔍",
    layout=APP_CONFIG["layout"],
)

load_css(ASSETS_DIR / "style.css")

st.title("Démonstration Interactive")

# Contexte métier PoC
st.info("""
**Proof of Concept — Classificateur de produits Rakuten France**

Ce PoC simule le processus de classification automatique des produits sur une marketplace e-commerce.
**Avant** : un opérateur classifie manuellement chaque produit (~5 min, 10-15% d'erreur).
**Après** : le système IA classifie en <1 seconde avec ~15% d'erreur, permettant de traiter **100K+ produits/jour**.
Le taux d'automatisation avec seuil de confiance à 80% est de **~70%** (les ~30% restants partent en revue humaine).
""")
st.markdown("---")

# Architecture visuelle de la demo
with st.expander("Architecture du systeme de classification", expanded=False):
    img_explain = str(ASSETS_DIR / "explainability_drive.png")
    if os.path.exists(img_explain):
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            st.image(img_explain, width="stretch")
            st.caption("Les 3 modeles du Voting analysent chaque image differemment : "
                       "DINOv3 (attention globale), EfficientNet (details), XGBoost (features statistiques).")

    img_accuracy = str(ASSETS_DIR / "model_accuracy_comparison.png")
    if os.path.exists(img_accuracy):
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            st.image(img_accuracy, width="stretch")
            st.caption("Le VOTING combine 3 architectures complementaires pour une accuracy image de 79.28%.")

# Download models from HF Hub if not present locally (runs once, cached)
@st.cache_resource
def get_clf():
    ensure_models(MODELS_DIR)
    return MultimodalClassifier()

clf = get_clf()

# --- FONCTIONS UTILITAIRES ---

def show_results(results, title="Résultats"):
    """Affiche le gagnant et le top 5 avec des barres."""
    if not results:
        st.error("Le modèle n'a renvoyé aucun résultat.")
        return

    top = results[0]
    c1, c2 = st.columns([1, 3])
    with c1:
        st.metric("Prédiction", top['label'])
    with c2:
        st.success(f"**{top['name']}**")
        st.progress(top['confidence'])
        st.caption(f"Confiance: {top['confidence']:.1%}")

    chart_data = [{"Produit": r['name'], "Confiance": r['confidence']} for r in results[:5]]
    df_chart = pd.DataFrame(chart_data)
    st.dataframe(
        df_chart.style.background_gradient(cmap="Greens", subset=["Confiance"]).format({"Confiance": "{:.1%}"}),
        width="stretch",
        hide_index=True
    )


def show_text_explainability(text, explain_data, predicted_name):
    """Affiche l'explicabilite texte : mots-cles + bar chart TF-IDF."""
    st.markdown("---")
    st.markdown("#### Explicabilite Texte")

    # Mots-cles detectes (simple)
    tokens = text.lower().split()
    kept = [w for w in tokens if len(w) > 3]
    if kept:
        st.markdown("**Mots-cles detectes :** " + " ".join([f"`{w}`" for w in kept]))

    # Bar chart des contributions TF-IDF
    if explain_data:
        df_exp = pd.DataFrame(explain_data)
        # Clean feature names (remove pipeline prefixes)
        df_exp['feature'] = df_exp['feature'].apply(
            lambda x: x.split('__')[-1] if '__' in x else x)

        fig = px.bar(
            df_exp, x='contribution', y='feature', orientation='h',
            color='contribution',
            color_continuous_scale=['#BF0000', '#FFE5E5', '#C8E6C9', '#2E7D32'],
            color_continuous_midpoint=0,
        )
        fig.update_layout(
            height=min(50 * len(df_exp) + 60, 400),
            margin=dict(l=0, r=10, t=10, b=10),
            xaxis_title="Contribution a la decision",
            yaxis_title="",
            coloraxis_showscale=False,
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig, width="stretch")
        st.caption(f"Les features TF-IDF qui poussent le plus vers **{predicted_name}**. "
                   "Positif = confirme la categorie, Negatif = pousse vers une autre.")
    else:
        st.caption("Explicabilite indisponible pour cette prediction.")


def show_image_explainability(per_model):
    """Affiche le breakdown par modele du Voting System."""
    st.markdown("---")
    st.markdown("#### Explicabilite Image — Conseil des Sages")

    if not per_model:
        st.caption("Breakdown par modele indisponible.")
        return

    # Modeles et leur config
    model_config = {
        "DINOv3": {"icon": "🦖", "role": "Le Patron", "desc": "Vision Transformer — contexte global", "weight": "4/7"},
        "EfficientNet": {"icon": "👁️", "role": "L'Expert", "desc": "CNN — details et textures", "weight": "2/7"},
        "XGBoost": {"icon": "⚡", "role": "Le Statisticien", "desc": "ML classique — correction", "weight": "1/7"},
    }

    cols = st.columns(len(per_model))
    predictions = []
    for i, (model_name, info) in enumerate(per_model.items()):
        cfg = model_config.get(model_name, {"icon": "🤖", "role": model_name, "desc": "", "weight": "?"})
        with cols[i]:
            st.metric(
                f"{cfg['icon']} {model_name}",
                f"{info['confidence']:.0%}",
                f"Poids {cfg['weight']}"
            )
            st.progress(min(info['confidence'], 1.0))
            st.caption(f"**{info['name']}**")
            st.caption(cfg['desc'])
        predictions.append(info['label'])

    # Accord / Desaccord
    unique_preds = set(predictions)
    if len(unique_preds) == 1:
        st.success("**Unanime** — Les 3 modeles sont d'accord.")
    else:
        st.warning(f"**Desaccord** — {len(unique_preds)} predictions differentes. "
                   "Le vote pondere tranche.")

    # Expander avec image statique de reference
    with st.expander("Comment les modeles voient une image", expanded=False):
        img_drive = str(ASSETS_DIR / "explainability_drive.png")
        if os.path.exists(img_drive):
            st.image(img_drive, width="stretch")
            st.caption("DINOv3 (attention globale), EfficientNet (zones d'activation), "
                       "XGBoost (features statistiques).")


def show_fusion_explainability(text_top, image_top, per_model, text_explain,
                                w_text, w_image, predicted_name):
    """Affiche l'explicabilite fusion : contributions texte vs image."""
    st.markdown("---")
    st.markdown("#### Explicabilite Fusion")

    # Qui dit quoi
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### Le Texte dit...")
        if text_top:
            st.metric("Prediction Texte", text_top['name'])
            st.progress(text_top['confidence'])
            st.caption(f"Confiance: {text_top['confidence']:.1%}")
    with c2:
        st.markdown("##### L'Image dit...")
        if image_top:
            st.metric("Prediction Image", image_top['name'])
            st.progress(image_top['confidence'])
            st.caption(f"Confiance: {image_top['confidence']:.1%}")

    # Formule de fusion
    st.markdown("##### Formule de fusion")
    st.code(
        f"P_fusion = {w_text:.0%} x P_texte + {w_image:.0%} x P_image",
        language=None
    )

    # Qui a ete decisif ?
    if text_top and image_top:
        text_contrib = text_top['confidence'] * w_text
        image_contrib = image_top['confidence'] * w_image

        fig_contrib = go.Figure()
        fig_contrib.add_trace(go.Bar(
            x=[text_contrib], y=["Fusion"], name=f"Texte ({w_text:.0%})",
            orientation='h', marker_color='#1565C0',
            text=[f"Texte: {text_contrib:.1%}"], textposition='inside',
        ))
        fig_contrib.add_trace(go.Bar(
            x=[image_contrib], y=["Fusion"], name=f"Image ({w_image:.0%})",
            orientation='h', marker_color='#BF0000',
            text=[f"Image: {image_contrib:.1%}"], textposition='inside',
        ))
        fig_contrib.update_layout(
            barmode='stack', height=100,
            margin=dict(l=0, r=10, t=10, b=10),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            xaxis=dict(tickformat=".0%", range=[0, 1]),
            yaxis=dict(visible=False),
        )
        st.plotly_chart(fig_contrib, width="stretch")

        if text_contrib > image_contrib:
            st.info(f"**Modalite decisive : Texte** — contribue {text_contrib:.1%} vs Image {image_contrib:.1%}")
        else:
            st.info(f"**Modalite decisive : Image** — contribue {image_contrib:.1%} vs Texte {text_contrib:.1%}")

    # Per-model image breakdown (compact)
    if per_model:
        with st.expander("Detail par modele image", expanded=False):
            show_image_explainability(per_model)

    # Top features texte (compact)
    if text_explain:
        with st.expander("Mots-cles decisifs (TF-IDF)", expanded=False):
            df_exp = pd.DataFrame(text_explain[:5])
            df_exp['feature'] = df_exp['feature'].apply(
                lambda x: x.split('__')[-1] if '__' in x else x)
            fig = px.bar(
                df_exp, x='contribution', y='feature', orientation='h',
                color='contribution',
                color_continuous_scale=['#BF0000', '#FFE5E5', '#C8E6C9', '#2E7D32'],
                color_continuous_midpoint=0,
            )
            fig.update_layout(
                height=200, margin=dict(l=0, r=10, t=10, b=10),
                xaxis_title="Contribution", yaxis_title="",
                coloraxis_showscale=False,
                yaxis=dict(autorange="reversed"),
            )
            st.plotly_chart(fig, width="stretch")


# --- INTERFACE PRINCIPALE ---

tabs = st.tabs(["📝 Analyse Texte", "🖼️ Analyse Image", "🚀 FUSION Multimodale"])

# ==========================================
# ONGLET 1 : TEXTE
# ==========================================
with tabs[0]:
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.subheader("Entrée Texte")
        
        if "demo_text_val" not in st.session_state:
            st.session_state.demo_text_val = ""

        txt_input = st.text_area("Description du produit", height=200,
                                 value=st.session_state.demo_text_val,
                                 placeholder="Ex: Piscine gonflable pour enfants intex...")
        
        # Callback to update session state if user types manually
        if txt_input != st.session_state.demo_text_val:
            st.session_state.demo_text_val = txt_input

        btn_txt = st.button("Analyser le Texte", type="primary")

    with col2:
        st.subheader("Résultats")
        if btn_txt and txt_input:
            with st.spinner("Analyse sémantique en cours..."):
                time.sleep(0.3)
                res = clf.predict_text(txt_input)
                explain_data = clf.explain_text(txt_input)

                show_results(res)

                # Explicabilite inline
                if res:
                    show_text_explainability(txt_input, explain_data, res[0]['name'])

                    # Additional explanation for low confidence
                    confidence = res[0]['confidence']
                    if confidence < 0.6:
                        with st.expander("⚠️ Pourquoi ce score est-il si bas ?", expanded=True):
                            st.warning(f"**Score de confiance faible ({confidence:.1%})**")
                            st.markdown("""
                            Le modèle hésite car :
                            - Le texte est **trop court** ou **ambigu** (ex: "Tableau").
                            - Il contient des **mots-clés contradictoires** (ex: "Piscine" vs "Livre").
                            - La catégorie prédite partage des similitudes avec d'autres (ex: Jeu Vidéo Digital vs Boîte).
                            
                            C'est un comportement **honnête** du modèle : il ne "sur-vend" pas sa certitude quand l'information est insuffisante.
                            """)

# ==========================================
# ONGLET 2 : IMAGE
# ==========================================
with tabs[1]:
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.subheader("Entrée Image")
        img_file = st.file_uploader("Image du produit", type=['jpg', 'png', 'jpeg', 'webp'])

        if img_file:
            st.image(img_file, caption="Aperçu", width="stretch")
            import tempfile
            _tmp_demo = os.path.join(tempfile.gettempdir(), "temp_demo.jpg")
            with open(_tmp_demo, "wb") as f:
                f.write(img_file.getbuffer())

    with col2:
        st.subheader("Analyse Visuelle")
        if img_file:
            if st.button("Lancer le Voting Image", type="primary"):
                with st.spinner("Réunion du Conseil (DINO + EfficientNet + XGBoost)..."):
                    img_detail = clf.predict_image_detailed(_tmp_demo)
                    st.session_state["img_results"] = img_detail

            if "img_results" in st.session_state:
                img_detail = st.session_state["img_results"]
                show_results(img_detail['results'])
                show_image_explainability(img_detail['per_model'])

# ==========================================
# ONGLET 3 : FUSION (SLIDER INTERACTIF)
# ==========================================
with tabs[2]:
    st.markdown("### Cockpit de Fusion")
    st.info("Ajustez le curseur pour voir comment le Texte et l'Image s'influencent mutuellement.")

    # slider interactif poids
    fusion_weight = st.slider("Équilibre de la Décision", 0.0, 1.0, 0.6,
                              format="Image: %d%%")

    # mise a jour dynamique des poids du classifieur
    clf.w_image = fusion_weight
    clf.w_text = 1.0 - fusion_weight

    # affichage visuel des poids
    c_txt, c_mid, c_img = st.columns([1, 6, 1])
    with c_txt: st.write(f"Texte **{int((1-fusion_weight)*100)}%**")
    with c_img: st.write(f"Image **{int(fusion_weight*100)}%**")
    with c_mid: st.progress(fusion_weight)

    st.divider()

    c1, c2 = st.columns(2, gap="large")
    with c1:
        f_txt = st.text_area("1. Description", height=100, key="fusion_txt")
        f_img = st.file_uploader("2. Image", type=['jpg', 'png'], key="fusion_img")

        launch = st.button("Calculer la Fusion", type="primary")

    with c2:
        if launch and f_txt and f_img:
            import tempfile
            _tmp_fusion = os.path.join(tempfile.gettempdir(), "temp_fusion.jpg")
            with open(_tmp_fusion, "wb") as f: f.write(f_img.getbuffer())

            with st.spinner("Fusion pondérée en cours..."):
                detail = clf.predict_fusion_detailed(f_txt, _tmp_fusion)

                show_results(detail['fusion'], title="Résultat Fusionné")

                # Explicabilite fusion inline
                show_fusion_explainability(
                    text_top=detail['text_top'],
                    image_top=detail['image_top'],
                    per_model=detail['per_model'],
                    text_explain=detail['text_explain'],
                    w_text=clf.w_text,
                    w_image=clf.w_image,
                    predicted_name=detail['fusion'][0]['name'] if detail['fusion'] else "",
                )
        elif launch:
            st.warning("Remplissez les deux champs (Texte et Image) !")

# Sidebar
with st.sidebar:
    st.markdown("### Démo Interactive")
    st.divider()
    st.markdown("**Poids de fusion actuels**")
    st.metric("Image", f"{int(clf.w_image*100)}%")
    st.metric("Texte", f"{int(clf.w_text*100)}%")
    st.divider()
    st.markdown("**Modèles chargés**")
    st.success("Voting (3 modèles)" if clf.voting else "Voting indisponible")
    st.success("LinearSVC" if clf.text_model else "Texte indisponible")
