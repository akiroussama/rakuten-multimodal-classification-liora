"""
Utilitaires UI pour l'application Streamlit Rakuten.
"""
import streamlit as st
from pathlib import Path

def load_css(css_file_path: Path):
    """Charge un fichier CSS et l'injecte dans la page Streamlit."""
    try:
        with open(css_file_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.error(f"Fichier CSS non trouvé : {css_file_path}")
