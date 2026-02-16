"""
Central configuration — single source of truth for all paths, weights, and settings.

All other modules import from here. Never hardcode paths or weights elsewhere.
"""
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent.parent  # repo root

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
IMAGES_DIR = RAW_DATA_DIR / "images"

MODELS_DIR = PROJECT_ROOT / "models"
IMAGE_MODEL_PATH = MODELS_DIR / "M1_IMAGE_DeepLearning_DINOv3.pth"
XGB_MODEL_PATH = MODELS_DIR / "M2_IMAGE_Classic_XGBoost.json"
XGB_ENC_PATH = MODELS_DIR / "M2_IMAGE_XGBoost_Encoder.pkl"
EFF_MODEL_PATH = MODELS_DIR / "M3_IMAGE_Classic_EfficientNetB0.pth"
TEXT_MODEL_PATH = MODELS_DIR / "text_classifier.joblib"
TFIDF_VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.joblib"
CATEGORY_MAPPING_PATH = MODELS_DIR / "category_mapping.json"

IMPLEMENTATION_DIR = PROJECT_ROOT / "implementation"
FEATURES_DIR = IMPLEMENTATION_DIR / "outputs"
METADATA_PATH = FEATURES_DIR / "metadata_augmented.json"

STREAMLIT_DIR = Path(__file__).parent
ASSETS_DIR = STREAMLIT_DIR / "assets"

# ── Hugging Face Hub ──────────────────────────────────────────────────────────
# Model repository for cloud deployment (auto-download at startup).
# Override via Streamlit secrets [huggingface] repo_id or HF_REPO_ID env var.
HF_REPO_ID = "akiroussama/rakuten-models"

# ── Fusion Weights ─────────────────────────────────────────────────────────────
# Late fusion: weighted average of image and text prediction scores.
# 60% image (voting system) + 40% text (LinearSVC) = F1 ~0.85.
FUSION_W_IMAGE = 0.6
FUSION_W_TEXT = 0.4

# ── App Settings ───────────────────────────────────────────────────────────────
APP_CONFIG = {
    "title": "Rakuten Product Classifier",
    "icon": "\U0001F6D2",      # shopping cart emoji
    "layout": "wide",
    "initial_sidebar_state": "expanded",
}

MODEL_CONFIG = {
    "use_mock": False,                              # True = demo mode (no real models needed)
    "fusion_weights": (FUSION_W_IMAGE, FUSION_W_TEXT),
    "top_k": 5,                                     # number of top predictions to return
    "confidence_threshold": 0.1,                     # minimum confidence to display
}

IMAGE_CONFIG = {
    "target_size": (224, 224),                       # standard CNN input size
    "allowed_formats": ["jpg", "jpeg", "png", "webp"],
    "max_size_mb": 10,
}

# ── UI Theme ───────────────────────────────────────────────────────────────────
THEME = {
    "primary_color": "#BF0000",       # Rakuten red
    "secondary_color": "#FFFFFF",
    "background_color": "#F5F5F5",
    "text_color": "#333333",
    "success_color": "#28A745",
    "warning_color": "#FFC107",
    "error_color": "#DC3545",
}
