"""
Model downloader — fetches trained weights from Hugging Face Hub.

On first run, downloads all required model files (~2.5 GB) to the local
models/ directory. Subsequent runs skip files that already exist locally.

Configuration (pick one):
  1. Streamlit secrets:  .streamlit/secrets.toml  [huggingface] repo_id = "..."
  2. Environment var:    HF_REPO_ID=your-org/your-repo
  3. config.py:          HF_REPO_ID = "your-org/your-repo"

For private repos, also set huggingface.token in secrets or HF_TOKEN env var.
"""
import os
import streamlit as st
from pathlib import Path

# Files required by the app (filename → human-readable size for progress UI)
REQUIRED_MODELS = {
    "category_mapping.json": "4 KB",
    "M1_IMAGE_DeepLearning_DINOv3.pth": "1.2 GB",
    "M2_IMAGE_XGBoost_Encoder.pkl": "<1 KB",
    "M3_IMAGE_Classic_EfficientNetB0.pth": "16 MB",
    "text_classifier.joblib": "32 MB",
}

# Optional models — downloaded if available, but app works without them
OPTIONAL_MODELS = {
    "M2_IMAGE_Classic_XGBoost.json": "~50 MB",
}


def _get_hf_repo_id():
    """Resolve Hugging Face repo ID from secrets, env, or config."""
    # 1. Streamlit secrets (recommended for cloud deployment)
    try:
        return st.secrets["huggingface"]["repo_id"]
    except (KeyError, FileNotFoundError):
        pass

    # 2. Environment variable
    repo = os.environ.get("HF_REPO_ID")
    if repo:
        return repo

    # 3. config.py fallback
    try:
        from config import HF_REPO_ID
        return HF_REPO_ID
    except (ImportError, AttributeError):
        return None


def _get_hf_token():
    """Resolve Hugging Face token (only needed for private repos)."""
    try:
        return st.secrets["huggingface"]["token"]
    except (KeyError, FileNotFoundError):
        pass
    return os.environ.get("HF_TOKEN")


def ensure_models(models_dir):
    """
    Download missing model files from Hugging Face Hub with progress UI.

    Args:
        models_dir: Path to the local models/ directory.

    Returns:
        True if all models are available, False if any are missing.
    """
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    # Check which files need downloading
    missing = [f for f in REQUIRED_MODELS if not (models_dir / f).exists()]

    if not missing:
        return True  # All models already present

    # Resolve HF configuration
    repo_id = _get_hf_repo_id()
    if not repo_id:
        st.warning(
            "Models not found locally and no Hugging Face repo configured. "
            "Set `HF_REPO_ID` in secrets or environment. "
            "The app will run in limited mode (text predictions may still work)."
        )
        return False

    # Import huggingface_hub (fail gracefully)
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        st.error(
            "Package `huggingface_hub` not installed. "
            "Run: `pip install huggingface_hub`"
        )
        return False

    token = _get_hf_token()

    # Download with progress bar
    progress = st.progress(0, text="Downloading models from Hugging Face Hub...")

    for i, filename in enumerate(missing):
        size_label = REQUIRED_MODELS[filename]
        progress.progress(
            i / len(missing),
            text=f"Downloading {filename} ({size_label})..."
        )
        try:
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=str(models_dir),
                local_dir_use_symlinks=False,
                token=token,
            )
        except Exception as e:
            progress.empty()
            st.error(f"Download failed for {filename}: {e}")
            return False

    progress.progress(1.0, text="All models downloaded successfully!")

    # Try optional models (non-blocking)
    optional_missing = [f for f in OPTIONAL_MODELS if not (models_dir / f).exists()]
    for filename in optional_missing:
        try:
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=str(models_dir),
                local_dir_use_symlinks=False,
                token=token,
            )
        except Exception:
            pass  # Optional model not available — app works without it

    return True
