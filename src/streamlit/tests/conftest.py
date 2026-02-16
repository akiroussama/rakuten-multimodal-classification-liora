"""
Fixtures partagées pour tous les tests Rakuten Product Classifier.

Ce fichier contient les fixtures pytest réutilisables à travers
tous les modules de test.
"""
import pytest
import sys
from pathlib import Path
from PIL import Image
import pandas as pd
import numpy as np
import io
import json

# =============================================================================
# CONFIGURATION PATH
# =============================================================================
# Ajouter le répertoire source au path pour les imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.mock_classifier import (
    DemoClassifier,
    MultiModelClassifier,
    TEXT_MODELS,
    IMAGE_MODELS,
    ModelConfig,
    ClassificationResult,
)
from utils.category_mapping import CATEGORY_MAPPING, get_category_info


# =============================================================================
# FIXTURES CLASSIFIER
# =============================================================================
@pytest.fixture
def demo_classifier():
    """Retourne un DemoClassifier initialisé avec config par défaut."""
    return DemoClassifier()


@pytest.fixture
def text_classifier():
    """Classifier configuré spécifiquement pour le texte (CamemBERT)."""
    config = TEXT_MODELS["camembert"]
    return DemoClassifier(model_config=config)


@pytest.fixture
def image_classifier():
    """Classifier configuré spécifiquement pour les images (ResNet50+SVM)."""
    config = IMAGE_MODELS["resnet50_svm"]
    return DemoClassifier(model_config=config)


@pytest.fixture
def multi_model_classifier():
    """MultiModelClassifier avec tous les modèles."""
    return MultiModelClassifier()


@pytest.fixture(params=list(TEXT_MODELS.keys()))
def all_text_models(request):
    """Paramétrise tous les modèles texte disponibles."""
    model_key = request.param
    config = TEXT_MODELS[model_key]
    return model_key, DemoClassifier(model_config=config)


@pytest.fixture(params=list(IMAGE_MODELS.keys()))
def all_image_models(request):
    """Paramétrise tous les modèles image disponibles."""
    model_key = request.param
    config = IMAGE_MODELS[model_key]
    return model_key, DemoClassifier(model_config=config)


# =============================================================================
# FIXTURES DONNÉES TEXTE
# =============================================================================
@pytest.fixture
def sample_text():
    """Texte produit exemple simple."""
    return "Console PlayStation 5 nouvelle génération Sony"


@pytest.fixture
def sample_designation():
    """Désignation produit exemple."""
    return "iPhone 15 Pro Max 256GB"


@pytest.fixture
def sample_description():
    """Description produit exemple."""
    return "Smartphone Apple dernière génération, écran OLED, 5G, appareil photo 48MP"


@pytest.fixture
def sample_product_texts():
    """Liste de textes produits pour tests batch."""
    return [
        ("Harry Potter à l'école des sorciers", "Roman fantastique J.K. Rowling"),
        ("Console PlayStation 5", "Console de jeux vidéo nouvelle génération"),
        ("Piscine gonflable ronde 3m", "Piscine autoportante pour jardin été"),
        ("Figurine Pop Marvel Spider-Man", "Figurine de collection Funko Pop vinyle"),
        ("Tondeuse à gazon électrique Bosch", "Tondeuse électrique 1400W coupe 37cm"),
    ]


# =============================================================================
# FIXTURES DONNÉES IMAGE
# =============================================================================
@pytest.fixture
def sample_image():
    """Image produit exemple (224x224 RGB)."""
    img = Image.new('RGB', (224, 224), color='red')
    return img


@pytest.fixture
def sample_image_small():
    """Petite image (50x50)."""
    return Image.new('RGB', (50, 50), color='blue')


@pytest.fixture
def sample_image_large():
    """Grande image (1000x1000)."""
    return Image.new('RGB', (1000, 1000), color='green')


@pytest.fixture
def sample_image_bytes():
    """Image en bytes (simule un upload Streamlit)."""
    img = Image.new('RGB', (224, 224), color='blue')
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG')
    buffer.seek(0)
    return buffer


@pytest.fixture
def sample_image_png_bytes():
    """Image PNG en bytes."""
    img = Image.new('RGBA', (224, 224), color='green')
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    buffer.seek(0)
    return buffer


@pytest.fixture
def corrupted_image_bytes():
    """Bytes invalides (pas une vraie image)."""
    return io.BytesIO(b"this is not a valid image content at all")


@pytest.fixture
def oversized_image():
    """Image très grande (4000x4000) pour test de redimensionnement."""
    return Image.new('RGB', (4000, 4000), color='purple')


# =============================================================================
# FIXTURES CATÉGORIES
# =============================================================================
@pytest.fixture
def all_category_codes():
    """Liste de tous les codes catégorie (27)."""
    return list(CATEGORY_MAPPING.keys())


@pytest.fixture
def sample_category_code():
    """Un code catégorie valide."""
    return "2583"  # Téléphones


@pytest.fixture
def invalid_category_code():
    """Un code catégorie invalide."""
    return "9999"


# =============================================================================
# FIXTURES EDGE CASES
# =============================================================================
@pytest.fixture(params=[
    "",                                    # Empty string
    " " * 100,                            # Whitespace only
    "a",                                   # Single character
    "a" * 10000,                          # Very long text
    "<script>alert('xss')</script>",      # XSS payload
    "<p>Hello <b>World</b></p>",          # HTML tags
    "🎮📱💻🖥️🎧",                          # Emojis only
    "价格便宜质量好",                        # Chinese characters
    "Привет мир",                          # Russian characters
    "Test\nwith\nnewlines",               # Newlines
    "Test\twith\ttabs",                   # Tabs
    "  leading and trailing spaces  ",    # Extra spaces
])
def edge_case_text(request):
    """Paramétrise les cas limites de texte pour tests de robustesse."""
    return request.param


@pytest.fixture(params=[
    (10, 10),      # Very small
    (1, 1000),     # Very thin
    (1000, 1),     # Very tall
])
def edge_case_image_size(request):
    """Tailles d'image limites."""
    return request.param


# =============================================================================
# FIXTURES ML METRICS
# =============================================================================
@pytest.fixture
def model_baseline():
    """Métriques baseline pour tests de non-régression."""
    return {
        "accuracy": 0.75,
        "f1_macro": 0.70,
        "f1_weighted": 0.75,
        "precision_macro": 0.70,
        "recall_macro": 0.65,
        "inference_time_ms": 100,
    }


@pytest.fixture
def golden_predictions():
    """Prédictions de référence pour tests de déterminisme."""
    return {
        "Console PlayStation 5": "2462",
        "iPhone 15 Pro Max": "2583",
        "Harry Potter livre": "2403",
        "Piscine gonflable": "2582",
        "Figurine Pop Marvel": "1281",
    }


# =============================================================================
# FIXTURES DONNÉES TRAINING (si disponibles)
# =============================================================================
@pytest.fixture(scope="session")
def data_paths():
    """Chemins vers les données."""
    base = PROJECT_ROOT.parent.parent / "data"
    return {
        "X_train": base / "X_train_update.csv",
        "Y_train": base / "Y_train_CVw08PX.csv",
    }


@pytest.fixture(scope="session")
def training_data_available(data_paths):
    """Vérifie si les données d'entraînement sont disponibles."""
    return all(Path(p).exists() for p in data_paths.values())


@pytest.fixture(scope="session")
def training_data(data_paths, training_data_available):
    """Charge les données d'entraînement (si disponibles)."""
    if not training_data_available:
        pytest.skip("Training data not available")

    X_train = pd.read_csv(data_paths["X_train"])
    Y_train = pd.read_csv(data_paths["Y_train"])
    return X_train, Y_train


# =============================================================================
# FIXTURES STREAMLIT
# =============================================================================
@pytest.fixture
def mock_session_state():
    """Mock du session_state Streamlit."""
    class MockSessionState(dict):
        def __getattr__(self, key):
            return self.get(key)

        def __setattr__(self, key, value):
            self[key] = value

    return MockSessionState()


@pytest.fixture
def streamlit_app_test():
    """Factory pour créer des AppTest Streamlit."""
    try:
        from streamlit.testing.v1 import AppTest
        return AppTest
    except ImportError:
        pytest.skip("streamlit.testing not available")


# =============================================================================
# MARKERS PERSONNALISÉS
# =============================================================================
def pytest_configure(config):
    """Configuration pytest au démarrage."""
    config.addinivalue_line("markers", "unit: Unit tests (fast, isolated)")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "ml: Machine Learning specific tests")
    config.addinivalue_line("markers", "e2e: End-to-end browser tests")
    config.addinivalue_line("markers", "slow: Slow tests (>5 seconds)")
    config.addinivalue_line("markers", "security: Security tests")
    config.addinivalue_line("markers", "regression: Regression tests")
    config.addinivalue_line("markers", "smoke: Smoke tests (critical path)")


# =============================================================================
# HELPERS
# =============================================================================
@pytest.fixture
def assert_valid_prediction():
    """Helper pour valider une prédiction."""
    def _assert(result: ClassificationResult):
        assert result is not None, "Result should not be None"
        assert isinstance(result, ClassificationResult), "Should be ClassificationResult"
        assert result.category in CATEGORY_MAPPING, f"Invalid category: {result.category}"
        assert 0.0 <= result.confidence <= 1.0, f"Confidence out of range: {result.confidence}"
        assert len(result.top_k_predictions) > 0, "Should have top_k predictions"
        assert result.source is not None, "Source should not be None"

        # Vérifier que top_k est trié par score décroissant
        scores = [score for _, score in result.top_k_predictions]
        assert scores == sorted(scores, reverse=True), "Top-k should be sorted by score"

        # Vérifier que la somme des probabilités <= 1
        total_prob = sum(scores)
        assert total_prob <= 1.01, f"Probabilities sum > 1: {total_prob}"

    return _assert


@pytest.fixture
def measure_time():
    """Helper pour mesurer le temps d'exécution."""
    import time

    class Timer:
        def __init__(self):
            self.elapsed = 0

        def __enter__(self):
            self.start = time.perf_counter()
            return self

        def __exit__(self, *args):
            self.elapsed = time.perf_counter() - self.start

    return Timer
