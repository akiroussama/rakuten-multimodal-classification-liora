"""
Tests d'intégration pour le chargement des pages Streamlit.

Ces tests vérifient que:
- Chaque page se charge sans erreur
- Les imports fonctionnent
- Les composants de base sont présents
"""
import pytest
import sys
from pathlib import Path

# Chemin vers le répertoire Streamlit
STREAMLIT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(STREAMLIT_DIR))
PAGES_DIR = STREAMLIT_DIR / "pages"


def _list_page_files():
    """Retourne les fichiers de pages Streamlit (hors __init__)."""
    return sorted(
        [f for f in PAGES_DIR.glob("*.py") if not f.name.startswith("__")],
        key=lambda p: p.name
    )


# =============================================================================
# TESTS Import des Modules
# =============================================================================
@pytest.mark.integration
class TestModuleImports:
    """Tests d'import des modules."""

    def test_import_config(self):
        """config.py peut être importé."""
        from config import APP_CONFIG, MODEL_CONFIG, THEME, ASSETS_DIR

        assert APP_CONFIG is not None
        assert MODEL_CONFIG is not None
        assert THEME is not None
        assert ASSETS_DIR is not None

    def test_import_category_mapping(self):
        """category_mapping.py peut être importé."""
        from utils.category_mapping import (
            CATEGORY_MAPPING,
            get_category_info,
            get_category_emoji,
            get_all_categories,
        )

        assert len(CATEGORY_MAPPING) == 27

    def test_import_mock_classifier(self):
        """mock_classifier.py peut être importé."""
        from utils.mock_classifier import (
            DemoClassifier,
            TEXT_MODELS,
            IMAGE_MODELS,
            MultiModelClassifier,
        )

        assert len(TEXT_MODELS) == 3
        assert len(IMAGE_MODELS) == 3

    def test_import_preprocessing(self):
        """preprocessing.py peut être importé."""
        from utils.preprocessing import (
            preprocess_product_text,
            validate_text_input,
        )

        assert callable(preprocess_product_text)
        assert callable(validate_text_input)

    def test_import_data_loader(self):
        """data_loader.py peut être importé."""
        from utils.data_loader import (
            is_data_available,
            get_dataset_summary,
            get_category_distribution,
        )

        assert callable(is_data_available)
        assert callable(get_dataset_summary)

    def test_import_image_utils(self):
        """image_utils.py peut être importé."""
        from utils.image_utils import (
            load_image_from_upload,
            validate_image,
            get_image_info,
        )

        assert callable(load_image_from_upload)
        assert callable(validate_image)

    def test_import_ui_utils(self):
        """ui_utils.py peut être importé."""
        from utils.ui_utils import load_css

        assert callable(load_css)


# =============================================================================
# TESTS Syntax des Pages (sans Streamlit runtime)
# =============================================================================
@pytest.mark.integration
class TestPageSyntax:
    """Tests de syntaxe des fichiers de page."""

    PAGE_FILES = ["app.py"] + [f"pages/{p.name}" for p in _list_page_files()]

    @pytest.mark.parametrize("page_file", PAGE_FILES)
    def test_page_syntax_valid(self, page_file):
        """Chaque fichier de page a une syntaxe Python valide."""
        file_path = STREAMLIT_DIR / page_file

        assert file_path.exists(), f"Page file not found: {page_file}"

        # Compile le fichier pour vérifier la syntaxe
        with open(file_path, 'r', encoding='utf-8') as f:
            source = f.read()

        try:
            compile(source, file_path, 'exec')
        except SyntaxError as e:
            pytest.fail(f"Syntax error in {page_file}: {e}")

    @pytest.mark.parametrize("page_file", PAGE_FILES)
    def test_page_has_docstring(self, page_file):
        """Chaque page a une docstring."""
        file_path = STREAMLIT_DIR / page_file

        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Vérifie la présence d'une docstring au début
        content_stripped = content.lstrip()
        has_docstring = (
            content_stripped.startswith('"""') or
            content_stripped.startswith("'''") or
            content_stripped.startswith('#')
        )

        assert has_docstring, f"{page_file} should have a docstring or comment"


# =============================================================================
# TESTS Configuration App
# =============================================================================
@pytest.mark.integration
class TestAppConfiguration:
    """Tests de la configuration de l'application."""

    def test_app_config_has_required_keys(self):
        """APP_CONFIG contient les clés requises."""
        from config import APP_CONFIG

        required_keys = ["title", "icon", "layout"]
        for key in required_keys:
            assert key in APP_CONFIG, f"Missing key in APP_CONFIG: {key}"

    def test_model_config_has_required_keys(self):
        """MODEL_CONFIG contient les clés requises."""
        from config import MODEL_CONFIG

        required_keys = ["use_mock"]
        for key in required_keys:
            assert key in MODEL_CONFIG, f"Missing key in MODEL_CONFIG: {key}"

    def test_theme_has_colors(self):
        """THEME contient les couleurs."""
        from config import THEME

        assert "primary" in THEME or "color" in str(THEME).lower()

    def test_assets_dir_exists(self):
        """Le répertoire assets existe."""
        from config import ASSETS_DIR

        assert ASSETS_DIR.exists(), f"Assets directory not found: {ASSETS_DIR}"

    def test_style_css_exists(self):
        """Le fichier style.css existe."""
        from config import ASSETS_DIR

        css_file = ASSETS_DIR / "style.css"
        assert css_file.exists(), f"Style file not found: {css_file}"


# =============================================================================
# TESTS Structure des Pages
# =============================================================================
@pytest.mark.integration
class TestPageStructure:
    """Tests de la structure des pages."""

    def test_pages_directory_exists(self):
        """Le répertoire pages/ existe."""
        pages_dir = STREAMLIT_DIR / "pages"
        assert pages_dir.exists()
        assert pages_dir.is_dir()

    def test_has_expected_number_of_pages(self):
        """Le répertoire contient toutes les pages prévues de la soutenance."""
        page_files = _list_page_files()
        assert len(page_files) >= 8, f"Expected at least 8 pages, found {len(page_files)}"

    def test_pages_have_correct_numbering(self):
        """Les pages sont numérotées de manière séquentielle (1..N)."""
        page_files = _list_page_files()

        numbers_found = set()
        for page in page_files:
            # Extrait le premier caractère qui devrait être le numéro
            first_char = page.name[0]
            if first_char.isdigit():
                numbers_found.add(int(first_char))

        expected = set(range(1, len(page_files) + 1))
        assert numbers_found == expected, f"Page numbering incorrect: {numbers_found}"

    def test_pages_have_emojis(self):
        """Les noms de pages contiennent des emojis."""
        page_files = _list_page_files()

        for page in page_files:
            # Un emoji a généralement un point de code > 127
            has_emoji = any(ord(c) > 127 for c in page.name)
            assert has_emoji, f"Page {page.name} should have an emoji"


# =============================================================================
# TESTS Imports Circulaires
# =============================================================================
@pytest.mark.integration
class TestCircularImports:
    """Tests pour détecter les imports circulaires."""

    MODULES = [
        "config",
        "utils.category_mapping",
        "utils.mock_classifier",
        "utils.preprocessing",
        "utils.data_loader",
        "utils.image_utils",
        "utils.ui_utils",
    ]

    @pytest.mark.parametrize("module_name", MODULES)
    def test_no_circular_import(self, module_name):
        """Chaque module peut être importé sans import circulaire."""
        try:
            # Fresh import
            if module_name in sys.modules:
                del sys.modules[module_name]

            __import__(module_name)
        except ImportError as e:
            if "circular" in str(e).lower():
                pytest.fail(f"Circular import in {module_name}: {e}")
            # Autres ImportError peuvent être acceptables (dépendances manquantes)


# =============================================================================
# TESTS Data Loader
# =============================================================================
@pytest.mark.integration
class TestDataLoaderIntegration:
    """Tests d'intégration du data loader."""

    def test_is_data_available_returns_bool(self):
        """is_data_available retourne un booléen."""
        from utils.data_loader import is_data_available

        result = is_data_available()
        assert isinstance(result, bool)

    def test_get_dataset_summary_returns_dict(self):
        """get_dataset_summary retourne un dictionnaire."""
        from utils.data_loader import get_dataset_summary

        result = get_dataset_summary()
        assert isinstance(result, dict)

    def test_dataset_summary_has_required_keys(self):
        """Le résumé du dataset a les clés requises."""
        from utils.data_loader import get_dataset_summary

        result = get_dataset_summary()
        required_keys = ["train_samples", "num_categories"]

        for key in required_keys:
            assert key in result, f"Missing key: {key}"

    def test_get_category_distribution_returns_dataframe(self):
        """get_category_distribution retourne un DataFrame."""
        from utils.data_loader import get_category_distribution
        import pandas as pd

        result = get_category_distribution()
        assert isinstance(result, pd.DataFrame)

    def test_category_distribution_has_27_rows(self):
        """La distribution a 27 lignes (une par catégorie)."""
        from utils.data_loader import get_category_distribution

        result = get_category_distribution()
        assert len(result) == 27


# =============================================================================
# TESTS Classifier Integration
# =============================================================================
@pytest.mark.integration
class TestClassifierIntegration:
    """Tests d'intégration du classifier."""

    def test_classifier_initialization_chain(self):
        """La chaîne d'initialisation fonctionne."""
        from utils.mock_classifier import (
            DemoClassifier,
            TEXT_MODELS,
            IMAGE_MODELS,
        )

        # Test avec config par défaut
        clf_default = DemoClassifier()
        assert clf_default is not None

        # Test avec config texte
        for key, config in TEXT_MODELS.items():
            clf = DemoClassifier(model_config=config)
            assert clf is not None

        # Test avec config image
        for key, config in IMAGE_MODELS.items():
            clf = DemoClassifier(model_config=config)
            assert clf is not None

    def test_full_classification_pipeline(self):
        """Le pipeline complet de classification fonctionne."""
        from utils.mock_classifier import DemoClassifier, TEXT_MODELS
        from utils.preprocessing import preprocess_product_text
        from utils.category_mapping import get_category_info

        # 1. Preprocessing
        raw_designation = "<p>iPhone 15 Pro</p>"
        raw_description = "Smartphone Apple"
        processed_text = preprocess_product_text(raw_designation, raw_description)

        # 2. Classification
        clf = DemoClassifier(model_config=TEXT_MODELS["camembert"])
        result = clf.predict(text=processed_text)

        # 3. Post-processing (get category info)
        name, full_name, emoji = get_category_info(result.category)

        # Vérifications
        assert result.category is not None
        assert 0 <= result.confidence <= 1
        assert len(name) > 0
        assert len(emoji) > 0


# =============================================================================
# TESTS Assets
# =============================================================================
@pytest.mark.integration
class TestAssets:
    """Tests des ressources statiques."""

    def test_css_file_valid(self):
        """Le fichier CSS est valide (non vide)."""
        from config import ASSETS_DIR

        css_file = ASSETS_DIR / "style.css"
        assert css_file.exists()

        content = css_file.read_text()
        assert len(content) > 100  # Au moins 100 caractères de CSS

    def test_css_has_rakuten_colors(self):
        """Le CSS utilise les couleurs Rakuten."""
        from config import ASSETS_DIR

        css_file = ASSETS_DIR / "style.css"
        content = css_file.read_text().lower()

        # Couleur principale Rakuten (rouge)
        has_rakuten_red = (
            "#bf0000" in content or
            "#BF0000" in content.upper() or
            "191,0,0" in content or
            "rgb(191" in content
        )

        assert has_rakuten_red, "CSS should include Rakuten red color"
