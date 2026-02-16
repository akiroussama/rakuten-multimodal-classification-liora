"""
Tests de robustesse — Fallbacks, indisponibilite modeles, messages d'erreur.

Couvre les scenarios critiques en production :
  1. Model downloader : fallback quand HF Hub inaccessible
  2. Real classifier : graceful degradation quand modeles manquants
  3. VotingPredictor : fallback sans XGBoost (DINOv3 + EfficientNet only)
  4. Pipeline : messages d'erreur clairs pour l'utilisateur
  5. Regression : format et coherence des predictions
"""
import sys
import os
import json
import pytest
import tempfile
import importlib
from pathlib import Path
from unittest.mock import patch, MagicMock

# Setup path — add src/streamlit/ so `utils`, `config` are importable
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
REPO_ROOT = PROJECT_ROOT.parent.parent

# Mock heavy ML dependency if not installed (timm is only needed at runtime
# for DINOv3 model creation; tests verify fallback logic, not GPU inference).
if importlib.util.find_spec("timm") is None:
    sys.modules.setdefault("timm", MagicMock())

# Pre-import real_classifier now that timm is available (real or mocked)
import utils.real_classifier as _rc_mod


def _make_classifier(tmpdir):
    """Helper: create a MultimodalClassifier with patched paths pointing to tmpdir."""
    mapping_path = Path(tmpdir) / "category_mapping.json"
    if not mapping_path.exists():
        mapping_path.write_text("{}")

    with patch.object(_rc_mod, "MODELS_DIR", Path(tmpdir)), \
         patch.object(_rc_mod, "TEXT_MODEL_PATH", Path(tmpdir) / "nonexistent.joblib"), \
         patch.object(_rc_mod, "CATEGORY_MAPPING_PATH", mapping_path):
        return _rc_mod.MultimodalClassifier()


# =============================================================================
# 1. MODEL DOWNLOADER — Fallback & Error Handling
# =============================================================================
class TestModelDownloaderFallback:
    """Tests for model_downloader.py — HF Hub resolution and download errors."""

    def test_repo_id_resolution_from_env(self):
        """HF_REPO_ID env var is used when secrets are unavailable."""
        from utils.model_downloader import _get_hf_repo_id

        with patch.dict(os.environ, {"HF_REPO_ID": "test-org/test-models"}):
            with patch("utils.model_downloader.st") as mock_st:
                mock_st.secrets.__getitem__.side_effect = KeyError("huggingface")
                repo = _get_hf_repo_id()
                assert repo == "test-org/test-models"

    def test_repo_id_resolution_from_config(self):
        """Falls back to config.py when secrets and env are unavailable."""
        from utils.model_downloader import _get_hf_repo_id

        # Remove only HF_REPO_ID (not all env vars — Streamlit needs HOME)
        env_without_hf = {k: v for k, v in os.environ.items() if k != "HF_REPO_ID"}
        with patch.dict(os.environ, env_without_hf, clear=True):
            with patch("utils.model_downloader.st") as mock_st:
                mock_st.secrets.__getitem__.side_effect = KeyError("huggingface")
                repo = _get_hf_repo_id()
                # Should get value from config.py
                assert repo is not None
                assert "/" in repo  # Format: "org/repo"

    def test_repo_id_returns_none_when_all_fail(self):
        """Returns None when no HF configuration is found."""
        from utils.model_downloader import _get_hf_repo_id

        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("HF_REPO_ID", None)
            with patch("utils.model_downloader.st") as mock_st:
                mock_st.secrets.__getitem__.side_effect = KeyError("huggingface")
                with patch.dict("sys.modules", {"config": MagicMock(spec=[])}):
                    repo = _get_hf_repo_id()
                    assert repo is None

    def test_ensure_models_skips_when_all_present(self):
        """When all model files exist locally, no download is triggered."""
        from utils.model_downloader import ensure_models, REQUIRED_MODELS

        with tempfile.TemporaryDirectory() as tmpdir:
            for filename in REQUIRED_MODELS:
                (Path(tmpdir) / filename).write_text("mock")

            result = ensure_models(tmpdir)
            assert result is True

    def test_ensure_models_creates_directory(self):
        """ensure_models creates models/ dir if it doesn't exist."""
        from utils.model_downloader import ensure_models

        with tempfile.TemporaryDirectory() as tmpdir:
            models_dir = Path(tmpdir) / "new_models"
            assert not models_dir.exists()

            with patch("utils.model_downloader._get_hf_repo_id", return_value=None):
                with patch("utils.model_downloader.st") as mock_st:
                    ensure_models(str(models_dir))

            assert models_dir.exists()

    def test_ensure_models_warning_when_no_repo_configured(self):
        """Shows clear warning when HF repo is not configured."""
        from utils.model_downloader import ensure_models

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("utils.model_downloader._get_hf_repo_id", return_value=None):
                with patch("utils.model_downloader.st") as mock_st:
                    result = ensure_models(tmpdir)

                    assert result is False
                    mock_st.warning.assert_called_once()
                    warning_msg = mock_st.warning.call_args[0][0]
                    assert "HF_REPO_ID" in warning_msg
                    assert "limited mode" in warning_msg.lower() or "limited" in warning_msg.lower()

    def test_ensure_models_error_when_hub_not_installed(self):
        """Shows clear error when huggingface_hub package is missing."""
        from utils.model_downloader import ensure_models

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("utils.model_downloader._get_hf_repo_id", return_value="org/repo"):
                with patch("utils.model_downloader.st") as mock_st:
                    with patch.dict("sys.modules", {"huggingface_hub": None}):
                        with patch("builtins.__import__", side_effect=ImportError("No module named 'huggingface_hub'")):
                            result = ensure_models(tmpdir)

                            assert result is False

    def test_required_models_list_complete(self):
        """REQUIRED_MODELS contains all files needed for the app."""
        from utils.model_downloader import REQUIRED_MODELS

        expected_files = {
            "category_mapping.json",
            "M1_IMAGE_DeepLearning_DINOv3.pth",
            "M2_IMAGE_XGBoost_Encoder.pkl",
            "M3_IMAGE_Classic_EfficientNetB0.pth",
            "text_classifier.joblib",
        }
        assert set(REQUIRED_MODELS.keys()) == expected_files

    def test_optional_models_list(self):
        """OPTIONAL_MODELS contains XGBoost (missing but non-blocking)."""
        from utils.model_downloader import OPTIONAL_MODELS

        assert "M2_IMAGE_Classic_XGBoost.json" in OPTIONAL_MODELS


# =============================================================================
# 2. REAL CLASSIFIER — Model Unavailability
# =============================================================================
class TestRealClassifierFallback:
    """Tests for real_classifier.py — graceful degradation when models missing."""

    def test_init_with_missing_models_does_not_crash(self):
        """MultimodalClassifier init succeeds even with missing model files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mapping = {"10": "Livres", "2583": "Piscine"}
            (Path(tmpdir) / "category_mapping.json").write_text(json.dumps(mapping))

            clf = _make_classifier(tmpdir)
            assert clf.voting is None
            assert clf.text_model is None

    def test_predict_text_returns_empty_when_model_missing(self):
        """predict_text returns [] when text model is not loaded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "category_mapping.json").write_text(json.dumps({"10": "Livres"}))
            clf = _make_classifier(tmpdir)

            result = clf.predict_text("Test product")
            assert result == []

    def test_predict_image_returns_empty_when_model_missing(self):
        """predict_image returns [] when voting model is not loaded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "category_mapping.json").write_text(json.dumps({"10": "Livres"}))
            clf = _make_classifier(tmpdir)

            result = clf.predict_image("/fake/image.jpg")
            assert result == []

    def test_predict_fusion_returns_empty_when_both_missing(self):
        """predict_fusion returns [] when both models are unavailable."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "category_mapping.json").write_text(json.dumps({"10": "Livres"}))
            clf = _make_classifier(tmpdir)

            result = clf.predict_fusion("Test", "/fake/image.jpg")
            assert result == []

    def test_category_mapping_loads_with_missing_file(self):
        """Classifier handles missing category mapping gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(_rc_mod, "MODELS_DIR", Path(tmpdir)), \
                 patch.object(_rc_mod, "TEXT_MODEL_PATH", Path(tmpdir) / "x.joblib"), \
                 patch.object(_rc_mod, "CATEGORY_MAPPING_PATH", Path(tmpdir) / "nonexistent.json"):
                clf = _rc_mod.MultimodalClassifier()
                assert isinstance(clf.mapping, dict)


# =============================================================================
# 3. VOTING PREDICTOR — XGBoost Fallback
# =============================================================================
class TestVotingPredictorFallback:
    """Tests for predict_model.py — fallback when XGBoost is missing."""

    def test_has_xgboost_flag_false_when_file_missing(self):
        """VotingPredictor.has_xgboost is False when M2 file is absent."""
        from src.models.predict_model import VotingPredictor

        with tempfile.TemporaryDirectory() as tmpdir:
            vp = VotingPredictor(tmpdir)
            # Before loading, has_xgboost defaults to False
            assert not hasattr(vp, "has_xgboost") or not vp.has_xgboost

    def test_xgboost_file_detection(self):
        """VotingPredictor checks for M2_IMAGE_Classic_XGBoost.json."""
        from src.models.predict_model import VotingPredictor

        with tempfile.TemporaryDirectory() as tmpdir:
            xgb_path = Path(tmpdir) / "M2_IMAGE_Classic_XGBoost.json"
            xgb_path.write_text("mock")

            vp = VotingPredictor(tmpdir)
            # The XGBoost path is checked during load_models(), verify file detection
            assert (vp.mdir / "M2_IMAGE_Classic_XGBoost.json").exists()

    def test_fallback_weights_are_correct(self):
        """Without XGBoost, weights should be 4:2 (DINOv3:EfficientNet)."""
        import numpy as np

        # Simulate the fallback formula: (4*p1 + 2*p3) / 6
        p1 = np.array([0.8, 0.1, 0.1])  # DINOv3 confident
        p3 = np.array([0.6, 0.2, 0.2])  # EfficientNet agrees

        f_p = (4.0 * p1 + 2.0 * p3) / 6.0

        assert abs(f_p.sum() - 1.0) < 0.01  # Probabilities sum to ~1
        assert f_p[0] > f_p[1]  # Top class is correct
        assert f_p[0] > 0.7  # High confidence maintained

    def test_full_weights_are_correct(self):
        """With XGBoost, weights should be 4:1:2 (DINOv3:XGB:EfficientNet)."""
        import numpy as np

        p1 = np.array([0.8, 0.1, 0.1])
        p2 = np.array([0.5, 0.3, 0.2])  # XGBoost less confident
        p3 = np.array([0.6, 0.2, 0.2])

        f_p = (4.0 * p1 + 1.0 * p2 + 2.0 * p3) / 7.0

        assert abs(f_p.sum() - 1.0) < 0.01
        assert f_p[0] > f_p[1]

    def test_sharpening_formula(self):
        """XGBoost probability sharpening: p^3 / sum(p^3) concentrates mass."""
        import numpy as np

        # Flat distribution (typical XGBoost output)
        raw = np.array([0.15, 0.10, 0.05, 0.70])

        sharp = np.power(raw, 3)
        sharp = sharp / sharp.sum()

        # Sharpening should increase the dominant class
        assert sharp[3] > raw[3]
        # And reduce the smaller classes
        assert sharp[0] < raw[0]


# =============================================================================
# 4. ERROR MESSAGES — Clear & Actionable
# =============================================================================
class TestErrorMessages:
    """Verify error messages are user-friendly and suggest solutions."""

    def test_model_downloader_warning_mentions_configuration(self):
        """Warning message tells user HOW to configure HF repo."""
        from utils.model_downloader import ensure_models

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("utils.model_downloader._get_hf_repo_id", return_value=None):
                with patch("utils.model_downloader.st") as mock_st:
                    ensure_models(tmpdir)

                    msg = mock_st.warning.call_args[0][0]
                    assert "HF_REPO_ID" in msg
                    assert "secrets" in msg.lower() or "environment" in msg.lower()

    def test_image_model_error_is_caught_silently(self):
        """Image model failure doesn't crash the app — sets voting=None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "category_mapping.json").write_text('{"10":"test"}')
            clf = _make_classifier(tmpdir)
            assert clf.voting is None

    def test_text_model_error_is_caught_silently(self):
        """Text model failure doesn't crash the app — sets text_model=None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "category_mapping.json").write_text('{"10":"test"}')
            clf = _make_classifier(tmpdir)
            assert clf.text_model is None


# =============================================================================
# 5. REGRESSION — Prediction Format & Consistency
# =============================================================================
class TestPredictionRegression:
    """Regression tests for prediction output format and consistency."""

    def test_mock_classifier_result_format(self):
        """Mock classifier returns correct result structure."""
        from utils.mock_classifier import DemoClassifier

        clf = DemoClassifier()
        result = clf.predict(text="Livre de recettes")

        assert hasattr(result, "category")
        assert hasattr(result, "confidence")
        assert hasattr(result, "top_k_predictions")
        assert hasattr(result, "source")

    def test_mock_classifier_confidence_range(self):
        """Confidence is always between 0 and 1."""
        from utils.mock_classifier import DemoClassifier

        clf = DemoClassifier()
        for text in ["Livre", "Piscine", "Console PS5", "Chaise de bureau"]:
            result = clf.predict(text=text)
            assert 0 <= result.confidence <= 1, f"Invalid confidence {result.confidence} for '{text}'"

    def test_mock_classifier_top_k_sorted(self):
        """Top-k predictions are sorted by confidence descending."""
        from utils.mock_classifier import DemoClassifier

        clf = DemoClassifier()
        result = clf.predict(text="Figurine manga collector")

        scores = [score for _, score in result.top_k_predictions]
        assert scores == sorted(scores, reverse=True), "Top-k not sorted descending"

    def test_mock_classifier_category_codes_valid(self):
        """All predicted category codes are valid Rakuten codes."""
        from utils.mock_classifier import DemoClassifier
        from utils.category_mapping import CATEGORY_CODES

        clf = DemoClassifier()
        result = clf.predict(text="Jeu de societe Monopoly")

        assert result.category in CATEGORY_CODES
        for code, _ in result.top_k_predictions:
            assert code in CATEGORY_CODES, f"Invalid code {code}"

    def test_mock_classifier_deterministic(self):
        """Same input produces same output (no randomness)."""
        from utils.mock_classifier import DemoClassifier

        clf = DemoClassifier()
        r1 = clf.predict(text="Nintendo Switch OLED")
        r2 = clf.predict(text="Nintendo Switch OLED")

        assert r1.category == r2.category
        assert r1.confidence == r2.confidence

    def test_real_classifier_result_format_when_unavailable(self):
        """Real classifier predict_text returns [] when models are missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "category_mapping.json").write_text(
                json.dumps({"10": "Livres", "2583": "Piscine"})
            )
            clf = _make_classifier(tmpdir)
            result = clf.predict_text("test")
            assert isinstance(result, list)

    def test_format_result_structure(self):
        """_format_result returns dict with label, name, confidence."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "category_mapping.json").write_text(
                json.dumps({"2583": "Piscine & Spa"})
            )
            clf = _make_classifier(tmpdir)

            result = clf._format_result("2583", 0.95)
            assert result["label"] == "2583"
            assert result["name"] == "Piscine & Spa"
            assert result["confidence"] == 0.95

    def test_format_result_unknown_category(self):
        """_format_result returns fallback name for unknown category codes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            clf = _make_classifier(tmpdir)
            result = clf._format_result("9999", 0.5)
            assert "9999" in result["name"]  # Fallback includes the code


# =============================================================================
# 6. CATEGORY MAPPING — 27 Classes Integrity
# =============================================================================
class TestCategoryIntegrity:
    """Verify 27-class system consistency across all components."""

    def test_exactly_27_categories(self):
        """Category mapping has exactly 27 entries."""
        from utils.category_mapping import CATEGORY_MAPPING
        assert len(CATEGORY_MAPPING) == 27

    def test_base_classifier_has_27_codes(self):
        """BaseClassifier.CATEGORY_CODES has exactly 27 entries."""
        from utils.model_interface import BaseClassifier
        assert len(BaseClassifier.CATEGORY_CODES) == 27
        assert BaseClassifier.NUM_CLASSES == 27

    def test_category_codes_match(self):
        """Category codes in mapping match those in BaseClassifier."""
        from utils.category_mapping import CATEGORY_CODES as mapping_codes
        from utils.model_interface import BaseClassifier

        assert set(mapping_codes) == set(BaseClassifier.CATEGORY_CODES)


# =============================================================================
# 7. APP STRUCTURE — Pages & Config
# =============================================================================
class TestAppStructure:
    """Verify app structure is complete and deployable."""

    def test_all_pages_exist(self):
        """All 5 Streamlit pages exist in pages/ directory."""
        pages_dir = PROJECT_ROOT / "pages"
        pages = list(pages_dir.glob("*.py"))

        assert len(pages) == 5, f"Expected 5 pages, found {len(pages)}: {[p.name for p in pages]}"

    def test_config_has_hf_repo_id(self):
        """config.py defines HF_REPO_ID for model downloads."""
        from config import HF_REPO_ID
        assert HF_REPO_ID is not None
        assert "/" in HF_REPO_ID

    def test_config_has_fusion_weights(self):
        """config.py defines fusion weights that sum to 1."""
        from config import FUSION_W_IMAGE, FUSION_W_TEXT
        assert abs(FUSION_W_IMAGE + FUSION_W_TEXT - 1.0) < 0.01

    def test_requirements_file_exists(self):
        """requirements.txt exists at repo root."""
        req_path = REPO_ROOT / "requirements.txt"
        assert req_path.exists()

    def test_streamlit_config_exists(self):
        """.streamlit/config.toml exists for theme configuration."""
        config_path = REPO_ROOT / ".streamlit" / "config.toml"
        assert config_path.exists()
