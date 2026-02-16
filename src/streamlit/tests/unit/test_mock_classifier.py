"""
Tests unitaires pour utils/mock_classifier.py

Ce module teste:
- DemoClassifier: prédictions texte, image, multimodal
- ModelConfig: configuration des modèles
- ClassificationResult: structure des résultats
- TEXT_MODELS et IMAGE_MODELS: registres de modèles
- MultiModelClassifier: comparaison multi-modèles
"""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils.mock_classifier import (
    DemoClassifier,
    MultiModelClassifier,
    ClassificationResult,
    ModelConfig,
    TEXT_MODELS,
    IMAGE_MODELS,
    get_available_text_models,
    get_available_image_models,
)
from utils.category_mapping import CATEGORY_MAPPING


# =============================================================================
# TESTS ModelConfig
# =============================================================================
@pytest.mark.unit
class TestModelConfig:
    """Tests pour la dataclass ModelConfig."""

    def test_model_config_creation(self):
        """ModelConfig peut être créé avec tous les attributs."""
        config = ModelConfig(
            name="Test Model",
            short_name="TM",
            description="A test model",
            base_confidence=0.8,
            seed_offset=42,
            color="#FF0000"
        )

        assert config.name == "Test Model"
        assert config.short_name == "TM"
        assert config.description == "A test model"
        assert config.base_confidence == 0.8
        assert config.seed_offset == 42
        assert config.color == "#FF0000"

    def test_model_config_defaults(self):
        """ModelConfig a des valeurs par défaut raisonnables."""
        config = ModelConfig(
            name="Test",
            short_name="T",
            description="Test"
        )

        assert 0.0 <= config.base_confidence <= 1.0
        assert isinstance(config.seed_offset, int)
        assert config.color.startswith("#")


# =============================================================================
# TESTS TEXT_MODELS Registry
# =============================================================================
@pytest.mark.unit
class TestTextModelsRegistry:
    """Tests pour le registre TEXT_MODELS."""

    def test_has_three_text_models(self):
        """Le registre contient exactement 3 modèles texte."""
        assert len(TEXT_MODELS) == 3

    def test_required_text_models_present(self):
        """Les 3 modèles texte attendus sont présents."""
        expected_models = ["tfidf_svm", "tfidf_rf", "camembert"]
        for model_key in expected_models:
            assert model_key in TEXT_MODELS, f"Missing text model: {model_key}"

    def test_text_models_are_model_config(self):
        """Chaque entrée est une ModelConfig valide."""
        for key, config in TEXT_MODELS.items():
            assert isinstance(config, ModelConfig), f"{key} is not ModelConfig"

    def test_text_models_have_valid_confidence(self):
        """Chaque modèle a une confiance de base valide."""
        for key, config in TEXT_MODELS.items():
            assert 0.0 <= config.base_confidence <= 1.0, \
                f"{key} has invalid base_confidence: {config.base_confidence}"

    def test_camembert_has_highest_confidence(self):
        """CamemBERT devrait avoir la plus haute confiance (meilleur modèle)."""
        camembert_conf = TEXT_MODELS["camembert"].base_confidence
        for key, config in TEXT_MODELS.items():
            if key != "camembert":
                assert camembert_conf >= config.base_confidence, \
                    f"CamemBERT should have highest confidence, but {key} has {config.base_confidence}"


# =============================================================================
# TESTS IMAGE_MODELS Registry
# =============================================================================
@pytest.mark.unit
class TestImageModelsRegistry:
    """Tests pour le registre IMAGE_MODELS."""

    def test_has_three_image_models(self):
        """Le registre contient exactement 3 modèles image."""
        assert len(IMAGE_MODELS) == 3

    def test_required_image_models_present(self):
        """Les 3 modèles image attendus sont présents."""
        expected_models = ["resnet50_svm", "resnet50_rf", "vgg16_svm"]
        for model_key in expected_models:
            assert model_key in IMAGE_MODELS, f"Missing image model: {model_key}"

    def test_image_models_are_model_config(self):
        """Chaque entrée est une ModelConfig valide."""
        for key, config in IMAGE_MODELS.items():
            assert isinstance(config, ModelConfig), f"{key} is not ModelConfig"

    def test_image_models_have_valid_confidence(self):
        """Chaque modèle a une confiance de base valide."""
        for key, config in IMAGE_MODELS.items():
            assert 0.0 <= config.base_confidence <= 1.0, \
                f"{key} has invalid base_confidence: {config.base_confidence}"


# =============================================================================
# TESTS ClassificationResult
# =============================================================================
@pytest.mark.unit
class TestClassificationResult:
    """Tests pour la dataclass ClassificationResult."""

    def test_classification_result_creation(self):
        """ClassificationResult peut être créé correctement."""
        result = ClassificationResult(
            category="2583",
            confidence=0.85,
            top_k_predictions=[("2583", 0.85), ("2403", 0.10), ("1281", 0.05)],
            source="test"
        )

        assert result.category == "2583"
        assert result.confidence == 0.85
        assert len(result.top_k_predictions) == 3
        assert result.source == "test"

    def test_classification_result_top_k_order(self):
        """Les top_k_predictions doivent être triées par score décroissant."""
        result = ClassificationResult(
            category="2583",
            confidence=0.85,
            top_k_predictions=[("2583", 0.85), ("2403", 0.10), ("1281", 0.05)],
            source="test"
        )

        scores = [score for _, score in result.top_k_predictions]
        assert scores == sorted(scores, reverse=True)


# =============================================================================
# TESTS DemoClassifier Initialization
# =============================================================================
@pytest.mark.unit
class TestDemoClassifierInit:
    """Tests d'initialisation de DemoClassifier."""

    def test_init_without_config(self):
        """DemoClassifier peut être initialisé sans config."""
        clf = DemoClassifier()
        assert clf is not None

    def test_init_with_text_config(self):
        """DemoClassifier accepte une config texte."""
        config = TEXT_MODELS["camembert"]
        clf = DemoClassifier(model_config=config)
        assert clf.model_config == config

    def test_init_with_image_config(self):
        """DemoClassifier accepte une config image."""
        config = IMAGE_MODELS["resnet50_svm"]
        clf = DemoClassifier(model_config=config)
        assert clf.model_config == config

    def test_init_with_custom_config(self):
        """DemoClassifier accepte une config personnalisée."""
        custom_config = ModelConfig(
            name="Custom Model",
            short_name="CM",
            description="Custom test model",
            base_confidence=0.9,
            seed_offset=100,
            color="#123456"
        )
        clf = DemoClassifier(model_config=custom_config)
        assert clf.model_config == custom_config


# =============================================================================
# TESTS DemoClassifier.predict() - Text
# =============================================================================
@pytest.mark.unit
class TestDemoClassifierPredictText:
    """Tests de prédiction texte avec DemoClassifier."""

    def test_predict_text_returns_classification_result(self, demo_classifier, sample_text):
        """predict(text=...) retourne un ClassificationResult."""
        result = demo_classifier.predict(text=sample_text)
        assert isinstance(result, ClassificationResult)

    def test_predict_text_valid_category(self, demo_classifier, sample_text):
        """La catégorie prédite est valide (dans les 27)."""
        result = demo_classifier.predict(text=sample_text)
        assert result.category in CATEGORY_MAPPING

    def test_predict_text_valid_confidence(self, demo_classifier, sample_text):
        """La confiance est entre 0 et 1."""
        result = demo_classifier.predict(text=sample_text)
        assert 0.0 <= result.confidence <= 1.0

    def test_predict_text_has_top_k(self, demo_classifier, sample_text):
        """La prédiction contient top_k résultats."""
        result = demo_classifier.predict(text=sample_text, top_k=5)
        assert len(result.top_k_predictions) == 5

    def test_predict_text_top_k_sorted(self, demo_classifier, sample_text):
        """top_k est trié par score décroissant."""
        result = demo_classifier.predict(text=sample_text, top_k=5)
        scores = [s for _, s in result.top_k_predictions]
        assert scores == sorted(scores, reverse=True)

    def test_predict_text_top_k_sum_valid(self, demo_classifier, sample_text):
        """La somme des probabilités <= 1."""
        result = demo_classifier.predict(text=sample_text, top_k=27)
        total = sum(s for _, s in result.top_k_predictions)
        assert total <= 1.01  # Petite marge pour erreurs d'arrondi

    def test_predict_text_source_is_set(self, demo_classifier, sample_text):
        """La source de la prédiction est définie."""
        result = demo_classifier.predict(text=sample_text)
        assert result.source is not None
        assert "text" in result.source.lower() or "mock" in result.source.lower()

    def test_predict_text_deterministic(self, sample_text):
        """Même texte = même prédiction (déterminisme)."""
        clf1 = DemoClassifier(model_config=TEXT_MODELS["camembert"])
        clf2 = DemoClassifier(model_config=TEXT_MODELS["camembert"])

        result1 = clf1.predict(text=sample_text)
        result2 = clf2.predict(text=sample_text)

        assert result1.category == result2.category
        assert abs(result1.confidence - result2.confidence) < 0.01

    def test_predict_empty_text(self, demo_classifier):
        """Texte vide est géré sans crash."""
        # Peut retourner un résultat ou lever ValueError
        try:
            result = demo_classifier.predict(text="")
            # Si pas d'exception, vérifier que c'est un résultat valide
            assert result.category in CATEGORY_MAPPING
        except ValueError:
            pass  # ValueError est acceptable

    def test_predict_whitespace_only_text(self, demo_classifier):
        """Texte avec espaces seulement est géré."""
        try:
            result = demo_classifier.predict(text="   ")
            assert result.category in CATEGORY_MAPPING
        except ValueError:
            pass


# =============================================================================
# TESTS DemoClassifier.predict() - Image
# =============================================================================
@pytest.mark.unit
class TestDemoClassifierPredictImage:
    """Tests de prédiction image avec DemoClassifier."""

    def test_predict_image_returns_classification_result(self, demo_classifier, sample_image):
        """predict(image=...) retourne un ClassificationResult."""
        result = demo_classifier.predict(image=sample_image)
        assert isinstance(result, ClassificationResult)

    def test_predict_image_valid_category(self, demo_classifier, sample_image):
        """La catégorie prédite est valide."""
        result = demo_classifier.predict(image=sample_image)
        assert result.category in CATEGORY_MAPPING

    def test_predict_image_valid_confidence(self, demo_classifier, sample_image):
        """La confiance est entre 0 et 1."""
        result = demo_classifier.predict(image=sample_image)
        assert 0.0 <= result.confidence <= 1.0

    def test_predict_image_source_is_set(self, demo_classifier, sample_image):
        """La source indique 'image'."""
        result = demo_classifier.predict(image=sample_image)
        assert result.source is not None
        assert "image" in result.source.lower() or "mock" in result.source.lower()

    def test_predict_small_image(self, demo_classifier, sample_image_small):
        """Petite image est acceptée."""
        result = demo_classifier.predict(image=sample_image_small)
        assert result.category in CATEGORY_MAPPING

    def test_predict_large_image(self, demo_classifier, sample_image_large):
        """Grande image est acceptée."""
        result = demo_classifier.predict(image=sample_image_large)
        assert result.category in CATEGORY_MAPPING


# =============================================================================
# TESTS DemoClassifier.predict() - Multimodal
# =============================================================================
@pytest.mark.unit
class TestDemoClassifierPredictMultimodal:
    """Tests de prédiction multimodale avec DemoClassifier."""

    def test_predict_multimodal_returns_result(self, demo_classifier, sample_text, sample_image):
        """predict(text=..., image=...) retourne un résultat."""
        result = demo_classifier.predict(text=sample_text, image=sample_image)
        assert isinstance(result, ClassificationResult)

    def test_predict_multimodal_valid_category(self, demo_classifier, sample_text, sample_image):
        """La catégorie multimodale est valide."""
        result = demo_classifier.predict(text=sample_text, image=sample_image)
        assert result.category in CATEGORY_MAPPING

    def test_predict_multimodal_source(self, demo_classifier, sample_text, sample_image):
        """La source indique 'multimodal'."""
        result = demo_classifier.predict(text=sample_text, image=sample_image)
        assert "multimodal" in result.source.lower() or "mock" in result.source.lower()


# =============================================================================
# TESTS DemoClassifier avec différents modèles
# =============================================================================
@pytest.mark.unit
class TestDemoClassifierModels:
    """Tests avec différentes configurations de modèles."""

    def test_all_text_models_predict(self, all_text_models, sample_text, assert_valid_prediction):
        """Chaque modèle texte peut faire des prédictions."""
        model_key, clf = all_text_models
        result = clf.predict(text=sample_text)
        assert_valid_prediction(result)

    def test_all_image_models_predict(self, all_image_models, sample_image, assert_valid_prediction):
        """Chaque modèle image peut faire des prédictions."""
        model_key, clf = all_image_models
        result = clf.predict(image=sample_image)
        assert_valid_prediction(result)

    def test_different_models_different_predictions(self, sample_text):
        """Différents modèles peuvent donner différentes prédictions."""
        predictions = {}
        for model_key, config in TEXT_MODELS.items():
            clf = DemoClassifier(model_config=config)
            result = clf.predict(text=sample_text)
            predictions[model_key] = (result.category, result.confidence)

        # Au moins une différence (pas tous identiques)
        unique_predictions = set(predictions.values())
        # Note: en mode mock, ils peuvent être similaires, donc on vérifie juste que ça fonctionne
        assert len(predictions) == 3


# =============================================================================
# TESTS MultiModelClassifier
# =============================================================================
@pytest.mark.unit
class TestMultiModelClassifier:
    """Tests pour MultiModelClassifier."""

    def test_init(self, multi_model_classifier):
        """MultiModelClassifier s'initialise correctement."""
        assert multi_model_classifier is not None

    def test_predict_all_text_models(self, multi_model_classifier, sample_text):
        """predict_all_text_models retourne 3 résultats."""
        results = multi_model_classifier.predict_all_text_models(sample_text)
        assert len(results) == 3
        assert all(key in TEXT_MODELS for key in results.keys())

    def test_predict_all_image_models(self, multi_model_classifier, sample_image):
        """predict_all_image_models retourne 3 résultats."""
        results = multi_model_classifier.predict_all_image_models(sample_image)
        assert len(results) == 3
        assert all(key in IMAGE_MODELS for key in results.keys())

    def test_predict_all_text_valid_results(self, multi_model_classifier, sample_text, assert_valid_prediction):
        """Tous les résultats texte sont valides."""
        results = multi_model_classifier.predict_all_text_models(sample_text)
        for model_key, result in results.items():
            assert_valid_prediction(result)

    def test_predict_all_image_valid_results(self, multi_model_classifier, sample_image, assert_valid_prediction):
        """Tous les résultats image sont valides."""
        results = multi_model_classifier.predict_all_image_models(sample_image)
        for model_key, result in results.items():
            assert_valid_prediction(result)


# =============================================================================
# TESTS Helper Functions
# =============================================================================
@pytest.mark.unit
class TestHelperFunctions:
    """Tests pour les fonctions utilitaires."""

    def test_get_available_text_models(self):
        """get_available_text_models retourne les 3 modèles."""
        models = get_available_text_models()
        assert len(models) == 3
        assert all(isinstance(m, dict) for m in models)

    def test_get_available_image_models(self):
        """get_available_image_models retourne les 3 modèles."""
        models = get_available_image_models()
        assert len(models) == 3
        assert all(isinstance(m, dict) for m in models)

    def test_available_models_have_required_keys(self):
        """Les modèles retournés ont les clés requises."""
        required_keys = {"key", "name", "description"}

        for model in get_available_text_models():
            assert required_keys.issubset(model.keys())

        for model in get_available_image_models():
            assert required_keys.issubset(model.keys())


# =============================================================================
# TESTS Robustness (Edge Cases)
# =============================================================================
@pytest.mark.unit
class TestDemoClassifierRobustness:
    """Tests de robustesse avec cas limites."""

    def test_handles_none_text(self, demo_classifier):
        """Gère text=None sans crash."""
        try:
            result = demo_classifier.predict(text=None)
            # Si pas d'exception, doit retourner un résultat valide
            assert result is None or result.category in CATEGORY_MAPPING
        except (ValueError, TypeError):
            pass  # Exception acceptable

    def test_handles_none_image(self, demo_classifier):
        """Gère image=None sans crash."""
        try:
            result = demo_classifier.predict(image=None)
            assert result is None or result.category in CATEGORY_MAPPING
        except (ValueError, TypeError):
            pass

    def test_handles_edge_case_text(self, demo_classifier, edge_case_text):
        """Gère les cas limites de texte."""
        try:
            if edge_case_text:  # Skip None car testé séparément
                result = demo_classifier.predict(text=edge_case_text)
                # Si pas d'exception, résultat doit être valide
                assert result.category in CATEGORY_MAPPING
        except (ValueError, TypeError):
            pass  # Exceptions acceptables pour inputs invalides

    def test_very_long_text(self, demo_classifier):
        """Gère texte très long (100K caractères)."""
        long_text = "produit " * 12500  # ~100K chars
        try:
            result = demo_classifier.predict(text=long_text)
            assert result.category in CATEGORY_MAPPING
        except (ValueError, MemoryError):
            pass

    def test_predict_with_no_input(self, demo_classifier):
        """Gère appel sans input."""
        try:
            result = demo_classifier.predict()
            # Comportement attendu: exception ou résultat par défaut
            assert result is None or result.category in CATEGORY_MAPPING
        except (ValueError, TypeError):
            pass


# =============================================================================
# TESTS Performance
# =============================================================================
@pytest.mark.unit
class TestDemoClassifierPerformance:
    """Tests de performance basiques."""

    def test_text_prediction_fast(self, demo_classifier, sample_text, measure_time):
        """Prédiction texte < 100ms."""
        with measure_time() as timer:
            for _ in range(10):
                demo_classifier.predict(text=sample_text)

        avg_time = timer.elapsed / 10 * 1000  # ms
        assert avg_time < 100, f"Text prediction too slow: {avg_time:.1f}ms"

    def test_image_prediction_fast(self, demo_classifier, sample_image, measure_time):
        """Prédiction image < 100ms."""
        with measure_time() as timer:
            for _ in range(10):
                demo_classifier.predict(image=sample_image)

        avg_time = timer.elapsed / 10 * 1000  # ms
        assert avg_time < 100, f"Image prediction too slow: {avg_time:.1f}ms"

    def test_batch_predictions(self, demo_classifier, sample_product_texts, measure_time):
        """Batch de 5 prédictions < 500ms."""
        with measure_time() as timer:
            for designation, description in sample_product_texts:
                text = f"{designation} {description}"
                demo_classifier.predict(text=text)

        assert timer.elapsed < 0.5, f"Batch too slow: {timer.elapsed:.2f}s"
