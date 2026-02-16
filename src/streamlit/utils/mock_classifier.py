"""
Classifieur mock pour le développement et les tests UI.

Ce classifieur génère des prédictions simulées réalistes sans
nécessiter de vrais modèles ML. Il est utilisé pour:
- Développer et tester l'interface utilisateur
- Démontrer le fonctionnement de l'application
- Servir de fallback si les vrais modèles ne sont pas disponibles

Supporte plusieurs modèles simulés pour texte et image:
- Texte: TF-IDF + SVM, TF-IDF + Random Forest, CamemBERT
- Image: ResNet50 + SVM, ResNet50 + Random Forest, VGG16 + SVM
"""
import hashlib
import numpy as np
from typing import Optional, Dict, List, Tuple
from PIL import Image
from dataclasses import dataclass

from .model_interface import BaseClassifier, ClassificationResult
from .category_mapping import CATEGORY_CODES, get_category_name


# =============================================================================
# Configuration des modèles disponibles
# =============================================================================
@dataclass
class ModelConfig:
    """Configuration d'un modèle simulé."""
    name: str
    short_name: str
    description: str
    base_confidence: float  # Confiance de base (moyenne)
    confidence_std: float   # Écart-type de la confiance
    seed_offset: int        # Offset pour différencier les résultats
    color: str              # Couleur pour les graphiques


# Modèles de texte disponibles
TEXT_MODELS: Dict[str, ModelConfig] = {
    "tfidf_svm": ModelConfig(
        name="TF-IDF + SVM",
        short_name="TF-IDF/SVM",
        description="Vectorisation TF-IDF avec classifieur SVM linéaire",
        base_confidence=0.78,
        confidence_std=0.12,
        seed_offset=100,
        color="#2196F3"  # Bleu
    ),
    "tfidf_rf": ModelConfig(
        name="TF-IDF + Random Forest",
        short_name="TF-IDF/RF",
        description="Vectorisation TF-IDF avec Random Forest (100 arbres)",
        base_confidence=0.75,
        confidence_std=0.15,
        seed_offset=200,
        color="#4CAF50"  # Vert
    ),
    "camembert": ModelConfig(
        name="CamemBERT",
        short_name="CamemBERT",
        description="Modèle transformer pré-entraîné sur le français",
        base_confidence=0.85,
        confidence_std=0.08,
        seed_offset=300,
        color="#9C27B0"  # Violet
    ),
}

# Modèles d'image disponibles
IMAGE_MODELS: Dict[str, ModelConfig] = {
    "resnet50_svm": ModelConfig(
        name="ResNet50 + SVM",
        short_name="ResNet50/SVM",
        description="Features ResNet50 (2048D) avec classifieur SVM",
        base_confidence=0.72,
        confidence_std=0.14,
        seed_offset=400,
        color="#FF5722"  # Orange
    ),
    "resnet50_rf": ModelConfig(
        name="ResNet50 + Random Forest",
        short_name="ResNet50/RF",
        description="Features ResNet50 avec Random Forest (200 arbres)",
        base_confidence=0.70,
        confidence_std=0.16,
        seed_offset=500,
        color="#795548"  # Marron
    ),
    "vgg16_svm": ModelConfig(
        name="VGG16 + SVM",
        short_name="VGG16/SVM",
        description="Features VGG16 (4096D) avec classifieur SVM",
        base_confidence=0.68,
        confidence_std=0.15,
        seed_offset=600,
        color="#607D8B"  # Gris-bleu
    ),
}


def get_available_text_models() -> Dict[str, ModelConfig]:
    """Retourne les modèles de texte disponibles."""
    return TEXT_MODELS


def get_available_image_models() -> Dict[str, ModelConfig]:
    """Retourne les modèles d'image disponibles."""
    return IMAGE_MODELS


# =============================================================================
# Classifieur Mock de base
# =============================================================================
class MockClassifier(BaseClassifier):
    """
    Classifieur simulé pour le développement de l'interface.

    Ce classifieur génère des prédictions pseudo-aléatoires mais
    déterministes basées sur le hash des entrées.
    """

    def __init__(self, seed: int = 42, model_config: Optional[ModelConfig] = None):
        """
        Initialise le classifieur mock.

        Args:
            seed: Graine de base pour la génération aléatoire
            model_config: Configuration du modèle simulé (optionnel)
        """
        self._ready = True
        self._seed = seed
        self._model_config = model_config

    def predict(
        self,
        image: Optional[Image.Image] = None,
        text: Optional[str] = None,
        top_k: int = 5
    ) -> ClassificationResult:
        """
        Génère une prédiction simulée basée sur les entrées.
        """
        if image is None and (text is None or text.strip() == ""):
            raise ValueError("Au moins une image ou un texte est requis")

        # Générer une graine déterministe basée sur les entrées
        seed_offset = self._model_config.seed_offset if self._model_config else 0
        hash_input = self._generate_hash(image, text, seed_offset)
        rng = np.random.RandomState(hash_input)

        # Générer des probabilités avec les caractéristiques du modèle
        probabilities = self._generate_probabilities(rng)

        # Déterminer la source
        if image is not None and text and text.strip():
            source = "mock_multimodal"
        elif image is not None:
            source = "mock_image"
        else:
            source = "mock_text"

        # Ajouter le nom du modèle si configuré
        if self._model_config:
            source = f"{source}_{self._model_config.short_name}"

        # Construire le résultat
        top_predictions = self._probabilities_to_predictions(probabilities, top_k)
        best_category, best_confidence = top_predictions[0]

        return ClassificationResult(
            category=best_category,
            confidence=best_confidence,
            top_k_predictions=top_predictions,
            source=source,
            raw_probabilities=probabilities
        )

    def _generate_probabilities(self, rng: np.random.RandomState) -> np.ndarray:
        """Génère des probabilités avec les caractéristiques du modèle."""
        alpha = np.ones(self.NUM_CLASSES) * 0.5
        peak_indices = rng.choice(self.NUM_CLASSES, size=3, replace=False)

        if self._model_config:
            # Ajuster selon la confiance de base du modèle
            peak_strength = 2.0 + (self._model_config.base_confidence - 0.7) * 10
            alpha[peak_indices] = rng.uniform(peak_strength, peak_strength + 3.0, size=3)
        else:
            alpha[peak_indices] = rng.uniform(2.0, 5.0, size=3)

        probabilities = rng.dirichlet(alpha)

        # Ajuster la confiance maximale selon le modèle
        if self._model_config:
            max_idx = np.argmax(probabilities)
            target_conf = np.clip(
                rng.normal(self._model_config.base_confidence, self._model_config.confidence_std),
                0.4, 0.98
            )
            # Rescale pour atteindre la confiance cible
            current_max = probabilities[max_idx]
            if current_max > 0:
                scale_factor = target_conf / current_max
                probabilities[max_idx] = target_conf
                # Redistribuer le reste
                other_mask = np.ones(len(probabilities), dtype=bool)
                other_mask[max_idx] = False
                remaining = 1.0 - target_conf
                if probabilities[other_mask].sum() > 0:
                    probabilities[other_mask] *= remaining / probabilities[other_mask].sum()

        return probabilities

    def load_model(self, path: str) -> None:
        """Simule le chargement d'un modèle."""
        self._ready = True

    @property
    def is_ready(self) -> bool:
        """Le mock est toujours prêt."""
        return self._ready

    @property
    def model_config(self) -> Optional[ModelConfig]:
        """Retourne la configuration du modèle."""
        return self._model_config

    def _generate_hash(
        self,
        image: Optional[Image.Image],
        text: Optional[str],
        seed_offset: int = 0
    ) -> int:
        """Génère un hash déterministe à partir des entrées."""
        hash_parts = [str(self._seed + seed_offset)]

        if image is not None:
            hash_parts.append(f"{image.size}")
            small = image.resize((8, 8)).convert("L")
            hash_parts.append(small.tobytes().hex()[:32])

        if text and text.strip():
            hash_parts.append(text.strip()[:200])

        combined = "|".join(hash_parts)
        hash_bytes = hashlib.md5(combined.encode()).digest()
        return int.from_bytes(hash_bytes[:4], byteorder="big")


# =============================================================================
# Classifieur de démonstration avec mots-clés
# =============================================================================
class DemoClassifier(MockClassifier):
    """
    Classifieur de démonstration avec des prédictions prédéfinies
    pour certains mots-clés, permettant des démos contrôlées.
    """

    KEYWORD_PREDICTIONS = {
        "piscine": ("2583", 0.92),
        "pool": ("2583", 0.88),
        "livre": ("2403", 0.85),
        "book": ("2403", 0.82),
        "harry potter": ("2403", 0.94),
        "roman": ("2403", 0.80),
        "jeu vidéo": ("2462", 0.90),
        "console": ("2462", 0.87),
        "playstation": ("2462", 0.95),
        "xbox": ("2462", 0.94),
        "nintendo": ("2462", 0.93),
        "figurine": ("1281", 0.88),
        "funko": ("1281", 0.91),
        "pokemon": ("1280", 0.91),
        "jouet": ("1280", 0.84),
        "lego": ("1280", 0.93),
        "bébé": ("1320", 0.86),
        "meuble": ("1560", 0.83),
        "jardin": ("2582", 0.89),
        "tondeuse": ("2585", 0.87),
        "outil": ("2585", 0.85),
        "iphone": ("2583", 0.90),
        "smartphone": ("2583", 0.88),
        "téléphone": ("2583", 0.85),
        "coque": ("2583", 0.82),
        "robe": ("1920", 0.86),
        "vêtement": ("1920", 0.83),
        "maquillage": ("1301", 0.89),
        "parfum": ("1301", 0.87),
        "bougie": ("1302", 0.84),
    }

    def __init__(self, seed: int = 42, model_config: Optional[ModelConfig] = None):
        super().__init__(seed, model_config)
        # Ajuster les confiances selon le modèle
        self._adjusted_predictions = self._adjust_predictions_for_model()

    def _adjust_predictions_for_model(self) -> Dict[str, Tuple[str, float]]:
        """Ajuste les prédictions selon les caractéristiques du modèle."""
        if not self._model_config:
            return self.KEYWORD_PREDICTIONS.copy()

        adjusted = {}
        rng = np.random.RandomState(self._model_config.seed_offset)

        for keyword, (category, base_conf) in self.KEYWORD_PREDICTIONS.items():
            # Ajuster la confiance avec une variation selon le modèle
            variation = rng.normal(0, 0.05)
            model_factor = self._model_config.base_confidence / 0.80  # Normaliser sur 0.80
            new_conf = np.clip(base_conf * model_factor + variation, 0.5, 0.98)
            adjusted[keyword] = (category, new_conf)

        return adjusted

    def predict(
        self,
        image: Optional[Image.Image] = None,
        text: Optional[str] = None,
        top_k: int = 5
    ) -> ClassificationResult:
        """Génère une prédiction basée sur des mots-clés ou le mock standard."""
        predictions_to_use = self._adjusted_predictions if self._model_config else self.KEYWORD_PREDICTIONS

        if text:
            text_lower = text.lower()
            for keyword, (category, confidence) in predictions_to_use.items():
                if keyword in text_lower:
                    probabilities = self._generate_keyword_probabilities(
                        category, confidence
                    )
                    top_predictions = self._probabilities_to_predictions(
                        probabilities, top_k
                    )

                    source = "demo"
                    if self._model_config:
                        source = f"demo_{self._model_config.short_name}"

                    return ClassificationResult(
                        category=category,
                        confidence=confidence,
                        top_k_predictions=top_predictions,
                        source=source,
                        raw_probabilities=probabilities
                    )

        return super().predict(image, text, top_k)

    def _generate_keyword_probabilities(
        self,
        main_category: str,
        main_confidence: float
    ) -> np.ndarray:
        """Génère des probabilités cohérentes avec une prédiction principale."""
        seed = self._model_config.seed_offset if self._model_config else 42
        rng = np.random.RandomState(seed)
        probabilities = rng.dirichlet(np.ones(self.NUM_CLASSES) * 0.3)

        main_idx = CATEGORY_CODES.index(main_category)
        remaining = 1.0 - main_confidence
        other_sum = probabilities.sum() - probabilities[main_idx]
        if other_sum > 0:
            scale = remaining / other_sum
            probabilities *= scale
        probabilities[main_idx] = main_confidence

        return probabilities


# =============================================================================
# Factory pour créer des classifieurs avec différents modèles
# =============================================================================
class MultiModelClassifier:
    """
    Gestionnaire de plusieurs modèles pour comparaison.

    Permet de créer et gérer plusieurs classifieurs avec différentes
    configurations pour comparer leurs performances.
    """

    def __init__(self):
        self._text_classifiers: Dict[str, DemoClassifier] = {}
        self._image_classifiers: Dict[str, DemoClassifier] = {}
        self._initialize_classifiers()

    def _initialize_classifiers(self):
        """Initialise tous les classifieurs disponibles."""
        for model_id, config in TEXT_MODELS.items():
            self._text_classifiers[model_id] = DemoClassifier(model_config=config)

        for model_id, config in IMAGE_MODELS.items():
            self._image_classifiers[model_id] = DemoClassifier(model_config=config)

    def get_text_classifier(self, model_id: str) -> DemoClassifier:
        """Retourne un classifieur texte spécifique."""
        if model_id not in self._text_classifiers:
            raise ValueError(f"Modèle texte inconnu: {model_id}")
        return self._text_classifiers[model_id]

    def get_image_classifier(self, model_id: str) -> DemoClassifier:
        """Retourne un classifieur image spécifique."""
        if model_id not in self._image_classifiers:
            raise ValueError(f"Modèle image inconnu: {model_id}")
        return self._image_classifiers[model_id]

    def predict_all_text_models(
        self,
        text: str,
        top_k: int = 5
    ) -> Dict[str, ClassificationResult]:
        """
        Exécute tous les modèles texte sur la même entrée.

        Returns:
            Dict avec model_id -> ClassificationResult
        """
        results = {}
        for model_id, classifier in self._text_classifiers.items():
            results[model_id] = classifier.predict(text=text, top_k=top_k)
        return results

    def predict_all_image_models(
        self,
        image: Image.Image,
        top_k: int = 5
    ) -> Dict[str, ClassificationResult]:
        """
        Exécute tous les modèles image sur la même entrée.

        Returns:
            Dict avec model_id -> ClassificationResult
        """
        results = {}
        for model_id, classifier in self._image_classifiers.items():
            results[model_id] = classifier.predict(image=image, top_k=top_k)
        return results

    def get_comparison_metrics(
        self,
        results: Dict[str, ClassificationResult]
    ) -> Dict[str, any]:
        """
        Calcule des métriques de comparaison entre modèles.

        Returns:
            Dict avec métriques agrégées
        """
        confidences = {k: r.confidence for k, r in results.items()}
        categories = {k: r.category for k, r in results.items()}

        # Trouver le consensus
        from collections import Counter
        category_counts = Counter(categories.values())
        consensus_category = category_counts.most_common(1)[0][0]
        agreement_ratio = category_counts[consensus_category] / len(results)

        return {
            "confidences": confidences,
            "categories": categories,
            "consensus_category": consensus_category,
            "agreement_ratio": agreement_ratio,
            "best_model": max(confidences, key=confidences.get),
            "avg_confidence": np.mean(list(confidences.values())),
            "std_confidence": np.std(list(confidences.values())),
        }
