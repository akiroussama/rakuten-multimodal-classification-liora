"""
Interface abstraite pour les classifieurs de produits Rakuten.

Ce module définit le contrat que tous les classifieurs doivent respecter,
permettant une intégration facile des modèles développés par l'équipe.

Architecture:
- BaseClassifier: Classe abstraite définissant l'interface
- ClassificationResult: Dataclass pour structurer les résultats
- Les implémentations concrètes héritent de BaseClassifier
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import numpy as np
from PIL import Image


@dataclass
class ClassificationResult:
    """
    Résultat structuré d'une classification de produit.

    Attributes:
        category: Code de la catégorie prédite (ex: "2583")
        confidence: Score de confiance [0, 1] pour la prédiction principale
        top_k_predictions: Liste des (code_catégorie, score) triées par score décroissant
        source: Source de la prédiction ("image", "text", "multimodal", "mock")
        raw_probabilities: Vecteur complet des probabilités (27 classes)
    """
    category: str
    confidence: float
    top_k_predictions: List[Tuple[str, float]] = field(default_factory=list)
    source: str = "unknown"
    raw_probabilities: Optional[np.ndarray] = None

    def __post_init__(self):
        """Validation des données après initialisation."""
        if not 0 <= self.confidence <= 1:
            raise ValueError(f"Confidence must be in [0, 1], got {self.confidence}")

    def to_dict(self) -> dict:
        """Convertit le résultat en dictionnaire pour affichage."""
        return {
            "category": self.category,
            "confidence": self.confidence,
            "top_k": self.top_k_predictions,
            "source": self.source,
        }


class BaseClassifier(ABC):
    """
    Classe abstraite définissant l'interface pour tous les classifieurs.

    Cette interface permet d'intégrer facilement différents types de modèles:
    - Classifieur image (ResNet50 features + ML)
    - Classifieur texte (TF-IDF + ML)
    - Classifieur multimodal (fusion des deux)
    - Mock classifieur (pour tests UI)

    Usage:
        class MyClassifier(BaseClassifier):
            def predict(self, image=None, text=None) -> ClassificationResult:
                # Implémentation spécifique
                pass

            def load_model(self, path):
                # Chargement du modèle
                pass

    Les modèles des collègues doivent respecter cette interface pour
    être intégrés dans l'application Streamlit.
    """

    # Liste des 27 codes de catégories Rakuten
    CATEGORY_CODES = [
        "10", "40", "50", "60", "1140", "1160", "1180", "1280", "1281",
        "1300", "1301", "1302", "1320", "1560", "1920", "1940", "2060",
        "2220", "2280", "2403", "2462", "2522", "2582", "2583", "2585",
        "2705", "2905"
    ]

    NUM_CLASSES = 27

    @abstractmethod
    def predict(
        self,
        image: Optional[Image.Image] = None,
        text: Optional[str] = None,
        top_k: int = 5
    ) -> ClassificationResult:
        """
        Effectue une prédiction de catégorie pour un produit.

        Args:
            image: Image PIL du produit (optionnel selon le type de classifieur)
            text: Texte du produit (designation + description, optionnel)
            top_k: Nombre de prédictions à retourner

        Returns:
            ClassificationResult contenant la catégorie prédite et les scores

        Raises:
            ValueError: Si ni image ni texte n'est fourni (selon le type)
        """
        pass

    @abstractmethod
    def load_model(self, path: str) -> None:
        """
        Charge un modèle depuis un fichier.

        Args:
            path: Chemin vers le fichier du modèle (.joblib, .h5, etc.)

        Raises:
            FileNotFoundError: Si le fichier n'existe pas
            RuntimeError: Si le chargement échoue
        """
        pass

    @property
    @abstractmethod
    def is_ready(self) -> bool:
        """
        Vérifie si le classifieur est prêt à effectuer des prédictions.

        Returns:
            True si le modèle est chargé et fonctionnel
        """
        pass

    @property
    def name(self) -> str:
        """Retourne le nom du classifieur."""
        return self.__class__.__name__

    def _validate_inputs(
        self,
        image: Optional[Image.Image],
        text: Optional[str],
        require_image: bool = False,
        require_text: bool = False
    ) -> None:
        """
        Valide les entrées selon les exigences du classifieur.

        Args:
            image: Image à valider
            text: Texte à valider
            require_image: Si True, l'image est obligatoire
            require_text: Si True, le texte est obligatoire

        Raises:
            ValueError: Si les entrées requises sont manquantes
        """
        if require_image and image is None:
            raise ValueError(f"{self.name} requires an image input")
        if require_text and (text is None or text.strip() == ""):
            raise ValueError(f"{self.name} requires a text input")

    def _probabilities_to_predictions(
        self,
        probabilities: np.ndarray,
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Convertit un vecteur de probabilités en liste de prédictions triées.

        Args:
            probabilities: Array de shape (27,) avec les probabilités par classe
            top_k: Nombre de prédictions à retourner

        Returns:
            Liste de tuples (code_catégorie, score) triés par score décroissant
        """
        if len(probabilities) != self.NUM_CLASSES:
            raise ValueError(
                f"Expected {self.NUM_CLASSES} probabilities, got {len(probabilities)}"
            )

        # Indices triés par probabilité décroissante
        sorted_indices = np.argsort(probabilities)[::-1]

        predictions = []
        for idx in sorted_indices[:top_k]:
            category_code = self.CATEGORY_CODES[idx]
            score = float(probabilities[idx])
            predictions.append((category_code, score))

        return predictions
