"""
Chargement et gestion des données pour l'application Rakuten.

Ce module gère le chargement des données d'entraînement et de test,
avec fallback sur des données mock si les fichiers ne sont pas disponibles.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import RAW_DATA_DIR, PROCESSED_DATA_DIR, IMAGES_DIR
from utils.category_mapping import CATEGORY_CODES, CATEGORY_MAPPING


# =============================================================================
# Chemins des fichiers de données
# =============================================================================
X_TRAIN_PATH = RAW_DATA_DIR / "X_train_update.csv"
Y_TRAIN_PATH = RAW_DATA_DIR / "Y_train_CVw08PX.csv"
X_TEST_PATH = RAW_DATA_DIR / "X_test_update.csv"
IMAGES_TRAIN_DIR = IMAGES_DIR / "images" / "image_train"
IMAGES_TEST_DIR = IMAGES_DIR / "images" / "image_test"


def is_data_available() -> bool:
    """Vérifie si les données réelles sont disponibles."""
    return X_TRAIN_PATH.exists() and Y_TRAIN_PATH.exists()


def load_training_data() -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Charge les données d'entraînement (X et Y).

    Returns:
        Tuple (X_train, Y_train) ou (None, None) si non disponible
    """
    if not is_data_available():
        return None, None

    try:
        X_train = pd.read_csv(X_TRAIN_PATH, index_col=0)
        Y_train = pd.read_csv(Y_TRAIN_PATH, index_col=0)
        return X_train, Y_train
    except Exception as e:
        print(f"Erreur chargement données: {e}")
        return None, None


def load_test_data() -> Optional[pd.DataFrame]:
    """Charge les données de test."""
    if not X_TEST_PATH.exists():
        return None

    try:
        return pd.read_csv(X_TEST_PATH, index_col=0)
    except Exception:
        return None


def get_category_distribution(Y_train: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Retourne la distribution des catégories.

    Si Y_train est None, retourne des données mock réalistes basées
    sur les statistiques connues du dataset Rakuten.
    """
    if Y_train is not None:
        # Données réelles
        dist = Y_train['prdtypecode'].value_counts().reset_index()
        dist.columns = ['category_code', 'count']
        dist['category_code'] = dist['category_code'].astype(str)
    else:
        # Données mock réalistes (basées sur distribution connue)
        mock_distribution = {
            "2583": 8560, "1280": 7550, "2705": 6850, "2522": 6100,
            "1560": 5800, "2060": 5500, "1920": 5200, "1140": 4800,
            "1180": 4500, "1300": 4200, "2403": 4000, "2582": 3800,
            "1281": 3600, "1320": 3400, "40": 3200, "1302": 3000,
            "2280": 2800, "2220": 2600, "10": 2400, "1160": 2200,
            "50": 2000, "60": 1800, "1301": 1600, "2585": 1400,
            "1940": 1200, "2462": 1000, "2905": 800
        }
        dist = pd.DataFrame([
            {"category_code": code, "count": count}
            for code, count in mock_distribution.items()
        ])

    # Ajouter les noms de catégories
    dist['category_name'] = dist['category_code'].apply(
        lambda x: CATEGORY_MAPPING.get(str(x), ("Inconnu", "", ""))[0]
    )
    dist['category_full'] = dist['category_code'].apply(
        lambda x: CATEGORY_MAPPING.get(str(x), ("", "Inconnu", ""))[1]
    )
    dist['emoji'] = dist['category_code'].apply(
        lambda x: CATEGORY_MAPPING.get(str(x), ("", "", "❓"))[2]
    )

    # Calculer les pourcentages
    total = dist['count'].sum()
    dist['percentage'] = (dist['count'] / total * 100).round(2)

    return dist.sort_values('count', ascending=False).reset_index(drop=True)


def get_text_statistics(X_train: Optional[pd.DataFrame] = None) -> Dict:
    """
    Calcule les statistiques sur les textes.

    Returns:
        Dict avec les statistiques texte
    """
    if X_train is not None:
        # Statistiques réelles
        designation_lengths = X_train['designation'].str.len()
        description_lengths = X_train['description'].fillna('').str.len()

        # Langues (si disponible)
        if 'lang' in X_train.columns:
            lang_dist = X_train['lang'].value_counts().to_dict()
        else:
            lang_dist = None

        return {
            "total_products": len(X_train),
            "designation": {
                "mean_length": designation_lengths.mean(),
                "min_length": designation_lengths.min(),
                "max_length": designation_lengths.max(),
                "median_length": designation_lengths.median(),
            },
            "description": {
                "mean_length": description_lengths.mean(),
                "min_length": description_lengths.min(),
                "max_length": description_lengths.max(),
                "non_empty_pct": (description_lengths > 0).mean() * 100,
            },
            "languages": lang_dist,
            "is_mock": False
        }
    else:
        # Statistiques mock réalistes
        return {
            "total_products": 84916,
            "designation": {
                "mean_length": 48.5,
                "min_length": 3,
                "max_length": 250,
                "median_length": 42,
            },
            "description": {
                "mean_length": 312.8,
                "min_length": 0,
                "max_length": 5000,
                "non_empty_pct": 78.5,
            },
            "languages": {
                "fr": 72500,
                "en": 8200,
                "de": 2100,
                "es": 1200,
                "it": 600,
                "other": 316
            },
            "is_mock": True
        }


def get_sample_products(
    X_train: Optional[pd.DataFrame] = None,
    Y_train: Optional[pd.DataFrame] = None,
    category_code: str = None,
    n_samples: int = 5
) -> pd.DataFrame:
    """
    Retourne des exemples de produits.

    Args:
        X_train: DataFrame des features
        Y_train: DataFrame des labels
        category_code: Code catégorie pour filtrer (optionnel)
        n_samples: Nombre d'exemples à retourner
    """
    if X_train is not None and Y_train is not None:
        # Données réelles
        data = X_train.join(Y_train)
        if category_code:
            data = data[data['prdtypecode'] == int(category_code)]
        return data.sample(min(n_samples, len(data)))
    else:
        # Données mock
        mock_products = [
            {"designation": "Harry Potter à l'école des sorciers", "description": "Roman fantastique de J.K. Rowling", "prdtypecode": "10"},
            {"designation": "Console PlayStation 5", "description": "Console de jeux nouvelle génération Sony", "prdtypecode": "60"},
            {"designation": "Piscine gonflable ronde 3m", "description": "Piscine familiale pour jardin", "prdtypecode": "2583"},
            {"designation": "Figurine Dragon Ball Z Goku", "description": "Figurine collector 25cm", "prdtypecode": "1140"},
            {"designation": "Canapé convertible 3 places", "description": "Canapé-lit confortable gris", "prdtypecode": "1560"},
            {"designation": "Lot cartes Pokemon Pikachu", "description": "Collection 50 cartes Pokemon rares", "prdtypecode": "1160"},
            {"designation": "Tondeuse à gazon électrique", "description": "Tondeuse 1800W coupe 42cm", "prdtypecode": "2582"},
            {"designation": "Puzzle 1000 pièces paysage", "description": "Puzzle adulte montagne", "prdtypecode": "1281"},
            {"designation": "Croquettes chien adulte 15kg", "description": "Alimentation premium pour chien", "prdtypecode": "2220"},
            {"designation": "Perceuse visseuse sans fil", "description": "Perceuse 18V avec 2 batteries", "prdtypecode": "2585"},
        ]
        df = pd.DataFrame(mock_products)
        if category_code:
            df = df[df['prdtypecode'] == category_code]
        return df.head(n_samples)


def get_dataset_summary() -> Dict:
    """Retourne un résumé du dataset."""
    X_train, Y_train = load_training_data()

    if X_train is not None:
        return {
            "train_samples": len(X_train),
            "test_samples": len(load_test_data()) if load_test_data() is not None else "N/A",
            "num_categories": 27,
            "features": ["designation", "description", "productid", "imageid"],
            "is_mock": False
        }
    else:
        return {
            "train_samples": 84916,
            "test_samples": 13812,
            "num_categories": 27,
            "features": ["designation", "description", "productid", "imageid"],
            "is_mock": True
        }
