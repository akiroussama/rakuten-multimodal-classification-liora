"""
Mapping des catégories de produits Rakuten.

Ce module contient les correspondances entre les codes de catégories
numériques et leurs descriptions lisibles en français.

Les 27 catégories couvrent différents types de produits vendus sur Rakuten:
- Livres et médias (livres, magazines, BD, etc.)
- Jeux et jouets (jeux vidéo, figurines, etc.)
- Équipement maison (piscine, mobilier jardin, etc.)
- High-tech (accessoires gaming, etc.)
"""
from typing import Dict, Optional, Tuple

# =============================================================================
# Mapping des 27 catégories Rakuten
# =============================================================================
# Format: code -> (nom_court, nom_complet, emoji)
CATEGORY_MAPPING: Dict[str, Tuple[str, str, str]] = {
    "10": ("Livres", "Livres occasion", "📚"),
    "40": ("Jeux vidéo", "Jeux vidéo, consoles et accessoires", "🎮"),
    "50": ("Gaming PC", "Accessoires gaming PC", "🖥️"),
    "60": ("Consoles", "Consoles de jeux vidéo", "🕹️"),
    "1140": ("Figurines", "Figurines et objets de collection", "🎭"),
    "1160": ("Cartes", "Cartes de collection (Pokemon, Magic, etc.)", "🃏"),
    "1180": ("Figurines pop", "Figurines de jeux et mangas", "🦸"),
    "1280": ("Jouets enfants", "Jouets et jeux pour enfants", "🧸"),
    "1281": ("Jeux société", "Jeux de société et puzzles", "🎲"),
    "1300": ("Modélisme", "Modélisme et miniatures", "✈️"),
    "1301": ("Loisirs créatifs", "Loisirs créatifs et bricolage enfant", "🎨"),
    "1302": ("Déguisements", "Déguisements et accessoires de fête", "🎃"),
    "1320": ("Puériculture", "Équipement bébé et puériculture", "👶"),
    "1560": ("Mobilier", "Mobilier intérieur", "🪑"),
    "1920": ("Literie", "Literie et linge de maison", "🛏️"),
    "1940": ("Alimentation", "Épicerie et alimentation", "🍽️"),
    "2060": ("Déco maison", "Décoration intérieure", "🏠"),
    "2220": ("Animalerie", "Produits pour animaux", "🐾"),
    "2280": ("Magazines", "Magazines et revues", "📰"),
    "2403": ("Livres neufs", "Livres, BD, magazines neufs", "📖"),
    "2462": ("Jeux PC", "Jeux vidéo PC en téléchargement", "💿"),
    "2522": ("Papeterie", "Fournitures de bureau et papeterie", "✏️"),
    "2582": ("Jardin", "Mobilier et équipement de jardin", "🌳"),
    "2583": ("Piscine", "Piscines et accessoires", "🏊"),
    "2585": ("Bricolage", "Outillage et bricolage", "🔧"),
    "2705": ("Livres anciens", "Livres anciens et de collection", "📜"),
    "2905": ("Jeux PC box", "Jeux vidéo PC en boîte", "📦"),
}

# Dictionnaire simplifié: code -> nom court
CATEGORY_NAMES: Dict[str, str] = {
    code: info[0] for code, info in CATEGORY_MAPPING.items()
}

# Liste ordonnée des codes (même ordre que metadata_augmented.json)
CATEGORY_CODES = [
    "10", "40", "50", "60", "1140", "1160", "1180", "1280", "1281",
    "1300", "1301", "1302", "1320", "1560", "1920", "1940", "2060",
    "2220", "2280", "2403", "2462", "2522", "2582", "2583", "2585",
    "2705", "2905"
]

# Index pour conversion rapide code <-> index
CODE_TO_INDEX = {code: idx for idx, code in enumerate(CATEGORY_CODES)}
INDEX_TO_CODE = {idx: code for idx, code in enumerate(CATEGORY_CODES)}


def get_category_name(code: str) -> str:
    """
    Retourne le nom court d'une catégorie à partir de son code.

    Args:
        code: Code de catégorie (ex: "2583")

    Returns:
        Nom court de la catégorie (ex: "Piscine")
        Retourne "Catégorie inconnue" si le code n'existe pas
    """
    return CATEGORY_NAMES.get(str(code), "Catégorie inconnue")


def get_category_info(code: str) -> Tuple[str, str, str]:
    """
    Retourne les informations complètes d'une catégorie.

    Args:
        code: Code de catégorie (ex: "2583")

    Returns:
        Tuple (nom_court, nom_complet, emoji)
        Retourne ("Inconnu", "Catégorie inconnue", "❓") si le code n'existe pas
    """
    return CATEGORY_MAPPING.get(
        str(code),
        ("Inconnu", "Catégorie inconnue", "❓")
    )


def get_category_emoji(code: str) -> str:
    """
    Retourne l'emoji associé à une catégorie.

    Args:
        code: Code de catégorie

    Returns:
        Emoji de la catégorie
    """
    return get_category_info(code)[2]


def get_all_categories() -> Dict[str, Tuple[str, str, str]]:
    """
    Retourne le dictionnaire complet des catégories.

    Returns:
        Dict avec code -> (nom_court, nom_complet, emoji)
    """
    return CATEGORY_MAPPING.copy()


def code_to_index(code: str) -> Optional[int]:
    """
    Convertit un code de catégorie en index (0-26).

    Args:
        code: Code de catégorie

    Returns:
        Index correspondant ou None si code invalide
    """
    return CODE_TO_INDEX.get(str(code))


def index_to_code(index: int) -> Optional[str]:
    """
    Convertit un index (0-26) en code de catégorie.

    Args:
        index: Index de la catégorie

    Returns:
        Code correspondant ou None si index invalide
    """
    return INDEX_TO_CODE.get(index)
