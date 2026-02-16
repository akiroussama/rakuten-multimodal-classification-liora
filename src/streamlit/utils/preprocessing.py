"""
Prétraitement du texte pour la classification de produits Rakuten.

Ce module fournit des fonctions de nettoyage et préparation du texte,
basées sur le pipeline NLP développé par l'équipe dans build_features_nlp.py.

Étapes de prétraitement:
1. Nettoyage HTML (unescape, suppression tags)
2. Normalisation (lowercase, suppression caractères spéciaux)
3. Gestion des nombres
4. Concaténation designation + description
"""
import re
import html
from typing import Optional, Tuple
import unicodedata


def clean_text(text: str) -> str:
    """
    Nettoie un texte brut pour la classification.

    Applique les transformations suivantes:
    - Décodage des entités HTML
    - Suppression des balises HTML
    - Conversion en minuscules
    - Normalisation des espaces
    - Suppression des caractères spéciaux excessifs

    Args:
        text: Texte brut à nettoyer

    Returns:
        Texte nettoyé
    """
    if not text or not isinstance(text, str):
        return ""

    # Décoder les entités HTML (&amp; -> &, etc.)
    text = html.unescape(text)

    # Supprimer les balises HTML
    text = _remove_html_tags(text)

    # Convertir en minuscules
    text = text.lower()

    # Normaliser les espaces
    text = _normalize_whitespace(text)

    # Supprimer les accents problématiques (optionnel, configurable)
    # text = _remove_accents(text)

    return text.strip()


def _remove_html_tags(text: str) -> str:
    """
    Supprime les balises HTML d'un texte.

    Args:
        text: Texte avec potentiellement des balises HTML

    Returns:
        Texte sans balises HTML
    """
    # Pattern pour matcher les balises HTML
    html_pattern = re.compile(r'<[^>]+>')
    text = html_pattern.sub(' ', text)

    # Supprimer les commentaires HTML
    comment_pattern = re.compile(r'<!--.*?-->', re.DOTALL)
    text = comment_pattern.sub(' ', text)

    return text


def _normalize_whitespace(text: str) -> str:
    """
    Normalise les espaces dans un texte.

    Args:
        text: Texte avec espaces irréguliers

    Returns:
        Texte avec espaces normalisés
    """
    # Remplacer les caractères de contrôle par des espaces
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', ' ', text)

    # Normaliser tous les types d'espaces
    text = re.sub(r'\s+', ' ', text)

    return text


def _remove_accents(text: str) -> str:
    """
    Supprime les accents d'un texte.

    Note: Cette fonction est optionnelle et peut affecter
    la qualité de la classification pour le français.

    Args:
        text: Texte avec accents

    Returns:
        Texte sans accents
    """
    nfkd_form = unicodedata.normalize('NFKD', text)
    return ''.join(c for c in nfkd_form if not unicodedata.combining(c))


def preprocess_product_text(
    designation: str,
    description: Optional[str] = None,
    remove_numbers: bool = False
) -> str:
    """
    Prétraite le texte complet d'un produit Rakuten.

    Combine et nettoie la désignation et la description du produit
    selon le pipeline établi par l'équipe NLP.

    Args:
        designation: Titre/nom du produit (obligatoire)
        description: Description détaillée (optionnel)
        remove_numbers: Si True, supprime les tokens contenant des chiffres

    Returns:
        Texte nettoyé et combiné prêt pour la vectorisation

    Example:
        >>> text = preprocess_product_text(
        ...     "Livre Harry Potter",
        ...     "Roman fantastique pour enfants"
        ... )
        >>> print(text)
        "livre harry potter . -//- roman fantastique pour enfants"
    """
    # Nettoyer la désignation
    clean_designation = clean_text(designation or "")

    # Nettoyer la description
    clean_description = clean_text(description or "")

    # Appliquer les remplacements spécifiques du pipeline Rakuten
    clean_designation = _apply_rakuten_replacements(clean_designation)
    clean_description = _apply_rakuten_replacements(clean_description)

    # Supprimer les mots avec chiffres si demandé
    if remove_numbers:
        clean_designation = _remove_words_with_numbers(clean_designation)
        clean_description = _remove_words_with_numbers(clean_description)

    # Combiner avec le séparateur du pipeline original
    if clean_description:
        combined = f"{clean_designation} . -//- {clean_description}"
    else:
        combined = clean_designation

    return combined.strip()


def _apply_rakuten_replacements(text: str) -> str:
    """
    Applique les remplacements spécifiques au pipeline Rakuten.

    Basé sur le code de build_features_nlp.py.

    Args:
        text: Texte à transformer

    Returns:
        Texte avec remplacements appliqués
    """
    # Remplacer n° par numéro
    text = re.sub(r'n°', ' numéro ', text)

    # Ajouter des espaces autour de la ponctuation
    # (sauf pour certains caractères spéciaux)
    text = re.sub(r"[^\d\w\s¿?'\-]", r' \g<0> ', text)

    # Normaliser les espaces après les remplacements
    text = _normalize_whitespace(text)

    return text


def _remove_words_with_numbers(text: str) -> str:
    """
    Supprime les mots contenant des chiffres.

    Args:
        text: Texte à traiter

    Returns:
        Texte sans les mots contenant des chiffres
    """
    pattern = r'\b\S*[0-9]+\S*\b'
    return re.sub(pattern, '', text)


def detect_language_simple(text: str) -> str:
    """
    Détection simplifiée de la langue (heuristique).

    Pour une détection plus précise, utiliser langid.
    Cette fonction est un fallback rapide.

    Args:
        text: Texte à analyser

    Returns:
        Code langue probable ('fr', 'en', 'de', etc.) ou 'unknown'
    """
    if not text or len(text) < 10:
        return "unknown"

    text_lower = text.lower()

    # Mots indicateurs par langue
    french_indicators = [
        'le', 'la', 'les', 'de', 'du', 'des', 'un', 'une', 'et', 'est',
        'pour', 'avec', 'dans', 'sur', 'par', 'qui', 'que', 'ce', 'cette'
    ]
    english_indicators = [
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
        'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with'
    ]
    german_indicators = [
        'der', 'die', 'das', 'ein', 'eine', 'und', 'ist', 'sind',
        'für', 'mit', 'auf', 'bei', 'nach', 'von', 'zu'
    ]

    words = set(re.findall(r'\b\w+\b', text_lower))

    fr_score = len(words & set(french_indicators))
    en_score = len(words & set(english_indicators))
    de_score = len(words & set(german_indicators))

    scores = {'fr': fr_score, 'en': en_score, 'de': de_score}
    best_lang = max(scores, key=scores.get)

    if scores[best_lang] >= 2:
        return best_lang
    return "unknown"


def validate_text_input(
    designation: str,
    description: Optional[str] = None,
    min_length: int = 3,
    max_length: int = 5000
) -> Tuple[bool, str]:
    """
    Valide les entrées texte d'un produit.

    Args:
        designation: Titre du produit
        description: Description du produit
        min_length: Longueur minimale de la désignation
        max_length: Longueur maximale totale

    Returns:
        Tuple (is_valid, error_message)
    """
    if not designation or not designation.strip():
        return False, "La désignation du produit est obligatoire"

    if len(designation.strip()) < min_length:
        return False, f"La désignation doit contenir au moins {min_length} caractères"

    total_length = len(designation) + len(description or "")
    if total_length > max_length:
        return False, f"Le texte total ne doit pas dépasser {max_length} caractères"

    return True, ""
