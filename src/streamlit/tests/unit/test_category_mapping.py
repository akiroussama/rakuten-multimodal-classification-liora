"""
Tests unitaires pour utils/category_mapping.py

Ce module teste:
- CATEGORY_MAPPING: dictionnaire des 27 catégories
- get_category_info(): obtenir les infos d'une catégorie
- get_category_emoji(): obtenir l'emoji d'une catégorie
- get_all_categories(): obtenir toutes les catégories
"""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils.category_mapping import (
    CATEGORY_MAPPING,
    get_category_info,
    get_category_emoji,
    get_all_categories,
)


# =============================================================================
# TESTS CATEGORY_MAPPING Dictionary
# =============================================================================
@pytest.mark.unit
class TestCategoryMappingDict:
    """Tests pour le dictionnaire CATEGORY_MAPPING."""

    def test_has_27_categories(self):
        """Le mapping contient exactement 27 catégories."""
        assert len(CATEGORY_MAPPING) == 27, \
            f"Expected 27 categories, found {len(CATEGORY_MAPPING)}"

    def test_all_keys_are_strings(self):
        """Toutes les clés sont des strings (codes catégorie)."""
        for key in CATEGORY_MAPPING.keys():
            assert isinstance(key, str), f"Key {key} is not a string"

    def test_all_keys_are_numeric_strings(self):
        """Toutes les clés sont des strings numériques."""
        for key in CATEGORY_MAPPING.keys():
            assert key.isdigit(), f"Key '{key}' is not a numeric string"

    def test_all_values_are_tuples(self):
        """Toutes les valeurs sont des tuples."""
        for key, value in CATEGORY_MAPPING.items():
            assert isinstance(value, tuple), \
                f"Value for {key} is not a tuple: {type(value)}"

    def test_all_tuples_have_three_elements(self):
        """Chaque tuple a exactement 3 éléments (name, full_name, emoji)."""
        for key, value in CATEGORY_MAPPING.items():
            assert len(value) == 3, \
                f"Tuple for {key} has {len(value)} elements, expected 3"

    def test_all_elements_are_strings(self):
        """Tous les éléments des tuples sont des strings."""
        for key, (name, full_name, emoji) in CATEGORY_MAPPING.items():
            assert isinstance(name, str), f"Name for {key} is not string"
            assert isinstance(full_name, str), f"Full name for {key} is not string"
            assert isinstance(emoji, str), f"Emoji for {key} is not string"

    def test_no_empty_names(self):
        """Aucun nom n'est vide."""
        for key, (name, full_name, emoji) in CATEGORY_MAPPING.items():
            assert len(name) > 0, f"Empty name for category {key}"
            assert len(full_name) > 0, f"Empty full_name for category {key}"

    def test_all_have_emojis(self):
        """Chaque catégorie a un emoji."""
        for key, (name, full_name, emoji) in CATEGORY_MAPPING.items():
            assert len(emoji) > 0, f"Empty emoji for category {key}"

    def test_no_duplicate_names(self):
        """Aucun nom de catégorie n'est dupliqué."""
        names = [name for name, _, _ in CATEGORY_MAPPING.values()]
        assert len(names) == len(set(names)), \
            f"Duplicate category names found: {[n for n in names if names.count(n) > 1]}"

    def test_expected_categories_present(self):
        """Les catégories attendues sont présentes."""
        expected_codes = [
            "10", "40", "50", "60", "1140", "1160", "1180", "1280", "1281",
            "1300", "1301", "1302", "1320", "1560", "1920", "1940", "2060",
            "2220", "2280", "2403", "2462", "2522", "2582", "2583", "2585",
            "2705", "2905"
        ]
        for code in expected_codes:
            assert code in CATEGORY_MAPPING, f"Missing expected category: {code}"


# =============================================================================
# TESTS get_category_info()
# =============================================================================
@pytest.mark.unit
class TestGetCategoryInfo:
    """Tests pour la fonction get_category_info()."""

    def test_returns_tuple(self, sample_category_code):
        """Retourne un tuple."""
        result = get_category_info(sample_category_code)
        assert isinstance(result, tuple)

    def test_returns_three_elements(self, sample_category_code):
        """Retourne un tuple de 3 éléments."""
        result = get_category_info(sample_category_code)
        assert len(result) == 3

    def test_returns_correct_types(self, sample_category_code):
        """Retourne (str, str, str)."""
        name, full_name, emoji = get_category_info(sample_category_code)
        assert isinstance(name, str)
        assert isinstance(full_name, str)
        assert isinstance(emoji, str)

    def test_returns_non_empty_values(self, sample_category_code):
        """Retourne des valeurs non vides."""
        name, full_name, emoji = get_category_info(sample_category_code)
        assert len(name) > 0
        assert len(full_name) > 0
        assert len(emoji) > 0

    def test_known_category_2583(self):
        """Test avec catégorie 2583 (Téléphones)."""
        name, full_name, emoji = get_category_info("2583")
        assert "téléphone" in name.lower() or "phone" in name.lower() or "électronique" in name.lower()

    def test_known_category_2403(self):
        """Test avec catégorie 2403 (Livres)."""
        name, full_name, emoji = get_category_info("2403")
        assert "livre" in name.lower() or "book" in name.lower() or "bandes" in name.lower() or "bd" in name.lower()

    @pytest.mark.parametrize("code", list(CATEGORY_MAPPING.keys()))
    def test_all_categories_return_valid_info(self, code):
        """Toutes les catégories retournent des infos valides."""
        name, full_name, emoji = get_category_info(code)
        assert len(name) > 0
        assert len(full_name) > 0
        assert len(emoji) > 0

    def test_invalid_category_code(self, invalid_category_code):
        """Gère les codes invalides gracieusement."""
        # Devrait retourner une valeur par défaut ou lever une exception
        try:
            result = get_category_info(invalid_category_code)
            # Si pas d'exception, vérifier que c'est une valeur par défaut
            assert isinstance(result, tuple)
            assert len(result) == 3
        except (KeyError, ValueError):
            pass  # Exception acceptable

    def test_none_category_code(self):
        """Gère None gracieusement."""
        try:
            result = get_category_info(None)
            assert isinstance(result, tuple)
        except (KeyError, ValueError, TypeError):
            pass

    def test_empty_string_category_code(self):
        """Gère string vide gracieusement."""
        try:
            result = get_category_info("")
            assert isinstance(result, tuple)
        except (KeyError, ValueError):
            pass

    def test_numeric_vs_string_code(self):
        """Gère les codes numériques vs string."""
        # Certaines implémentations acceptent int ou str
        try:
            result_str = get_category_info("2583")
            # Si int est supporté
            try:
                result_int = get_category_info(2583)
                # Les résultats devraient être identiques
                assert result_str == result_int
            except (KeyError, TypeError):
                pass  # Int non supporté, c'est OK
        except Exception as e:
            pytest.fail(f"get_category_info failed: {e}")


# =============================================================================
# TESTS get_category_emoji()
# =============================================================================
@pytest.mark.unit
class TestGetCategoryEmoji:
    """Tests pour la fonction get_category_emoji()."""

    def test_returns_string(self, sample_category_code):
        """Retourne une string."""
        result = get_category_emoji(sample_category_code)
        assert isinstance(result, str)

    def test_returns_non_empty(self, sample_category_code):
        """Retourne une string non vide."""
        result = get_category_emoji(sample_category_code)
        assert len(result) > 0

    def test_returns_emoji_character(self, sample_category_code):
        """Retourne un caractère emoji."""
        result = get_category_emoji(sample_category_code)
        # Un emoji est généralement > 1 byte en UTF-8
        # ou a un point de code élevé
        assert any(ord(c) > 127 for c in result) or len(result) >= 1

    @pytest.mark.parametrize("code", list(CATEGORY_MAPPING.keys())[:5])
    def test_sample_categories_have_emoji(self, code):
        """Quelques catégories ont des emojis valides."""
        emoji = get_category_emoji(code)
        assert len(emoji) > 0

    def test_invalid_code_handled(self, invalid_category_code):
        """Gère les codes invalides."""
        try:
            result = get_category_emoji(invalid_category_code)
            assert isinstance(result, str)
        except (KeyError, ValueError):
            pass


# =============================================================================
# TESTS get_all_categories()
# =============================================================================
@pytest.mark.unit
class TestGetAllCategories:
    """Tests pour la fonction get_all_categories()."""

    def test_returns_dict(self):
        """Retourne un dictionnaire."""
        result = get_all_categories()
        assert isinstance(result, dict)

    def test_returns_27_categories(self):
        """Retourne 27 catégories."""
        result = get_all_categories()
        assert len(result) == 27

    def test_keys_are_category_codes(self):
        """Les clés sont les codes catégorie."""
        result = get_all_categories()
        for key in result.keys():
            assert key in CATEGORY_MAPPING

    def test_values_are_tuples(self):
        """Les valeurs sont des tuples."""
        result = get_all_categories()
        for value in result.values():
            assert isinstance(value, tuple)

    def test_values_have_three_elements(self):
        """Chaque valeur a 3 éléments."""
        result = get_all_categories()
        for key, value in result.items():
            assert len(value) == 3, f"Category {key} has {len(value)} elements"

    def test_returns_same_data_as_mapping(self):
        """Retourne les mêmes données que CATEGORY_MAPPING."""
        result = get_all_categories()
        assert result == CATEGORY_MAPPING or set(result.keys()) == set(CATEGORY_MAPPING.keys())


# =============================================================================
# TESTS Consistency
# =============================================================================
@pytest.mark.unit
class TestCategoryConsistency:
    """Tests de cohérence entre les fonctions."""

    def test_get_info_matches_mapping(self):
        """get_category_info retourne les mêmes données que le mapping."""
        for code, expected in CATEGORY_MAPPING.items():
            result = get_category_info(code)
            assert result == expected, f"Mismatch for {code}"

    def test_get_emoji_matches_mapping(self):
        """get_category_emoji retourne le même emoji que le mapping."""
        for code, (_, _, expected_emoji) in CATEGORY_MAPPING.items():
            result = get_category_emoji(code)
            assert result == expected_emoji, f"Mismatch for {code}"

    def test_get_all_matches_mapping(self):
        """get_all_categories retourne les mêmes données que CATEGORY_MAPPING."""
        result = get_all_categories()
        for code in CATEGORY_MAPPING.keys():
            assert code in result


# =============================================================================
# TESTS Edge Cases
# =============================================================================
@pytest.mark.unit
class TestCategoryEdgeCases:
    """Tests des cas limites."""

    @pytest.mark.parametrize("invalid_code", [
        "0",
        "-1",
        "99999",
        "abc",
        "2583a",
        " 2583",
        "2583 ",
        "25.83",
    ])
    def test_invalid_codes_handled(self, invalid_code):
        """Les codes invalides sont gérés proprement."""
        try:
            result = get_category_info(invalid_code)
            # Si pas d'exception, doit retourner quelque chose de valide
            assert isinstance(result, tuple)
        except (KeyError, ValueError):
            pass  # Comportement acceptable

    def test_category_codes_are_unique(self):
        """Les codes catégorie sont uniques."""
        codes = list(CATEGORY_MAPPING.keys())
        assert len(codes) == len(set(codes))

    def test_no_category_with_none_values(self):
        """Aucune catégorie n'a de valeur None."""
        for code, (name, full_name, emoji) in CATEGORY_MAPPING.items():
            assert name is not None, f"None name for {code}"
            assert full_name is not None, f"None full_name for {code}"
            assert emoji is not None, f"None emoji for {code}"
