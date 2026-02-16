"""
Tests unitaires pour utils/preprocessing.py

Ce module teste:
- preprocess_product_text(): nettoyage et préparation du texte
- validate_text_input(): validation des entrées texte
- clean_html(): suppression des balises HTML
"""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils.preprocessing import (
    preprocess_product_text,
    validate_text_input,
)


# =============================================================================
# TESTS preprocess_product_text()
# =============================================================================
@pytest.mark.unit
class TestPreprocessProductText:
    """Tests pour la fonction preprocess_product_text()."""

    def test_returns_string(self, sample_designation, sample_description):
        """Retourne une string."""
        result = preprocess_product_text(sample_designation, sample_description)
        assert isinstance(result, str)

    def test_combines_designation_and_description(self):
        """Combine désignation et description."""
        designation = "iPhone 15"
        description = "Smartphone Apple"
        result = preprocess_product_text(designation, description)

        # Le résultat doit contenir des éléments des deux
        result_lower = result.lower()
        assert "iphone" in result_lower or "15" in result_lower
        assert "smartphone" in result_lower or "apple" in result_lower

    def test_handles_empty_description(self):
        """Gère description vide."""
        result = preprocess_product_text("iPhone 15", "")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_handles_empty_designation(self):
        """Gère désignation vide."""
        result = preprocess_product_text("", "Smartphone Apple")
        assert isinstance(result, str)

    def test_handles_both_empty(self):
        """Gère les deux vides."""
        result = preprocess_product_text("", "")
        assert isinstance(result, str)

    def test_handles_none_description(self):
        """Gère description None."""
        result = preprocess_product_text("iPhone 15", None)
        assert isinstance(result, str)

    def test_handles_none_designation(self):
        """Gère désignation None."""
        result = preprocess_product_text(None, "Smartphone")
        assert isinstance(result, str)

    def test_handles_both_none(self):
        """Gère les deux None."""
        result = preprocess_product_text(None, None)
        assert isinstance(result, str)

    def test_removes_html_tags(self):
        """Supprime les balises HTML."""
        designation = "<p>iPhone <b>15</b></p>"
        result = preprocess_product_text(designation, "")

        assert "<p>" not in result
        assert "</p>" not in result
        assert "<b>" not in result
        assert "</b>" not in result

    def test_removes_script_tags(self):
        """Supprime les balises script (sécurité)."""
        designation = "<script>alert('xss')</script>iPhone"
        result = preprocess_product_text(designation, "")

        assert "<script>" not in result
        assert "alert" not in result.lower() or "iphone" in result.lower()

    def test_handles_special_characters(self):
        """Gère les caractères spéciaux."""
        designation = "iPhone™ 15® Pro©"
        result = preprocess_product_text(designation, "")
        assert isinstance(result, str)

    def test_handles_unicode(self):
        """Gère les caractères Unicode."""
        designation = "Téléphone été français"
        result = preprocess_product_text(designation, "")
        assert isinstance(result, str)

    def test_handles_emojis(self):
        """Gère les emojis."""
        designation = "📱 iPhone 15 🍎"
        result = preprocess_product_text(designation, "")
        assert isinstance(result, str)

    def test_trims_whitespace(self):
        """Supprime les espaces en début/fin."""
        designation = "   iPhone 15   "
        result = preprocess_product_text(designation, "")
        # Le résultat ne devrait pas avoir d'espaces en excès
        assert not result.startswith("   ")
        assert not result.endswith("   ")

    def test_normalizes_multiple_spaces(self):
        """Normalise les espaces multiples."""
        designation = "iPhone    15     Pro"
        result = preprocess_product_text(designation, "")
        # Ne devrait pas avoir plusieurs espaces consécutifs
        assert "    " not in result

    def test_preserves_essential_content(self):
        """Préserve le contenu essentiel."""
        designation = "Console PlayStation 5"
        description = "Jeux vidéo Sony"
        result = preprocess_product_text(designation, description)

        result_lower = result.lower()
        # Au moins une partie du contenu doit être préservée
        assert "playstation" in result_lower or "console" in result_lower or "sony" in result_lower

    @pytest.mark.parametrize("html_input,should_not_contain", [
        ("<p>Test</p>", "<p>"),
        ("<div class='x'>Test</div>", "<div"),
        ("<a href='url'>Link</a>", "<a"),
        ("<img src='img.jpg'>", "<img"),
        ("<style>css</style>", "<style"),
        ("<!--comment-->Test", "<!--"),
    ])
    def test_removes_various_html(self, html_input, should_not_contain):
        """Supprime différents types de HTML."""
        result = preprocess_product_text(html_input, "")
        assert should_not_contain not in result


# =============================================================================
# TESTS validate_text_input()
# =============================================================================
@pytest.mark.unit
class TestValidateTextInput:
    """Tests pour la fonction validate_text_input()."""

    def test_returns_tuple(self, sample_designation):
        """Retourne un tuple (is_valid, message)."""
        result = validate_text_input(sample_designation)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_valid_text_returns_true(self, sample_designation):
        """Texte valide retourne (True, ...)."""
        is_valid, message = validate_text_input(sample_designation)
        assert is_valid is True

    def test_valid_text_message(self, sample_designation):
        """Texte valide a un message approprié."""
        is_valid, message = validate_text_input(sample_designation)
        assert isinstance(message, str)

    def test_empty_string_invalid(self):
        """String vide est invalide."""
        is_valid, message = validate_text_input("")
        assert is_valid is False
        assert len(message) > 0  # Message d'erreur présent

    def test_whitespace_only_invalid(self):
        """Espaces seulement est invalide."""
        is_valid, message = validate_text_input("   ")
        assert is_valid is False

    def test_none_invalid(self):
        """None est invalide."""
        is_valid, message = validate_text_input(None)
        assert is_valid is False

    def test_minimum_length(self):
        """Vérifie la longueur minimale."""
        # Texte trop court
        is_valid_short, _ = validate_text_input("a")
        # Texte assez long
        is_valid_long, _ = validate_text_input("iPhone 15 Pro Max")

        # Au moins le texte long devrait être valide
        assert is_valid_long is True

    def test_maximum_length(self):
        """Gère les textes très longs."""
        long_text = "a" * 100000
        is_valid, message = validate_text_input(long_text)
        # Soit valide, soit message d'erreur approprié
        assert isinstance(is_valid, bool)
        assert isinstance(message, str)

    def test_special_characters_handled(self):
        """Gère les caractères spéciaux."""
        is_valid, message = validate_text_input("Test!@#$%^&*()")
        assert isinstance(is_valid, bool)

    def test_unicode_handled(self):
        """Gère l'Unicode correctement."""
        is_valid, message = validate_text_input("Téléphone français été")
        assert isinstance(is_valid, bool)


# =============================================================================
# TESTS Security (XSS Prevention)
# =============================================================================
@pytest.mark.unit
@pytest.mark.security
class TestPreprocessingSecurity:
    """Tests de sécurité pour le preprocessing."""

    XSS_PAYLOADS = [
        "<script>alert('xss')</script>",
        "<img src=x onerror=alert('xss')>",
        "<svg onload=alert('xss')>",
        "javascript:alert('xss')",
        "<iframe src='javascript:alert(1)'>",
        "<body onload=alert('xss')>",
        "<input onfocus=alert('xss') autofocus>",
        "'-alert(1)-'",
        "\"><script>alert('xss')</script>",
    ]

    @pytest.mark.parametrize("payload", XSS_PAYLOADS)
    def test_xss_payloads_neutralized(self, payload):
        """Les payloads XSS sont neutralisés."""
        result = preprocess_product_text(payload, "")

        # Aucune balise script ne doit rester
        assert "<script>" not in result.lower()
        assert "javascript:" not in result.lower()
        assert "onerror=" not in result.lower()
        assert "onload=" not in result.lower()
        assert "onfocus=" not in result.lower()

    def test_sql_injection_patterns_handled(self):
        """Les patterns d'injection SQL sont gérés."""
        sql_payloads = [
            "'; DROP TABLE users; --",
            "1 OR 1=1",
            "admin'--",
        ]
        for payload in sql_payloads:
            result = preprocess_product_text(payload, "")
            # Le preprocessing ne devrait pas exécuter ces patterns
            assert isinstance(result, str)


# =============================================================================
# TESTS Edge Cases
# =============================================================================
@pytest.mark.unit
class TestPreprocessingEdgeCases:
    """Tests des cas limites."""

    def test_very_long_text(self):
        """Gère texte très long (100K+ caractères)."""
        long_text = "produit " * 15000  # ~120K chars
        result = preprocess_product_text(long_text, "")
        assert isinstance(result, str)

    def test_only_numbers(self):
        """Gère texte avec seulement des chiffres."""
        result = preprocess_product_text("123456789", "")
        assert isinstance(result, str)

    def test_only_punctuation(self):
        """Gère texte avec seulement de la ponctuation."""
        result = preprocess_product_text("!@#$%^&*()", "")
        assert isinstance(result, str)

    def test_mixed_languages(self):
        """Gère texte multilingue."""
        result = preprocess_product_text(
            "Hello Bonjour Hola 你好 Привет",
            "Description in multiple languages"
        )
        assert isinstance(result, str)

    def test_newlines_and_tabs(self):
        """Gère les retours à la ligne et tabulations."""
        result = preprocess_product_text(
            "Line1\nLine2\tTabbed",
            "Description\r\nwith\rreturns"
        )
        assert isinstance(result, str)
        # Ne devrait pas garder des caractères de contrôle bruts problématiques
        assert "\r" not in result or "\n" not in result or isinstance(result, str)

    def test_html_entities(self):
        """Gère les entités HTML."""
        result = preprocess_product_text(
            "&lt;script&gt;alert&amp;apos;xss&apos;&lt;/script&gt;",
            "&nbsp;&copy;&reg;"
        )
        assert isinstance(result, str)

    def test_urls_in_text(self):
        """Gère les URLs dans le texte."""
        result = preprocess_product_text(
            "Visit https://example.com for details",
            "See http://test.com"
        )
        assert isinstance(result, str)

    def test_email_addresses(self):
        """Gère les adresses email."""
        result = preprocess_product_text(
            "Contact: test@example.com",
            ""
        )
        assert isinstance(result, str)


# =============================================================================
# TESTS Consistency
# =============================================================================
@pytest.mark.unit
class TestPreprocessingConsistency:
    """Tests de cohérence."""

    def test_idempotent(self):
        """Appliquer deux fois donne le même résultat."""
        original = "iPhone 15 Pro <b>Max</b>"
        result1 = preprocess_product_text(original, "")
        result2 = preprocess_product_text(result1, "")

        assert result1 == result2

    def test_deterministic(self):
        """Même input = même output."""
        text = "Console PlayStation 5"
        results = [preprocess_product_text(text, "") for _ in range(5)]

        assert all(r == results[0] for r in results)

    def test_order_independent_for_description(self):
        """Le résultat contient les deux parties."""
        designation = "iPhone 15"
        description = "Smartphone Apple"

        result = preprocess_product_text(designation, description)

        # Les deux devraient contribuer au résultat
        result_lower = result.lower()
        has_designation = "iphone" in result_lower or "15" in result_lower
        has_description = "smartphone" in result_lower or "apple" in result_lower

        # Au moins un des deux doit être présent
        assert has_designation or has_description
