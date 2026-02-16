"""
Tests de sécurité - Sanitization des entrées utilisateur.

Ces tests vérifient:
- Protection contre XSS
- Protection contre l'injection
- Gestion des entrées malveillantes
"""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils.preprocessing import preprocess_product_text, validate_text_input


# =============================================================================
# TESTS XSS Prevention
# =============================================================================
@pytest.mark.security
class TestXSSPrevention:
    """Tests de prévention XSS."""

    XSS_PAYLOADS = [
        # Script tags
        "<script>alert('xss')</script>",
        "<script src='evil.js'></script>",
        "<script>document.location='http://evil.com'</script>",

        # Event handlers
        "<img src=x onerror=alert('xss')>",
        "<img src=x onload=alert('xss')>",
        "<body onload=alert('xss')>",
        "<svg onload=alert('xss')>",
        "<input onfocus=alert('xss') autofocus>",
        "<marquee onstart=alert('xss')>",

        # JavaScript URLs
        "javascript:alert('xss')",
        "<a href='javascript:alert(1)'>click</a>",
        "<iframe src='javascript:alert(1)'>",

        # Data URLs
        "<a href='data:text/html,<script>alert(1)</script>'>",

        # CSS injection
        "<style>body{background:url('javascript:alert(1)')}</style>",

        # SVG
        "<svg><script>alert('xss')</script></svg>",
        "<svg/onload=alert('xss')>",

        # Encoded payloads
        "%3Cscript%3Ealert('xss')%3C/script%3E",
        "&#60;script&#62;alert('xss')&#60;/script&#62;",

        # Obfuscated
        "<ScRiPt>alert('xss')</ScRiPt>",
        "<script>alert`xss`</script>",
        "<script>alert(String.fromCharCode(88,83,83))</script>",
    ]

    @pytest.mark.parametrize("payload", XSS_PAYLOADS)
    def test_xss_payload_neutralized(self, payload):
        """Les payloads XSS sont neutralisés après preprocessing."""
        result = preprocess_product_text(payload, "")

        # Vérifications strictes
        result_lower = result.lower()

        assert "<script" not in result_lower, f"Script tag found in: {result}"
        assert "javascript:" not in result_lower, f"JavaScript URL found in: {result}"
        assert "onerror=" not in result_lower, f"onerror handler found in: {result}"
        assert "onload=" not in result_lower, f"onload handler found in: {result}"
        assert "onfocus=" not in result_lower, f"onfocus handler found in: {result}"
        assert "onclick=" not in result_lower, f"onclick handler found in: {result}"

    def test_nested_xss_payloads(self):
        """Les payloads XSS imbriqués sont neutralisés."""
        nested_payloads = [
            "<scr<script>ipt>alert('xss')</scr</script>ipt>",
            "<<script>script>alert('xss')<</script>/script>",
            "<img src='x' onerror='<script>alert(1)</script>'>",
        ]

        for payload in nested_payloads:
            result = preprocess_product_text(payload, "")
            assert "<script" not in result.lower()


# =============================================================================
# TESTS HTML Injection
# =============================================================================
@pytest.mark.security
class TestHTMLInjection:
    """Tests de protection contre l'injection HTML."""

    HTML_PAYLOADS = [
        # Balises dangereuses
        "<iframe src='http://evil.com'>",
        "<object data='http://evil.com'>",
        "<embed src='http://evil.com'>",
        "<form action='http://evil.com'><input type='submit'></form>",

        # Balises de style
        "<style>@import 'http://evil.com/evil.css';</style>",
        "<link rel='stylesheet' href='http://evil.com/evil.css'>",

        # Meta refresh
        "<meta http-equiv='refresh' content='0;url=http://evil.com'>",

        # Base tag hijacking
        "<base href='http://evil.com'>",
    ]

    @pytest.mark.parametrize("payload", HTML_PAYLOADS)
    def test_html_injection_blocked(self, payload):
        """Les balises HTML dangereuses sont supprimées."""
        result = preprocess_product_text(payload, "")

        result_lower = result.lower()
        assert "<iframe" not in result_lower
        assert "<object" not in result_lower
        assert "<embed" not in result_lower
        assert "<form" not in result_lower
        assert "<style" not in result_lower
        assert "<link" not in result_lower
        assert "<meta" not in result_lower
        assert "<base" not in result_lower


# =============================================================================
# TESTS SQL Injection (informational)
# =============================================================================
@pytest.mark.security
class TestSQLInjection:
    """Tests informatifs pour les patterns SQL injection.

    Note: L'application n'utilise pas de SQL directement,
    mais ces tests vérifient que les inputs sont sanitisés
    pour éviter des problèmes en aval.
    """

    SQL_PAYLOADS = [
        "'; DROP TABLE users; --",
        "1 OR 1=1",
        "admin'--",
        "1'; DELETE FROM products WHERE 1=1; --",
        "' UNION SELECT * FROM users --",
        "1; UPDATE products SET price=0 --",
    ]

    @pytest.mark.parametrize("payload", SQL_PAYLOADS)
    def test_sql_payloads_handled(self, payload):
        """Les payloads SQL sont gérés sans crash."""
        # Le preprocessing ne devrait pas crash
        result = preprocess_product_text(payload, "")
        assert isinstance(result, str)

        # La validation devrait fonctionner
        is_valid, message = validate_text_input(payload)
        assert isinstance(is_valid, bool)


# =============================================================================
# TESTS Path Traversal (informational)
# =============================================================================
@pytest.mark.security
class TestPathTraversal:
    """Tests pour les tentatives de path traversal."""

    PATH_PAYLOADS = [
        "../../../etc/passwd",
        "..\\..\\..\\windows\\system32\\config\\sam",
        "/etc/passwd",
        "C:\\Windows\\System32\\config\\SAM",
        "....//....//....//etc/passwd",
        "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd",
    ]

    @pytest.mark.parametrize("payload", PATH_PAYLOADS)
    def test_path_traversal_handled(self, payload):
        """Les tentatives de path traversal sont gérées."""
        result = preprocess_product_text(payload, "")
        assert isinstance(result, str)
        # Le résultat ne devrait pas permettre l'accès aux fichiers système


# =============================================================================
# TESTS DoS Prevention
# =============================================================================
@pytest.mark.security
class TestDoSPrevention:
    """Tests de prévention des attaques DoS."""

    def test_very_long_input_handled(self):
        """Les inputs très longs sont gérés sans crash."""
        # 1 MB de texte
        long_input = "A" * (1024 * 1024)

        # Ne devrait pas crash
        result = preprocess_product_text(long_input, "")
        assert isinstance(result, str)

    def test_deeply_nested_html_handled(self):
        """HTML profondément imbriqué est géré."""
        # 1000 niveaux d'imbrication
        nested = "<div>" * 1000 + "content" + "</div>" * 1000

        result = preprocess_product_text(nested, "")
        assert isinstance(result, str)
        assert "<div>" not in result

    def test_repeated_patterns_handled(self):
        """Les patterns répétés sont gérés."""
        # Tentative de ReDoS
        repeated = "a" * 10000 + "b"

        result = preprocess_product_text(repeated, "")
        assert isinstance(result, str)

    def test_unicode_bomb_handled(self):
        """Les 'unicode bombs' sont gérés."""
        # Caractères Unicode qui peuvent causer des problèmes
        unicode_bomb = "\ufeff" * 1000 + "content"

        result = preprocess_product_text(unicode_bomb, "")
        assert isinstance(result, str)


# =============================================================================
# TESTS Input Validation
# =============================================================================
@pytest.mark.security
class TestInputValidation:
    """Tests de validation des entrées."""

    def test_validate_rejects_empty(self):
        """La validation rejette les entrées vides."""
        is_valid, message = validate_text_input("")
        assert is_valid is False

    def test_validate_rejects_whitespace_only(self):
        """La validation rejette les espaces seuls."""
        is_valid, message = validate_text_input("   \t\n   ")
        assert is_valid is False

    def test_validate_rejects_none(self):
        """La validation rejette None."""
        is_valid, message = validate_text_input(None)
        assert is_valid is False

    def test_validate_accepts_normal_input(self):
        """La validation accepte les entrées normales."""
        is_valid, message = validate_text_input("iPhone 15 Pro Max")
        assert is_valid is True

    def test_validate_provides_error_message(self):
        """La validation fournit un message d'erreur."""
        is_valid, message = validate_text_input("")
        assert isinstance(message, str)
        assert len(message) > 0


# =============================================================================
# TESTS Character Encoding
# =============================================================================
@pytest.mark.security
class TestCharacterEncoding:
    """Tests de gestion de l'encodage des caractères."""

    def test_handles_null_bytes(self):
        """Les null bytes sont gérés."""
        input_with_null = "Product\x00Name"
        result = preprocess_product_text(input_with_null, "")
        assert "\x00" not in result

    def test_handles_control_characters(self):
        """Les caractères de contrôle sont gérés."""
        control_chars = "".join(chr(i) for i in range(32))
        input_with_control = f"Product{control_chars}Name"

        result = preprocess_product_text(input_with_control, "")
        assert isinstance(result, str)

    def test_handles_mixed_encodings(self):
        """Les encodages mixtes sont gérés."""
        mixed = "Téléphone 电话 телефон"
        result = preprocess_product_text(mixed, "")
        assert isinstance(result, str)

    def test_handles_rtl_characters(self):
        """Les caractères RTL sont gérés."""
        rtl = "Product \u202e\u0645\u0646\u062a\u062c"
        result = preprocess_product_text(rtl, "")
        assert isinstance(result, str)
