"""
PhishGuard - Automated Regression Test Suite

Covers mandatory Priority 8 requirements:
- Category A: Official domain recognition
- Category B: Typographical impersonation candidates
- Category C: Misleading hostname structures
- Category D: General domain handling & edge cases
- Category E: Model & scoring regression, rule evaluation, and V4 report compatibility
"""

import sys
import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from detection_logic.INFO_POUCH.url_features import (
    extract_url_features,
    detect_brand_impersonation,
    BRAND_REGISTRY,
    KNOWN_BRANDS,
    damerau_levenshtein_distance,
)
from detection_logic.INFO_POUCH.domain_features import extract_domain_features
from detection_logic.INFO_POUCH.dns_features import extract_dns_features
from detection_logic.risk_scoring import evaluate_risk
from Models.predict_v3 import predict_v3, FEATURE_COLUMNS
from ui.analysis_service import analyze_url


class TestCategoryAOfficialDomains(unittest.TestCase):
    """Category A: Official reference domain recognition."""

    def test_official_reference_domains(self):
        official_cases = [
            ("youtube.com", "YouTube", "youtube.com"),
            ("www.youtube.com", "YouTube", "youtube.com"),
            ("https://youtube.com", "YouTube", "youtube.com"),
            ("google.com", "Google", "google.com"),
            ("www.google.com", "Google", "google.com"),
            ("github.com", "GitHub", "github.com"),
            ("microsoft.com", "Microsoft", "microsoft.com"),
            ("apple.com", "Apple", "apple.com"),
            ("amazon.com", "Amazon", "amazon.com"),
            ("paypal.com", "PayPal", "paypal.com"),
            ("spotify.com", "Spotify", "spotify.com"),
            ("reddit.com", "Reddit", "reddit.com"),
            ("discord.com", "Discord", "discord.com"),
            ("dropbox.com", "Dropbox", "dropbox.com"),
            ("netflix.com", "Netflix", "netflix.com"),
        ]

        for raw_url, expected_brand, expected_ref in official_cases:
            with self.subTest(url=raw_url):
                df = extract_domain_features(raw_url)
                self.assertEqual(
                    df["brand_domain_status"],
                    "official_reference",
                    f"{raw_url} should be recognized as official_reference",
                )
                self.assertEqual(
                    df["brand_name"],
                    expected_brand,
                    f"{raw_url} brand name should be {expected_brand}, got {df['brand_name']}",
                )
                self.assertEqual(df["brand_reference_domain"], expected_ref)


class TestCategoryBTyposquatting(unittest.TestCase):
    """Category B: Typographical impersonation candidates."""

    def test_typo_candidates(self):
        typo_cases = [
            ("outube.com", "YouTube"),          # single character deletion (missing Y)
            ("youtbe.com", "YouTube"),          # single character deletion (missing U)
            ("youtub3.com", "YouTube"),         # character / leet substitution (3 for e)
            ("y0utube.com", "YouTube"),         # leet substitution (0 for o)
            ("youtubee.com", "YouTube"),        # extra character insertion
            ("yotuube.com", "YouTube"),         # adjacent transposition
            ("youtube-login.com", "YouTube"),   # brand token in registered domain
            ("g00gle.com", "Google"),           # leet substitution
            ("gogle.com", "Google"),           # single character deletion
            ("paypa1.com", "PayPal"),           # leet substitution
            ("micros0ft.com", "Microsoft"),     # leet substitution
            ("app1e.com", "Apple"),             # leet substitution
        ]

        for raw_url, expected_brand in typo_cases:
            with self.subTest(url=raw_url):
                df = extract_domain_features(raw_url)
                self.assertEqual(
                    df["brand_domain_status"],
                    "possible_impersonation",
                    f"{raw_url} should be detected as possible_impersonation",
                )
                self.assertEqual(
                    df["brand_name"],
                    expected_brand,
                    f"{raw_url} brand name must never be truncated. Expected {expected_brand}, got {df['brand_name']}",
                )
                self.assertNotEqual(
                    df["brand_name"],
                    df["domain_name"],
                    f"Brand name should not be conflated with truncated domain label {df['domain_name']}",
                )


class TestCategoryCMisleadingHostnames(unittest.TestCase):
    """Category C: Misleading hostname structures."""

    def test_misleading_subdomains_and_structures(self):
        cases = [
            ("youtube.com.attacker.example", "YouTube"),
            ("google.com.login.example", "Google"),
            ("paypal.com.secure-login.example", "PayPal"),
            ("https://youtube.com.evil-example.org/path", "YouTube"),
        ]

        for raw_url, expected_brand in cases:
            with self.subTest(url=raw_url):
                df = extract_domain_features(raw_url)
                self.assertNotEqual(
                    df["brand_domain_status"],
                    "official_reference",
                    f"{raw_url} must NOT be recognized as official_reference",
                )
                self.assertEqual(
                    df["brand_domain_status"],
                    "possible_impersonation",
                    f"{raw_url} should trigger possible_impersonation due to misleading structure",
                )
                self.assertEqual(df["brand_name"], expected_brand)


class TestCategoryDGeneralDomains(unittest.TestCase):
    """Category D: General domain handling, edge cases, and validation."""

    def test_unfamiliar_legitimate_domains(self):
        clean_cases = [
            "https://wikipedia.org",
            "https://python.org",
            "https://example.com",
            "https://sub.valid-service.co.uk",
            "my-legitimate-store.com",
        ]
        for url in clean_cases:
            with self.subTest(url=url):
                df = extract_domain_features(url)
                self.assertEqual(df["brand_domain_status"], "unrecognized")
                self.assertIsNone(df["brand_name"])

    def test_ip_based_urls(self):
        ip_urls = [
            "http://192.168.1.100/index.html",
            "http://8.8.8.8:8080/test",
        ]
        for url in ip_urls:
            with self.subTest(url=url):
                uf = extract_url_features(url)
                self.assertEqual(uf["has_ip_address"], 1)

    def test_invalid_urls_raise_value_error(self):
        invalid_inputs = [
            "",
            "   ",
            None,
            "ftp://example.com",
            "http://",
            "https://with spaces.com",
            "https://bad\\slash.com",
        ]
        for bad_input in invalid_inputs:
            with self.subTest(url=bad_input):
                with self.assertRaises(ValueError):
                    analyze_url(bad_input)

    def test_punycode_and_homographs(self):
        # Cyrillic 'о' (U+043E) in google.com -> xn--ggle-55da.com
        fake_google_punycode = "xn--ggle-55da.com"
        df = extract_domain_features(fake_google_punycode)
        self.assertEqual(df["has_punycode"], 1)
        self.assertEqual(df["brand_domain_status"], "possible_impersonation")
        self.assertEqual(df["brand_name"], "Google")


class TestCategoryEModelAndScoring(unittest.TestCase):
    """Category E: Model inference, risk formula, and V4 report contracts."""

    def test_feature_columns_and_ordering(self):
        self.assertEqual(len(FEATURE_COLUMNS), 19)
        self.assertNotIn("is_https", FEATURE_COLUMNS)

    def test_model_predict_v3_structure(self):
        uf = extract_url_features("https://example.com")
        df = extract_domain_features("https://example.com")
        prediction = predict_v3({**uf, **df})

        self.assertIn("predicted_label", prediction)
        self.assertIn("probability_label_1", prediction)
        self.assertIn(prediction["predicted_label"], [0, 1])
        self.assertTrue(0.0 <= prediction["probability_label_1"] <= 1.0)
        self.assertTrue(prediction["label_meaning_verified"])

    def test_hybrid_risk_formula_calculation(self):
        # 1. Unrecognized domain retains raw uncalibrated model score
        unrecognized_report = {
            "url_features": {"has_ip_address": 0, "num_at_symbols": 0, "scheme_status": "https", "suspicious_keyword_count": 0},
            "domain_features": {"has_punycode": 0, "subdomain_count": 0, "brand_domain_status": "unrecognized"},
            "webpage_features": {"page_has_password_field": None, "external_action_forms": None},
        }
        # ml_prob = 0.50 -> ml_score = 50.0. rule_score = 0.0.
        # weighted_score = 0.60 * 50 + 0.40 * 0 = 30.0.
        # combined = max(50.0, 30.0) = 50.0.
        risk_unrec = evaluate_risk(unrecognized_report, ml_prediction=1, ml_probability=0.50, phishing_label=1)
        self.assertEqual(risk_unrec["model_score"], 50.0)
        self.assertEqual(risk_unrec["rule_score"], 0.0)
        self.assertEqual(risk_unrec["risk_score"], 50.0)
        self.assertTrue(risk_unrec["model_is_inconclusive"])

        # 2. Verified official reference domain homepage calibrates model score to baseline low risk (10.0)
        official_report = {
            "url_features": {"has_ip_address": 0, "num_at_symbols": 0, "scheme_status": "https", "suspicious_keyword_count": 0},
            "domain_features": {"has_punycode": 0, "subdomain_count": 0, "brand_domain_status": "official_reference"},
            "webpage_features": {"page_has_password_field": None, "external_action_forms": None},
        }
        risk_off = evaluate_risk(official_report, ml_prediction=1, ml_probability=0.549, phishing_label=1)
        self.assertEqual(risk_off["model_score"], 10.0)
        self.assertEqual(risk_off["raw_model_score"], 54.9)
        self.assertEqual(risk_off["rule_score"], 0.0)
        self.assertEqual(risk_off["risk_score"], 10.0)
        self.assertEqual(risk_off["risk_level"], "Low")
        self.assertFalse(risk_off["model_is_inconclusive"])

    def test_brand_alert_triggers_potential_impersonation(self):
        dummy_report = {
            "url_features": {"has_ip_address": 0, "num_at_symbols": 0, "scheme_status": "https", "suspicious_keyword_count": 0,
                             "brand_name": "YouTube", "brand_domain_status": "possible_impersonation", "brand_detection_reason": "Single-character deletion"},
            "domain_features": {"has_punycode": 0, "subdomain_count": 0, "brand_domain_status": "possible_impersonation"},
            "webpage_features": {},
        }
        risk = evaluate_risk(dummy_report, ml_prediction=0, ml_probability=0.10, phishing_label=1)
        self.assertTrue(risk["brand_alert"])
        self.assertEqual(risk["assessment_status"], "potential_impersonation")
        self.assertGreaterEqual(risk["rule_score"], 50.0)

    @patch("detection_logic.INFO_POUCH.dns_features._lookup_record")
    def test_v4_report_payload_complete_contract(self, mock_lookup):
        # Mock DNS records for deterministic unit test
        mock_lookup.return_value = ([], "no_record")

        report = analyze_url("https://www.youtube.com")

        # Top-level keys expected by report.html and JavaScript
        required_keys = [
            "url",
            "scan_id",
            "scanned_at",
            "model",
            "feature_count",
            "prediction",
            "risk_assessment",
            "dns_features",
            "webpage_features",
            "coverage",
            "evidence",
            "brand_analysis",
            "brand_alert",
            "assessment_status",
            "assessment_warnings",
            "model_is_inconclusive",
            "disclaimer",
        ]
        for key in required_keys:
            self.assertIn(key, report, f"Missing required V4 report field: {key}")

        # Check sub-structures
        self.assertIn("risk_score", report["risk_assessment"])
        self.assertIn("risk_level", report["risk_assessment"])
        self.assertIn("rule_evaluations", report["risk_assessment"])
        self.assertIn("scoring_breakdown", report["risk_assessment"])
        self.assertEqual(len(report["risk_assessment"]["rule_evaluations"]), 9)


if __name__ == "__main__":
    unittest.main()
