
"""
PhishGuard V3 - Streamlit Analysis Service

Features:
1. URL lexical analysis
2. Domain structural analysis
3. Random Forest V3 prediction
4. DNS A and MX record inspection
5. Hybrid risk scoring
6. Scan ID and timestamp
7. Evidence for Risk Assessment Report

Live webpage inspection remains disabled.

DNS records are evidence, not proof of legitimacy.
Risk scores are provisional.
"""

import ipaddress
import re

from datetime import datetime, timezone
from uuid import uuid4

from detection_logic.INFO_POUCH.url_features import (
    extract_url_features
)

from detection_logic.INFO_POUCH.domain_features import (
    extract_domain_features
)

from detection_logic.INFO_POUCH.dns_features import (
    extract_dns_features
)

from Models.predict_v3 import (
    predict_v3,
    FEATURE_COLUMNS
)

from detection_logic.risk_scoring import evaluate_risk


# ==========================================
# 1. DNS SAFETY CONFIGURATION
# ==========================================

SKIP_DNS_SUFFIXES = (
    ".localhost",
    ".local",
    ".internal",
    ".test",
    ".invalid",
    ".example",
    ".onion",
)


def _ip(value):
    """Check whether a hostname is an IP address."""

    try:
        ipaddress.ip_address(value)
        return True

    except ValueError:
        return False


def _dns_display(value, status):
    """
    Convert DNS status to human-readable text.

    Do not confuse a failed DNS lookup with
    a confirmed missing DNS record.
    """

    if status == "found" and value == 1:
        return "Found"

    if status == "null_mx":
        return "Null MX (mail intentionally disabled)"

    if status == "no_record":
        return "No record returned"

    if status == "nxdomain":
        return "Domain not found (NXDOMAIN)"

    if status == "timeout":
        return "Unknown (DNS timeout)"

    if status == "resolver_error":
        return "Unknown (DNS resolver error)"

    if status == "not_checked":
        return "Not checked"

    return f"Unknown ({status or 'unavailable'})"


# ==========================================
# 2. MAIN URL ANALYSIS
# ==========================================

def analyze_url(url):
    """
    Analyze a submitted URL and return
    the complete PhishGuard V3 report.
    """

    # --------------------------------------
    # VALIDATE INPUT
    # --------------------------------------

    if not isinstance(url, str):
        raise ValueError(
            "Please enter a website URL."
        )

    url = url.strip()

    if not url or len(url) > 2048:
        raise ValueError(
            "Enter a website URL up to 2048 characters long."
        )

    if re.search(r"[\s\x00-\x1f\x7f\\]", url):
        raise ValueError(
            "URL must not contain spaces, control characters "
            "or backslashes."
        )

    # Only HTTP, HTTPS, or scheme-less host-first URLs.
    if (
        re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", url)
        and not re.match(
            r"^https?://",
            url,
            re.IGNORECASE
        )
    ):
        raise ValueError(
            "Only HTTP and HTTPS website URLs are supported."
        )

    # --------------------------------------
    # URL AND DOMAIN FEATURES
    # --------------------------------------

    try:
        url_features = extract_url_features(url)

        domain_features = extract_domain_features(url)

    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"URL could not be analyzed: {exc}"
        ) from exc

    hostname = url_features.get("hostname") or ""

    if not hostname or not (
        "." in hostname or _ip(hostname)
    ):
        raise ValueError(
            "Please enter a valid domain or IP address."
        )

    # --------------------------------------
    # RANDOM FOREST V3
    # --------------------------------------

    numeric_features = {
        **url_features,
        **domain_features,
    }

    prediction = predict_v3(numeric_features)

    # --------------------------------------
    # DNS FEATURE EXTRACTION
    # --------------------------------------

    lower_hostname = hostname.lower().rstrip(".")

    if lower_hostname.endswith(SKIP_DNS_SUFFIXES):

        dns_features = {
            "has_dns_a_record": 0,
            "has_dns_mx_record": 0,
            "dns_resolved": False,
            "ip_addresses": [],
            "mx_records": [],
            "dns_status": (
                "Reserved/local domain: DNS lookup skipped"
            ),
            "dns_a_status": "not_checked",
            "dns_mx_status": "not_checked",
            "dns_lookup_performed": False,
        }

    else:

        # Uses the working DNS collector.
        # No webpage is fetched or opened.
        dns_features = extract_dns_features(
            hostname
        )

    # --------------------------------------
    # WEBPAGE INSPECTION
    # --------------------------------------

    # Webpage inspection stays disabled until
    # safe redirect and network protections
    # have been implemented.

    webpage_features = {
        "page_has_title": None,
        "page_form_count": None,
        "page_has_password_field": None,
        "page_has_iframe": None,
        "external_action_forms": None,
        "status_code": None,
        "webpage_status": (
            "Not inspected (webpage fetching disabled)"
        ),
    }

    # --------------------------------------
    # CENTRALIZED REPORT STRUCTURE
    # --------------------------------------

    report = {
        "url": url,
        "url_features": url_features,
        "domain_features": domain_features,
        "dns_features": dns_features,
        "webpage_features": webpage_features,
        "numeric_features": numeric_features,
    }

    # --------------------------------------
    # HYBRID RISK SCORING
    # --------------------------------------

    # DEVELOPMENT ASSUMPTION:
    # Numerical class 1 represents phishing.
    # Must be verified against the data source.

    risk = evaluate_risk(
        report,
        ml_prediction=prediction["predicted_label"],
        ml_probability=prediction["probability_label_1"],
        phishing_label=1,
    )

    # --------------------------------------
    # SCAN COVERAGE
    # --------------------------------------

    dns_attempted = bool(
        dns_features.get("dns_lookup_performed", False)
    )

    a_status = dns_features.get(
        "dns_a_status", "not_checked"
    )

    mx_status = dns_features.get(
        "dns_mx_status", "not_checked"
    )

    coverage = [
        "URL structure: analyzed",
        "Domain structure: analyzed",
        (
            f"DNS attempted - A: {a_status}; MX: {mx_status}"
            if dns_attempted
            else (
                "DNS skipped - "
                + dns_features.get(
                    "dns_status", "not checked"
                )
            )
        ),
        "Live webpage: not inspected",
    ]

    # --------------------------------------
    # REPORT EVIDENCE
    # --------------------------------------

    evidence = [

        {
            "label": "Hostname",
            "value": hostname,
        },

        {
            "label": "Protocol observed",
            "value": url_features.get(
                "scheme_status", "unknown"
            ),
        },

        {
            "label": "URL length",
            "value": url_features.get("url_length"),
        },

        {
            "label": "Path length",
            "value": url_features.get("path_length"),
        },

        {
            "label": "Dots",
            "value": url_features.get("num_dots"),
        },

        {
            "label": "Hyphens",
            "value": url_features.get("num_hyphens"),
        },

        {
            "label": "Subdomains",
            "value": domain_features.get(
                "subdomain_count"
            ),
        },

        {
            "label": "Suspicious keyword matches",
            "value": url_features.get(
                "suspicious_keyword_count"
            ),
        },

        {
            "label": "DNS A record",
            "value": _dns_display(
                dns_features.get("has_dns_a_record"),
                a_status,
            ),
        },

        {
            "label": "DNS MX record",
            "value": _dns_display(
                dns_features.get("has_dns_mx_record"),
                mx_status,
            ),
        },

        {
            "label": "IPv4 addresses",
            "value": (
                ", ".join(
                    dns_features.get("ip_addresses") or []
                )
                or "Not available"
            ),
        },

        {
            "label": "Mail servers",
            "value": (
                ", ".join(
                    dns_features.get("mx_records") or []
                )
                or "Not available"
            ),
        },

        {
            "label": "DNS diagnostic",
            "value": dns_features.get(
                "dns_status", "Unavailable"
            ),
        },

        {
            "label": "Webpage inspection",
            "value": "Not performed",
        },
    ]

    # --------------------------------------
    # FINAL REPORT
    # --------------------------------------

    return {

        "url": url,

        "scan_id": uuid4().hex[:10],

        "scanned_at": datetime.now(
            timezone.utc
        ).isoformat(),

        "model": "Random Forest V3",

        "feature_count": len(FEATURE_COLUMNS),

        "prediction": prediction,

        "risk_assessment": risk,

        "dns_features": dns_features,

        "webpage_features": webpage_features,

        "coverage": coverage,

        "evidence": evidence,

        "disclaimer": (
            "Experimental assessment. Numerical class 1 "
            "is provisionally treated as phishing, pending "
            "dataset source verification. Model estimates "
            "are uncalibrated and thresholds are provisional. "
            "DNS presence does not establish legitimacy. "
            "A low score is not proof of safety."
        ),
    }
