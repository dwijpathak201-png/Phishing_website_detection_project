"""
PhishGuard V3.1 — Analysis service.

Compatible with the project's existing Random Forest V3 predictor and
detection_logic/risk_scoring.py V3.1.

The service does not download webpage content. DNS evidence is informational,
not proof that a domain is safe or malicious.
"""

import ipaddress
import re
from datetime import datetime, timezone
from urllib.parse import urlsplit
from uuid import uuid4

from detection_logic.INFO_POUCH.url_features import extract_url_features
from detection_logic.INFO_POUCH.domain_features import extract_domain_features
from detection_logic.INFO_POUCH.dns_features import extract_dns_features
from Models.predict_v3 import predict_v3, FEATURE_COLUMNS
from detection_logic.risk_scoring import evaluate_risk


SKIP_DNS_SUFFIXES = (
    ".localhost", ".local", ".internal", ".test",
    ".invalid", ".example", ".onion",
)
SKIP_DNS_EXACT = {
    "localhost", "local", "internal", "test", "invalid", "example", "onion"
}


def _is_ip(hostname):
    try:
        ipaddress.ip_address(hostname)
        return True
    except ValueError:
        return False


def _skip_dns(hostname):
    """Avoid DNS lookups for reserved/local names and IP literals."""
    name = hostname.lower().rstrip(".")
    return (
        _is_ip(name)
        or name in SKIP_DNS_EXACT
        or name.endswith(SKIP_DNS_SUFFIXES)
    )


def _validate_url(raw_url):
    """Validate input without modifying what the feature extractor receives."""
    if not isinstance(raw_url, str):
        raise ValueError("Please enter a website URL.")

    url = raw_url.strip()
    if not url or len(url) > 2048:
        raise ValueError("Enter a website URL up to 2048 characters long.")

    if re.search(r"[\s\x00-\x1f\x7f\\]", url):
        raise ValueError(
            "URL must not contain spaces, control characters or backslashes."
        )

    if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", url) and not re.match(
        r"^https?://", url, flags=re.IGNORECASE
    ):
        raise ValueError("Only HTTP and HTTPS website URLs are supported.")

    # For validation only. Keep the original input for the V3 feature pipeline.
    candidate = url if re.match(r"^https?://", url, re.I) else "https://" + url
    try:
        parsed = urlsplit(candidate)
        hostname = parsed.hostname
        _ = parsed.port  # Detect malformed ports.
    except ValueError as exc:
        raise ValueError(f"Invalid website URL: {exc}") from exc

    if parsed.scheme.lower() not in ("http", "https") or not hostname:
        raise ValueError("Please enter a valid HTTP or HTTPS URL.")

    if "." not in hostname and not _is_ip(hostname):
        raise ValueError("Please enter a valid domain or IP address.")

    return url


def _dns_not_checked(reason):
    return {
        "has_dns_a_record": 0,
        "has_dns_mx_record": 0,
        "dns_resolved": False,
        "ip_addresses": [],
        "mx_records": [],
        "dns_status": reason,
        "dns_a_status": "not_checked"}