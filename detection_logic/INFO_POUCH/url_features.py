
"""
PhishGuard - URL Lexical and Structural Feature Extraction.

Supports URLs with or without HTTP/HTTPS schemes.

Uses scheme-independent lexical features for consistency
between training data and live predictions.

Version 3:
Canonicalizes an empty homepage path to "/".
"""

import re
import ipaddress

from typing import Dict, Any
from urllib.parse import urlsplit


# --------------------------------------------------
# SUSPICIOUS KEYWORDS
# --------------------------------------------------

SUSPICIOUS_KEYWORDS = [
    "login",
    "signin",
    "verify",
    "verification",
    "account",
    "update",
    "secure",
    "security",
    "password",
    "bank",
    "wallet",
    "payment",
    "confirm",
    "recover",
]


# --------------------------------------------------
# IP ADDRESS DETECTION
# --------------------------------------------------

def _is_ip_address(hostname: str) -> bool:
    """
    Check whether a hostname is an IPv4 or IPv6 address.
    """

    if not hostname:
        return False

    try:
        ipaddress.ip_address(hostname)
        return True

    except ValueError:
        return False


# --------------------------------------------------
# URL PARSING
# --------------------------------------------------

def _parse_input_url(url: str):
    """
    Parse URLs with or without an HTTP/HTTPS scheme.

    A protocol embedded inside a URL query or path
    must not be mistaken for the main URL's scheme.

    Returns:
        parsed URL
        scheme-independent lexical URL
        scheme status
    """

    if not isinstance(url, str) or not url.strip():
        raise ValueError(
            "URL must be a non-empty string."
        )

    original = url.strip()

    # Detect a protocol ONLY at the beginning.
    scheme_match = re.match(
        r"^([a-zA-Z][a-zA-Z0-9+.-]*)://",
        original
    )

    if original.startswith("//"):

        parsed = urlsplit(original)

        lexical_url = original[2:]

        scheme_status = "unknown"

    elif scheme_match:

        scheme_status = scheme_match.group(1).lower()

        if scheme_status not in ("http", "https"):
            raise ValueError(
                "Only HTTP and HTTPS URLs are supported."
            )

        parsed = urlsplit(original)

        lexical_url = original[scheme_match.end():]

    else:

        # The main URL has no explicit protocol.
        parsed = urlsplit("//" + original)

        lexical_url = original

        scheme_status = "unknown"

    if not parsed.hostname:
        raise ValueError(
            f"URL has no valid hostname: {url!r}"
        )

    return parsed, lexical_url, scheme_status


# --------------------------------------------------
# FEATURE EXTRACTION
# --------------------------------------------------

def extract_url_features(url: str) -> Dict[str, Any]:
    """
    Extract lexical and structural features.

    HTTP/HTTPS prefixes are excluded from lexical counts.

    Empty homepage paths are normalized to "/".

    Examples:
        example.com
        example.com/
        https://example.com
        https://example.com/

    All four produce the same structural features,
    excluding protocol-specific metadata.
    """

    parsed, lexical_url, scheme_status = (
        _parse_input_url(url)
    )

    hostname = parsed.hostname or ""

    # --------------------------------------------------
    # V3 FIX: HOMEPAGE PATH NORMALIZATION
    # --------------------------------------------------

    # A missing path and "/" represent the same
    # homepage structure.

    if parsed.path == "":

        canonical_path = "/"

        # lexical_url starts with parsed.netloc.
        # Insert "/" immediately after the authority.
        #
        # This also handles URLs containing queries:
        #
        # example.com?x=1
        # becomes
        # example.com/?x=1

        authority_end = len(parsed.netloc)

        lexical_url = (
            lexical_url[:authority_end]
            + "/"
            + lexical_url[authority_end:]
        )

    else:

        canonical_path = parsed.path

    # --------------------------------------------------
    # PATH ANALYSIS
    # --------------------------------------------------

    path_segments = [
        segment
        for segment in canonical_path.strip("/").split("/")
        if segment
    ]

    num_path_segments = len(path_segments)

    # --------------------------------------------------
    # QUERY ANALYSIS
    # --------------------------------------------------

    query_parameters = [
        parameter
        for parameter in parsed.query.split("&")
        if parameter
    ]

    num_query_parameters = len(query_parameters)

    # --------------------------------------------------
    # IP ADDRESS CHECK
    # --------------------------------------------------

    has_ip_address = int(
        _is_ip_address(hostname)
    )

    # --------------------------------------------------
    # HTTPS CHECK
    # --------------------------------------------------

    # Unknown protocol is represented as 0 in the
    # existing numerical feature schema.
    #
    # scheme_status records the distinction between
    # an unknown scheme and explicit HTTP.

    is_https = int(
        scheme_status == "https"
    )

    # --------------------------------------------------
    # SUSPICIOUS KEYWORD ANALYSIS
    # --------------------------------------------------

    lower_url = lexical_url.lower()

    detected_keywords = [
        keyword
        for keyword in SUSPICIOUS_KEYWORDS
        if keyword in lower_url
    ]

    # --------------------------------------------------
    # FINAL FEATURE DICTIONARY
    # --------------------------------------------------

    features = {

        # URL-level lexical features
        "url_length": len(lexical_url),

        "num_dots": lexical_url.count("."),

        "num_hyphens": lexical_url.count("-"),

        "num_underscores": lexical_url.count("_"),

        "num_slashes": lexical_url.count("/"),

        "num_question_marks": lexical_url.count("?"),

        "num_equal_signs": lexical_url.count("="),

        "num_at_symbols": lexical_url.count("@"),

        "num_percent_symbols": lexical_url.count("%"),

        # Path features
        "num_path_segments": num_path_segments,

        "path_length": len(canonical_path),

        # Query features
        "query_length": len(parsed.query),

        "num_query_parameters": num_query_parameters,

        # Host-related features
        "hostname_length": len(hostname),

        "has_ip_address": has_ip_address,

        # Protocol feature
        "is_https": is_https,

        # Keyword feature
        "suspicious_keyword_count": len(
            detected_keywords
        ),

        # --------------------------------------------------
        # METADATA FOR EXPLAINABLE RISK REPORTS
        # --------------------------------------------------

        "detected_keywords": detected_keywords,

        "hostname": hostname,

        "path": canonical_path,

        "query": parsed.query,

        "scheme_status": scheme_status,

        "https_observed": (
            None
            if scheme_status == "unknown"
            else scheme_status == "https"
        ),
    }

    return features
