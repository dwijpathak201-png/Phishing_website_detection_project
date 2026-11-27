
"""
PhishGuard - Domain Feature Extraction (V3.1)

Features:
1. Subdomain depth detection
2. Domain length extraction
3. Public suffix / TLD identification
4. Punycode detection
5. Registered domain identification
6. Official reference domain matching
7. Brand impersonation detection
8. Typosquatting detection for compound TLDs

IMPORTANT:
The existing numerical domain features remain
unchanged for Random Forest V3 compatibility.

Brand detection is heuristic metadata.
It does not guarantee website legitimacy.
"""

from typing import Dict, Any
import ipaddress

from detection_logic.INFO_POUCH.url_features import (
    _parse_input_url,
    detect_brand_impersonation,
    BRAND_REGISTRY,
    KNOWN_BRANDS,
    _tld_extractor,
)


# ============================================
# 1. IP ADDRESS DETECTION
# ============================================

def _is_ip_address(hostname: str) -> bool:
    try:
        ipaddress.ip_address(hostname)
        return True
    except ValueError:
        return False


# ============================================
# 2. DOMAIN IDENTITY ANALYSIS
# ============================================

def analyze_domain_identity(
    hostname: str,
    domain_name: str,
    registered_domain: str,
) -> Dict[str, Any]:
    """
    Identify configured reference domains and possible brand impersonation.
    Coordinates directly with the unified brand detector using PSL-resolved domain labels.
    """
    return detect_brand_impersonation(
        hostname=hostname,
        domain_name=domain_name,
        registered_domain=registered_domain,
    )


# ============================================
# 3. MAIN DOMAIN FEATURE EXTRACTION
# ============================================

def extract_domain_features(
    url: str
) -> Dict[str, Any]:
    """
    Extract domain structure and identity metadata from a validated URL.
    Supports scheme-less and HTTP(S) URLs.
    """

    parsed, _, _ = _parse_input_url(url)

    hostname = (
        parsed.hostname or ""
    ).lower().rstrip(".")

    if not hostname:
        raise ValueError(
            "A valid hostname is required."
        )

    # Public suffix extraction using offline snapshot
    extracted = _tld_extractor(hostname)

    subdomain = extracted.subdomain.lower()
    domain = extracted.domain.lower()
    suffix = extracted.suffix.lower()

    # Subdomain depth (generic "www" prefix is normalized so www.domain.tld has subdomain_count=0)
    if subdomain:
        subdomain_parts = [
            part for part in subdomain.split(".")
            if part and part.lower() != "www"
        ]
        subdomain_count = len(subdomain_parts)
    else:
        subdomain_count = 0

    # Punycode detection
    has_punycode = int(
        "xn--" in domain
        or "xn--" in subdomain
    )

    # Registered domain
    registered_domain = (
        f"{domain}.{suffix}"
        if domain and suffix
        else domain
    )

    # Brand identity analysis
    brand_info = analyze_domain_identity(
        hostname=hostname,
        domain_name=domain,
        registered_domain=registered_domain,
    )

    is_ip_hostname = _is_ip_address(hostname)

    has_unicode_hostname = any(
        ord(character) > 127
        for character in hostname
    )

    return {
        # Original ML numerical features
        "domain_length": len(domain),
        "subdomain_count": subdomain_count,
        "has_punycode": has_punycode,

        # Metadata
        "subdomain": subdomain,
        "domain_name": domain,
        "suffix": suffix,
        "registered_domain": registered_domain,
        "hostname": hostname,
        "is_ip_hostname": is_ip_hostname,
        "has_unicode_hostname": has_unicode_hostname,

        # Brand analysis metadata
        "brand_name": brand_info["brand_name"],
        "brand_reference_domain": brand_info["brand_reference_domain"],
        "brand_domain_status": brand_info["brand_domain_status"],
        "brand_detection_reason": brand_info["brand_detection_reason"],
    }
