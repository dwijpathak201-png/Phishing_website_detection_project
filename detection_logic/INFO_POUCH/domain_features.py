"""
Domain Feature Extraction using tldextract.

Extracts:
- Subdomain depth
- Domain length
- TLD / suffix
- Punycode indicator
- Registered domain
"""

from typing import Dict, Any

import tldextract


def extract_domain_features(url: str) -> Dict[str, Any]:
    """
    Extract domain-level features from a validated URL.

    Args:
        url: Validated and normalized URL.

    Returns:
        Dictionary containing domain features.
    """

    extracted = tldextract.extract(url)

    subdomain = extracted.subdomain.lower()
    domain = extracted.domain.lower()
    suffix = extracted.suffix.lower()

    # Count subdomain levels
    if subdomain:
        subdomain_count = len(subdomain.split("."))
    else:
        subdomain_count = 0

    # Check for Punycode
    has_punycode = int(
        "xn--" in domain or "xn--" in subdomain
    )

    # Registered domain
    registered_domain = (
        f"{domain}.{suffix}"
        if domain and suffix
        else domain
    )

    features = {
        "domain_length": len(domain),
        "subdomain_count": subdomain_count,
        "has_punycode": has_punycode,

        # Metadata
        "subdomain": subdomain,
        "domain_name": domain,
        "suffix": suffix,
        "registered_domain": registered_domain,
    }

    return features