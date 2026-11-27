
"""
Central Feature Extraction Pipeline.

Coordinates URL, domain, DNS, and webpage
feature extraction for PhishGuard.
"""

from typing import Dict, Any
from urllib.parse import urlparse

from detection_logic.INFO_POUCH.url_features import (
    extract_url_features,
)
from detection_logic.INFO_POUCH.domain_features import (
    extract_domain_features,
)
from detection_logic.INFO_POUCH.dns_features import (
    extract_dns_features,
)
from detection_logic.INFO_POUCH.webpage_analysis import (
    extract_webpage_features,
)


def extract_all_features(
    url: str,
    check_live_webpage: bool = True,
) -> Dict[str, Any]:

    # 1. URL features
    url_feats = extract_url_features(url)

    # 2. Domain features
    domain_feats = extract_domain_features(url)

    # 3. Extract the validated hostname for DNS analysis
    hostname = url_feats.get("hostname") or domain_feats.get("hostname") or ""

    # 4. DNS features
    dns_feats = extract_dns_features(hostname)

    # 5. Webpage features
    if check_live_webpage:
        webpage_feats = extract_webpage_features(url)
    else:
        webpage_feats = {
            "page_has_title": None,
            "page_form_count": None,
            "page_has_password_field": None,
            "page_has_iframe": None,
            "external_action_forms": None,
            "status_code": None,
            "page_title": "Inspection Skipped",
            "webpage_status": "Live HTTP inspection disabled",
        }

    # 6. Combine numeric features from all pouches
    raw_feature_map = {
        # URL
        "url_length": url_feats["url_length"],
        "num_dots": url_feats["num_dots"],
        "num_hyphens": url_feats["num_hyphens"],
        "num_underscores": url_feats["num_underscores"],
        "num_slashes": url_feats["num_slashes"],
        "num_question_marks": url_feats["num_question_marks"],
        "num_equal_signs": url_feats["num_equal_signs"],
        "num_at_symbols": url_feats["num_at_symbols"],
        "num_percent_symbols": url_feats["num_percent_symbols"],
        "num_path_segments": url_feats["num_path_segments"],
        "path_length": url_feats["path_length"],
        "query_length": url_feats["query_length"],
        "num_query_parameters": url_feats["num_query_parameters"],
        "hostname_length": url_feats["hostname_length"],
        "has_ip_address": url_feats["has_ip_address"],
        "is_https": url_feats["is_https"],
        "suspicious_keyword_count": (
            url_feats["suspicious_keyword_count"]
        ),

        # Domain
        "domain_length": domain_feats["domain_length"],
        "subdomain_count": domain_feats["subdomain_count"],
        "has_punycode": domain_feats["has_punycode"],

        # DNS
        "has_dns_a_record": dns_feats["has_dns_a_record"],
        "has_dns_mx_record": dns_feats["has_dns_mx_record"],

        # Webpage
        "page_has_title": webpage_feats["page_has_title"],
        "page_form_count": webpage_feats["page_form_count"],
        "page_has_password_field": (
            webpage_feats["page_has_password_field"]
        ),
        "page_has_iframe": webpage_feats["page_has_iframe"],
        "external_action_forms": (
            webpage_feats["external_action_forms"]
        ),
    }

    # 7. Detailed report
    report = {
        "url": url,
        "url_features": url_feats,
        "domain_features": domain_feats,
        "dns_features": dns_feats,
        "webpage_features": webpage_feats,
        "numeric_features": raw_feature_map,
    }

    return report
