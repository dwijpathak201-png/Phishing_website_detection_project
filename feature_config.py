
# Features used as inputs to the PhishGuard ML model

FEATURE_COLUMNS = [
    "url_length",
    "num_dots",
    "num_hyphens",
    "num_underscores",
    "num_slashes",
    "num_question_marks",
    "num_equal_signs",
    "num_at_symbols",
    "num_percent_symbols",
    "num_path_segments",
    "path_length",
    "query_length",
    "num_query_parameters",
    "hostname_length",
    "has_ip_address",
    "is_https",
    "suspicious_keyword_count",
    "domain_length",
    "subdomain_count",
    "has_punycode",
]

TARGET_COLUMN = "label"
