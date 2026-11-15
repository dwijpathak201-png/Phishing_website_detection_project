
# config.py - PhishGuard Configuration

# ==========================================
# NETWORK TIMEOUTS
# ==========================================

DNS_TIMEOUT_SECONDS = 5
REQUEST_TIMEOUT_SECONDS = 10

# Backward compatibility with older modules
DNS_TIMEOUT = DNS_TIMEOUT_SECONDS
REQUEST_TIMEOUT = REQUEST_TIMEOUT_SECONDS


# ==========================================
# RISK SCORE THRESHOLDS
# ==========================================

# Legacy thresholds retained for compatibility.
# Current hybrid risk scoring uses its own
# Low: 0-34.9, Medium: 35-69.9, High: 70-100.

LOW_RISK_THRESHOLD = 25
MEDIUM_RISK_THRESHOLD = 50
HIGH_RISK_THRESHOLD = 75


# ==========================================
# HTTP USER AGENT
# ==========================================

DEFAULT_USER_AGENT = (
    "Mozilla/5.0 "
    "(Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 "
    "(KHTML, like Gecko) "
    "Chrome/154.0.0.0 Safari/537.36"
)
