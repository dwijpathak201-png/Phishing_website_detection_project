
"""
PhishGuard - DNS Feature Extraction

Checks:
1. IPv4 A records
2. Mail Exchange (MX) records
3. DNS resolution status
4. Timeout and error handling

Uses preferred DNS servers for development.

DNS results are evidence, NOT proof that a
website is safe or malicious.
"""

import ipaddress

from typing import Dict, Any, List

import dns.resolver
import dns.exception

from config import DNS_TIMEOUT_SECONDS


# ==================================================
# 1. DNS CONFIGURATION
# ==================================================

# Temporary development DNS servers.
# These were previously reachable using Windows nslookup.
# Their accessibility through dnspython must still be tested.

DNS_SERVERS = [
    "192.168.1.1",
    "1.1.1.1",
]

DNS_QUERY_TIMEOUT = 2.0


# ==================================================
# 2. DEFAULT DNS REPORT
# ==================================================

def _empty_dns_report(status: str) -> Dict[str, Any]:
    """
    Return a consistent report when DNS
    lookup cannot or should not run.
    """

    return {
        "has_dns_a_record": 0,
        "has_dns_mx_record": 0,
        "dns_resolved": False,
        "ip_addresses": [],
        "mx_records": [],
        "dns_status": status,
        "dns_a_status": "not_checked",
        "dns_mx_status": "not_checked",
        "dns_lookup_performed": False,
    }


# ==================================================
# 3. DNS RECORD LOOKUP
# ==================================================

def _lookup_record(
    resolver,
    hostname: str,
    record_type: str
):
    """
    Perform one DNS record lookup.

    Returns:
        (answers, status)

    Status values:
        found
        no_record
        nxdomain
        timeout
        resolver_error
    """

    try:
        answers = resolver.resolve(
            hostname,
            record_type,
            search=False
        )

        records = list(answers)

        if records:
            return records, "found"

        return [], "no_record"

    except dns.resolver.NXDOMAIN:
        return [], "nxdomain"

    except dns.resolver.NoAnswer:
        return [], "no_record"

    except dns.resolver.NoNameservers:
        return [], "resolver_error"

    except dns.exception.Timeout:
        return [], "timeout"

    except dns.exception.DNSException:
        return [], "resolver_error"

    except (ValueError, UnicodeError):
        return [], "resolver_error"


# ==================================================
# 4. MAIN DNS FEATURE EXTRACTION
# ==================================================

def extract_dns_features(
    domain_or_host: str
) -> Dict[str, Any]:
    """
    Extract DNS information for a hostname.

    Args:
        domain_or_host:
            Hostname only, e.g. example.com.

    Returns:
        DNS features and inspection metadata.

    Note:
        Missing DNS records are not automatically
        an indication of phishing.
    """

    # --------------------------------------------------
    # VALIDATE HOSTNAME
    # --------------------------------------------------

    if not isinstance(domain_or_host, str):
        return _empty_dns_report(
            "Invalid hostname"
        )

    hostname = domain_or_host.strip().rstrip(".")

    if not hostname:
        return _empty_dns_report(
            "Empty hostname"
        )

    # Handle bracketed IPv6 addresses.
    if (
        hostname.startswith("[")
        and hostname.endswith("]")
    ):
        hostname = hostname[1:-1]

    # --------------------------------------------------
    # HANDLE DIRECT IP ADDRESSES
    # --------------------------------------------------

    try:
        ip = ipaddress.ip_address(hostname)

        result = _empty_dns_report(
            "Direct IP address - DNS lookup bypassed"
        )

        result["ip_addresses"] = [str(ip)]

        return result

    except ValueError:
        pass

    # --------------------------------------------------
    # REJECT INVALID DNS HOSTNAME INPUT
    # --------------------------------------------------

    if any(char in hostname for char in "/?#@:"):
        return _empty_dns_report(
            "Invalid hostname - supply hostname only"
        )

    lower_hostname = hostname.lower()

    if (
        lower_hostname == "localhost"
        or lower_hostname.endswith(".localhost")
    ):
        return _empty_dns_report(
            "Local hostname - DNS lookup skipped"
        )

    # Support internationalized domain names.
    try:
        hostname = hostname.encode(
            "idna"
        ).decode("ascii")

    except UnicodeError:
        return _empty_dns_report(
            "Invalid internationalized hostname"
        )

    # --------------------------------------------------
    # CONFIGURE DNS RESOLVER - UPDATED
    # --------------------------------------------------

    timeout = float(DNS_TIMEOUT_SECONDS)

    if timeout <= 0:
        raise ValueError(
            "DNS_TIMEOUT_SECONDS must be positive."
        )

    try:

        # Do not load the problematic automatic
        # DNS server ordering discovered on Windows.
        resolver = dns.resolver.Resolver(
            configure=False
        )

        # Prefer the local network DNS server.
        # Cloudflare DNS is the alternative.
        resolver.nameservers = DNS_SERVERS.copy()

        # Maximum wait per DNS server attempt.
        resolver.timeout = min(
            DNS_QUERY_TIMEOUT,
            timeout
        )

        # Total lifetime for each record lookup.
        resolver.lifetime = timeout

    except (
        dns.exception.DNSException,
        ValueError,
        OSError
    ):
        return _empty_dns_report(
            "Could not initialize DNS resolver"
        )

    # --------------------------------------------------
    # LOOK UP A RECORD
    # --------------------------------------------------

    a_answers, a_status = _lookup_record(
        resolver,
        hostname,
        "A"
    )

    resolved_ips: List[str] = [
        answer.to_text()
        for answer in a_answers
    ]

    has_a_record = int(
        bool(resolved_ips)
    )

    # --------------------------------------------------
    # LOOK UP MX RECORD
    # --------------------------------------------------

    mx_records: List[str] = []

    # If the hostname does not exist,
    # skip the second lookup.
    if a_status == "nxdomain":

        mx_status = "not_checked"

    else:

        mx_answers, mx_status = _lookup_record(
            resolver,
            hostname,
            "MX"
        )

        if mx_status == "found":

            for answer in mx_answers:

                exchange = (
                    answer.exchange
                    .to_text()
                    .rstrip(".")
                )

                # A root exchange "." indicates
                # a null MX (no incoming email).
                if exchange:
                    mx_records.append(exchange)

            if not mx_records:
                mx_status = "null_mx"

    has_mx_record = int(
        bool(mx_records)
    )

    # --------------------------------------------------
    # DNS RESOLUTION STATUS
    # --------------------------------------------------

    dns_resolved = bool(
        has_a_record or has_mx_record
    )

    # --------------------------------------------------
    # HUMAN-READABLE STATUS
    # --------------------------------------------------

    if a_status == "nxdomain":

        dns_status = (
            "Domain does not exist (NXDOMAIN)"
        )

    elif has_a_record and has_mx_record:

        dns_status = (
            "A and MX records resolved successfully"
        )

    elif has_a_record:

        dns_status = (
            f"A record resolved; MX status: {mx_status}"
        )

    elif has_mx_record:

        dns_status = (
            f"MX record resolved; A status: {a_status}"
        )

    elif (
        a_status == "timeout"
        or mx_status == "timeout"
    ):

        dns_status = (
            "DNS lookup timed out or was incomplete"
        )

    elif (
        a_status == "resolver_error"
        or mx_status == "resolver_error"
    ):

        dns_status = (
            "DNS resolver error - results may be incomplete"
        )

    else:

        dns_status = (
            "No usable A/MX records found "
            f"(A: {a_status}, MX: {mx_status})"
        )

    # --------------------------------------------------
    # FINAL DNS REPORT
    # --------------------------------------------------

    return {

        # Numerical DNS features
        "has_dns_a_record": has_a_record,
        "has_dns_mx_record": has_mx_record,

        # DNS resolution summary
        "dns_resolved": dns_resolved,

        # DNS evidence
        "ip_addresses": resolved_ips,
        "mx_records": mx_records,

        # Human-readable status
        "dns_status": dns_status,

        # Detailed lookup status
        "dns_a_status": a_status,
        "dns_mx_status": mx_status,

        # Indicates DNS was attempted
        "dns_lookup_performed": True,
    }
