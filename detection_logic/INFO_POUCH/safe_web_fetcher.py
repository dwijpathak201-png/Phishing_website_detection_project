"""PhishGuard - guarded, opt-in HTTP(S) HTML retrieval (Phase 3, step 2).

Requires urllib3 >= 2.0 and the existing webpage_analysis.py.

DO NOT automatically call this function on untrusted URLs in a publicly
accessible app. It enforces several SSRF defenses, but an isolated network
worker with outbound firewall restrictions remains advisable in production.

Safety controls:
- Only HTTP/HTTPS on standard ports; no URL userinfo.
- DNS resolution validates ALL returned addresses as public, then pins the
  outbound TCP connection to one validated address (avoids DNS rebinding).
- HTTPS SNI and certificate verification use the original DNS hostname.
- Redirects followed only after validating their destinations (max 3).
- urllib3 direct pools do not consult environment HTTP(S)_PROXY variables.
- No cookies, credentials, form submits, JS, or browser scripting.
- 1 MB maximum HTML body, disabled compression, connection/read timeouts.
- Explicit blocklist for local/reserved names and non-global IP ranges.
"""

from __future__ import annotations

import ipaddress
import re
import socket
import ssl
import time
from urllib.parse import urljoin, urlsplit, urlunsplit

import urllib3

from detection_logic.INFO_POUCH.webpage_analysis import (
    _empty_result,
    extract_webpage_features_from_html,
)


MAX_BYTES = 1_000_000
MAX_REDIRECTS = 3
MAX_DNS_ADDRESSES = 32
CONNECT_TIMEOUT_SECONDS = 3.0
READ_TIMEOUT_SECONDS = 4.0
DOWNLOAD_DEADLINE_SECONDS = 12.0  # Cooperative deadline, not DNS hard timeout.
ALLOWED_PORTS = {"http": 80, "https": 443}
REDIRECT_STATUS_CODES = {301, 302, 303, 307, 308}
BLOCKED_SUFFIXES = (
    ".localhost", ".local", ".internal", ".invalid",
    ".test", ".onion", ".home.arpa", ".example",
)


class WebpageFetchError(Exception):
    """An unsafe or unsuccessful inspection attempt."""


def _is_public_ip(value: str) -> bool:
    try:
        address = ipaddress.ip_address(value)
        # IPv4-mapped IPv6 addresses must be checked against IPv4 policy too.
        if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped:
            address = address.ipv4_mapped
        return (
            address.is_global
            and not address.is_multicast
            and not address.is_reserved
            and not address.is_unspecified
        )
    except ValueError:
        return False


def _validate_url(raw_url: str) -> tuple[str, str, str, int, str]:
    """Return normalized URL, scheme, IDNA host, port, request target."""
    if not isinstance(raw_url, str) or not raw_url or len(raw_url) > 2048:
        raise WebpageFetchError("Blocked: URL must contain 1-2048 characters")
    if re.search(r"[\s\x00-\x1f\x7f\\]", raw_url):
        raise WebpageFetchError("Blocked: whitespace, controls or backslashes in URL")

    try:
        parts = urlsplit(raw_url)
        scheme = parts.scheme.lower()
        if scheme not in ALLOWED_PORTS or not parts.netloc:
            raise WebpageFetchError("Blocked: only absolute HTTP(S) URLs supported")
        if parts.username is not None or parts.password is not None or "@" in parts.netloc:
            raise WebpageFetchError("Blocked: URL credentials / userinfo are not allowed")
        raw_host = parts.hostname
        if not raw_host or "%" in raw_host:
            raise WebpageFetchError("Blocked: invalid hostname")
        host = raw_host.rstrip(".").encode("idna").decode("ascii").lower()
        port = parts.port or ALLOWED_PORTS[scheme]
        if port != ALLOWED_PORTS[scheme]:
            raise WebpageFetchError("Blocked: nonstandard HTTP(S) port")
        if host in {"localhost", "local", "internal"} or any(
            host.endswith(suffix) for suffix in BLOCKED_SUFFIXES
        ):
            raise WebpageFetchError("Blocked: local or reserved hostname")
        try:
            # Allow valid IP literal hostnames, but their public status is checked later.
            ipaddress.ip_address(host)
            ip_literal = True
        except ValueError:
            ip_literal = False
        if not ip_literal:
            if len(host) > 253 or "." not in host or not all(
                re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?", label)
                for label in host.split(".")
            ):
                raise WebpageFetchError("Blocked: invalid public DNS hostname")
        authority = f"[{host}]" if ":" in host else host
        normalized = urlunsplit((scheme, authority, parts.path or "/", parts.query, ""))
        target = parts.path or "/"
        if parts.query:
            target += "?" + parts.query
        return normalized, scheme, host, port, target
    except (ValueError, UnicodeError) as exc:
        raise WebpageFetchError(f"Blocked: malformed URL ({type(exc).__name__})") from exc


def _resolve_public_target(host: str, port: int) -> str:
    """Require all A/AAAA results to be globally routable; pin first answer."""
    try:
        literal = ipaddress.ip_address(host)
        addresses = [str(literal)]
    except ValueError:
        try:
            answers = socket.getaddrinfo(
                host, port, family=socket.AF_UNSPEC,
                type=socket.SOCK_STREAM, proto=socket.IPPROTO_TCP,
            )
        except (socket.gaierror, OSError) as exc:
            raise WebpageFetchError(f"DNS lookup failed: {type(exc).__name__}") from exc
        addresses = []
        for family, _type, _proto, _canon, sockaddr in answers:
            if family not in (socket.AF_INET, socket.AF_INET6):
                raise WebpageFetchError("Blocked: unexpected DNS address family")
            addresses.append(sockaddr[0])
            if len(addresses) > MAX_DNS_ADDRESSES:
                raise WebpageFetchError("Blocked: too many DNS results")

    if not addresses:
        raise WebpageFetchError("DNS lookup returned no usable addresses")
    if any(not _is_public_ip(address) for address in addresses):
        raise WebpageFetchError("Blocked: destination resolves to non-public IP")
    return addresses[0]


def _request_one_hop(
    ip: str, scheme: str, host: str, port: int, target: str, deadline: float
) -> tuple[int, dict[str, str], bytes]:
    """GET directly from pinned IP; preserve Host and HTTPS SNI/cert checks."""
    if time.monotonic() > deadline:
        raise WebpageFetchError("Inspection deadline exceeded")

    timeout = urllib3.Timeout(
        connect=CONNECT_TIMEOUT_SECONDS, read=READ_TIMEOUT_SECONDS
    )
    pool_kwargs = {"maxsize": 1, "block": True, "retries": False, "timeout": timeout}
    if scheme == "https":
        pool = urllib3.HTTPSConnectionPool(
            host=ip,
            port=port,
            ssl_context=ssl.create_default_context(),
            server_hostname=host,
            assert_hostname=host,
            **pool_kwargs,
        )
    else:
        pool = urllib3.HTTPConnectionPool(host=ip, port=port, **pool_kwargs)

    response = None
    try:
        response = pool.urlopen(
            "GET", target,
            headers={
                "Host": f"[{host}]" if ":" in host else host,
                "User-Agent": "PhishGuard-Research/0.1 (passive URL inspector)",
                "Accept": "text/html,application/xhtml+xml",
                "Accept-Encoding": "identity",
                "Connection": "close",
            },
            redirect=False,
            retries=False,
            preload_content=False,
            decode_content=False,
            timeout=timeout,
        )
        status = int(response.status)
        headers = {str(k).lower(): str(v) for k, v in response.headers.items()}

        # Redirect bodies are never needed, and must not be fetched.
        if status in REDIRECT_STATUS_CODES:
            return status, headers, b""
        if status < 200 or status >= 300:
            return status, headers, b""
        content_type = headers.get("content-type", "").lower()
        if not ("text/html" in content_type or "application/xhtml+xml" in content_type):
            return status, headers, b""
        if headers.get("content-encoding", "identity").lower() not in ("", "identity"):
            raise WebpageFetchError("Unsupported compressed response")
        try:
            size = int(headers.get("content-length", "0"))
        except ValueError:
            size = 0
        if size > MAX_BYTES:
            raise WebpageFetchError("HTML body exceeds 1 MB limit")

        body = bytearray()
        while True:
            if time.monotonic() > deadline:
                raise WebpageFetchError("Inspection deadline exceeded")
            chunk = response.read(min(16_384, MAX_BYTES + 1 - len(body)), decode_content=False)
            if not chunk:
                break
            body.extend(chunk)
            if len(body) > MAX_BYTES:
                raise WebpageFetchError("HTML body exceeds 1 MB limit")
        return status, headers, bytes(body)
    finally:
        if response is not None:
            response.close()
        pool.close()


def fetch_webpage_features(url: str) -> dict:
    """Guarded, opt-in fetcher; **not** automatically used by analysis_service.

    On unsafe URL, DNS/network failure or non-HTML response, the feature values
    stay None, so the risk engine won't mistake missing checks for safe checks.
    """
    result = _empty_result("Not inspected")
    redirects = 0
    visited = set()
    current = url
    deadline = time.monotonic() + DOWNLOAD_DEADLINE_SECONDS
    try:
        while True:
            normalized, scheme, host, port, target = _validate_url(current)
            if normalized in visited:
                raise WebpageFetchError("Redirect loop detected")
            visited.add(normalized)
            ip = _resolve_public_target(host, port)
            status, headers, body = _request_one_hop(
                ip, scheme, host, port, target, deadline
            )
            result["status_code"] = status
            result["final_url"] = normalized
            result["redirect_count"] = redirects

            if status in REDIRECT_STATUS_CODES:
                location = headers.get("location")
                if not location:
                    raise WebpageFetchError("Redirect without Location header")
                if redirects >= MAX_REDIRECTS:
                    raise WebpageFetchError("Too many redirects")
                destination = urljoin(normalized, location)
                _, next_scheme, _, _, _ = _validate_url(destination)
                if scheme == "https" and next_scheme != "https":
                    raise WebpageFetchError("Blocked: HTTPS-to-HTTP redirect")
                current = destination
                redirects += 1
                continue

            if status < 200 or status >= 300:
                raise WebpageFetchError(f"HTTP {status}: page not inspected")
            content_type = headers.get("content-type", "").lower()
            if not ("text/html" in content_type or "application/xhtml+xml" in content_type):
                raise WebpageFetchError("Non-HTML response: page not inspected")
            if not body:
                raise WebpageFetchError("Empty HTML response")
            inspected = extract_webpage_features_from_html(
                body, normalized, status_code=status
            )
            inspected["redirect_count"] = redirects
            if inspected.get("inspection_performed"):
                inspected["webpage_status"] = f"HTML inspected (HTTP {status}; {redirects} redirects)"
            return inspected
    except (WebpageFetchError, urllib3.exceptions.HTTPError, ssl.SSLError, OSError, ValueError) as exc:
        result["webpage_status"] = f"Not inspected: {exc}"
        result.setdefault("redirect_count", redirects)
        return result
