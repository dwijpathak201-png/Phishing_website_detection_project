"""PhishGuard: offline HTML feature extraction (Phase 3, step 1).

No network access happens in this module. HTML must be provided explicitly.
A future secure fetcher can supply inspected HTML after validating and pinning
public destinations at each redirect.
"""

from typing import Any, Dict, Optional, Union
from urllib.parse import urljoin, urlsplit

from bs4 import BeautifulSoup


MAX_HTML_BYTES = 1_000_000


def _empty_result(status: str = "Not inspected (network fetching disabled)") -> Dict[str, Any]:
    """Unknown evidence must stay None, never silently become 0."""
    return {
        "page_has_title": None,
        "page_form_count": None,
        "page_has_password_field": None,
        "page_has_iframe": None,
        "external_action_forms": None,
        "status_code": None,
        "page_title": "",
        "webpage_status": status,
        "inspection_performed": False,
        "final_url": None,
        "external_action_form_count": None,
        "non_http_action_form_count": None,
    }


def _origin(url: str) -> Optional[tuple]:
    """Normalize an HTTP(S) origin as (scheme, IDNA hostname, effective port)."""
    try:
        parts = urlsplit(url)
        scheme = parts.scheme.lower()
        if scheme not in ("http", "https") or not parts.hostname:
            return None
        hostname = parts.hostname.rstrip(".").encode("idna").decode("ascii").lower()
        port = parts.port or (443 if scheme == "https" else 80)
        return scheme, hostname, port
    except (ValueError, UnicodeError):
        return None


def extract_webpage_features_from_html(
    html: Union[str, bytes], page_url: str, *, status_code: Optional[int] = None
) -> Dict[str, Any]:
    """Analyze supplied HTML without making any network requests.

    `page_url` is the URL that actually served the HTML, rather than
    necessarily the original URL before redirects.
    """
    result = _empty_result()

    if not isinstance(html, (str, bytes)) or not html:
        result["webpage_status"] = "No HTML content supplied"
        return result

    if not isinstance(page_url, str) or _origin(page_url) is None:
        result["webpage_status"] = "Invalid HTTP(S) page URL for HTML analysis"
        return result

    size = len(html.encode("utf-8")) if isinstance(html, str) else len(html)
    if size > MAX_HTML_BYTES:
        result["webpage_status"] = "HTML exceeds the inspection size limit"
        return result

    try:
        soup = BeautifulSoup(html, "html.parser")

        title_tag = soup.find("title")
        title = title_tag.get_text(" ", strip=True) if title_tag else ""
        forms = soup.find_all("form")
        password_present = any(
            str(item.get("type", "")).strip().lower() == "password"
            for item in soup.find_all("input")
        )

        # The first HTML <base href> affects resolution of relative actions.
        effective_base = page_url
        base_tag = soup.find("base", href=True)
        if base_tag:
            candidate = urljoin(page_url, str(base_tag["href"]).strip())
            if _origin(candidate) is not None:
                effective_base = candidate

        page_origin = _origin(page_url)
        external_count = 0
        non_http_count = 0
        for form in forms:
            raw_action = str(form.get("action") or "").strip()
            # No action or a fragment submits to this page, not an external site.
            if not raw_action or raw_action.startswith("#"):
                continue

            action_url = urljoin(effective_base, raw_action)
            action_origin = _origin(action_url)
            if action_origin is None:
                # javascript:, data:, mailto:, etc. are not external HTTP(S)
                # submissions. Keep a separate diagnostic instead of mislabeling.
                non_http_count += 1
            elif action_origin != page_origin:
                external_count += 1

        result.update({
            "page_has_title": int(bool(title)),
            "page_form_count": len(forms),
            "page_has_password_field": int(password_present),
            "page_has_iframe": int(bool(soup.find("iframe"))),
            "external_action_forms": int(external_count > 0),
            "status_code": status_code,
            "page_title": title[:100],
            "webpage_status": "HTML analyzed (offline; no network request)",
            "inspection_performed": True,
            "final_url": page_url,
            "external_action_form_count": external_count,
            "non_http_action_form_count": non_http_count,
        })
    except Exception as exc:
        result["webpage_status"] = f"HTML analysis error: {type(exc).__name__}"

    return result


def extract_webpage_features(url: str) -> Dict[str, Any]:
    """Compatibility entry point: do not fetch user-provided URLs yet.

    Call `extract_webpage_features_from_html` with supplied test HTML now.
    Only connect a live fetcher after public-IP pinning, redirect validation,
    response-size limits, timeouts, and proxy protections are implemented.
    """
    return _empty_result()
