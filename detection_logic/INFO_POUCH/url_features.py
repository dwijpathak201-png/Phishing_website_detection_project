
"""
PhishGuard - URL Feature Extraction (V3.1)

Features:
1. URL lexical feature extraction
2. URL path normalization
3. HTTP/HTTPS detection
4. Suspicious keyword detection
5. IP address detection
6. Popular brand domain recognition
7. Typosquatting and lookalike detection
8. Basic Unicode impersonation detection

IMPORTANT:
Existing Random Forest V3 numerical feature
definitions remain unchanged.

Brand detection produces additional metadata.
It does not automatically change the ML score.
"""

import re
import ipaddress
import unicodedata
from typing import Dict, Any, Optional
from urllib.parse import urlsplit

try:
    import tldextract
    _tld_extractor = tldextract.TLDExtract(
        suffix_list_urls=None,
        cache_dir=None,
        fallback_to_snapshot=True,
    )
except ImportError:
    class _PurePythonTLDExtractor:
        TWO_PART_SUFFIXES = {
            "co.uk", "org.uk", "me.uk", "ltd.uk", "plc.uk", "net.uk", "sch.uk", "ac.uk", "gov.uk",
            "com.au", "net.au", "org.au", "edu.au", "gov.au",
            "co.in", "net.in", "org.in", "gen.in", "firm.in", "ind.in", "ac.in", "edu.in", "res.in", "gov.in",
            "co.nz", "net.nz", "org.nz", "govt.nz", "ac.nz",
            "co.za", "org.za", "net.za", "ac.za", "gov.za",
            "co.jp", "ne.jp", "or.jp", "ac.jp", "go.jp",
            "com.br", "net.br", "org.br", "gov.br",
            "com.cn", "net.cn", "org.cn", "gov.cn", "edu.cn",
            "com.sg", "net.sg", "org.sg", "gov.sg", "edu.sg",
            "com.mx", "net.mx", "org.mx", "edu.mx", "gob.mx",
        }

        class _ExtractResult:
            def __init__(self, subdomain, domain, suffix):
                self.subdomain = subdomain
                self.domain = domain
                self.suffix = suffix
                self.registered_domain = f"{domain}.{suffix}" if domain and suffix else domain

        def __call__(self, hostname: str):
            host = hostname.strip(".").lower()
            labels = host.split(".")
            if len(labels) <= 1:
                return self._ExtractResult("", host, "")
            if len(labels) >= 3:
                two_part = f"{labels[-2]}.{labels[-1]}"
                if two_part in self.TWO_PART_SUFFIXES:
                    domain = labels[-3]
                    subdomain = ".".join(labels[:-3])
                    return self._ExtractResult(subdomain, domain, two_part)
            suffix = labels[-1]
            domain = labels[-2]
            subdomain = ".".join(labels[:-2])
            return self._ExtractResult(subdomain, domain, suffix)

    _tld_extractor = _PurePythonTLDExtractor()


# ============================================
# 1. SUSPICIOUS KEYWORDS
# ============================================

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


# ============================================
# 2. REFERENCE BRAND REGISTRY
# ============================================

# Extensible registry of well-known organizations and their verified official domains.
# A match against an official domain indicates identity match, NOT guaranteed safety.
BRAND_REGISTRY = {
    "Google": {
        "domains": ["google.com"],
        "primary_label": "google",
    },
    "YouTube": {
        "domains": ["youtube.com", "youtu.be"],
        "primary_label": "youtube",
    },
    "Microsoft": {
        "domains": ["microsoft.com", "live.com", "office.com"],
        "primary_label": "microsoft",
    },
    "GitHub": {
        "domains": ["github.com"],
        "primary_label": "github",
    },
    "Amazon": {
        "domains": ["amazon.com"],
        "primary_label": "amazon",
    },
    "Apple": {
        "domains": ["apple.com", "icloud.com"],
        "primary_label": "apple",
    },
    "Facebook": {
        "domains": ["facebook.com", "fb.com"],
        "primary_label": "facebook",
    },
    "Instagram": {
        "domains": ["instagram.com"],
        "primary_label": "instagram",
    },
    "WhatsApp": {
        "domains": ["whatsapp.com"],
        "primary_label": "whatsapp",
    },
    "Netflix": {
        "domains": ["netflix.com"],
        "primary_label": "netflix",
    },
    "PayPal": {
        "domains": ["paypal.com"],
        "primary_label": "paypal",
    },
    "LinkedIn": {
        "domains": ["linkedin.com"],
        "primary_label": "linkedin",
    },
    "Reddit": {
        "domains": ["reddit.com"],
        "primary_label": "reddit",
    },
    "Dropbox": {
        "domains": ["dropbox.com"],
        "primary_label": "dropbox",
    },
    "Discord": {
        "domains": ["discord.com", "discord.gg"],
        "primary_label": "discord",
    },
    "Spotify": {
        "domains": ["spotify.com"],
        "primary_label": "spotify",
    },
}

# Backward compatibility map: brand name -> primary official domain
KNOWN_BRANDS = {brand: data["domains"][0] for brand, data in BRAND_REGISTRY.items()}


# ============================================
# 3. COMMON LOOKALIKE CHARACTERS & SUBSTITUTIONS
# ============================================

CONFUSABLES = str.maketrans({
    "а": "a", "α": "a", "ạ": "a", "à": "a", "á": "a", "â": "a", "ã": "a", "ä": "a", "å": "a",
    "Ь": "b", "ь": "b", "ъ": "b",
    "с": "c", "ç": "c",
    "ԁ": "d", "ժ": "d",
    "е": "e", "ε": "e", "é": "e", "è": "e", "ê": "e", "ë": "e", "ё": "e", "ē": "e",
    "ɡ": "g", "ġ": "g",
    "һ": "h",
    "і": "i", "ι": "i", "í": "i", "ì": "i", "î": "i", "ï": "i",
    "ј": "j",
    "к": "k",
    "ӏ": "l", "ł": "l",
    "м": "m",
    "ո": "n", "ñ": "n",
    "о": "o", "ο": "o", "ó": "o", "ò": "o", "ô": "o", "õ": "o", "ö": "o", "ø": "o",
    "р": "p", "ρ": "p",
    "ԛ": "q",
    "г": "r",
    "ѕ": "s", "ʂ": "s",
    "т": "t", "τ": "t",
    "υ": "u", "ú": "u", "ù": "u", "û": "u", "ü": "u", "µ": "u",
    "ν": "v",
    "ԝ": "w",
    "х": "x", "χ": "x",
    "у": "y", "ý": "y", "ÿ": "y",
    "z": "z",
})

LEET_TRANSLATION = str.maketrans({
    "0": "o",
    "1": "l",
    "2": "z",
    "3": "e",
    "4": "a",
    "5": "s",
    "6": "g",
    "7": "t",
    "8": "b",
    "9": "g",
    "@": "a",
    "$": "s",
})


# ============================================
# 4. IP ADDRESS DETECTION
# ============================================

def _is_ip_address(hostname: str) -> bool:
    if not hostname:
        return False
    try:
        ipaddress.ip_address(hostname)
        return True
    except ValueError:
        return False


# ============================================
# 5. URL PARSING
# ============================================

def _parse_input_url(url: str):
    if not isinstance(url, str) or not url.strip():
        raise ValueError("URL must be a non-empty string.")

    original = url.strip()

    scheme_match = re.match(r"^([a-zA-Z][a-zA-Z0-9+.-]*)://", original)

    if original.startswith("//"):
        parsed = urlsplit(original)
        lexical_url = original[2:]
        scheme_status = "unknown"
    elif scheme_match:
        scheme_status = scheme_match.group(1).lower()
        if scheme_status not in ("http", "https"):
            raise ValueError("Only HTTP and HTTPS URLs are supported.")
        parsed = urlsplit(original)
        lexical_url = original[scheme_match.end():]
    else:
        parsed = urlsplit("//" + original)
        lexical_url = original
        scheme_status = "unknown"

    if not parsed.hostname:
        raise ValueError(f"URL has no valid hostname: {url!r}")

    return parsed, lexical_url, scheme_status


# ============================================
# 6. DOMAIN NORMALIZATION & SKELETON
# ============================================

def _normalize_hostname(hostname: str) -> str:
    """Normalize hostname for reliable domain comparisons."""
    return hostname.strip(".").lower()


def _decode_hostname(hostname: str) -> str:
    """Convert punycode labels to Unicode when possible."""
    labels = hostname.split(".")
    decoded = []
    for label in labels:
        try:
            decoded.append(label.encode("ascii").decode("idna"))
        except (UnicodeError, ValueError):
            decoded.append(label)
    return ".".join(decoded)


def _domain_skeleton(hostname: str) -> str:
    """Create a visual-comparison representation for homograph detection."""
    decoded = _decode_hostname(hostname)
    normalized = unicodedata.normalize("NFKC", decoded).casefold()
    return normalized.translate(CONFUSABLES)


# ============================================
# 7. STRING DISTANCE (DAMERAU-LEVENSHTEIN)
# ============================================

def damerau_levenshtein_distance(s1: str, s2: str) -> int:
    """
    Compute Damerau-Levenshtein distance between two strings,
    supporting insertions, deletions, substitutions, and adjacent transpositions.
    """
    if s1 == s2:
        return 0
    len1 = len(s1)
    len2 = len(s2)
    if not len1:
        return len2
    if not len2:
        return len1

    d = {}
    for i in range(-1, len1 + 1):
        d[(i, -1)] = i + 1
    for j in range(-1, len2 + 1):
        d[(-1, j)] = j + 1

    for i in range(len1):
        for j in range(len2):
            cost = 0 if s1[i] == s2[j] else 1
            d[(i, j)] = min(
                d[(i - 1, j)] + 1,        # deletion
                d[(i, j - 1)] + 1,        # insertion
                d[(i - 1, j - 1)] + cost, # substitution
            )
            if i > 0 and j > 0 and s1[i] == s2[j - 1] and s1[i - 1] == s2[j]:
                d[(i, j)] = min(d[(i, j)], d[(i - 2, j - 2)] + 1)  # transposition

    return d[(len1 - 1, len2 - 1)]


def _is_one_edit_away(a: str, b: str) -> bool:
    """Detect single edit distance (insertion, deletion, substitution, transposition)."""
    return damerau_levenshtein_distance(a, b) == 1


# ============================================
# 8. BRAND IMPERSONATION DETECTION
# ============================================

def detect_brand_impersonation(
    hostname: str,
    domain_name: Optional[str] = None,
    registered_domain: Optional[str] = None,
) -> dict:
    """
    Compare a hostname against configured reference brand domains.

    Evaluates:
    1. Exact official reference domain or subdomain match.
    2. Misleading hostname structures (e.g. youtube.com.attacker.example).
    3. Brand names combined with keywords/hyphens in registered domain (e.g. youtube-login.com).
    4. Exact character/leet substitutions (e.g. g00gle.com, paypa1.com, youtub3.com).
    5. Single-edit typographical variations (Damerau-Levenshtein: deletion, insertion, substitution, transposition).
    6. Unicode homoglyphs and Punycode spoofing.

    Returns:
        dict with keys:
            brand_name: Optional[str] (always canonical name, e.g. "YouTube")
            brand_reference_domain: Optional[str] (e.g. "youtube.com")
            brand_domain_status: "official_reference" | "possible_impersonation" | "unrecognized"
            brand_detection_reason: str
    """
    result = {
        "brand_name": None,
        "brand_reference_domain": None,
        "brand_domain_status": "unrecognized",
        "brand_detection_reason": "No configured brand match detected.",
    }

    normalized_host = _normalize_hostname(hostname)
    if not normalized_host or _is_ip_address(normalized_host):
        return result

    # 1. Check exact official domain or legitimate subdomain
    for brand, data in BRAND_REGISTRY.items():
        for official_domain in data["domains"]:
            if normalized_host == official_domain or normalized_host.endswith("." + official_domain):
                result.update({
                    "brand_name": brand,
                    "brand_reference_domain": official_domain,
                    "brand_domain_status": "official_reference",
                    "brand_detection_reason": (
                        "Hostname matches a configured reference domain or its subdomain. "
                        "This is not a safety guarantee."
                    ),
                })
                return result

    # Decode Punycode / IDNA and convert homoglyphs to skeleton
    skeleton = _domain_skeleton(normalized_host)

    # PSL extraction for registered domain and domain label
    if not domain_name or not registered_domain:
        extracted = _tld_extractor(skeleton)
        reg_domain = f"{extracted.domain}.{extracted.suffix}" if extracted.domain and extracted.suffix else extracted.domain
        reg_label = extracted.domain.lower()
    else:
        reg_domain = registered_domain.lower()
        reg_label = domain_name.lower()

    # 2. Misleading hostname structures: embedded reference domain
    # e.g. youtube.com.evil-example.org or google.com.login.example
    for brand, data in BRAND_REGISTRY.items():
        primary_domain = data["domains"][0]
        for official_domain in data["domains"]:
            if (official_domain + ".") in (skeleton + ".") or (f".{official_domain}." in f".{skeleton}."):
                if not (normalized_host == official_domain or normalized_host.endswith("." + official_domain)):
                    result.update({
                        "brand_name": brand,
                        "brand_reference_domain": primary_domain,
                        "brand_domain_status": "possible_impersonation",
                        "brand_detection_reason": (
                            "Hostname embeds a reference domain inside an untrusted or different hostname."
                        ),
                    })
                    return result

    # 3. Check registered domain label against brand registry
    for brand, data in BRAND_REGISTRY.items():
        primary_domain = data["domains"][0]
        brand_label = data["primary_label"]

        # 3a. Brand token embedded in registered domain (e.g. youtube-login, login-youtube)
        if reg_label != brand_label:
            label_tokens = reg_label.split("-")
            if (
                brand_label in label_tokens
                or reg_label.startswith(brand_label + "-")
                or reg_label.endswith("-" + brand_label)
            ):
                result.update({
                    "brand_name": brand,
                    "brand_reference_domain": primary_domain,
                    "brand_domain_status": "possible_impersonation",
                    "brand_detection_reason": (
                        "Hostname contains a popular brand name but does not match its configured reference domain."
                    ),
                })
                return result

        # 3b. Exact leet / numerical character substitution (e.g. g00gle, paypa1, youtub3, y0utube, micros0ft, app1e)
        reg_leet = reg_label.translate(LEET_TRANSLATION)
        if reg_leet == brand_label:
            result.update({
                "brand_name": brand,
                "brand_reference_domain": primary_domain,
                "brand_domain_status": "possible_impersonation",
                "brand_detection_reason": (
                    "Hostname resembles a popular brand through character substitutions."
                ),
            })
            return result

        # 3c. Damerau-Levenshtein edit distance = 1 (typosquatting: deletion, insertion, substitution, transposition)
        if len(brand_label) >= 5:
            dist = damerau_levenshtein_distance(reg_label, brand_label)
            if dist == 1:
                if len(reg_label) < len(brand_label):
                    reason = f"Hostname is a single-character deletion from {brand}."
                elif len(reg_label) > len(brand_label):
                    reason = f"Hostname contains an extra character compared to {brand}."
                elif sorted(reg_label) == sorted(brand_label):
                    reason = f"Hostname contains transposed characters compared to {brand}."
                else:
                    reason = f"Hostname contains a single-character substitution compared to {brand}."

                result.update({
                    "brand_name": brand,
                    "brand_reference_domain": primary_domain,
                    "brand_domain_status": "possible_impersonation",
                    "brand_detection_reason": reason,
                })
                return result

            # Also check if leet-translated label is 1 edit away
            leet_dist = damerau_levenshtein_distance(reg_leet, brand_label)
            if leet_dist == 1:
                result.update({
                    "brand_name": brand,
                    "brand_reference_domain": primary_domain,
                    "brand_domain_status": "possible_impersonation",
                    "brand_detection_reason": (
                        f"Hostname resembles {brand} through character substitutions and typo variations."
                    ),
                })
                return result

    return result


# ============================================
# 9. MAIN FEATURE EXTRACTION
# ============================================

def extract_url_features(url: str) -> Dict[str, Any]:

    parsed, lexical_url, scheme_status = (
        _parse_input_url(url)
    )

    hostname = parsed.hostname or ""

    # ----------------------------------------
    # HOMEPAGE PATH NORMALIZATION
    # ----------------------------------------

    if parsed.path == "":
        canonical_path = "/"
        if "?" in lexical_url:
            base, rest = lexical_url.split("?", 1)
            if not base.endswith("/"):
                lexical_url = base + "/?" + rest
        elif "#" in lexical_url:
            base, rest = lexical_url.split("#", 1)
            if not base.endswith("/"):
                lexical_url = base + "/#" + rest
        elif not lexical_url.endswith("/"):
            lexical_url = lexical_url + "/"
    else:
        canonical_path = parsed.path

    # ----------------------------------------
    # PATH FEATURES
    # ----------------------------------------

    path_segments = [
        segment
        for segment in canonical_path.strip("/").split("/")
        if segment
    ]

    num_path_segments = len(path_segments)

    # ----------------------------------------
    # QUERY FEATURES
    # ----------------------------------------

    query_parameters = [
        parameter
        for parameter in parsed.query.split("&")
        if parameter
    ]

    num_query_parameters = len(query_parameters)

    # ----------------------------------------
    # PROTOCOL AND HOST CHECKS
    # ----------------------------------------

    has_ip_address = int(
        _is_ip_address(hostname)
    )

    is_https = int(
        scheme_status == "https"
    )

    # ----------------------------------------
    # SUSPICIOUS KEYWORDS
    # ----------------------------------------

    lower_url = lexical_url.lower()

    detected_keywords = [
        keyword
        for keyword in SUSPICIOUS_KEYWORDS
        if keyword in lower_url
    ]

    # ----------------------------------------
    # BRAND IMPERSONATION
    # ----------------------------------------

    brand_info = detect_brand_impersonation(
        hostname
    )

    # ----------------------------------------
    # ORIGINAL NUMERICAL FEATURES
    # ----------------------------------------

    features = {

        "url_length": len(lexical_url),

        "num_dots": lexical_url.count("."),

        "num_hyphens": lexical_url.count("-"),

        "num_underscores": lexical_url.count("_"),

        "num_slashes": lexical_url.count("/"),

        "num_question_marks": lexical_url.count("?"),

        "num_equal_signs": lexical_url.count("="),

        "num_at_symbols": lexical_url.count("@"),

        "num_percent_symbols": lexical_url.count("%"),

        "num_path_segments": num_path_segments,

        "path_length": len(canonical_path),

        "query_length": len(parsed.query),

        "num_query_parameters": num_query_parameters,

        "hostname_length": len(hostname),

        "has_ip_address": has_ip_address,

        "is_https": is_https,

        "suspicious_keyword_count": len(
            detected_keywords
        ),

        # ------------------------------------
        # ORIGINAL METADATA
        # ------------------------------------

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

        # ------------------------------------
        # NEW BRAND ANALYSIS METADATA
        # ------------------------------------

        "brand_name": brand_info["brand_name"],

        "brand_reference_domain": (
            brand_info["brand_reference_domain"]
        ),

        "brand_domain_status": (
            brand_info["brand_domain_status"]
        ),

        "brand_detection_reason": (
            brand_info["brand_detection_reason"]
        ),
    }

    return features
