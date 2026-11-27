
"""
PhishGuard - Explainable Hybrid Risk Scoring Engine (V3.1).

Preserves Random Forest V3 scoring inputs and the existing
hybrid scoring formula.

Adds:
1. Brand impersonation detection
2. Inconclusive model assessment
3. Nine explainable cybersecurity rules
4. Backward-compatible report fields

IMPORTANT:
The class-label mapping and scoring thresholds remain
provisional. This module makes no network requests.
"""

from math import isfinite
from urllib.parse import urlsplit


# ==============================================
# 1. CONFIGURATION
# ==============================================

RISK_THRESHOLD_MEDIUM = 35.0
RISK_THRESHOLD_HIGH = 70.0

# Provisional uncertainty range
MODEL_INCONCLUSIVE_MIN = 40.0
MODEL_INCONCLUSIVE_MAX = 60.0

# Provisional impersonation penalty
BRAND_IMPERSONATION_POINTS = 50.0


# ==============================================
# 2. NUMERIC VALIDATION
# ==============================================

def _number(value):

    if isinstance(value, bool):
        return None

    if not isinstance(value, (int, float)):
        return None

    return value if isfinite(value) else None


# ==============================================
# 3. BRAND METADATA
# ==============================================

def _brand_metadata(url_features, domain_features):
    """
    Retrieve brand analysis from the URL
    and domain feature extractors.

    Possible impersonation takes priority
    over a conflicting reference match.
    """

    sources = (domain_features, url_features)

    for status in (
        "possible_impersonation",
        "official_reference",
        "unrecognized",
    ):

        for source in sources:

            if source.get("brand_domain_status") == status:

                return {
                    "status": status,
                    "brand_name": source.get("brand_name"),
                    "reference_domain": source.get(
                        "brand_reference_domain"
                    ),
                    "reason": source.get(
                        "brand_detection_reason"
                    ) or "",
                }

    return {
        "status": "not_checked",
        "brand_name": None,
        "reference_domain": None,
        "reason": "Brand metadata unavailable.",
    }


# ==============================================
# 4. RULE EVALUATION ENGINE
# ==============================================

def _evaluate_rules(report):

    url = report.get("url_features") or {}
    domain = report.get("domain_features") or {}
    webpage = report.get("webpage_features") or {}

    findings = []
    evaluations = []

    def record(
        rule_id,
        title,
        feature,
        observed,
        triggered,
        maximum_points,
        description,
        checked=True,
        awarded_points=None,
    ):

        if not checked:
            status = "not_checked"

        elif triggered:
            status = "triggered"

        else:
            status = "not_triggered"

        if status == "triggered":

            points = float(
                awarded_points
                if awarded_points is not None
                else maximum_points
            )

        else:
            points = 0.0

        evaluations.append({
            "rule_id": rule_id,
            "title": title,
            "feature": feature,
            "status": status,
            "observed_value": observed,
            "points": points,
            "maximum_points": float(maximum_points),
            "description": description,
        })

        if status == "triggered":

            findings.append({
                "feature": feature,
                "description": description,
                "points": points,
            })

    # ------------------------------------------
    # RULE 1: IP HOSTNAME
    # ------------------------------------------

    ip_host = _number(
        url.get("has_ip_address")
    )

    record(
        "ip_hostname",
        "IP address used as hostname",
        "has_ip_address",
        ip_host,
        ip_host == 1,
        25,
        "URL uses an IP address as hostname.",
        checked=ip_host is not None,
    )

    # ------------------------------------------
    # RULE 2: @ SYMBOL
    # ------------------------------------------

    at_count = _number(
        url.get("num_at_symbols")
    )

    record(
        "at_symbol",
        "@ symbol in URL",
        "num_at_symbols",
        at_count,
        at_count is not None and at_count > 0,
        20,
        "URL contains an @ symbol.",
        checked=at_count is not None,
    )

    # ------------------------------------------
    # RULE 3: PUNYCODE
    # ------------------------------------------

    punycode = _number(
        domain.get("has_punycode")
    )

    record(
        "punycode",
        "Punycode in domain",
        "has_punycode",
        punycode,
        punycode == 1,
        8,
        "Domain contains punycode characters.",
        checked=punycode is not None,
    )

    # ------------------------------------------
    # RULE 4: SUBDOMAINS
    # ------------------------------------------

    subdomains = _number(
        domain.get("subdomain_count")
    )

    record(
        "subdomains",
        "Multiple subdomain levels",
        "subdomain_count",
        subdomains,
        subdomains is not None and subdomains >= 3,
        12,
        "Domain contains multiple subdomain levels.",
        checked=subdomains is not None,
    )

    # ------------------------------------------
    # RULE 5: SUSPICIOUS KEYWORDS
    # ------------------------------------------

    keywords = _number(
        url.get("suspicious_keyword_count")
    )

    keyword_points = (
        min(keywords * 12.0, 36.0)
        if keywords is not None and keywords > 0
        else 0.0
    )

    record(
        "keywords",
        "Suspicious URL keywords",
        "suspicious_keyword_count",
        keywords,
        keywords is not None and keywords > 0,
        36,
        "Suspicious keywords detected in URL.",
        checked=keywords is not None,
        awarded_points=keyword_points,
    )

    # ------------------------------------------
    # RULE 6: HTTP CONNECTION
    # ------------------------------------------

    scheme = url.get("scheme_status")

    scheme_checked = scheme in (
        "http",
        "https",
    )

    brand = _brand_metadata(
        url,
        domain,
    )

    is_clean_official = (
        brand.get("status") == "official_reference"
        and (_number(url.get("suspicious_keyword_count")) or 0) == 0
        and (_number(url.get("has_ip_address")) or 0) == 0
        and (_number(domain.get("has_punycode")) or 0) == 0
    )

    if is_clean_official and scheme == "http":
        # Official reference domain over HTTP:
        # HTTP alone does not penalize an exact official reference with heuristic phishing points.
        # Transport security advisory is disclosed separately without heuristic penalty.
        record(
            "http",
            "Explicit HTTP connection",
            "scheme_status",
            scheme,
            False,
            12,
            "Website URL explicitly uses unencrypted HTTP instead of HTTPS; transport security advisory noted separately without heuristic phishing penalty.",
            checked=scheme_checked,
            awarded_points=0.0,
        )
    else:
        record(
            "http",
            "Explicit HTTP connection",
            "scheme_status",
            scheme if scheme is not None else "unknown",
            scheme == "http",
            12,
            "Website URL explicitly uses HTTP instead of HTTPS.",
            checked=scheme_checked,
        )

    # ------------------------------------------
    # RULE 7: PASSWORD FIELD OVER HTTP
    # ------------------------------------------

    password_field = _number(
        webpage.get("page_has_password_field")
    )

    effective_url = (
        webpage.get("final_url")
        or webpage.get("effective_url")
    )

    effective_scheme = None

    if isinstance(effective_url, str):

        effective_scheme = urlsplit(
            effective_url
        ).scheme.lower()

    page_scheme = (
        effective_scheme
        if effective_scheme in ("http", "https")
        else scheme
    )

    password_checked = (
        password_field is not None
        and (
            page_scheme in ("http", "https")
            or password_field == 0
        )
    )

    record(
        "password_http",
        "Password field over HTTP",
        "page_has_password_field",
        password_field,
        (
            password_field == 1
            and page_scheme == "http"
        ),
        28,
        "Password field detected on an HTTP page.",
        checked=password_checked,
    )

    # ------------------------------------------
    # RULE 8: EXTERNAL FORM SUBMISSION
    # ------------------------------------------

    external_form = _number(
        webpage.get("external_action_forms")
    )

    record(
        "external_form",
        "External form submission",
        "external_action_forms",
        external_form,
        external_form == 1,
        20,
        "Form submits data to an external origin.",
        checked=external_form is not None,
    )

    # ------------------------------------------
    # RULE 9: BRAND IMPERSONATION
    # ------------------------------------------

    brand = _brand_metadata(
        url,
        domain,
    )

    brand_warning = (
        brand["status"] == "possible_impersonation"
    )

    if brand_warning:

        description = (
            f"Possible impersonation of "
            f"{brand['brand_name']} "
            f"({brand['reference_domain']}). "
            f"{brand['reason']}"
        )

    else:

        description = (
            "No configured brand impersonation "
            "indicator detected. This does not "
            "establish website safety."
        )

    record(
        "brand_impersonation",
        "Possible brand impersonation",
        "brand_domain_status",
        brand["status"],
        brand_warning,
        BRAND_IMPERSONATION_POINTS,
        description,
        checked=brand["status"] != "not_checked",
    )

    # ------------------------------------------
    # FINAL RULE SCORE
    # ------------------------------------------

    score = min(
        sum(
            item["points"]
            for item in findings
        ),
        100.0,
    )

    return round(score, 1), findings, evaluations


# ==============================================
# 5. COMPATIBILITY FUNCTIONS
# ==============================================

def _rule_findings(report):

    score, findings, _ = _evaluate_rules(report)

    return score, findings


def calculate_rule_score(report):

    score, _, _ = _evaluate_rules(report)

    return score


# ==============================================
# 6. HYBRID RISK ASSESSMENT
# ==============================================

def evaluate_risk(
    report,
    ml_prediction,
    ml_probability,
    *,
    phishing_label,
):
    """
    Calculate provisional risk assessment.

    Preserves the existing hybrid scoring
    formula and adds explanatory statuses.
    """

    # ------------------------------------------
    # VALIDATE INPUTS
    # ------------------------------------------

    if phishing_label not in (0, 1):

        raise ValueError(
            "phishing_label must be 0 or 1."
        )

    if ml_prediction not in (0, 1):

        raise ValueError(
            "ml_prediction must be 0 or 1."
        )

    try:

        ml_probability = float(
            ml_probability
        )

    except (TypeError, ValueError, OverflowError) as exc:

        raise ValueError(
            "ml_probability must be between 0 and 1."
        ) from exc

    if (
        not isfinite(ml_probability)
        or not 0.0 <= ml_probability <= 1.0
    ):

        raise ValueError(
            "ml_probability must be between 0 and 1."
        )

    # ------------------------------------------
    # ML SCORE & OFFICIAL DOMAIN CALIBRATION
    # ------------------------------------------

    phishing_probability = (
        ml_probability
        if phishing_label == 1
        else 1.0 - ml_probability
    )

    raw_ml_score = (
        phishing_probability * 100.0
    )

    brand = _brand_metadata(
        report.get("url_features") or {},
        report.get("domain_features") or {},
    )

    url_f = report.get("url_features") or {}
    domain_f = report.get("domain_features") or {}
    webpage_f = report.get("webpage_features") or {}

    is_official_reference = (brand.get("status") == "official_reference")

    keywords_count = _number(url_f.get("suspicious_keyword_count")) or 0
    has_ip = _number(url_f.get("has_ip_address")) or 0
    has_punycode = _number(domain_f.get("has_punycode")) or 0
    path = url_f.get("path") or "/"
    path_segments = [s for s in path.strip("/").split("/") if s]
    query = url_f.get("query") or ""
    password_field = _number(webpage_f.get("page_has_password_field")) or 0
    scheme = url_f.get("scheme_status") or "unknown"
    external_form = _number(webpage_f.get("external_action_forms")) or 0

    has_independent_suspicious_signals = bool(
        keywords_count > 0
        or has_ip == 1
        or has_punycode == 1
        or (password_field == 1 and scheme == "http")
        or external_form == 1
        or len(path_segments) > 2
        or len(query) > 100
    )

    is_clean_official_homepage = (
        is_official_reference
        and not has_independent_suspicious_signals
    )

    if is_clean_official_homepage:
        # Verified official reference domain homepage with no independent suspicious indicators:
        # Uncalibrated Random Forest tree vote (~50-68%) is biased by underrepresentation
        # of clean brand homepages in training data. Under application scoring policy,
        # calibrate model risk index to baseline low risk (10.0/100, targeting 0-20/100 Green / Low Risk).
        ml_score = 10.0
        is_calibrated = True
        calibration_rationale = (
            "Verified official reference domain homepage with no independent suspicious indicators "
            "calibrated to baseline low risk (10.0/100)."
        )
    else:
        ml_score = raw_ml_score
        is_calibrated = False
        calibration_rationale = "Raw uncalibrated Random Forest class-1 probability index."

    # ------------------------------------------
    # RULE SCORE
    # ------------------------------------------

    rule_score, findings, evaluations = (
        _evaluate_rules(report)
    )

    # ------------------------------------------
    # EXISTING HYBRID FORMULA
    # ------------------------------------------

    weighted_score = (
        0.60 * ml_score
        + 0.40 * rule_score
    )

    combined_score = max(
        ml_score,
        weighted_score,
    )

    final_score = round(
        min(
            max(combined_score, 0.0),
            100.0,
        ),
        1,
    )

    # ------------------------------------------
    # NUMERICAL RISK LEVEL
    # ------------------------------------------

    if final_score >= RISK_THRESHOLD_HIGH:

        level = "High"
        verdict = "Elevated risk"

    elif final_score >= RISK_THRESHOLD_MEDIUM:

        level = "Medium"
        verdict = "Needs review"

    else:

        level = "Low"
        verdict = (
            "Verified official reference domain homepage"
            if is_clean_official_homepage
            else "Lower observed risk"
        )

    # ------------------------------------------
    # INCONCLUSIVE MODEL DETECTION
    # ------------------------------------------

    # Inconclusive describes model uncertainty when an uncalibrated model score
    # falls in 40-60%. A calibrated official reference domain (10.0) is not inconclusive.
    inconclusive = (
        not is_calibrated
        and (
            MODEL_INCONCLUSIVE_MIN
            <= ml_score
            <= MODEL_INCONCLUSIVE_MAX
        )
    )

    # ------------------------------------------
    # BRAND IMPERSONATION DETECTION & WARNINGS
    # ------------------------------------------

    brand_alert = (
        brand["status"] == "possible_impersonation"
    )

    warnings = []

    if brand_alert:

        warnings.append(
            "Possible brand impersonation: "
            + brand["reason"]
        )

    if inconclusive:

        warnings.append(
            "Model estimate is near 50%; "
            "additional verification is needed."
        )

    if scheme == "http":

        warnings.append(
            "Transport security warning: Website URL explicitly uses unencrypted HTTP. "
            "Data in transit is unencrypted and vulnerable to interception."
        )

    # ------------------------------------------
    # EXPLANATORY ASSESSMENT STATUS
    # ------------------------------------------

    if brand_alert:

        assessment_status = (
            "potential_impersonation"
        )

        verdict = (
            "Potential brand impersonation - "
            "verify the domain"
        )

    elif inconclusive:

        assessment_status = "inconclusive"

        verdict = (
            "Inconclusive - additional "
            "verification required"
        )

    else:

        assessment_status = "provisional"

    # ------------------------------------------
    # FINAL REPORT
    # ------------------------------------------

    return {

        # Existing fields
        "risk_score": final_score,

        "risk_level": level,

        "assessment_verdict": verdict,

        "model_score": round(
            ml_score, 1
        ),

        "raw_model_score": round(
            raw_ml_score, 1
        ),

        "rule_score": rule_score,

        "ml_predicted_label": ml_prediction,

        "ml_predicts_phishing": (
            ml_prediction == phishing_label
        ),

        "phishing_class_label": phishing_label,

        "rule_findings": findings,

        "score_is_calibrated": is_calibrated,

        # Rule breakdown
        "rule_evaluations": evaluations,

        "rules_triggered": sum(
            row["status"] == "triggered"
            for row in evaluations
        ),

        "rules_not_triggered": sum(
            row["status"] == "not_triggered"
            for row in evaluations
        ),

        "rules_not_checked": sum(
            row["status"] == "not_checked"
            for row in evaluations
        ),

        # Hybrid scoring explanation
        "scoring_breakdown": {

            "raw_model_score": round(
                raw_ml_score, 1
            ),

            "model_score": round(
                ml_score, 1
            ),

            "rule_score": rule_score,

            "weighted_score": round(
                weighted_score, 1
            ),

            "final_score": final_score,

            "formula": (
                "max(model_score, "
                "0.60 * model_score + "
                "0.40 * rule_score)"
            ),

            "is_calibrated": is_calibrated,

            "calibration_rationale": calibration_rationale,
        },

        # New V3.1 fields
        "assessment_status": (
            assessment_status
        ),

        "assessment_warnings": warnings,

        "brand_alert": brand_alert,

        "brand_alert_details": brand,

        "model_is_inconclusive": (
            inconclusive
        ),

        "model_uncertainty_band": [
            MODEL_INCONCLUSIVE_MIN,
            MODEL_INCONCLUSIVE_MAX,
        ],
    }
