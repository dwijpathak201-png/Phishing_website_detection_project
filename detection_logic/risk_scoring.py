"""PhishGuard - Explainable Hybrid Risk Scoring Engine (V3).

The score is experimental. Numerical class 1 is not verified as phishing,
Random Forest probabilities are uncalibrated, and thresholds are provisional.

This module does not perform network requests. It evaluates the supplied report.
"""

from math import isfinite

RISK_THRESHOLD_MEDIUM = 35.0
RISK_THRESHOLD_HIGH = 70.0


def _number(value):
    """Accept finite numeric evidence without interpreting missing data as zero."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        if not isfinite(value):
            return None
    except (TypeError, ValueError, OverflowError):
        return None
    return value


def _evaluate_rules(report):
    """Return (capped_score, triggered_findings, all_rule_evaluations)."""
    url = report.get("url_features") or {}
    domain = report.get("domain_features") or {}
    webpage = report.get("webpage_features") or {}

    findings = []
    evaluations = []

    def record(rule_id, title, feature, observed, triggered, points,
               description, checked=True):
        status = (
            "not_checked" if not checked
            else "triggered" if triggered
            else "not_triggered"
        )
        awarded_points = float(points) if status == "triggered" else 0.0
        row = {
            "rule_id": rule_id,
            "title": title,
            "feature": feature,
            "status": status,
            "observed_value": observed,
            "points": awarded_points,
            "maximum_points": float(points),
            "description": description,
        }
        evaluations.append(row)
        if status == "triggered":
            findings.append({
                "feature": feature,
                "description": description,
                "points": awarded_points,
            })

    # 1. Literal IP address as the host.
    ip_host = _number(url.get("has_ip_address"))
    record(
        "ip_hostname", "IP address used as hostname", "has_ip_address",
        ip_host, ip_host == 1, 25,
        "URL uses an IP address as hostname.", checked=ip_host is not None,
    )

    # 2. The @ delimiter can conceal the effective host in a URL.
    at_count = _number(url.get("num_at_symbols"))
    record(
        "at_symbol", "@ symbol in URL", "num_at_symbols",
        at_count, at_count is not None and at_count > 0, 20,
        "URL contains an @ symbol.", checked=at_count is not None,
    )

    # 3. Internationalized-domain encoding is an indicator, not a verdict.
    punycode = _number(domain.get("has_punycode"))
    record(
        "punycode", "Punycode in domain", "has_punycode",
        punycode, punycode == 1, 8,
        "Domain contains punycode characters.", checked=punycode is not None,
    )

    # 4. Many nested subdomains can be suspicious in some URL structures.
    subdomains = _number(domain.get("subdomain_count"))
    record(
        "subdomains", "Multiple subdomain levels", "subdomain_count",
        subdomains, subdomains is not None and subdomains >= 3, 12,
        "Domain contains multiple subdomain levels.",
        checked=subdomains is not None,
    )

    # 5. Keyword matches: 12 points each, capped at 36.
    keywords = _number(url.get("suspicious_keyword_count"))
    keyword_points = min(keywords * 12.0, 36.0) if keywords is not None and keywords > 0 else 0.0
    record(
        "keywords", "Suspicious URL keywords", "suspicious_keyword_count",
        keywords, keywords is not None and keywords > 0,
        keyword_points,
        "Suspicious keywords detected in URL.",
        checked=keywords is not None,
    )
    # The cap for this rule is 36, regardless of the number detected.
    evaluations[-1]["maximum_points"] = 36.0

    # 6. Unknown scheme is not assumed to be HTTP.
    scheme = url.get("scheme_status")
    scheme_checked = scheme in ("http", "https")
    record(
        "http", "Explicit HTTP connection", "scheme_status",
        scheme if scheme is not None else "unknown",
        scheme == "http", 12,
        "Website URL explicitly uses HTTP instead of HTTPS.",
        checked=scheme_checked,
    )

    # 7. Password field transmitted on explicitly HTTP URL.
    password_field = _number(webpage.get("page_has_password_field"))
    password_checked = password_field is not None and (scheme_checked or password_field == 0)
    record(
        "password_http", "Password field over HTTP", "page_has_password_field",
        password_field,
        password_field == 1 and scheme == "http", 28,
        "Password field detected on an HTTP page.",
        checked=password_checked,
    )

    # 8. Form submits data to a different origin.
    external_form = _number(webpage.get("external_action_forms"))
    record(
        "external_form", "External form submission", "external_action_forms",
        external_form, external_form == 1, 20,
        "Form submits data to an external origin.",
        checked=external_form is not None,
    )

    score = min(sum(finding["points"] for finding in findings), 100.0)
    return round(score, 1), findings, evaluations


def _rule_findings(report):
    """Keep the existing two-value private helper for compatibility."""
    score, findings, _ = _evaluate_rules(report)
    return score, findings


def calculate_rule_score(report):
    """Return the provisional 0–100 rule-based score."""
    score, _, _ = _evaluate_rules(report)
    return score


def evaluate_risk(report, ml_prediction, ml_probability, *, phishing_label):
    """Calculate risk and expose how each available rule was evaluated.

    ml_probability is an uncalibrated estimate for numerical class 1.
    phishing_label is a provisional mapping until dataset provenance is checked.
    """
    if phishing_label not in (0, 1):
        raise ValueError("phishing_label must be 0 or 1.")
    if ml_prediction not in (0, 1):
        raise ValueError("ml_prediction must be 0 or 1.")

    try:
        ml_probability = float(ml_probability)
    except (TypeError, ValueError) as exc:
        raise ValueError("ml_probability must be a number between 0 and 1.") from exc

    if not isfinite(ml_probability) or not 0.0 <= ml_probability <= 1.0:
        raise ValueError("ml_probability must be between 0 and 1.")

    phishing_probability = (
        ml_probability if phishing_label == 1
        else 1.0 - ml_probability
    )
    ml_score = phishing_probability * 100.0
    rule_score, findings, evaluations = _evaluate_rules(report)

    weighted_score = 0.60 * ml_score + 0.40 * rule_score
    combined_score = max(ml_score, weighted_score)
    final_score = round(min(max(combined_score, 0.0), 100.0), 1)

    if final_score >= RISK_THRESHOLD_HIGH:
        level, verdict = "High", "Elevated risk"
    elif final_score >= RISK_THRESHOLD_MEDIUM:
        level, verdict = "Medium", "Needs review"
    else:
        level, verdict = "Low", "Lower observed risk"

    return {
        "risk_score": final_score,
        "risk_level": level,
        "assessment_verdict": verdict,
        "model_score": round(ml_score, 1),
        "rule_score": rule_score,
        "ml_predicted_label": ml_prediction,
        "ml_predicts_phishing": ml_prediction == phishing_label,
        "phishing_class_label": phishing_label,
        "rule_findings": findings,
        "score_is_calibrated": False,
        # Added V3 explanatory fields; existing return keys are unchanged.
        "rule_evaluations": evaluations,
        "rules_triggered": sum(row["status"] == "triggered" for row in evaluations),
        "rules_not_triggered": sum(row["status"] == "not_triggered" for row in evaluations),
        "rules_not_checked": sum(row["status"] == "not_checked" for row in evaluations),
        "scoring_breakdown": {
            "model_score": round(ml_score, 1),
            "rule_score": rule_score,
            "weighted_score": round(weighted_score, 1),
            "final_score": final_score,
            "formula": "max(model_score, 0.60 * model_score + 0.40 * rule_score)",
            "is_calibrated": False,
        },
    }
