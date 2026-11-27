"""
PhishGuard - Mandatory Comparative Regression Test Matrix (Section 6)

Compares official reference domains against typosquatting/lookalike candidates:
- YouTube (https, www, http) vs outube, youtbe, y0utube, youtub3
- Facebook vs faceb00k
- WhatsApp Web vs whatsappp
- GitHub vs githuub
- LinkedIn vs linkedln

Runs deterministically with mocked DNS.
Outputs exact required fields per test and asserts critical invariants.
"""

import sys
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ui.analysis_service import analyze_url

TEST_URLS = [
    # YouTube official variants
    "https://youtube.com",
    "https://www.youtube.com",
    "http://youtube.com",
    "http://www.youtube.com",
    # YouTube lookalikes
    "https://outube.com",
    "https://youtbe.com",
    "https://y0utube.com",
    "https://youtub3.com",
    # Facebook pair
    "https://facebook.com",
    "https://faceb00k.com",
    # WhatsApp pair
    "https://web.whatsapp.com",
    "https://whatsappp.com",
    # GitHub pair
    "https://github.com",
    "https://githuub.com",
    # LinkedIn pair
    "https://linkedin.com",
    "https://linkedln.com",
]


def run_comparative_matrix():
    print("=" * 80)
    print("PHISHGUARD — OFFICIAL VS IMPERSONATION COMPARATIVE REGRESSION MATRIX")
    print("=" * 80)

    results = []

    with patch("detection_logic.INFO_POUCH.dns_features._lookup_record", return_value=([], "mock_tested")):
        for url in TEST_URLS:
            report = analyze_url(url)

            uf = report.get("url_features") or {}
            df = report.get("domain_features") or {}
            ra = report.get("risk_assessment") or {}
            ba = report.get("brand_analysis") or {}
            sb = ra.get("scoring_breakdown") or {}

            norm_host = df.get("hostname") or uf.get("hostname") or ""
            reg_domain = df.get("registered_domain") or ""
            brand_detected = ba.get("brand_name") or "None"
            brand_status = ba.get("status") or "unrecognized"
            raw_ml = sb.get("raw_model_score")
            heuristic = sb.get("rule_score")
            final_score = ra.get("risk_score")
            risk_level = ra.get("risk_level")
            warnings = ra.get("assessment_warnings") or []
            explanation = ba.get("reason") or "No explanation"

            item = {
                "url": url,
                "norm_host": norm_host,
                "reg_domain": reg_domain,
                "brand_detected": brand_detected,
                "brand_status": brand_status,
                "raw_ml": raw_ml,
                "heuristic": heuristic,
                "final_score": final_score,
                "risk_level": risk_level,
                "warnings": warnings,
                "explanation": explanation,
                "report": report,
            }
            results.append(item)

            print(f"Input URL:                 {url}")
            print(f"Normalized hostname:       {norm_host}")
            print(f"Registered domain:         {reg_domain}")
            print(f"Brand detected:            {brand_detected}")
            print(f"Brand reference status:    {brand_status}")
            print(f"Raw ML score:              {raw_ml:.1f}/100" if raw_ml is not None else "Raw ML score:              N/A")
            print(f"Heuristic score:           {heuristic:.1f}/100" if heuristic is not None else "Heuristic score:           N/A")
            print(f"Final phishing-risk score: {final_score:.1f}/100" if final_score is not None else "Final phishing-risk score: N/A")
            print(f"Risk level:                {risk_level}")
            print(f"Assessment warnings:       {'; '.join(warnings) if warnings else 'None'}")
            print(f"Explanation:               {explanation}")
            print("-" * 80)

    # -------------------------------------------------------------
    # RIGOROUS ASSERTIONS
    # -------------------------------------------------------------
    print("\nExecuting Invariant Verifications...")

    by_url = {r["url"]: r for r in results}

    # 1. Official YouTube variants: all must be Low Risk, 0-20 score, exact match
    for yt_url in ["https://youtube.com", "https://www.youtube.com", "http://youtube.com", "http://www.youtube.com"]:
        r = by_url[yt_url]
        assert r["brand_status"] == "official_reference", f"{yt_url} status not official_reference"
        assert r["brand_detected"] == "YouTube", f"{yt_url} brand name truncated or incorrect"
        assert r["risk_level"] == "Low", f"{yt_url} risk level not Low"
        assert 0.0 <= r["final_score"] <= 20.0, f"{yt_url} score {r['final_score']} outside 0-20"
        assert r["heuristic"] == 0.0, f"{yt_url} should have 0 heuristic phishing points"

    # 2. HTTP variants: Must have transport security warning, but 0 phishing heuristic penalty
    for http_yt in ["http://youtube.com", "http://www.youtube.com"]:
        r = by_url[http_yt]
        assert any("unencrypted HTTP" in w for w in r["warnings"]), f"{http_yt} missing HTTP transport warning"
        assert r["final_score"] == by_url[http_yt.replace("http://", "https://")]["final_score"], \
            f"HTTP should have identical score to HTTPS for clean official domain"

    # 3. Lookalike variants: Must be detected as possible_impersonation with Medium/High risk
    for lookalike in ["https://outube.com", "https://youtbe.com", "https://y0utube.com", "https://youtub3.com"]:
        r = by_url[lookalike]
        assert r["brand_status"] == "possible_impersonation", f"{lookalike} status not possible_impersonation"
        assert r["brand_detected"] == "YouTube", f"{lookalike} brand name truncated or incorrect"
        assert r["risk_level"] in ("Medium", "High"), f"{lookalike} risk level not Medium/High"
        assert r["final_score"] >= 35.0, f"{lookalike} score {r['final_score']} below 35"
        assert r["final_score"] > by_url["https://youtube.com"]["final_score"], \
            f"{lookalike} score must be higher than official youtube.com"

    # 4. Facebook pair
    fb_off = by_url["https://facebook.com"]
    fb_typo = by_url["https://faceb00k.com"]
    assert fb_off["brand_status"] == "official_reference" and fb_off["risk_level"] == "Low" and fb_off["final_score"] <= 20.0
    assert fb_typo["brand_status"] == "possible_impersonation" and fb_typo["brand_detected"] == "Facebook"
    assert fb_typo["final_score"] >= 35.0 and fb_typo["final_score"] > fb_off["final_score"]

    # 5. WhatsApp pair
    wa_off = by_url["https://web.whatsapp.com"]
    wa_typo = by_url["https://whatsappp.com"]
    assert wa_off["brand_status"] == "official_reference" and wa_off["risk_level"] == "Low" and wa_off["final_score"] <= 20.0
    assert wa_typo["brand_status"] == "possible_impersonation" and wa_typo["brand_detected"] == "WhatsApp"
    assert wa_typo["final_score"] >= 35.0 and wa_typo["final_score"] > wa_off["final_score"]

    # 6. GitHub pair
    gh_off = by_url["https://github.com"]
    gh_typo = by_url["https://githuub.com"]
    assert gh_off["brand_status"] == "official_reference" and gh_off["risk_level"] == "Low" and gh_off["final_score"] <= 20.0
    assert gh_typo["brand_status"] == "possible_impersonation" and gh_typo["brand_detected"] == "GitHub"
    assert gh_typo["final_score"] >= 35.0 and gh_typo["final_score"] > gh_off["final_score"]

    # 7. LinkedIn pair
    li_off = by_url["https://linkedin.com"]
    li_typo = by_url["https://linkedln.com"]
    assert li_off["brand_status"] == "official_reference" and li_off["risk_level"] == "Low" and li_off["final_score"] <= 20.0
    assert li_typo["brand_status"] == "possible_impersonation" and li_typo["brand_detected"] == "LinkedIn"
    assert li_typo["final_score"] >= 35.0 and li_typo["final_score"] > li_off["final_score"]

    print("ALL 7 COMPARATIVE INVARIANTS PASSED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    run_comparative_matrix()
