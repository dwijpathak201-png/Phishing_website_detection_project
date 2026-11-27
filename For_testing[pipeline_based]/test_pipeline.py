
from detection_logic.Centralized_feature_pipelines import extract_all_features


def main():
    report = extract_all_features(
        "https://example.com",
        check_live_webpage=False
    )

    print("URL:", report["url"])

    print("\nNumeric features:")
    for name, value in report["numeric_features"].items():
        print(f"{name}: {value}")

    print("\nDNS status:")
    print(report["dns_features"]["dns_status"])

    print("\nWebpage status:")
    print(report["webpage_features"]["webpage_status"])


if __name__ == "__main__":
    main()
