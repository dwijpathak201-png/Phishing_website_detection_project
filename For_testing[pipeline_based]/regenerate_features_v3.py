"""Regenerate PhishGuard V3 feature CSV from original labeled URL records.
No DNS requests, webpage fetches, or modification of source CSV.
"""
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from feature_config import FEATURE_COLUMNS, TARGET_COLUMN
from detection_logic.INFO_POUCH.url_features import extract_url_features
from detection_logic.INFO_POUCH.domain_features import extract_domain_features

FOLDER = ROOT / 'For_testing[pipeline_based]'
SOURCE = FOLDER / 'phishguard_ml_features.csv'
OUTPUT = FOLDER / 'phishguard_ml_features_v3.csv'


def main():
    if not SOURCE.is_file():
        raise FileNotFoundError(f'Original dataset missing: {SOURCE}')
    source = pd.read_csv(SOURCE, low_memory=False)
    url_column = 'url' if 'url' in source.columns else 'normalized_url'
    required = [url_column, TARGET_COLUMN]
    missing = [name for name in required if name not in source.columns]
    if missing:
        raise ValueError(f'Missing required columns: {missing}')
    rows = []
    for _, original_row in source.iterrows():
        record = original_row.to_dict()
        try:
            url = original_row[url_column]
            if pd.isna(url):
                raise ValueError('Missing URL')
            url = str(url)
            features = {**extract_url_features(url), **extract_domain_features(url)}
            for feature in FEATURE_COLUMNS:
                record[feature] = features[feature]
            record['extraction_success'] = True
            record['error'] = ''
        except Exception as exc:
            for feature in FEATURE_COLUMNS:
                record[feature] = None
            record['extraction_success'] = False
            record['error'] = str(exc)
        rows.append(record)
    result = pd.DataFrame(rows)
    if not result[TARGET_COLUMN].equals(source[TARGET_COLUMN]):
        raise ValueError('Source labels changed unexpectedly')
    result.to_csv(OUTPUT, index=False)
    successes = result['extraction_success'].sum()
    print('\n=== PHISHGUARD V3 DATASET ===')
    print('Original rows:', len(source))
    print('Successful:', int(successes))
    print('Failed:', len(source) - int(successes))
    print('Class distribution:\n', result[TARGET_COLUMN].value_counts().sort_index())
    print('Empty URL paths should now be represented as / with path_length=1.')
    if successes != len(source):
        print('Failure reasons:\n', result.loc[~result['extraction_success'], 'error'].value_counts().head(10))
        raise ValueError('Fix extraction failures before V3 training')
    print('V3 saved:', OUTPUT)
    print('Original data unchanged.')


if __name__ == '__main__':
    main()
