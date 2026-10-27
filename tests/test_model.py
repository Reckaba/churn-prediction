import json

import pytest

from churn.pipeline import FEATURES, METADATA_PATH, build_pipeline, load_data


def test_data_has_no_missing_values_after_cleaning():
    X, y = load_data()
    assert len(X) == 7043
    assert X.isna().sum().sum() == 0
    assert set(y.unique()) == {0, 1}


def test_saved_model_quality_does_not_regress():
    metrics = json.loads(METADATA_PATH.read_text())["metrics"]
    assert metrics["test_roc_auc"] >= 0.83


@pytest.mark.filterwarnings("ignore:Found unknown categories")
def test_unknown_category_does_not_crash():
    X, y = load_data()
    pipe = build_pipeline().fit(X[FEATURES], y)
    row = X[FEATURES].iloc[[0]].copy()
    row["PaymentMethod"] = "Crypto"
    proba = pipe.predict_proba(row)[0, 1]
    assert 0.0 <= proba <= 1.0
