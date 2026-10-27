"""Train the churn model, evaluate it on a held-out test set and save it for the API.

Usage:
    python train.py            # train, evaluate, save models/churn_model.joblib
    python train.py --mlflow   # same, and log params/metrics/model to MLflow
"""
import argparse
import json
from datetime import datetime, timezone

import joblib
import numpy as np
import sklearn
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

from churn.pipeline import (FEATURES, METADATA_PATH, MODEL_DIR, MODEL_PATH, ROOT,
                            build_pipeline, load_data)

SEED = 42


def f1_optimal_threshold(y_true, proba) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    f1 = 2 * precision[:-1] * recall[:-1] / np.clip(precision[:-1] + recall[:-1], 1e-12, None)
    return float(thresholds[np.argmax(f1)])


def top_k_capture(y_true, proba, share: float = 0.2) -> float:
    """Share of all churners found among the `share` highest-risk customers."""
    order = np.argsort(-proba)
    k = int(round(share * len(proba)))
    y_sorted = np.asarray(y_true)[order]
    return float(y_sorted[:k].sum() / y_sorted.sum())


def main(use_mlflow: bool = False) -> dict:
    X, y = load_data()
    X = X[FEATURES]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    # Out-of-fold predictions on the training set: CV score and decision threshold
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = cross_val_predict(build_pipeline(), X_train, y_train, cv=cv, method="predict_proba")[:, 1]
    threshold = f1_optimal_threshold(y_train, oof)

    # Evaluate once on the untouched test set
    model = build_pipeline().fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    pred = proba >= threshold
    metrics = {
        "cv_oof_roc_auc": roc_auc_score(y_train, oof),
        "test_roc_auc": roc_auc_score(y_test, proba),
        "test_pr_auc": average_precision_score(y_test, proba),
        "test_recall": float(pred[y_test == 1].mean()),
        "test_precision": float(y_test[pred].mean()),
        "test_top20_capture": top_k_capture(y_test, proba, 0.2),
    }
    metrics = {k: round(float(v), 4) for k, v in metrics.items()}

    # Refit on all data for serving; the metrics above estimate its quality
    final_model = build_pipeline().fit(X, y)
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(final_model, MODEL_PATH)
    metadata = {
        "model": "LogisticRegression (one-hot + scaling + log(tenure))",
        "features": FEATURES,
        "threshold": round(threshold, 4),
        "metrics": metrics,
        "n_train_rows": int(len(X)),
        "sklearn_version": sklearn.__version__,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2))

    if use_mlflow:
        import mlflow
        import mlflow.sklearn

        mlflow.set_tracking_uri(f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}")
        mlflow.set_experiment("churn-prediction")
        with mlflow.start_run(run_name="logreg_final"):
            mlflow.log_params({"model": "LogisticRegression", "C": 1.0,
                               "features": ",".join(FEATURES), "threshold": metadata["threshold"]})
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(final_model, name="model", input_example=X.head(3))

    print(json.dumps({"threshold": metadata["threshold"], **metrics}, indent=2))
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mlflow", action="store_true", help="log the run to MLflow")
    main(use_mlflow=parser.parse_args().mlflow)
