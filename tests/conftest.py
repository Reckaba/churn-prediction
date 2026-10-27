from churn.pipeline import MODEL_PATH


def pytest_sessionstart(session):
    """Train the model once if it is missing, so `pytest` works on a fresh clone."""
    if not MODEL_PATH.exists():
        import train
        train.main()
