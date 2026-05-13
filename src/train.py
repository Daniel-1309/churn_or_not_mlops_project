# src/train.py

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def train_logistic_regression(preprocessor, X_train, y_train):
    """
    Train baseline Logistic Regression model.
    """
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(max_iter=1000, random_state=42))
    ])

    pipeline.fit(X_train, y_train)
    return pipeline


def train_random_forest(preprocessor, X_train, y_train):
    """
    Train improved Random Forest model.
    """
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            random_state=42,
            n_jobs=-1
        ))
    ])

    pipeline.fit(X_train, y_train)
    return pipeline