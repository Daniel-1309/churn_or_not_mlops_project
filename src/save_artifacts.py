# src/save_artifacts.py

import os
import joblib
import pandas as pd


def save_pipeline(pipeline, artifacts_dir="artifacts"):
    """
    Save the complete sklearn pipeline.
    """
    os.makedirs(artifacts_dir, exist_ok=True)
    joblib.dump(pipeline, os.path.join(artifacts_dir, "pipeline.pkl"))


def save_model_and_preprocessor(pipeline, artifacts_dir="artifacts"):
    """
    Save model and preprocessor separately.
    """
    os.makedirs(artifacts_dir, exist_ok=True)

    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    joblib.dump(preprocessor, os.path.join(artifacts_dir, "preprocessor.pkl"))
    joblib.dump(model, os.path.join(artifacts_dir, "model.pkl"))


def save_experiment_log(log_data, artifacts_dir="artifacts"):
    """
    Append experiment metrics/settings to experiment_log.csv.
    """
    os.makedirs(artifacts_dir, exist_ok=True)

    log_path = os.path.join(artifacts_dir, "experiment_log.csv")
    df = pd.DataFrame([log_data])

    if os.path.exists(log_path):
        existing = pd.read_csv(log_path)
        df = pd.concat([existing, df], ignore_index=True)

    df.to_csv(log_path, index=False)