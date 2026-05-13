# main.py

from src.preprocess import load_data, split_data, build_preprocessor
from src.train import train_logistic_regression, train_random_forest
from src.evaluate import evaluate_model
from src.save_artifacts import (
    save_pipeline,
    save_model_and_preprocessor,
    save_experiment_log
)


def print_metrics(model_name, metrics):
    print(f"\n=== {model_name} Metrics ===")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")


def main():
    # ==========================================================
    # 1. LOAD AND PREPROCESS DATA
    # ==========================================================
    print("Loading data...")
    df = load_data("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

    print("Splitting data...")
    X_train, X_test, y_train, y_test = split_data(df)

    print("Building preprocessor...")
    preprocessor = build_preprocessor(X_train)

    # ==========================================================
    # 2. TRAIN MODELS
    # ==========================================================
    print("Training Logistic Regression...")
    lr_pipeline = train_logistic_regression(
        preprocessor,
        X_train,
        y_train
    )

    print("Training Random Forest...")
    rf_pipeline = train_random_forest(
        preprocessor,
        X_train,
        y_train
    )

    # ==========================================================
    # 3. EVALUATE MODELS
    # ==========================================================
    print("Evaluating Logistic Regression...")
    lr_metrics = evaluate_model(lr_pipeline, X_test, y_test)

    print("Evaluating Random Forest...")
    rf_metrics = evaluate_model(rf_pipeline, X_test, y_test)

    print_metrics("Logistic Regression", lr_metrics)
    print_metrics("Random Forest", rf_metrics)

    # ==========================================================
    # 4. SELECT BEST MODEL
    # ==========================================================
    if rf_metrics["roc_auc"] >= lr_metrics["roc_auc"]:
        best_pipeline = rf_pipeline
        best_metrics = rf_metrics
        best_model_name = "RandomForest"
    else:
        best_pipeline = lr_pipeline
        best_metrics = lr_metrics
        best_model_name = "LogisticRegression"

    print(f"\nBest model selected: {best_model_name}")

    # ==========================================================
    # 5. SAVE ARTIFACTS
    # ==========================================================
    print("Saving artifacts...")
    save_pipeline(best_pipeline)
    save_model_and_preprocessor(best_pipeline)

    # ==========================================================
    # 6. SAVE EXPERIMENT LOG
    # ==========================================================
    experiment_log = {
        "model_name": best_model_name,
        "accuracy": best_metrics["accuracy"],
        "precision": best_metrics["precision"],
        "recall": best_metrics["recall"],
        "f1_score": best_metrics["f1_score"],
        "roc_auc": best_metrics["roc_auc"]
    }

    # Save some model settings
    model = best_pipeline.named_steps["model"]

    if hasattr(model, "get_params"):
        params = model.get_params()
        experiment_log["model_params"] = str(params)

    save_experiment_log(experiment_log)

    print("Training pipeline completed successfully.")
    print("Artifacts saved in the 'artifacts/' directory.")


if __name__ == "__main__":
    main()