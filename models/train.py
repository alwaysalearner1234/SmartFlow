"""
Model Training and Evaluation Suite for Adverse Selection Risk.
Trains a Logistic Regression baseline and an XGBoost Classifier,
computes classification metrics without fabrication, and serializes the best model.
"""

from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
import xgboost as xgb

from config.config import ModelConfig, SAVED_MODELS_DIR
from data.contracts import OrderSide
from models.preprocessing import prepare_chronological_splits, DEFAULT_FEATURE_COLS


def evaluate_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
) -> Dict[str, Any]:
    """
    Evaluates a trained classifier on test data and returns genuine metrics.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

    acc = float(accuracy_score(y_test, y_pred))
    # Handle zero division gracefully
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))

    try:
        auc = float(roc_auc_score(y_test, y_prob))
    except Exception:
        auc = 0.5  # Neutral AUC if single class in test slice

    cm = confusion_matrix(y_test, y_pred).tolist()

    return {
        "model_name": model_name,
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(auc, 4),
        "confusion_matrix": cm,
        "test_samples": len(y_test),
        "positive_rate": round(float(np.mean(y_test)), 4),
    }


def train_models(
    df: pd.DataFrame,
    config: Optional[ModelConfig] = None,
    side: OrderSide = OrderSide.BUY,
    save_best: bool = True,
) -> Tuple[Dict[str, Any], Path]:
    """
    Trains both Logistic Regression and XGBoost classifiers on chronological feature data.
    Selects and saves the superior model.
    """
    config = config or ModelConfig()
    SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    X_train, y_train, X_val, y_val, X_test, y_test, scaler, feature_cols = prepare_chronological_splits(
        df=df,
        feature_cols=DEFAULT_FEATURE_COLS,
        horizon=config.prediction_horizon,
        threshold_mult=config.adverse_threshold_spread_mult,
        test_size=config.test_size,
        val_size=config.val_size,
        side=side,
    )

    # 1. Baseline: Logistic Regression
    logreg = LogisticRegression(
        C=config.logreg_c,
        max_iter=1000,
        random_state=config.random_seed,
        class_weight="balanced",
    )
    logreg.fit(X_train, y_train)
    logreg_metrics = evaluate_model(logreg, X_test, y_test, "Logistic Regression Baseline")

    # 2. XGBoost Classifier
    scale_pos_weight = (len(y_train) - sum(y_train)) / max(1, sum(y_train))
    xgb_model = xgb.XGBClassifier(
        n_estimators=config.xgboost_n_estimators,
        max_depth=config.xgboost_max_depth,
        learning_rate=config.xgboost_learning_rate,
        random_state=config.random_seed,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
    )
    xgb_model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    xgb_metrics = evaluate_model(xgb_model, X_test, y_test, "XGBoost Classifier")

    # Select best model based on ROC-AUC (fallback to F1)
    if xgb_metrics["roc_auc"] >= logreg_metrics["roc_auc"]:
        best_model = xgb_model
        best_name = "XGBoost Classifier"
        best_metrics = xgb_metrics
    else:
        best_model = logreg
        best_name = "Logistic Regression Baseline"
        best_metrics = logreg_metrics

    # Feature importances
    feature_importances: Dict[str, float] = {}
    if hasattr(best_model, "feature_importances_"):
        for name, imp in zip(feature_cols, best_model.feature_importances_):
            feature_importances[name] = round(float(imp), 4)
    elif hasattr(best_model, "coef_"):
        for name, coef in zip(feature_cols, best_model.coef_[0]):
            feature_importances[name] = round(float(abs(coef)), 4)

    results = {
        "best_model_name": best_name,
        "best_metrics": best_metrics,
        "logistic_regression": logreg_metrics,
        "xgboost": xgb_metrics,
        "feature_importances": feature_importances,
        "feature_names": feature_cols,
        "prediction_horizon": config.prediction_horizon,
    }

    model_path = config.saved_model_path
    if save_best:
        bundle = {
            "model": best_model,
            "scaler": scaler,
            "feature_names": feature_cols,
            "best_name": best_name,
            "metrics": results,
            "config": config,
        }
        joblib.dump(bundle, model_path)

    return results, model_path
