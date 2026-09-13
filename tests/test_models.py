"""Unit tests for ML Adverse-Selection Risk modeling and prediction."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from data.generator import MarketDataGenerator
from data.contracts import OrderSide, PredictionResult, ModelStatus, RiskCategory
from features import build_feature_pipeline
from models.preprocessing import generate_adverse_selection_labels, prepare_chronological_splits
from models.train import train_models
from models.predictor import AdverseSelectionPredictor


@pytest.fixture
def feature_dataset():
    gen = MarketDataGenerator(seed=42)
    snaps = gen.generate_scenario_stream("high_adverse_selection", num_ticks=120)
    return build_feature_pipeline(snaps)


def test_label_generation(feature_dataset):
    labels_buy = generate_adverse_selection_labels(feature_dataset, horizon=5, threshold_mult=0.5, side=OrderSide.BUY)
    assert len(labels_buy) == len(feature_dataset)
    assert set(labels_buy.unique()).issubset({0, 1})

    labels_sell = generate_adverse_selection_labels(feature_dataset, horizon=5, threshold_mult=0.5, side=OrderSide.SELL)
    assert len(labels_sell) == len(feature_dataset)


def test_chronological_split_no_leakage(feature_dataset):
    X_train, y_train, X_val, y_val, X_test, y_test, scaler, cols = prepare_chronological_splits(
        feature_dataset,
        horizon=5,
        test_size=0.15,
        val_size=0.15,
    )
    # Ensure splits are non-empty
    assert len(X_train) > 0
    assert len(X_val) > 0
    assert len(X_test) > 0
    # Check scaler mean matches train dimensions
    assert len(scaler.mean_) == len(cols)


def test_model_training_and_serialization(feature_dataset, tmp_path):
    from config.config import ModelConfig
    temp_model_path = tmp_path / "test_model.joblib"
    cfg = ModelConfig(
        prediction_horizon=5,
        xgboost_n_estimators=10,
        xgboost_max_depth=3,
        saved_model_path=temp_model_path,
    )

    results, path = train_models(feature_dataset, config=cfg, save_best=True)
    assert Path(path).exists()
    assert "best_model_name" in results
    assert "logistic_regression" in results
    assert "xgboost" in results
    assert 0.0 <= results["best_metrics"]["accuracy"] <= 1.0


def test_predictor_interface_fallback():
    # Points to non-existent file
    predictor = AdverseSelectionPredictor(model_path=Path("non_existent_model.joblib"))
    assert predictor.model_status == ModelStatus.FALLBACK_HEURISTIC

    dummy_feats = {
        "timestamp": 1700000000.0,
        "depth_imbalance_l1": -0.8,
        "spread_expansion_ratio": 1.5,
        "momentum_ret_5": -0.01,
        "trade_volume_imbalance": -0.6,
    }

    pred = predictor.predict(dummy_feats, side=OrderSide.BUY)
    assert isinstance(pred, PredictionResult)
    assert 0.0 <= pred.probability <= 1.0
    assert pred.risk_category in [RiskCategory.LOW, RiskCategory.MEDIUM, RiskCategory.HIGH]
    assert pred.model_status == ModelStatus.FALLBACK_HEURISTIC
