from __future__ import annotations

from pathlib import Path
import argparse
import importlib.util
import json
import sys
import time
import warnings
from typing import Any

import joblib          # [FIX 3] dùng joblib thay pickle
import numpy as np
import pandas as pd
from env_config import load_env, resolve_directory as env_resolve_directory

warnings.filterwarnings("ignore")


DEFAULT_FEATURES = ["CR", "ROS", "DS", "TAT", "SIZE"]
DEFAULT_TARGET_COL = "target"
DEFAULT_PREPROCESSING_SUBDIR = Path("output") / "preprocessing"
DEFAULT_OUTPUT_SUBDIR = Path("output") / "models"
DEFAULT_PREPROCESSED_ENV_VAR = "PREPROCESSING_OUTPUT_DIR"
DEFAULT_MODELS_ENV_VAR = "MODELS_OUTPUT_DIR"

# Bài toán ưu tiên bắt đúng doanh nghiệp ROS < 5% (class dương),
# nên ưu tiên recall hơn precision → dùng F2-score.
GRID_SEARCH_BETA = 2.0
THRESHOLD_GRID = np.round(np.arange(0.10, 0.91, 0.02), 2)

MODEL_DISPLAY_NAMES = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
}
MODEL_DATASETS = {
    "logistic_regression": "scaled",   # LR nhạy với thang đo → cần scale
    "random_forest": "raw",            # RF dùng tree split → không cần scale
    "xgboost": "raw",                  # XGB dùng tree split → không cần scale
}
MODEL_FILE_NAMES = {
    "logistic_regression": "logistic_regression_model.pkl",
    "random_forest": "random_forest_model.pkl",
    "xgboost": "xgboost_model.pkl",
}
MODEL_ALIASES = {
    "lr": "logistic_regression",
    "logistic": "logistic_regression",
    "logistic_regression": "logistic_regression",
    "rf": "random_forest",
    "random_forest": "random_forest",
    "randomforest": "random_forest",
    "xgb": "xgboost",
    "xgboost": "xgboost",
    "all": "all",
}


# ============================================================
# Utilities
# ============================================================

def ensure_utf8_output() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")


def resolve_preprocessed_dir(user_input: str | None) -> Path:
    path = env_resolve_directory(
        user_input,
        DEFAULT_PREPROCESSED_ENV_VAR,
        Path("src") / DEFAULT_PREPROCESSING_SUBDIR,
    )
    if path.exists() and path.is_dir():
        return path.resolve()
    raise FileNotFoundError(f"Khong tim thay thu muc preprocessing: {path}")


def normalize_requested_models(models: list[str] | tuple[str, ...] | None) -> list[str]:
    if not models:
        return list(MODEL_DISPLAY_NAMES.keys())
    normalized: list[str] = []
    for model in models:
        key = model.strip().lower()
        if key not in MODEL_ALIASES:
            valid = ", ".join(sorted(MODEL_ALIASES))
            raise ValueError(f"Model '{model}' khong hop le. Gia tri hop le: {valid}")
        canonical = MODEL_ALIASES[key]
        if canonical == "all":
            return list(MODEL_DISPLAY_NAMES.keys())
        if canonical not in normalized:
            normalized.append(canonical)
    return normalized


def dependency_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def collect_dependency_status() -> dict[str, bool]:
    return {
        "scikit_learn": dependency_available("sklearn"),
        "xgboost": dependency_available("xgboost"),
    }


def require_sklearn() -> dict[str, Any]:
    try:
        from sklearn.base import clone
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import (
            accuracy_score,
            confusion_matrix,
            f1_score,
            fbeta_score,
            make_scorer,
            precision_score,
            recall_score,
            roc_auc_score,
        )
        from sklearn.model_selection import (
            GridSearchCV,
            StratifiedKFold,
            cross_val_predict,
            cross_val_score,   # [FIX 2] thêm để tính baseline
        )
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Thieu scikit-learn. Chay: pip install scikit-learn") from exc

    return {
        "clone": clone,
        "RandomForestClassifier": RandomForestClassifier,
        "LogisticRegression": LogisticRegression,
        "GridSearchCV": GridSearchCV,
        "StratifiedKFold": StratifiedKFold,
        "cross_val_predict": cross_val_predict,
        "cross_val_score": cross_val_score,    # [FIX 2]
        "accuracy_score": accuracy_score,
        "confusion_matrix": confusion_matrix,
        "f1_score": f1_score,
        "fbeta_score": fbeta_score,
        "make_scorer": make_scorer,
        "precision_score": precision_score,
        "recall_score": recall_score,
        "roc_auc_score": roc_auc_score,
    }


def require_xgboost():
    try:
        from xgboost import XGBClassifier
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Thieu xgboost. Chay: pip install xgboost") from exc
    return XGBClassifier


# ============================================================
# Data loading
# ============================================================

def read_csv_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Khong tim thay file bat buoc: {path}")
    return pd.read_csv(path)


def read_csv_optional(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def dataframe_to_target(series_or_frame: pd.Series | pd.DataFrame, target_col: str) -> pd.Series:
    if isinstance(series_or_frame, pd.DataFrame):
        series = series_or_frame[target_col] if target_col in series_or_frame.columns else series_or_frame.iloc[:, 0]
    else:
        series = series_or_frame
    clean = pd.to_numeric(series, errors="coerce")
    if clean.isna().any():
        raise ValueError("Cot target co gia tri khong hop le.")
    return clean.astype(int).rename(target_col)


def validate_feature_frame(df: pd.DataFrame, features: list[str], file_name: str) -> pd.DataFrame:
    missing = [col for col in features if col not in df.columns]
    if missing:
        raise ValueError(f"File {file_name} thieu feature: {missing}")
    return df[features].copy()


def load_preprocessed_bundle(preprocessed_dir: Path) -> dict[str, Any]:
    config_path = preprocessed_dir / "preprocessing_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}

    features   = config.get("features", DEFAULT_FEATURES)
    target_col = config.get("target_col", DEFAULT_TARGET_COL)

    X_train_raw    = validate_feature_frame(read_csv_required(preprocessed_dir / "X_train.csv"),        features, "X_train.csv")
    X_test_raw     = validate_feature_frame(read_csv_required(preprocessed_dir / "X_test.csv"),         features, "X_test.csv")
    X_train_scaled = validate_feature_frame(read_csv_required(preprocessed_dir / "X_train_scaled.csv"), features, "X_train_scaled.csv")
    X_test_scaled  = validate_feature_frame(read_csv_required(preprocessed_dir / "X_test_scaled.csv"),  features, "X_test_scaled.csv")
    y_train        = dataframe_to_target(read_csv_required(preprocessed_dir / "y_train.csv"), target_col)
    y_test         = dataframe_to_target(read_csv_required(preprocessed_dir / "y_test.csv"),  target_col)
    meta_train     = read_csv_optional(preprocessed_dir / "meta_train.csv")
    meta_test      = read_csv_optional(preprocessed_dir / "meta_test.csv")

    if len(X_train_scaled) != len(X_train_raw) or len(y_train) != len(X_train_raw):
        raise ValueError("So dong train khong khop giua cac file.")
    if len(X_test_scaled) != len(X_test_raw) or len(y_test) != len(X_test_raw):
        raise ValueError("So dong test khong khop giua cac file.")

    return {
        "preprocessed_dir": preprocessed_dir,
        "config": config,
        "features": features,
        "target_col": target_col,
        "X_train_raw": X_train_raw,
        "X_test_raw": X_test_raw,
        "X_train_scaled": X_train_scaled,
        "X_test_scaled": X_test_scaled,
        "y_train": y_train,
        "y_test": y_test,
        "meta_train": meta_train,
        "meta_test": meta_test,
    }


def build_bundle_from_frames(
    X_train, X_test, y_train, y_test,
    X_train_scaled=None, X_test_scaled=None, meta_test=None,
) -> dict[str, Any]:
    features   = list(X_train.columns)
    target_col = y_train.name if isinstance(y_train, pd.Series) and y_train.name else DEFAULT_TARGET_COL
    y_train_c  = dataframe_to_target(y_train, target_col)
    y_test_c   = dataframe_to_target(y_test,  target_col)

    if X_train_scaled is None or X_test_scaled is None:
        warnings.warn("Khong co X_train_scaled. Logistic Regression se dung du lieu raw.", stacklevel=2)
        X_train_scaled = X_train.copy()
        X_test_scaled  = X_test.copy()

    return {
        "preprocessed_dir": None,
        "config": {},
        "features": features,
        "target_col": target_col,
        "X_train_raw": X_train.copy(),
        "X_test_raw": X_test.copy(),
        "X_train_scaled": validate_feature_frame(X_train_scaled, features, "X_train_scaled"),
        "X_test_scaled":  validate_feature_frame(X_test_scaled,  features, "X_test_scaled"),
        "y_train": y_train_c,
        "y_test":  y_test_c,
        "meta_train": None,
        "meta_test": meta_test.copy() if meta_test is not None else None,
    }


# ============================================================
# CV / Scoring helpers
# ============================================================

def get_cv():
    return require_sklearn()["StratifiedKFold"](n_splits=5, shuffle=True, random_state=42)


def get_grid_scorer():
    sk = require_sklearn()
    return sk["make_scorer"](sk["fbeta_score"], beta=GRID_SEARCH_BETA, zero_division=0)


def get_dataset_pair(bundle: dict[str, Any], model_key: str) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    kind = MODEL_DATASETS[model_key]
    if kind == "scaled":
        return bundle["X_train_scaled"], bundle["X_test_scaled"], kind
    return bundle["X_train_raw"], bundle["X_test_raw"], kind


def safe_predict_scores(model: Any, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X)[:, 1], dtype=float)
    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(X), dtype=float)
        lo, hi = float(scores.min()), float(scores.max())
        return (scores - lo) / (hi - lo) if hi > lo else np.zeros_like(scores)
    return np.asarray(model.predict(X), dtype=float)


def get_scale_pos_weight(y_train: pd.Series) -> float:
    counts = y_train.value_counts().to_dict()
    pos = int(counts.get(1, 0))
    neg = int(counts.get(0, 0))
    return max(neg / pos, 1.0) if pos > 0 else 1.0


# ============================================================
# [FIX 2] Baseline evaluation — chạy trước grid.fit()
# ============================================================

def print_baseline(model_name: str, default_model: Any, X_train: pd.DataFrame,
                   y_train: pd.Series) -> dict[str, float]:
    """
    Tính cross_val_score với default model (không tune) để có baseline.
    Thầy hướng dẫn thường hỏi: 'GridSearch cải thiện được bao nhiêu so với default?'
    Hàm này cho số liệu để trả lời câu đó.
    """
    sk = require_sklearn()
    scorer = get_grid_scorer()
    cv = get_cv()

    # F2 baseline
    f2_scores = sk["cross_val_score"](default_model, X_train, y_train,
                                      cv=cv, scoring=scorer, n_jobs=-1)
    # AUC baseline
    auc_scores = sk["cross_val_score"](default_model, X_train, y_train,
                                       cv=cv, scoring="roc_auc", n_jobs=-1)

    baseline = {
        "f2_mean":  float(f2_scores.mean()),
        "f2_std":   float(f2_scores.std()),
        "auc_mean": float(auc_scores.mean()),
        "auc_std":  float(auc_scores.std()),
    }

    print(f"  Baseline {model_name} (default params, cv=5):")
    print(f"    F2-score : {baseline['f2_mean']:.4f} ± {baseline['f2_std']:.4f}")
    print(f"    AUC-ROC  : {baseline['auc_mean']:.4f} ± {baseline['auc_std']:.4f}")
    return baseline


# ============================================================
# Threshold tuning (OOF — không chạm test set)
# ============================================================

def find_best_threshold(y_true, scores, *, beta=GRID_SEARCH_BETA,
                        thresholds=THRESHOLD_GRID) -> tuple[float, dict[str, float]]:
    sk = require_sklearn()
    best_t = 0.50
    best = {"f2": -1.0, "recall": 0.0, "precision": 0.0}
    y_arr = np.asarray(y_true, dtype=int)

    for t in thresholds:
        preds = (scores >= t).astype(int)
        f2  = float(sk["fbeta_score"](y_arr, preds, beta=beta, zero_division=0))
        rec = float(sk["recall_score"](y_arr, preds, zero_division=0))
        pre = float(sk["precision_score"](y_arr, preds, zero_division=0))
        if (f2 > best["f2"]
                or (np.isclose(f2, best["f2"]) and rec > best["recall"])
                or (np.isclose(f2, best["f2"]) and np.isclose(rec, best["recall"]) and pre > best["precision"])):
            best_t = float(t)
            best = {"f2": f2, "recall": rec, "precision": pre}

    return best_t, best


def tune_threshold_from_oof(model: Any, X_train: pd.DataFrame,
                             y_train: pd.Series) -> tuple[float, dict[str, float]]:
    sk = require_sklearn()
    oof_scores = sk["cross_val_predict"](
        sk["clone"](model), X_train, y_train,
        cv=get_cv(), method="predict_proba", n_jobs=-1,
    )[:, 1]
    return find_best_threshold(y_train, oof_scores)


# ============================================================
# Evaluation
# ============================================================

def evaluate_model(model, X_test, y_test, threshold=0.50):
    sk = require_sklearn()
    y_score = pd.Series(safe_predict_scores(model, X_test), index=X_test.index, name="score")
    y_pred  = pd.Series((y_score >= threshold).astype(int), index=X_test.index, name="pred")

    try:
        auc = sk["roc_auc_score"](y_test, y_score)
    except ValueError:
        auc = float("nan")

    tn, fp, fn, tp = sk["confusion_matrix"](y_test, y_pred, labels=[0, 1]).ravel()

    metrics = {
        "accuracy":    float(sk["accuracy_score"](y_test, y_pred)),
        "auc":         float(auc),
        "f1":          float(sk["f1_score"](y_test, y_pred, zero_division=0)),
        "f2":          float(sk["fbeta_score"](y_test, y_pred, beta=GRID_SEARCH_BETA, zero_division=0)),
        "precision":   float(sk["precision_score"](y_test, y_pred, zero_division=0)),
        "recall":      float(sk["recall_score"](y_test, y_pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan"),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }
    return metrics, y_pred, y_score


def print_metrics(model_name: str, metrics: dict[str, float], threshold: float,
                  baseline: dict[str, float] | None = None) -> None:
    print(f"\n--- {model_name} — Test Performance ---")
    print(f"Threshold : {threshold:.2f}")
    print(f"Accuracy  : {metrics['accuracy']:.4f}")
    print(f"AUC-ROC   : {metrics['auc']:.4f}", end="")
    # [FIX 2] In cải thiện so với baseline nếu có
    if baseline:
        delta_auc = metrics['auc'] - baseline['auc_mean']
        delta_f2  = metrics['f2']  - baseline['f2_mean']
        print(f"  (baseline {baseline['auc_mean']:.4f}, +{delta_auc:+.4f})", end="")
    print()
    print(f"F1-score  : {metrics['f1']:.4f}")
    print(f"F2-score  : {metrics['f2']:.4f}", end="")
    if baseline:
        print(f"  (baseline {baseline['f2_mean']:.4f}, {delta_f2:+.4f})", end="")
    print()
    print(f"Precision : {metrics['precision']:.4f}")
    print(f"Recall    : {metrics['recall']:.4f}")
    print(f"Specific. : {metrics['specificity']:.4f}")
    print(f"ConfMat   : TN={metrics['tn']} FP={metrics['fp']} FN={metrics['fn']} TP={metrics['tp']}")


# ============================================================
# [FIX 3] Lưu artifact bằng joblib (thống nhất với preprocessing.py)
# ============================================================

def save_artifact(obj: Any, output_path: Path) -> None:
    """Dùng joblib thay pickle — nhanh hơn và được sklearn khuyến nghị."""
    joblib.dump(obj, output_path)


def save_prediction_table(model_key, y_test, y_pred, y_score, threshold,
                          output_dir, meta_test=None) -> Path:
    result_df = pd.DataFrame({
        "actual_target":       y_test.reset_index(drop=True),
        "predicted_target":    y_pred.reset_index(drop=True),
        "score_positive_class": y_score.reset_index(drop=True),
        "threshold_used":      threshold,
    })
    if meta_test is not None:
        result_df = pd.concat([meta_test.reset_index(drop=True), result_df], axis=1)
    path = output_dir / f"predictions_{model_key}.csv"
    result_df.to_csv(path, index=False)
    return path


# ============================================================
# Model trainers
# ============================================================

def train_logistic_regression(bundle: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    sk = require_sklearn()
    LogisticRegression = sk["LogisticRegression"]
    GridSearchCV       = sk["GridSearchCV"]

    X_train, X_test, kind = get_dataset_pair(bundle, "logistic_regression")
    y_train, y_test = bundle["y_train"], bundle["y_test"]

    print("\n==============================")
    print("MODEL 1 - Logistic Regression")
    print("==============================")
    print(f"Input dataset: {kind} (StandardScaler đã áp dụng)")

    # [FIX 2] Baseline trước khi tune
    default_lr = LogisticRegression(random_state=42, max_iter=1000, class_weight="balanced",
                                    solver="liblinear")
    baseline = print_baseline("Logistic Regression", default_lr, X_train, y_train)

    # GridSearch
    start = time.time()
    param_grid = {
        "C":       [0.001, 0.01, 0.1, 1, 10, 100],
        "penalty": ["l1", "l2"],
        "solver":  ["liblinear"],
    }
    grid = GridSearchCV(
        LogisticRegression(random_state=42, max_iter=1000, class_weight="balanced"),
        param_grid, cv=get_cv(), scoring=get_grid_scorer(), n_jobs=-1, verbose=0,
    )
    grid.fit(X_train, y_train)
    elapsed = time.time() - start

    best_model = grid.best_estimator_
    best_threshold, threshold_stats = tune_threshold_from_oof(best_model, X_train, y_train)
    metrics, y_pred, y_score = evaluate_model(best_model, X_test, y_test, threshold=best_threshold)

    print(f"\nBest params   : {grid.best_params_}")
    print(f"Best CV F2    : {grid.best_score_:.4f}  (baseline {baseline['f2_mean']:.4f}, "
          f"{grid.best_score_ - baseline['f2_mean']:+.4f})")
    print(f"Threshold OOF : {best_threshold:.2f}  {threshold_stats}")
    print(f"Train time    : {elapsed:.1f}s")
    print_metrics(MODEL_DISPLAY_NAMES["logistic_regression"], metrics, best_threshold, baseline)

    # [FIX 3] Lưu bằng joblib
    model_path = output_dir / MODEL_FILE_NAMES["logistic_regression"]
    save_artifact({"model": best_model, "threshold": best_threshold,
                   "features": bundle["features"], "input_data": kind}, model_path)
    pred_path = save_prediction_table("logistic_regression", y_test, y_pred, y_score,
                                      best_threshold, output_dir, bundle["meta_test"])
    print(f"Saved model      : {model_path}")
    print(f"Saved predictions: {pred_path}")

    return {
        "model_key": "logistic_regression",
        "model": MODEL_DISPLAY_NAMES["logistic_regression"],
        "input_data": kind,
        "baseline_f2": baseline["f2_mean"],
        "baseline_auc": baseline["auc_mean"],
        "best_params": json.dumps(grid.best_params_, ensure_ascii=False),
        "cv_best_f2": float(grid.best_score_),
        "cv_improvement_f2": float(grid.best_score_ - baseline["f2_mean"]),
        "threshold": float(best_threshold),
        "train_time_seconds": float(elapsed),
        "model_path": str(model_path),
        "prediction_path": str(pred_path),
        **metrics,
    }


def train_random_forest(bundle: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    sk = require_sklearn()
    RandomForestClassifier = sk["RandomForestClassifier"]
    GridSearchCV           = sk["GridSearchCV"]

    X_train, X_test, kind = get_dataset_pair(bundle, "random_forest")
    y_train, y_test = bundle["y_train"], bundle["y_test"]

    print("\n========================")
    print("MODEL 2 - Random Forest")
    print("========================")
    print(f"Input dataset: {kind} (scale không cần thiết cho tree-based model)")

    # [FIX 2] Baseline
    default_rf = RandomForestClassifier(random_state=42, n_jobs=1, class_weight="balanced")
    baseline = print_baseline("Random Forest", default_rf, X_train, y_train)

    # [FIX 1] Thêm max_features vào grid — hyperparameter quan trọng nhất của RF
    # max_features kiểm soát số feature được xem xét tại mỗi split:
    #   "sqrt" ≈ sqrt(5) ≈ 2 features → tăng diversity giữa các cây
    #   "log2" ≈ log2(5) ≈ 2 features → tương tự nhưng log scale
    # Thiếu max_features là lý do RF thường bị underfit hoặc overfit mà không rõ nguyên nhân.
    start = time.time()
    param_grid = {
        "n_estimators":     [100, 200],
        "max_depth":        [3, 5, 10, None],
        "max_features":     ["sqrt", "log2"],     # [FIX 1] thêm mới
        "min_samples_split": [2, 5],
        "min_samples_leaf":  [1, 2],
    }
    grid = GridSearchCV(
        RandomForestClassifier(random_state=42, n_jobs=1, class_weight="balanced"),
        param_grid, cv=get_cv(), scoring=get_grid_scorer(), n_jobs=-1, verbose=0,
    )
    grid.fit(X_train, y_train)
    elapsed = time.time() - start

    best_model = grid.best_estimator_
    best_threshold, threshold_stats = tune_threshold_from_oof(best_model, X_train, y_train)
    metrics, y_pred, y_score = evaluate_model(best_model, X_test, y_test, threshold=best_threshold)

    print(f"\nBest params   : {grid.best_params_}")
    print(f"Best CV F2    : {grid.best_score_:.4f}  (baseline {baseline['f2_mean']:.4f}, "
          f"{grid.best_score_ - baseline['f2_mean']:+.4f})")
    print(f"Threshold OOF : {best_threshold:.2f}  {threshold_stats}")
    print(f"Train time    : {elapsed:.1f}s")
    print_metrics(MODEL_DISPLAY_NAMES["random_forest"], metrics, best_threshold, baseline)

    # [FIX 3] joblib
    model_path = output_dir / MODEL_FILE_NAMES["random_forest"]
    save_artifact({"model": best_model, "threshold": best_threshold,
                   "features": bundle["features"], "input_data": kind}, model_path)
    pred_path = save_prediction_table("random_forest", y_test, y_pred, y_score,
                                      best_threshold, output_dir, bundle["meta_test"])
    print(f"Saved model      : {model_path}")
    print(f"Saved predictions: {pred_path}")

    return {
        "model_key": "random_forest",
        "model": MODEL_DISPLAY_NAMES["random_forest"],
        "input_data": kind,
        "baseline_f2": baseline["f2_mean"],
        "baseline_auc": baseline["auc_mean"],
        "best_params": json.dumps(grid.best_params_, ensure_ascii=False),
        "cv_best_f2": float(grid.best_score_),
        "cv_improvement_f2": float(grid.best_score_ - baseline["f2_mean"]),
        "threshold": float(best_threshold),
        "train_time_seconds": float(elapsed),
        "model_path": str(model_path),
        "prediction_path": str(pred_path),
        **metrics,
    }


def train_xgboost(bundle: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    sk = require_sklearn()
    GridSearchCV  = sk["GridSearchCV"]
    XGBClassifier = require_xgboost()

    X_train, X_test, kind = get_dataset_pair(bundle, "xgboost")
    y_train, y_test = bundle["y_train"], bundle["y_test"]
    scale_pos_weight = get_scale_pos_weight(y_train)

    print("\n===================")
    print("MODEL 3 - XGBoost")
    print("===================")
    print(f"Input dataset    : {kind}")
    print(f"scale_pos_weight : {scale_pos_weight:.4f}  (neg/pos ratio để cân bằng class)")

    # [FIX 2] Baseline
    default_xgb = XGBClassifier(random_state=42, eval_metric="logloss",
                                 n_jobs=1, scale_pos_weight=scale_pos_weight)
    baseline = print_baseline("XGBoost", default_xgb, X_train, y_train)

    start = time.time()
    param_grid = {
        "n_estimators":    [100, 200],
        "max_depth":       [3, 5],
        "learning_rate":   [0.01, 0.1],
        "subsample":       [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
    }
    grid = GridSearchCV(
        XGBClassifier(random_state=42, eval_metric="logloss",
                      n_jobs=1, scale_pos_weight=scale_pos_weight),
        param_grid, cv=get_cv(), scoring=get_grid_scorer(), n_jobs=-1, verbose=0,
    )
    grid.fit(X_train, y_train)
    elapsed = time.time() - start

    best_model = grid.best_estimator_
    best_threshold, threshold_stats = tune_threshold_from_oof(best_model, X_train, y_train)
    metrics, y_pred, y_score = evaluate_model(best_model, X_test, y_test, threshold=best_threshold)

    print(f"\nBest params   : {grid.best_params_}")
    print(f"Best CV F2    : {grid.best_score_:.4f}  (baseline {baseline['f2_mean']:.4f}, "
          f"{grid.best_score_ - baseline['f2_mean']:+.4f})")
    print(f"Threshold OOF : {best_threshold:.2f}  {threshold_stats}")
    print(f"Train time    : {elapsed:.1f}s")
    print_metrics(MODEL_DISPLAY_NAMES["xgboost"], metrics, best_threshold, baseline)

    # [FIX 3] joblib
    model_path = output_dir / MODEL_FILE_NAMES["xgboost"]
    save_artifact({"model": best_model, "threshold": best_threshold,
                   "features": bundle["features"], "input_data": kind,
                   "scale_pos_weight": scale_pos_weight}, model_path)
    pred_path = save_prediction_table("xgboost", y_test, y_pred, y_score,
                                      best_threshold, output_dir, bundle["meta_test"])
    print(f"Saved model      : {model_path}")
    print(f"Saved predictions: {pred_path}")

    return {
        "model_key": "xgboost",
        "model": MODEL_DISPLAY_NAMES["xgboost"],
        "input_data": kind,
        "baseline_f2": baseline["f2_mean"],
        "baseline_auc": baseline["auc_mean"],
        "best_params": json.dumps(grid.best_params_, ensure_ascii=False),
        "cv_best_f2": float(grid.best_score_),
        "cv_improvement_f2": float(grid.best_score_ - baseline["f2_mean"]),
        "threshold": float(best_threshold),
        "train_time_seconds": float(elapsed),
        "model_path": str(model_path),
        "prediction_path": str(pred_path),
        **metrics,
    }


MODEL_TRAINERS = {
    "logistic_regression": train_logistic_regression,
    "random_forest":        train_random_forest,
    "xgboost":              train_xgboost,
}


# ============================================================
# Summary helpers
# ============================================================

def print_bundle_summary(bundle: dict[str, Any]) -> None:
    print("=" * 65)
    print("TRAINING INPUT SUMMARY")
    print("=" * 65)
    if bundle["preprocessed_dir"]:
        print(f"Preprocessed dir : {bundle['preprocessed_dir']}")
    print(f"Features         : {bundle['features']}")
    print(f"Target column    : {bundle['target_col']}")
    print(f"Train raw        : {bundle['X_train_raw'].shape}")
    print(f"Train scaled     : {bundle['X_train_scaled'].shape}")
    print(f"Test raw         : {bundle['X_test_raw'].shape}")
    print(f"Test scaled      : {bundle['X_test_scaled'].shape}")
    print(f"Train classes    : {bundle['y_train'].value_counts().sort_index().to_dict()}")
    print(f"Test classes     : {bundle['y_test'].value_counts().sort_index().to_dict()}")


def save_training_summary(bundle, requested_models, results_df, output_dir) -> None:
    summary = {
        "requested_models":     requested_models,
        "features":             bundle["features"],
        "target_col":           bundle["target_col"],
        "preprocessed_dir":     str(bundle["preprocessed_dir"]) if bundle["preprocessed_dir"] else None,
        "train_rows":           int(len(bundle["X_train_raw"])),
        "test_rows":            int(len(bundle["X_test_raw"])),
        "train_class_balance":  {str(k): int(v) for k, v in bundle["y_train"].value_counts().sort_index().items()},
        "test_class_balance":   {str(k): int(v) for k, v in bundle["y_test"].value_counts().sort_index().items()},
        "grid_search_objective": f"F{GRID_SEARCH_BETA}-score (uu tien recall)",
        "threshold_grid":       THRESHOLD_GRID.tolist(),
        "fixes_applied": [
            "[FIX 1] RF grid: them max_features=['sqrt','log2']",
            "[FIX 2] Baseline cross_val_score truoc grid.fit()",
            "[FIX 3] Luu artifact bang joblib thay pickle",
        ],
        "results": results_df.to_dict(orient="records"),
    }
    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )


# ============================================================
# Public API
# ============================================================

def train_and_evaluate_models(
    X_train=None, X_test=None, y_train=None, y_test=None,
    output_dir=None, *, X_train_scaled=None, X_test_scaled=None,
    meta_test=None, preprocessed_dir=None, models=None,
) -> pd.DataFrame:
    requested_models = normalize_requested_models(models)

    if X_train is None or X_test is None or y_train is None or y_test is None:
        resolved = resolve_preprocessed_dir(str(preprocessed_dir) if preprocessed_dir else None)
        bundle = load_preprocessed_bundle(resolved)
        default_output = env_resolve_directory(
            None,
            DEFAULT_MODELS_ENV_VAR,
            Path("src") / DEFAULT_OUTPUT_SUBDIR,
        )
    else:
        bundle = build_bundle_from_frames(X_train, X_test, y_train, y_test,
                                          X_train_scaled, X_test_scaled, meta_test)
        default_output = env_resolve_directory(None, DEFAULT_MODELS_ENV_VAR, Path("models"))

    final_output = Path(output_dir).expanduser() if output_dir else default_output
    final_output.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    for model_key in requested_models:
        try:
            result = MODEL_TRAINERS[model_key](bundle, final_output)
            results.append(result)
        except Exception as exc:
            print(f"\n[ERROR] {MODEL_DISPLAY_NAMES[model_key]} failed: {exc}")

    if not results:
        raise RuntimeError("Tat ca model deu that bai. Kiem tra preprocessing output va dependency.")

    comparison_df = pd.DataFrame(results)[[
        "model", "input_data",
        "baseline_f2", "baseline_auc",       # [FIX 2] thêm baseline vào bảng so sánh
        "cv_best_f2", "cv_improvement_f2",   # [FIX 2] thêm cải thiện so baseline
        "threshold",
        "accuracy", "auc", "f1", "f2",
        "precision", "recall", "specificity",
        "train_time_seconds", "best_params",
    ]].sort_values(["f2", "recall", "auc"], ascending=False, na_position="last")

    print("\n" + "=" * 65)
    print("BANG SO SANH MODEL")
    print("=" * 65)
    print(comparison_df[["model", "baseline_f2", "cv_best_f2", "cv_improvement_f2",
                          "auc", "f1", "f2", "recall", "threshold"]].to_string(index=False))

    comparison_df.to_csv(final_output / "model_comparison.csv",  index=False)
    comparison_df.to_excel(final_output / "model_comparison.xlsx", index=False)
    save_training_summary(bundle, requested_models, comparison_df, final_output)

    print(f"\nOutput luu tai: {final_output}")
    return comparison_df


# ============================================================
# CLI
# ============================================================

def main() -> None:
    ensure_utf8_output()
    load_env()

    parser = argparse.ArgumentParser(description="Train models from preprocessing outputs.")
    parser.add_argument("--preprocessed-dir", type=str, default=None)
    parser.add_argument("--output-dir",        type=str, default=None)
    parser.add_argument("--models",            nargs="*", default=None,
                        help="lr rf xgb hoac all")
    parser.add_argument("--check-only",        action="store_true",
                        help="Chi kiem tra setup, khong train.")
    args = parser.parse_args()

    requested_models   = normalize_requested_models(args.models)
    dependency_status  = collect_dependency_status()
    preprocessed_dir   = resolve_preprocessed_dir(args.preprocessed_dir)
    bundle             = load_preprocessed_bundle(preprocessed_dir)

    print_bundle_summary(bundle)
    print()
    print("Dependency:")
    print(f"  scikit-learn : {'OK' if dependency_status['scikit_learn'] else 'MISSING'}")
    print(f"  xgboost      : {'OK' if dependency_status['xgboost'] else 'MISSING'}")
    print(f"Models         : {requested_models}")
    print(f"Objective      : F{GRID_SEARCH_BETA}-score + OOF threshold tuning")
    print(f"Fixes applied  : [FIX1] max_features RF  [FIX2] baseline  [FIX3] joblib")

    if args.check_only:
        return

    try:
        train_and_evaluate_models(
            output_dir=args.output_dir,
            preprocessed_dir=preprocessed_dir,
            models=requested_models,
        )
    except Exception as exc:
        print(f"\n[FATAL] {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
