from __future__ import annotations

from pathlib import Path
import argparse
import json
import pickle
import sys
import warnings
from typing import Any

import joblib
import numpy as np
import pandas as pd
from env_config import load_env, resolve_directory as env_resolve_directory

warnings.filterwarnings("ignore")


DEFAULT_FEATURES = ["CR", "ROS", "DS", "TAT", "SIZE", "is_HNX"]
DEFAULT_TARGET_COL = "target"
DEFAULT_PREPROCESSING_SUBDIR = Path("output") / "preprocessing"
DEFAULT_MODELS_SUBDIR = Path("output") / "models"
DEFAULT_OUTPUT_FILE = "model_evaluation.xlsx"
DEFAULT_AUDIT_FILE = "dataset_split_audit.json"
DEFAULT_PREPROCESSED_ENV_VAR = "PREPROCESSING_OUTPUT_DIR"
DEFAULT_MODELS_ENV_VAR = "MODELS_OUTPUT_DIR"

GRID_SEARCH_BETA = 2.0
ALTMAN_DISTRESS_CUTOFF = 1.81
ALTMAN_MODEL_KEY = "altman_z_score"

MODEL_DISPLAY_NAMES = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
}
MODEL_DATASETS = {
    "logistic_regression": "scaled",
    "random_forest": "raw",
    "xgboost": "raw",
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
COMPARISON_METRICS = [
    ("auc", "AUC"),
    ("f1", "F1"),
    ("f2", "F2"),
    ("precision", "Precision"),
    ("recall", "Recall"),
    ("specificity", "Specificity"),
    ("accuracy", "Accuracy"),
]
PLOT_COLORS = ["#4C78A8", "#F58518", "#54A24B", "#E45756"]
THRESHOLD_CURVE_GRID = np.round(np.arange(0.0, 1.001, 0.01), 2)
ALTMAN_REQUIRED_FEATURES = ["CR", "ROS", "DS", "TAT"]


def ensure_utf8_output() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")


def require_sklearn() -> dict[str, Any]:
    try:
        from sklearn.calibration import calibration_curve
        from sklearn.metrics import (
            accuracy_score,
            confusion_matrix,
            f1_score,
            fbeta_score,
            precision_recall_curve,
            precision_score,
            recall_score,
            roc_curve,
            roc_auc_score,
        )
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Thieu scikit-learn. Hay cai dependency roi chay lai, vi du: pip install scikit-learn"
        ) from exc

    return {
        "calibration_curve": calibration_curve,
        "accuracy_score": accuracy_score,
        "confusion_matrix": confusion_matrix,
        "f1_score": f1_score,
        "fbeta_score": fbeta_score,
        "precision_recall_curve": precision_recall_curve,
        "precision_score": precision_score,
        "recall_score": recall_score,
        "roc_curve": roc_curve,
        "roc_auc_score": roc_auc_score,
    }


def require_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Thieu matplotlib. Hay cai dependency roi chay lai, vi du: pip install matplotlib"
        ) from exc
    return plt


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
        if target_col in series_or_frame.columns:
            series = series_or_frame[target_col]
        else:
            series = series_or_frame.iloc[:, 0]
    else:
        series = series_or_frame

    clean = pd.to_numeric(series, errors="coerce")
    if clean.isna().any():
        raise ValueError("Cot target co gia tri khong hop le.")
    return clean.astype(int).rename(target_col)


def validate_feature_frame(df: pd.DataFrame, features: list[str], file_name: str) -> pd.DataFrame:
    missing = [col for col in features if col not in df.columns]
    if missing:
        raise ValueError(f"File {file_name} thieu feature bat buoc: {missing}")
    return df[features].copy()


def resolve_preprocessed_dir(user_input: str | None) -> Path:
    path = env_resolve_directory(
        user_input,
        DEFAULT_PREPROCESSED_ENV_VAR,
        Path("src") / DEFAULT_PREPROCESSING_SUBDIR,
    )
    if path.exists() and path.is_dir():
        return path.resolve()
    raise FileNotFoundError(f"Khong tim thay thu muc preprocessing theo --preprocessed-dir: {path}")


def resolve_models_dir(user_input: str | None) -> Path:
    path = env_resolve_directory(
        user_input,
        DEFAULT_MODELS_ENV_VAR,
        Path("src") / DEFAULT_MODELS_SUBDIR,
    )
    if path.exists() and path.is_dir():
        return path.resolve()
    raise FileNotFoundError(f"Khong tim thay thu muc models theo --models-dir: {path}")


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


def load_preprocessed_bundle(preprocessed_dir: Path) -> dict[str, Any]:
    config_path = preprocessed_dir / "preprocessing_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}

    # Neu config khong co, suy ra feature tu file X_train
    X_train_raw_full = read_csv_required(preprocessed_dir / "X_train.csv")
    inferred_features = list(X_train_raw_full.columns)
    features = config.get("features", inferred_features if inferred_features else DEFAULT_FEATURES)
    target_col = config.get("target_col", DEFAULT_TARGET_COL)

    X_train_raw = validate_feature_frame(X_train_raw_full, features, "X_train.csv")
    X_test_raw = validate_feature_frame(read_csv_required(preprocessed_dir / "X_test.csv"), features, "X_test.csv")
    X_train_scaled = validate_feature_frame(read_csv_required(preprocessed_dir / "X_train_scaled.csv"), features, "X_train_scaled.csv")
    X_test_scaled = validate_feature_frame(read_csv_required(preprocessed_dir / "X_test_scaled.csv"), features, "X_test_scaled.csv")
    y_train = dataframe_to_target(read_csv_required(preprocessed_dir / "y_train.csv"), target_col)
    y_test = dataframe_to_target(read_csv_required(preprocessed_dir / "y_test.csv"), target_col)
    meta_train = read_csv_optional(preprocessed_dir / "meta_train.csv")
    meta_test = read_csv_optional(preprocessed_dir / "meta_test.csv")

    if len(X_train_raw) != len(X_train_scaled) or len(X_train_raw) != len(y_train):
        raise ValueError("So dong train giua X_train/X_train_scaled/y_train khong khop.")
    if len(X_test_raw) != len(X_test_scaled) or len(X_test_raw) != len(y_test):
        raise ValueError("So dong test giua X_test/X_test_scaled/y_test khong khop.")
    if meta_train is not None and len(meta_train) != len(X_train_raw):
        raise ValueError("So dong meta_train khong khop voi X_train.")
    if meta_test is not None and len(meta_test) != len(X_test_raw):
        raise ValueError("So dong meta_test khong khop voi X_test.")

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


def get_dataset_pair(bundle: dict[str, Any], dataset_kind: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if dataset_kind == "scaled":
        return bundle["X_train_scaled"], bundle["X_test_scaled"]
    return bundle["X_train_raw"], bundle["X_test_raw"]


def safe_predict_scores(model: Any, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X)[:, 1], dtype=float)
    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(X), dtype=float)
        score_min, score_max = float(scores.min()), float(scores.max())
        if score_max > score_min:
            return (scores - score_min) / (score_max - score_min)
        return np.zeros_like(scores, dtype=float)
    return np.asarray(model.predict(X), dtype=float)


def evaluate_predictions(
    y_true: pd.Series,
    scores: np.ndarray,
    threshold: float,
) -> tuple[dict[str, float], np.ndarray]:
    sklearn_api = require_sklearn()
    y_pred = (scores >= threshold).astype(int)

    try:
        auc = sklearn_api["roc_auc_score"](y_true, scores)
    except ValueError:
        auc = float("nan")

    tn, fp, fn, tp = sklearn_api["confusion_matrix"](y_true, y_pred, labels=[0, 1]).ravel()

    metrics = {
        "accuracy": float(sklearn_api["accuracy_score"](y_true, y_pred)),
        "auc": float(auc),
        "f1": float(sklearn_api["f1_score"](y_true, y_pred, zero_division=0)),
        "f2": float(sklearn_api["fbeta_score"](y_true, y_pred, beta=GRID_SEARCH_BETA, zero_division=0)),
        "precision": float(sklearn_api["precision_score"](y_true, y_pred, zero_division=0)),
        "recall": float(sklearn_api["recall_score"](y_true, y_pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan"),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    return metrics, y_pred


def restore_altman_input_features(X_raw: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    missing = [col for col in ALTMAN_REQUIRED_FEATURES if col not in X_raw.columns]
    if missing:
        raise ValueError(f"Khong the tinh Altman Z-score vi thieu feature: {missing}")

    restored = X_raw[ALTMAN_REQUIRED_FEATURES].copy()
    for col in ALTMAN_REQUIRED_FEATURES:
        restored[col] = pd.to_numeric(restored[col], errors="coerce")

    log_plan = config.get("log_plan", {})
    for col in ["CR", "TAT"]:
        if col in restored.columns and bool(log_plan.get(col, {}).get("apply", False)):
            restored[col] = np.expm1(restored[col])

    if restored.isna().any().any():
        bad_cols = restored.columns[restored.isna().any()].tolist()
        raise ValueError(f"Altman input co gia tri khong hop le o cac cot: {bad_cols}")

    return restored


def compute_altman_z_score_proxy(X_raw: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.Series, pd.DataFrame]:
    features = restore_altman_input_features(X_raw, config)

    # Available engineered features:
    # CR = current_assets / current_liabilities
    # ROS = pre_tax_profit / revenue
    # DS = total_liabilities / revenue
    # TAT = revenue / total_assets
    #
    # Original Altman Z requires WC/TA, RE/TA, EBIT/TA, MVE/TL, Sales/TA.
    # RE/TA and market equity are unavailable, so this is an interpretable proxy.
    total_liabilities_to_assets = (features["DS"] * features["TAT"]).replace([np.inf, -np.inf], np.nan)
    total_liabilities_to_assets = total_liabilities_to_assets.clip(lower=1e-6)
    working_capital_to_assets = (features["CR"] - 1.0) * total_liabilities_to_assets
    ebit_to_assets = features["ROS"] * features["TAT"]
    book_equity_to_liabilities = ((1.0 - total_liabilities_to_assets) / total_liabilities_to_assets).clip(-10, 10)
    sales_to_assets = features["TAT"]

    z_score = (
        1.2 * working_capital_to_assets
        + 3.3 * ebit_to_assets
        + 0.6 * book_equity_to_liabilities
        + 1.0 * sales_to_assets
    )
    z_score = z_score.replace([np.inf, -np.inf], np.nan)

    if z_score.isna().any():
        raise ValueError("Altman Z-score tinh ra NaN/inf. Kiem tra CR, ROS, DS, TAT.")

    components = pd.DataFrame(
        {
            "CR": features["CR"].reset_index(drop=True),
            "ROS": features["ROS"].reset_index(drop=True),
            "DS": features["DS"].reset_index(drop=True),
            "TAT": features["TAT"].reset_index(drop=True),
            "working_capital_to_assets_proxy": working_capital_to_assets.reset_index(drop=True),
            "ebit_to_assets_proxy": ebit_to_assets.reset_index(drop=True),
            "book_equity_to_liabilities_proxy": book_equity_to_liabilities.reset_index(drop=True),
            "sales_to_assets": sales_to_assets.reset_index(drop=True),
            "altman_z_score_proxy": z_score.reset_index(drop=True),
        }
    )
    return z_score.reset_index(drop=True), components


def evaluate_altman_z_score_baseline(bundle: dict[str, Any], output_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    z_score, components = compute_altman_z_score_proxy(bundle["X_test_raw"], bundle.get("config", {}))

    # Lower Altman Z means higher distress risk. Convert to a positive-class score
    # where 0.50 corresponds to the classic distress cutoff Z <= 1.81.
    z_delta = np.clip(z_score.to_numpy(dtype=float) - ALTMAN_DISTRESS_CUTOFF, -50, 50)
    scores = 1.0 / (1.0 + np.exp(z_delta))
    threshold = 0.50
    y_test = bundle["y_test"]

    metrics, y_pred = evaluate_predictions(y_test, scores, threshold=threshold)
    prediction_path = save_prediction_table(
        model_key=ALTMAN_MODEL_KEY,
        meta_test=bundle["meta_test"],
        y_test=y_test,
        y_pred=y_pred,
        scores=scores,
        threshold=threshold,
        output_dir=output_dir,
    )

    component_df = components.copy()
    if bundle["meta_test"] is not None:
        component_df = pd.concat([bundle["meta_test"].reset_index(drop=True), component_df], axis=1)
    component_path = output_dir / "altman_z_score_components.csv"
    component_df.to_csv(component_path, index=False)

    print_metrics("Altman Z-score Proxy", "rule_based_raw_proxy", threshold, metrics)
    print(f"Altman distress cutoff: Z <= {ALTMAN_DISTRESS_CUTOFF:.2f}")
    print(f"Saved Altman components: {component_path}")

    result = {
        "model": "Altman Z-score Proxy",
        "input_data": "rule_based_raw_proxy",
        "threshold": threshold,
        "artifact_loader": "formula",
        "model_path": "",
        "prediction_path": str(prediction_path),
        **metrics,
    }
    payload = {
        "model_key": ALTMAN_MODEL_KEY,
        "model": "Altman Z-score Proxy",
        "threshold": threshold,
        "scores": np.asarray(scores, dtype=float),
        "y_true": y_test.reset_index(drop=True),
        "metrics": metrics,
    }
    return result, payload


def print_metrics(model_name: str, dataset_kind: str, threshold: float, metrics: dict[str, float]) -> None:
    print("\n" + "=" * 60)
    print(f"{model_name} | input_data={dataset_kind} | threshold={threshold:.2f}")
    print("=" * 60)
    print(f"Accuracy   : {metrics['accuracy']:.4f}")
    print(f"AUC-ROC    : {metrics['auc']:.4f}")
    print(f"F1-score   : {metrics['f1']:.4f}")
    print(f"F2-score   : {metrics['f2']:.4f}")
    print(f"Precision  : {metrics['precision']:.4f}")
    print(f"Recall     : {metrics['recall']:.4f}")
    print(f"Specificity: {metrics['specificity']:.4f}")
    print(f"Confusion  : TN={metrics['tn']} FP={metrics['fp']} FN={metrics['fn']} TP={metrics['tp']}")


def try_load_with_joblib_then_pickle(model_path: Path) -> Any:
    try:
        return joblib.load(model_path)
    except Exception:
        with model_path.open("rb") as fh:
            return pickle.load(fh)


def load_model_artifact(model_path: Path, model_key: str) -> dict[str, Any]:
    if not model_path.exists():
        raise FileNotFoundError(f"Khong tim thay model file: {model_path}")

    obj = try_load_with_joblib_then_pickle(model_path)

    if isinstance(obj, dict) and "model" in obj:
        artifact = obj.copy()
        artifact.setdefault("threshold", 0.50)
        artifact.setdefault("features", DEFAULT_FEATURES)
        artifact.setdefault("input_data", MODEL_DATASETS.get(model_key, "raw"))
        artifact["loader"] = "joblib_or_pickle"
        return artifact

    return {
        "model": obj,
        "threshold": 0.50,
        "features": DEFAULT_FEATURES,
        "input_data": MODEL_DATASETS.get(model_key, "raw"),
        "loader": "joblib_or_pickle",
    }


def save_prediction_table(
    model_key: str,
    meta_test: pd.DataFrame | None,
    y_test: pd.Series,
    y_pred: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    output_dir: Path,
) -> Path:
    result_df = pd.DataFrame(
        {
            "actual_target": y_test.reset_index(drop=True),
            "predicted_target": pd.Series(y_pred).reset_index(drop=True),
            "score_positive_class": pd.Series(scores).reset_index(drop=True),
            "threshold_used": threshold,
        }
    )
    if meta_test is not None:
        result_df = pd.concat([meta_test.reset_index(drop=True), result_df], axis=1)

    output_path = output_dir / f"evaluation_predictions_{model_key}.csv"
    result_df.to_csv(output_path, index=False)
    return output_path


def find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lowered = {col.lower(): col for col in df.columns}
    for name in candidates:
        if name.lower() in lowered:
            return lowered[name.lower()]
    return None


def infer_split_semantics(config: dict[str, Any]) -> dict[str, Any]:
    split_method = config.get("split_method", "unknown")
    return {
        "split_method": split_method,
        "train_cutoff_year": config.get("train_cutoff_year"),
        "test_year": config.get("test_year"),
        "exclude_finance": config.get("exclude_finance"),
    }


def audit_split(bundle: dict[str, Any]) -> dict[str, Any]:
    meta_train = bundle.get("meta_train")
    meta_test = bundle.get("meta_test")
    config = bundle.get("config", {})

    audit: dict[str, Any] = {
        "available": meta_train is not None and meta_test is not None,
        "notes": [],
        "split_info": infer_split_semantics(config),
    }

    if meta_train is None or meta_test is None:
        audit["notes"].append("Khong co meta_train/meta_test, bo qua split audit.")
        return audit

    firm_col = find_column(meta_train, ["Ma", "ma", "ticker", "symbol", "firm_id", "company_id", "code"])
    year_col = find_column(meta_train, ["nam", "year", "fiscal_year"])
    finance_col = find_column(meta_train, ["is_finance", "finance_flag"])

    audit["firm_column"] = firm_col
    audit["year_column"] = year_col
    audit["finance_column"] = finance_col
    audit["train_rows"] = int(len(meta_train))
    audit["test_rows"] = int(len(meta_test))

    if year_col is not None:
        audit["train_year_distribution"] = {
            str(k): int(v) for k, v in meta_train[year_col].value_counts().sort_index().to_dict().items()
        }
        audit["test_year_distribution"] = {
            str(k): int(v) for k, v in meta_test[year_col].value_counts().sort_index().to_dict().items()
        }

    if firm_col is None or year_col is None:
        audit["notes"].append("Khong xac dinh duoc cot cong ty/nam de audit chi tiet.")
        return audit

    train_firms = meta_train[firm_col].astype(str)
    test_firms = meta_test[firm_col].astype(str)
    train_years = pd.to_numeric(meta_train[year_col], errors="coerce")
    test_years = pd.to_numeric(meta_test[year_col], errors="coerce")

    train_unique_firms = set(train_firms.dropna().unique().tolist())
    test_unique_firms = set(test_firms.dropna().unique().tolist())
    overlap_firms = train_unique_firms & test_unique_firms

    train_keys = set(
        zip(
            meta_train[firm_col].astype(str),
            pd.to_numeric(meta_train[year_col], errors="coerce").fillna(-1).astype(int),
        )
    )
    test_keys = set(
        zip(
            meta_test[firm_col].astype(str),
            pd.to_numeric(meta_test[year_col], errors="coerce").fillna(-1).astype(int),
        )
    )
    exact_key_overlap = train_keys & test_keys

    firm_to_train_years = (
        pd.DataFrame({firm_col: train_firms, year_col: train_years})
        .dropna()
        .groupby(firm_col)[year_col]
        .apply(lambda s: sorted(set(int(v) for v in s.tolist())))
        .to_dict()
    )

    same_company_in_train = 0
    past_year_in_train = 0
    future_year_in_train = 0
    both_past_and_future_in_train = 0

    for firm, year in zip(test_firms.tolist(), test_years.tolist()):
        if pd.isna(firm) or pd.isna(year):
            continue
        year_int = int(year)
        seen_years = firm_to_train_years.get(str(firm), [])
        if not seen_years:
            continue

        same_company_in_train += 1
        has_past_year = any(train_year < year_int for train_year in seen_years)
        has_future_year = any(train_year > year_int for train_year in seen_years)

        if has_past_year:
            past_year_in_train += 1
        if has_future_year:
            future_year_in_train += 1
        if has_past_year and has_future_year:
            both_past_and_future_in_train += 1

    audit["unique_firms"] = {
        "train": int(len(train_unique_firms)),
        "test": int(len(test_unique_firms)),
        "overlap": int(len(overlap_firms)),
    }
    audit["time_based_panel_check"] = {
        "same_company_in_both_sets": int(same_company_in_train),
        "past_year_in_train_for_test_rows": int(past_year_in_train),
        "future_year_in_train_for_test_rows": int(future_year_in_train),
        "both_past_and_future_in_train_for_test_rows": int(both_past_and_future_in_train),
        "fully_unseen_company_rows": int(len(meta_test) - same_company_in_train),
        "exact_same_ma_year_overlap": int(len(exact_key_overlap)),
    }

    split_method = audit["split_info"].get("split_method")
    if split_method == "time_based":
        if len(exact_key_overlap) == 0:
            audit["notes"].append(
                "Time-based split: cung cong ty xuat hien o train/test la binh thuong voi panel data; "
                "khong co overlap chinh xac theo (Ma, nam)."
            )
        else:
            audit["notes"].append(
                "CANH BAO: phat hien overlap chinh xac theo (Ma, nam) giua train va test."
            )
    else:
        if len(overlap_firms) > 0:
            audit["notes"].append(
                "Co overlap cong ty giua train va test. Neu day la random split panel data thi can xem xet leakage."
            )

    if finance_col is not None:
        train_finance_mask = meta_train[finance_col].fillna(False).astype(bool)
        test_finance_mask = meta_test[finance_col].fillna(False).astype(bool)

        train_finance_firms = set(meta_train.loc[train_finance_mask, firm_col].astype(str).unique().tolist())
        test_finance_firms = set(meta_test.loc[test_finance_mask, firm_col].astype(str).unique().tolist())

        audit["finance_rows"] = {
            "train": int(train_finance_mask.sum()),
            "test": int(test_finance_mask.sum()),
            "total": int(train_finance_mask.sum() + test_finance_mask.sum()),
        }
        audit["finance_unique_firms"] = {
            "train": int(len(train_finance_firms)),
            "test": int(len(test_finance_firms)),
            "union": int(len(train_finance_firms | test_finance_firms)),
            "overlap": int(len(train_finance_firms & test_finance_firms)),
        }

    return audit


def print_audit_summary(audit: dict[str, Any]) -> None:
    print("\n" + "=" * 60)
    print("DATA SPLIT AUDIT")
    print("=" * 60)

    if not audit.get("available", False):
        print("Khong co du meta_train/meta_test de audit.")
        return

    split_info = audit.get("split_info", {})
    if split_info:
        print(
            "Split info   : "
            f"method={split_info.get('split_method')} | "
            f"train_cutoff={split_info.get('train_cutoff_year')} | "
            f"test_year={split_info.get('test_year')} | "
            f"exclude_finance={split_info.get('exclude_finance')}"
        )

    print(f"Train rows    : {audit.get('train_rows')}")
    print(f"Test rows     : {audit.get('test_rows')}")

    unique_firms = audit.get("unique_firms", {})
    if unique_firms:
        print(
            "Unique firms  : "
            f"train={unique_firms.get('train')} | "
            f"test={unique_firms.get('test')} | "
            f"overlap={unique_firms.get('overlap')}"
        )

    year_train = audit.get("train_year_distribution")
    year_test = audit.get("test_year_distribution")
    if year_train is not None:
        print(f"Train years   : {year_train}")
    if year_test is not None:
        print(f"Test years    : {year_test}")

    panel_check = audit.get("time_based_panel_check", {})
    if panel_check:
        print(
            "Panel overlap : "
            f"same_company={panel_check.get('same_company_in_both_sets')} | "
            f"past_year={panel_check.get('past_year_in_train_for_test_rows')} | "
            f"future_year={panel_check.get('future_year_in_train_for_test_rows')} | "
            f"both={panel_check.get('both_past_and_future_in_train_for_test_rows')} | "
            f"unseen={panel_check.get('fully_unseen_company_rows')} | "
            f"same_(Ma,nam)={panel_check.get('exact_same_ma_year_overlap')}"
        )

    finance_rows = audit.get("finance_rows")
    finance_firms = audit.get("finance_unique_firms")
    if finance_rows is not None:
        print(
            "Finance rows  : "
            f"train={finance_rows.get('train')} | "
            f"test={finance_rows.get('test')} | "
            f"total={finance_rows.get('total')}"
        )
    if finance_firms is not None:
        print(
            "Finance firms : "
            f"train={finance_firms.get('train')} | "
            f"test={finance_firms.get('test')} | "
            f"union={finance_firms.get('union')} | "
            f"overlap={finance_firms.get('overlap')}"
        )

    notes = audit.get("notes", [])
    if notes:
        print("Notes        :")
        for note in notes:
            print(f"  - {note}")


def slugify_model_name(model_name: str) -> str:
    return model_name.strip().lower().replace(" ", "_")


def make_plots_dir(output_dir: Path) -> Path:
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    return plots_dir


def compute_threshold_metrics(y_true: pd.Series, scores: np.ndarray) -> pd.DataFrame:
    sklearn_api = require_sklearn()
    rows: list[dict[str, float]] = []

    for threshold in THRESHOLD_CURVE_GRID:
        y_pred = (scores >= threshold).astype(int)
        tn, fp, fn, tp = sklearn_api["confusion_matrix"](y_true, y_pred, labels=[0, 1]).ravel()
        rows.append(
            {
                "threshold": float(threshold),
                "accuracy": float(sklearn_api["accuracy_score"](y_true, y_pred)),
                "precision": float(sklearn_api["precision_score"](y_true, y_pred, zero_division=0)),
                "recall": float(sklearn_api["recall_score"](y_true, y_pred, zero_division=0)),
                "f2": float(sklearn_api["fbeta_score"](y_true, y_pred, beta=GRID_SEARCH_BETA, zero_division=0)),
                "specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan"),
            }
        )

    return pd.DataFrame(rows)


def save_metric_bar_comparison(comparison_df: pd.DataFrame, output_dir: Path) -> Path:
    plt = require_matplotlib()

    chart_df = comparison_df[["model", *(metric_key for metric_key, _ in COMPARISON_METRICS)]].copy()
    chart_df = chart_df.set_index("model")
    model_names = chart_df.index.tolist()
    metric_labels = [metric_label for _, metric_label in COMPARISON_METRICS]
    x = np.arange(len(metric_labels))
    bar_width = 0.22 if len(model_names) >= 3 else 0.8 / max(len(model_names), 1)

    fig, ax = plt.subplots(figsize=(13, 6.5))
    for idx, model_name in enumerate(model_names):
        offset = (idx - (len(model_names) - 1) / 2) * bar_width
        values = [float(chart_df.loc[model_name, metric_key]) for metric_key, _ in COMPARISON_METRICS]
        bars = ax.bar(
            x + offset,
            values,
            width=bar_width,
            label=model_name,
            color=PLOT_COLORS[idx % len(PLOT_COLORS)],
            edgecolor="white",
            linewidth=0.8,
        )
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.01, f"{value:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_title("Metric Bar Comparison", fontsize=14, fontweight="bold")
    ax.set_xlabel("Evaluation Metrics")
    ax.set_ylabel("Metric Value")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0, 1.08)
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.legend(title="Models")
    fig.tight_layout()

    output_path = output_dir / "metric_bar_comparison.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_roc_curve_comparison(model_payloads: list[dict[str, Any]], output_dir: Path) -> Path | None:
    plt = require_matplotlib()
    sklearn_api = require_sklearn()

    if len(pd.Series(model_payloads[0]["y_true"]).unique()) < 2:
        print("[WARN] Bo qua ROC curve comparison vi y_test chi co 1 class.")
        return None

    fig, ax = plt.subplots(figsize=(8, 6.5))
    for idx, payload in enumerate(model_payloads):
        fpr, tpr, _ = sklearn_api["roc_curve"](payload["y_true"], payload["scores"])
        ax.plot(
            fpr,
            tpr,
            label=f"{payload['model']} (AUC={payload['metrics']['auc']:.3f})",
            color=PLOT_COLORS[idx % len(PLOT_COLORS)],
            linewidth=2,
        )

    ax.plot([0, 1], [0, 1], linestyle="--", color="#666666", linewidth=1)
    ax.set_title("ROC Curve Comparison", fontsize=14, fontweight="bold")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.legend()
    fig.tight_layout()

    output_path = output_dir / "roc_curve_comparison.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_precision_recall_curve_comparison(model_payloads: list[dict[str, Any]], output_dir: Path) -> Path | None:
    plt = require_matplotlib()
    sklearn_api = require_sklearn()

    if len(pd.Series(model_payloads[0]["y_true"]).unique()) < 2:
        print("[WARN] Bo qua Precision-Recall curve comparison vi y_test chi co 1 class.")
        return None

    fig, ax = plt.subplots(figsize=(8, 6.5))
    for idx, payload in enumerate(model_payloads):
        precision, recall, _ = sklearn_api["precision_recall_curve"](payload["y_true"], payload["scores"])
        ax.plot(
            recall,
            precision,
            label=f"{payload['model']} (F2={payload['metrics']['f2']:.3f})",
            color=PLOT_COLORS[idx % len(PLOT_COLORS)],
            linewidth=2,
        )

    ax.set_title("Precision-Recall Curve Comparison", fontsize=14, fontweight="bold")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.legend()
    fig.tight_layout()

    output_path = output_dir / "precision_recall_curve_comparison.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_calibration_curve_comparison(model_payloads: list[dict[str, Any]], output_dir: Path) -> Path | None:
    plt = require_matplotlib()
    sklearn_api = require_sklearn()

    if len(pd.Series(model_payloads[0]["y_true"]).unique()) < 2:
        print("[WARN] Bo qua Calibration curve comparison vi y_test chi co 1 class.")
        return None

    fig, ax = plt.subplots(figsize=(8, 6.5))
    for idx, payload in enumerate(model_payloads):
        frac_pos, mean_pred = sklearn_api["calibration_curve"](
            payload["y_true"],
            payload["scores"],
            n_bins=10,
            strategy="uniform",
        )
        ax.plot(
            mean_pred,
            frac_pos,
            marker="o",
            label=payload["model"],
            color=PLOT_COLORS[idx % len(PLOT_COLORS)],
            linewidth=2,
        )

    ax.plot([0, 1], [0, 1], linestyle="--", color="#666666", linewidth=1, label="Perfect calibration")
    ax.set_title("Calibration Curve Comparison", fontsize=14, fontweight="bold")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.legend()
    fig.tight_layout()

    output_path = output_dir / "calibration_curve_comparison.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_confusion_matrix_plot(payload: dict[str, Any], output_dir: Path) -> Path:
    plt = require_matplotlib()

    matrix = np.array(
        [
            [payload["metrics"]["tn"], payload["metrics"]["fp"]],
            [payload["metrics"]["fn"], payload["metrics"]["tp"]],
        ]
    )

    fig, ax = plt.subplots(figsize=(6, 5.5))
    im = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for row_idx in range(2):
        for col_idx in range(2):
            ax.text(col_idx, row_idx, f"{matrix[row_idx, col_idx]}", ha="center", va="center", color="#111111", fontsize=12)

    ax.set_title(f"Confusion Matrix - {payload['model']}", fontsize=13, fontweight="bold")
    ax.set_xticks([0, 1], labels=["Pred 0", "Pred 1"])
    ax.set_yticks([0, 1], labels=["True 0", "True 1"])
    fig.tight_layout()

    output_path = output_dir / "confusion_matrix.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_threshold_curves(payload: dict[str, Any], output_dir: Path) -> list[Path]:
    plt = require_matplotlib()
    threshold_df = compute_threshold_metrics(payload["y_true"], payload["scores"])
    threshold_df.to_csv(output_dir / "threshold_metrics.csv", index=False)

    saved_paths: list[Path] = []

    fig, ax = plt.subplots(figsize=(9.5, 6))
    ax.plot(threshold_df["threshold"], threshold_df["precision"], label="Precision", linewidth=2, color="#F58518")
    ax.plot(threshold_df["threshold"], threshold_df["recall"], label="Recall", linewidth=2, color="#54A24B")
    ax.plot(threshold_df["threshold"], threshold_df["f2"], label="F2", linewidth=2, color="#4C78A8")
    ax.plot(threshold_df["threshold"], threshold_df["accuracy"], label="Accuracy", linewidth=2, color="#E45756")
    ax.plot(threshold_df["threshold"], threshold_df["specificity"], label="Specificity", linewidth=2, color="#72B7B2")
    ax.axvline(payload["threshold"], linestyle="--", color="#333333", linewidth=1.2, label=f"Chosen threshold={payload['threshold']:.2f}")
    ax.set_title(f"Threshold Curve - {payload['model']}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Metric Value")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    threshold_curve_path = output_dir / "threshold_curve.png"
    fig.savefig(threshold_curve_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    saved_paths.append(threshold_curve_path)

    metric_specs = [
        ("precision", "Precision theo Threshold", "#F58518", "precision_by_threshold.png"),
        ("recall", "Recall theo Threshold", "#54A24B", "recall_by_threshold.png"),
        ("f2", "F2 theo Threshold", "#4C78A8", "f2_by_threshold.png"),
    ]
    for metric_key, title, color, file_name in metric_specs:
        fig, ax = plt.subplots(figsize=(8.5, 5.5))
        ax.plot(threshold_df["threshold"], threshold_df[metric_key], color=color, linewidth=2.2)
        ax.axvline(payload["threshold"], linestyle="--", color="#333333", linewidth=1.2, label=f"Chosen threshold={payload['threshold']:.2f}")
        ax.set_title(f"{title} - {payload['model']}", fontsize=13, fontweight="bold")
        ax.set_xlabel("Threshold")
        ax.set_ylabel(metric_key.capitalize())
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
        ax.grid(True, linestyle="--", alpha=0.25)
        ax.legend()
        fig.tight_layout()
        metric_path = output_dir / file_name
        fig.savefig(metric_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(metric_path)

    return saved_paths


def save_score_distribution_plot(payload: dict[str, Any], output_dir: Path) -> Path:
    plt = require_matplotlib()

    y_true = pd.Series(payload["y_true"]).reset_index(drop=True)
    scores = pd.Series(payload["scores"]).reset_index(drop=True)
    negative_scores = scores[y_true == 0]
    positive_scores = scores[y_true == 1]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.hist(negative_scores, bins=20, alpha=0.6, color="#4C78A8", label="Class 0", density=True)
    ax.hist(positive_scores, bins=20, alpha=0.6, color="#E45756", label="Class 1", density=True)
    ax.axvline(payload["threshold"], linestyle="--", color="#222222", linewidth=1.2, label=f"Chosen threshold={payload['threshold']:.2f}")
    ax.set_title(f"Score Distribution - {payload['model']}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicted Probability of Positive Class")
    ax.set_ylabel("Density")
    ax.set_xlim(0, 1)
    ax.grid(True, linestyle="--", alpha=0.2)
    ax.legend()
    fig.tight_layout()

    output_path = output_dir / "score_distribution.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_model_level_plots(payload: dict[str, Any], plots_dir: Path) -> list[Path]:
    model_dir = plots_dir / slugify_model_name(payload["model"])
    model_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = [save_confusion_matrix_plot(payload, model_dir)]
    saved_paths.extend(save_threshold_curves(payload, model_dir))
    saved_paths.append(save_score_distribution_plot(payload, model_dir))
    return saved_paths


def save_comparison_plots(comparison_df: pd.DataFrame, model_payloads: list[dict[str, Any]], output_dir: Path) -> list[Path]:
    plots_dir = make_plots_dir(output_dir)
    saved_paths: list[Path] = []

    saved_paths.append(save_metric_bar_comparison(comparison_df, plots_dir))

    roc_path = save_roc_curve_comparison(model_payloads, plots_dir)
    if roc_path is not None:
        saved_paths.append(roc_path)

    pr_path = save_precision_recall_curve_comparison(model_payloads, plots_dir)
    if pr_path is not None:
        saved_paths.append(pr_path)

    calibration_path = save_calibration_curve_comparison(model_payloads, plots_dir)
    if calibration_path is not None:
        saved_paths.append(calibration_path)

    for payload in model_payloads:
        saved_paths.extend(save_model_level_plots(payload, plots_dir))

    return saved_paths


def evaluate_saved_models(
    preprocessed_dir: str | Path | None = None,
    models_dir: str | Path | None = None,
    models: list[str] | tuple[str, ...] | None = None,
    output_dir: str | Path | None = None,
    run_audit: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any] | None]:
    requested_models = normalize_requested_models(models)
    bundle = load_preprocessed_bundle(resolve_preprocessed_dir(str(preprocessed_dir) if preprocessed_dir is not None else None))
    resolved_models_dir = resolve_models_dir(str(models_dir) if models_dir is not None else None)

    if output_dir is None:
        final_output_dir = resolved_models_dir
    else:
        final_output_dir = Path(output_dir).expanduser().resolve()
        final_output_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    model_payloads: list[dict[str, Any]] = []

    try:
        altman_result, altman_payload = evaluate_altman_z_score_baseline(bundle, final_output_dir)
        results.append(altman_result)
        model_payloads.append(altman_payload)
    except Exception as exc:
        print(f"\n[ERROR] Altman Z-score Proxy failed: {exc}")

    for model_key in requested_models:
        try:
            model_path = resolved_models_dir / MODEL_FILE_NAMES[model_key]
            artifact = load_model_artifact(model_path, model_key)
            model = artifact["model"]
            threshold = float(artifact.get("threshold", 0.50))
            dataset_kind = str(artifact.get("input_data", MODEL_DATASETS[model_key]))
            artifact_features = list(artifact.get("features", bundle["features"]))

            available_bundle_features = set(bundle["features"])
            missing = [feat for feat in artifact_features if feat not in available_bundle_features]
            if missing:
                raise ValueError(f"Feature mismatch. Cac feature thieu trong preprocessing bundle: {missing}")

            _, X_test = get_dataset_pair(bundle, dataset_kind)
            X_test_eval = X_test[artifact_features].copy()
            y_test = bundle["y_test"]

            scores = safe_predict_scores(model, X_test_eval)
            metrics, y_pred = evaluate_predictions(y_test, scores, threshold=threshold)
            prediction_path = save_prediction_table(
                model_key=model_key,
                meta_test=bundle["meta_test"],
                y_test=y_test,
                y_pred=y_pred,
                scores=scores,
                threshold=threshold,
                output_dir=final_output_dir,
            )

            print_metrics(MODEL_DISPLAY_NAMES[model_key], dataset_kind, threshold, metrics)

            result = {
                "model": MODEL_DISPLAY_NAMES[model_key],
                "input_data": dataset_kind,
                "threshold": threshold,
                "artifact_loader": artifact.get("loader", "unknown"),
                "model_path": str(model_path),
                "prediction_path": str(prediction_path),
                **metrics,
            }
            results.append(result)
            model_payloads.append(
                {
                    "model_key": model_key,
                    "model": MODEL_DISPLAY_NAMES[model_key],
                    "threshold": threshold,
                    "scores": np.asarray(scores, dtype=float),
                    "y_true": y_test.reset_index(drop=True),
                    "metrics": metrics,
                }
            )
        except Exception as exc:
            print(f"\n[ERROR] {MODEL_DISPLAY_NAMES[model_key]} failed: {exc}")

    if not results:
        raise RuntimeError("Khong model nao evaluate duoc. Hay kiem tra model file va preprocessing output.")

    comparison_df = pd.DataFrame(results)[
        [
            "model",
            "input_data",
            "threshold",
            "accuracy",
            "auc",
            "f1",
            "f2",
            "precision",
            "recall",
            "specificity",
            "tn",
            "fp",
            "fn",
            "tp",
            "artifact_loader",
            "model_path",
            "prediction_path",
        ]
    ].sort_values(["f2", "recall", "auc"], ascending=[False, False, False], na_position="last")

    eval_path = final_output_dir / DEFAULT_OUTPUT_FILE
    comparison_df.to_excel(eval_path, index=False)
    comparison_df.to_csv(eval_path.with_suffix(".csv"), index=False)
    saved_plot_paths = save_comparison_plots(comparison_df, model_payloads, final_output_dir)

    audit_result = audit_split(bundle) if run_audit else None
    if audit_result is not None:
        audit_path = final_output_dir / DEFAULT_AUDIT_FILE
        audit_path.write_text(json.dumps(audit_result, indent=2, ensure_ascii=False), encoding="utf-8")
        print_audit_summary(audit_result)
        print(f"\nSaved split audit to: {audit_path}")

    print("\n" + "=" * 60)
    print("MODEL EVALUATION SUMMARY")
    print("=" * 60)
    print(comparison_df.to_string(index=False))
    print(f"\nSaved evaluation table to: {eval_path}")
    print(f"Saved evaluation table to: {eval_path.with_suffix('.csv')}")
    if saved_plot_paths:
        print("Saved plots:")
        for plot_path in saved_plot_paths:
            print(f"  - {plot_path}")

    return comparison_df, audit_result


def main() -> None:
    ensure_utf8_output()
    load_env()

    parser = argparse.ArgumentParser(description="Evaluate saved models on preprocessing outputs.")
    parser.add_argument(
        "--preprocessed-dir",
        type=str,
        default=None,
        help="Thu muc chua output preprocessing.",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default=None,
        help="Thu muc chua model da train.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Thu muc luu bang evaluation va file audit. Mac dinh = models_dir",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Danh sach model can evaluate. Gia tri hop le: lr rf xgb logistic_regression random_forest xgboost all",
    )
    parser.add_argument(
        "--skip-audit",
        action="store_true",
        help="Bo qua split audit train/test.",
    )
    args = parser.parse_args()

    try:
        evaluate_saved_models(
            preprocessed_dir=args.preprocessed_dir,
            models_dir=args.models_dir,
            output_dir=args.output_dir,
            models=args.models,
            run_audit=not args.skip_audit,
        )
    except Exception as exc:
        print(f"\n[FATAL] {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
