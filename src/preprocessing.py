from __future__ import annotations

from pathlib import Path
import argparse
import json
import pickle
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import skew
from env_config import (
    load_env,
    resolve_directory as env_resolve_directory,
    resolve_input_file as env_resolve_input_file,
)


FEATURES = ["CR", "ROS", "DS", "TAT", "SIZE", "is_HNX"]
TARGET_COL = "target"
DEFAULT_FILE_NAME = "DATA_SUPERVISED.xlsx"
DEFAULT_OUTPUT_SUBDIR = Path("output") / "preprocessing"
DEFAULT_INPUT_ENV_VAR = "SUPERVISED_DATA_FILE"
DEFAULT_OUTPUT_ENV_VAR = "PREPROCESSING_OUTPUT_DIR"

YEAR_COL = "nam"
FINANCE_COL = "is_finance"
META_COLUMNS = ["Ma", "San", "nam", "is_finance", "ROS_next"]

TRAIN_CUTOFF_YEAR = 2023
TEST_YEAR = 2024

# "raw" trong pipeline model = da qua feature engineering + clean +
# winsorize/log transform, NHUNG chua standard scale.
LOG_FEATURES = ["CR", "TAT"]
WINSORIZE_FEATURES = ["ROS", "TAT"]
WINSORIZE_BOUNDS = (0.01, 0.99)


class ManualStandardScaler:
    def __init__(self) -> None:
        self.feature_names_: list[str] | None = None
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def fit(self, X: pd.DataFrame) -> "ManualStandardScaler":
        mean = X.mean(axis=0)
        scale = X.std(axis=0, ddof=0).replace(0, 1.0)
        self.feature_names_ = list(X.columns)
        self.mean_ = mean.to_numpy(dtype=float)
        self.scale_ = scale.to_numpy(dtype=float)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.feature_names_ is None or self.mean_ is None or self.scale_ is None:
            raise ValueError("Scaler chua duoc fit.")
        mean = pd.Series(self.mean_, index=self.feature_names_)
        scale = pd.Series(self.scale_, index=self.feature_names_)
        values = (X[self.feature_names_] - mean) / scale
        return pd.DataFrame(values.to_numpy(), columns=self.feature_names_, index=X.index)

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)


def ensure_utf8_output() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")


def resolve_input_file(user_input: str | None) -> Path:
    return env_resolve_input_file(
        user_input,
        DEFAULT_INPUT_ENV_VAR,
        Path("Data") / "train" / DEFAULT_FILE_NAME,
    )


def safe_skew(series: pd.Series) -> float:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if len(clean) < 3 or clean.nunique() < 2:
        return float("nan")
    return float(skew(clean, bias=False))


def prepare_dataframe(df: pd.DataFrame, allow_build_target_from_current_ros: bool) -> tuple[pd.DataFrame, list[str], str, int]:
    prepared = df.copy()

    if "San" in prepared.columns:
        prepared["San"] = prepared["San"].astype(str).str.strip().str.upper()
        prepared["San"] = prepared["San"].replace({"HSX": "HOSE"})
        prepared["is_HNX"] = (prepared["San"] == "HNX").astype(int)
    elif "is_HNX" not in prepared.columns:
        raise ValueError("Thieu cot 'San' hoac 'is_HNX' de tao feature is_HNX.")

    missing_features = [col for col in FEATURES if col not in prepared.columns]
    if missing_features:
        raise ValueError(f"Thieu feature bat buoc: {missing_features}")

    for col in FEATURES:
        prepared[col] = pd.to_numeric(prepared[col], errors="coerce")

    target_source = "existing_target"
    if TARGET_COL not in prepared.columns:
        if not allow_build_target_from_current_ros:
            raise ValueError(
                "Khong co cot target. "
                "Voi bai toan nay, ban nen dua vao DATA_SUPERVISED.xlsx co target san. "
                "Chi bat fallback bang --allow-build-target-from-current-ros khi that su can."
            )
        if "ROS" not in prepared.columns:
            raise ValueError("Thieu ca target va ROS, khong the tao target fallback.")
        prepared[TARGET_COL] = (pd.to_numeric(prepared["ROS"], errors="coerce") < 0.05).astype(float)
        target_source = "built_from_current_ros"

    prepared[TARGET_COL] = pd.to_numeric(prepared[TARGET_COL], errors="coerce")
    meta_cols = [col for col in META_COLUMNS if col in prepared.columns]

    row_count_before = len(prepared)
    prepared = prepared.dropna(subset=FEATURES + [TARGET_COL]).copy()
    prepared = prepared[prepared[TARGET_COL].isin([0, 1])].copy()
    prepared[TARGET_COL] = prepared[TARGET_COL].astype(int)
    dropped_rows = row_count_before - len(prepared)

    if prepared.empty:
        raise ValueError("Khong con dong hop le sau khi lam sach du lieu.")
    if YEAR_COL not in prepared.columns:
        raise ValueError(f"Thieu cot '{YEAR_COL}' trong data.")
    if prepared[TARGET_COL].nunique() < 2:
        raise ValueError("Target chi con 1 class sau khi lam sach, khong the train.")

    dup_keys = prepared.duplicated(subset=["Ma", YEAR_COL]).sum() if "Ma" in prepared.columns else 0
    if dup_keys > 0:
        raise ValueError(f"Phat hien {dup_keys} dong trung (Ma, nam). Can xu ly duplicate truoc khi preprocess.")

    return prepared, meta_cols, target_source, dropped_rows


def time_based_split(
    df: pd.DataFrame,
    train_cutoff: int,
    test_year: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = df.loc[df[YEAR_COL] <= train_cutoff].copy()
    test_df = df.loc[df[YEAR_COL] == test_year].copy()

    if train_df.empty:
        raise ValueError(f"Khong co dong nao co nam <= {train_cutoff}.")
    if test_df.empty:
        raise ValueError(f"Khong co dong nao co nam == {test_year}.")

    if train_df[TARGET_COL].nunique() < 2:
        raise ValueError("Train chi co 1 class target.")
    if test_df[TARGET_COL].nunique() < 2:
        print("[WARN] Test chi co 1 class target. AUC/threshold evaluation co the kem on dinh.")

    return train_df, test_df


def fit_winsorize_params(df_train_features: pd.DataFrame) -> dict[str, dict[str, float]]:
    params: dict[str, dict[str, float]] = {}
    for col in WINSORIZE_FEATURES:
        lo = float(df_train_features[col].quantile(WINSORIZE_BOUNDS[0]))
        hi = float(df_train_features[col].quantile(WINSORIZE_BOUNDS[1]))
        params[col] = {"lo": lo, "hi": hi}
    return params


def apply_winsorize(df_features: pd.DataFrame, params: dict[str, dict[str, float]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    processed = df_features.copy()
    rows: list[dict[str, Any]] = []

    for col, bounds in params.items():
        lo = float(bounds["lo"])
        hi = float(bounds["hi"])
        affected = int(((processed[col] < lo) | (processed[col] > hi)).sum())
        processed[col] = processed[col].clip(lower=lo, upper=hi)
        rows.append({
            "feature": col,
            "step": "winsorize",
            "lower_bound": lo,
            "upper_bound": hi,
            "n_affected": affected,
        })

    return processed, pd.DataFrame(rows)


def fit_log_plan(df_train_features: pd.DataFrame) -> dict[str, dict[str, Any]]:
    plan: dict[str, dict[str, Any]] = {}
    for col in LOG_FEATURES:
        negative_count = int((df_train_features[col] < 0).sum())
        apply_log = negative_count == 0
        plan[col] = {
            "apply": apply_log,
            "negative_count_train": negative_count,
            "skew_before_train": safe_skew(df_train_features[col]),
        }
    return plan


def apply_log_plan(df_features: pd.DataFrame, plan: dict[str, dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    processed = df_features.copy()
    rows: list[dict[str, Any]] = []

    for col, info in plan.items():
        apply_log = bool(info["apply"])
        if apply_log:
            processed[col] = np.log1p(processed[col])
        rows.append({
            "feature": col,
            "step": "log1p",
            "applied": apply_log,
            "negative_count_in_data": int((df_features[col] < 0).sum()),
            "skew_before": safe_skew(df_features[col]),
            "skew_after": safe_skew(processed[col]) if apply_log else np.nan,
            "note": "" if apply_log else "skip_due_to_negative_values_in_train",
        })

    return processed, pd.DataFrame(rows)


def preprocess_train_test(
    X_train_raw: pd.DataFrame,
    X_test_raw: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], pd.DataFrame]:
    # Fit transform params tren TRAIN ONLY
    winsor_params = fit_winsorize_params(X_train_raw)
    X_train_win, train_win_report = apply_winsorize(X_train_raw, winsor_params)
    X_test_win, test_win_report = apply_winsorize(X_test_raw, winsor_params)

    log_plan = fit_log_plan(X_train_win)
    X_train_proc, train_log_report = apply_log_plan(X_train_win, log_plan)
    X_test_proc, test_log_report = apply_log_plan(X_test_win, log_plan)

    report = pd.concat([
        train_win_report.assign(dataset="train"),
        test_win_report.assign(dataset="test"),
        train_log_report.assign(dataset="train"),
        test_log_report.assign(dataset="test"),
    ], ignore_index=True, sort=False)

    artifacts = {
        "winsorize_params": winsor_params,
        "log_plan": log_plan,
    }
    return X_train_proc, X_test_proc, artifacts, report


def plot_transform_before_after(
    train_before: pd.DataFrame,
    train_after: pd.DataFrame,
    test_before: pd.DataFrame,
    test_after: pd.DataFrame,
    output_path: Path,
) -> None:
    transform_cols = list(dict.fromkeys(WINSORIZE_FEATURES + LOG_FEATURES))
    fig, axes = plt.subplots(len(transform_cols), 4, figsize=(18, 4 * len(transform_cols)))

    if len(transform_cols) == 1:
        axes = np.array([axes])

    for idx, col in enumerate(transform_cols):
        panels = [
            ("Train before", train_before[col]),
            ("Train after", train_after[col]),
            ("Test before", test_before[col]),
            ("Test after", test_after[col]),
        ]

        for j, (title, series) in enumerate(panels):
            s = pd.to_numeric(series, errors="coerce").dropna()
            ax = axes[idx, j]
            ax.hist(s, bins=40, density=True, alpha=0.7, edgecolor="white")
            ax.set_title(f"{col} - {title}\nskew={safe_skew(s):.3f}")
            ax.set_xlabel(col)
            ax.set_ylabel("Density")
            ax.grid(alpha=0.25)

    plt.suptitle("Feature distributions before and after preprocessing", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_stats_table(df_features: pd.DataFrame) -> pd.DataFrame:
    stats_df = df_features.describe().T
    stats_df["skewness"] = df_features.apply(safe_skew)
    ordered_cols = ["count", "mean", "std", "min", "25%", "50%", "75%", "max", "skewness"]
    return stats_df[ordered_cols]


def save_target_csv(target: pd.Series, output_path: Path) -> None:
    pd.DataFrame({TARGET_COL: target}).to_csv(output_path, index=False)


def save_meta_csv(meta: pd.DataFrame | None, output_path: Path) -> None:
    if meta is not None and not meta.empty:
        meta.to_csv(output_path, index=False)


def print_year_distribution(df: pd.DataFrame, label: str) -> None:
    print(f"\nPhan phoi nam [{label}]:")
    for yr, cnt in sorted(df[YEAR_COL].value_counts().sort_index().items()):
        print(f"  {yr}: {cnt}")


def main() -> None:
    ensure_utf8_output()
    load_env()

    parser = argparse.ArgumentParser(description="Preprocess DATA_SUPERVISED for modeling.")
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--train-cutoff", type=int, default=TRAIN_CUTOFF_YEAR)
    parser.add_argument("--test-year", type=int, default=TEST_YEAR)
    parser.add_argument("--keep-finance", action="store_true", help="Giu lai cong ty tai chinh (mac dinh: loai bo)")
    parser.add_argument(
        "--allow-build-target-from-current-ros",
        action="store_true",
        help="Chi bat khi input khong co target va ban chap nhan fallback tao target tu ROS hien tai.",
    )
    args = parser.parse_args()

    output_dir = env_resolve_directory(
        args.output_dir,
        DEFAULT_OUTPUT_ENV_VAR,
        Path("src") / DEFAULT_OUTPUT_SUBDIR,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    file_path = resolve_input_file(args.input)

    print("=" * 80)
    print("PREPROCESSING PIPELINE (updated)")
    print("=" * 80)
    print(f"Input file      : {file_path}")
    print(f"Output dir      : {output_dir.resolve()}")
    print(f"Train nam <=    : {args.train_cutoff}")
    print(f"Test  nam ==    : {args.test_year}")
    print(f"Loai tai chinh  : {not args.keep_finance}")

    df = pd.read_excel(file_path)
    prepared_df, meta_cols, target_source, dropped_rows = prepare_dataframe(
        df,
        allow_build_target_from_current_ros=args.allow_build_target_from_current_ros,
    )

    print(f"\nSo dong ban dau : {len(df)}")
    print(f"So dong hop le  : {len(prepared_df)}")
    print(f"So dong bi loai : {dropped_rows}")
    print(f"Nguon target    : {target_source}")

    excluded_finance = 0
    if not args.keep_finance and FINANCE_COL in prepared_df.columns:
        mask_tc = prepared_df[FINANCE_COL].fillna(False).astype(bool)
        excluded_finance = int(mask_tc.sum())
        prepared_df = prepared_df.loc[~mask_tc].copy()
        print(f"Loai cong ty TC : {excluded_finance} dong")

    print(f"Con lai de split: {len(prepared_df)}")
    print(f"Class balance   : {prepared_df[TARGET_COL].value_counts().sort_index().to_dict()}")

    if "is_HNX" in prepared_df.columns:
        hnx_count = int(prepared_df["is_HNX"].sum())
        hose_count = len(prepared_df) - hnx_count
        hnx_target = prepared_df.loc[prepared_df["is_HNX"] == 1, TARGET_COL].mean()
        hose_target = prepared_df.loc[prepared_df["is_HNX"] == 0, TARGET_COL].mean()
        print("\nFeature is_HNX:")
        print(f"  HNX  (is_HNX=1): {hnx_count:4d} dong | target ratio = {hnx_target:.1%}")
        print(f"  HOSE (is_HNX=0): {hose_count:4d} dong | target ratio = {hose_target:.1%}")

    print_year_distribution(prepared_df, "all_cleaned")

    train_df, test_df = time_based_split(prepared_df, args.train_cutoff, args.test_year)
    print_year_distribution(train_df, "train")
    print_year_distribution(test_df, "test")

    if "Ma" in train_df.columns and "Ma" in test_df.columns:
        company_overlap = set(train_df["Ma"].astype(str).unique()) & set(test_df["Ma"].astype(str).unique())
        train_keys = set(zip(train_df["Ma"].astype(str), train_df[YEAR_COL].astype(int)))
        test_keys = set(zip(test_df["Ma"].astype(str), test_df[YEAR_COL].astype(int)))
        key_overlap = train_keys & test_keys
        print("\nKiem tra leakage (time-based split):")
        print(f"  Cong ty xuat hien ca 2 tap : {len(company_overlap)} (binh thuong voi panel data)")
        print(f"  Trung theo (Ma, nam)       : {len(key_overlap)} ({'CANH BAO' if key_overlap else 'OK'})")

    X_train_base = train_df[FEATURES].copy()
    X_test_base = test_df[FEATURES].copy()
    y_train = train_df[TARGET_COL].copy()
    y_test = test_df[TARGET_COL].copy()
    meta_train = train_df[meta_cols].copy() if meta_cols else None
    meta_test = test_df[meta_cols].copy() if meta_cols else None

    # FIX QUAN TRONG: fit transform params tren TRAIN ONLY, sau do apply sang TEST
    X_train, X_test, preprocess_artifacts, transform_report = preprocess_train_test(X_train_base, X_test_base)

    plot_transform_before_after(
        train_before=X_train_base,
        train_after=X_train,
        test_before=X_test_base,
        test_after=X_test,
        output_path=output_dir / "transform_before_after.png",
    )

    scaler = ManualStandardScaler().fit(X_train)
    scaler_mean = pd.Series(scaler.mean_, index=FEATURES)
    scaler_scale = pd.Series(scaler.scale_, index=FEATURES)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    stats_train_after = build_stats_table(X_train)
    stats_test_after = build_stats_table(X_test)

    X_train.to_csv(output_dir / "X_train.csv", index=False)
    X_test.to_csv(output_dir / "X_test.csv", index=False)
    X_train_scaled.to_csv(output_dir / "X_train_scaled.csv", index=False)
    X_test_scaled.to_csv(output_dir / "X_test_scaled.csv", index=False)
    save_target_csv(y_train, output_dir / "y_train.csv")
    save_target_csv(y_test, output_dir / "y_test.csv")
    save_meta_csv(meta_train, output_dir / "meta_train.csv")
    save_meta_csv(meta_test, output_dir / "meta_test.csv")

    scaler_payload = {
        "scaler_type": "manual_standard_scaler",
        "feature_names": FEATURES,
        "mean": {col: float(scaler_mean[col]) for col in FEATURES},
        "scale": {col: float(scaler_scale[col]) for col in FEATURES},
    }
    with (output_dir / "scaler.pkl").open("wb") as fh:
        pickle.dump(scaler_payload, fh)

    with (output_dir / "preprocess_artifacts.pkl").open("wb") as fh:
        pickle.dump(preprocess_artifacts, fh)

    with pd.ExcelWriter(output_dir / "stats_after_transform.xlsx", engine="openpyxl") as writer:
        stats_train_after.to_excel(writer, sheet_name="train_after")
        stats_test_after.to_excel(writer, sheet_name="test_after")

    transform_report.to_csv(output_dir / "transform_report.csv", index=False)

    config = {
        "features": FEATURES,
        "target_col": TARGET_COL,
        "target_source": target_source,
        "split_method": "time_based",
        "train_cutoff_year": args.train_cutoff,
        "test_year": args.test_year,
        "exclude_finance": not args.keep_finance,
        "excluded_finance_rows": excluded_finance,
        "scaler_type": "manual_standard_scaler",
        "winsorize_features": WINSORIZE_FEATURES,
        "winsorize_bounds": list(WINSORIZE_BOUNDS),
        "winsorize_params_fit_on": "train_only",
        "winsorize_params": preprocess_artifacts["winsorize_params"],
        "log_features": LOG_FEATURES,
        "log_plan_fit_on": "train_only",
        "log_plan": preprocess_artifacts["log_plan"],
        "meta_columns": meta_cols,
        "train_size": int(len(X_train)),
        "test_size_rows": int(len(X_test)),
        "train_class_1_ratio": float(y_train.mean()),
        "test_class_1_ratio": float(y_test.mean()),
        "is_HNX_feature": True,
        "is_HNX_note": "HNX=1, HOSE=0, tao tu cot San",
    }
    (output_dir / "preprocessing_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\nTrain : {len(X_train)} dong | class 1 = {y_train.mean():.3f}")
    print(f"Test  : {len(X_test)} dong | class 1 = {y_test.mean():.3f}")

    print("\nScaler mean/std (fit tren train only):")
    for col in FEATURES:
        print(f"  {col}: mean={scaler_mean[col]:.6f}, std={scaler_scale[col]:.6f}")

    print("\nDa luu file:")
    saved = [
        "X_train.csv", "X_test.csv",
        "X_train_scaled.csv", "X_test_scaled.csv",
        "y_train.csv", "y_test.csv",
        "meta_train.csv" if meta_train is not None else None,
        "meta_test.csv" if meta_test is not None else None,
        "scaler.pkl", "preprocess_artifacts.pkl",
        "stats_after_transform.xlsx", "transform_report.csv",
        "transform_before_after.png", "preprocessing_config.json",
    ]
    for item in saved:
        if item:
            print(f"  - {item}")


if __name__ == "__main__":
    main()
