from pathlib import Path
import argparse
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis, gaussian_kde
from env_config import (
    load_env,
    resolve_directory as env_resolve_directory,
    resolve_input_file as env_resolve_input_file,
)

FEATURES = ["CR", "ROS", "DS", "TAT", "SIZE"]
DEFAULT_FILE_NAME = "DATA_FEATURED.xlsx"
DEFAULT_INPUT_ENV_VAR = "FEATURED_DATA_FILE"
DEFAULT_OUTPUT_ENV_VAR = "ANALYSIS_OUTPUT_DIR"


def ensure_utf8_output() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")


def resolve_input_file(user_path: str | None = None, default_file_name: str = DEFAULT_FILE_NAME) -> Path:
    return env_resolve_input_file(
        user_path,
        DEFAULT_INPUT_ENV_VAR,
        Path("Data") / "train" / default_file_name,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="EDA phan phoi cho 5 financial features.")
    parser.add_argument("--input", type=str, default=None, help="Duong dan file input.")
    parser.add_argument("--output-dir", type=str, default=None, help="Thu muc output.")
    parser.add_argument(
        "--exclude-finance",
        action="store_true",
        help="Loai cac dong is_finance=True truoc khi phan tich.",
    )
    parser.add_argument(
        "--by-year",
        action="store_true",
        help="Luu them thong ke mo ta theo nam.",
    )
    return parser.parse_args()


def summarize_missing(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in FEATURES:
        s = pd.to_numeric(df[col], errors="coerce")
        rows.append({
            "feature": col,
            "n_total": int(len(s)),
            "n_missing": int(s.isna().sum()),
            "pct_missing": float(s.isna().mean() * 100),
        })
    return pd.DataFrame(rows)


def build_descriptive_stats(data: pd.DataFrame) -> pd.DataFrame:
    desc = data.describe(percentiles=[0.25, 0.5, 0.75]).T
    desc["skewness"] = data.apply(lambda s: skew(s.dropna(), bias=False) if s.dropna().shape[0] > 2 else np.nan)
    desc["kurtosis"] = data.apply(lambda s: kurtosis(s.dropna(), fisher=True, bias=False) if s.dropna().shape[0] > 3 else np.nan)
    desc = desc[["count", "mean", "std", "min", "25%", "50%", "75%", "max", "skewness", "kurtosis"]]
    desc = desc.rename(columns={"25%": "q1", "50%": "median", "75%": "q3"})
    return desc


def build_outlier_iqr(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in FEATURES:
        s = data[col].dropna()
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        mask = (s < lower) | (s > upper)
        rows.append({
            "feature": col,
            "Q1": q1,
            "Q3": q3,
            "IQR": iqr,
            "lower_bound": lower,
            "upper_bound": upper,
            "n_outlier": int(mask.sum()),
            "pct_outlier": float(mask.mean() * 100) if len(mask) else np.nan,
        })
    return pd.DataFrame(rows)


def plot_histograms(data: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    axes = axes.flatten()

    for i, col in enumerate(FEATURES):
        ax = axes[i]
        s = data[col].dropna().astype(float)
        ax.hist(s, bins=35, density=True, alpha=0.65, edgecolor="black")

        if len(s) > 1 and s.nunique() > 1:
            x_grid = np.linspace(s.min(), s.max(), 500)
            kde = gaussian_kde(s)
            ax.plot(x_grid, kde(x_grid), linewidth=2, label="KDE")

        if col == "ROS":
            ax.axvline(0, linestyle="--", linewidth=1.5, label="0 (profit/loss)")
        if col == "CR" and len(s) > 0:
            ax.axvline(s.max(), linestyle=":", linewidth=1.5, label=f"max={s.max():.2f}")

        ax.set_title(f"Histogram + KDE: {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Density")
        ax.grid(alpha=0.25)
        ax.legend()

    for j in range(len(FEATURES), len(axes)):
        axes[j].axis("off")

    plt.suptitle("Distribution of 5 Financial Features", fontsize=14, y=0.98)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_boxplot(data: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    box_data = [data[col].dropna().values for col in FEATURES]
    ax.boxplot(box_data, tick_labels=FEATURES, showfliers=True)
    ax.set_yscale("symlog", linthresh=1)
    ax.axhline(0, linestyle="--", alpha=0.6)
    ax.set_title("Boxplot of 5 Features (Y-axis = symlog)")
    ax.set_ylabel("Value")
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    ensure_utf8_output()
    load_env()
    args = parse_args()
    file_path = resolve_input_file(args.input, DEFAULT_FILE_NAME)

    output_dir = env_resolve_directory(
        args.output_dir,
        DEFAULT_OUTPUT_ENV_VAR,
        Path("src") / "output",
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_excel(file_path)

    required_cols = [c for c in FEATURES if c in df.columns]
    if len(required_cols) != len(FEATURES):
        missing = [c for c in FEATURES if c not in df.columns]
        preview_cols = ", ".join(str(c) for c in list(df.columns)[:10])
        raise ValueError(
            f"Thieu feature bat buoc: {missing}. "
            f"Cac cot dau file hien tai: {preview_cols}. "
            f"Co the file FEATURED_DATA_FILE dang mat header hoac khong phai DATA_FEATURED dung dinh dang."
        )

    if args.exclude_finance and "is_finance" in df.columns:
        before = len(df)
        df = df.loc[~df["is_finance"].fillna(False).astype(bool)].copy()
        print(f"Da loai finance rows: {before - len(df)}")

    for col in FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    data = df[FEATURES].copy()

    print(f"Input file: {file_path.resolve()}")
    print(f"Output folder: {output_dir.resolve()}")
    print(f"So dong du lieu sau loc: {len(df)}")
    if "nam" in df.columns:
        print("So dong theo nam:")
        print(df["nam"].value_counts().sort_index())
    if "San" in df.columns:
        print("So dong theo san:")
        print(df["San"].value_counts(dropna=False))
    if "is_finance" in df.columns:
        print("So dong theo is_finance:")
        print(df["is_finance"].value_counts(dropna=False))
    print()

    desc = build_descriptive_stats(data)
    outlier_df = build_outlier_iqr(data)
    missing_df = summarize_missing(df)

    print("=" * 80)
    print("BANG THONG KE MO TA")
    print("=" * 80)
    print(desc.round(4))
    print()

    print("=" * 80)
    print("SO LUONG OUTLIER THEO IQR")
    print("=" * 80)
    print(outlier_df.round(4))
    print()

    plot_histograms(data, output_dir / "histogram_kde_features.png")
    plot_boxplot(data, output_dir / "boxplot_features.png")

    desc.to_excel(output_dir / "descriptive_statistics.xlsx")
    outlier_df.to_excel(output_dir / "outlier_iqr_summary.xlsx", index=False)
    missing_df.to_excel(output_dir / "missing_summary.xlsx", index=False)

    with pd.ExcelWriter(output_dir / "eda_summary.xlsx", engine="openpyxl") as writer:
        desc.to_excel(writer, sheet_name="descriptive_stats")
        outlier_df.to_excel(writer, sheet_name="iqr_outliers", index=False)
        missing_df.to_excel(writer, sheet_name="missing_summary", index=False)
        if args.by_year and "nam" in df.columns:
            yearly_tables = []
            for year, sub in df.groupby("nam"):
                tmp = build_descriptive_stats(sub[FEATURES].copy())
                tmp.insert(0, "year", year)
                yearly_tables.append(tmp.reset_index(names="feature"))
            if yearly_tables:
                pd.concat(yearly_tables, ignore_index=True).to_excel(writer, sheet_name="descriptive_by_year", index=False)

    print("Da luu file:")
    print(f"- {output_dir / 'histogram_kde_features.png'}")
    print(f"- {output_dir / 'boxplot_features.png'}")
    print(f"- {output_dir / 'descriptive_statistics.xlsx'}")
    print(f"- {output_dir / 'outlier_iqr_summary.xlsx'}")
    print(f"- {output_dir / 'missing_summary.xlsx'}")
    print(f"- {output_dir / 'eda_summary.xlsx'}")


if __name__ == "__main__":
    main()
