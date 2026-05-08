from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
from env_config import (
    load_env,
    resolve_directory as env_resolve_directory,
    resolve_input_file as env_resolve_input_file,
)

FEATURES = ["CR", "ROS", "DS", "TAT", "SIZE"]
TARGET_COL = "target"
DEFAULT_FILE_NAME = "DATA_SUPERVISED.xlsx"
DEFAULT_INPUT_ENV_VAR = "SUPERVISED_DATA_FILE"
DEFAULT_OUTPUT_ENV_VAR = "ANALYSIS_OUTPUT_DIR"


def resolve_input_file(user_input: str | None, default_file_name: str = DEFAULT_FILE_NAME) -> Path:
    return env_resolve_input_file(
        user_input,
        DEFAULT_INPUT_ENV_VAR,
        Path("Data") / "train" / default_file_name,
    )


def rank_biserial_from_u(u_stat: float, n1: int, n0: int) -> float:
    return (2.0 * u_stat) / (n1 * n0) - 1.0


def significance_label(p: float, alpha: float = 0.05) -> str:
    return "Co y nghia thong ke" if p < alpha else "Khong co y nghia thong ke"


def main():
    load_env()
    parser = argparse.ArgumentParser(description="So sanh 5 features theo target tren DATA_SUPERVISED.")
    parser.add_argument("--input", type=str, default=None, help="Duong dan DATA_SUPERVISED.xlsx")
    parser.add_argument("--output-dir", type=str, default=None, help="Thu muc output")
    parser.add_argument("--exclude-finance", action="store_true", help="Loai is_finance=True")
    args = parser.parse_args()

    file_path = resolve_input_file(args.input, DEFAULT_FILE_NAME)
    output_dir = env_resolve_directory(
        args.output_dir,
        DEFAULT_OUTPUT_ENV_VAR,
        Path("src") / "output",
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Dang doc file: {file_path}")
    print(f"Thu muc output: {output_dir}")

    df = pd.read_excel(file_path)

    required_cols = FEATURES + [TARGET_COL]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Thieu cot bat buoc: {missing_cols}")

    if args.exclude_finance and "is_finance" in df.columns:
        before = len(df)
        df = df.loc[~df["is_finance"].fillna(False).astype(bool)].copy()
        print(f"Da loai finance rows: {before - len(df)}")

    for col in FEATURES + [TARGET_COL]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[df[TARGET_COL].isin([0, 1])].copy()
    df[TARGET_COL] = df[TARGET_COL].astype(int)
    df = df.dropna(subset=FEATURES + [TARGET_COL]).copy()

    print(f"So dong dung de phan tich: {len(df)}")
    print(df[TARGET_COL].value_counts().sort_index())
    if "nam" in df.columns:
        print("So dong theo nam:")
        print(df["nam"].value_counts().sort_index())
    print()

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    axes = axes.flatten()

    for i, feature in enumerate(FEATURES):
        ax = axes[i]
        x0 = df.loc[df[TARGET_COL] == 0, feature].dropna().values
        x1 = df.loc[df[TARGET_COL] == 1, feature].dropna().values

        ax.boxplot([x0, x1], tick_labels=["target=0", "target=1"], showfliers=True)
        ax.set_title(f"Boxplot: {feature}")
        ax.set_ylabel(feature)
        ax.grid(alpha=0.25)

    for j in range(len(FEATURES), len(axes)):
        axes[j].axis("off")

    plt.suptitle("So sanh boxplot theo target", fontsize=14, y=0.98)
    plt.tight_layout()
    boxplot_path = output_dir / "boxplot_by_target.png"
    plt.savefig(boxplot_path, bbox_inches="tight", dpi=200)
    plt.close()

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    axes = axes.flatten()

    for i, feature in enumerate(FEATURES):
        ax = axes[i]
        x0 = df.loc[df[TARGET_COL] == 0, feature].dropna().values
        x1 = df.loc[df[TARGET_COL] == 1, feature].dropna().values

        ax.violinplot([x0, x1], positions=[1, 2], showmeans=True, showmedians=True, showextrema=True)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["target=0", "target=1"])
        ax.set_title(f"Violin plot: {feature}")
        ax.set_ylabel(feature)
        ax.grid(alpha=0.25)

    for j in range(len(FEATURES), len(axes)):
        axes[j].axis("off")

    plt.suptitle("So sanh violin plot theo target", fontsize=14, y=0.98)
    plt.tight_layout()
    violin_path = output_dir / "violin_by_target.png"
    plt.savefig(violin_path, bbox_inches="tight", dpi=200)
    plt.close()

    results = []
    for feature in FEATURES:
        class0 = df.loc[df[TARGET_COL] == 0, feature].dropna().values
        class1 = df.loc[df[TARGET_COL] == 1, feature].dropna().values

        n0 = len(class0)
        n1 = len(class1)
        if n0 == 0 or n1 == 0:
            raise ValueError(f"Feature {feature} khong du du lieu cho mot trong hai nhom.")

        u_stat, p_value = mannwhitneyu(class1, class0, alternative="two-sided")
        rbc = rank_biserial_from_u(u_stat, n1, n0)

        results.append({
            "feature": feature,
            "n_class_0": n0,
            "n_class_1": n1,
            "mean_class_0": np.mean(class0),
            "mean_class_1": np.mean(class1),
            "median_class_0": np.median(class0),
            "median_class_1": np.median(class1),
            "U_statistic": u_stat,
            "p_value": p_value,
            "rank_biserial_corr": rbc,
            "conclusion": significance_label(p_value),
        })

    results_df = pd.DataFrame(results)
    effect_df = results_df.copy()
    effect_df["abs_rbc"] = effect_df["rank_biserial_corr"].abs()
    effect_df = effect_df.sort_values("abs_rbc", ascending=False)

    plt.figure(figsize=(10, 5))
    bars = plt.bar(effect_df["feature"], effect_df["rank_biserial_corr"])
    plt.axhline(0, linewidth=1)
    plt.title("Effect size theo Rank-Biserial Correlation")
    plt.ylabel("Rank-biserial correlation")
    plt.xlabel("Feature")
    plt.grid(axis="y", alpha=0.25)

    for bar, val in zip(bars, effect_df["rank_biserial_corr"]):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            val,
            f"{val:.3f}",
            ha="center",
            va="bottom" if val >= 0 else "top",
        )

    effect_path = output_dir / "effect_size_rbc.png"
    plt.tight_layout()
    plt.savefig(effect_path, bbox_inches="tight", dpi=200)
    plt.close()

    summary_df = results_df[["feature", "mean_class_0", "mean_class_1", "p_value", "conclusion"]].copy()

    print("=" * 100)
    print("KET QUA MANN-WHITNEY U TEST")
    print("=" * 100)
    print(results_df[["feature", "U_statistic", "p_value", "rank_biserial_corr", "conclusion"]].round(6))
    print()

    excel_path = output_dir / "comparison_summary.xlsx"
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        results_df.to_excel(writer, sheet_name="detailed_results", index=False)
        summary_df.to_excel(writer, sheet_name="summary_table", index=False)

    print("Da luu file:")
    print(f"- {boxplot_path}")
    print(f"- {violin_path}")
    print(f"- {effect_path}")
    print(f"- {excel_path}")


if __name__ == "__main__":
    main()
