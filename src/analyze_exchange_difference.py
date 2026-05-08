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
EXCHANGE_COL = "San"
DEFAULT_FILE_NAME = "DATA_SUPERVISED.xlsx"
DEFAULT_INPUT_ENV_VAR = "SUPERVISED_DATA_FILE"
DEFAULT_OUTPUT_ENV_VAR = "ANALYSIS_OUTPUT_DIR"


def resolve_input_file(user_input: str | None, default_file_name: str = DEFAULT_FILE_NAME) -> Path:
    return env_resolve_input_file(
        user_input,
        DEFAULT_INPUT_ENV_VAR,
        Path("Data") / "train" / default_file_name,
    )


def significance_label(p: float, alpha: float = 0.05) -> str:
    return "Co y nghia thong ke" if p < alpha else "Khong co y nghia thong ke"


def clean_exchange_series(s: pd.Series) -> pd.Series:
    cleaned = s.astype(str).str.strip().str.upper()
    return cleaned.replace({"HSX": "HOSE", "HOSE ": "HOSE", "HNX ": "HNX", "NAN": np.nan, "NONE": np.nan})


def rank_biserial_from_u(u_stat: float, n1: int, n2: int) -> float:
    return (2.0 * u_stat) / (n1 * n2) - 1.0


def build_summary_tables(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    counts = df.groupby(EXCHANGE_COL).size().rename("so_dong")
    target_rate = df.groupby(EXCHANGE_COL)[TARGET_COL].mean().mul(100).rename("ty_le_target_1_pct")
    overall_summary = pd.concat([counts, target_rate], axis=1).reset_index()

    mean_std = df.groupby(EXCHANGE_COL)[FEATURES].agg(["mean", "std"])
    mean_std.columns = [f"{col}_{stat}" for col, stat in mean_std.columns]
    mean_std = mean_std.reset_index()
    return overall_summary, mean_std


def plot_boxplots_by_exchange(df: pd.DataFrame, output_path: Path):
    exchanges = ["HNX", "HOSE"]
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    axes = axes.flatten()

    for i, feature in enumerate(FEATURES):
        ax = axes[i]
        hnx = df.loc[df[EXCHANGE_COL] == "HNX", feature].dropna().values
        hose = df.loc[df[EXCHANGE_COL] == "HOSE", feature].dropna().values

        bp = ax.boxplot([hnx, hose], tick_labels=exchanges, patch_artist=True, showfliers=True)
        facecolors = ["tab:blue", "tab:orange"]
        for patch, color in zip(bp["boxes"], facecolors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        ax.set_title(f"Boxplot theo san: {feature}")
        ax.set_ylabel(feature)
        ax.grid(alpha=0.25)
        if feature == "ROS":
            ax.axhline(0, linestyle="--", linewidth=1)

    for j in range(len(FEATURES), len(axes)):
        axes[j].axis("off")

    plt.suptitle("So sanh phan phoi features giua HNX va HOSE", fontsize=14, y=0.98)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()


def mannwhitney_by_exchange(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature in FEATURES:
        hnx = df.loc[df[EXCHANGE_COL] == "HNX", feature].dropna().values
        hose = df.loc[df[EXCHANGE_COL] == "HOSE", feature].dropna().values

        if len(hnx) == 0 or len(hose) == 0:
            raise ValueError(f"Feature {feature} khong du du lieu o mot trong hai san.")

        u_stat, p_value = mannwhitneyu(hnx, hose, alternative="two-sided")
        rbc = rank_biserial_from_u(u_stat, len(hnx), len(hose))

        rows.append({
            "feature": feature,
            "n_HNX": len(hnx),
            "n_HOSE": len(hose),
            "mean_HNX": np.mean(hnx),
            "std_HNX": np.std(hnx, ddof=1) if len(hnx) > 1 else np.nan,
            "mean_HOSE": np.mean(hose),
            "std_HOSE": np.std(hose, ddof=1) if len(hose) > 1 else np.nan,
            "median_HNX": np.median(hnx),
            "median_HOSE": np.median(hose),
            "U_statistic": u_stat,
            "p_value": p_value,
            "rank_biserial_corr": rbc,
            "conclusion": significance_label(p_value),
        })
    return pd.DataFrame(rows)


def plot_stacked_target_bar(df: pd.DataFrame, output_path: Path):
    target_counts = (
        df.groupby([EXCHANGE_COL, TARGET_COL]).size().unstack(fill_value=0)
        .reindex(index=["HNX", "HOSE"], columns=[0, 1], fill_value=0)
    )
    target_ratio = target_counts.div(target_counts.sum(axis=1), axis=0)

    ax = target_ratio.plot(kind="bar", stacked=True, figsize=(8, 5))
    ax.set_title("Ty le target = 0 / 1 theo san")
    ax.set_xlabel("San")
    ax.set_ylabel("Ty le")
    ax.legend(title="target")
    ax.grid(axis="y", alpha=0.25)

    for i, exchange in enumerate(target_ratio.index):
        cum = 0.0
        for cls in target_ratio.columns:
            val = target_ratio.loc[exchange, cls]
            if val > 0:
                ax.text(i, cum + val / 2, f"{val:.1%}", ha="center", va="center")
                cum += val

    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()


def build_conclusion_text(overall_summary: pd.DataFrame, mw_df: pd.DataFrame) -> str:
    lines = ["KET LUAN TU DONG", "=" * 80]
    if {"HNX", "HOSE"}.issubset(set(overall_summary[EXCHANGE_COL])):
        hnx_rate = overall_summary.loc[overall_summary[EXCHANGE_COL] == "HNX", "ty_le_target_1_pct"].iloc[0]
        hose_rate = overall_summary.loc[overall_summary[EXCHANGE_COL] == "HOSE", "ty_le_target_1_pct"].iloc[0]
        lines.append(f"1) Ty le target = 1: HNX = {hnx_rate:.2f}%, HOSE = {hose_rate:.2f}%.")

    sig_df = mw_df[mw_df["p_value"] < 0.05].copy().sort_values("p_value")

    if sig_df.empty:
        lines.append("2) Khong co feature nao khac biet co y nghia thong ke giua HNX va HOSE o muc alpha = 0.05.")
        lines.append("3) Chua co bang chung manh rang can tach mo hinh rieng theo san.")
    else:
        lines.append("2) Cac feature khac biet co y nghia thong ke giua HNX va HOSE:")
        for _, row in sig_df.iterrows():
            direction = "HNX > HOSE" if row["mean_HNX"] > row["mean_HOSE"] else "HOSE > HNX"
            lines.append(
                f"   - {row['feature']}: p={row['p_value']:.4g}, mean_HNX={row['mean_HNX']:.4f}, mean_HOSE={row['mean_HOSE']:.4f} ({direction})"
            )
        lines.append(f"3) Co {sig_df['feature'].nunique()} / {len(FEATURES)} feature khac biet theo san.")
        lines.append("4) Goi y: co the them bien San vao model baseline truoc khi can nhac tach mo hinh rieng.")

    return "\n".join(lines)


def main():
    load_env()
    parser = argparse.ArgumentParser(description="So sanh HNX va HOSE tren DATA_SUPERVISED.")
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
    df = pd.read_excel(file_path)

    required_cols = FEATURES + [TARGET_COL, EXCHANGE_COL]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Thieu cot bat buoc: {missing_cols}")

    if args.exclude_finance and "is_finance" in df.columns:
        before = len(df)
        df = df.loc[~df["is_finance"].fillna(False).astype(bool)].copy()
        print(f"Da loai finance rows: {before - len(df)}")

    for col in FEATURES + [TARGET_COL]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df[EXCHANGE_COL] = clean_exchange_series(df[EXCHANGE_COL])
    df = df[df[EXCHANGE_COL].isin(["HNX", "HOSE"])].copy()
    df = df[df[TARGET_COL].isin([0, 1])].copy()
    df[TARGET_COL] = df[TARGET_COL].astype(int)
    df = df.dropna(subset=required_cols).copy()

    print(f"So dong dung de phan tich: {len(df)}")
    if "nam" in df.columns:
        print("So dong theo nam:")
        print(df["nam"].value_counts().sort_index())
    print()

    overall_summary, mean_std_table = build_summary_tables(df)
    mw_df = mannwhitney_by_exchange(df)

    plot_boxplots_by_exchange(df, output_dir / "boxplot_hnx_vs_hose.png")
    plot_stacked_target_bar(df, output_dir / "stacked_target_by_exchange.png")

    conclusion_text = build_conclusion_text(overall_summary, mw_df)
    print(conclusion_text)
    print()

    summary_table = mw_df[["feature", "mean_HNX", "std_HNX", "mean_HOSE", "std_HOSE", "p_value", "conclusion"]].copy()

    excel_path = output_dir / "hnx_hose_comparison.xlsx"
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        overall_summary.to_excel(writer, sheet_name="overall_summary", index=False)
        mean_std_table.to_excel(writer, sheet_name="mean_std_by_exchange", index=False)
        mw_df.to_excel(writer, sheet_name="mannwhitney", index=False)
        summary_table.to_excel(writer, sheet_name="summary_table", index=False)

    txt_path = output_dir / "hnx_hose_conclusion.txt"
    txt_path.write_text(conclusion_text, encoding="utf-8")

    print("Da luu file:")
    print(f"- {output_dir / 'boxplot_hnx_vs_hose.png'}")
    print(f"- {output_dir / 'stacked_target_by_exchange.png'}")
    print(f"- {excel_path}")
    print(f"- {txt_path}")


if __name__ == "__main__":
    main()
