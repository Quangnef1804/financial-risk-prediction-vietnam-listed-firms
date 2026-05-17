from pathlib import Path
import argparse
import itertools
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from statsmodels.stats.outliers_influence import variance_inflation_factor
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


def plot_heatmap(ax, corr_df: pd.DataFrame, title: str, vmin: float = -1, vmax: float = 1):
    im = ax.imshow(corr_df.values, cmap="coolwarm", vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(corr_df.columns)))
    ax.set_xticklabels(corr_df.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr_df.index)))
    ax.set_yticklabels(corr_df.index)
    ax.set_title(title)

    for i in range(corr_df.shape[0]):
        for j in range(corr_df.shape[1]):
            val = corr_df.iat[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=9)

    return im


def draw_pairplot(df: pd.DataFrame, features: list[str], target_col: str, out_path: Path):
    n = len(features)
    fig, axes = plt.subplots(n, n, figsize=(18, 18))
    target_values = sorted(df[target_col].dropna().unique())
    if target_values != [0, 1]:
        raise ValueError("Cot target phai chi chua 0 va 1 de ve pairplot theo class.")

    class_styles = {
        0: {"label": "target=0", "color": "tab:blue"},
        1: {"label": "target=1", "color": "tab:orange"},
    }

    for i, y_col in enumerate(features):
        for j, x_col in enumerate(features):
            ax = axes[i, j]

            if i == j:
                for cls in [0, 1]:
                    s = df.loc[df[target_col] == cls, x_col].dropna()
                    ax.hist(s, bins=20, alpha=0.5, density=False, label=class_styles[cls]["label"], color=class_styles[cls]["color"])
            else:
                for cls in [0, 1]:
                    sub = df.loc[df[target_col] == cls, [x_col, y_col]].dropna()
                    ax.scatter(
                        sub[x_col],
                        sub[y_col],
                        s=10,
                        alpha=0.55,
                        color=class_styles[cls]["color"],
                        label=class_styles[cls]["label"] if (i == 0 and j == 1) else None,
                    )

            if i == n - 1:
                ax.set_xlabel(x_col)
            else:
                ax.set_xticklabels([])

            if j == 0:
                ax.set_ylabel(y_col)
            else:
                ax.set_yticklabels([])

            ax.grid(alpha=0.2)

    handles = [
        Line2D([0], [0], marker='o', linestyle='', label='target=0', markersize=7, color='tab:blue'),
        Line2D([0], [0], marker='o', linestyle='', label='target=1', markersize=7, color='tab:orange')
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=True)
    fig.suptitle("Pairplot of 5 Features Colored by Target", fontsize=16, y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()


def compute_vif(df_features: pd.DataFrame) -> pd.DataFrame:
    clean = df_features.dropna().copy()
    x = clean.values.astype(float)
    vif_rows = []

    for i, col in enumerate(clean.columns):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            vif_val = variance_inflation_factor(x, i)
        vif_rows.append({
            "feature": col,
            "VIF": float(vif_val),
            "warning": "Canh bao: VIF > 5" if vif_val > 5 else ""
        })

    return pd.DataFrame(vif_rows).sort_values("VIF", ascending=False)


def summarize_correlations(corr_df: pd.DataFrame, vif_df: pd.DataFrame) -> str:
    lines = []
    pairs = []

    for a, b in itertools.combinations(corr_df.columns, 2):
        val = corr_df.loc[a, b]
        pairs.append((a, b, val, abs(val)))

    pairs_sorted = sorted(pairs, key=lambda x: x[3], reverse=True)
    high_pairs = [(a, b, v) for a, b, v, av in pairs_sorted if av >= 0.7]
    medium_pairs = [(a, b, v) for a, b, v, av in pairs_sorted if 0.5 <= av < 0.7]

    lines.append("NHAN XET TU DONG")
    lines.append("=" * 80)

    if high_pairs:
        lines.append("1) Cac cap features co tuong quan cao (|r| >= 0.70):")
        for a, b, v in high_pairs:
            lines.append(f"   - {a} vs {b}: r = {v:.3f}")
    else:
        lines.append("1) Khong co cap features nao co tuong quan cao o muc |r| >= 0.70.")

    if medium_pairs:
        lines.append("")
        lines.append("2) Cac cap features co tuong quan trung binh-kha (0.50 <= |r| < 0.70):")
        for a, b, v in medium_pairs:
            lines.append(f"   - {a} vs {b}: r = {v:.3f}")

    lines.append("")
    lines.append("3) Kiem tra da cong tuyen bang VIF:")
    problematic = vif_df[vif_df["VIF"] > 5].copy()
    if problematic.empty:
        lines.append("   - Khong co feature nao co VIF > 5. Chua co dau hieu manh phai loai feature truoc Logistic Regression.")
    else:
        lines.append("   - Co feature VIF > 5, can xem xet them:")
        for _, row in problematic.iterrows():
            lines.append(f"     * {row['feature']}: VIF = {row['VIF']:.3f}")

    lines.append("")
    lines.append("4) Goi y:")
    if problematic.empty and not high_pairs:
        lines.append("   - Co the giu lai toan bo 5 features o buoc baseline.")
    else:
        lines.append("   - Nen xem lai cac cap tuong quan cao va feature co VIF > 5.")
        lines.append("   - Chua nen loai bo chi dua tren tuong quan don thuan; uu tien can nhac theo VIF, nghiep vu va hieu qua mo hinh.")

    return "\n".join(lines)


def main():
    load_env()
    parser = argparse.ArgumentParser(description="Phan tich tuong quan features tren DATA_SUPERVISED.")
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

    required_cols = FEATURES + [TARGET_COL]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Thieu cot bat buoc: {missing_cols}")

    if args.exclude_finance and "is_finance" in df.columns:
        before = len(df)
        df = df.loc[~df["is_finance"].fillna(False).astype(bool)].copy()
        print(f"Da loai finance rows: {before - len(df)}")

    for col in required_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[df[TARGET_COL].isin([0, 1])].copy()
    df[TARGET_COL] = df[TARGET_COL].astype(int)
    df = df.dropna(subset=required_cols).copy()

    feat_df = df[FEATURES].copy()
    pearson_corr = feat_df.corr(method="pearson")
    spearman_corr = feat_df.corr(method="spearman")

    print("=" * 100)
    print("MA TRAN TUONG QUAN PEARSON")
    print("=" * 100)
    print(pearson_corr.round(4))
    print()

    print("=" * 100)
    print("MA TRAN TUONG QUAN SPEARMAN")
    print("=" * 100)
    print(spearman_corr.round(4))
    print()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    plot_heatmap(axes[0], pearson_corr, "Pearson Correlation")
    im2 = plot_heatmap(axes[1], spearman_corr, "Spearman Correlation")
    cbar = fig.colorbar(im2, ax=axes.ravel().tolist(), shrink=0.88)
    cbar.set_label("Correlation coefficient")
    plt.suptitle("Correlation Heatmaps among 5 Features", fontsize=15, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    heatmap_path = output_dir / "correlation_heatmaps.png"
    plt.savefig(heatmap_path, dpi=220, bbox_inches="tight")
    plt.close()

    pairplot_path = output_dir / "pairplot_by_target.png"
    draw_pairplot(df, FEATURES, TARGET_COL, pairplot_path)

    vif_df = compute_vif(feat_df)
    print("=" * 100)
    print("VIF (VARIANCE INFLATION FACTOR)")
    print("=" * 100)
    print(vif_df.round(4))
    print()

    remarks_text = summarize_correlations(pearson_corr, vif_df)
    print(remarks_text)
    print()

    excel_path = output_dir / "correlation_analysis.xlsx"
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        pearson_corr.to_excel(writer, sheet_name="pearson_corr")
        spearman_corr.to_excel(writer, sheet_name="spearman_corr")
        vif_df.to_excel(writer, sheet_name="vif", index=False)

    txt_path = output_dir / "correlation_remarks.txt"
    txt_path.write_text(remarks_text, encoding="utf-8")

    print("Da luu file:")
    print(f"- {heatmap_path}")
    print(f"- {pairplot_path}")
    print(f"- {excel_path}")
    print(f"- {txt_path}")


if __name__ == "__main__":
    main()
