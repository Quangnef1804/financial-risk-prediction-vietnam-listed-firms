import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from env_config import get_path, load_env

load_env()


def ensure_utf8_output() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

HNX_PATH = str(get_path("RAW_HNX_FILE", Path("Data") / "raw" / "HNX_MERGED.xlsx"))
HOSE_PATH = str(get_path("RAW_HOSE_FILE", Path("Data") / "raw" / "HOSE_MERGED.xlsx"))

# File feature giữ đủ toàn bộ năm, kể cả năm cuối chưa có target.
FEATURED_OUTPUT_PATH = str(get_path("FEATURED_DATA_FILE", Path("Data") / "train" / "DATA_FEATURED.xlsx"))

# File supervised chỉ giữ các dòng có ROS_next để phục vụ train model.
SUPERVISED_OUTPUT_PATH = str(get_path("SUPERVISED_DATA_FILE", Path("Data") / "train" / "DATA_SUPERVISED.xlsx"))

# Tùy chọn: nếu muốn loại công ty tài chính ra khỏi dataset train, bật flag CLI --exclude-finance.
FINANCE_TICKERS = {
    "ACB", "BID", "CTG", "EIB", "HDB", "LPB", "MBB", "MSB", "NAB", "OCB",
    "SHB", "SSB", "STB", "TCB", "TPB", "VCB", "VIB", "VPB", "BAB", "ABB",
    "BVB", "KLB", "PGB", "SGB", "VAB",
    "SSI", "VND", "HCM", "VCI", "FTS", "CTS", "BSI", "MBS", "SHS", "VIX",
    "ORS", "AGR", "BVS", "TVS", "AAS",
    "BVH", "BMI", "MIG", "PVI", "PGI", "BIC",
}

# Các biến kiểu "stock" có thể forward fill tương đối an toàn theo thời gian.
# Không forward fill các biến kiểu "flow" như doanh thu, lợi nhuận, chi phí...
FLOW_KEYWORDS = [
    "DOANH THU",
    "LỢI NHUẬN",
    "KHẤU HAO",
    "CHI PHÍ",
    "THUẾ",
    "LÃI",
    "LỖ",
    "DÒNG TIỀN",
    "CỔ TỨC",
]


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        pd.Index(df.columns)
        .astype(str)
        .str.replace("\n", " ", regex=False)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    return df


def normalize_text(value: object) -> str:
    if value is None:
        return ""
    text = str(value).replace("\n", " ").strip()
    if not text:
        return ""
    text = text.replace("Đ", "D").replace("đ", "d")
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return " ".join(text.split()).strip().lower()


def detect_header_row(raw_df: pd.DataFrame, search_limit: int = 30) -> int:
    required_tokens = {"stt", "ma", "ten cong ty", "san"}

    for idx in range(min(search_limit, len(raw_df))):
        row_tokens = {normalize_text(value) for value in raw_df.iloc[idx].tolist()}
        row_tokens.discard("")
        if required_tokens.issubset(row_tokens):
            return idx

    raise ValueError(
        "Khong xac dinh duoc dong header trong file raw. "
        "Can tim thay day du cac cot STT, Ma, Ten cong ty, San."
    )


def rename_standard_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map: dict[str, str] = {}

    for col in df.columns:
        normalized = normalize_text(col)
        if normalized == "stt":
            rename_map[col] = "STT"
        elif normalized == "ma":
            rename_map[col] = "Ma"
        elif normalized == "ten cong ty":
            rename_map[col] = "Ten cong ty"
        elif normalized == "san":
            rename_map[col] = "San"

    return df.rename(columns=rename_map)


def load_raw_financial_file(path: str | Path, expected_exchange: str) -> pd.DataFrame:
    path = Path(path)
    raw_preview = pd.read_excel(path, header=None)
    header_row = detect_header_row(raw_preview)

    df = pd.read_excel(path, header=header_row)
    df = clean_columns(df)
    df = rename_standard_columns(df)
    df = df.dropna(axis=1, how="all")
    df = df.dropna(how="all").copy()

    required_cols = {"Ma", "San"}
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"File {path.name} thieu cot bat buoc sau khi doc header: {missing}")

    ma_clean = df["Ma"].where(df["Ma"].notna(), "").astype(str).str.strip()
    san_clean = (
        df["San"]
        .where(df["San"].notna(), "")
        .astype(str)
        .str.strip()
        .str.upper()
        .replace({"HSX": "HOSE"})
    )

    valid_rows = (
        ma_clean.ne("")
        & ma_clean.str.upper().ne("NAN")
        & san_clean.eq(expected_exchange.upper())
    )

    removed_rows = int((~valid_rows).sum())
    df = df.loc[valid_rows].copy()
    df["Ma"] = ma_clean.loc[valid_rows]
    df["San"] = san_clean.loc[valid_rows]

    print(
        f"Doc {path.name}: header_row={header_row + 1}, "
        f"giu {len(df)} dong hop le, loai {removed_rows} dong metadata/footer/trong."
    )
    return df


def extract_indicator_name(column_name: object) -> str:
    text = " ".join(str(column_name).replace("\n", " ").split()).strip()
    for marker in [" Hợp nhất ", " Riêng lẻ ", " Quý:", " Năm:"]:
        if marker in text:
            return text.split(marker, 1)[0].strip()
    return text


def extract_year_from_column_name(column_name: object) -> int | None:
    text = " ".join(str(column_name).replace("\n", " ").split()).strip()
    match = re.search(r"Năm:\s*(\d{4})", text)
    if match:
        return int(match.group(1))
    return None


def resolve_column_name(columns: pd.Index | list[str], target_name: str) -> str:
    normalized_target = normalize_text(target_name)
    normalized_map = {normalize_text(col): str(col) for col in columns}
    if normalized_target not in normalized_map:
        raise KeyError(target_name)
    return normalized_map[normalized_target]


def resolve_existing_columns(columns: pd.Index | list[str], target_names: list[str]) -> list[str]:
    normalized_map = {normalize_text(col): str(col) for col in columns}
    resolved: list[str] = []
    for target_name in target_names:
        normalized_target = normalize_text(target_name)
        if normalized_target in normalized_map:
            resolved.append(normalized_map[normalized_target])
    return resolved


def load_and_standardize(hnx_path: str, hose_path: str) -> pd.DataFrame:
    hnx = load_raw_financial_file(hnx_path, expected_exchange="HNX")
    hose = load_raw_financial_file(hose_path, expected_exchange="HOSE")

    hnx = hnx.drop(
        columns=["STT", "Ten cong ty", "STT_bs", "Ten cong ty_bs", "San_bs"],
        errors="ignore",
    )
    hose = hose.drop(
        columns=["STT", "Ten cong ty", "STT_bs", "Ten cong ty_bs", "San_bs"],
        errors="ignore",
    )

    common_indicators = [c for c in hose.columns if c in hnx.columns or c in {"Ma", "San"}]
    df = pd.concat([hnx, hose[common_indicators]], ignore_index=True)
    return df


def wide_to_panel(df: pd.DataFrame) -> pd.DataFrame:
    value_cols = [c for c in df.columns if extract_year_from_column_name(c) is not None]
    id_cols = [c for c in ["Ma", "San"] if c in df.columns]

    if not value_cols:
        raise ValueError("Không tìm thấy cột năm dạng 'Năm: YYYY' trong raw data.")

    df_long = df.melt(
        id_vars=id_cols,
        value_vars=value_cols,
        var_name="chi_tieu_nam",
        value_name="gia_tri",
    )

    df_long["chi_tieu"] = df_long["chi_tieu_nam"].map(extract_indicator_name)
    df_long["nam"] = df_long["chi_tieu_nam"].map(extract_year_from_column_name).astype(int)

    df_long = df_long.drop(columns=["chi_tieu_nam"])

    df_panel = (
        df_long.pivot_table(
            index=["Ma", "San", "nam"],
            columns="chi_tieu",
            values="gia_tri",
            aggfunc="first",
        )
        .reset_index()
    )
    df_panel.columns.name = None
    return df_panel


def check_duplicates(df_panel: pd.DataFrame) -> None:
    dup_mask = df_panel.duplicated(subset=["Ma", "nam"], keep=False)
    dup_count = int(dup_mask.sum())
    if dup_count > 0:
        sample = (
            df_panel.loc[dup_mask, ["Ma", "nam"]]
            .drop_duplicates()
            .sort_values(["Ma", "nam"])
            .head(10)
            .to_dict(orient="records")
        )
        raise ValueError(
            f"Phát hiện {dup_count} dòng bị trùng Ma-nam sau pivot. Ví dụ: {sample}"
        )


def infer_forward_fill_columns(indicator_cols: list[str]) -> list[str]:
    ffill_cols: list[str] = []
    for col in indicator_cols:
        upper = str(col).upper()
        if any(keyword in upper for keyword in FLOW_KEYWORDS):
            continue
        ffill_cols.append(col)
    return ffill_cols


def handle_missing_values(df_panel: pd.DataFrame) -> pd.DataFrame:
    df_panel = df_panel.copy()
    indicator_cols = [c for c in df_panel.columns if c not in ["Ma", "San", "nam"]]

    for c in indicator_cols:
        df_panel[c] = pd.to_numeric(df_panel[c], errors="coerce")

    # Loại các dòng thiếu quá nhiều chỉ tiêu.
    missing_by_row = df_panel[indicator_cols].isna().sum(axis=1)
    threshold = len(indicator_cols) // 2
    df_panel = df_panel.loc[missing_by_row <= threshold].copy()

    # Chỉ forward fill các biến stock; không fill doanh thu/lợi nhuận.
    ffill_cols = infer_forward_fill_columns(indicator_cols)
    df_panel = df_panel.sort_values(["Ma", "nam"]).copy()
    df_panel[ffill_cols] = (
        df_panel.groupby("Ma", group_keys=False)[ffill_cols]
        .ffill()
    )

    required_cols = [
        "A. TỔNG TÀI SẢN",
        "3. Doanh thu thuần",
        "1. Nợ ngắn hạn",
        "1. Lợi nhuận trước thuế (GT)",
        "I. TỔNG NỢ PHẢI TRẢ",
        "I. TÀI SẢN NGẮN HẠN",
    ]
    required_cols = resolve_existing_columns(df_panel.columns, required_cols)
    df_panel = df_panel.dropna(subset=required_cols).copy()
    return df_panel


def handle_outliers_and_invalids(df_panel: pd.DataFrame, finance_tickers=None) -> pd.DataFrame:
    if finance_tickers is None:
        finance_tickers = set()

    df_panel = df_panel.copy()

    # Doanh thu <= 0 làm ROS, DS, TAT mất ý nghĩa -> loại.
    revenue_col = next(
        iter(resolve_existing_columns(df_panel.columns, ["3. Doanh thu thuần"])),
        None,
    )
    if revenue_col is not None:
        mask_dt_invalid = df_panel[revenue_col] <= 0
        print(f"Loại {int(mask_dt_invalid.sum())} dòng doanh thu <= 0")
        df_panel = df_panel.loc[~mask_dt_invalid].copy()

    # Gắn nhãn công ty tài chính để có thể lọc sau nếu cần.
    df_panel["is_finance"] = df_panel["Ma"].isin(set(finance_tickers))
    return df_panel


def create_features(df_panel: pd.DataFrame) -> pd.DataFrame:
    p = df_panel.copy()

    required_feature_inputs = [
        "I. TÀI SẢN NGẮN HẠN",
        "1. Nợ ngắn hạn",
        "1. Lợi nhuận trước thuế (GT)",
        "3. Doanh thu thuần",
        "I. TỔNG NỢ PHẢI TRẢ",
        "A. TỔNG TÀI SẢN",
    ]
    resolved_inputs = resolve_existing_columns(p.columns, required_feature_inputs)
    missing_inputs = [c for c in required_feature_inputs if normalize_text(c) not in {normalize_text(x) for x in resolved_inputs}]
    if missing_inputs:
        raise ValueError(f"Thiếu cột để tính feature: {missing_inputs}")

    current_assets_col = resolve_column_name(p.columns, "I. TÀI SẢN NGẮN HẠN")
    short_debt_col = resolve_column_name(p.columns, "1. Nợ ngắn hạn")
    pre_tax_profit_col = resolve_column_name(p.columns, "1. Lợi nhuận trước thuế (GT)")
    revenue_col = resolve_column_name(p.columns, "3. Doanh thu thuần")
    total_liabilities_col = resolve_column_name(p.columns, "I. TỔNG NỢ PHẢI TRẢ")
    total_assets_col = resolve_column_name(p.columns, "A. TỔNG TÀI SẢN")

    p["CR"] = np.where(
        p[short_debt_col] != 0,
        p[current_assets_col] / p[short_debt_col],
        np.nan,
    )

    p["ROS"] = np.where(
        p[revenue_col] != 0,
        p[pre_tax_profit_col] / p[revenue_col],
        np.nan,
    )

    p["DS"] = np.where(
        p[revenue_col] != 0,
        p[total_liabilities_col] / p[revenue_col],
        np.nan,
    )

    p["TAT"] = np.where(
        p[total_assets_col] != 0,
        p[revenue_col] / p[total_assets_col],
        np.nan,
    )

    p["SIZE"] = np.where(
        p[total_assets_col] > 0,
        np.log(p[total_assets_col]),
        np.nan,
    )

    p = p.sort_values(["Ma", "nam"]).copy()

    # Chỉ lấy ROS_next khi năm kế tiếp thực sự là nam + 1.
    next_year = p.groupby("Ma")["nam"].shift(-1)
    next_ros = p.groupby("Ma")["ROS"].shift(-1)
    consecutive_next = next_year.eq(p["nam"] + 1)

    p["ROS_next"] = np.where(consecutive_next, next_ros, np.nan)

    # Chỉ gán target khi thực sự có ROS năm sau hợp lệ.
    p["target"] = pd.Series(pd.NA, index=p.index, dtype="Int64")
    has_next = p["ROS_next"].notna()
    p.loc[has_next, "target"] = (p.loc[has_next, "ROS_next"] < 0.05).astype("Int64")

    return p


def reorder_columns(p: pd.DataFrame) -> pd.DataFrame:
    front_cols = ["Ma", "San", "nam", "is_finance", "CR", "ROS", "DS", "TAT", "SIZE", "ROS_next", "target"]
    existing_front_cols = [c for c in front_cols if c in p.columns]
    other_cols = [c for c in p.columns if c not in existing_front_cols]
    return p[existing_front_cols + other_cols].copy()


def build_featured_dataset(df_panel: pd.DataFrame) -> pd.DataFrame:
    featured = create_features(df_panel)

    feature_cols = ["CR", "ROS", "DS", "TAT", "SIZE"]
    featured = featured.dropna(subset=feature_cols).copy()
    featured = reorder_columns(featured)
    return featured


def build_supervised_dataset(featured_df: pd.DataFrame, exclude_finance: bool = False) -> pd.DataFrame:
    supervised = featured_df.copy()

    # Chỉ dataset train mới bỏ các dòng năm cuối chưa có ROS_next hoặc bị gãy chuỗi năm.
    supervised = supervised.dropna(subset=["ROS_next", "target"]).copy()

    if exclude_finance and "is_finance" in supervised.columns:
        before = len(supervised)
        supervised = supervised.loc[~supervised["is_finance"].fillna(False)].copy()
        print(f"Loại {before - len(supervised)} dòng công ty tài chính khỏi DATA_SUPERVISED")

    feature_cols = ["CR", "ROS", "DS", "TAT", "SIZE"]
    supervised = supervised.dropna(subset=feature_cols).copy()
    supervised["target"] = supervised["target"].astype("Int64")

    supervised = reorder_columns(supervised)
    return supervised


def save_dataframe(df: pd.DataFrame, output_path: str) -> None:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    if out.suffix.lower() == ".xlsx":
        df.to_excel(out, index=False)
    else:
        df.to_csv(out, index=False, encoding="utf-8-sig")

    print(f"Đã lưu: {out}")


def print_year_summary(df: pd.DataFrame, label: str) -> None:
    year_counts = df["nam"].value_counts().sort_index().to_dict()
    print(f"{label} theo năm: {year_counts}")


def audit_year_gaps(df: pd.DataFrame) -> dict:
    gap_records = []
    for ma, g in df.sort_values(["Ma", "nam"]).groupby("Ma"):
        years = g["nam"].dropna().astype(int).tolist()
        for current_year, next_year in zip(years, years[1:]):
            if next_year - current_year != 1:
                gap_records.append({
                    "Ma": ma,
                    "current_year": int(current_year),
                    "next_year": int(next_year),
                    "gap": int(next_year - current_year),
                })

    result = {
        "companies_with_non_consecutive_years": len({r["Ma"] for r in gap_records}),
        "non_consecutive_transitions": len(gap_records),
        "examples": gap_records[:10],
    }
    return result


def print_gap_summary(df: pd.DataFrame, label: str) -> None:
    gap_info = audit_year_gaps(df)
    print(
        f"{label} | công ty bị gãy chuỗi năm: {gap_info['companies_with_non_consecutive_years']} | "
        f"số transition không liên tiếp: {gap_info['non_consecutive_transitions']}"
    )
    if gap_info["examples"]:
        print(f"Ví dụ gap: {gap_info['examples'][:5]}")


def build_datasets(
    hnx_path: str,
    hose_path: str,
    featured_output_path: str | None = None,
    supervised_output_path: str | None = None,
    finance_tickers=None,
    exclude_finance: bool = False,
):
    df = load_and_standardize(hnx_path, hose_path)
    print(f"Sau merge: {df.shape[0]} dòng, {df.shape[1]} cột")

    df_panel = wide_to_panel(df)
    check_duplicates(df_panel)
    print(f"Sau reshape panel: {df_panel.shape[0]} dòng, {df_panel.shape[1]} cột")
    print_year_summary(df_panel, "Panel gốc")
    print_gap_summary(df_panel, "Panel gốc")

    df_panel = handle_missing_values(df_panel)
    print(f"Sau xử lý missing: {df_panel.shape[0]} dòng")
    print_year_summary(df_panel, "Panel sau missing")
    print_gap_summary(df_panel, "Panel sau missing")

    df_panel = handle_outliers_and_invalids(df_panel, finance_tickers=finance_tickers)
    print(f"Sau xử lý giá trị bất thường: {df_panel.shape[0]} dòng")
    print_year_summary(df_panel, "Panel sau invalid/outlier")
    print_gap_summary(df_panel, "Panel sau invalid/outlier")

    featured_df = build_featured_dataset(df_panel)
    print(f"DATA_FEATURED: {featured_df.shape[0]} dòng, {featured_df.shape[1]} cột")
    print_year_summary(featured_df, "DATA_FEATURED")

    supervised_df = build_supervised_dataset(featured_df, exclude_finance=exclude_finance)
    print(f"DATA_SUPERVISED: {supervised_df.shape[0]} dòng, {supervised_df.shape[1]} cột")
    print_year_summary(supervised_df, "DATA_SUPERVISED")

    if featured_output_path:
        save_dataframe(featured_df, featured_output_path)
    if supervised_output_path:
        save_dataframe(supervised_df, supervised_output_path)

    return featured_df, supervised_df


def main():
    ensure_utf8_output()
    parser = argparse.ArgumentParser(description="Build DATA_FEATURED và DATA_SUPERVISED từ raw financial files.")
    parser.add_argument("--hnx-path", type=str, default=HNX_PATH)
    parser.add_argument("--hose-path", type=str, default=HOSE_PATH)
    parser.add_argument("--featured-output", type=str, default=FEATURED_OUTPUT_PATH)
    parser.add_argument("--supervised-output", type=str, default=SUPERVISED_OUTPUT_PATH)
    parser.add_argument(
        "--exclude-finance",
        action="store_true",
        help="Loại các dòng is_finance=True khỏi DATA_SUPERVISED.",
    )
    args = parser.parse_args()

    featured_df, supervised_df = build_datasets(
        hnx_path=args.hnx_path,
        hose_path=args.hose_path,
        featured_output_path=args.featured_output,
        supervised_output_path=args.supervised_output,
        finance_tickers=FINANCE_TICKERS,
        exclude_finance=args.exclude_finance,
    )

    print("\nPreview DATA_FEATURED:")
    print(featured_df[["Ma", "San", "nam", "CR", "ROS", "DS", "TAT", "SIZE", "ROS_next", "target"]].head(10))

    print("\nPreview DATA_SUPERVISED:")
    print(supervised_df[["Ma", "San", "nam", "CR", "ROS", "DS", "TAT", "SIZE", "ROS_next", "target"]].head(10))


if __name__ == "__main__":
    main()
