from __future__ import annotations

import argparse
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

from env_config import get_path, load_env


load_env()


HNX_PATH = str(get_path("RAW_HNX_FILE", Path("Data") / "raw" / "HNX_V2.xlsx"))
HOSE_PATH = str(get_path("RAW_HOSE_FILE", Path("Data") / "raw" / "HOSE_V2.xlsx"))
OUTPUT_DIR = Path(get_path("RAW_OVERVIEW_OUTPUT_DIR", Path("Data") / "raw" / "overview"))
VND_TO_BILLION = 1_000_000_000

MAIN_VARIABLES = [
    ("A. TỔNG TÀI SẢN", "Tổng tài sản"),
    ("I. TÀI SẢN NGẮN HẠN", "Tài sản ngắn hạn"),
    ("1. Nợ ngắn hạn", "Nợ ngắn hạn"),
    ("I. TỔNG NỢ PHẢI TRẢ", "Tổng nợ phải trả"),
    ("3. Doanh thu thuần", "Doanh thu thuần"),
    ("1. Lợi nhuận trước thuế (GT)", "Lợi nhuận trước thuế"),
]

ASSETS_INDICATOR = "A. TỔNG TÀI SẢN"
REVENUE_INDICATOR = "3. Doanh thu thuần"


def ensure_utf8_output() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")


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
        row_tokens = {normalize_text(v) for v in raw_df.iloc[idx].tolist()}
        row_tokens.discard("")
        if required_tokens.issubset(row_tokens):
            return idx
    raise ValueError("Không xác định được dòng header trong file raw.")


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
        raise ValueError(f"File {path.name} thiếu cột bắt buộc: {missing}")

    ma_clean = df["Ma"].where(df["Ma"].notna(), "").astype(str).str.strip()
    san_clean = (
        df["San"].where(df["San"].notna(), "")
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
        f"Đọc {path.name}: header_row={header_row + 1}, "
        f"giữ {len(df)} dòng hợp lệ, loại {removed_rows} dòng metadata/footer/trống."
    )
    return df


def extract_indicator_name(column_name: object) -> str:
    text = " ".join(str(column_name).replace("\n", " ").split()).strip()
    text = re.split(r"\s+(Hợp nhất|Riêng lẻ)\s+", text, maxsplit=1)[0].strip()
    text = re.split(r"\s+Quý:\s+", text, maxsplit=1)[0].strip()
    text = re.split(r"\s+Năm:\s+", text, maxsplit=1)[0].strip()
    return text


def extract_year_from_column_name(column_name: object) -> int | None:
    match = re.search(r"Năm:\s*(\d{4})", str(column_name))
    if match:
        return int(match.group(1))
    return None


def resolve_indicator_name(columns: pd.Index | list[str], target_name: str) -> str:
    normalized_target = normalize_text(target_name)
    indicator_map: dict[str, str] = {}
    for col in columns:
        raw_col = str(col)
        indicator_map.setdefault(normalize_text(extract_indicator_name(raw_col)), raw_col)

    if normalized_target not in indicator_map:
        raise KeyError(
            f"Không tìm thấy chỉ tiêu '{target_name}'. "
            f"Cần kiểm tra lại cấu trúc file raw."
        )
    return indicator_map[normalized_target]


def load_and_standardize(hnx_path: str, hose_path: str) -> pd.DataFrame:
    hnx = load_raw_financial_file(hnx_path, expected_exchange="HNX")
    hose = load_raw_financial_file(hose_path, expected_exchange="HOSE")

    all_cols = sorted(set(hnx.columns).union(hose.columns), key=str)
    hnx = hnx.reindex(columns=all_cols)
    hose = hose.reindex(columns=all_cols)
    return pd.concat([hnx, hose], ignore_index=True)


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


def identify_all_missing_firms(df: pd.DataFrame) -> pd.DataFrame:
    value_cols = [c for c in df.columns if extract_year_from_column_name(c) is not None]
    if not value_cols:
        return df.iloc[0:0][["Ma", "San"]].copy()

    mask = df[value_cols].isna().all(axis=1)
    base_cols = [c for c in ["Ma", "San", "Ten cong ty"] if c in df.columns]
    return df.loc[mask, base_cols].copy()


def format_note(label: str, missing_rate: float, min_value_billion: float) -> str:
    notes: list[str] = []
    if label == "Lợi nhuận trước thuế" and missing_rate > 10:
        notes.append(
            f"Tỷ lệ thiếu {missing_rate:.1f}%, cần lưu ý khi diễn giải kết quả."
        )
    if label == "Doanh thu thuần" and min_value_billion < 0:
        notes.append(
            "Có quan sát âm; có thể do điều chỉnh doanh thu, hàng bán bị trả lại, hoặc lỗi dữ liệu nguồn."
        )
    return " ".join(notes)


def build_main_variable_table(df_panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for raw_name, label in MAIN_VARIABLES:
        col = resolve_indicator_name(df_panel.columns, raw_name)
        s = pd.to_numeric(df_panel[col], errors="coerce")
        s_billion = s / VND_TO_BILLION
        missing_rate = float(s.isna().mean() * 100)
        min_value_billion = float(s_billion.min()) if s_billion.notna().any() else np.nan

        rows.append(
            {
                "Biến tài chính": label,
                "Tên chỉ tiêu gốc": raw_name,
                "Đơn vị": "Tỷ đồng",
                "Số quan sát": int(s.notna().sum()),
                "Số obs thiếu": int(s.isna().sum()),
                "Tỷ lệ thiếu (%)": missing_rate,
                "Mean": float(s_billion.mean()) if s_billion.notna().any() else np.nan,
                "Median": float(s_billion.median()) if s_billion.notna().any() else np.nan,
                "Std": float(s_billion.std()) if s_billion.notna().any() else np.nan,
                "Min": min_value_billion,
                "Max": float(s_billion.max()) if s_billion.notna().any() else np.nan,
                "Skewness": float(s_billion.skew()) if s_billion.notna().any() else np.nan,
                "Ghi chú": format_note(label, missing_rate, min_value_billion),
            }
        )
    return pd.DataFrame(rows)


def build_exchange_size_table(df_panel: pd.DataFrame) -> pd.DataFrame:
    assets_col = resolve_indicator_name(df_panel.columns, ASSETS_INDICATOR)
    revenue_col = resolve_indicator_name(df_panel.columns, REVENUE_INDICATOR)
    tmp = df_panel[["Ma", "San", "nam", assets_col, revenue_col]].copy()
    tmp[assets_col] = pd.to_numeric(tmp[assets_col], errors="coerce")
    tmp[revenue_col] = pd.to_numeric(tmp[revenue_col], errors="coerce")
    tmp = tmp.dropna(subset=[assets_col]).copy()
    tmp["tong_tai_san_ty"] = tmp[assets_col] / VND_TO_BILLION
    tmp["doanh_thu_thuan_ty"] = tmp[revenue_col] / VND_TO_BILLION
    tmp["log_tong_tai_san"] = np.where(tmp[assets_col] > 0, np.log(tmp[assets_col]), np.nan)

    summary = (
        tmp.groupby("San")
        .agg(
            so_quan_sat=("Ma", "size"),
            so_doanh_nghiep=("Ma", "nunique"),
            nam_min=("nam", "min"),
            nam_max=("nam", "max"),
            tong_tai_san_tb=("tong_tai_san_ty", "mean"),
            tong_tai_san_median=("tong_tai_san_ty", "median"),
            doanh_thu_thuan_tb=("doanh_thu_thuan_ty", "mean"),
            doanh_thu_thuan_median=("doanh_thu_thuan_ty", "median"),
            log_tong_tai_san_tb=("log_tong_tai_san", "mean"),
        )
        .reset_index()
    )

    return summary.rename(
        columns={
            "San": "Sàn",
            "don_vi": "Đơn vị",
            "so_quan_sat": "Số quan sát",
            "so_doanh_nghiep": "Số doanh nghiệp",
            "nam_min": "Năm bắt đầu",
            "nam_max": "Năm kết thúc",
            "tong_tai_san_tb": "Tổng tài sản TB",
            "tong_tai_san_median": "Tổng tài sản Median",
            "doanh_thu_thuan_tb": "Doanh thu thuần TB",
            "doanh_thu_thuan_median": "Doanh thu thuần Median",
            "log_tong_tai_san_tb": "Log(Tổng tài sản) TB",
        }
    ).assign(**{"Đơn vị": "Tỷ đồng"})[
        [
            "Sàn",
            "Đơn vị",
            "Số quan sát",
            "Số doanh nghiệp",
            "Năm bắt đầu",
            "Năm kết thúc",
            "Tổng tài sản TB",
            "Tổng tài sản Median",
            "Doanh thu thuần TB",
            "Doanh thu thuần Median",
            "Log(Tổng tài sản) TB",
        ]
    ]


def main() -> None:
    ensure_utf8_output()

    parser = argparse.ArgumentParser(
        description=(
            "Tạo 2 bảng overview từ HNX_V2 và HOSE_V2: "
            "thống kê mô tả biến tài chính chính và so sánh quy mô doanh nghiệp theo sàn."
        )
    )
    parser.add_argument("--hnx-path", type=str, default=HNX_PATH)
    parser.add_argument("--hose-path", type=str, default=HOSE_PATH)
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_and_standardize(args.hnx_path, args.hose_path)
    all_missing_firms = identify_all_missing_firms(df)
    df_panel = wide_to_panel(df)

    print(
        f"\nRaw input: {len(df)} dòng doanh nghiệp | {df['Ma'].nunique()} mã doanh nghiệp từ 2 file raw"
    )
    print(f"Raw panel: {len(df_panel)} dòng | {df_panel['Ma'].nunique()} doanh nghiệp")
    print(f"Số năm: {sorted(df_panel['nam'].dropna().astype(int).unique().tolist())}")
    if not all_missing_firms.empty:
        print(
            f"Có {len(all_missing_firms)} doanh nghiệp bị loại khỏi panel vì thiếu toàn bộ giá trị tài chính."
        )

    main_table = build_main_variable_table(df_panel)
    size_table = build_exchange_size_table(df_panel)

    excel_path = output_dir / "raw_data_overview.xlsx"
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        main_table.to_excel(writer, sheet_name="main_financial_variables", index=False)
        size_table.to_excel(writer, sheet_name="exchange_size_comparison", index=False)

    main_table.to_csv(output_dir / "main_financial_variables.csv", index=False, encoding="utf-8-sig")
    size_table.to_csv(output_dir / "exchange_size_comparison.csv", index=False, encoding="utf-8-sig")

    print("\nBẢNG 1 - THỐNG KÊ MÔ TẢ CÁC BIẾN TÀI CHÍNH CHÍNH (GỘP HOSE VÀ HNX)")
    print(main_table.round(4).to_string(index=False))

    print("\nBẢNG 2 - SO SÁNH QUY MÔ DOANH NGHIỆP THEO SÀN GIAO DỊCH")
    print(size_table.round(4).to_string(index=False))

    print(f"\nĐã lưu file tại: {output_dir}")
    print(f"- {excel_path}")
    print(f"- {output_dir / 'main_financial_variables.csv'}")
    print(f"- {output_dir / 'exchange_size_comparison.csv'}")


if __name__ == "__main__":
    main()
