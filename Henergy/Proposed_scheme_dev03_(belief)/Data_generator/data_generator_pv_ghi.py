# add_time_features_only.py
import os
import argparse
import numpy as np
import pandas as pd
from typing import Optional

def read_excel_as_df(path: str, sheet_name: Optional[str] = None) -> pd.DataFrame:
    obj = pd.read_excel(path, sheet_name=sheet_name)
    if isinstance(obj, pd.DataFrame):
        return obj
    # 여러 시트일 때: sheet_name이 있으면 우선, 없으면 첫 시트
    if sheet_name and sheet_name in obj:
        return obj[sheet_name]
    first = next(iter(obj.keys()))
    return obj[first]

def add_time_features_no_touch(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
) -> pd.DataFrame:
    """
    기존 df의 데이터는 그대로 두고, timestamp를 이용해
    tod_sin, tod_cos, doy_sin, doy_cos 4개 컬럼만 추가해서 반환.
    """
    if timestamp_col not in df.columns:
        raise ValueError(f"'{timestamp_col}' 컬럼을 찾을 수 없습니다.")

    # 원본은 그대로 두고, 새 컬럼만 추가
    out = df.copy()

    # timestamp를 읽기만 하고 df의 원본 값은 변경하지 않음
    ts = pd.to_datetime(df[timestamp_col], errors="coerce")

    # Time-of-day (0~1)
    tod = (ts.dt.hour * 60 + ts.dt.minute) / (24 * 60)
    # Day-of-year (0~1) — 간단히 365로 스케일
    doy = ts.dt.dayofyear / 365.0

    out["tod_sin"] = np.sin(2 * np.pi * tod)
    out["tod_cos"] = np.cos(2 * np.pi * tod)
    out["doy_sin"] = np.sin(2 * np.pi * doy)
    out["doy_cos"] = np.cos(2 * np.pi * doy)
    return out

def parse_args():
    ap = argparse.ArgumentParser(description="기존 엑셀의 데이터는 변경하지 않고 시간 임베딩 컬럼만 추가")
    ap.add_argument("--excel_path", type=str, default="./Project/Henergy/Proposed_scheme_dev01/Data_generator/Data_input/generated_1hour_data_pv_ghi_clearghi.xlsx",
                    help="입력할 엑셀 경로(.xlsx)")
    ap.add_argument("--sheet_name", type=str, default=None, help="시트명 (없으면 첫 시트)")
    ap.add_argument("--timestamp_col", type=str, default="timestamp", help="타임스탬프 컬럼명")
    ap.add_argument("--pv_col", type=str, default="pv", help="PV 컬럼명(변경하지 않음, 존재 확인용)")
    ap.add_argument("--ghi_col", type=str, default="ghi", help="GHI 컬럼명(변경하지 않음, 존재 확인용)")
    ap.add_argument("--clear_ghi_col", type=str, default="clear_ghi", help="Clear-sky GHI 컬럼명(변경하지 않음, 존재 확인용)")
    ap.add_argument("--out_path", type=str, default="./Project/Henergy/Proposed_scheme_dev01/Data_generator/Data_output/generated_1hour_data_pv_ghi_clearghi_output.xlsx",
                    help="저장할 출력 엑셀 경로(.xlsx)")
    ap.add_argument("--out_sheet", type=str, default="data_with_time_feats", help="출력 시트명")
    return ap.parse_args()

def main():
    args = parse_args()

    if not os.path.exists(args.excel_path):
        raise FileNotFoundError(args.excel_path)

    df = read_excel_as_df(args.excel_path, sheet_name=args.sheet_name)

    # 필수 4컬럼이 존재하는지만 체크 (값/타입/순서는 손대지 않음)
    for c in [args.timestamp_col, args.pv_col, args.ghi_col, args.clear_ghi_col]:
        if c not in df.columns:
            raise ValueError(f"필수 컬럼이 없습니다: '{c}'")

    # 새 컬럼만 추가
    out = add_time_features_no_touch(df, timestamp_col=args.timestamp_col)

    # 저장
    os.makedirs(os.path.dirname(args.out_path) or ".", exist_ok=True)
    with pd.ExcelWriter(args.out_path, engine="openpyxl") as writer:
        out.to_excel(writer, index=False, sheet_name=args.out_sheet)

    print("완료:", os.path.abspath(args.out_path))
    print("추가된 컬럼 ->", ["tod_sin", "tod_cos", "doy_sin", "doy_cos"])

if __name__ == "__main__":
    main()
