# data_generator.py
import os
from typing import Optional, Union
import numpy as np
import pandas as pd
import argparse  

def resample_impute(df: pd.DataFrame, ts_col: str, freq_minutes: int) -> pd.DataFrame:
    out = df.copy()
    out[ts_col] = pd.to_datetime(out[ts_col])
    out = out.sort_values(ts_col).set_index(ts_col)
    rule = f"{freq_minutes}T"
    out = out.resample(rule).mean()
    out = out.interpolate(limit_direction="both")
    out = out.reset_index()
    return out

def _find_col(candidates, columns):
    cols_lower = {c.lower(): c for c in columns}
    for name in candidates:
        if name.lower() in cols_lower:
            return cols_lower[name.lower()]
    return None

def load_pv_15min(excel_path: Optional[str], sheet_name: Optional[str], freq_minutes: int = 15) -> pd.DataFrame:
    """
    15분 PV 총발전량 시계열 로드(또는 모의 데이터 생성).
    반환 컬럼: ['timestamp','pv_total']
    """
    if excel_path and os.path.exists(excel_path):
        try:
            df = pd.read_excel(excel_path, sheet_name=sheet_name)
            ts_col = _find_col(["timestamp","time","date","datetime"], df.columns)
            pv_col = _find_col(["pv_total","sum_energy","pv","generation","energy"], df.columns)
            if ts_col is None or pv_col is None:
                raise ValueError("엑셀에 timestamp/pv_total 열을 찾을 수 없습니다.")
            df = df[[ts_col, pv_col]].rename(columns={ts_col:"timestamp", pv_col:"pv_total"})
            df = resample_impute(df, "timestamp", freq_minutes)
            return df[["timestamp","pv_total"]]
        except Exception as e:
            print(f"[경고] 엑셀 로드 실패: {e} → 모의 PV 데이터 생성으로 대체합니다.")

    print("[정보] 모의 pv 15분 데이터를 생성합니다.")
    periods = 96 * 30  # 30일
    idx = pd.date_range("2025-07-09", periods=periods, freq=f"{freq_minutes}T")

    tod = (idx.hour * 60 + idx.minute) / (24 * 60)           # [0,1)
    irradiance = np.clip(np.sin(np.pi * tod), 0, 1)          # 밤 0, 정오 1
    clouds = np.clip(np.random.beta(2, 5, len(idx)) * 0.7, 0, 1)  # 0~1
    pv = 3000 * irradiance * (1 - 0.6 * clouds)
    pv += np.random.normal(0, 40, len(idx))
    pv = np.clip(pv, 0, None)

    return pd.DataFrame({"timestamp": idx, "pv_total": pv})

def make_synthetic_weather(index: Union[pd.Series, pd.DatetimeIndex, np.ndarray, list]) -> pd.DataFrame:
    """
    15분 간격 임의 기상 생성.
    어떤 형태로 들어와도 DatetimeIndex 로 캐스팅 후 계산.
    """
    idx = pd.DatetimeIndex(index)
    n = len(idx)

    tod = (idx.hour * 60 + idx.minute) / (24 * 60)  # [0,1)
    doy = idx.dayofyear

    ghi   = np.clip(np.sin(np.pi * tod), 0, 1)

    cloud = np.zeros(n, dtype=np.float32)
    for i in range(n):
        eps = np.random.normal(0, 0.08)
        prev = cloud[i-1] if i else 0.3
        cloud[i] = np.float32(np.clip(0.85 * prev + 0.15 * np.random.rand() + eps, 0, 1))

    temp = (12
            + 10 * np.sin(2 * np.pi * (doy / 365.0))
            + 7 * np.sin(2 * np.pi * tod)
            + np.random.normal(0, 1.2, n))

    wind = np.clip(2.0 + 2.0 * (1 - ghi) + np.random.normal(0, 0.6, n), 0, None)

    return pd.DataFrame({
        "ghi_sim":  ghi.astype(np.float32),
        "temp_sim": temp.astype(np.float32),
        "wind_sim": wind.astype(np.float32),
        "cloud_sim": cloud.astype(np.float32),
    })

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    ts = pd.to_datetime(df["timestamp"])
    tod = (ts.dt.hour * 60 + ts.dt.minute) / (24 * 60)
    doy = ts.dt.dayofyear / 365.0
    df["tod_sin"] = np.sin(2 * np.pi * tod)
    df["tod_cos"] = np.cos(2 * np.pi * tod)
    df["doy_sin"] = np.sin(2 * np.pi * doy)
    df["doy_cos"] = np.cos(2 * np.pi * doy)
    return df

# ==============================
# main: .xlsx 출력 유틸
# ==============================
def parse_args():
    ap = argparse.ArgumentParser(description="15분 PV + 임의 기상 + 시간피처를 생성/정리하여 .xlsx로 저장")
    ap.add_argument("--excel_path", type=str, default="./Project/Henergy/Proposed_scheme/Data_generator/Data_input/sum_energy_15min.xlsx",
                    help="입력할 엑셀 경로(.xlsx)")
    ap.add_argument("--sheet_name", type=str, default=None, help="엑셀 시트명")
    ap.add_argument("--freq_minutes", type=int, default=15, help="리샘플 기준 분 단위(기본 15)")
    ap.add_argument("--out_path", type=str, default="./Project/Henergy/Proposed_scheme/Data_generator/Data_output/generated_15min_data.xlsx",
                    help="저장할 출력 엑셀 경로(.xlsx)")
    ap.add_argument("--out_sheet", type=str, default="data15m",
                    help="출력 시트명 (병합 데이터)")
    ap.add_argument("--separate_sheets", action="store_true",
                    help="PV/Weather/Combined를 각각 시트로 저장")
    return ap.parse_args()

def main():
    args = parse_args()

    # 1) PV 로드(또는 생성) — timestamp/PV 15분 정렬
    pv_df = load_pv_15min(args.excel_path, args.sheet_name, args.freq_minutes)

    # 2) 임의 기상 생성 (동일 타임스탬프)
    wx_df = make_synthetic_weather(pd.to_datetime(pv_df["timestamp"]))

    # 3) 병합 + 시간 피처
    merged = pd.concat([pv_df.reset_index(drop=True), wx_df.reset_index(drop=True)], axis=1)
    merged = add_time_features(merged)

    # 4) 엑셀 저장
    os.makedirs(os.path.dirname(args.out_path) or ".", exist_ok=True)
    with pd.ExcelWriter(args.out_path, engine="openpyxl") as writer:
        if args.separate_sheets:
            pv_df.to_excel(writer, index=False, sheet_name="pv")
            wx_df.to_excel(writer, index=False, sheet_name="weather")
            merged.to_excel(writer, index=False, sheet_name=args.out_sheet)
        else:
            merged.to_excel(writer, index=False, sheet_name=args.out_sheet)

    # 5) 요약 출력
    ts = pd.to_datetime(merged["timestamp"])
    print("[완료] 저장 경로:", os.path.abspath(args.out_path))
    print(f" - 행 수: {len(merged)}  (기간: {ts.min()}  ~  {ts.max()})")
    print(f" - 컬럼: {list(merged.columns)}")

if __name__ == "__main__":
    main()