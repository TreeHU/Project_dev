import re
import pandas as pd
import numpy as np

# ------------------------
# 설정
# ------------------------


EXCEL_PATH =  "./Project/Henergy/EM/Data/pv_gens_weathers_safe_pv_id.xlsx" 
SHEET_NAME = "pv_gens_weathers"   # 시트명/인덱스
START = pd.Timestamp("2025-07-09 00:00", tz="Asia/Seoul")
END   = pd.Timestamp("2025-08-18 00:00", tz="Asia/Seoul")
ENERGY_IS_POWER_KW = False  # True면 값이 전력(kW) -> kWh로 적분

# ------------------------
# 유틸: 컬럼 자동 탐지
# ------------------------
hdr = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME, nrows=0)
cols = [str(c).strip() for c in hdr.columns]

def find_col(cols, pats, required=False):
    for c in cols:
        k = c.lower().replace(" ", "").replace("-", "_")
        if any(re.search(p, k) for p in pats):
            return c
    if required:
        raise ValueError(f"Column not found for patterns: {pats}")
    return None

pv_col   = find_col(cols, [r"^pv_id$", r"^pvid$"], required=True)
time_col = find_col(cols, [r"timestamp", r"datetime", r"^date$", r"^time$", r"ymd", r"^dt$", r"^unnamed:0$"], required=True)
energy_col = find_col(cols, [r"\benergy\b", r"\bgen\b", r"power", r"output", r"\bpv\b"], required=True)

print(f"[INFO] columns -> pv_id='{pv_col}', time='{time_col}', energy='{energy_col}'")

# ------------------------
# 데이터 로드/전처리
# ------------------------
usecols = [pv_col, time_col, energy_col]
df = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME, usecols=usecols)

# 시간 파싱 + KST로 정규화(타임존 있으면 KST로 변환, 없으면 KST로 가정)
t = pd.to_datetime(df[time_col], errors="coerce")
if pd.api.types.is_datetime64tz_dtype(t):
    t = t.dt.tz_convert("Asia/Seoul")
else:
    t = t.dt.tz_localize("Asia/Seoul")
df[time_col] = t
df = df.dropna(subset=[time_col, energy_col])

# 관심 구간 필터 (START <= t < END)
mask = (df[time_col] >= START) & (df[time_col] < END)
df = df.loc[mask].copy()

# 표본 간격(분) 추정
step_min = (df[time_col].sort_values().diff().dt.total_seconds()/60).dropna().round().mode()
step_min = int(step_min.iloc[0]) if len(step_min) else 15
print(f"[INFO] inferred sampling step ≈ {step_min} min")

# 전력(kW)라면 kWh로 적분
if ENERGY_IS_POWER_KW:
    df["_energy_kwh_"] = df[energy_col].astype(float) * (step_min / 60.0)
    val_col = "_energy_kwh_"
else:
    val_col = energy_col

# ------------------------
# 한시간 단위 총합(모든 pv_id 합산)
# ------------------------
# 1) 동일 시각·pv_id 중복 제거
df = df.sort_values([pv_col, time_col]).drop_duplicates(subset=[pv_col, time_col], keep="last")

# 2) 시각 기준으로 합산(모든 pv_id)
g = df.groupby(time_col, as_index=True)[val_col].sum().to_frame("sum_energy")
# 3) 한시간 단위로 리샘플(합)
hourly = g.resample("1H").sum()

print(hourly.head(12))          # 앞 부분 확인
# 저장 원하면 주석 해제
hourly.to_csv("./Projecct/EM/Data/hourly_energy_sum_20250709_20250818.csv")

# (선택) 간단한 라인 그래프
import matplotlib.pyplot as plt
plt.figure(figsize=(12,4))
plt.plot(hourly.index.tz_convert("Asia/Seoul"), hourly["sum_energy"])
plt.title("Hourly total energy (all pv_id)")
plt.xlabel("Time (KST)"); plt.ylabel("Energy (kWh)" if ENERGY_IS_POWER_KW else "Energy (unit)")
plt.tight_layout()
plt.savefig('./Project/EM/Data/energy_total_sum_true.png')
