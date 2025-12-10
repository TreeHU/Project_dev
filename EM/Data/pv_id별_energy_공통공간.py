import os, re
import pandas as pd
import numpy as np

# =========================
# 설정
# =========================
excel_path = "./Project/Henergy/EM/Data/pv_gens_weathers.xlsx"   # 파일 경로

sheet_name = "pv_gens_weathers"        # 시트명
energy_col_override = None             # 예: "pv_energy" (None이면 자동 탐지)
positive_only = False                  # True면 energy > 0 만 "존재"로 간주

# =========================
# 유틸 함수
# =========================
def find_col(cols, patterns, required=False):
    """패턴 중 하나라도 매칭되는 첫 컬럼 반환"""
    for c in cols:
        k = str(c).strip().lower().replace(" ", "").replace("-", "_")
        if any(re.search(p, k) for p in patterns):
            return c
    if required:
        raise ValueError(f"Column not found for patterns: {patterns}")
    return None

def infer_step(times: pd.DatetimeIndex) -> pd.Timedelta:
    """시간 간격 추정(최빈 간격 → 중앙값 대체)"""
    if len(times) < 2: 
        return pd.Timedelta(0)
    diffs = pd.Series(times).diff().dropna()
    if diffs.empty:
        return pd.Timedelta(0)
    mode = diffs.mode()
    if len(mode) > 0:
        return mode.iloc[0]
    return diffs.median()

def segments_from_times(times: pd.DatetimeIndex, step: pd.Timedelta, tol_factor: float = 1.1) -> pd.DataFrame:
    """연속 구간 계산: times는 정렬된 DatetimeIndex"""
    if len(times) == 0:
        return pd.DataFrame(columns=["start","end","steps","duration"])
    tser = pd.Series(times)
    tol = step * tol_factor if step != pd.Timedelta(0) else pd.Timedelta(0)
    groups = (tser.diff() > tol).cumsum()
    segs = tser.groupby(groups).agg(["min","max","count"]).reset_index(drop=True)
    segs.rename(columns={"min":"start","max":"end","count":"steps"}, inplace=True)
    segs["duration"] = segs["end"] - segs["start"] + (step if step!=pd.Timedelta(0) else pd.Timedelta(0))
    return segs[["start","end","steps","duration"]]

# =========================
# 1) 헤더 스캔 및 컬럼 자동 탐지
# =========================
hdr = pd.read_excel(excel_path, sheet_name=sheet_name, nrows=0)
cols = [str(c) for c in hdr.columns]

pv_col = find_col(cols, [r"^pv_id$", r"^pvid$"], required=True)
time_col = find_col(cols, [r"timestamp", r"datetime", r"^date$", r"^time$", r"ymd", r"dt", r"^unnamed:0$"], required=True)

if energy_col_override is not None:
    energy_col = energy_col_override
else:
    # energy/gen/power/output/pv 등에서 첫 후보
    energy_col = None
    for c in cols:
        k = c.lower().replace(" ", "")
        if c not in (pv_col, time_col) and any(p in k for p in ["energy","gen","power","output","pv"]):
            energy_col = c; break
    if energy_col is None:
        raise ValueError("발전량(energy) 컬럼을 찾지 못했습니다. energy_col_override에 직접 지정하세요.")

print(f"[INFO] pv_id={pv_col}, time={time_col}, energy={energy_col}")

# =========================
# 2) 데이터 로드/전처리
# =========================
usecols = [pv_col, time_col, energy_col]
df = pd.read_excel(excel_path, sheet_name=sheet_name, usecols=usecols)

df = df.dropna(subset=[pv_col, time_col]).copy()
df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
df = df.dropna(subset=[time_col])

# 동일 (pv_id, time) 중복 제거
df = df.sort_values([pv_col, time_col]).drop_duplicates(subset=[pv_col, time_col], keep="last")

# 존재 조건: NaN 제거 (+옵션으로 양수만)
if positive_only:
    df = df[df[energy_col] > 0]
else:
    df = df.dropna(subset=[energy_col])

# =========================
# 3) 피벗: 시각 x pv_id → 값 존재 여부
# =========================
wide = df.pivot_table(index=time_col, columns=pv_col, values=energy_col, aggfunc="first").sort_index()

# 각 pv_id 별 존재 시각
present_times_by_pv = {pid: wide.index[wide[pid].notna()] for pid in wide.columns}

# 전체 타임라인의 대표 간격(모드) 추정
global_step = infer_step(wide.index)

# =========================
# 4) (A) pv_id 별 존재 구간 출력
# =========================
print("\n[A] pv_id 별 energy 존재 연속 구간")
per_pv_segments = {}
for pid, times in present_times_by_pv.items():
    segs = segments_from_times(times, infer_step(times) or global_step)
    per_pv_segments[pid] = segs
    if len(segs)==0:
        print(f"- pv_id={pid}: (no segments)")
    else:
        for _, r in segs.iterrows():
            print(f"- pv_id={pid}: {r['start']} ~ {r['end']} | steps={int(r['steps'])} | duration={r['duration']}")

# (필요 시 CSV로 저장)
# with pd.ExcelWriter("pv_presence_segments.xlsx") as xw:
#     for pid, segs in per_pv_segments.items():
#         segs.to_excel(xw, sheet_name=str(pid), index=False)

# =========================
# 5) (B) 모든 pv_id 공통 존재 구간 출력
# =========================
mask_all = wide.notna().all(axis=1)
common_times = wide.index[mask_all]
common_segs = segments_from_times(common_times, infer_step(common_times) or global_step)

print("\n[B] 모든 pv_id 공통 존재 연속 구간")
if len(common_segs)==0:
    print("(no common segments)")
else:
    for _, r in common_segs.iterrows():
        print(f"- {r['start']} ~ {r['end']} | steps={int(r['steps'])} | duration={r['duration']}")

# (필요 시 CSV)
# common_segs.to_csv("pv_common_presence_segments.csv", index=False)
