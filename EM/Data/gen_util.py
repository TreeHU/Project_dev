import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, re, hashlib

# -----------------------------------------
# 설정
# -----------------------------------------
excel_path = "./Project/EM/Data/pv_gens_weathers.xlsx"   # 파일 경로
sheet_name = "pv_gens_weathers"        # 시트명
save_dir = './Project/EM/Data/Fig/'                        # 예: "plots" (PNG 저장 원하면 폴더명 지정)
max_rows_to_read = None                # 메모리 부담시 정수로 제한 (예: 200000)

# -----------------------------------------
# 유틸: 헤더만 먼저 읽어 컬럼 자동 탐지
# -----------------------------------------
hdr = pd.read_excel(excel_path, sheet_name=sheet_name, nrows=0)
cols = [str(c).strip() for c in hdr.columns]

def find_col(cols, patterns, required=False):
    """패턴 리스트 중 하나라도 매칭되는 첫 컬럼 반환"""
    for c in cols:
        lc = c.lower().replace(" ", "").replace("-", "_")
        for pat in patterns:
            if re.search(pat, lc):
                return c
    if required:
        raise ValueError(f"Required column not found for patterns: {patterns}")
    return None


def safe_name(x):
    # 1) bytes면 디코드
    if isinstance(x, (bytes, bytearray)):
        try:
            s = x.decode('utf-8', 'ignore')
        except Exception:
            s = x.decode('latin1', 'ignore')
    else:
        s = str(x)
    # 2) 경로 구분자 제거
    s = s.replace(os.sep, "_").replace("/", "_").replace("\\", "_")
    # 3) 파일명에 안전한 문자만 남김
    s = re.sub(r'[^A-Za-z0-9._-]+', '_', s).strip("_")
    # 4) 완전히 비면 해시로 대체
    if not s:
        s = hashlib.md5(str(x).encode()).hexdigest()[:8]
    return s



# pv_id 컬럼
pv_col = find_col(cols, patterns=[r"^pv_id$", r"^pvid$"], required=True)

# 시간(날짜) 컬럼 후보 (가능하면 1개만)
time_col = find_col(cols, patterns=[
    r"timestamp", r"datetime", r"date", r"^\s*time\s*$", r"\by?m?d?h?m?\b", r"dt"
], required=False)

# 에너지(발전량) 컬럼 후보 (여러 패턴)
# 흔한 명칭: energy, gen, power, output, pv_power 등
energy_patterns = [r"energy", r"\bgen\b", r"power", r"output", r"pv"]
energy_cols = [c for c in cols
               if any(re.search(p, c.lower().replace(" ", "")) for p in energy_patterns)
               and c != pv_col and c != time_col]

if not energy_cols:
    raise ValueError("발전량(energy) 컬럼을 자동으로 찾지 못했습니다. 파일의 발전량 컬럼명을 확인해 주세요.")

# 첫 번째 후보를 기본 energy로 사용 (여러 개면 바꿔가며 시각화 가능)
energy_col = energy_cols[0]
print(f"[INFO] pv_id: {pv_col}, time: {time_col}, energy: {energy_col}")

# -----------------------------------------
# 필요한 컬럼만 읽기 (메모리/속도 최적화)
# -----------------------------------------
usecols = [pv_col, energy_col] + ([time_col] if time_col else [])
df = pd.read_excel(
    excel_path, sheet_name=sheet_name,
    usecols=usecols, nrows=max_rows_to_read
)

# 시간 파싱
if time_col:
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")

# 결측/이상치 간단 처리
df = df.dropna(subset=[pv_col, energy_col]).copy()
if time_col:
    # 시간 정렬
    df = df.sort_values(time_col)

# -----------------------------------------
# pv_id 리스트
# -----------------------------------------
pv_ids = df[pv_col].dropna().unique().tolist()
print(f"[INFO] unique pv_id count = {len(pv_ids)}")

# 저장 폴더 생성(옵션)
if save_dir:
    os.makedirs(save_dir, exist_ok=True)

# -----------------------------------------
# 시각화: pv_id별로 1개 차트씩
# (규칙: 한 플롯에 한 차트, seaborn 사용 금지, 색 지정 금지)
# -----------------------------------------
for pid in pv_ids:
    sub = df[df[pv_col] == pid]
    if sub.empty:
        continue

    plt.figure(figsize=(10, 4))
    if time_col:
        plt.plot(sub[time_col], sub[energy_col], label=str(pid))
        plt.xlabel("Time")
    else:
        plt.plot(np.arange(len(sub)), sub[energy_col], label=str(pid))
        plt.xlabel("Index")
    plt.ylabel(energy_col)
    plt.title(f"Energy time series for pv_id = {pid}")
    plt.legend()
    plt.tight_layout()

    if save_dir:
        fname = f"pv_id : {safe_name(pid)}.png"
        out = os.path.join(save_dir, fname)
        plt.savefig(out, dpi=150)
    plt.show()

# -----------------------------------------
# (옵션) 집계 예시: 일 단위 평균/합계
# -----------------------------------------
if time_col:
    # 15분 단위가 아니라면 다운샘플링/업샘플링 자유롭게 변경
    # 예) 일(day) 평균
    daily = (
        df.set_index(time_col)
          .groupby(pv_col)[energy_col]
          .resample("1D").mean()
          .reset_index()
    )
    # pv_id별 일평균 그래프(각 pv_id를 한 차트에 겹쳐서 그릴 수도 있으나,
    # 차트 1개에는 여러 라인이 들어가도 '한 차트'이므로 규칙에 부합합니다.)
    plt.figure(figsize=(10, 4))
    for pid in pv_ids:
        ss = daily[daily[pv_col] == pid]
        plt.plot(ss[time_col], ss[energy_col], label=str(pid))
    plt.xlabel("Date")
    plt.ylabel(f"Daily mean of {energy_col}")
    plt.title("Daily mean energy by pv_id")
    plt.legend()
    plt.tight_layout()
    if save_dir:
        out = os.path.join(save_dir, "pv_daily_mean.png")
        plt.savefig(out, dpi=150)
    plt.show()
