import os, re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------
# 설정
# -----------------------
PATH = "./Project/EM/Data/Excel/hourly_energy_sum_20250709_20250818.xlsx"  # 첨부 엑셀
TIME_COL_HINT = r"(time|date|datetime|timestamp|^Unnamed: ?0$)"  # 시간 컬럼 힌트(없으면 1열 사용)
SUM_COL_HINT  = r"^sum_energy$"                                   # 값 컬럼 힌트
TZ = "Asia/Seoul"                                                 # 표시/저장 타임존
DISTRIBUTE_HOURLY = True   # True: 1시간 값 → 15분 값으로 1/4씩 분배, False: 계단(step) 유지

# -----------------------
# 1) 데이터 로드
# -----------------------
df = pd.read_excel(PATH, engine="openpyxl")

# 시간 컬럼/값 컬럼 자동 탐지
def find_col(cols, pattern, default=None):
    for c in cols:
        if re.search(pattern, str(c), re.IGNORECASE):
            return c
    return default

time_col = find_col(df.columns, TIME_COL_HINT, default=df.columns[0])
sum_col  = find_col(df.columns, SUM_COL_HINT,  default=None)
if sum_col is None:
    # 이름이 다르면 수치형 마지막 컬럼을 선택
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not num_cols:
        raise ValueError("sum_energy 컬럼을 찾지 못했습니다. 숫자형 컬럼이 없습니다.")
    sum_col = num_cols[-1]

# 시간 파싱 & 타임존 부여/정렬
t = pd.to_datetime(df[time_col], errors="coerce")
if pd.api.types.is_datetime64tz_dtype(t):
    t = t.dt.tz_convert(TZ)
else:
    t = t.dt.tz_localize(TZ)
df[time_col] = t
df = df.dropna(subset=[time_col, sum_col]).sort_values(time_col).set_index(time_col)

# -----------------------
# 2) 15분 해상도로 업샘플
#    - DISTRIBUTE_HOURLY=True  : 1시간 합계를 15분 *4 구간에 1/4씩 분배 (kWh 의미 보존)
#    - DISTRIBUTE_HOURLY=False : 계단형으로 4개 구간에 동일값 복제
# -----------------------
# 15분 인덱스 생성 (양 끝 포함)
full_15m = pd.date_range(df.index.min(), df.index.max() + pd.Timedelta(hours=1) - pd.Timedelta(minutes=45),
                         freq="15T", tz=TZ)

# 시간단위 값이 각 정시를 대표한다고 가정(예: 10:00 행은 10:00~11:00 구간)
df_15m = df.reindex(full_15m, method="ffill")  # 시간당 값을 15분 그리드에 퍼뜨림

if DISTRIBUTE_HOURLY:
    df_15m[sum_col] = df_15m[sum_col] / 4.0  # 한 시간 값 → 15분 값 1/4로 분배

df_15m.rename(columns={sum_col: "sum_energy_15min"}, inplace=True)

# -----------------------
# 3) 출력/시각화/저장
# -----------------------
print("[INFO] 15분 해상도 데이터 (앞 12행)")
print(df_15m.head(12))

# 그래프 (한 개 차트)
plt.figure(figsize=(12,4))
plt.plot(df_15m.index, df_15m["sum_energy_15min"])
plt.title("sum_energy @ 15-min resolution")
plt.xlabel("Time (KST)")
plt.ylabel("sum_energy per 15 min" if DISTRIBUTE_HOURLY else "sum_energy (step)")
plt.tight_layout()
plt.savefig('./Project/EM/Data/sum_energy_15min.png')
# 엑셀 저장(원하면 주석 해제; Excel은 tz를 못 저장하므로 tz 제거)
out = df_15m.copy()
out.index = out.index.tz_localize(None)
out.to_excel("./Project/EM/Data/Excel/sum_energy_15min.xlsx", engine="xlsxwriter")