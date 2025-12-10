import re, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =======================
# 설정
# =======================
excel_path = "./Project/EM/Data/pv_gens_weathers.xlsx"   # 엑셀 경로
sheet_name = "pv_gens_weathers"        # 시트명
positive_only = False                  # True면 energy>0 인 경우만 유효로 간주
out_dir = './Project/EM/Data/Fig/'              # 결과 저장 폴더
max_rows_to_read = None                # 메모리 제한시 정수(예: 300000)

os.makedirs(out_dir, exist_ok=True)

# =======================
# 1) 헤더 스캔(자동 컬럼 탐지)
# =======================
hdr = pd.read_excel(excel_path, sheet_name=sheet_name, nrows=0)
cols = [str(c).strip() for c in hdr.columns]

def find_col(cols, patterns, required=False):
    for c in cols:
        k = c.lower().replace(" ", "").replace("-", "_")
        if any(re.search(p, k) for p in patterns):
            return c
    if required:
        raise ValueError(f"Column not found: {patterns}")
    return None

pv_col   = find_col(cols, [r"^pv_id$", r"^pvid$"], required=True)
time_col = find_col(cols, [r"^unnamed:0$", r"timestamp", r"datetime", r"^date$", r"^time$", r"\by?m?d?h?m?\b", r"^dt$"], required=True)
energy_col_candidates = [c for c in cols if c not in (pv_col, time_col)]
energy_col = None
for c in energy_col_candidates:
    k = c.lower().replace(" ", "")
    if any(p in k for p in ["energy","gen","power","output","pv"]):
        energy_col = c; break
if energy_col is None:
    raise ValueError("발전량(energy) 컬럼을 찾지 못했습니다.")

print(f"[INFO] pv_id={pv_col}, time={time_col}, energy={energy_col}")

# =======================
# 2) 데이터 로드/전처리
# =======================
usecols = [pv_col, time_col, energy_col]
df = pd.read_excel(excel_path, sheet_name=sheet_name, usecols=usecols, nrows=max_rows_to_read)
df = df.dropna(subset=[pv_col, time_col]).copy()
df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
df = df.dropna(subset=[time_col])
df = df.sort_values([pv_col, time_col])
df = df.drop_duplicates(subset=[pv_col, time_col], keep="last")

if positive_only:
    df = df[df[energy_col] > 0]
else:
    df = df.dropna(subset=[energy_col])

# =======================
# 3) 피벗 → 시각별 모든 pv_id 유무 마스크
# =======================
wide = df.pivot_table(index=time_col, columns=pv_col, values=energy_col, aggfunc="first").sort_index()
mask_all = wide.notna().all(axis=1)
avail_times = wide.index[mask_all]

if len(avail_times) == 0:
    print("모든 pv_id가 동시에 존재하는 시각이 없습니다.")
else:
    # 기준 간격 추정(모드/중앙값 기반, 예외시 최빈값 대체)
    diffs = pd.Series(avail_times).to_series().diff().dropna()
    step = diffs.mode().iloc[0] if len(diffs)>0 else pd.Timedelta(0)
    if step == pd.Timedelta(0) and len(diffs)>0:
        step = diffs.median()

    # 연속 세그먼트 그룹핑 (허용 오차 1.1배)
    tol = step * 1.1 if step != pd.Timedelta(0) else pd.Timedelta(0)
    s = pd.Series(avail_times)
    groups = (s.diff() > tol).cumsum()
    segs = s.groupby(groups).agg(["min","max","count"]).reset_index(drop=True)
    segs["duration"] = segs["max"] - segs["min"] + (step if step!=pd.Timedelta(0) else pd.Timedelta(0))

    # 출력
    print("\n[모든 pv_id가 동시에 존재하는 연속 구간 목록]")
    for _, r in segs.iterrows():
        print(f"- {r['min']}  ~  {r['max']} | steps={int(r['count'])} | duration={r['duration']}")

    longest = segs.sort_values("count", ascending=False).iloc[0]
    print("\n[가장 긴 구간]")
    print(f"{longest['min']}  ~  {longest['max']} | steps={int(longest['count'])} | duration={longest['duration']}")

    # 저장
    segs.to_csv(os.path.join(out_dir, "overlap_segments.csv"), index=False)
    pd.DataFrame({"time": avail_times}).to_csv(os.path.join(out_dir, "overlap_times.csv"), index=False)

    # =======================
    # 4) 히트맵(가용성) + 최장구간 강조
    # =======================
    # 큰 데이터면 다운샘플(예: 1H)
    
    resample_rule = None  # 예: "1H"
    W = wide.notna().astype(int)
    if resample_rule:
        W = W.resample(resample_rule).min()  # 모든 pv_id가 그 시간창에 존재해야 1

    plt.figure(figsize=(10, 4 + 0.15*W.shape[1]))
    plt.imshow(W.T.values, aspect="auto", interpolation="nearest")
    plt.yticks(range(W.shape[1]), W.columns.astype(str))
    plt.xticks([0, W.shape[0]-1], [W.index[0].strftime("%Y-%m-%d %H:%M"), W.index[-1].strftime("%Y-%m-%d %H:%M")])
    plt.title("Availability heatmap by pv_id (1=present)")
    plt.xlabel("time"); plt.ylabel("pv_id")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "availability_heatmap.png"), dpi=150)
    plt.show()