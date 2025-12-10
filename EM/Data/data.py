# Gantt-like timeline graph for pv_id start~end ranges
import os, re, hashlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

path = "./Project/EM/Data/pv_energy_presence_segments.xlsx"
assert os.path.exists(path), "엑셀 파일이 보이지 않습니다: /mnt/data/pv_energy_presence_segments.xlsx"

# Load
df = pd.read_excel(path, sheet_name=0)

# Find columns (case-insensitive)
def find_col(cols, pattern):
    for c in cols:
        if re.search(pattern, str(c), re.IGNORECASE):
            return c
    return None

pv_col    = find_col(df.columns, r"^pv_id$|^pvid$|pv.?id")
start_col = find_col(df.columns, r"^start$")
end_col   = find_col(df.columns, r"^end$")
assert pv_col and start_col and end_col, f"필수 컬럼을 찾지 못했습니다. columns={list(df.columns)}"

# Normalize datetimes
def to_naive(series):
    s = pd.to_datetime(series, errors="coerce")
    if pd.api.types.is_datetime64tz_dtype(s):
        try:
            s = s.dt.tz_convert("Asia/Seoul")
        except Exception:
            pass
        s = s.dt.tz_localize(None)
    return s

df[start_col] = to_naive(df[start_col])
df[end_col]   = to_naive(df[end_col])

df = df.dropna(subset=[pv_col, start_col, end_col]).sort_values([pv_col, start_col]).reset_index(drop=True)

# Prepare plotting data
start_num = mdates.date2num(df[start_col])
width_num = (df[end_col] - df[start_col]).dt.total_seconds().to_numpy() / 86400.0  # days

# y positions
pv_ids = df[pv_col].astype(str).unique().tolist()
ypos_map = {pid: i for i, pid in enumerate(pv_ids)}
df["_y"] = df[pv_col].astype(str).map(ypos_map)

# Shorten labels for readability
def safe_label(x, maxlen=18):
    s = str(x)
    if s.startswith("b'") or s.startswith('b"'):
        s = "id_" + hashlib.md5(s.encode()).hexdigest()[:8]
    if len(s) > maxlen:
        s = s[:maxlen-3] + "..."
    return s

labels = [safe_label(pid) for pid in pv_ids]

# Figure size scales with number of pv_ids
height = max(4, 0.35 * len(pv_ids) + 1)
fig, ax = plt.subplots(figsize=(12, height))

# Single chart: barh for all segments
ax.barh(df["_y"], width_num, left=start_num, height=0.8)

# Format axes
ax.xaxis_date()
ax.xaxis.set_major_locator(mdates.AutoDateLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d\n%H:%M"))
ax.set_yticks(range(len(pv_ids)))
ax.set_yticklabels(labels)
ax.set_xlabel("Time")
ax.set_ylabel("pv_id")
ax.set_title("PV energy availability (start~end) by pv_id")
plt.tight_layout()
plt.savefig('./Project/EM/Data/data_%d.png'%(1))