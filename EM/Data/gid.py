import os, glob, math, re
import matplotlib.pyplot as plt

# ===== 설정 =====

folder = "./Project/EM/Data/Fig"                 # PNG들이 들어있는 폴더
out_path = "./Project/EM/Data/Fig/Grid/grid_page2.png"        # 저장 파일 경로 (원치 않으면 None)
cols = 5                            # 그리드 열 개수(행은 자동 계산)
max_images = None                   # 최대 이미지 수 제한 (None이면 전부)
title_with_filename = True          # 각 칸에 파일명 표시 여부

# 숫자 자연 정렬용 키(파일명에 숫자 포함 시 1,2,10 순서 유지)
def natural_key(s):
    b = os.path.basename(s)
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", b)]

# 1) 파일 모으기
files = sorted(glob.glob(os.path.join(folder, "*.png")), key=natural_key)
if max_images is not None:
    files = files[:max_images]
n = len(files)
if n == 0:
    raise SystemExit("폴더에 .png 파일이 없습니다.")

# 2) 그리드 크기 계산
rows = math.ceil(n / cols)

# 3) 그림 사이즈(비례 조절)
fig_w = cols * 4       # 열당 가로 3인치
fig_h = rows * 1.5     # 행당 세로 3인치
fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h), squeeze=False)

# 4) 채우기
for i, ax in enumerate(axes.ravel()):
    if i < n:
        img = plt.imread(files[i])
        ax.imshow(img)
        ax.axis("off")
        if title_with_filename:
            ax.set_title(os.path.basename(files[i]), fontsize=8)
    else:
        ax.axis("off")

plt.tight_layout()

# 5) 저장 및 표시
if out_path:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.show()