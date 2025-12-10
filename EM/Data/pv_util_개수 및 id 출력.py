import pandas as pd


# 엑셀 파일 불러오기 (pv_gens_weathers 시트 사용)
file_path = "./Project/EM/Data/pv_gens_weathers.xlsx"
sheet = "pv_gens_weathers"          # 시트명이 다르면 바꿔주세요
pv_col_name = "pv_id"               # 컬럼명이 다르면 바꿔주세요

# pv_id 컬럼만 읽기 (엑셀이 매우 클 경우 속도에 도움이 됩니다)
df = pd.read_excel(file_path, sheet_name=sheet, usecols=[pv_col_name])

# ===== 2) pv_id 고유값 추출 =====
# 원래 인덱스를 유지하면 0, 944, 1893 같은 인덱스가 남습니다.
# 보기 좋게 0..N-1로 다시 매기려면 reset_index(drop=True) 사용
unique_pv_ids = df[pv_col_name].drop_duplicates().reset_index(drop=True)

# ===== 3) 콘솔에서 줄임표 없이 전부 보이기(선택) =====
pd.set_option("display.max_rows", None)     # 모든 행 표시
pd.set_option("display.max_colwidth", None) # 긴 문자열 자르지 않음
pd.set_option("display.width", 0)           # 화면 폭에 맞춰 줄바꿈 비활성(노트북 기준)

# print(unique_pv_ids.to_string(index=False))  # 인덱스 없이 전부 출력하고 싶을 때

# ===== 4) 텍스트 파일로 저장 =====
# (a) 값만 한 줄에 하나씩 저장
out_txt_path = "./Project/EM/Data/unique_pv_ids.txt"
unique_pv_ids.to_csv(out_txt_path, index=False, header=False)

# (b) 헤더 포함 CSV로 저장하고 싶으면 아래 사용
# out_csv_path = "/mnt/data/unique_pv_ids.csv"
# unique_pv_ids.to_csv(out_csv_path, index=False, header=[pv_col_name])

print(f"고유 pv_id 개수: {unique_pv_ids.shape[0]}")
print(f"텍스트 파일 저장 완료: {out_txt_path}")