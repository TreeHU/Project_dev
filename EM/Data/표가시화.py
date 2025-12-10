import re
import pandas as pd

# 1) 아래 triple quotes 안에 질문의 리스트를 그대로 붙여 넣으세요.
raw = r"""
[A] pv_id 별 energy 존재 연속 구간
- pv_id=b"\x9csft\xc0\xf4DV\xbc\xe6\x9f\x95'\xbb\x8ab": 2025-07-09 12:00:00+09:00 ~ 2025-08-18 00:00:00+09:00 | steps=949 | duration=39 days 13:00:00
- pv_id=b"nNxB'\xf6J\xdb\xb9\x12E\x94\xa8B\xdd^": 2024-06-12 12:00:00+09:00 ~ 2024-11-26 15:00:00+09:00 | steps=4012 | duration=167 days 04:00:00
- pv_id=b' x\xe3T\x975K\x0f\x87\xee\xc0IN\xc29\xa9': 2025-07-09 17:00:00+09:00 ~ 2025-08-18 00:00:00+09:00 | steps=944 | duration=39 days 08:00:00
- pv_id=b'#H\xcf:\xaa\xdaM\x05\xbd\xff\xa4\x98\x17\xed\xa3\xdb': 2025-07-09 11:00:00+09:00 ~ 2025-08-18 00:00:00+09:00 | steps=950 | duration=39 days 14:00:00
- pv_id=b'#\x072\xed\xc9\xa0I\xdf\x9b=x\xe1\xaa\x16&\xbd': 2025-04-16 14:00:00+09:00 ~ 2025-08-18 00:00:00+09:00 | steps=2963 | duration=123 days 11:00:00
- pv_id=b'(\x9a?\x0b3\x13O\xac\x80\xdfl\x891\x94\xeb\x80': 2025-07-09 15:00:00+09:00 ~ 2025-08-18 00:00:00+09:00 | steps=946 | duration=39 days 10:00:00
- pv_id=b'-\xa9\xfd\xc78\x9aHd\xa1\x125D\x11 x\xa1': 2025-05-20 14:00:00+09:00 ~ 2025-08-18 00:00:00+09:00 | steps=2147 | duration=89 days 11:00:00
- pv_id=b'6?N\xf0-\xdbE\xbd\x90U\xe4\xea\x0fI:1': 2025-02-14 14:00:00+09:00 ~ 2025-08-18 00:00:00+09:00 | steps=4427 | duration=184 days 11:00:00
- pv_id=b'<"\xa5\xd2)]A5\x81\xb2H\xbd\x86#\xfe\x03': 2025-05-20 13:00:00+09:00 ~ 2025-08-18 00:00:00+09:00 | steps=2148 | duration=89 days 12:00:00
- pv_id=b'<\xc2r\x11\xd1\x08Jk\x83\xe3\x9f\x92\x1a\x8e2\x88': 2024-03-04 16:00:00+09:00 ~ 2025-08-18 00:00:00+09:00 | steps=12753 | duration=531 days 09:00:00
- pv_id=b'=\xf5\x87\xea\x99\x85J\x89\xbe\x86\xec\x81\xd9\x83\t\x11': 2025-02-14 11:00:00+09:00 ~ 2025-08-18 00:00:00+09:00 | steps=4430 | duration=184 days 14:00:00
- pv_id=b'>\x1b\xaf\x04E\xbbG\xd4\x86\xd3\xc79\r\x15)\xc7': 2025-05-20 10:00:00+09:00 ~ 2025-08-18 00:00:00+09:00 | steps=2151 | duration=89 days 15:00:00
- pv_id=b'C\xc5\xd6\x82O\x82I\x18\xa5\xcb\xd1H\xfc\xe8\xba\xfe': 2025-05-21 09:00:00+09:00 ~ 2025-08-18 00:00:00+09:00 | steps=2128 | duration=88 days 16:00:00
- pv_id=b'L\xac\xf3\x16?\xa5KP\xa2\xad\x85\xa6s\x04n/': 2025-07-09 18:00:00+09:00 ~ 2025-08-18 00:00:00+09:00 | steps=943 | duration=39 days 07:00:00
- pv_id=b'S<c\xcc\x18\x1bD\xc4\xb4*\xd6(\xc0\xfb\xae\xbc': 2025-02-14 08:00:00+09:00 ~ 2025-08-18 00:00:00+09:00 | steps=4433 | duration=184 days 17:00:00
- pv_id=b'WU\x08!\x14xDz\x96\x8d\xe9\x95\x0ca\t\xd4': 2025-01-10 14:00:00+09:00 ~ 2025-08-18 00:00:00+09:00 | steps=5267 | duration=219 days 11:00:00
- pv_id=b'W^T\xf5\xce\xb9Cr\xa2\x97J\xba\x07\xce\x98e': 2025-03-05 15:00:00+09:00 ~ 2025-08-18 00:00:00+09:00 | steps=3970 | duration=165 days 10:00:00
- pv_id=b'Y\x04\xc7\x087lO\xc9\xa3\x8d\x9d-\x0e\x0b\\\xab': 2025-03-05 18:00:00+09:00 ~ 2025-08-18 00:00:00+09:00 | steps=3967 | duration=165 days 07:00:00
- pv_id=b'[`D\xff\xd9\r@\x98\xaf\xa0Z\xd2\x82+\xdf,': 2025-07-09 11:00:00+09:00 ~ 2025-08-18 00:00:00+09:00 | steps=950 | duration=39 days 14:00:00
- pv_id=b'\x00\xaa\xeb\xcd;\x9bA\xeb\x90Y\xf8\xa4M\x19/1': 2025-01-10 10:00:00+09:00 ~ 2025-08-18 00:00:00+09:00 | steps=5271 | duration=219 days 15:00:00
- pv_id=b'\x03\x0f\xcb\xa3\xdfgAl\x8a\x1af\xc7\xf9\xd6Pw': 2024-03-03 15:00:00+09:00 ~ 2025-08-18 00:00:00+09:00 | steps=12778 | duration=532 days 10:00:00
- pv_id=b'\x08\xb7EX\xb2\x92E"\x92\x90\xe1\x10\xf2\x02I\xa6': 2025-01-11 10:00:00+09:00 ~ 2025-08-18 00:00:00+09:00 | steps=5247 | duration=218 days 15:00:00
- pv_id=b'\x157\xae\xf7\x00\x92N5\x82\xde/\xac\x11\x0eK\xed': 2025-01-11 10:00:00+09:00 ~ 2025-08-18 00:00:00+09:00 | steps=5247 | duration=218 days 15:00:00
- pv_id=b'\x1b~\xce\xe4v\xe8@\xf3\xa0\xf7\xfd\xc3\xa1\x8d(:': 2025-04-16 09:00:00+09:00 ~ 2025-08-18 00:00:00+09:00 | steps=2968 | duration=123 days 16:00:00
- pv_id=b'\x1e\t\xbd\x8c\xed\x13I\xfc\xa3"\x98ij:\xa5\x9f': 2025-03-05 14:00:00+09:00 ~ 2025-08-18 00:00:00+09:00 | steps=3971 | duration=165 days 11:00:00
- pv_id=b'\x82\xd6\xcc`?SBm\xa2\x87W\xff`\x0e\xab\x89': 2025-03-05 14:00:00+09:00 ~ 2025-08-18 00:00:00+09:00 | steps=3971 | duration=165 days 11:00:00
- pv_id=b'\x91\x85\xb1\xd2\xf9\xd3Ag\xb7\xa3\xf2\xe7\xf2f\r4': 2025-01-10 11:00:00+09:00 ~ 2025-08-18 00:00:00+09:00 | steps=5270 | duration=219 days 14:00:00
- pv_id=b'\x91i\xf4\xf9\xca\x1bFR\xaa>\xfah9F0\xf4': 2025-07-09 12:00:00+09:00 ~ 2025-08-18 00:00:00+09:00 | steps=949 | duration=39 days 13:00:00
- pv_id=b'\x9a\x89\xa4\x8f\xdfl@_\xae\xcdS.\x0b:\x1f\x93': 2025-07-09 19:00:00+09:00 ~ 2025-08-18 00:00:00+09:00 | steps=942 | duration=39 days 06:00:00
- pv_id=b'\x9c\x97}\xd1\x98PL\x91\x90\x96\xde\xcfk\xc4+7': 2025-07-09 12:00:00+09:00 ~ 2025-08-18 00:00:00+09:00 | steps=949 | duration=39 days 13:00:00
- pv_id=b'\xa0&\x88\x9em\xb5O\xe3\x8e\xbc?\xa0:Z\x99\xc2': 2024-06-12 10:00:00+09:00 ~ 2025-08-18 00:00:00+09:00 | steps=10359 | duration=431 days 15:00:00
- pv_id=b'\xa0\xe2M3\x0c\x9dER\x9b\x9a\xd1T\r\x84X\x81': 2025-07-09 11:00:00+09:00 ~ 2025-08-18 00:00:00+09:00 | steps=950 | duration=39 days 14:00:00
- pv_id=b'\xa6\x7fO\x97\x9b\xceF#\xac\x1f\x0b\x8d\x9a\x9dT\xb6': 2025-04-23 17:00:00+09:00 ~ 2025-08-18 00:00:00+09:00 | steps=2792 | duration=116 days 08:00:00
- pv_id=b'\xa9\xeeL\x93n;E\\\xa5;f\xe6\x8d\x04E\xc9': 2025-01-09 12:00:00+09:00 ~ 2025-08-18 00:00:00+09:00 | steps=5293 | duration=220 days 13:00:00
- pv_id=b'\xb0^mo\xa2\xa7E>\xa4\xf72\xc4*T`g': 2025-01-10 16:00:00+09:00 ~ 2025-08-18 00:00:00+09:00 | steps=5265 | duration=219 days 09:00:00
- pv_id=b'\xb6V\xfd\x81b\xffN\xb7\x8f\xfeO2\xbd\xaa\x95#': 2024-03-03 16:00:00+09:00 ~ 2025-08-18 00:00:00+09:00 | steps=12777 | duration=532 days 09:00:00
- pv_id=b'\xbb\xf0nH\xdc\x98I)\xbd<\xa6\xbe)0Q?': 2025-02-14 15:00:00+09:00 ~ 2025-08-18 00:00:00+09:00 | steps=4426 | duration=184 days 10:00:00
- pv_id=b'\xbf\x96\xf3\x1e\x86\rO\xb8\xba\x0cB@\x82\xeeb\x0e': 2025-07-09 08:00:00+09:00 ~ 2025-08-18 00:00:00+09:00 | steps=953 | duration=39 days 17:00:00
- pv_id=b'\xc6\x932@(\x0cJ\xa1\xb5\x10q\xeb\x9f\xd9B{': 2024-03-04 11:00:00+09:00 ~ 2025-08-18 00:00:00+09:00 | steps=12758 | duration=531 days 14:00:00
- pv_id=b'\xc8\xec\xd1*P\xe5@\xd5\xa4\xa4\xf3@\x16\xc8b(': 2024-03-04 10:00:00+09:00 ~ 2025-08-18 00:00:00+09:00 | steps=12759 | duration=531 days 15:00:00
- pv_id=b'\xcc\x9c2Y&\xa6HB\x98V\x8f\xb2\xd5 \xfb\x86': 2025-08-14 11:00:00+09:00 ~ 2025-08-18 00:00:00+09:00 | steps=86 | duration=3 days 14:00:00
- pv_id=b'\xcf\xa9\xfc*\x01\xdaA\x08\xb5|\xaa\xa5\x1dv$\x84': 2024-06-12 10:00:00+09:00 ~ 2025-08-18 00:00:00+09:00 | steps=10359 | duration=431 days 15:00:00
- pv_id=b'\xd9\x119\x87[\xd7Oj\xa5$\xb1\xadt\xf5\x87\x18': 2025-03-05 10:00:00+09:00 ~ 2025-08-18 00:00:00+09:00 | steps=3975 | duration=165 days 15:00:00
- pv_id=b'\xdf\xd2ZL\x92\xa9K\x8c\x83\x96K\xcd\xe7\xcd\xadM': 2025-05-20 13:00:00+09:00 ~ 2025-08-18 00:00:00+09:00 | steps=2148 | duration=89 days 12:00:00
- pv_id=b'\xe6\xadS\x10hsDu\xa6{\xebmz\xdd&\x8f': 2024-03-04 10:00:00+09:00 ~ 2025-08-18 00:00:00+09:00 | steps=12759 | duration=531 days 15:00:00
- pv_id=b'\xe9n|\x13|OA#\x95\xc7\xb0\xf2\x8e\xcf\x16J': 2024-03-03 13:00:00+09:00 ~ 2025-08-18 00:00:00+09:00 | steps=12780 | duration=532 days 12:00:00
- pv_id=b'\xf1\xfc\xc9wX\xb0@,\x80\xefu\xcc]\x12?\x7f': 2025-01-10 15:00:00+09:00 ~ 2025-08-18 00:00:00+09:00 | steps=5266 | duration=219 days 10:00:00
- pv_id=b'\xfeNS^H\xe8O\xdb\x92\xc9\xf4X,7\x91\x86': 2025-07-09 14:00:00+09:00 ~ 2025-08-18 00:00:00+09:00 | steps=947 | duration=39 days 11:00:00
- pv_id=b'cv\xb3(\x83\x97L\xba\xa2\xd6\xb8\x9c\xd6\xf2\xc9&': 2025-07-09 12:00:00+09:00 ~ 2025-08-18 00:00:00+09:00 | steps=949 | duration=39 days 13:00:00
- pv_id=b'h\x97i\x8a\x83\x04Gg\xa2Ce\t(\xdc\xbe\xfc': 2025-03-05 11:00:00+09:00 ~ 2025-08-18 00:00:00+09:00 | steps=3974 | duration=165 days 14:00:00
- pv_id=b'i\xd5E8YJL\x15\x90\x01z\xe0\xe7\x10uH': 2025-01-10 15:00:00+09:00 ~ 2025-08-18 00:00:00+09:00 | steps=5266 | duration=219 days 10:00:00
- pv_id=b'r\x82\xed(\xb7\rE\x81\xb7]\xe7\x11Ly\xb7\xaf': 2025-01-10 12:00:00+09:00 ~ 2025-08-18 00:00:00+09:00 | steps=5269 | duration=219 days 13:00:00
- pv_id=b'r\xe8\xf1\xeeU\xe8O|\x8b"\xa3cL\x9a\xdd?': 2025-07-09 09:00:00+09:00 ~ 2025-08-18 00:00:00+09:00 | steps=952 | duration=39 days 16:00:00
- pv_id=b'uW\x07\x00\xe2{G\xd4\xbf\xdc\x9d\x17\x95\xd6Et': 2025-01-09 17:00:00+09:00 ~ 2025-08-18 00:00:00+09:00 | steps=5288 | duration=220 days 08:00:00
- pv_id=b'}\xb9\x8d=*\xc2O\xef\x95:[\x06TP\xd3E': 2025-05-20 12:00:00+09:00 ~ 2025-08-18 00:00:00+09:00 | steps=2149 | duration=89 days 13:00:00
- pv_id=b'~N~\xda\xe3\xafM\x04\x8a9\xef{\xb3\x98\x11s': 2025-01-10 18:00:00+09:00 ~ 2025-08-18 00:00:00+09:00 | steps=5263 | duration=219 days 07:00:00
"""

# 2) 각 라인을 파싱해서 표로 만들기
rows = []
for line in raw.splitlines():
    line = line.strip()
    if not line.startswith("- pv_id="):
        continue

    # pv_id 부분과 나머지 분리
    try:
        left, right = line.split(":", 1)
    except ValueError:
        continue

    pv_id = left.replace("- pv_id=", "").strip()

    # 날짜 구간 / steps / duration 분리
    # 예: " 2025-07-09 ... ~ 2025-08-18 ... | steps=949 | duration=39 days 13:00:00"
    try:
        date_part, rest = right.split("| steps=", 1)
        start_str, end_str = [s.strip() for s in date_part.split("~")]
        steps_str, duration_str = rest.split("| duration=", 1)
        steps = int(steps_str.strip())
        duration_txt = duration_str.strip()
    except ValueError:
        continue

    rows.append({
        "pv_id": pv_id,
        "start": start_str,
        "end": end_str,
        "steps": steps,
        "duration_text": duration_txt,
    })

df = pd.DataFrame(rows)

# 3) 타입 변환 및 파생값
df["start"] = pd.to_datetime(df["start"], errors="coerce")
df["end"] = pd.to_datetime(df["end"], errors="coerce")
df["duration_text_td"] = pd.to_timedelta(df["duration_text"], errors="coerce")
df["duration_calc"] = df["end"] - df["start"]
df["duration_gap"] = (df["duration_calc"] - df["duration_text_td"]).dt.total_seconds()

# 보기 좋게 정렬/선택
out = df[["pv_id","start","end","steps","duration_text","duration_calc","duration_gap"]].sort_values("start")
print(out.to_string(index=False))


out.to_csv("./Project/EM/Data/pv_gens_weathers.xlsx" , index=False, encoding="utf-8")
out2 = out.copy()
for c in ["start", "end"]:
    out2[c] = pd.to_datetime(out2[c], errors="coerce").dt.tz_convert("Asia/Seoul").dt.tz_localize(None).dt.strftime("%Y-%m-%d %H:%M:%S")
out2["duration_calc"] = out2["duration_calc"].astype("timedelta64[ns]").astype(str)

out2.to_excel("./Project/EM/Data/pv_energy_presence_segments.xlsx", index=False)
