# layout_builder_v4.py
# ------------------------------------------------------------
# 원하는 레이아웃을 정밀 재현 (가로 3열 + 우측 "결과창" 스타일) 후 ./Project/GUI/layout.png 저장
# ------------------------------------------------------------
import os, sys
if not os.environ.get("DISPLAY"):
    os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PyQt5 import QtCore, QtGui, QtWidgets

# ---------- 경로 ----------
SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
SAVE_DIR   = os.path.join(SCRIPT_DIR, "Project", "GUI")
os.makedirs(SAVE_DIR, exist_ok=True)
SAVE_PATH  = os.path.join(SAVE_DIR, "layout.png")

# ---------- 색/치수 ----------
BG_GREY      = "#d9d9d9"
PANEL_BORDER = "#b7b7b7"
HEADER_BLUE  = "#1b3c89"
HEADER_TXT   = "#ffffff"
LABEL_COLOR  = "#111111"
HANDLE_W_MAIN, HANDLE_W_COL = 8, 6
LEFT_MIN_W, LEFT_MAX_W = 430, 450

# ---------- 공통 위젯 ----------
def mk_le():
    w = QtWidgets.QLineEdit(); w.setFixedWidth(160); w.setMinimumHeight(18); return w

def mk_cb(items):
    w = QtWidgets.QComboBox(); w.addItems(items); w.setFixedWidth(160); w.setMinimumHeight(20); return w

class BlueHeader(QtWidgets.QFrame):
    def __init__(self, title):
        super().__init__()
        self.setFixedHeight(28)
        self.setStyleSheet(f"QFrame {{ background:{HEADER_BLUE}; border:1px solid {HEADER_BLUE}; border-radius:4px; }}")
        lab = QtWidgets.QLabel(title); lab.setStyleSheet(f"QLabel {{ color:{HEADER_TXT}; font-weight:600; }}")
        lay = QtWidgets.QHBoxLayout(self); lay.setContentsMargins(8, 2, 8, 2)
        lay.addWidget(lab); lay.addStretch(1)

class Section(QtWidgets.QWidget):
    def __init__(self, title, content):
        super().__init__()
        header = BlueHeader(title)
        wrap = QtWidgets.QFrame(); wrap.setStyleSheet(f"QFrame {{ background:{BG_GREY}; }}")
        v = QtWidgets.QVBoxLayout(wrap); v.setContentsMargins(6,6,6,6); v.addWidget(content)
        lay = QtWidgets.QVBoxLayout(self); lay.setContentsMargins(0,4,0,0); lay.setSpacing(6)
        lay.addWidget(header); lay.addWidget(wrap)

def mk_form(rows):
    grid = QtWidgets.QGridLayout(); grid.setHorizontalSpacing(8); grid.setVerticalSpacing(6)
    for r,(label,w) in enumerate(rows):
        lab = QtWidgets.QLabel(label); lab.setStyleSheet(f"QLabel {{ color:{LABEL_COLOR}; }}")
        grid.addWidget(lab, r, 0); grid.addWidget(w, r, 1)
    host = QtWidgets.QWidget(); host.setLayout(grid); return host

def white_panel(min_h=None):
    fr = QtWidgets.QFrame()
    fr.setFrameShape(QtWidgets.QFrame.Panel); fr.setFrameShadow(QtWidgets.QFrame.Plain); fr.setLineWidth(1)
    fr.setStyleSheet(f"QFrame {{ background:#ffffff; border:1px solid {PANEL_BORDER}; }}")
    if min_h: fr.setMinimumHeight(min_h)
    return fr

class ResultWindow(QtWidgets.QWidget):
    """우측 열: 결과창 스타일(타이틀바 + 컨트롤 버튼 느낌)"""
    def __init__(self):
        super().__init__()
        # 타이틀바
        titlebar = QtWidgets.QFrame()
        titlebar.setFixedHeight(22)
        titlebar.setStyleSheet(f"QFrame {{ background:#cfcfcf; border:1px solid {PANEL_BORDER}; border-bottom:none; }}")
        ttl = QtWidgets.QLabel(" "); ttl.setStyleSheet("QLabel { color:#444; }")
        # 의사 컨트롤 버튼 3개
        def dot(size=12):
            b = QtWidgets.QToolButton(); b.setFixedSize(size,size)
            b.setStyleSheet("QToolButton { background:#e5e5e5; border:1px solid #bdbdbd; border-radius:2px; }")
            b.setEnabled(False); return b
        tb = QtWidgets.QHBoxLayout(titlebar); tb.setContentsMargins(6,2,6,2)
        tb.addWidget(ttl); tb.addStretch(1); tb.addWidget(dot()); tb.addWidget(dot()); tb.addWidget(dot())

        # 내용 패널
        body = QtWidgets.QFrame()
        body.setStyleSheet(f"QFrame {{ background:#ffffff; border:1px solid {PANEL_BORDER}; border-top:none; }}")
        v = QtWidgets.QVBoxLayout(body); v.setContentsMargins(0,0,0,0)  # 내용은 비움(자리만)

        # 전체
        lay = QtWidgets.QVBoxLayout(self); lay.setContentsMargins(0,0,0,0); lay.setSpacing(0)
        lay.addWidget(titlebar); lay.addWidget(body)

# ---------- 좌측 패널 ----------
class LeftPanel(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        # Sim
        sim_rows = [
            ("Episodes", mk_le()), 
            ("Actor learning rate", mk_le()), 
            ("Critic learning rate", mk_le()), 
            ("Soft update period", mk_le()),
            ("Disconunt factor", mk_le()),
            ("Tau", mk_le()), 
            ("Memory capacity", mk_le()),
            ("Max epsilon", mk_le()), 
            ("Min epsilon", mk_le()), 
            ("Epsilon decay", mk_le()), 
            ("Optimizer", mk_le()), 

        ]
        sim_form = mk_form(sim_rows)
        fixed = QtWidgets.QHBoxLayout()
        fixed.addWidget(QtWidgets.QCheckBox("Fixed seed")); fixed.addStretch(1); fixed.addWidget(QtWidgets.QCheckBox("None"))
        fixed_w = QtWidgets.QWidget(); fixed_w.setLayout(fixed)
        rl = QtWidgets.QHBoxLayout()
        reset_btn, load_btn = QtWidgets.QPushButton("Reset"), QtWidgets.QPushButton("Load")
        for b in (reset_btn, load_btn): b.setFixedHeight(26); b.setMinimumWidth(120)
        rl.addWidget(reset_btn); rl.addWidget(load_btn); rl_w = QtWidgets.QWidget(); rl_w.setLayout(rl)
        sim_v = QtWidgets.QVBoxLayout(); sim_v.setSpacing(6)
        sim_v.addWidget(sim_form); sim_v.addWidget(fixed_w); sim_v.addWidget(rl_w)
        sim_host = QtWidgets.QWidget(); sim_host.setLayout(sim_v)
        sim_sec = Section("Simulation Parameters", sim_host)

        # Weather
        wg = QtWidgets.QGridLayout(); wg.setHorizontalSpacing(8); wg.setVerticalSpacing(6); r=0
        wg.addWidget(QtWidgets.QCheckBox("Use real-time weather data"), r,0,1,2); r+=1
        for lab in ["Temperature (°C)","Relative humidity (%)","Rain rate (mm/h)","Snow rate (mm/h)"]:
            l=QtWidgets.QLabel(lab); l.setStyleSheet(f"QLabel {{ color:{LABEL_COLOR}; }}")
            wg.addWidget(l,r,0); wg.addWidget(mk_le(), r,1); r+=1
        w_host = QtWidgets.QWidget(); w_host.setLayout(wg)
        weather_sec = Section("Weather Conditions", w_host)

        # Algo
        labels = ["Improved SA for UAVs' coordinates","Cooling rate","Termination threshold","Iterations",
                  "Stagnation count","Adaptative factor","PSO for UAVs' transmit powers","w1, w2","c1, c2",
                  "Num of particles","Iterations"]
        ap_form = mk_form([(lab, mk_le()) for lab in labels])
        ap_sec   = Section("Algorithm Parameters", ap_form)

        start = QtWidgets.QPushButton("START"); start.setFixedHeight(32); start.setEnabled(False)

        v = QtWidgets.QVBoxLayout(self); v.setContentsMargins(8,8,8,8); v.setSpacing(10)
        v.addWidget(sim_sec); v.addWidget(weather_sec); v.addWidget(ap_sec); v.addWidget(start)
        self.setMinimumWidth(LEFT_MIN_W); self.setMaximumWidth(LEFT_MAX_W)

# ---------- 메인 윈도우 ----------
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Deployment Simulator (Exact Layout with Result Windows)")
        self.resize(1700, 850)
        self.setStyleSheet(
            f"QWidget#root {{ background:{BG_GREY}; }} "
            f"QLabel {{ color:{LABEL_COLOR}; }} "
            "QPushButton { font-weight:600; } "
            "QSplitter::handle { background:#c9c9c9; }"
        )
        root = QtWidgets.QWidget(objectName="root"); self.setCentralWidget(root)

        # 좌/우
        hsplit = QtWidgets.QSplitter(QtCore.Qt.Horizontal); hsplit.setHandleWidth(HANDLE_W_MAIN)
        left = LeftPanel(); hsplit.addWidget(left)

        # 가운데 열: 큰 패널 + 하단 얇은 패널
        center_col = QtWidgets.QSplitter(QtCore.Qt.Vertical); center_col.setHandleWidth(HANDLE_W_COL)
        center_top = white_panel()               # 메인 큰 패널
        center_bot = white_panel(min_h=140)      # 얇은 패널
        center_col.addWidget(center_top); center_col.addWidget(center_bot)

        # 오른쪽 열: 결과창 스타일 2개(상/하)
        right_col = QtWidgets.QSplitter(QtCore.Qt.Vertical); right_col.setHandleWidth(HANDLE_W_COL)
        right_top = ResultWindow(); right_bot = ResultWindow()
        right_col.addWidget(right_top); right_col.addWidget(right_bot)

        # 가운데 + 오른쪽 묶기
        mid_right = QtWidgets.QSplitter(QtCore.Qt.Horizontal); mid_right.setHandleWidth(HANDLE_W_MAIN)
        mid_right.addWidget(center_col); mid_right.addWidget(right_col)

        hsplit.addWidget(mid_right)

        # 비율 (원본 우측 이미지에 맞춤)
        hsplit.setSizes([420, 1240])     # 좌(≈24%) / 우(≈76%)
        mid_right.setSizes([880, 360])   # 가운데(≈71%) / 오른쪽(≈29%)
        center_col.setSizes([600, 200])  # 가운데 상:하 ≈ 75:25
        right_col.setSizes([400, 400])   # 우측 상:하 = 50:50

        outer = QtWidgets.QHBoxLayout(root); outer.setContentsMargins(6,6,6,6); outer.addWidget(hsplit)

    def save_snapshot(self, path):
        self.show(); QtWidgets.QApplication.processEvents()
        pm: QtGui.QPixmap = self.grab(); pm.save(path, "PNG")

# ---------- 실행 ----------
def main():
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.save_snapshot(SAVE_PATH)
    print(f"[OK] layout saved to: {SAVE_PATH}")
    sys.exit(0)

if __name__ == "__main__":
    main()
