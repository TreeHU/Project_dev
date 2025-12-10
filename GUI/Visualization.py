# layout_builder_v3.py
# ------------------------------------------------------------
# 가운데/오른쪽 패널을 스크린샷과 동일하게:
# - 오른쪽 패널: 타이틀바(의사 최소화/최대화/닫기 버튼) 포함
# - 가운데:오른쪽 폭 비율/상하 비율 조정
# 저장 경로: ./Project/GUI/layout.png
# ------------------------------------------------------------
import os, sys
if not os.environ.get("DISPLAY"):
    os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PyQt5 import QtCore, QtGui, QtWidgets

# -----------------------------
# 경로
# -----------------------------
SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
SAVE_DIR   = os.path.join(SCRIPT_DIR, "Project", "GUI")
os.makedirs(SAVE_DIR, exist_ok=True)
SAVE_PATH  = os.path.join(SAVE_DIR, "layout.png")

# -----------------------------
# 색상/크기 상수
# -----------------------------
BG_GREY      = "#d9d9d9"      # 전체 배경
PANEL_BORDER = "#b7b7b7"      # 얇은 테두리
HEADER_BLUE  = "#1b3c89"      # 그룹 헤더
HEADER_TEXT  = "#ffffff"
LABEL_COLOR  = "#111111"

LEFT_WIDTH_MIN = 430
LEFT_WIDTH_MAX = 450
HANDLE_W_MAIN  = 8
HANDLE_W_COL   = 6

# -----------------------------
# 공통 위젯
# -----------------------------
def make_lineedit():
    le = QtWidgets.QLineEdit()
    le.setFixedWidth(160)
    le.setMinimumHeight(18)
    return le

def make_combo(items):
    cb = QtWidgets.QComboBox()
    cb.addItems(items)
    cb.setFixedWidth(160)
    cb.setMinimumHeight(20)
    return cb

def make_load_button():
    btn = QtWidgets.QPushButton("Load")
    btn.setFixedHeight(22)
    btn.setFixedWidth(80)
    return btn

class HeaderBar(QtWidgets.QFrame):
    """좌측 파란색 제목 막대"""
    def __init__(self, title):
        super().__init__()
        self.setFixedHeight(28)
        self.setStyleSheet(
            f"QFrame {{ background:{HEADER_BLUE}; border:1px solid {HEADER_BLUE}; border-radius:4px; }}"
        )
        lab = QtWidgets.QLabel(title)
        lab.setStyleSheet(f"QLabel {{ color:{HEADER_TEXT}; font-weight:600; }}")
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(8, 2, 8, 2)
        lay.addWidget(lab)
        lay.addStretch(1)

class GroupSection(QtWidgets.QWidget):
    """좌측 섹션: 파란 헤더 + 회색 내용 래퍼"""
    def __init__(self, title, content_widget):
        super().__init__()
        header = HeaderBar(title)
        content_wrap = QtWidgets.QFrame()
        content_wrap.setStyleSheet(f"QFrame {{ background:{BG_GREY}; }}")
        v = QtWidgets.QVBoxLayout(content_wrap)
        v.setContentsMargins(6, 6, 6, 6)
        v.addWidget(content_widget)

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 4, 0, 0)
        lay.setSpacing(6)
        lay.addWidget(header)
        lay.addWidget(content_wrap)

def make_form(rows):
    """라벨-입력 2열 Grid 폼"""
    grid = QtWidgets.QGridLayout()
    grid.setHorizontalSpacing(8)
    grid.setVerticalSpacing(6)
    r = 0
    for label, widget in rows:
        lab = QtWidgets.QLabel(label)
        lab.setStyleSheet(f"QLabel {{ color:{LABEL_COLOR}; }}")
        grid.addWidget(lab, r, 0)
        grid.addWidget(widget, r, 1)
        r += 1
    w = QtWidgets.QWidget()
    w.setLayout(grid)
    return w

def thin_panel(min_h=None):
    """흰 배경 + 얇은 테두리 패널 (중앙 칸에 사용)"""
    fr = QtWidgets.QFrame()
    fr.setFrameShape(QtWidgets.QFrame.Panel)
    fr.setFrameShadow(QtWidgets.QFrame.Plain)
    fr.setLineWidth(1)
    fr.setStyleSheet(f"QFrame {{ background:#ffffff; border:1px solid {PANEL_BORDER}; }}")
    if min_h:
        fr.setMinimumHeight(min_h)
    return fr

class ResultWindow(QtWidgets.QWidget):
    """
    오른쪽 패널: 스샷처럼 타이틀바(작은 컨트롤 버튼) + 컨텐츠 영역.
    """
    def __init__(self):
        super().__init__()
        # 타이틀바
        titlebar = QtWidgets.QFrame()
        titlebar.setFixedHeight(22)
        titlebar.setStyleSheet(
            f"QFrame {{ background:#cfcfcf; border:1px solid {PANEL_BORDER}; border-bottom:none; }}"
        )
        # 의사 컨트롤 버튼
        def dot():
            b = QtWidgets.QToolButton()
            b.setFixedSize(12, 12)
            b.setStyleSheet("QToolButton { background:#e5e5e5; border:1px solid #bdbdbd; border-radius:2px; }")
            b.setEnabled(False)
            return b
        tb = QtWidgets.QHBoxLayout(titlebar)
        tb.setContentsMargins(6, 2, 6, 2)
        tb.addStretch(1)
        tb.addWidget(dot()); tb.addSpacing(4); tb.addWidget(dot()); tb.addSpacing(4); tb.addWidget(dot())

        # 본문
        body = QtWidgets.QFrame()
        body.setStyleSheet(
            f"QFrame {{ background:#ffffff; border:1px solid {PANEL_BORDER}; border-top:none; }}"
        )
        body_l = QtWidgets.QVBoxLayout(body)
        body_l.setContentsMargins(0, 0, 0, 0)

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(titlebar)
        lay.addWidget(body)

# -----------------------------
# 좌측 패널
# -----------------------------
class LeftPanel(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        # Simulation Parameters
        sim_rows = [
            ("Episodes", make_lineedit()), 
            ("Actor learning rate", make_lineedit()),
            ("Critic learning rate", make_lineedit()),
            ("Soft update period", make_lineedit()), 
            ("Disconunt factor", make_combo([str(i) for i in range(1, 11)])),
            ("Tau", make_lineedit()),
            ("Memory capacity", make_lineedit()),
            ("Max epsilon", make_lineedit()),
            ("Min epsilon", make_lineedit()), 
            ("Epsilon decay", make_lineedit()), 
            ("Optimizer", make_combo(["Adam","Adam","Adam","Adam"])),
            ("Loss function", make_combo(["Mse","Mse","Mse","Mse"])),
        ]
        sim_form = make_form(sim_rows)

        # Fixed seed/None + Reset/Load
        fixed = QtWidgets.QHBoxLayout()
        fixed.addWidget(QtWidgets.QCheckBox("Fixed seed"))
        fixed.addStretch(1)
        fixed.addWidget(QtWidgets.QCheckBox("None"))
        fixed_w = QtWidgets.QWidget(); fixed_w.setLayout(fixed)

        rl = QtWidgets.QHBoxLayout()
        reset_btn = QtWidgets.QPushButton("Reset")
        load_btn  = QtWidgets.QPushButton("Load")
        for b in (reset_btn, load_btn):
            b.setFixedHeight(26)
            b.setMinimumWidth(120)
        rl.addWidget(reset_btn); rl.addWidget(load_btn)
        rl_w = QtWidgets.QWidget(); rl_w.setLayout(rl)

        sim_v = QtWidgets.QVBoxLayout()
        sim_v.setSpacing(6)
        sim_v.addWidget(sim_form)
        sim_v.addWidget(fixed_w)
        sim_v.addWidget(rl_w)
        sim_vw = QtWidgets.QWidget(); sim_vw.setLayout(sim_v)
        sim_sec = GroupSection("Simulation Parameters", sim_vw)

        # Weather Conditions (Load 버튼 4개)
        wc_form_grid = QtWidgets.QGridLayout()
        wc_form_grid.setHorizontalSpacing(8); wc_form_grid.setVerticalSpacing(6)
        r = 0
        cbx = QtWidgets.QCheckBox("Use real-time weather data")
        wc_form_grid.addWidget(cbx, r, 0, 1, 2); r += 1

        wc_items = [
            "Temperature map (°C)",
            "Humidity map (%)",
            "Fine dust map (µg/m³)",
            "Cloud map",
        ]
        for label in wc_items:
            lab = QtWidgets.QLabel(label); lab.setStyleSheet(f"QLabel {{ color:{LABEL_COLOR}; }}")
            wc_form_grid.addWidget(lab, r, 0)
            wc_form_grid.addWidget(make_load_button(), r, 1)
            r += 1

        wc_form = QtWidgets.QWidget(); wc_form.setLayout(wc_form_grid)
        wc_sec = GroupSection("Weather Conditions", wc_form)

        # Algorithm Parameters
        labels = [
            "Encoder nodes", "LSTM nodes", "Batch nodes", "Decoder nodes",
            "Actor moment", "Activation"
        ]
        ap_form = make_form([(lab, make_lineedit()) for lab in labels])
        ap_sec = GroupSection("Algorithm Parameters", ap_form)

        # START 버튼(비활성)
        start = QtWidgets.QPushButton("START")
        start.setFixedHeight(32)
        start.setEnabled(False)

        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(8, 8, 8, 8)
        v.setSpacing(10)
        v.addWidget(sim_sec)
        v.addWidget(wc_sec)
        v.addWidget(ap_sec)
        v.addWidget(start)

        self.setMinimumWidth(LEFT_WIDTH_MIN)
        self.setMaximumWidth(LEFT_WIDTH_MAX)

# -----------------------------
# 메인 윈도우
# -----------------------------
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Deployment Simulator (Exact Layout)")
        self.resize(1700, 850)

        self.setStyleSheet(
            f"QWidget#root {{ background:{BG_GREY}; }} "
            f"QLabel {{ color:{LABEL_COLOR}; }} "
            "QPushButton { font-weight:600; } "
            f"QSplitter::handle {{ background:{'#c9c9c9'}; }}"
        )

        root = QtWidgets.QWidget(objectName="root")
        self.setCentralWidget(root)

        # 좌/우 메인 스플리터
        hsplit = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        hsplit.setHandleWidth(HANDLE_W_MAIN)

        # 좌측
        left = LeftPanel()
        hsplit.addWidget(left)

        # 가운데(상: 큰 패널 / 하: 얇은 패널)
        center_col = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        center_col.setHandleWidth(HANDLE_W_COL)
        center_top = thin_panel()
        center_bot = thin_panel(min_h=140)
        center_col.addWidget(center_top)
        center_col.addWidget(center_bot)
        
        # 오른쪽(상/하) — 스샷처럼 타이틀바 포함
        right_col = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        #right_col.setMinimumWidth(RIGHT_WIDTH // 2)
        right_col.setHandleWidth(HANDLE_W_COL)
        right_top = ResultWindow()
        right_bot = ResultWindow()
        right_col.addWidget(right_top)
        right_col.addWidget(right_bot)

        # 가운데+오른쪽
        mid_right = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        mid_right.setHandleWidth(HANDLE_W_MAIN)
        mid_right.addWidget(center_col)
        mid_right.addWidget(right_col)

        hsplit.addWidget(mid_right)

        # ===== 비율 조정 (스샷 느낌으로 중앙이 더 넓고, 우측은 타이틀바 달린 2칸) =====
        hsplit.setSizes([420, 1240])        # 좌 / (가운데+오른쪽)
        #mid_right.setSizes([900, 340])      # 가운데 : 오른쪽  ≈ 72 : 28
        center_col.setSizes([610, 190])     # 중앙 상 : 하     ≈ 76 : 24
        right_col.setSizes([360, 360])      # 우측 상 : 하     =  50 : 50

                # 교체 — 오른쪽 폭만 키움 (예: 680px)
        RIGHT_WIDTH = 600
        TOTAL_WIDTH = 1500                     # hsplit.setSizes([420, 1240]) 에서 두 번째 값과 일치
        mid_right.setSizes([TOTAL_WIDTH - RIGHT_WIDTH, RIGHT_WIDTH])
        mid_right.setStretchFactor(0, 1)   # 가운데
        mid_right.setStretchFactor(1, 2)   # 오른쪽(더 넓게 유지)

        outer = QtWidgets.QHBoxLayout(root)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.addWidget(hsplit)

    def save_snapshot(self, path):
        self.show()
        QtWidgets.QApplication.processEvents()
        pm: QtGui.QPixmap = self.grab()
        pm.save(path, "PNG")

# -----------------------------
# 실행
# -----------------------------
def main():
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.save_snapshot(SAVE_PATH)
    print(f"[OK] layout saved to: {SAVE_PATH}")
    sys.exit(0)

if __name__ == "__main__":
    main()
