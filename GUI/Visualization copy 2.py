# layout_builder_v3.py
# ------------------------------------------------------------
# "원하는 layout" 정확 재현: 색상/간격/비율 정밀 조정 + layout.png 저장
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

LEFT_WIDTH_MIN = 430           # 좌측 폭 (원본 스샷 감성)
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
    """파란색 제목 막대 (그룹 타이틀)"""
    def __init__(self, title):
        super().__init__()
        self.setFixedHeight(28)
        self.setStyleSheet(f"QFrame {{ background:{HEADER_BLUE}; border:1px solid {HEADER_BLUE}; "
                           "border-radius:4px; }}")
        lab = QtWidgets.QLabel(title)
        lab.setStyleSheet(f"QLabel {{ color:{HEADER_TEXT}; font-weight:600; }}")
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(8, 2, 8, 2)
        lay.addWidget(lab)
        lay.addStretch(1)

class GroupSection(QtWidgets.QWidget):
    """원하는 레이아웃처럼: 파란 헤더 + 내용 영역(회색 배경, 테두리 없음 느낌)"""
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
    """흰 배경 + 얇은 테두리 패널 (중앙/오른쪽 칸)"""
    fr = QtWidgets.QFrame()
    fr.setFrameShape(QtWidgets.QFrame.Panel)
    fr.setFrameShadow(QtWidgets.QFrame.Plain)
    fr.setLineWidth(1)
    fr.setStyleSheet(f"QFrame {{ background:#ffffff; border:1px solid {PANEL_BORDER}; }}")
    if min_h:
        fr.setMinimumHeight(min_h)
    return fr

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

        # Weather Conditions (Load 버튼으로 교체)
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
            "The number of Encoder nodes' The number of LSTM nodes", "The number of Batch nodes", "The number of Decoder nodes",
            "Actor moment", "Activation"
        ]
        ap_form = make_form([(lab, make_lineedit()) for lab in labels])
        ap_sec = GroupSection("Algorithm Parameters", ap_form)

        # START 버튼(비활성 회색)
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

        # 전반 스타일
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

        # 가운데(상: 큰 패널 / 하: 낮은 패널)
        center_col = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        center_col.setHandleWidth(HANDLE_W_COL)
        center_top = thin_panel()
        center_bot = thin_panel(min_h=140)
        center_col.addWidget(center_top)
        center_col.addWidget(center_bot)

        # 오른쪽(상/하 동일)
        right_col = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        right_col.setHandleWidth(HANDLE_W_COL)
        right_top = thin_panel()
        right_bot = thin_panel()
        right_col.addWidget(right_top)
        right_col.addWidget(right_bot)

        # 가운데+오른쪽
        mid_right = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        mid_right.setHandleWidth(HANDLE_W_MAIN)
        mid_right.addWidget(center_col)
        mid_right.addWidget(right_col)

        hsplit.addWidget(mid_right)

        # 비율 조정
        hsplit.setSizes([420, 1240])     # 좌 전체 / (가운데+오른쪽)
        # 🔸 오른쪽 폭을 "기존의 약 2배"로: 이전 [900, 340] → [560, 680]
        mid_right.setSizes([620, 680])
        center_col.setSizes([600, 200])  # 가운데 상:하 = 75:25
        right_col.setSizes([400, 400])   # 오른쪽 상:하 = 50:50

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
