from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QGroupBox,
    QCheckBox,
)
import numpy as np


def np_to_qpixmap(frame: np.ndarray) -> QPixmap:
    if frame is None:
        return QPixmap()
    h, w, ch = frame.shape
    bytes_per_line = ch * w
    img = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_BGR888)
    return QPixmap.fromImage(img.copy())


class SettingsPanel(QWidget):
    def __init__(self, on_start, on_stop, on_update_flags, on_reset, parent=None):
        super().__init__(parent)
        self.on_start = on_start
        self.on_stop = on_stop
        self.on_update_flags = on_update_flags
        self.on_reset = on_reset
        self._build()

    def _build(self):
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # Controls
        ctrl_box = QGroupBox("Controls")
        ctrl_layout = QHBoxLayout()
        btn_start = QPushButton("Start Camera")
        btn_stop = QPushButton("Stop Camera")
        btn_reset = QPushButton("Reset Timer/Alarm")
        btn_start.clicked.connect(self.on_start)
        btn_stop.clicked.connect(self.on_stop)
        btn_reset.clicked.connect(self.on_reset)
        ctrl_layout.addWidget(btn_start)
        ctrl_layout.addWidget(btn_stop)
        ctrl_layout.addWidget(btn_reset)
        ctrl_box.setLayout(ctrl_layout)

        # Flags
        flags_box = QGroupBox("Landmarks & Preprocess")
        flags_layout = QVBoxLayout()
        self.cb_tes = QCheckBox("Tesselation")
        self.cb_con = QCheckBox("Contours")
        self.cb_iri = QCheckBox("Irises")
        self.cb_pre = QCheckBox("Preprocess")
        self.cb_tes.setChecked(False)
        self.cb_con.setChecked(True)
        self.cb_iri.setChecked(False)
        self.cb_pre.setChecked(True)
        for cb, key in [
            (self.cb_tes, 'tesselation'),
            (self.cb_con, 'contours'),
            (self.cb_iri, 'irises'),
            (self.cb_pre, 'use_preprocessing'),
        ]:
            cb.stateChanged.connect(lambda _=None, k=key, c=cb: self.on_update_flags({k: c.isChecked()}))
            flags_layout.addWidget(cb)
        flags_box.setLayout(flags_layout)

        layout.addWidget(ctrl_box)
        layout.addWidget(flags_box)


class CameraPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._build()

    def _build(self):
        layout = QVBoxLayout(self)
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setText("No camera feed")
        self.video_label.setMinimumSize(640, 360)
        layout.addWidget(self.video_label)

    def set_frame(self, frame: np.ndarray):
        pix = np_to_qpixmap(frame)
        if not pix.isNull():
            self.video_label.setPixmap(pix)
        else:
            self.video_label.setText("No frame")

