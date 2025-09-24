from PyQt5.QtWidgets import QMainWindow, QWidget, QHBoxLayout, QApplication
from PyQt5.QtCore import Qt
import sys

from .backend import CameraWorker
from .panels import SettingsPanel, CameraPanel


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Driver Monitoring App")
        self.worker = None

        central = QWidget()
        layout = QHBoxLayout(central)

        # Panels
        self.settings = SettingsPanel(
            on_start=self.start_camera,
            on_stop=self.stop_camera,
            on_update_flags=self.update_flags,
            on_reset=self.reset_timer,
        )
        self.camera = CameraPanel()

        layout.addWidget(self.settings, 0, Qt.AlignmentFlag.AlignTop)
        layout.addWidget(self.camera, 1)

        self.setCentralWidget(central)
        self.resize(1100, 600)

    def start_camera(self):
        if self.worker and self.worker.isRunning():
            return
        self.worker = CameraWorker(camera_id=0)
        self.worker.frame_ready.connect(self.camera.set_frame)
        self.worker.start()

    def stop_camera(self):
        if self.worker:
            self.worker.stop()
            self.worker.wait(500)
            self.worker = None

    def update_flags(self, flags: dict):
        if self.worker:
            self.worker.update_config(**flags)

    def reset_timer(self):
        if self.worker:
            self.worker.reset_timer()


def run_app():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())

