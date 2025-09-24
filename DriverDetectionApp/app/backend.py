import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal

from DriverDetection.driver_monitor import DriverMonitor


class CameraWorker(QThread):
    frame_ready = pyqtSignal(np.ndarray)
    error = pyqtSignal(str)

    def __init__(self, camera_id=0, parent=None):
        super().__init__(parent)
        self.camera_id = camera_id
        self._running = False
        self.cap = None
        self.monitor = DriverMonitor()

    def run(self):
        try:
            self.cap = cv2.VideoCapture(self.camera_id)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self._running = True

            while self._running:
                ret, frame = self.cap.read()
                if not ret:
                    self.error.emit("Cannot read from camera")
                    break

                frame = cv2.flip(frame, 1)
                try:
                    results = self.monitor.process_frame(frame)
                    self.frame_ready.emit(results['frame'])
                except Exception as e:  # noqa: BLE001
                    self.error.emit(str(e))
                    self.frame_ready.emit(frame)

        finally:
            self._cleanup()

    def stop(self):
        self._running = False

    def _cleanup(self):
        try:
            if self.monitor:
                self.monitor.cleanup()
            if self.cap:
                self.cap.release()
        except Exception:
            pass

    # Public config methods
    def update_config(self, **kwargs):
        self.monitor.update_config(**kwargs)

    def reset_timer(self):
        self.monitor.reset_timer()

    def stop_alarm(self):
        try:
            # Access private method safely via exposed API
            self.monitor._DriverMonitor__stop_alarm()  # type: ignore[attr-defined]
        except Exception:
            pass

