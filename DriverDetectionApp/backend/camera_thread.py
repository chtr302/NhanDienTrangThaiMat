"""
Camera Thread  camera và kết nối với DriverMonitor
"""

import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import QMessageBox
import traceback
import os
import sys
import platform
from pathlib import Path
from importlib.util import find_spec
import importlib

from .audio_manager import AudioManager, AlertType


def _prepare_mediapipe_environment():
    """Ensure Mediapipe native DLLs are discoverable at runtime."""
    try:
        spec = find_spec("mediapipe")
        if spec and spec.origin:
            mp_root = Path(spec.origin).parent
            dll_dir = mp_root / "python"
            if dll_dir.exists():
                if hasattr(os, "add_dll_directory"):
                    os.add_dll_directory(str(dll_dir))
                else:
                    os.environ["PATH"] = str(dll_dir) + os.pathsep + os.environ.get("PATH", "")
                return None
            return f"Mediapipe python directory not found at {dll_dir}"
        return "Unable to locate mediapipe package to configure DLL path"
    except Exception as exc:
        return f"Failed to prepare mediapipe DLL directory: {exc}"
    return None


def _gather_import_diagnostics():
    """Try importing key dependencies to pinpoint failing module."""
    modules = [
        "numpy",
        "cv2",
        "mediapipe",
        "tensorflow",
        "keras",
        "PyQt5",
        "PyQt5.QtCore",
    ]
    lines = []
    for name in modules:
        try:
            mod = importlib.import_module(name)
            version = getattr(mod, "__version__", None)
            path = getattr(mod, "__file__", None)
            lines.append(f"OK {name} version={version} file={path}")
        except Exception as exc:
            lines.append(f"FAIL {name}: {exc}")
    return lines


def _diagnose_mediapipe_bindings():
    """Collect details about MediaPipe installation and native bindings."""
    info = []
    try:
        spec = find_spec("mediapipe")
        origin = spec.origin if spec and getattr(spec, "origin", None) else None
        mp_root = Path(origin).parent if origin else None
        info.append(f"mediapipe origin: {origin}")
        if mp_root:
            py_dir = mp_root / "python"
            info.append(f"mediapipe/python dir: {py_dir}")
            if py_dir.exists():
                try:
                    entries = list(py_dir.glob("_framework_bindings*.pyd"))
                    info.append("bindings present: " + (", ".join(e.name for e in entries) or "<none>"))
                except Exception:
                    pass
    except Exception:
        pass
    return info


class CameraThread(QThread):
    """Thread xá»­ lÃ½ camera vÃ  káº¿t ná»‘i vá»›i DriverMonitor"""

    frame_ready = pyqtSignal(np.ndarray)
    error_occurred = pyqtSignal(str)
    alert_triggered = pyqtSignal(str, str)  # (alert_type, message)

    def __init__(self, camera_id=0):
        super().__init__()
        self.camera_id = camera_id
        self.running = False
        self.cap = None
        self.monitor = None
        self._ai_loaded = False
        self._ai_error_reported = False

        # Yawn detection state
        self.yawn_enabled = False
        self._pending_max_yawn_count = 5
        self._pending_yawn_reset_minutes = 10
        
        # Eye threshold
        self._pending_eye_threshold = 2.0  # Default threshold

        # Audio manager for alerts
        self.audio_manager = AudioManager()
        self.last_drowsy_alert = False
        self.last_yawn_alert = False
        self.last_max_yawn_alert = False

    # ---- Control helpers (thread-safe enough for simple flags) ----
    def toggle_detection_flag(self, flag_name: str):
        """Toggle detection flag (tesselation, contours, irises)"""
        try:
            if self.monitor is None:
                return
            current = bool(self.monitor.detection_config.get(flag_name, False))
            self.monitor.detection_config[flag_name] = not current
        except Exception as e:
            pass

    def reset_landmarks_flags(self):
        """Reset all landmark flags to True"""
        try:
            if self.monitor is None:
                return
            self.monitor.detection_config.update({
                'tesselation': True,
                'contours': True,
                'irises': True
            })
        except Exception as e:
            pass

    def reset_timer_and_alarm(self):
        """Reset timer and alarm"""
        try:
            if self.monitor is None:
                return
            self.monitor.reset_timer()
        except Exception as e:
            pass

    def load_ai_modules(self):
        """Lazy load AI modules"""
        if self._ai_loaded:
            return True

        proj_root = None
        dll_hint = None

        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            proj_root = os.path.dirname(os.path.dirname(current_dir))
            if proj_root not in sys.path:
                sys.path.append(proj_root)

            dll_hint = _prepare_mediapipe_environment()

            from DriverDetection.driver_monitor import DriverMonitor

            # Test model loading
            self.monitor = DriverMonitor()
            # ĐỒNG BỘ TOÀN BỘ THÔNG SỐ NHẬN DIỆN NGÁP VÀO monitor
            self.monitor.update_config(
                enable_yawn_detection=self.yawn_enabled,
                max_yawn_count=self._pending_max_yawn_count,
                yawn_reset_minutes=self._pending_yawn_reset_minutes,
                sleep_threshold=self._pending_eye_threshold  # Apply pending eye threshold
            )
            self._ai_loaded = True
            return True

        except Exception as e:
            # Build a detailed diagnostic message with traceback and common hints
            tb = traceback.format_exc()
            msg_lines = [
                "AI modules failed to load",
                f"Exception: {str(e)}",
                "\nTraceback:",
                tb,
            ]

            runtime_info = [
                f"Python executable: {sys.executable}",
                f"Python version: {sys.version}",
                f"Working directory: {os.getcwd()}",
                f"Platform: {platform.platform()} ({platform.machine()})",
            ]
            msg_lines.append("Runtime info:\n" + "\n".join(runtime_info))

            sys_path_preview = [
                f"sys.path[{i}]: {entry}"
                for i, entry in enumerate(sys.path[:5])
            ]
            msg_lines.append("sys.path preview:\n" + "\n".join(sys_path_preview))

            path_entries = os.environ.get("PATH", "").split(os.pathsep)
            path_preview = [
                f"PATH[{i}]: {entry}"
                for i, entry in enumerate(path_entries[:6])
            ]
            msg_lines.append("PATH preview:\n" + "\n".join(path_preview))

            if dll_hint:
                msg_lines.append(f"Mediapipe DLL hint: {dll_hint}")

            # Import diagnostics (which module actually fails?)
            try:
                import_diag = _gather_import_diagnostics()
                msg_lines.append("Import checks:\n" + "\n".join(import_diag))
            except Exception:
                pass

            # Extra MediaPipe specifics if error message references it
            if "mediapipe" in tb.lower() or "_framework_bindings" in tb:
                mp_details = _diagnose_mediapipe_bindings()
                if mp_details:
                    msg_lines.append("Mediapipe details:\n" + "\n".join(mp_details))

                msg_lines.append(
                    "Hints (Windows): Ensure 'Microsoft Visual C++ 2015-2022 Redistributable (x64)' is installed,\n"
                    "and that Python/mediapipe wheel versions are compatible."
                )

            try:
                if proj_root is None:
                    proj_root = os.path.dirname(
                        os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                eye_best = os.path.join(proj_root, 'models', 'best_model_first_try.keras')
                eye_main = os.path.join(proj_root, 'models', 'model.keras')
                yawn_path = os.path.join(proj_root, 'models', 'yawn_model.keras')
                hints = [
                    f"Exists {eye_best}: {os.path.exists(eye_best)}",
                    f"Exists {eye_main}: {os.path.exists(eye_main)}",
                    f"Exists {yawn_path}: {os.path.exists(yawn_path)}",
                ]
                msg_lines.append("Model file checks:\n" + "\n".join(hints))
            except Exception:
                pass

            detailed_msg = "\n".join(msg_lines)

            # Log to file for later inspection (project root)
            try:
                log_base = proj_root or os.getcwd()
                log_path = os.path.join(log_base, 'startup_errors.log')
                with open(log_path, 'a', encoding='utf-8') as f:
                    f.write("\n=== AI Load Error ===\n")
                    f.write(detailed_msg)
                    f.write("\n")
            except Exception:
                pass

            # Emit once to UI to avoid dialog spam
            if not self._ai_error_reported:
                self._ai_error_reported = True
                self.error_occurred.emit(detailed_msg)
            return False

    def load_yawn_processor(self):
        """Check if YawnProcessor is available in DriverMonitor"""
        if self.monitor is None:
            return False
        return hasattr(self.monitor.yawn_processor, 'available') and self.monitor.yawn_processor.available

    def set_yawn_enabled(self, enabled: bool):
        self.yawn_enabled = enabled
        if self.monitor is not None:
            self.monitor.update_config(enable_yawn_detection=enabled)

    def set_max_yawn_count(self, value):
        self._pending_max_yawn_count = value
        if self.monitor is not None:
            self.monitor.update_config(max_yawn_count=value)

    def reset_yawn_count(self):
        if self.monitor is not None:
            self.monitor.reset_yawn_count()

    def set_yawn_reset_minutes(self, value):
        self._pending_yawn_reset_minutes = value
        if self.monitor is not None:
            self.monitor.update_config(yawn_reset_minutes=value)

    def set_audio_file(self, file_path: str):
        """Set audio file for alerts"""
        self.audio_manager.set_audio_file(file_path, AlertType.GENERAL)

    def set_audio_duration(self, duration: float):
        """Set maximum audio duration"""
        self.audio_manager.set_max_duration(duration)

    def set_eye_threshold(self, threshold: float):
        """Set eye closure threshold (seconds)"""
        self._pending_eye_threshold = threshold  # Lưu trữ giá trị để apply khi monitor được tạo
        if self.monitor is not None:
            self.monitor.update_config(sleep_threshold=threshold)

    def start_camera(self):
        """Start camera thread"""
        # Attempt to load AI early to surface errors before entering run loop
        self.load_ai_modules()
        # Luôn đồng bộ trạng thái yawn_enabled vào monitor
        if self.monitor is not None:
            self.monitor.update_config(enable_yawn_detection=self.yawn_enabled)
        self.running = True
        if not self.isRunning():
            self.start()

    def stop_camera(self):
        """Stop camera thread"""
        self.running = False
        if self.cap:
            self.cap.release()
        self.wait()

    def run(self):
        """Main camera processing loop"""
        # Try to load AI modules
        ai_available = self.load_ai_modules()

        self.cap = cv2.VideoCapture(self.camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        last_yawn_enabled = self.yawn_enabled  # Track last state

        while self.running:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                # Chỉ update trạng thái yawn_enabled nếu có thay đổi
                if self.monitor is not None and last_yawn_enabled != self.yawn_enabled:
                    self.monitor.update_config(enable_yawn_detection=self.yawn_enabled)
                    last_yawn_enabled = self.yawn_enabled
                if ai_available and self.monitor:
                    try:
                        # Process frame with DriverMonitor (Ä‘Ã£ cÃ³ Ä‘áº§y Ä‘á»§ logic)
                        results = self.monitor.process_frame(frame)

                        # Check for alerts and play audio + show dialog
                        if 'is_drowsy' in results and results['is_drowsy']:
                            if not self.last_drowsy_alert:
                                self.audio_manager.play_alert(AlertType.DROWSY)
                                self.alert_triggered.emit("DROWSY", "CẢNH BÁO: Phát hiện buồn ngủ! Hãy nghỉ ngơi.")
                                self.last_drowsy_alert = True
                        else:
                            self.last_drowsy_alert = False

                        if 'yawn_alert' in results and results['yawn_alert']:
                            if not self.last_yawn_alert:
                                self.audio_manager.play_alert(AlertType.YAWN)
                                self.alert_triggered.emit("YAWN", "CẢNH BÁO: Phát hiện ngáp! Hãy nghỉ ngơi.")
                                self.last_yawn_alert = True
                        else:
                            self.last_yawn_alert = False

                        # Kiểm tra cảnh báo vượt quá số lần ngáp tối đa
                        if ('yawn_count' in results and 'max_yawn_count' in results and
                            results['yawn_count'] >= results['max_yawn_count']):
                            # Chỉ cảnh báo một lần khi vượt quá, không cảnh báo liên tục
                            if not self.last_max_yawn_alert:
                                self.audio_manager.play_alert(AlertType.GENERAL)
                                self.alert_triggered.emit("GENERAL",
                                    f"CẢNH BÁO: Bạn đã ngáp quá {results['max_yawn_count']} lần! Hãy nghỉ ngơi.")
                                self.last_max_yawn_alert = True
                        else:
                            # Reset flag khi không còn vượt quá
                            self.last_max_yawn_alert = False

                        # Emit processed frame
                        self.frame_ready.emit(results['frame'])

                    except Exception as e:
                        # Fallback to basic frame
                        cv2.putText(frame, f"AI Error: {str(e)[:50]}",
                                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        self.frame_ready.emit(frame)
                else:
                    # AI not available -> emit raw frame (camera preview)
                    self.frame_ready.emit(frame)
            self.msleep(33)  # ~30 FPS

        if self.cap:
            self.cap.release()