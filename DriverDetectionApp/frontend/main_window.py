import sys
import os
import json
import numpy as np
import threading
from PyQt5.QtWidgets import (QApplication, QMainWindow, QHBoxLayout, QWidget, 
                            QMessageBox, QShortcut)
from PyQt5.QtCore import Qt, pyqtSlot
from PyQt5.QtGui import QKeySequence

from .camera_panel import CameraPanel
from .settings_panel import SettingsPanel
from .alert_manager import AlertManager
from ..backend.camera_thread import CameraThread


class MainWindow(QMainWindow):
    """Cửa sổ chính của ứng dụng"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hệ Thống Nhận Diện Trạng Thái Mất Tập Trung của Tài Xế lái xe")
        self.setMinimumSize(1000, 600)
        
        self.alert_manager = AlertManager(self)
        self.settings_file = "settings.json"

        self.init_ui()

        self.camera_thread = CameraThread(camera_id=1)

        self.camera_thread.frame_ready.connect(self.camera_panel.update_camera)
        self.camera_thread.error_occurred.connect(self.handle_error)
        self.camera_thread.alert_triggered.connect(self.handle_alert)

        self.preload_thread = threading.Thread(target=self.camera_thread.load_ai_modules, daemon=True)
        self.preload_thread.start()

        self.load_settings()
        self.current_eye_threshold = float(self.settings_panel.eye_threshold_spin.value())
        self.setup_shortcuts()
        
    def init_ui(self):
        """Khởi tạo giao diện người dùng"""

        central_widget = QWidget()
        central_widget.setStyleSheet("background-color: white;")
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout(central_widget)
        
        # Panel cài đặt (bên trái)
        self.settings_panel = SettingsPanel(self)
        main_layout.addWidget(self.settings_panel)
        
        # Panel camera (bên phải)
        self.camera_panel = CameraPanel()
        main_layout.addWidget(self.camera_panel)
        
        # Kết nối các nút với hàm xử lý
        self.settings_panel.start_btn.clicked.connect(self.start_camera)
        self.settings_panel.stop_btn.clicked.connect(self.stop_camera)
        # Kết nối switch nhận diện ngáp
        self.settings_panel.yawn_enable_switch.toggled.connect(self.on_yawn_switch_toggled)
        # Kết nối thay đổi số lần ngáp tối đa
        self.settings_panel.yawn_count_spin.valueChanged.connect(self.on_max_yawn_count_changed)
        # Kết nối nút reset yawn count
        self.settings_panel.reset_yawn_btn.clicked.connect(self.on_reset_yawn_count)
        # Kết nối thay đổi thời gian reset yawn count
        self.settings_panel.yawn_reset_spin.valueChanged.connect(self.on_yawn_reset_minutes_changed)
        # Kết nối âm thanh
        self.settings_panel.audio_duration_spin.valueChanged.connect(self.on_audio_duration_changed)
        # Kết nối ngưỡng mắt
        self.settings_panel.eye_threshold_spin.valueChanged.connect(self.on_eye_threshold_changed)

        # Kết nối cài đặt điểm buồn ngủ (MỚI)
        self.settings_panel.w1_spin.valueChanged.connect(self.on_score_setting_changed)
        self.settings_panel.w2_spin.valueChanged.connect(self.on_score_setting_changed)
        self.settings_panel.w3_spin.valueChanged.connect(self.on_score_setting_changed)
        self.settings_panel.pitch_thresh_spin.valueChanged.connect(self.on_score_setting_changed)
        self.settings_panel.yaw_thresh_spin.valueChanged.connect(self.on_score_setting_changed)
        self.settings_panel.alert1_thresh_spin.valueChanged.connect(self.on_score_setting_changed)
        self.settings_panel.alert2_thresh_spin.valueChanged.connect(self.on_score_setting_changed)
        
        # Hiển thị placeholder cho camera
        self.camera_panel.show_placeholder()
        
        # Cập nhật trạng thái
        self.settings_panel.update_status("Sẵn sàng", "success")
        
    def setup_shortcuts(self):
        """Thiết lập phím tắt"""
        # Phím tắt Q để thoát
        quit_shortcut = QShortcut(QKeySequence("Q"), self)
        quit_shortcut.activated.connect(self.close)
        
        # Phím tắt T để bật/tắt tesselation
        t_shortcut = QShortcut(QKeySequence("T"), self)
        t_shortcut.activated.connect(lambda: self.toggle_detection_flag('tesselation'))
        
        # Phím tắt C để bật/tắt contours
        c_shortcut = QShortcut(QKeySequence("C"), self)
        c_shortcut.activated.connect(lambda: self.toggle_detection_flag('contours'))
        
        # Phím tắt I để bật/tắt irises
        i_shortcut = QShortcut(QKeySequence("I"), self)
        i_shortcut.activated.connect(lambda: self.toggle_detection_flag('irises'))
        
        # Phím tắt P để bật/tắt preprocessing
        p_shortcut = QShortcut(QKeySequence("P"), self)
        p_shortcut.activated.connect(lambda: self.toggle_detection_flag('use_preprocessing'))
        
        # Phím tắt R để reset landmarks
        r_shortcut = QShortcut(QKeySequence("R"), self)
        r_shortcut.activated.connect(self.reset_landmarks)
        
        # Phím tắt S để dừng âm thanh
        s_shortcut = QShortcut(QKeySequence("S"), self)
        s_shortcut.activated.connect(self.reset_alarm)
        
    def on_yawn_switch_toggled(self, checked):
        if self.camera_thread is not None:
            self.camera_thread.set_yawn_enabled(checked)
            # Đồng bộ số lần ngáp tối đa và phút reset khi bật ON
            if checked:
                self.camera_thread.set_max_yawn_count(self.settings_panel.yawn_count_spin.value())
                self.camera_thread.set_yawn_reset_minutes(self.settings_panel.yawn_reset_spin.value())
        # Enable/disable nút reset yawn dựa trên trạng thái switch
        self.settings_panel.reset_yawn_btn.setEnabled(checked)
        self.save_settings()

    def on_max_yawn_count_changed(self, value):
        """Cập nhật số lần ngáp tối đa vào DriverMonitor"""
        if self.camera_thread is not None:
            self.camera_thread.set_max_yawn_count(value)
        self.save_settings()

    def on_reset_yawn_count(self):
        """Reset số lần ngáp về 0"""
        if self.camera_thread is not None:
            self.camera_thread.reset_yawn_count()

    def on_yawn_reset_minutes_changed(self, value):
        """Cập nhật thời gian reset yawn count vào DriverMonitor"""
        if self.camera_thread is not None:
            self.camera_thread.set_yawn_reset_minutes(value)
        self.save_settings()

    def on_audio_duration_changed(self, value):
        """Cập nhật thời gian âm thanh tối đa"""
        if self.camera_thread is not None:
            self.camera_thread.set_audio_duration(float(value))
        self.save_settings()

    def on_eye_threshold_changed(self, value):
        """Cập nhật ngưỡng nhắm mắt"""
        self.current_eye_threshold = float(value)  # Lưu trữ giá trị hiện tại
        if self.camera_thread is not None:
            self.camera_thread.set_eye_threshold(float(value))
        self.save_settings()

    def on_score_setting_changed(self):
        """Cập nhật tất cả các thông số của điểm buồn ngủ"""
        if self.camera_thread is not None:
            # Gọi một hàm mới trong camera_thread để cập nhật tất cả các giá trị
            self.camera_thread.update_score_config(
                w1_eye=self.settings_panel.w1_spin.value(),
                w2_yawn=self.settings_panel.w2_spin.value(),
                w3_distraction=self.settings_panel.w3_spin.value(),
                pitch_threshold=self.settings_panel.pitch_thresh_spin.value(),
                yaw_threshold=self.settings_panel.yaw_thresh_spin.value(),
                alert_level_1_threshold=self.settings_panel.alert1_thresh_spin.value(),
                alert_level_2_threshold=self.settings_panel.alert2_thresh_spin.value(),
            )
        self.save_settings()

    def start_camera(self):
        """Bắt đầu camera và xử lý"""
        try:
            if self.camera_thread is None or self.camera_thread.isRunning():
                return

            # --- ĐỒNG BỘ TOÀN BỘ THÔNG SỐ TỪ UI TRƯỚC KHI CHẠY ---
            self.on_score_setting_changed() # Đồng bộ cài đặt điểm
            self.camera_thread.set_eye_threshold(self.current_eye_threshold)
            self.camera_thread.set_yawn_enabled(self.settings_panel.yawn_enable_switch.isChecked())
            self.camera_thread.set_max_yawn_count(self.settings_panel.yawn_count_spin.value())
            self.camera_thread.set_yawn_reset_minutes(self.settings_panel.yawn_reset_spin.value())

            audio_file_full_path = self.settings_panel.audio_file_label.property("full_path")
            if audio_file_full_path:
                self.camera_thread.set_audio_file(audio_file_full_path)
            else:
                default_audio = os.path.join(os.getcwd(), "alarm.wav")
                self.camera_thread.set_audio_file(default_audio)
            self.camera_thread.set_audio_duration(float(self.settings_panel.audio_duration_spin.value()))
            # ------------------------------------------------------

            # Cập nhật giao diện - khóa các settings khi camera đang chạy
            self.settings_panel.start_btn.setEnabled(False)
            self.settings_panel.stop_btn.setEnabled(True)
            self.settings_panel.update_status("Đang xử lý...", "success")
            self.lock_settings_controls(True)
            
            # Bắt đầu thread (vòng lặp run)
            self.camera_thread.start_camera()
            
        except Exception as e:
            self.handle_error(f"Lỗi khởi động camera: {str(e)}")
    
    def stop_camera(self):
        """Dừng camera và xử lý"""
        if self.camera_thread is not None and self.camera_thread.isRunning():
            # Chỉ dừng vòng lặp của luồng, không hủy đối tượng
            self.camera_thread.stop_camera()
            
            # Cập nhật giao diện - mở khóa các settings khi camera dừng
            self.settings_panel.start_btn.setEnabled(True)
            self.settings_panel.stop_btn.setEnabled(False)
            self.settings_panel.update_status("Đã dừng", "warning")

            # Mở khóa tất cả các controls trong settings khi camera dừng
            self.lock_settings_controls(False)
            
            # Hiển thị placeholder
            self.camera_panel.show_placeholder()
    
    def toggle_detection_flag(self, flag_name):
        """Bật/tắt các cờ hiển thị"""
        if self.camera_thread is not None and self.camera_thread.isRunning():
            self.camera_thread.toggle_detection_flag(flag_name)
    
    def reset_landmarks(self):
        """Reset tất cả landmarks"""
        if self.camera_thread is not None and self.camera_thread.isRunning():
            self.camera_thread.reset_landmarks_flags()
    
    def reset_alarm(self):
        """Reset timer và dừng âm thanh"""
        if self.camera_thread is not None and self.camera_thread.isRunning():
            self.camera_thread.reset_timer_and_alarm()
    
    @pyqtSlot(str, str)
    def handle_alert(self, alert_type, message):
        """Xử lý cảnh báo từ camera thread"""
        # Hiển thị cảnh báo tùy chỉnh với animation
        self.alert_manager.show_alert(alert_type, message)

    @pyqtSlot(str)
    def handle_error(self, error_message):
        """Xử lý lỗi từ camera thread"""
        # Cập nhật trạng thái trên UI
        self.settings_panel.update_status("Lỗi AI: xem terminal để biết chi tiết", "error")

        # In lỗi ra terminal thay vì hiện QMessageBox
        try:
            print("=== LỖI KHỞI TẠO AI ===", file=sys.stderr)
            if not isinstance(error_message, str):
                error_message = str(error_message)

            if len(error_message) > 6000:
                print(error_message[:6000] + "\n... (rút gọn)", file=sys.stderr)
            else:
                print(error_message, file=sys.stderr)
            print("=== KẾT THÚC LỖI ===", file=sys.stderr)
        except Exception as e:
            print(f"Lỗi khi in ra terminal: {e}", file=sys.stderr)
        
        
    def lock_settings_controls(self, lock: bool):
        """Khóa/mở khóa các controls trong settings panel"""
        # Khóa các controls âm thanh
        self.settings_panel.audio_browse_btn.setEnabled(not lock)
        self.settings_panel.audio_duration_spin.setEnabled(not lock)

        # Khóa các controls mắt
        self.settings_panel.eye_threshold_spin.setEnabled(not lock)

        # Khóa các controls ngáp khi camera đang chạy
        self.settings_panel.yawn_enable_switch.setEnabled(not lock)  # Switch cũng bị khóa khi chạy
        self.settings_panel.yawn_count_spin.setEnabled(not lock)
        self.settings_panel.yawn_reset_spin.setEnabled(not lock)
        # Nút reset yawn được enable nếu switch ngáp được bật, bất kể camera có chạy hay không
        self.settings_panel.reset_yawn_btn.setEnabled(self.settings_panel.yawn_enable_switch.isChecked())

        # Khóa các controls điểm buồn ngủ (MỚI)
        self.settings_panel.w1_spin.setEnabled(not lock)
        self.settings_panel.w2_spin.setEnabled(not lock)
        self.settings_panel.w3_spin.setEnabled(not lock)
        self.settings_panel.pitch_thresh_spin.setEnabled(not lock)
        self.settings_panel.yaw_thresh_spin.setEnabled(not lock)
        self.settings_panel.alert1_thresh_spin.setEnabled(not lock)
        self.settings_panel.alert2_thresh_spin.setEnabled(not lock)

    def save_settings(self):
        """Lưu cài đặt hiện tại vào file JSON"""
        settings = {
            "eye_threshold": self.settings_panel.eye_threshold_spin.value(),
            "yawn_enabled": self.settings_panel.yawn_enable_switch.isChecked(),
            "yawn_max_count": self.settings_panel.yawn_count_spin.value(),
            "yawn_reset_minutes": self.settings_panel.yawn_reset_spin.value(),
            "audio_duration": self.settings_panel.audio_duration_spin.value(),
            "audio_file": self.settings_panel.audio_file_label.property("full_path"),
            # Cài đặt điểm (MỚI)
            "w1_eye": self.settings_panel.w1_spin.value(),
            "w2_yawn": self.settings_panel.w2_spin.value(),
            "w3_distraction": self.settings_panel.w3_spin.value(),
            "pitch_threshold": self.settings_panel.pitch_thresh_spin.value(),
            "yaw_threshold": self.settings_panel.yaw_thresh_spin.value(),
            "alert_level_1_threshold": self.settings_panel.alert1_thresh_spin.value(),
            "alert_level_2_threshold": self.settings_panel.alert2_thresh_spin.value(),
        }
        try:
            settings_file_path = os.path.abspath(os.path.join(os.path.dirname(__file__),"..","settings.json"))
            with open(settings_file_path, 'w') as f:
                json.dump(settings, f, indent=4)
        except Exception as e:
            print(f"Lỗi khi lưu cài đặt: {e}", file=sys.stderr)

    def load_settings(self):
        """Tải cài đặt từ file JSON và áp dụng vào UI"""
        try:
            settings_file_path = os.path.abspath(os.path.join(os.path.dirname(__file__),"..","settings.json"))
            if os.path.exists(settings_file_path):
                with open(settings_file_path, 'r') as f:
                    settings = json.load(f)
                    
                self.settings_panel.eye_threshold_spin.setValue(settings.get("eye_threshold", 2))
                self.settings_panel.yawn_enable_switch.setChecked(settings.get("yawn_enabled", False))
                self.settings_panel.yawn_count_spin.setValue(settings.get("yawn_max_count", 5))
                self.settings_panel.yawn_reset_spin.setValue(settings.get("yawn_reset_minutes", 10))
                self.settings_panel.audio_duration_spin.setValue(settings.get("audio_duration", 10))
                
                audio_file = settings.get("audio_file")
                if audio_file and os.path.exists(audio_file):
                    self.settings_panel.audio_file_label.setText(os.path.basename(audio_file))
                    self.settings_panel.audio_file_label.setProperty("full_path", audio_file)

                # Tải cài đặt điểm (MỚI)
                self.settings_panel.w1_spin.setValue(settings.get("w1_eye", 50))
                self.settings_panel.w2_spin.setValue(settings.get("w2_yawn", 20))
                self.settings_panel.w3_spin.setValue(settings.get("w3_distraction", 30))
                self.settings_panel.pitch_thresh_spin.setValue(settings.get("pitch_threshold", 20))
                self.settings_panel.yaw_thresh_spin.setValue(settings.get("yaw_threshold", 30))
                self.settings_panel.alert1_thresh_spin.setValue(settings.get("alert_level_1_threshold", 40))
                self.settings_panel.alert2_thresh_spin.setValue(settings.get("alert_level_2_threshold", 70))

        except Exception as e:
            print(f"Lỗi khi tải cài đặt: {e}", file=sys.stderr)

    def resizeEvent(self, event):
        """Xử lý sự kiện thay đổi kích thước cửa sổ"""
        # Cập nhật vị trí các alert khi cửa sổ thay đổi kích thước
        self.alert_manager.update_positions_on_resize()
        super().resizeEvent(event)

    def closeEvent(self, event):
        """Xử lý sự kiện đóng cửa sổ"""
        self.save_settings()
        # Dừng camera thread nếu đang chạy
        if self.camera_thread is not None and self.camera_thread.isRunning():
            self.camera_thread.stop_camera()

        # Chấp nhận sự kiện đóng
        event.accept()


def main():
    """Hàm main để chạy ứng dụng"""
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
