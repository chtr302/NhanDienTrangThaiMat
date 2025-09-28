"""
Main Window cho ứng dụng nhận diện ngủ gật
"""

import sys
import os
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QHBoxLayout, QWidget, 
                            QMessageBox, QShortcut)
from PyQt5.QtCore import Qt, pyqtSlot
from PyQt5.QtGui import QKeySequence

from .camera_panel import CameraPanel
from .settings_panel import SettingsPanel
from ..backend.camera_thread import CameraThread


class MainWindow(QMainWindow):
    """Cửa sổ chính của ứng dụng"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hệ Thống Nhận Diện Ngủ Gật")
        self.setMinimumSize(1000, 600)
        
        self.camera_thread = None
        self.init_ui()
        self.setup_shortcuts()
        
    def init_ui(self):
        """Khởi tạo giao diện người dùng"""
        # Widget chính
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Layout chính
        main_layout = QHBoxLayout(central_widget)
        
        # Panel cài đặt (bên trái)
        self.settings_panel = SettingsPanel()
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
        # Disable/enable chỉnh số lần ngáp tối đa và thời gian reset khi bật/tắt nhận diện ngáp
        self.settings_panel.yawn_count_spin.setEnabled(not checked)
        self.settings_panel.yawn_reset_spin.setEnabled(not checked)

    def on_max_yawn_count_changed(self, value):
        """Cập nhật số lần ngáp tối đa vào DriverMonitor"""
        if self.camera_thread is not None:
            self.camera_thread.set_max_yawn_count(value)

    def on_reset_yawn_count(self):
        """Reset số lần ngáp về 0"""
        if self.camera_thread is not None:
            self.camera_thread.reset_yawn_count()

    def on_yawn_reset_minutes_changed(self, value):
        """Cập nhật thời gian reset yawn count vào DriverMonitor"""
        if self.camera_thread is not None:
            self.camera_thread.set_yawn_reset_minutes(value)

    def start_camera(self):
        """Bắt đầu camera và xử lý"""
        try:
            if self.camera_thread is not None and self.camera_thread.isRunning():
                return
                
            # Lấy camera ID
            camera_id = 0  # Mặc định là camera 0
            
            # Khởi tạo camera thread
            self.camera_thread = CameraThread(camera_id)
            
            # Kết nối tín hiệu từ camera thread
            self.camera_thread.frame_ready.connect(self.camera_panel.update_camera)
            self.camera_thread.error_occurred.connect(self.handle_error)

            # --- ĐỒNG BỘ TOÀN BỘ THÔNG SỐ NHẬN DIỆN NGÁP TỪ UI ---
            # Luôn cập nhật trạng thái bật/tắt, số lần ngáp tối đa, thời gian reset từ UI
            self.camera_thread.set_yawn_enabled(self.settings_panel.yawn_enable_switch.isChecked())
            self.camera_thread.set_max_yawn_count(self.settings_panel.yawn_count_spin.value())
            self.camera_thread.set_yawn_reset_minutes(self.settings_panel.yawn_reset_spin.value())
            # Cập nhật trạng thái enable/disable cho spinbox
            is_on = self.settings_panel.yawn_enable_switch.isChecked()
            self.settings_panel.yawn_count_spin.setEnabled(not is_on)
            self.settings_panel.yawn_reset_spin.setEnabled(not is_on)
            # ------------------------------------------------------

            # Cập nhật giao diện
            self.settings_panel.start_btn.setEnabled(False)
            self.settings_panel.stop_btn.setEnabled(True)
            self.settings_panel.update_status("Đang xử lý...", "success")
            
            # Bắt đầu thread
            self.camera_thread.start_camera()
            
        except Exception as e:
            self.handle_error(f"Lỗi khởi động camera: {str(e)}")
    
    def stop_camera(self):
        """Dừng camera và xử lý"""
        if self.camera_thread is not None and self.camera_thread.isRunning():
            self.camera_thread.stop_camera()
            self.camera_thread = None
            
            # Cập nhật giao diện
            self.settings_panel.start_btn.setEnabled(True)
            self.settings_panel.stop_btn.setEnabled(False)
            self.settings_panel.update_status("Đã dừng", "warning")
            
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
        
        
    def closeEvent(self, event):
        """Xử lý sự kiện đóng cửa sổ"""
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