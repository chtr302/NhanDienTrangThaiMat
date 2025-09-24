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
            
            # Bật nhận diện ngáp nếu có
            self.camera_thread.yawn_enabled = self.settings_panel.yawn_enable_switch.isChecked()
            
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
    
    # Phương thức update_main_frame không còn cần thiết vì đã kết nối trực tiếp
    
    # @pyqtSlot(str)
    # def handle_error(self, error_message):
    #     """Xử lý lỗi từ camera thread"""
    #     # Update status label
    #     self.settings_panel.update_status("Lỗi AI: xem chi tiết", "error")

    #     # Show a detailed dialog so user can see the exact error/traceback
    #     try:
    #         dlg = QMessageBox(self)
    #         dlg.setIcon(QMessageBox.Icon.Critical)
    #         dlg.setWindowTitle("Lỗi khởi tạo AI")
    #         # Truncate excessive length visually but keep most details
    #         shown = error_message if len(error_message) < 6000 else (error_message[:6000] + "\n... (rút gọn)")
    #         dlg.setText("Không thể khởi tạo mô-đun AI.")
    #         dlg.setDetailedText(shown)
    #         dlg.setStandardButtons(QMessageBox.StandardButton.Ok)
    #         dlg.exec_()
    #     except Exception:
    #         pass
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
