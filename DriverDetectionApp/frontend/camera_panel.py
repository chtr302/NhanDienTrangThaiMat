"""
Camera Panel cho hiển thị camera và kết quả AI
"""

import numpy as np
import cv2
from PyQt5.QtWidgets import (QFrame, QVBoxLayout, QLabel, QGroupBox)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap


class CameraPanel(QFrame):
    """Panel hiển thị camera bên phải của ứng dụng"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameStyle(QFrame.Shape.StyledPanel)
        self.init_ui()

    def init_ui(self):
        """Khởi tạo giao diện camera"""
        layout = QVBoxLayout(self)

        # Camera chính - hiển thị kết quả nhận diện
        self.create_main_camera_group(layout)

    def create_main_camera_group(self, parent_layout):
        """Tạo nhóm camera chính hiển thị kết quả nhận diện"""
        camera_group = QGroupBox("Nhận Diện Trạng Thái Tài Xế")
        camera_layout = QVBoxLayout(camera_group)

        self.main_camera_label = QLabel()
        self.main_camera_label.setMinimumSize(640, 480)
        self.main_camera_label.setStyleSheet("QLabel { border: 2px solid #ddd; background-color: #f8f8f8; border-radius: 8px; }")
        self.main_camera_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.main_camera_label.clear()
        camera_layout.addWidget(self.main_camera_label)

        parent_layout.addWidget(camera_group)

    def update_camera(self, frame):
        """Cập nhật hiển thị camera chính"""
        if frame is None:
            return

        try:
            height, width, channel = frame.shape
            bytes_per_line = 3 * width
            q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format.Format_RGB888).rgbSwapped()

            # Scale ảnh để fit với label
            pixmap = QPixmap.fromImage(q_image)
            scaled_pixmap = pixmap.scaled(self.main_camera_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            self.main_camera_label.setPixmap(scaled_pixmap)

            # Clear text when showing camera feed
            self.main_camera_label.setText("")
        except Exception as e:
            print(f"Lỗi hiển thị camera: {e}")
            self.main_camera_label.setText("Lỗi hiển thị camera")

    # Giữ lại các phương thức cũ để tương thích với code hiện tại
    def update_eye_camera(self, frame):
        """Cập nhật hiển thị camera nhận diện mắt (chuyển hướng đến camera chính)"""
        self.update_camera(frame)

    def update_yawn_camera(self, frame):
        """Cập nhật hiển thị camera nhận diện ngáp (chuyển hướng đến camera chính)"""
        self.update_camera(frame)

    def show_placeholder(self):
        """Hiển thị placeholder khi không có camera"""
        self.main_camera_label.clear()
        self.main_camera_label.setText("Nhấn 'Bắt đầu' để khởi động camera\nHoặc kiểm tra kết nối camera")

