"""
Camera Panel cho hiển thị camera và kết quả AI
"""

import numpy as np
import cv2
import os
from PyQt5.QtWidgets import (QFrame, QVBoxLayout, QLabel, QGroupBox, QWidget)
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
        camera_group.setStyleSheet(
            "QGroupBox { background-color: white; border: 2px solid rgb(200, 200, 200); border-radius: 8px; margin-top: 1ex; padding-top: 15px; } "
            "QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; padding: 0 5px; background-color: white; color: rgb(80, 80, 80); font-weight: bold; }"
        )
        camera_layout = QVBoxLayout(camera_group)

        self.main_camera_label = QLabel()
        self.main_camera_label.setMinimumSize(640, 480)
        self.main_camera_label.setStyleSheet("QLabel { background-color: white; border-radius: 8px; }")
        self.main_camera_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.main_camera_label.clear()
        camera_layout.addWidget(self.main_camera_label)

        parent_layout.addWidget(camera_group)

    def update_camera(self, frame):
        """Cập nhật hiển thị camera chính"""
        if frame is None:
            return

        # Nếu có layout (tức là placeholder đang hiển thị), hãy xóa nó đi
        if self.main_camera_label.layout() is not None:
            # Xóa các widget con trong layout cũ
            while self.main_camera_label.layout().count():
                child = self.main_camera_label.layout().takeAt(0)
                if child.widget():
                    child.widget().deleteLater()
            # Xóa layout cũ bằng cách gán nó cho một widget tạm thời
            QWidget().setLayout(self.main_camera_label.layout())

        try:
            height, width, channel = frame.shape
            bytes_per_line = 3 * width
            q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format.Format_RGB888).rgbSwapped()

            # Scale ảnh để fit với label
            pixmap = QPixmap.fromImage(q_image)
            scaled_pixmap = pixmap.scaled(self.main_camera_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            self.main_camera_label.setPixmap(scaled_pixmap)

            # Clear text khi hiển thị video
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
        """Hiển thị logo và text khi không có camera"""
        # Xóa pixmap cũ và layout cũ nếu có
        self.main_camera_label.clear()
        if self.main_camera_label.layout() is not None:
            # Xóa các widget con trong layout cũ
            while self.main_camera_label.layout().count():
                child = self.main_camera_label.layout().takeAt(0)
                if child.widget():
                    child.widget().deleteLater()
            # Xóa layout cũ
            QWidget().setLayout(self.main_camera_label.layout())

        # --- Tạo layout mới ---
        placeholder_layout = QVBoxLayout(self.main_camera_label)
        placeholder_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.main_camera_label.setLayout(placeholder_layout)

        # --- Label cho Logo ---
        logo_label = QLabel()
        logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        logo_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'img', 'logo.png')

        if os.path.exists(logo_path):
            pixmap = QPixmap(logo_path)
            if not pixmap.isNull():
                scaled_pixmap = pixmap.scaled(400, 400, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                logo_label.setPixmap(scaled_pixmap)
            else:
                logo_label.setText("Không thể tải logo.")
        else:
            logo_label.setText(f"Không tìm thấy logo tại:\n{logo_path}")
        
        placeholder_layout.addWidget(logo_label)

        # --- Label cho Text ---
        text_label = QLabel("Nhấn 'Bắt đầu' để sử dụng chương trình")
        text_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        text_label.setStyleSheet("font-size: 16px; color: #888;")
        placeholder_layout.addWidget(text_label)

