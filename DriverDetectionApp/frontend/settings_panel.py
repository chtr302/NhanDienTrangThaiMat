"""
Settings Panel cho cấu hình hệ thống
"""

import os
from PyQt5.QtWidgets import (QFrame, QVBoxLayout, QHBoxLayout, QLabel,
                             QGroupBox, QPushButton, QSpinBox, QFileDialog, QApplication, QStyle, QFileIconProvider)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QIcon

from .ui_components import ToggleSwitch


class SettingsPanel(QFrame):
    """Panel cài đặt bên trái của ứng dụng"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameStyle(QFrame.Shape.StyledPanel)
        self.setMaximumWidth(350)
        self.init_ui()

    def init_ui(self):
        """Khởi tạo giao diện settings"""
        layout = QVBoxLayout(self)

        # Title
        title = QLabel("CÀI ĐẶT HỆ THỐNG")
        title.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # Nhóm cài đặt âm thanh
        self.create_audio_group(layout)

        # Nhóm cài đặt nhận diện mắt
        self.create_eye_group(layout)

        # Nhóm cài đặt nhận diện ngáp
        self.create_yawn_group(layout)

        # Nhóm điều khiển hệ thống
        self.create_control_group(layout)

        # Status
        self.status_label = QLabel("Trạng thái: Đang kiểm tra...")
        self.status_label.setStyleSheet("QLabel { background-color: #e8f5e8; color: black; padding: 10px; border-radius: 5px; font-weight: bold; }")
        layout.addWidget(self.status_label)

        layout.addStretch()

    def create_audio_group(self, parent_layout):
        """Tạo nhóm cài đặt âm thanh"""
        audio_group = QGroupBox("Cài Đặt Âm Thanh")
        audio_group.setStyleSheet(
            "QGroupBox { background-color: white; border: 2px solid rgb(200, 200, 200); border-radius: 8px; margin-top: 1ex; padding-top: 15px; } "
            "QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; padding: 0 5px; background-color: white; color: rgb(80, 80, 80); font-weight: bold; }"
        )
        audio_layout = QVBoxLayout(audio_group)

        # Chọn file âm thanh
        audio_file_layout = QHBoxLayout()
        self.audio_file_label = QLabel("alarm.wav")
        self.audio_browse_btn = QPushButton()
        self.audio_browse_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DirOpenIcon))
        self.audio_browse_btn.setFlat(True) # Ẩn shadow
        self.audio_browse_btn.clicked.connect(self.browse_audio_file)
        audio_file_label_title = QLabel("File âm thanh:")
        audio_file_label_title.setStyleSheet("font-weight: bold;")
        audio_file_layout.addWidget(audio_file_label_title)
        audio_file_layout.addWidget(self.audio_file_label)
        audio_file_layout.addWidget(self.audio_browse_btn)
        audio_layout.addLayout(audio_file_layout)

        # Giới hạn thời gian âm thanh
        duration_layout = QHBoxLayout()
        duration_label_title = QLabel("Thời gian tối đa (s):")
        duration_label_title.setStyleSheet("font-weight: bold;")
        duration_layout.addWidget(duration_label_title)
        self.audio_duration_spin = QSpinBox()
        self.audio_duration_spin.setRange(1, 60)
        self.audio_duration_spin.setValue(10)
        duration_layout.addWidget(self.audio_duration_spin)
        audio_layout.addLayout(duration_layout)

        parent_layout.addWidget(audio_group)

    def create_eye_group(self, parent_layout):
        """Tạo nhóm cài đặt nhận diện mắt"""
        eye_group = QGroupBox("Nhận Diện Mắt")
        eye_group.setStyleSheet(
            "QGroupBox { background-color: white; border: 2px solid rgb(200, 200, 200); border-radius: 8px; margin-top: 1ex; padding-top: 15px; } "
            "QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; padding: 0 5px; background-color: white; color: rgb(80, 80, 80); font-weight: bold; }"
        )
        eye_layout = QVBoxLayout(eye_group)

        # Ngưỡng thời gian nhắm mắt
        eye_threshold_layout = QHBoxLayout()
        eye_threshold_label_title = QLabel("Ngưỡng nhắm mắt (s):")
        eye_threshold_label_title.setStyleSheet("font-weight: bold;")
        eye_threshold_layout.addWidget(eye_threshold_label_title)
        self.eye_threshold_spin = QSpinBox()
        self.eye_threshold_spin.setRange(1, 2)
        self.eye_threshold_spin.setValue(2)
        eye_threshold_layout.addWidget(self.eye_threshold_spin)
        eye_layout.addLayout(eye_threshold_layout)

        parent_layout.addWidget(eye_group)

    def create_yawn_group(self, parent_layout):
        """Tạo nhóm cài đặt nhận diện ngáp"""
        yawn_group = QGroupBox("Nhận Diện Ngáp")
        yawn_group.setStyleSheet(
            "QGroupBox { background-color: white; border: 2px solid rgb(200, 200, 200); border-radius: 8px; margin-top: 1ex; padding-top: 15px; } "
            "QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; padding: 0 5px; background-color: white; color: rgb(80, 80, 80); font-weight: bold; }"
        )
        yawn_layout = QVBoxLayout(yawn_group)

        # Toggle switch bật/tắt
        yawn_toggle_layout = QHBoxLayout()
        yawn_toggle_label_title = QLabel("Nhận diện ngáp:")
        yawn_toggle_label_title.setStyleSheet("font-weight: bold;")
        yawn_toggle_layout.addWidget(yawn_toggle_label_title)
        self.yawn_enable_switch = ToggleSwitch()
        self.yawn_enable_switch.setChecked(False)  # Mặc định OFF khi khởi tạo
        yawn_toggle_layout.addWidget(self.yawn_enable_switch)
        yawn_toggle_layout.addStretch()
        yawn_layout.addLayout(yawn_toggle_layout)

        # Số lần ngáp tối đa
        yawn_count_layout = QHBoxLayout()
        yawn_count_label_title = QLabel("Số lần ngáp tối đa:")
        yawn_count_label_title.setStyleSheet("font-weight: bold;")
        yawn_count_layout.addWidget(yawn_count_label_title)
        self.yawn_count_spin = QSpinBox()
        self.yawn_count_spin.setRange(1, 20)
        self.yawn_count_spin.setValue(5)
        yawn_count_layout.addWidget(self.yawn_count_spin)
        yawn_layout.addLayout(yawn_count_layout)

        # Nút reset số lần ngáp
        self.reset_yawn_btn = QPushButton("Đặt lại số lần ngáp")
        self.reset_yawn_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                font-weight: bold;
                padding: 6px;
            }
        """)
        yawn_layout.addWidget(self.reset_yawn_btn)

        # Thời gian reset (phút)
        yawn_reset_layout = QHBoxLayout()
        yawn_reset_label_title = QLabel("Đặt lại sau (phút):")
        yawn_reset_label_title.setStyleSheet("font-weight: bold;")
        yawn_reset_layout.addWidget(yawn_reset_label_title)
        self.yawn_reset_spin = QSpinBox()
        self.yawn_reset_spin.setRange(1, 60)
        self.yawn_reset_spin.setValue(10)
        yawn_reset_layout.addWidget(self.yawn_reset_spin)
        yawn_layout.addLayout(yawn_reset_layout)

        parent_layout.addWidget(yawn_group)

    def create_control_group(self, parent_layout):
        """Tạo nhóm điều khiển hệ thống"""
        control_group = QGroupBox("Điều Khiển")
        control_group.setStyleSheet(
            "QGroupBox { background-color: white; border: 2px solid rgb(200, 200, 200); border-radius: 8px; margin-top: 1ex; padding-top: 15px; } "
            "QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; padding: 0 5px; background-color: white; color: rgb(80, 80, 80); font-weight: bold; }"
        )
        control_layout = QVBoxLayout(control_group)

        self.start_btn = QPushButton("Bắt Đầu")
        self.start_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                padding: 10px;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        control_layout.addWidget(self.start_btn)

        self.stop_btn = QPushButton("Dừng")
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                font-weight: bold;
                padding: 10px;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        control_layout.addWidget(self.stop_btn)

        parent_layout.addWidget(control_group)

    def browse_audio_file(self):
        """Chọn file âm thanh"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Chọn File Âm Thanh", "",
            "Audio Files (*.wav *.mp3 *.ogg);;All Files (*)"
        )
        if file_path:
            self.audio_file_label.setText(os.path.basename(file_path))
            # Lưu đường dẫn đầy đủ để sử dụng sau này
            self.audio_file_label.setProperty("full_path", file_path)

    def update_status(self, message, style="success"):
        """Cập nhật trạng thái hiển thị"""
        self.status_label.setText(f"Trạng thái: {message}")

        if style == "success":
            self.status_label.setStyleSheet("QLabel { background-color: #d4edda; color: black; padding: 10px; border-radius: 5px; }")
        elif style == "warning":
            self.status_label.setStyleSheet("QLabel { background-color: #fff3cd; color: black; padding: 10px; border-radius: 5px; }")
        elif style == "error":
            self.status_label.setStyleSheet("QLabel { background-color: #f8d7da; color: black; padding: 10px; border-radius: 5px; }")