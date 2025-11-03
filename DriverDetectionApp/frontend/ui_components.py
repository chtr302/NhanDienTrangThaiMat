
from PyQt5.QtWidgets import (QCheckBox, QLabel, QFrame, QPushButton,
                             QHBoxLayout, QVBoxLayout, QWidget)
from PyQt5.QtCore import (Qt, QTimer, QPropertyAnimation, QRect, QEasingCurve,
                         QPoint, pyqtSignal)
from PyQt5.QtGui import QPainter, QColor, QFont, QIcon


class ToggleSwitch(QCheckBox):
    """Custom toggle switch widget"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(60, 30)
        self.stateChanged.connect(self.update)

    def paintEvent(self, event):
        """Custom paint event để vẽ toggle switch"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Background
        if self.isChecked():
            painter.setBrush(QColor(76, 175, 80))  # Green when ON
        else:
            painter.setBrush(QColor(189, 189, 189))  # Gray when OFF

        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(0, 0, self.width(), self.height(), 15, 15)

        # Switch circle
        painter.setBrush(QColor(255, 255, 255))
        circle_x = self.width() - 26 if self.isChecked() else 4
        painter.drawEllipse(circle_x, 4, 22, 22)

        # Text
        painter.setPen(QColor(255, 255, 255))
        painter.setFont(QFont("Arial", 8, QFont.Weight.Bold))
        if self.isChecked():
            painter.drawText(8, 20, "BẬT")
        else:
            painter.drawText(38, 20, "TẮT")

    def animate_toggle(self):
        """Animation khi toggle"""
        self.update()

    def mousePressEvent(self, event):
        """Handle mouse click"""
        if self.isEnabled():
            self.setChecked(not self.isChecked())
        self.update()


class AlertWidget(QFrame):
    """Custom alert widget với animation và auto-hide"""

    # Signal khi alert bị đóng
    alert_closed = pyqtSignal(str)  # alert_id

    def __init__(self, alert_id, alert_type, message, parent=None):
        super().__init__(parent)
        self.alert_id = alert_id
        self.alert_type = alert_type
        self.message = message

        # Setup widget
        self.setFrameStyle(QFrame.Shape.Box)
        self.setLineWidth(2)
        self.setFixedWidth(350)
        self.setMinimumHeight(80)

        # Set colors based on alert type
        self._setup_colors()

        # Setup UI
        self._setup_ui()

        # Setup animation và timer
        self._setup_animation()
        self._setup_timer()

    def _setup_colors(self):
        """Thiết lập màu sắc theo loại cảnh báo"""
        if self.alert_type == "DROWSY":
            self.border_color = QColor(211, 47, 47)   # Red
            self.bg_color = QColor(255, 235, 238)     # Light Red
            self.text_color = QColor(198, 40, 40)     # Dark Red
            self.icon_text = "😴"
        elif self.alert_type == "YAWN": # Giữ lại để tương thích, cũng đổi sang vàng
            self.border_color = QColor(255, 179, 0)  # Amber
            self.bg_color = QColor(255, 248, 225)    # Light Amber
            self.text_color = QColor(255, 143, 0)    # Dark Amber
            self.icon_text = "🥱"
        else: # GENERAL - Dùng cho cảnh báo đếm số lần ngáp
            self.border_color = QColor(255, 179, 0)  # Amber
            self.bg_color = QColor(255, 248, 225)    # Light Amber
            self.text_color = QColor(255, 143, 0)    # Dark Amber
            self.icon_text = "⚠️"

    def _setup_ui(self):
        """Thiết lập giao diện"""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(15, 10, 15, 10)
        layout.setSpacing(10)

        # Icon label
        self.icon_label = QLabel(self.icon_text)
        self.icon_label.setFont(QFont("Arial", 24))
        self.icon_label.setFixedSize(40, 40)
        layout.addWidget(self.icon_label)

        # Text layout
        text_layout = QVBoxLayout()
        text_layout.setSpacing(2)

        # Title
        self.title_label = QLabel(f"CẢNH BÁO {self.alert_type}")
        self.title_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        self.title_label.setStyleSheet(f"color: {self.text_color.name()};")
        text_layout.addWidget(self.title_label)

        # Message
        self.message_label = QLabel(self.message)
        self.message_label.setFont(QFont("Arial", 10))
        self.message_label.setStyleSheet(f"color: {self.text_color.name()};")
        self.message_label.setWordWrap(True)
        text_layout.addWidget(self.message_label)

        layout.addLayout(text_layout, 1)

        # Close button
        self.close_btn = QPushButton("✕")
        self.close_btn.setFixedSize(30, 30)
        self.close_btn.setFont(QFont("Arial", 12))
        self.close_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: transparent;
                color: {self.text_color.name()};
                border: none;
                border-radius: 15px;
            }}
            QPushButton:hover {{
                background-color: rgba(0, 0, 0, 0.1);
            }}
        """)
        self.close_btn.clicked.connect(self.close_alert)
        layout.addWidget(self.close_btn)

        # Set background color
        self.setStyleSheet(f"""
            AlertWidget {{
                background-color: {self.bg_color.name()};
                border: 2px solid {self.border_color.name()};
                border-radius: 10px;
            }}
        """)

    def _setup_animation(self):
        """Thiết lập animation"""
        # Show animation - slide in from right
        self.show_animation = QPropertyAnimation(self, b"pos")
        self.show_animation.setDuration(500)
        self.show_animation.setEasingCurve(QEasingCurve.Type.OutCubic)

        # Hide animation - slide out to right
        self.hide_animation = QPropertyAnimation(self, b"pos")
        self.hide_animation.setDuration(300)
        self.hide_animation.setEasingCurve(QEasingCurve.Type.InCubic)
        self.hide_animation.finished.connect(self._on_hide_finished)

    def _setup_timer(self):
        """Thiết lập timer tự động ẩn"""
        self.auto_hide_timer = QTimer(self)
        self.auto_hide_timer.setSingleShot(True)
        self.auto_hide_timer.timeout.connect(self.close_alert)
        # Auto hide after 8 seconds
        self.auto_hide_timer.start(8000)

    def set_position(self, x, y):
        """Thiết lập vị trí widget"""
        self.move(x, y)
        # Update animation end position
        start_pos = QPoint(x + 400, y)  # Start from right
        end_pos = QPoint(x, y)
        self.show_animation.setStartValue(start_pos)
        self.show_animation.setEndValue(end_pos)

        hide_start_pos = QPoint(x, y)
        hide_end_pos = QPoint(x + 400, y)  # Hide to right
        self.hide_animation.setStartValue(hide_start_pos)
        self.hide_animation.setEndValue(hide_end_pos)

    def close_alert(self):
        """Đóng cảnh báo với animation"""
        # Stop timer
        self.auto_hide_timer.stop()
        # Start hide animation
        self.hide_animation.start()

    def _on_hide_finished(self):
        """Called when hide animation finishes"""
        # Emit signal and hide
        self.alert_closed.emit(self.alert_id)
        self.hide()

    def enterEvent(self, event):
        """Pause auto-hide when mouse enters"""
        self.auto_hide_timer.stop()
        super().enterEvent(event)

    def leaveEvent(self, event):
        """Resume auto-hide when mouse leaves"""
        self.auto_hide_timer.start(8000)
        super().leaveEvent(event)