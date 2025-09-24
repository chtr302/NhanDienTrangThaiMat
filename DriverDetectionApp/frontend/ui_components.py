"""
UI Components cho giao diện người dùng
"""

from PyQt5.QtWidgets import QCheckBox
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPainter, QColor, QFont


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
            painter.drawText(8, 20, "ON")
        else:
            painter.drawText(38, 20, "OFF")

    def animate_toggle(self):
        """Animation khi toggle"""
        self.update()

    def mousePressEvent(self, event):
        """Handle mouse click"""
        super().mousePressEvent(event)
        self.update()
