"""
Alert Manager để quản lý các cảnh báo tùy chỉnh
"""

import uuid
from PyQt5.QtWidgets import QWidget
from PyQt5.QtCore import QPoint, pyqtSignal, QObject
from .ui_components import AlertWidget


class AlertManager(QObject):
    """Quản lý các alert widget trên toàn bộ ứng dụng"""

    def __init__(self, parent_widget):
        super().__init__()
        self.parent_widget = parent_widget
        self.active_alerts = {}  # alert_id -> AlertWidget
        self.alert_spacing = 10  # Khoảng cách giữa các alert

    def show_alert(self, alert_type, message):
        """
        Hiển thị cảnh báo mới
        Returns: alert_id nếu thành công, None nếu thất bại
        """
        # Tạo ID duy nhất cho alert
        alert_id = str(uuid.uuid4())

        # Tạo AlertWidget
        alert_widget = AlertWidget(alert_id, alert_type, message, self.parent_widget)
        alert_widget.alert_closed.connect(self._on_alert_closed)

        # Tính vị trí
        position = self._calculate_position()
        alert_widget.set_position(position.x(), position.y())

        # Lưu vào danh sách active
        self.active_alerts[alert_id] = alert_widget

        # Hiển thị và bắt đầu animation
        alert_widget.show()
        alert_widget.show_animation.start()

        return alert_id

    def hide_alert(self, alert_id):
        """Ẩn một cảnh báo cụ thể"""
        if alert_id in self.active_alerts:
            self.active_alerts[alert_id].close_alert()

    def hide_all_alerts(self):
        """Ẩn tất cả cảnh báo"""
        for alert_id in list(self.active_alerts.keys()):
            self.hide_alert(alert_id)

    def _calculate_position(self):
        """Tính vị trí cho alert mới (từ góc trên bên phải)"""
        # Vị trí cơ bản: góc trên bên phải của parent widget
        base_x = self.parent_widget.width() - 370  # 350 (width) + 20 (margin)
        base_y = 20

        # Nếu có alert đang active, tính vị trí phía dưới
        if self.active_alerts:
            # Lấy alert cuối cùng để tính vị trí
            last_alert = list(self.active_alerts.values())[-1]
            last_y = last_alert.y()
            last_height = last_alert.height()

            base_y = last_y + last_height + self.alert_spacing

        return QPoint(base_x, base_y)

    def _on_alert_closed(self, alert_id):
        """Xử lý khi một alert bị đóng"""
        if alert_id in self.active_alerts:
            # Xóa khỏi danh sách
            del self.active_alerts[alert_id]

            # Cập nhật lại vị trí của các alert còn lại
            self._reposition_alerts()

    def _reposition_alerts(self):
        """Cập nhật lại vị trí của tất cả alert sau khi một alert bị xóa"""
        base_x = self.parent_widget.width() - 370
        current_y = 20

        for alert_widget in self.active_alerts.values():
            # Tạo animation để di chuyển đến vị trí mới
            alert_widget.set_position(base_x, current_y)
            alert_widget.show_animation.start()

            current_y += alert_widget.height() + self.alert_spacing

    def update_positions_on_resize(self):
        """Cập nhật vị trí khi parent widget thay đổi kích thước"""
        if self.active_alerts:
            self._reposition_alerts()

