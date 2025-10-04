"""
Audio Manager cho hệ thống cảnh báo
Quản lý âm thanh tập trung, tránh chồng/đè âm thanh và cho phép timeout.
"""
import os
import time
import threading
import platform
from enum import Enum

# playsound là tùy chọn. Trên Windows có thể phát WAV bằng winsound (built‑in),
# vì vậy thiếu playsound KHÔNG nên vô hiệu hóa âm thanh.
HAVE_PLAYSOUND = False
try:
    from playsound import playsound  # type: ignore
    HAVE_PLAYSOUND = True
except Exception:
    HAVE_PLAYSOUND = False

# Có audio nếu là Windows (winsound) hoặc có playsound
AUDIO_AVAILABLE = (platform.system() == "Windows") or HAVE_PLAYSOUND


class AlertType(Enum):
    DROWSY = "drowsy"
    YAWN = "yawn"
    GENERAL = "general"


class AudioManager:
    """Quản lý âm thanh cảnh báo tập trung"""

    def __init__(self, default_audio_file="alarm.wav", max_audio_duration=3.0):
        self.default_audio_file = default_audio_file
        self.custom_audio_files = {}
        self.max_audio_duration = max_audio_duration  # giới hạn thời gian phát (giây)

        # Audio state
        self.audio_playing = False
        self.last_alert_time = 0.0
        self.last_alert_type = None
        self.min_alert_interval = 3.0  # khoảng cách tối thiểu giữa các cảnh báo (giây)
        self.current_thread = None
        self.stop_audio_flag = False

        # Cooldowns riêng theo loại
        self.alert_cooldowns = {
            AlertType.DROWSY: 5.0,
            AlertType.YAWN: 3.0,
            AlertType.GENERAL: 2.0,
        }
        self.last_alert_times = {
            AlertType.DROWSY: 0.0,
            AlertType.YAWN: 0.0,
            AlertType.GENERAL: 0.0,
        }

    def set_audio_file(self, file_path: str, alert_type: AlertType = AlertType.GENERAL):
        """Đặt file âm thanh cho loại cảnh báo cụ thể"""
        if os.path.exists(file_path):
            if alert_type == AlertType.GENERAL:
                self.default_audio_file = file_path
            else:
                self.custom_audio_files[alert_type] = file_path

    def get_audio_file(self, alert_type: AlertType) -> str:
        return self.custom_audio_files.get(alert_type, self.default_audio_file)

    def can_play_alert(self, alert_type: AlertType) -> bool:
        if not AUDIO_AVAILABLE:
            return False
        current_time = time.time()
        if self.audio_playing:
            return False
        last_time = self.last_alert_times.get(alert_type, 0.0)
        cooldown = self.alert_cooldowns.get(alert_type, 2.0)
        if current_time - last_time < cooldown:
            return False
        if current_time - self.last_alert_time < self.min_alert_interval:
            return False
        return True

    def play_alert(self, alert_type: AlertType = AlertType.GENERAL) -> bool:
        """Phát cảnh báo âm thanh (có timeout)."""
        if not self.can_play_alert(alert_type):
            return False

        audio_file = self.get_audio_file(alert_type)
        if not os.path.exists(audio_file):
            return False

        # Cập nhật trạng thái
        now = time.time()
        self.audio_playing = True
        self.last_alert_time = now
        self.last_alert_times[alert_type] = now
        self.last_alert_type = alert_type

        def play_audio():
            try:
                self.stop_audio_flag = False
                if platform.system() == "Windows":
                    # WAV: dùng winsound; MP3/OGG: playsound nếu có
                    if audio_file.lower().endswith(".wav"):
                        import winsound
                        winsound.PlaySound(audio_file, winsound.SND_FILENAME | winsound.SND_ASYNC)
                        start = time.time()
                        while True:
                            if time.time() - start >= self.max_audio_duration or self.stop_audio_flag:
                                try:
                                    winsound.PlaySound(None, winsound.SND_PURGE)
                                except Exception:
                                    pass
                                break
                            time.sleep(0.1)
                    elif HAVE_PLAYSOUND:
                        start = time.time()
                        t = threading.Thread(target=lambda: playsound(audio_file), daemon=True)
                        t.start()
                        while t.is_alive():
                            if time.time() - start >= self.max_audio_duration or self.stop_audio_flag:
                                break
                            time.sleep(0.1)
                    else:
                        # Định dạng không hỗ trợ nếu thiếu playsound
                        pass
                else:
                    if HAVE_PLAYSOUND:
                        start = time.time()
                        t = threading.Thread(target=lambda: playsound(audio_file), daemon=True)
                        t.start()
                        while t.is_alive():
                            if time.time() - start >= self.max_audio_duration or self.stop_audio_flag:
                                break
                            time.sleep(0.1)
                    else:
                        pass
            except Exception:
                # Bỏ qua lỗi phát âm thanh
                pass
            finally:
                self.audio_playing = False
                self.stop_audio_flag = False

        if self.current_thread and self.current_thread.is_alive():
            # Không dừng thread cũ chủ động, flag sẽ ngăn chồng âm
            pass

        self.current_thread = threading.Thread(target=play_audio, daemon=True)
        self.current_thread.start()
        return True

    def stop_all_alerts(self):
        self.audio_playing = False
        self.stop_audio_flag = True

    def set_max_duration(self, duration: float):
        self.max_audio_duration = max(1.0, min(60.0, float(duration)))

    def reset_cooldowns(self):
        for k in list(self.last_alert_times.keys()):
            self.last_alert_times[k] = 0.0
        self.last_alert_time = 0.0

    def get_status(self) -> dict:
        current_time = time.time()
        return {
            "audio_available": AUDIO_AVAILABLE,
            "audio_playing": self.audio_playing,
            "last_alert_type": self.last_alert_type.value if self.last_alert_type else None,
            "time_since_last_alert": current_time - self.last_alert_time,
            "cooldowns_remaining": {
                k.value: max(0.0, self.alert_cooldowns[k] - (current_time - self.last_alert_times[k]))
                for k in self.last_alert_times
            },
        }

