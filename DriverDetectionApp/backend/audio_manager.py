"""
Audio Manager cho hệ thống cảnh báo
Quản lý âm thanh tập trung, tránh đè/lặp âm thanh
"""
import os
import time
import threading
from enum import Enum
from typing import Optional

try:
    from playsound import playsound
    AUDIO_AVAILABLE = True
except ImportError:
# Audio backend not available - alerts disabled
    AUDIO_AVAILABLE = False


class AlertType(Enum):
    """Loại cảnh báo"""
    DROWSY = "drowsy"
    YAWN = "yawn"
    GENERAL = "general"


class AudioManager:
    """Quản lý âm thanh cảnh báo tập trung"""
    
    def __init__(self, default_audio_file="alarm.wav", max_audio_duration=3.0):
        self.default_audio_file = default_audio_file
        self.custom_audio_files = {}
        self.max_audio_duration = max_audio_duration  # Giới hạn thời gian phát âm thanh (giây)
        
        # Audio state management
        self.audio_playing = False
        self.last_alert_time = 0
        self.last_alert_type = None
        self.min_alert_interval = 3.0  # Khoảng cách tối thiểu giữa các cảnh báo (giây)
        self.current_thread = None
        self.stop_audio_flag = False  # Flag để dừng âm thanh
        
        # Alert cooldowns để tránh spam
        self.alert_cooldowns = {
            AlertType.DROWSY: 5.0,  # 5 giây
            AlertType.YAWN: 3.0,    # 3 giây  
            AlertType.GENERAL: 2.0  # 2 giây
        }
        self.last_alert_times = {
            AlertType.DROWSY: 0,
            AlertType.YAWN: 0,
            AlertType.GENERAL: 0
        }
    
    def set_audio_file(self, file_path: str, alert_type: AlertType = AlertType.GENERAL):
        """Đặt file âm thanh cho loại cảnh báo cụ thể"""
        if os.path.exists(file_path):
            if alert_type == AlertType.GENERAL:
                self.default_audio_file = file_path
            else:
                self.custom_audio_files[alert_type] = file_path
        # File not found - keep default
    
    def get_audio_file(self, alert_type: AlertType) -> str:
        """Lấy file âm thanh cho loại cảnh báo"""
        return self.custom_audio_files.get(alert_type, self.default_audio_file)
    
    def can_play_alert(self, alert_type: AlertType) -> bool:
        """Kiểm tra xem có thể phát cảnh báo không"""
        if not AUDIO_AVAILABLE:
            return False
            
        current_time = time.time()
        
        # Kiểm tra nếu đang phát âm thanh khác
        if self.audio_playing:
            return False
            
        # Kiểm tra cooldown cho loại cảnh báo cụ thể
        last_time = self.last_alert_times.get(alert_type, 0)
        cooldown = self.alert_cooldowns.get(alert_type, 2.0)
        
        if current_time - last_time < cooldown:
            return False
            
        # Kiểm tra khoảng cách tối thiểu giữa bất kỳ cảnh báo nào
        if current_time - self.last_alert_time < self.min_alert_interval:
            return False
            
        return True
    
    def play_alert(self, alert_type: AlertType = AlertType.GENERAL) -> bool:
        """
        Phát cảnh báo âm thanh
        Returns: True nếu phát thành công, False nếu bị chặn
        """
        if not self.can_play_alert(alert_type):
            return False
            
        audio_file = self.get_audio_file(alert_type)
        if not os.path.exists(audio_file):
            return False
        
        # Cập nhật state
        current_time = time.time()
        self.audio_playing = True
        self.last_alert_time = current_time
        self.last_alert_times[alert_type] = current_time
        self.last_alert_type = alert_type
        
        # Phát âm thanh trong thread riêng với giới hạn thời gian
        def play_audio():
            try:
                # Play audio alert
                
                # Reset stop flag
                self.stop_audio_flag = False
                
                # Start audio in background
                import subprocess
                import platform
                
                if platform.system() == "Windows":
                    # Windows: sử dụng subprocess với timeout
                    proc = subprocess.Popen(
                        ['powershell', '-c', f'(New-Object Media.SoundPlayer "{audio_file}").PlaySync()'],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL
                    )
                    
                    # Wait với timeout
                    start_time = time.time()
                    while proc.poll() is None:
                        if time.time() - start_time >= self.max_audio_duration or self.stop_audio_flag:
                            proc.terminate()
                            break
                        time.sleep(0.1)
                else:
                    # Linux/Mac: fallback to playsound với timeout simulation
                    start_time = time.time()
                    audio_thread = threading.Thread(target=lambda: playsound(audio_file), daemon=True)
                    audio_thread.start()
                    
                    # Wait với timeout
                    while audio_thread.is_alive():
                        if time.time() - start_time >= self.max_audio_duration or self.stop_audio_flag:
                            break
                        time.sleep(0.1)
                    
            except Exception:
                # Audio playback failed
                pass
            finally:
                # Reset state sau khi phát xong
                self.audio_playing = False
                self.stop_audio_flag = False
        
        # Stop previous thread if still running
        if self.current_thread and self.current_thread.is_alive():
            # Note: playsound không thể stop easily, nhưng flag sẽ prevent overlap
            pass
            
        self.current_thread = threading.Thread(target=play_audio, daemon=True)
        self.current_thread.start()
        
        return True
    
    def stop_all_alerts(self):
        """Dừng tất cả cảnh báo (hard stop)"""
        self.audio_playing = False
        self.stop_audio_flag = True
    
    def set_max_duration(self, duration: float):
        """Đặt thời gian tối đa cho âm thanh"""
        self.max_audio_duration = max(1.0, min(60.0, duration))  # Giới hạn 1-60 giây
    
    def reset_cooldowns(self):
        """Reset tất cả cooldowns"""
        current_time = time.time()
        for alert_type in AlertType:
            self.last_alert_times[alert_type] = 0
        self.last_alert_time = 0
    
    def get_status(self) -> dict:
        """Lấy trạng thái hiện tại của audio manager"""
        current_time = time.time()
        return {
            "audio_available": AUDIO_AVAILABLE,
            "audio_playing": self.audio_playing,
            "last_alert_type": self.last_alert_type.value if self.last_alert_type else None,
            "time_since_last_alert": current_time - self.last_alert_time,
            "cooldowns_remaining": {
                alert_type.value: max(0, self.alert_cooldowns[alert_type] - (current_time - self.last_alert_times[alert_type]))
                for alert_type in AlertType
            }
        }
