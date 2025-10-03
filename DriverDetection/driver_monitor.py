from .frame_processor import FrameProcessor
from .eye_processor import EyeProcessor
from .yawn_processor import YawnProcessor
import cv2
import time
import threading

class DriverMonitor:
    def __init__(self, sleep_th=2):
        self.frame_processor = FrameProcessor()
        self.eye_processor = EyeProcessor()
        self.yawn_processor = YawnProcessor()

        self.detection_config = {
            'tesselation': False,
            'contours': True,
            'irises': False,
            'use_preprocessing': True,
            'enable_yawn_detection': True,
            'max_yawn_count': 5,  
            'yawn_reset_minutes': 10  
        }

        self.SLEEP_TH = sleep_th
        self.eyes_closed_start_time = None
        self.eyes_closed_duration = 0.0

        self.yawn_start_time = None
        self.yawn_duration = 0.0
        self.yawn_count = 0 
        self.last_yawn_state = False  # Để phát hiện rising edge
        self.last_yawn_reset_time = time.time()
        self.yawn_reset_countdown = None  

    def __update_eyes_closed_time(self, eyes_closed):
        current_time = time.time()
        if eyes_closed:
            if self.eyes_closed_start_time is None:
                self.eyes_closed_start_time = current_time
                self.eyes_closed_duration = 0.0
            else:
                self.eyes_closed_duration = current_time - self.eyes_closed_start_time
        else:
            self.eyes_closed_start_time = None
            self.eyes_closed_duration = 0.0

    def __check_drowsiness(self, eye_results):
        eyes_closed = eye_results['closed']
        self.__update_eyes_closed_time(eyes_closed)
        return self.eyes_closed_duration >= self.SLEEP_TH

    def __update_yawn_time(self, yawn):
        current_time = time.time()
        if yawn:
            if self.yawn_start_time is None:
                self.yawn_start_time = current_time
                self.yawn_duration = 0.0
            else:
                self.yawn_duration = current_time - self.yawn_start_time
        else:
            self.yawn_start_time = None
            self.yawn_duration = 0.0

    def process_frame(self, frame):
        try:
            frame_results = self.frame_processor.process_frame(frame, self.detection_config['use_preprocessing'])
            annotated_frame = self.frame_processor.draw_landmarks(
                frame,
                frame_results,
                self.detection_config['tesselation'],
                self.detection_config['contours'],
                self.detection_config['irises']
            )

            eye_results = self.__process_eyes(annotated_frame, frame_results)
            is_drowsy = self.__check_drowsiness(eye_results)

            # --- Yawn detection ---
            yawn_detected, yawn_conf = False, 0.0
            enable_yawn_detection = self.detection_config.get('enable_yawn_detection', True)
            max_yawn_count = self.detection_config.get('max_yawn_count', 5)
            # Đếm số lần ngáp
            if enable_yawn_detection:
                if hasattr(self.yawn_processor, 'available') and self.yawn_processor.available:
                    if frame_results.multi_face_landmarks:
                        mouth_img = self.yawn_processor.crop_mouth_region(frame, frame_results)
                        if mouth_img is not None:
                            yawn_detected, yawn_conf = self.yawn_processor.predict(mouth_img)
                            self.__update_yawn_time(yawn_detected)
                            # Đếm số lần ngáp chỉ khi cảnh báo xuất hiện (ngáp đủ lâu)
                            yawn_alert = (yawn_detected and self.yawn_duration >= 2.0)
                            if yawn_alert and not self.last_yawn_state:
                                self.yawn_count += 1
                            self.last_yawn_state = yawn_alert
                        else:
                            self.yawn_start_time = None
                            self.yawn_duration = 0.0
                            self.last_yawn_state = False
                    else:
                        self.yawn_start_time = None
                        self.yawn_duration = 0.0
                        self.last_yawn_state = False
            else:
                self.yawn_start_time = None
                self.yawn_duration = 0.0
                yawn_detected, yawn_conf = False, 0.0
                self.last_yawn_state = False

            # Tự động reset yawn_count sau số phút người dùng chọn
            reset_minutes = self.detection_config.get('yawn_reset_minutes', 10)
            now = time.time()

            # Nếu chưa đạt max yawn count, reset countdown
            if self.yawn_count < max_yawn_count:
                self.yawn_reset_countdown = None
                # Reset lại mốc thời gian nếu chưa đạt max
                self.last_yawn_reset_time = now
            else:
                # Đã đạt max yawn count, bắt đầu đếm ngược
                if self.yawn_reset_countdown is None:
                    self.yawn_reset_countdown = reset_minutes * 60
                    self.last_yawn_reset_time = now
                else:
                    elapsed = now - self.last_yawn_reset_time
                    self.yawn_reset_countdown = max(0, reset_minutes * 60 - elapsed)
                    # Khi hết thời gian, reset yawn count và countdown
                    if self.yawn_reset_countdown <= 0:
                        self.yawn_count = 0
                        self.yawn_reset_countdown = None
                        self.last_yawn_reset_time = now  # Đặt lại mốc cho lần tiếp theo

            self.__add_ui_elements(
                annotated_frame, eye_results, is_drowsy, yawn_detected, yawn_conf,
                yawn_count=self.yawn_count, max_yawn_count=max_yawn_count,
                yawn_alert=(yawn_detected and self.yawn_duration >= 2.0),
                yawn_reset_countdown=self.yawn_reset_countdown
            )
            return {
                'frame': annotated_frame,
                'face_detected': frame_results.multi_face_landmarks is not None,
                'eye_results': eye_results,
                'is_drowsy': is_drowsy,
                'closed_duration': self.eyes_closed_duration,
                'yawn': yawn_detected,
                'yawn_conf': yawn_conf,
                'yawn_duration': self.yawn_duration,
                'yawn_count': self.yawn_count,
                'max_yawn_count': max_yawn_count
            }
            
        except Exception as e:
            return {
                'frame': frame,
                'face_detected': False,
                'eye_results': {
                    'left_eye': {'state': 'error', 'confidence': 0.0},
                    'right_eye': {'state': 'error', 'confidence': 0.0},
                    'closed': False
                },
                'is_drowsy': False,
                'closed_duration': 0.0,
                'yawn': False,
                'yawn_conf': 0.0,
                'yawn_duration': 0.0,
                'yawn_count': self.yawn_count,
                'max_yawn_count': self.detection_config.get('max_yawn_count', 5)
            }
    
    def __process_eyes(self, frame, face):
        try:
            if face.multi_face_landmarks:
                # Do frame bị lật ngang, nên đổi thứ tự unpack để mắt trái/phải đúng
                # eye_processor trả về (left_from_processor, right_from_processor)
                # Nhưng do frame lật, left_from_processor thực chất là right_eye từ góc nhìn người dùng
                right_state, left_state, right_conf, left_conf = self.eye_processor.detect_eyes(frame, face.multi_face_landmarks[0])
                return {
                    'left_eye': {'state': left_state, 'confidence': left_conf},
                    'right_eye': {'state': right_state, 'confidence': right_conf},
                    'closed': left_state == 'closed' or right_state == 'closed'  # Chỉ cần 1 mắt nhắm
                }
        except Exception as e:
            print(f"Eye processing error: {e}")
        
        return {
            'left_eye': {'state': 'unknown', 'confidence': 0.0},
            'right_eye': {'state': 'unknown', 'confidence': 0.0},
            'closed': False
        }

    def __add_ui_elements(self, frame, eye_results, is_drowsy, yawn, yawn_conf, yawn_count=0, max_yawn_count=5, yawn_alert=False, yawn_reset_countdown=None):
        left_state = eye_results['left_eye']['state']
        right_state = eye_results['right_eye']['state']
        eye_text = f"Eyes: L={left_state} R={right_state}"
        cv2.putText(frame, eye_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        time_text = f"Closed time: {self.eyes_closed_duration:.1f}s / {self.SLEEP_TH:.1f}s"
        cv2.putText(frame, time_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Hiển thị thông tin ngáp và thời gian ngáp (đặt lệch dòng để tránh chồng chữ)
        yawn_text = f"Yawn: {yawn} ({yawn_conf:.2f}) | Yawn time: {self.yawn_duration:.1f}s"
        cv2.putText(frame, yawn_text, (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)


        # Hiển thị số lần ngáp chỉ khi bật nhận diện ngáp
        if self.detection_config.get('enable_yawn_detection', True):
            # Lấy đúng giá trị max_yawn_count từ config
            max_yawn_count = self.detection_config.get('max_yawn_count', 5)
            # Đếm số lần ngáp: đặt ở dòng dưới để không đè lên yawn_text
            yawn_count_text = f"Yawn count: {yawn_count}/{max_yawn_count}"
            cv2.putText(frame, yawn_count_text, (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # Cảnh báo nếu vượt quá số lần ngáp tối đa (logic giữ lại, hiển thị bằng popup)
        # Không hiển thị text trên camera nữa

    def update_config(self, **kwargs):
        changed = False
        for k, v in kwargs.items():
            if self.detection_config.get(k) != v:
                self.detection_config[k] = v
                changed = True
        # Đồng bộ trạng thái enable_yawn_detection vào yawn_processor nếu có thay đổi
        if changed and hasattr(self.yawn_processor, "enable") and hasattr(self.yawn_processor, "disable"):
            enabled = self.detection_config.get("enable_yawn_detection", True)
            if enabled:
                self.yawn_processor.enable()
            else:
                self.yawn_processor.disable()
        # Nếu có max_yawn_count hoặc yawn_reset_minutes thì reset lại mốc thời gian đếm ngược
        if 'max_yawn_count' in kwargs or 'yawn_reset_minutes' in kwargs:
            self.last_yawn_reset_time = time.time()
            # Nếu đang ở trạng thái đã đạt max yawn thì cập nhật lại countdown 
            max_yawn_count = self.detection_config.get('max_yawn_count', kwargs.get('max_yawn_count'))
            yawn_reset_minutes = self.detection_config.get('yawn_reset_minutes', kwargs.get('yawn_reset_minutes'))
            if self.yawn_count >= max_yawn_count:
                self.yawn_reset_countdown = yawn_reset_minutes * 60

        # Update SLEEP_TH if sleep_threshold is provided
        if 'sleep_threshold' in kwargs:
            self.SLEEP_TH = kwargs['sleep_threshold']
            changed = True
    
    def reset_timer(self):
        """Reset timer"""
        try:
            self.eyes_closed_start_time = None
            self.eyes_closed_duration = 0.0
        except Exception as e:
            print(f"Reset error: {e}")

    def reset_yawn_count(self):
        """Reset số lần ngáp về 0"""
        self.yawn_count = 0
