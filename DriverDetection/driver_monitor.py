import cv2
import time
import threading
from collections import deque
from .frame_processor import FrameProcessor
from .eye_processor import EyeProcessor
from .yawn_processor import YawnProcessor

class DriverMonitor:
    def __init__(self, sleep_th=2.0):
        self.frame_processor = FrameProcessor()
        self.eye_processor = EyeProcessor()
        self.yawn_processor = YawnProcessor()

        self.detection_config = {
            'tesselation': False,
            'contours': False,
            'irises': False,
            'use_preprocessing': True,
            'enable_yawn_detection': True,
            'max_yawn_count': 5,  
            'yawn_reset_minutes': 10,
            'eye_prediction_threshold': 0.3
        }

        self.prob_history = deque(maxlen=5) # Lưu 5 xác suất mắt mở gần nhất
        self.SMOOTHED_PROB_THRESHOLD = 0.4 # Ngưỡng cho giá trị xác suất đã được làm mượt

        self.SLEEP_TH = sleep_th
        self.eyes_closed_start_time = None
        self.eyes_closed_duration = 0.0

        self.yawn_start_time = None
        self.yawn_duration = 0.0
        self.yawn_count = 0 
        self.last_yawn_state = False
        self.last_yawn_reset_time = time.time()
        self.yawn_reset_countdown = None  

        self.frame_count = 0
        self.SKIP_FRAMES = 2
        self.cached_results = None

    def __update_eyes_closed_time(self, eyes_closed_this_frame):
        """
        Cập nhật bộ đếm thời gian mắt nhắm (đã đơn giản hóa).
        """
        current_time = time.time()

        if eyes_closed_this_frame:
            if self.eyes_closed_start_time is None:
                self.eyes_closed_start_time = current_time
            self.eyes_closed_duration = current_time - self.eyes_closed_start_time
        else:
            self.eyes_closed_start_time = None
            self.eyes_closed_duration = 0.0

    def __check_drowsiness(self, eye_data):
        """
        Kiểm tra trạng thái buồn ngủ dựa trên tín hiệu mắt đã được làm mượt.
        """
        is_closed_this_frame = eye_data['is_closed_this_frame']
        self.__update_eyes_closed_time(is_closed_this_frame)
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
        """
        Xử lý từng khung hình từ camera, phát hiện buồn ngủ và ngáp.
        Đảm bảo UI được vẽ trên mọi khung hình để tránh nhấp nháy.
        """
        self.frame_count += 1
        should_process_model = (self.frame_count % (self.SKIP_FRAMES + 1)) == 0

        current_eye_results_data = {'avg_prob_open': 1.0, 'smoothed_prob': 1.0, 'is_closed_this_frame': False}
        current_is_drowsy = False
        current_yawn_detected = False
        current_yawn_conf = 0.0
        current_face_detected = False
        annotated_frame = frame.copy()

        if should_process_model or self.cached_results is None:
            try:
                frame_results = self.frame_processor.process_frame(frame, self.detection_config['use_preprocessing'])
                current_face_detected = frame_results and frame_results.multi_face_landmarks

                if current_face_detected:
                    annotated_frame = self.frame_processor.draw_landmarks(
                        frame.copy(),
                        frame_results,
                        self.detection_config['tesselation'],
                        self.detection_config['contours'],
                        self.detection_config['irises']
                    )

                    current_eye_results_data = self.__process_eyes(annotated_frame, frame_results)
                    current_is_drowsy = self.__check_drowsiness(current_eye_results_data)


                    enable_yawn_detection = self.detection_config.get('enable_yawn_detection', True)

                    if enable_yawn_detection and self.yawn_processor is not None and hasattr(self.yawn_processor, 'available') and self.yawn_processor.available:
                        mouth_img = self.yawn_processor.crop_mouth_region(frame, frame_results)
                        if mouth_img is not None:
                            current_yawn_detected, current_yawn_conf = self.yawn_processor.predict(mouth_img)
                            self.__update_yawn_time(current_yawn_detected)
                            yawn_alert = (current_yawn_detected and self.yawn_duration >= 1.0)
                            if yawn_alert and not self.last_yawn_state:
                                self.yawn_count += 1
                            self.last_yawn_state = yawn_alert
                        else:
                            self.__update_yawn_time(False); self.last_yawn_state = False
                    else:
                        self.__update_yawn_time(False); current_yawn_detected, current_yawn_conf = False, 0.0; self.last_yawn_state = False

                else:
                    self.__update_eyes_closed_time(False)
                    self.__update_yawn_time(False)
                    self.last_yawn_state = False
                    self.prob_history.clear()

                self.cached_results = {
                    'face_detected': current_face_detected,
                    'eye_results_data': current_eye_results_data,
                    'is_drowsy': current_is_drowsy,
                    'closed_duration': self.eyes_closed_duration,
                    'yawn': current_yawn_detected,
                    'yawn_conf': current_yawn_conf,
                    'yawn_duration': self.yawn_duration,
                    'yawn_count': self.yawn_count,
                    'max_yawn_count': self.detection_config.get('max_yawn_count', 5)
                }

            except Exception as e:
                import traceback
                traceback.print_exc()
                self.cached_results = None
                self.__update_eyes_closed_time(False)
                self.__update_yawn_time(False)
                self.last_yawn_state = False
                self.prob_history.clear()
        
        display_eye_results_data = {'avg_prob_open': 1.0, 'smoothed_prob': 1.0, 'is_closed_this_frame': False}
        display_is_drowsy = False
        display_yawn_detected = False
        display_yawn_conf = 0.0
        display_face_detected = False
        display_yawn_count = self.yawn_count
        display_max_yawn_count = self.detection_config.get('max_yawn_count', 5)

        if self.cached_results is not None:
            cached_data = self.cached_results
            display_eye_results_data = cached_data['eye_results_data']
            display_is_drowsy = cached_data['is_drowsy']
            display_yawn_detected = cached_data['yawn']
            display_yawn_conf = cached_data['yawn_conf']
            display_face_detected = cached_data['face_detected']

            display_is_drowsy = self.__check_drowsiness(display_eye_results_data)
            self.__update_yawn_time(display_yawn_detected)

            display_yawn_count = self.yawn_count
            display_max_yawn_count = cached_data['max_yawn_count']

        reset_minutes = self.detection_config.get('yawn_reset_minutes', 10)
        now = time.time()

        if self.yawn_count < display_max_yawn_count:
            self.yawn_reset_countdown = None
            self.last_yawn_reset_time = now
        else:
            if self.yawn_reset_countdown is None:
                self.yawn_reset_countdown = reset_minutes * 60
                self.last_yawn_reset_time = now
            else:
                elapsed = now - self.last_yawn_reset_time
                self.yawn_reset_countdown = max(0, reset_minutes * 60 - elapsed)
                if self.yawn_reset_countdown <= 0:
                    self.yawn_count = 0
                    self.yawn_reset_countdown = None
                    self.last_yawn_reset_time = now

        self.__add_ui_elements(
            annotated_frame,
            display_eye_results_data,
            display_is_drowsy,
            display_yawn_detected,
            display_yawn_conf,
            yawn_count=display_yawn_count,
            max_yawn_count=display_max_yawn_count,
            yawn_alert=(display_yawn_detected and self.yawn_duration >= 1.0),
            yawn_reset_countdown=self.yawn_reset_countdown
        )

        if self.cached_results is not None:
             self.cached_results.update({
                'closed_duration': self.eyes_closed_duration,
                'yawn_duration': self.yawn_duration,
                'yawn_count': self.yawn_count,
                'is_drowsy': display_is_drowsy
             })
             final_frame_to_return = annotated_frame
        else:
            final_frame_to_return = annotated_frame


        return {
            'frame': final_frame_to_return,
            'face_detected': display_face_detected,
            'eye_results_data': display_eye_results_data,
            'is_drowsy': display_is_drowsy,
            'closed_duration': self.eyes_closed_duration,
            'yawn': display_yawn_detected,
            'yawn_conf': display_yawn_conf,
            'yawn_duration': self.yawn_duration,
            'yawn_count': self.yawn_count,
            'max_yawn_count': display_max_yawn_count
        }
        
    def __process_eyes(self, frame, face):
        """
        Gọi EyeProcessor để lấy xác suất mắt mở, làm mượt tín hiệu và ra quyết định.
        """
        try:
            if face.multi_face_landmarks:
                avg_prob_open = self.eye_processor.detect_eyes(frame, face.multi_face_landmarks[0])

                self.prob_history.append(avg_prob_open)

                smoothed_prob = sum(self.prob_history) / len(self.prob_history)

                is_closed_this_frame = smoothed_prob <= self.SMOOTHED_PROB_THRESHOLD

                return {
                    'avg_prob_open': avg_prob_open,
                    'smoothed_prob': smoothed_prob,
                    'is_closed_this_frame': is_closed_this_frame
                }
        except Exception as e:
            print(f"Eye processing error in DriverMonitor: {e}")

        return {
            'avg_prob_open': 1.0,
            'smoothed_prob': 1.0,
            'is_closed_this_frame': False
        }

    def __add_ui_elements(self, frame, eye_results_data, is_drowsy, yawn, yawn_conf, yawn_count=0, max_yawn_count=5, yawn_alert=False, yawn_reset_countdown=None):
        """
        Vẽ các thông tin giám sát lên khung hình.
        """
        is_closed_this_frame = eye_results_data['is_closed_this_frame']

        eye_state_text = "CLOSED" if is_closed_this_frame else "OPEN"
        cv2.putText(frame, f"Eyes: {eye_state_text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        time_text = f"Closed: {self.eyes_closed_duration:.1f}s / {self.SLEEP_TH:.1f}s"
        cv2.putText(frame, time_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        yawn_text = f"Yawn: {yawn} ({yawn_conf:.2f}) | Yawn time: {self.yawn_duration:.1f}s"
        cv2.putText(frame, yawn_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        if self.detection_config.get('enable_yawn_detection', True):
            max_yawn_count = self.detection_config.get('max_yawn_count', 5)
            yawn_count_text = f"Yawn count: {yawn_count}/{max_yawn_count}"
            cv2.putText(frame, yawn_count_text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    
    def update_config(self, **kwargs):
        """
        Cập nhật cấu hình giám sát từ bên ngoài.
        """
        changed = False
        for k, v in kwargs.items():
            if k == 'sleep_threshold':
                if self.SLEEP_TH != v:
                    self.SLEEP_TH = v
                    changed = True
            elif k == 'eye_prediction_threshold':
                if self.eye_processor.PREDICTION_THRESHOLD != v:
                    self.eye_processor.set_prediction_threshold(v)
                    self.detection_config[k] = v
                    changed = True
            elif self.detection_config.get(k) != v:
                self.detection_config[k] = v
                changed = True

        if changed and hasattr(self.yawn_processor, "enable") and hasattr(self.yawn_processor, "disable"):
            enabled = self.detection_config.get("enable_yawn_detection", True)
            if enabled:
                self.yawn_processor.enable()
            else:
                self.yawn_processor.disable()

        if 'max_yawn_count' in kwargs or 'yawn_reset_minutes' in kwargs:
            self.last_yawn_reset_time = time.time()
            max_yawn_count = self.detection_config.get('max_yawn_count', kwargs.get('max_yawn_count'))
            yawn_reset_minutes = self.detection_config.get('yawn_reset_minutes', kwargs.get('yawn_reset_minutes'))
            if self.yawn_count >= max_yawn_count:
                self.yawn_reset_countdown = yawn_reset_minutes * 60
    def reset_timer(self):
        """Reset timer cho mắt nhắm và bộ đếm miss."""
        try:
            self.eyes_closed_start_time = None
            self.eyes_closed_duration = 0.0
            self.miss_counter = 0 
        except Exception as e:
            print(f"Reset error: {e}")

    def reset_yawn_count(self):
        """Reset số lần ngáp về 0"""
        self.yawn_count = 0
