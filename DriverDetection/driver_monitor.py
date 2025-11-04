import cv2
import time
import threading
from collections import deque
from .frame_processor import FrameProcessor
from .eye_processor import EyeProcessor
from .yawn_processor import YawnProcessor
from .head_pose_estimator import HeadPoseEstimator # New import

class DriverMonitor:
    def __init__(self, sleep_th=2.0):
        self.frame_processor = FrameProcessor()
        self.eye_processor = EyeProcessor()
        self.yawn_processor = YawnProcessor()
        self.head_pose_estimator = HeadPoseEstimator() # New instance

        self.detection_config = {
            'tesselation': False,
            'contours': False,
            'irises': False,
            'use_preprocessing': True,
            'enable_yawn_detection': True,
            'max_yawn_count': 5,  
            'yawn_reset_minutes': 10,
            'eye_prediction_threshold': 0.3,
            'pitch_threshold': 20, # New: Ngưỡng góc Pitch (độ) để phát hiện nhìn xuống/lên
            'yaw_threshold': 30,   # New: Ngưỡng góc Yaw (độ) để phát hiện quay đầu
            'distraction_time_threshold': 3.0, # New: Ngưỡng thời gian mất tập trung (giây)
            'w1_eye': 50, # Trọng số cho điểm mắt
            'w2_yawn': 20, # Trọng số cho điểm ngáp
            'w3_distraction': 30 # Trọng số cho điểm mất tập trung
        }

        # --- Logic làm mượt tín hiệu mắt ---
        self.prob_history = deque(maxlen=5) # Lưu 5 xác suất mắt mở gần nhất
        self.SMOOTHED_PROB_THRESHOLD = 0.4 # Ngưỡng cho giá trị xác suất đã được làm mượt

        # --- Logic phát hiện buồn ngủ (mắt nhắm) ---
        self.SLEEP_TH = sleep_th
        self.eyes_closed_start_time = None
        self.eyes_closed_duration = 0.0

        # --- Logic phát hiện ngáp ---
        self.yawn_start_time = None
        self.yawn_duration = 0.0
        self.yawn_count = 0 
        self.last_yawn_state = False
        self.last_yawn_reset_time = time.time()
        self.yawn_reset_countdown = None  

        # --- Logic phát hiện mất tập trung (quay đầu) ---
        self.head_turned_away_start_time = None # New
        self.head_turned_away_duration = 0.0    # New
        self.current_pitch = 0.0 # New: Lưu trữ giá trị pitch hiện tại
        self.current_yaw = 0.0   # New: Lưu trữ giá trị yaw hiện tại
        self.current_roll = 0.0  # New: Lưu trữ giá trị roll hiện tại

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

    def __update_head_turned_away_time(self, is_head_turned_away_this_frame):
        """
        Cập nhật bộ đếm thời gian đầu quay đi.
        """
        current_time = time.time()

        if is_head_turned_away_this_frame:
            if self.head_turned_away_start_time is None:
                self.head_turned_away_start_time = current_time
            self.head_turned_away_duration = current_time - self.head_turned_away_start_time
        else:
            self.head_turned_away_start_time = None
            self.head_turned_away_duration = 0.0

    def __correct_eye_prob_for_pose(self, avg_prob_open, yaw):
        """
        Điều chỉnh xác suất mắt mở dựa trên góc Yaw của đầu.
        Nếu đầu quay nhiều, tăng nhẹ xác suất để bù lại sự biến dạng của ảnh mắt.
        """
        yaw_threshold_for_correction = self.detection_config.get('yaw_threshold', 30) * 0.5 # Ví dụ: bắt đầu hiệu chỉnh từ 15 độ
        correction_factor = 0.0

        if abs(yaw) > yaw_threshold_for_correction:
            # Tăng hệ số hiệu chỉnh tuyến tính theo độ lớn của góc yaw
            correction_factor = (abs(yaw) - yaw_threshold_for_correction) / (self.detection_config.get('yaw_threshold', 30) * 2) # Max correction at 60 degrees
            correction_factor = min(correction_factor, 0.2) # Giới hạn hệ số hiệu chỉnh tối đa

        corrected_prob = avg_prob_open + correction_factor
        return min(corrected_prob, 1.0) # Đảm bảo xác suất không vượt quá 1.0

    def __calculate_drowsiness_score(self, smoothed_corrected_prob, yawn_count, head_turned_away_duration):
        """
        Tính toán điểm buồn ngủ tổng hợp dựa trên nhiều yếu tố.
        Điểm càng cao càng buồn ngủ/mất tập trung.
        """
        score = 0.0

        # --- Eye Score ---
        # Nếu mắt nhắm (xác suất mở thấp), điểm mắt tăng
        eye_drowsiness_factor = 1.0 - smoothed_corrected_prob # 0 nếu mắt mở hoàn toàn, 1 nếu mắt nhắm hoàn toàn
        score += self.detection_config.get('w1_eye', 50) * eye_drowsiness_factor

        # --- Yawn Score ---
        # Điểm ngáp tăng theo số lần ngáp
        score += self.detection_config.get('w2_yawn', 20) * min(yawn_count, self.detection_config.get('max_yawn_count', 5)) / self.detection_config.get('max_yawn_count', 5)

        # --- Distraction Score ---
        # Điểm mất tập trung tăng theo thời gian đầu quay đi
        distraction_time_threshold = self.detection_config.get('distraction_time_threshold', 3.0)
        if head_turned_away_duration > 0:
            distraction_factor = min(head_turned_away_duration / distraction_time_threshold, 1.0)
            score += self.detection_config.get('w3_distraction', 30) * distraction_factor
        
        # Giới hạn điểm từ 0 đến 100
        return min(max(score, 0), 100)

    def process_frame(self, frame):
        """
        Xử lý từng khung hình từ camera, phát hiện buồn ngủ và ngáp.
        Đảm bảo UI được vẽ trên mọi khung hình để tránh nhấp nháy.
        """
        self.frame_count += 1
        should_process_model = (self.frame_count % (self.SKIP_FRAMES + 1)) == 0

        # Khởi tạo các giá trị mặc định
        current_eye_results_data = {'avg_prob_open': 1.0, 'corrected_prob_open': 1.0, 'smoothed_prob': 1.0, 'is_closed_this_frame': False}
        current_is_drowsy = False
        current_yawn_detected = False
        current_yawn_conf = 0.0
        current_face_detected = False
        current_pitch, current_yaw, current_roll = 0.0, 0.0, 0.0 # NEW
        current_is_head_turned_away = False # NEW
        current_drowsiness_score = 0.0 # NEW
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

                    # NEW: Ước tính tư thế đầu
                    img_h, img_w, _ = frame.shape
                    current_pitch, current_yaw, current_roll = self.head_pose_estimator.process_landmarks(
                        frame_results.multi_face_landmarks[0], img_w, img_h
                    )
                    self.current_pitch, self.current_yaw, self.current_roll = current_pitch, current_yaw, current_roll

                    # NEW: Kiểm tra đầu quay đi
                    if current_pitch is not None and current_yaw is not None:
                        pitch_threshold = self.detection_config.get('pitch_threshold', 20)
                        yaw_threshold = self.detection_config.get('yaw_threshold', 30)
                        current_is_head_turned_away = (abs(current_pitch) > pitch_threshold) or (abs(current_yaw) > yaw_threshold)
                    else:
                        current_is_head_turned_away = False # Không phát hiện được góc thì không coi là quay đi

                    self.__update_head_turned_away_time(current_is_head_turned_away) # NEW

                    # NEW: Truyền yaw vào __process_eyes
                    current_eye_results_data = self.__process_eyes(annotated_frame, frame_results, current_yaw)
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
                    
                    # NEW: Tính toán điểm buồn ngủ tổng hợp
                    current_drowsiness_score = self.__calculate_drowsiness_score(
                        current_eye_results_data['smoothed_prob'],
                        self.yawn_count,
                        self.head_turned_away_duration
                    )

                else: # No face detected
                    self.__update_eyes_closed_time(False)
                    self.__update_yawn_time(False)
                    self.__update_head_turned_away_time(False) # NEW
                    self.last_yawn_state = False
                    self.prob_history.clear()
                    self.current_pitch, self.current_yaw, self.current_roll = 0.0, 0.0, 0.0 # Reset angles
                    self.head_turned_away_duration = 0.0 # Reset duration
                    current_drowsiness_score = 0.0 # Reset score

                self.cached_results = {
                    'face_detected': current_face_detected,
                    'eye_results_data': current_eye_results_data,
                    'is_drowsy': current_is_drowsy,
                    'closed_duration': self.eyes_closed_duration,
                    'yawn': current_yawn_detected,
                    'yawn_conf': current_yawn_conf,
                    'yawn_duration': self.yawn_duration,
                    'yawn_count': self.yawn_count,
                    'max_yawn_count': self.detection_config.get('max_yawn_count', 5),
                    'pitch': self.current_pitch, # NEW
                    'yaw': self.current_yaw,     # NEW
                    'roll': self.current_roll,   # NEW
                    'is_head_turned_away': current_is_head_turned_away, # NEW
                    'head_turned_away_duration': self.head_turned_away_duration, # NEW
                    'drowsiness_score': current_drowsiness_score # NEW
                }

            except Exception as e:
                import traceback
                traceback.print_exc()
                self.cached_results = None
                self.__update_eyes_closed_time(False)
                self.__update_yawn_time(False)
                self.__update_head_turned_away_time(False) # NEW
                self.last_yawn_state = False
                self.prob_history.clear()
                self.current_pitch, self.current_yaw, self.current_roll = 0.0, 0.0, 0.0 # Reset angles
                self.head_turned_away_duration = 0.0 # Reset duration
                current_drowsiness_score = 0.0 # Reset score
        
        # --- Display logic (uses cached_results) ---
        display_eye_results_data = {'avg_prob_open': 1.0, 'corrected_prob_open': 1.0, 'smoothed_prob': 1.0, 'is_closed_this_frame': False}
        display_is_drowsy = False
        display_yawn_detected = False
        display_yawn_conf = 0.0
        display_face_detected = False
        display_yawn_count = self.yawn_count
        display_max_yawn_count = self.detection_config.get('max_yawn_count', 5)
        display_pitch, display_yaw, display_roll = 0.0, 0.0, 0.0 # NEW
        display_is_head_turned_away = False # NEW
        display_head_turned_away_duration = 0.0 # NEW
        display_drowsiness_score = 0.0 # NEW

        if self.cached_results is not None:
            cached_data = self.cached_results
            display_eye_results_data = cached_data['eye_results_data']
            display_is_drowsy = cached_data['is_drowsy']
            display_yawn_detected = cached_data['yawn']
            display_yawn_conf = cached_data['yawn_conf']
            display_face_detected = cached_data['face_detected']
            display_pitch = cached_data['pitch'] # NEW
            display_yaw = cached_data['yaw']     # NEW
            display_roll = cached_data['roll']   # NEW
            display_is_head_turned_away = cached_data['is_head_turned_away'] # NEW
            display_head_turned_away_duration = cached_data['head_turned_away_duration'] # NEW
            display_drowsiness_score = cached_data['drowsiness_score'] # NEW

            # These lines are redundant now as the logic is in should_process_model block
            # display_is_drowsy = self.__check_drowsiness(display_eye_results_data)
            # self.__update_yawn_time(display_yawn_detected)

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
            yawn_reset_countdown=self.yawn_reset_countdown,
            pitch=display_pitch, # NEW
            yaw=display_yaw,     # NEW
            roll=display_roll,   # NEW
            is_head_turned_away=display_is_head_turned_away, # NEW
            head_turned_away_duration=display_head_turned_away_duration, # NEW
            drowsiness_score=display_drowsiness_score # NEW
        )

        if self.cached_results is not None:
             self.cached_results.update({
                'closed_duration': self.eyes_closed_duration,
                'yawn_duration': self.yawn_duration,
                'yawn_count': self.yawn_count,
                'is_drowsy': display_is_drowsy,
                'pitch': self.current_pitch, # NEW
                'yaw': self.current_yaw,     # NEW
                'roll': self.current_roll,   # NEW
                'is_head_turned_away': current_is_head_turned_away, # NEW
                'head_turned_away_duration': self.head_turned_away_duration, # NEW
                'drowsiness_score': current_drowsiness_score # NEW
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
            'max_yawn_count': display_max_yawn_count,
            'pitch': display_pitch, # NEW
            'yaw': display_yaw,     # NEW
            'roll': display_roll,   # NEW
            'is_head_turned_away': display_is_head_turned_away, # NEW
            'head_turned_away_duration': display_head_turned_away_duration, # NEW
            'drowsiness_score': display_drowsiness_score # NEW
        }
        
    def __process_eyes(self, frame, face, yaw=0.0): # Added yaw=0.0
        """
        Gọi EyeProcessor để lấy xác suất mắt mở, làm mượt tín hiệu và ra quyết định.
        """
        try:
            if face.multi_face_landmarks:
                avg_prob_open = self.eye_processor.detect_eyes(frame, face.multi_face_landmarks[0])

                # NEW: Điều chỉnh xác suất mắt dựa trên góc Yaw
                corrected_prob_open = self.__correct_eye_prob_for_pose(avg_prob_open, yaw)

                self.prob_history.append(corrected_prob_open) # Use corrected_prob_open

                smoothed_prob = sum(self.prob_history) / len(self.prob_history)

                is_closed_this_frame = smoothed_prob <= self.SMOOTHED_PROB_THRESHOLD

                return {
                    'avg_prob_open': avg_prob_open,
                    'corrected_prob_open': corrected_prob_open, # New
                    'smoothed_prob': smoothed_prob,
                    'is_closed_this_frame': is_closed_this_frame
                }
        except Exception as e:
            print(f"Eye processing error in DriverMonitor: {e}")

        return {
            'avg_prob_open': 1.0,
            'corrected_prob_open': 1.0, # New
            'smoothed_prob': 1.0,
            'is_closed_this_frame': False
        }

    def __add_ui_elements(self, frame, eye_results_data, is_drowsy, yawn, yawn_conf, yawn_count=0, max_yawn_count=5, yawn_alert=False, yawn_reset_countdown=None, pitch=0.0, yaw=0.0, roll=0.0, is_head_turned_away=False, head_turned_away_duration=0.0, drowsiness_score=0.0):
        """
        Vẽ các thông tin giám sát lên khung hình.
        """
        # --- Hiển thị điểm buồn ngủ --- 
        score_text = f"Drowsiness Score: {drowsiness_score:.1f}"
        score_color = (0, 255, 255) # Vàng
        if drowsiness_score > 70:
            score_color = (0, 0, 255) # Đỏ
        elif drowsiness_score > 40:
            score_color = (0, 165, 255) # Cam
        cv2.putText(frame, score_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, score_color, 2)

        # --- Hiển thị thông tin mắt ---
        is_closed_this_frame = eye_results_data.get('is_closed_this_frame', False)
        eye_state_text = "CLOSED" if is_closed_this_frame else "OPEN"
        eye_prob_text = f"(Prob: {eye_results_data.get('smoothed_prob', 0.0):.2f})"
        cv2.putText(frame, f"Eyes: {eye_state_text} {eye_prob_text}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # --- Hiển thị thông tin ngáp ---
        if self.detection_config.get('enable_yawn_detection', True):
            yawn_text = f"Yawn Count: {yawn_count}/{max_yawn_count}"
            cv2.putText(frame, yawn_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            if yawn_reset_countdown is not None:
                reset_text = f"(Resets in: {int(yawn_reset_countdown)}s)"
                cv2.putText(frame, reset_text, (200, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # --- Hiển thị thông tin tư thế đầu và mất tập trung ---
        pose_text = f"Pitch: {pitch:.1f}, Yaw: {yaw:.1f}"
        distraction_text = f"Distracted: {head_turned_away_duration:.1f}s"
        pose_color = (0, 0, 255) if is_head_turned_away else (255, 255, 255)
        cv2.putText(frame, pose_text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, pose_color, 2)
        cv2.putText(frame, distraction_text, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, pose_color, 2)
    
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
