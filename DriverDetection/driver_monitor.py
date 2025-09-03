from .frame_processor import FrameProcessor
from .eye_processor import EyeProcessor
from .yawn_processor import YawnProcessor
import cv2, threading, os, time
from playsound import playsound

class DriverMonitor:
    def __init__(self, sleep_th=2, alarm_file="alarm.wav"):
        self.frame_processor = FrameProcessor()
        self.eye_processor = EyeProcessor()
        self.yawn_processor = YawnProcessor()

        self.detection_config = {
            'tesselation': False,
            'contours': True,
            'irises': False,
            'use_preprocessing': True
        }

        self.alert = None
        self.SLEEP_TH = sleep_th
        self.alarm_playing = False

        self.alarm_file = alarm_file
        self.alarm_thread = None
        self.stop_alarm_flag = False

        self.eyes_closed_start_time = None
        self.eyes_closed_duration = 0.0

        self.yawn_start_time = None
        self.yawn_duration = 0.0

    def __is_alarm_playing(self):
        return self.alarm_playing
    
    def __start_playing_alarm(self):
        """Start alarm với playsound"""
        if not os.path.exists(self.alarm_file):
            return
            
        if not self.alarm_playing:
            try:
                self.alarm_playing = True
                self.stop_alarm_flag = False
                
                self.alarm_thread = threading.Thread(target=self.__alarm_worker, daemon=True)
                self.alarm_thread.start()
                
            except Exception as e:
                print(f"starting alaerm error: {e}")
                self.alarm_playing = False
    
    def __alarm_worker(self):
        """Worker thread cho playsound alarm"""
        try:
            while self.alarm_playing and not self.stop_alarm_flag:
                try:
                    playsound(self.alarm_file, block=False)

                    for _ in range(20):
                        if self.stop_alarm_flag or not self.alarm_playing:
                            break
                        time.sleep(0.1)
                        
                except Exception as e:
                    print(f"Playsound error: {e}")
                    break
                    
        except Exception as e:
            print(f"alarm worker error: {e}")
        finally:
            self.alarm_playing = False

    def __stop_alarm(self):
        """Stop alarm"""
        if self.alarm_playing:
            try:
                self.alarm_playing = False
                self.stop_alarm_flag = True
            except Exception as e:
                print(f"stop alarm error: {e}")

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
            if hasattr(self.yawn_processor, 'available') and self.yawn_processor.available:
                if frame_results.multi_face_landmarks:
                    mouth_img = self.yawn_processor.crop_mouth_region(frame, frame_results)
                    if mouth_img is not None:
                        yawn_detected, yawn_conf = self.yawn_processor.predict(mouth_img)
                        self.__update_yawn_time(yawn_detected)
                    else:
                        self.yawn_start_time = None
                        self.yawn_duration = 0.0
                else:
                    self.yawn_start_time = None
                    self.yawn_duration = 0.0
            else:
                # Yawn processor not available, disable yawn detection
                self.yawn_start_time = None
                self.yawn_duration = 0.0

            try:
                if is_drowsy and not self.__is_alarm_playing():
                    self.__start_playing_alarm()
                elif not is_drowsy and self.__is_alarm_playing():
                    self.__stop_alarm()
            except Exception as alarm_error:
                print(f"alarm error: {alarm_error}")
                self.alarm_playing = False

            self.__add_ui_elements(annotated_frame, eye_results, is_drowsy, yawn_detected, yawn_conf)
            return {
                'frame': annotated_frame,
                'face_detected': frame_results.multi_face_landmarks is not None,
                'eye_results': eye_results,
                'is_drowsy': is_drowsy,
                'closed_duration': self.eyes_closed_duration,
                'yawn': yawn_detected,
                'yawn_conf': yawn_conf,
                'yawn_duration': self.yawn_duration
            }
            
        except Exception as e:
            print(f"process frame error: {e}")
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
                'yawn_duration': 0.0
            }
    
    def __process_eyes(self, frame, face):
        try:
            if face.multi_face_landmarks:
                left_state, right_state, left_conf, right_conf = self.eye_processor.detect_eyes(frame, face.multi_face_landmarks[0])
                return {
                    'left_eye': {'state': left_state, 'confidence': left_conf},
                    'right_eye': {'state': right_state, 'confidence': right_conf},
                    'closed': left_state == 'closed' and right_state == 'closed'
                }
        except Exception as e:
            print(f"Eye processing error: {e}")
        
        return {
            'left_eye': {'state': 'unknown', 'confidence': 0.0},
            'right_eye': {'state': 'unknown', 'confidence': 0.0},
            'closed': False
        }

    def __add_ui_elements(self, frame, eye_results, is_drowsy, yawn, yawn_conf):
        left_state = eye_results['left_eye']['state']
        right_state = eye_results['right_eye']['state']
        eye_text = f"Eyes: L={left_state} R={right_state}"
        cv2.putText(frame, eye_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        time_text = f"Closed time: {self.eyes_closed_duration:.1f}s / {self.SLEEP_TH:.1f}s"
        cv2.putText(frame, time_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        yawn_text = f"Yawn: {yawn} ({yawn_conf:.2f}) | Yawn time: {self.yawn_duration:.1f}s"
        cv2.putText(frame, yawn_text, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        if self.yawn_duration >= 2.0:
            cv2.putText(frame, "Ban co dau hieu buon ngu!", (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        if is_drowsy:
            cv2.putText(frame, "Day di, day di!", (10, 140), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    def update_config(self, **kwargs):
        """Update detection configuration"""
        self.detection_config.update(kwargs)
    
    def reset_timer(self):
        """Reset timer và stop alarm"""
        try:
            self.eyes_closed_start_time = None
            self.eyes_closed_duration = 0.0
            self.__stop_alarm()
        except Exception as e:
            print(f"Reset error: {e}")

    def cleanup(self):
        """Safe cleanup"""
        try:
            self.__stop_alarm()
            if self.alarm_thread and self.alarm_thread.is_alive():
                time.sleep(0.5)
        except Exception as e:
            print(f"Cleanup error: {e}")