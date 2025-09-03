"""
Safe version của main_app.py với lazy loading để tránh lỗi import
Chỉ import TensorFlow/MediaPipe khi thực sự cần
"""

import sys
import os
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QPushButton, QCheckBox,
                             QSpinBox, QFileDialog, QGroupBox, QGridLayout,
                             QSlider, QLineEdit, QFrame, QMessageBox, QShortcut)
from PyQt5.QtCore import QTimer, Qt, pyqtSignal, QThread, QPropertyAnimation, QRect, QEasingCurve
from PyQt5.QtGui import QImage, QPixmap, QFont, QPainter, QColor, QKeySequence
from PIL import Image

# Lazy import for YawnProcessor (avoid import errors)
def get_yawn_processor():
    """Lazy load YawnProcessor to avoid import errors"""
    try:
        from DriverDetection.yawn_processor import YawnProcessor
        return YawnProcessor
    except ImportError as e:
        print(f"⚠️ Could not import YawnProcessor: {e}")
        return None

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

class SafeCameraThread(QThread):
    """Thread an toàn với lazy loading"""
    frame_ready = pyqtSignal(np.ndarray)
    eye_regions_ready = pyqtSignal(np.ndarray)  # Signal for eye regions
    yawn_frame_ready = pyqtSignal(np.ndarray)   # Signal for yawn camera
    error_occurred = pyqtSignal(str)
    
    def __init__(self, camera_id=0):
        super().__init__()
        self.camera_id = camera_id
        self.running = False
        self.cap = None
        self.monitor = None
        self._ai_loaded = False
        # Yawn detection state
        self.yawn_enabled = False
        self.yawn_processor = None

    # ---- Control helpers (thread-safe enough for simple flags) ----
    def toggle_detection_flag(self, flag_name: str):
        try:
            if self.monitor is None:
                return
            current = bool(self.monitor.detection_config.get(flag_name, False))
            self.monitor.detection_config[flag_name] = not current
            print(f"⚙️  {flag_name} -> {not current}")
        except Exception as e:
            print(f"toggle flag error: {e}")

    def reset_landmarks_flags(self):
        try:
            if self.monitor is None:
                return
            self.monitor.detection_config.update({
                'tesselation': True,
                'contours': True,
                'irises': True
            })
            print("⚙️  Reset landmarks flags -> all ON")
        except Exception as e:
            print(f"reset flags error: {e}")

    def reset_timer_and_alarm(self):
        try:
            if self.monitor is None:
                return
            self.monitor.reset_timer()
            print("🛑 Alarm/Timer reset")
        except Exception as e:
            print(f"reset timer error: {e}")
        
    def load_ai_modules(self):
        """Lazy load AI modules"""
        if self._ai_loaded:
            return True
            
        try:
            # Try to import AI modules
            current_dir = os.path.dirname(os.path.abspath(__file__))
            parent_dir = os.path.dirname(current_dir)
            if parent_dir not in sys.path:
                sys.path.append(parent_dir)
                
            from DriverDetection.driver_monitor import DriverMonitor
            
            # Test model loading
            self.monitor = DriverMonitor()
            print("✅ AI modules loaded successfully")
            self._ai_loaded = True
            return True
            
        except Exception as e:
            print(f"❌ AI modules failed to load: {e}")
            self.error_occurred.emit(f"AI modules error: {str(e)[:100]}")
            return False

    def load_yawn_processor(self):
        """Load YawnProcessor once when needed"""
        if self.yawn_processor is not None:
            return self.yawn_processor.available
        try:
            YawnProcessorClass = get_yawn_processor()
            if YawnProcessorClass is None:
                print("❌ YawnProcessor class not available")
                return False

            self.yawn_processor = YawnProcessorClass()
            print(f"✅ YawnProcessor loaded, available: {self.yawn_processor.available}")
            return self.yawn_processor.available
        except Exception as e:
            print(f"❌ Load YawnProcessor error: {e}")
            self.yawn_processor = None
            return False
    
    def extract_eye_regions(self, frame, frame_results):
        """Extract và combine eye regions thành một frame"""
        try:
            if not frame_results.multi_face_landmarks:
                return None
                
            face_landmarks = frame_results.multi_face_landmarks[0]
            
            # Eye landmark indices từ MediaPipe
            LEFT_EYE_INDICES = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
            RIGHT_EYE_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
            
            height, width = frame.shape[:2]
            
            # Extract left eye region
            left_eye_points = []
            for i in LEFT_EYE_INDICES:
                if i < len(face_landmarks.landmark):
                    x = int(face_landmarks.landmark[i].x * width)
                    y = int(face_landmarks.landmark[i].y * height)
                    left_eye_points.append((x, y))
            
            # Extract right eye region  
            right_eye_points = []
            for i in RIGHT_EYE_INDICES:
                if i < len(face_landmarks.landmark):
                    x = int(face_landmarks.landmark[i].x * width)
                    y = int(face_landmarks.landmark[i].y * height)
                    right_eye_points.append((x, y))
                    
            # Crop eye regions
            left_eye_crop = self.crop_eye_region(frame, left_eye_points)
            right_eye_crop = self.crop_eye_region(frame, right_eye_points)

            # Handle cases where one or both eyes are None
            if left_eye_crop is None and right_eye_crop is None:
                return None

            # Create placeholder if eye is None
            placeholder = np.zeros((100, 200, 3), dtype=np.uint8)
            cv2.putText(placeholder, "Eye not detected", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Use placeholder if eye crop is None
            if left_eye_crop is None:
                left_eye_crop = placeholder.copy()
            if right_eye_crop is None:
                right_eye_crop = placeholder.copy()

            # Resize eye crops để consistent size (only if not already the right size)
            if left_eye_crop.shape[:2] != (100, 200):
                left_eye_resized = cv2.resize(left_eye_crop, (200, 100))
            else:
                left_eye_resized = left_eye_crop

            if right_eye_crop.shape[:2] != (100, 200):
                right_eye_resized = cv2.resize(right_eye_crop, (200, 100))
            else:
                right_eye_resized = right_eye_crop

            # NOTE: Khung hình đã bị lật ngang (cv2.flip(frame, 1)),
            # nên mắt "trái" theo landmark sẽ xuất hiện bên phải màn hình.
            # Để hiển thị đúng trực quan: ô bên trái hiển thị mắt trái (theo người dùng),
            # ta cần đặt thứ tự [right_eye, left_eye].
            combined_eyes = np.hstack([right_eye_resized, left_eye_resized])

            # Scale up để dễ nhìn
            final_eyes = cv2.resize(combined_eyes, (640, 160))
            return final_eyes
                
        except Exception as e:
            print(f"Eye extraction error: {e}")
            
        return None
    
    def crop_eye_region(self, frame, eye_points):
        """Crop eye region từ points"""
        if len(eye_points) < 4:
            return None
            
        try:
            xs, ys = zip(*eye_points)
            x_min, x_max = max(0, int(min(xs)) - 20), min(frame.shape[1], int(max(xs)) + 20)
            y_min, y_max = max(0, int(min(ys)) - 15), min(frame.shape[0], int(max(ys)) + 15)
            
            if x_max <= x_min or y_max <= y_min:
                return None
                
            crop = frame[y_min:y_max, x_min:x_max]
            return crop if crop.size > 0 else None
            
        except Exception as e:
            print(f"Crop error: {e}")
            return None

    def extract_mouth_region(self, frame, frame_results):
        """Extract mouth (lips) region based on FACEMESH_LIPS connections"""
        try:
            if not frame_results.multi_face_landmarks:
                return None
            face_landmarks = frame_results.multi_face_landmarks[0]
            height, width = frame.shape[:2]

            # Collect unique indices from FACEMESH_LIPS connections
            mp_face_mesh = self.monitor.frame_processor.mediapipe_face_mesh
            lips_conns = mp_face_mesh.FACEMESH_LIPS
            mouth_indices = set()
            for a, b in lips_conns:
                mouth_indices.add(a)
                mouth_indices.add(b)

            xs, ys = [], []
            for idx in mouth_indices:
                if idx < len(face_landmarks.landmark):
                    xs.append(int(face_landmarks.landmark[idx].x * width))
                    ys.append(int(face_landmarks.landmark[idx].y * height))

            if not xs or not ys:
                return None

            x_min, x_max = max(0, min(xs) - 20), min(width, max(xs) + 20)
            y_min, y_max = max(0, min(ys) - 20), min(height, max(ys) + 20)
            if x_max <= x_min or y_max <= y_min:
                return None
            crop = frame[y_min:y_max, x_min:x_max]
            return crop if crop.size > 0 else None
        except Exception as e:
            print(f"Mouth extraction error: {e}")
            return None
        
    def start_camera(self):
        self.running = True
        if not self.isRunning():
            self.start()
    
    def stop_camera(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self.wait()
    
    def run(self):
        # Try to load AI modules
        ai_available = self.load_ai_modules()
        
        self.cap = cv2.VideoCapture(self.camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                
                if ai_available and self.monitor:
                    try:
                        # Process frame with AI
                        results = self.monitor.process_frame(frame)
                        
                        # Get frame processing results from FrameProcessor
                        frame_results = self.monitor.frame_processor.process_frame(frame, True)
                        
                        # Extract eye regions
                        eye_regions = self.extract_eye_regions(frame, frame_results)
                        
                        if eye_regions is not None:
                            # Add eye state info to eye regions
                            eye_results = results.get('eye_results', {})
                            left_state = eye_results.get('left_eye', {}).get('state', 'unknown')
                            right_state = eye_results.get('right_eye', {}).get('state', 'unknown')
                            left_conf = eye_results.get('left_eye', {}).get('confidence', 0.0)
                            right_conf = eye_results.get('right_eye', {}).get('confidence', 0.0)
                            
                            # Add text overlay (đã đảo vị trí hiển thị)
                            # Bên trái màn hình: mắt trái theo người dùng => right_state
                            cv2.putText(eye_regions, f"Left: {right_state} ({right_conf:.2f})", 
                                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                            # Bên phải màn hình: mắt phải theo người dùng => left_state
                            cv2.putText(eye_regions, f"Right: {left_state} ({left_conf:.2f})", 
                                       (330, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                            
                            self.eye_regions_ready.emit(eye_regions)
                        else:
                            # Fallback eye display
                            fallback_eye = np.zeros((160, 640, 3), dtype=np.uint8)
                            cv2.putText(fallback_eye, "No face detected", 
                                       (250, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                            self.eye_regions_ready.emit(fallback_eye)
                        
                        # Still emit full frame for debugging
                        self.frame_ready.emit(results['frame'])
                        
                        # Yawn detection pipeline (optional)
                        if self.yawn_enabled and self.load_yawn_processor() and self.yawn_processor.available:
                            mouth = self.extract_mouth_region(frame, frame_results)
                            if mouth is not None:
                                try:
                                    # Sử dụng YawnProcessor để predict
                                    is_yawn, prob = self.yawn_processor.predict(mouth)
                                    # Build display frame for yawn camera
                                    display = cv2.resize(mouth, (640, 360))
                                    text = f"YAWN: {'YES' if is_yawn else 'NO'}  prob={prob:.2f}"
                                    color = (0, 0, 255) if is_yawn else (0, 255, 0)
                                    cv2.putText(display, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                                    # Emit to yawn camera
                                    self.yawn_frame_ready.emit(display)
                                except Exception as ye:
                                    # On any error, send placeholder
                                    fallback_yawn = np.zeros((360, 640, 3), dtype=np.uint8)
                                    cv2.putText(fallback_yawn, f"Yawn error: {str(ye)[:30]}", (80, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                                    self.yawn_frame_ready.emit(fallback_yawn)
                            else:
                                placeholder = np.zeros((360, 640, 3), dtype=np.uint8)
                                cv2.putText(placeholder, "Mouth not detected", (180, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
                                self.yawn_frame_ready.emit(placeholder)
                        elif self.yawn_enabled:
                            # Yawn enabled but processor not available
                            placeholder = np.zeros((360, 640, 3), dtype=np.uint8)
                            cv2.putText(placeholder, "Model not available", (220, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 100, 100), 2)
                            self.yawn_frame_ready.emit(placeholder)

                    except Exception as e:
                        # Fallback to basic frame
                        cv2.putText(frame, f"AI Error: {str(e)[:50]}", 
                                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        self.frame_ready.emit(frame)
                        
                        # Fallback eye display
                        fallback_eye = np.zeros((160, 640, 3), dtype=np.uint8)
                        cv2.putText(fallback_eye, f"AI Error: {str(e)[:30]}", 
                                   (200, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        self.eye_regions_ready.emit(fallback_eye)
                else:
                    # Basic frame without AI
                    cv2.putText(frame, "Camera OK - AI modules not loaded",
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, "Run fix_dependencies.py to fix AI",
                               (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    self.frame_ready.emit(frame)

                    # Emit placeholder for eye regions
                    placeholder_eye = np.zeros((160, 640, 3), dtype=np.uint8)
                    cv2.putText(placeholder_eye, "AI not loaded - Cannot detect eyes",
                               (150, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    self.eye_regions_ready.emit(placeholder_eye)

                    # Emit placeholder for yawn camera if enabled
                    if self.yawn_enabled:
                        placeholder_yawn = np.zeros((360, 640, 3), dtype=np.uint8)
                        cv2.putText(placeholder_yawn, "AI not available",
                                   (240, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 100, 100), 2)
                        self.yawn_frame_ready.emit(placeholder_yawn)
                    
            self.msleep(33)  # ~30 FPS
        
        if self.cap:
            self.cap.release()

class SafeDriverDetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.setup_camera()
        self.load_placeholder_image()
        self.ai_available = False
        self.check_ai_availability()
        
    def check_ai_availability(self):
        """Kiểm tra AI modules có sẵn không"""
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            parent_dir = os.path.dirname(current_dir)
            if parent_dir not in sys.path:
                sys.path.append(parent_dir)
                
            from DriverDetection.driver_monitor import DriverMonitor
            
            # Test import model loading
            test_monitor = DriverMonitor()
            
            self.ai_available = True
            self.status_label.setText("Trạng thái: AI modules OK - Sẵn sàng")
            self.status_label.setStyleSheet("QLabel { background-color: #d4edda; color: black; padding: 10px; border-radius: 5px; }")
            print("✅ AI availability check passed")
        except Exception as e:
            self.ai_available = False
            self.status_label.setText(f"Trạng thái: AI lỗi - Chỉ GUI mode")
            self.status_label.setStyleSheet("QLabel { background-color: #fff3cd; color: black; padding: 10px; border-radius: 5px; }")
            print(f"❌ AI availability check failed: {e}")
            
            # Show info dialog only if it's a serious error
            if "No module named" in str(e):
                self.show_ai_error_dialog(str(e))
    
    def show_ai_error_dialog(self, error_msg):
        """Hiển thị dialog thông báo lỗi AI"""
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Icon.Warning)
        msg.setWindowTitle("AI Modules Error")
        msg.setText("Không thể load AI modules")
        msg.setInformativeText(
            "Ứng dụng sẽ chạy ở chế độ demo GUI.\n\n"
            "Để fix:\n"
            "1. Chạy fix_dependencies.py\n"
            "2. Cài đặt Python 3.8-3.11\n"
            "3. Cài Visual C++ Redistributable"
        )
        msg.setDetailedText(f"Error: {error_msg}")
        msg.exec()
        
    def init_ui(self):
        self.setWindowTitle("Hệ Thống Nhận Diện Ngủ Gật (Safe Mode)")
        self.setGeometry(100, 100, 1200, 800)
        
        # Main widget và layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        
        # Tạo panel bên trái (Settings)
        self.create_settings_panel()
        main_layout.addWidget(self.settings_frame, 1)
        
        # Tạo panel bên phải (Camera displays)
        self.create_camera_panel()
        main_layout.addWidget(self.camera_frame, 2)

        # Shortcuts sẽ được gắn sau khi camera_thread sẵn sàng
        
    def create_settings_panel(self):
        """Tạo panel settings bên trái"""
        self.settings_frame = QFrame()
        self.settings_frame.setFrameStyle(QFrame.Shape.StyledPanel)
        self.settings_frame.setMaximumWidth(350)
        
        layout = QVBoxLayout(self.settings_frame)
        
        # Title
        title = QLabel("CÀI ĐẶT HỆ THỐNG")
        title.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        # Nhóm cài đặt âm thanh
        audio_group = QGroupBox("Cài Đặt Âm Thanh")
        audio_layout = QVBoxLayout(audio_group)
        
        # Chọn file âm thanh
        audio_file_layout = QHBoxLayout()
        self.audio_file_label = QLabel("alarm.wav")
        self.audio_browse_btn = QPushButton("Chọn File")
        self.audio_browse_btn.clicked.connect(self.browse_audio_file)
        audio_file_layout.addWidget(QLabel("File âm thanh:"))
        audio_file_layout.addWidget(self.audio_file_label)
        audio_file_layout.addWidget(self.audio_browse_btn)
        audio_layout.addLayout(audio_file_layout)
        
        # Giới hạn thời gian âm thanh
        duration_layout = QHBoxLayout()
        duration_layout.addWidget(QLabel("Thời gian tối đa (s):"))
        self.audio_duration_spin = QSpinBox()
        self.audio_duration_spin.setRange(1, 60)
        self.audio_duration_spin.setValue(10)
        duration_layout.addWidget(self.audio_duration_spin)
        audio_layout.addLayout(duration_layout)
        
        layout.addWidget(audio_group)
        
        # Nhóm cài đặt nhận diện mắt
        eye_group = QGroupBox("Nhận Diện Mắt")
        eye_layout = QVBoxLayout(eye_group)
        
        # Ngưỡng thời gian nhắm mắt
        eye_threshold_layout = QHBoxLayout()
        eye_threshold_layout.addWidget(QLabel("Ngưỡng nhắm mắt (s):"))
        self.eye_threshold_spin = QSpinBox()
        self.eye_threshold_spin.setRange(1, 10)
        self.eye_threshold_spin.setValue(2)
        eye_threshold_layout.addWidget(self.eye_threshold_spin)
        eye_layout.addLayout(eye_threshold_layout)
        
        layout.addWidget(eye_group)
        
        # Nhóm cài đặt nhận diện ngáp
        yawn_group = QGroupBox("Nhận Diện Ngáp")
        yawn_layout = QVBoxLayout(yawn_group)
        
        # Toggle switch bật/tắt
        yawn_toggle_layout = QHBoxLayout()
        yawn_toggle_layout.addWidget(QLabel("Nhận diện ngáp:"))
        self.yawn_enable_switch = ToggleSwitch()
        self.yawn_enable_switch.stateChanged.connect(self.toggle_yawn_detection)
        yawn_toggle_layout.addWidget(self.yawn_enable_switch)
        yawn_toggle_layout.addStretch()
        yawn_layout.addLayout(yawn_toggle_layout)
        
        # Số lần ngáp tối đa
        yawn_count_layout = QHBoxLayout()
        yawn_count_layout.addWidget(QLabel("Số lần ngáp tối đa:"))
        self.yawn_count_spin = QSpinBox()
        self.yawn_count_spin.setRange(1, 20)
        self.yawn_count_spin.setValue(5)
        yawn_count_layout.addWidget(self.yawn_count_spin)
        yawn_layout.addLayout(yawn_count_layout)
        
        # Thời gian reset (phút)
        yawn_reset_layout = QHBoxLayout()
        yawn_reset_layout.addWidget(QLabel("Reset sau (phút):"))
        self.yawn_reset_spin = QSpinBox()
        self.yawn_reset_spin.setRange(1, 60)
        self.yawn_reset_spin.setValue(10)
        yawn_reset_layout.addWidget(self.yawn_reset_spin)
        yawn_layout.addLayout(yawn_reset_layout)
        
        layout.addWidget(yawn_group)
        
        # Nhóm điều khiển hệ thống
        control_group = QGroupBox("Điều Khiển")
        control_layout = QVBoxLayout(control_group)
        
        self.start_btn = QPushButton("Bắt Đầu")
        self.start_btn.clicked.connect(self.start_monitoring)
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
        self.stop_btn.clicked.connect(self.stop_monitoring)
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
        
        layout.addWidget(control_group)
        
        # Status
        self.status_label = QLabel("Trạng thái: Đang kiểm tra...")
        self.status_label.setStyleSheet("QLabel { background-color: #e8f5e8; color: black; padding: 10px; border-radius: 5px; }")
        layout.addWidget(self.status_label)
        
        layout.addStretch()
        
    def create_camera_panel(self):
        """Tạo panel camera bên phải"""
        self.camera_frame = QFrame()
        self.camera_frame.setFrameStyle(QFrame.Shape.StyledPanel)
        
        layout = QVBoxLayout(self.camera_frame)
        
        # Camera nhận diện mắt (trên)
        eye_group = QGroupBox("Nhận Diện Trạng Thái Mắt")
        eye_layout = QVBoxLayout(eye_group)
        
        self.eye_camera_label = QLabel()
        self.eye_camera_label.setMinimumSize(640, 160)  # Reduced height for eye regions
        self.eye_camera_label.setStyleSheet("QLabel { border: 2px solid #ddd; background-color: #f8f8f8; border-radius: 8px; }")
        self.eye_camera_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.eye_camera_label.clear()
        eye_layout.addWidget(self.eye_camera_label)
        
        layout.addWidget(eye_group)
        
        # Camera nhận diện ngáp (dưới)
        yawn_group = QGroupBox("Nhận Diện Ngáp")
        yawn_layout = QVBoxLayout(yawn_group)
        
        self.yawn_camera_label = QLabel()
        self.yawn_camera_label.setMinimumSize(640, 360)
        self.yawn_camera_label.setStyleSheet("QLabel { border: 2px solid #ddd; background-color: #f8f8f8; border-radius: 8px; }")
        self.yawn_camera_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # Clean placeholder - no text, just empty camera frame
        self.yawn_camera_label.clear()
        yawn_layout.addWidget(self.yawn_camera_label)
        
        layout.addWidget(yawn_group)
        
    def setup_camera(self):
        """Khởi tạo camera thread"""
        self.camera_thread = SafeCameraThread()
        self.camera_thread.eye_regions_ready.connect(self.update_eye_camera)  # Use eye regions instead of full frame
        self.camera_thread.error_occurred.connect(self.handle_camera_error)
        self.camera_thread.yawn_frame_ready.connect(self.update_yawn_camera)
        
        # ---- Shortcuts (phụ thuộc camera_thread) ----
        QShortcut(QKeySequence("S"), self, activated=self.start_monitoring)
        QShortcut(QKeySequence("D"), self, activated=self.stop_monitoring)
        QShortcut(QKeySequence("Y"), self, activated=self._shortcut_toggle_yawn)
        QShortcut(QKeySequence("T"), self, activated=lambda: self.camera_thread.toggle_detection_flag('tesselation'))
        QShortcut(QKeySequence("C"), self, activated=lambda: self.camera_thread.toggle_detection_flag('contours'))
        QShortcut(QKeySequence("I"), self, activated=lambda: self.camera_thread.toggle_detection_flag('irises'))
        QShortcut(QKeySequence("R"), self, activated=self.camera_thread.reset_landmarks_flags)
        QShortcut(QKeySequence("A"), self, activated=self.camera_thread.reset_timer_and_alarm)
        QShortcut(QKeySequence("Q"), self, activated=self.close)
        
    def load_placeholder_image(self):
        """Load ảnh placeholder cho khi tắt nhận diện ngáp"""
        # Tạo ảnh placeholder đơn giản
        placeholder = np.zeros((360, 640, 3), dtype=np.uint8)
        placeholder.fill(200)  # Màu xám nhạt
        
        
        self.placeholder_image = placeholder
        self.update_yawn_camera(placeholder)
        
    def browse_audio_file(self):
        """Chọn file âm thanh"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Chọn File Âm Thanh", "", 
            "Audio Files (*.wav *.mp3 *.ogg);;All Files (*)"
        )
        if file_path:
            self.audio_file_label.setText(os.path.basename(file_path))
            
    def toggle_yawn_detection(self, state=None):
        """Bật/tắt nhận diện ngáp"""
        try:
            # Nếu state được truyền vào, sử dụng nó
            if state is not None:
                is_checked = (state == 2) if isinstance(state, int) else bool(state)
            else:
                # Nếu không có state, check từ switch
                is_checked = self.yawn_enable_switch.isChecked()

            if is_checked:
                # Enable yawn detection - clear text to show camera
                self.yawn_camera_label.clear()
                if hasattr(self, 'camera_thread') and self.camera_thread:
                    self.camera_thread.yawn_enabled = True
            else:
                # Disable yawn detection - clean placeholder
                self.yawn_camera_label.clear()
                if hasattr(self, 'camera_thread') and self.camera_thread:
                    self.camera_thread.yawn_enabled = False

        except Exception as e:
            print(f"❌ Error in toggle_yawn_detection: {e}")
            import traceback
            traceback.print_exc()
        
    def handle_camera_error(self, error_msg):
        """Xử lý lỗi camera"""
        self.status_label.setText(f"Lỗi camera: {error_msg[:50]}...")
        self.status_label.setStyleSheet("QLabel { background-color: #f8d7da; color: black; padding: 10px; border-radius: 5px; }")
        
    def start_monitoring(self):
        """Bắt đầu giám sát"""
        self.camera_thread.start_camera()
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        
        if self.ai_available:
            self.status_label.setText("Trạng thái: Đang giám sát (AI ON)")
        else:
            self.status_label.setText("Trạng thái: Đang giám sát (AI OFF)")
        self.status_label.setStyleSheet("QLabel { background-color: #fff3cd; color: black; padding: 10px; border-radius: 5px; }")
        
    def stop_monitoring(self):
        """Dừng giám sát"""
        self.camera_thread.stop_camera()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_label.setText("Trạng thái: Đã dừng")
        self.status_label.setStyleSheet("QLabel { background-color: #f8d7da; color: black; padding: 10px; border-radius: 5px; }")
        
        # Reset camera displays
        self.eye_camera_label.clear()

    # ---- Shortcuts handlers ----
    def _shortcut_toggle_yawn(self):
        """Toggle yawn detection via keyboard shortcut"""
        try:
            current = self.yawn_enable_switch.isChecked()
            new_state = not current

            # Update switch
            self.yawn_enable_switch.setChecked(new_state)

            # Call toggle function without parameter to use switch state
            self.toggle_yawn_detection()

        except Exception as e:
            print(f"❌ Error in shortcut toggle: {e}")
        
    def update_eye_camera(self, frame):
        """Cập nhật hiển thị camera nhận diện mắt"""
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format.Format_RGB888).rgbSwapped()
        
        # Scale ảnh để fit với label
        pixmap = QPixmap.fromImage(q_image)
        scaled_pixmap = pixmap.scaled(self.eye_camera_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.eye_camera_label.setPixmap(scaled_pixmap)
        
        # Cập nhật status
        # Clear text when showing camera feed
        self.eye_camera_label.setText("")
        
    def update_yawn_camera(self, frame):
        """Cập nhật hiển thị camera nhận diện ngáp"""
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format.Format_RGB888).rgbSwapped()

        pixmap = QPixmap.fromImage(q_image)
        scaled_pixmap = pixmap.scaled(self.yawn_camera_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.yawn_camera_label.setPixmap(scaled_pixmap)
        # Clear any text when showing camera feed
        self.yawn_camera_label.setText("")
        
    def closeEvent(self, event):
        """Đóng app an toàn"""
        if hasattr(self, 'camera_thread'):
            self.camera_thread.stop_camera()
        event.accept()

def main():
    app = QApplication(sys.argv)
    
    print("🛡️  SAFE MODE - Hệ Thống Nhận Diện Ngủ Gật")
    print("=" * 50)
    print("✅ Lazy loading AI modules")
    print("✅ Fallback to camera-only nếu AI lỗi")
    print("✅ Error handling toàn diện")
    print("✅ Nút Fix Dependencies tích hợp")
    print("=" * 50)
    
    window = SafeDriverDetectionApp()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()

