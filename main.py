import cv2
import numpy as np
from playsound import playsound
import mediapipe as mp
from tensorflow import keras
import os
import threading
import time

eye_model = keras.models.load_model(os.path.join('models','best_model_first_try.keras'))

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(
    model_selection=1,
    min_detection_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

LEFT_EYE_INDICES = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

face_present = True
last_face_time = time.time()
alarm_playing = False

def play_alarm():
    global alarm_playing
    if not alarm_playing:
        alarm_playing = True
        try:
            playsound(os.path.join('alarm.wav'))
        except Exception as e:
            print(f"Lỗi phát âm thanh: {e}")
        finally:
            alarm_playing = False

def check_if_face_present(frame):
    global face_present, last_face_time
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_frame)
    
    if results.detections:
        face_present = True
        last_face_time = time.time()
        return True

    if time.time() - last_face_time > 3:  # 3s để kiểm tra có mặt trong khung hình không
        face_present = False
    return face_present

def eye_cropper(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    
    if not results.multi_face_landmarks:
        return None, None

    face_landmarks = results.multi_face_landmarks[0]

    # Lấy landmarks cho cả hai mắt
    left_eye_img = extract_eye(frame, face_landmarks, LEFT_EYE_INDICES)
    right_eye_img = extract_eye(frame, face_landmarks, RIGHT_EYE_INDICES)
    
    return left_eye_img, right_eye_img

def extract_eye(frame, face_landmarks, eye_indices):
    eye_landmarks = []
    
    for idx in eye_indices:
        landmark = face_landmarks.landmark[idx]
        x = int(landmark.x * frame.shape[1])
        y = int(landmark.y * frame.shape[0])
        eye_landmarks.append((x, y))
    
    if len(eye_landmarks) != len(eye_indices):
        return None

    x_max = max([coordinate[0] for coordinate in eye_landmarks])
    x_min = min([coordinate[0] for coordinate in eye_landmarks])
    y_max = max([coordinate[1] for coordinate in eye_landmarks])
    y_min = min([coordinate[1] for coordinate in eye_landmarks])

    x_range = x_max - x_min
    y_range = y_max - y_min

    if x_range > y_range:
        right = round(.5*x_range) + x_max
        left = x_min - round(.5*x_range)
        bottom = round((((right-left) - y_range))/2) + y_max
        top = y_min - round((((right-left) - y_range))/2)
    else:
        bottom = round(.5*y_range) + y_max
        top = y_min - round(.5*y_range)
        right = round((((bottom-top) - x_range))/2) + x_max
        left = x_min - round((((bottom-top) - x_range))/2)

    top = max(0, top)
    bottom = min(frame.shape[0], bottom)
    left = max(0, left)
    right = min(frame.shape[1], right)

    cropped = frame[top:bottom, left:right]
    
    if cropped.size == 0:
        return None
    
    try:
        cropped = cv2.resize(cropped, (80, 80))
        image_for_prediction = cropped.reshape(-1, 80, 80, 3)
        return image_for_prediction
    except:
        return None

cap = cv2.VideoCapture(1) # Chuyển về 0 để sử dụng camera của máy
w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

if not cap.isOpened():
    raise IOError('Khong mo duoc camera')

counter = 0  # Đếm frame mắt đóng liên tục
blink_start = None  # Lúc bắt đầu nháy
consecutive_closed = 0  # Số frame mắt đóng liên tiếp
frame_count = 0  # tổng frame đã xử lý
dozing_count = 0  # Số lần ngủ gật được phát hiện

SLEEP_THRESHOLD_TIME = 1.2  # Mắt không được nhắm quá 1.2s

while True:
    frame_count += 1
    start_time = time.time()  # Bắt đầu đo thời gian xử lý frame

    ret, frame = cap.read()
    
    if not ret:
        print("Khong doc duoc tu camera")
        break

    face_detected = check_if_face_present(frame)

    if not face_detected:
        cv2.rectangle(frame, (0,0), (int(w), 50), (0,0,0), -1)
        cv2.putText(frame, "Khong co khuon mat trong khung hinh, nhin vao camera di nhaaaaa!", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow('Nhận diện trạng thái mắt', frame)

        counter = 0
        consecutive_closed = 0
        blink_start = None
        
        k = cv2.waitKey(1)
        if k == 27:
            break
        continue

    left_eye_image, right_eye_image = eye_cropper(frame)
    
    if left_eye_image is None and right_eye_image is None:
        cv2.putText(frame, "Khong tim thay mat", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.imshow('Drowsiness Detection', frame)
        k = cv2.waitKey(1)
        if k == 27:
            break
        continue

    # Process left eye if available
    left_prediction = None
    if left_eye_image is not None:
        left_eye_image = left_eye_image / 255.0
        left_prediction = eye_model.predict(left_eye_image, verbose=0)

    right_prediction = None
    if right_eye_image is not None:
        right_eye_image = right_eye_image / 255.0
        right_prediction = eye_model.predict(right_eye_image, verbose=0)

    if left_prediction is not None and right_prediction is not None:
        prediction = (left_prediction + right_prediction) / 2
    elif left_prediction is not None:
        prediction = left_prediction
    else:
        prediction = right_prediction

    # Hiển thị trạng thái của từng mắt
    if left_prediction is not None:
        eye_status = "Mo" if left_prediction >= 0.5 else "Dong"
        cv2.putText(frame, f"Mat Trai: {eye_status}", (10, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    if right_prediction is not None:
        eye_status = "Mo" if right_prediction >= 0.5 else "Dong"
        cv2.putText(frame, f"Mat Phai: {eye_status}", (10, 120), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    if prediction >= 0.5:  # Mắt mở
        if blink_start:
            blink_duration = time.time() - blink_start
            if blink_duration < 1.0:  # Nháy mắt bình thường < 1s
                counter = 0 # Nháy bình thường, counter về 0
            blink_start = None
        
        consecutive_closed = 0
        status = 'Dang Mo'
        cv2.putText(frame, status, (round(w/2)-80,70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 2, cv2.LINE_4)

    else:  # Mắt đóng
        consecutive_closed += 1
        
        # Đếm thời gian mắt đóng
        if consecutive_closed == 1:
            blink_start = time.time()
        
        # Tính thời gian nhắm
        current_closed_time = time.time() - blink_start if blink_start else 0
        
        if current_closed_time > SLEEP_THRESHOLD_TIME:
            counter = counter + 1
            status = 'Mat Dong'

            cv2.putText(frame, status, (round(w/2)-104,70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 2, cv2.LINE_4)

            # Hiển thị thời gian mắt đóng
            cv2.putText(frame, f"Mat dong duoc: {current_closed_time:.1f}s", 
                        (10, int(h)-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Cảnh báo nếu quá thời gian
            if counter > 2:
                cv2.rectangle(frame, (round(w/2) - 160, round(h) - 200), (round(w/2) + 160, round(h) - 120), (0,0,255), -1)
                cv2.putText(frame, 'DRIVER SLEEPING', (round(w/2)-136,round(h) - 146), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv2.LINE_4)
                
                # Tăng số lần ngủ gật
                if current_closed_time > SLEEP_THRESHOLD_TIME:  # Nếu nhắm mắt quá lâu thì tính là một lần ngủ gật
                    dozing_count += 1
                    blink_start = time.time()
                
                cv2.putText(frame, f"So lan ngu gat: {dozing_count}", 
                        (10, int(h)-80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
                # Hiển thị cảnh báo nếu ngủ gật quá 3 lần
                if dozing_count > 3:
                    cv2.putText(frame, 'BAN NEN DI NGU!', (round(w/2)-180, round(h/2) + 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 2, cv2.LINE_4)
                
                # Tạo thread riêng, nếu không có nó sẽ dẫn tới giao diện lúc phát hiện nó bị đơ
                if not alarm_playing:
                    alarm_thread = threading.Thread(target=play_alarm)
                    alarm_thread.daemon = True  # Huỷ thread
                    alarm_thread.start()
                
                counter = 1  # Giảm counter xuống
        else:
            status = 'Vua nhay mat'
            cv2.putText(frame, status, (round(w/2)-104,70), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,0), 2, cv2.LINE_4)

    # Luôn hiển thị số lần ngủ gật
    cv2.putText(frame, f"So lan ngu gat: {dozing_count}", 
                (10, int(h)-80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    processing_time = time.time() - start_time
    cv2.putText(frame, f"Process time: {processing_time*1000:.0f}ms", (10, int(h)-50), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Hiển thị frame đã xử lý
    cv2.imshow('Drowsiness Detection', frame)
    
    # Xử lý phím nhấn
    k = cv2.waitKey(1)
    if k == 27:  # ESC key
        break

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()
face_mesh.close()
face_detection.close()