import cv2
import numpy as np
from playsound import playsound
import mediapipe as mp
from tensorflow import keras
import os
import threading
import time
from collections import deque

eye_model = keras.models.load_model(os.path.join('models','best_model_first_try.keras'))

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    # các biến giúp nhận diện tốt hơn
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

LEFT_EYE_INDICES = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398] # Landmark mắt trái
RIGHT_EYE_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246] # Landmark mắt phải

face_present = True
last_face_time = time.time()
alarm_playing = False

is_dozing = False

def play_alarm():
    global alarm_playing
    if not alarm_playing:
        alarm_playing = True
        try:
            playsound(os.path.join('alarm.wav'))
        except Exception as e:
            print(f"hinh nhu la khong tim thay am thanh: {e}")
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
    results = face_mesh.process(rgb_frame) # phát hiện landmark từ frame đã đổi qua màu rgb
    
    if not results.multi_face_landmarks:
        return None, None

    face_landmarks = results.multi_face_landmarks[0] # lấy landmark cúa mặt đầu tiên

    # tính vị trí mắt xong trả về
    left_eye_img = extract_eye(frame, face_landmarks, LEFT_EYE_INDICES)
    right_eye_img = extract_eye(frame, face_landmarks, RIGHT_EYE_INDICES)
    
    return left_eye_img, right_eye_img

last_l_img, last_r_img = None, None

OPEN_TH = 0.98 # cai nay la gia tri cho chuong trinh biet mat mo
CLOSE_TH = 0.95 # cai nay cung giong the nhung la mat dong

def extract_eye(frame, face_landmarks, eye_indices):
    eye_lms = [(int(face_landmarks.landmark[i].x * frame.shape[1]),
                int(face_landmarks.landmark[i].y * frame.shape[0]))
               for i in eye_indices] # duyệt qua eye_indices 
    
    if not eye_lms:
        return None

    xs, ys = zip(*eye_lms)
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    w, h = x_max - x_min, y_max - y_min
    pad_w, pad_h = int(w*0.3), int(h*0.3)

    x1 = max(x_min - pad_w, 0);   x2 = min(x_max + pad_w, frame.shape[1])
    y1 = max(y_min - pad_h, 0);   y2 = min(y_max + pad_h, frame.shape[0])

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0: 
        return None
    try:
        crop = cv2.resize(crop, (80,80))
        if eye_indices is LEFT_EYE_INDICES:
            crop = cv2.flip(crop, 1)
        return crop.reshape(-1,80,80,3)
    except cv2.error:
        return None

def predict_eye(img, buf):
    if img is None:
        return None
    p = float(eye_model.predict(img/255.0, verbose=0)[0][0])
    buf.append(p)
    return sum(buf)/len(buf)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# lưu giá trị nhận diện của mắt
left_buf = deque(maxlen=3)
right_buf = deque(maxlen=3)

mesh_skip = 0
face_landmarks_cache = None

blink_start = None  # Lúc bắt đầu nháy
consecutive_closed = 0  # Số frame mắt đóng liên tiếp
frame_count = 0  # tổng frame đã xử lý
dozing_count = 0  # Số lần ngủ gật được phát hiện

SLEEP_THRESHOLD_TIME = 0.8  # Mắt không được nhắm quá 0.8s

while True:
    frame_count += 1
    start_time = time.time()  # Bắt đầu đo thời gian xử lý frame

    ret, frame = cap.read()
    if not ret:
        break
    h, w = frame.shape[:2]

    if mesh_skip == 0:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = face_mesh.process(rgb)
        face_landmarks_cache = res.multi_face_landmarks[0] if res.multi_face_landmarks else None
    mesh_skip = (mesh_skip + 1) % 3

    if not face_landmarks_cache:
        continue

    # Vẽ khung quanh mắt
    pad = 5
    for eye_inds in (LEFT_EYE_INDICES, RIGHT_EYE_INDICES):
        pts = [(int(face_landmarks_cache.landmark[i].x * w),
                int(face_landmarks_cache.landmark[i].y * h))
               for i in eye_inds]
        xs, ys = zip(*pts)
        x1, x2 = min(xs) - pad, max(xs) + pad
        y1, y2 = min(ys) - pad, max(ys) + pad
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 1)

    l_img = extract_eye(frame, face_landmarks_cache, LEFT_EYE_INDICES)
    r_img = extract_eye(frame, face_landmarks_cache, RIGHT_EYE_INDICES)

    # fallback về lần crop thành công trước đó
    if l_img is None:
        l_img = last_l_img
    else:
        last_l_img = l_img
    if r_img is None:
        r_img = last_r_img
    else:
        last_r_img = r_img

    lp = predict_eye(l_img, left_buf)
    rp = predict_eye(r_img, right_buf)

    # in debug
    cv2.putText(frame, f"trai:{lp or 0:.2f} phai:{rp or 0:.2f}",
                (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0),2)

    both_closed = (lp is not None and lp < CLOSE_TH) and (rp is not None and rp < CLOSE_TH)
    if both_closed:
        consecutive_closed += 1
        if consecutive_closed == 1:
            blink_start = time.time()
        closed_time = time.time() - blink_start

        # dozing_count tăng sau mỗi lần
        if closed_time > SLEEP_THRESHOLD_TIME and not is_dozing:
            dozing_count += 1
            is_dozing = True
            if not alarm_playing:
                t = threading.Thread(target=play_alarm)
                t.daemon = True
                t.start()

        # thông báo cứng nếu alarm bật
        if alarm_playing:
            cv2.putText(frame,
                        'TAI XE DANG NHAM MAT',
                        (round(w/2)-200, round(h/2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)
    # mắt mở thì bắt đầu lại
    else:
        is_dozing = False
        consecutive_closed = 0
        blink_start = None

    # ngu gat
    cv2.putText(frame,
                f"So lan ngu gat: {dozing_count}",
                (10, h-80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    processing_time = time.time() - start_time
    cv2.putText(frame, f"thoi gian xu ly frame: {processing_time*1000:.0f}ms", (10, int(h)-50), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.imshow('nhan dien trang thai mat', frame)
    
    # nhận ESC thoát
    k = cv2.waitKey(1)
    if k == 27:
        break

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()
face_mesh.close()
face_detection.close()