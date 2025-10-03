from tensorflow.keras import models as keras_models
import numpy as np
import cv2

class YawnProcessor:
    def __init__(self, model_path='models/yawn_model.keras'):
        try:
            # Load using tf.keras to be compatible with TF 2.x
            self.model = keras_models.load_model(model_path)
            self.available = True
        except Exception as e:
            self.model = None
            self.available = False
        self.input_shape = (80,80)
        self.enabled = True  # Thêm trạng thái bật/tắt

    def enable(self):
        """Bật nhận diện ngáp"""
        self.enabled = True

    def disable(self):
        """Tắt nhận diện ngáp"""
        self.enabled = False

    def predict(self, face_img):
        if not self.available or self.model is None or not self.enabled:
            return False, 0.0
        # face_img: BGR image (numpy array), cropped to mouth/yawn region
        img = cv2.resize(face_img, self.input_shape)
        img = img.astype('float32') / 255.0
        img = np.expand_dims(img, axis=0)
        pred = self.model.predict(img,verbose = 0)[0][0]
        return pred > 0.5, float(pred)  # (is_yawn, confidence)

        
    def crop_mouth_region(self, frame, face_results):
        """
        Crop vùng miệng dựa trên landmark mediapipe (dùng FACEMESH_LIPS)
        """
        if not face_results.multi_face_landmarks:
            return None
        face_landmarks = face_results.multi_face_landmarks[0]
        h, w = frame.shape[:2]
        mouth_indices = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 185, 40, 39, 37, 0, 267, 269, 270, 409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78]
        points = []
        for idx in mouth_indices:
            lm = face_landmarks.landmark[idx]
            x, y = int(lm.x * w), int(lm.y * h)
            points.append((x, y))
        points = np.array(points)
        x, y, w_box, h_box = cv2.boundingRect(points)
        margin = 10
        x = max(x - margin, 0)
        y = max(y - margin, 0)
        w_box = min(w_box + 2*margin, frame.shape[1] - x)
        h_box = min(h_box + 2*margin, frame.shape[0] - y)
        mouth_img = frame[y:y+h_box, x:x+w_box]
        return mouth_img