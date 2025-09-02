import os
import cv2
import numpy as np
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
except ImportError as e:
    raise ImportError("TensorFlow is required for yawn model loading. Please install tensorflow==2.13.0") from e


class YawnProcessor:
    def __init__(self, model_path='models/yawn_model_trained.keras'):
        # Resolve absolute model path relative to project root
        if not os.path.isabs(model_path):
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(current_dir)
            model_path = os.path.join(project_root, model_path)

        self.model = None
        self.input_shape = (80, 80)
        self.available = False

        if not os.path.exists(model_path):
            print(f"⚠️ Yawn model not found at: {model_path}")
            print("💡 Yawn detection will be disabled")
            return

        print(f"🔍 Loading yawn model from: {model_path}")

        # Thử load model với nhiều cách khác nhau
        try:
            # Cách 1: Direct load với compile=False
            self.model = load_model(model_path, compile=False)
            self.available = True
            print(f"✅ Yawn model loaded successfully: {os.path.basename(model_path)}")
        except Exception as e:
            print(f"⚠️ Primary model failed: {e}")
            print("🔄 Trying fallback models...")

            # Fallback 1: Try trained model
            trained_path = model_path.replace('yawn_model_trained.keras', 'yawn_model_trained.keras')
            if trained_path != model_path and os.path.exists(trained_path):
                try:
                    self.model = load_model(trained_path, compile=False)
                    self.available = True
                    print(f"✅ Loaded trained model from: {os.path.basename(trained_path)}")
                except Exception as e2:
                    print(f"❌ Trained model also failed: {e2}")

            # Fallback 2: Try TF213 compatible model
            if not self.available:
                compat_path = model_path.replace('.keras', '_tf213.keras')
                if os.path.exists(compat_path):
                    try:
                        self.model = load_model(compat_path, compile=False)
                        self.available = True
                        print(f"✅ Loaded compatible model from: {os.path.basename(compat_path)}")
                    except Exception as e3:
                        print(f"❌ Compatible model also failed: {e3}")

            # Fallback 3: Try original model
            if not self.available:
                orig_path = model_path.replace('yawn_model_trained.keras', 'yawn_model.keras')
                if orig_path != model_path and os.path.exists(orig_path):
                    try:
                        self.model = load_model(orig_path, compile=False)
                        self.available = True
                        print(f"✅ Loaded original model from: {os.path.basename(orig_path)}")
                    except Exception as e4:
                        print(f"❌ All models failed, yawn detection disabled: {e4}")

        if not self.available:
            print("⚠️ YawnProcessor initialized but model not available")
            print("💡 App will run without yawn detection")

    def predict(self, face_img):
        """
        Dự đoán trạng thái ngáp từ ảnh vùng miệng
        face_img: BGR image (numpy array), cropped to mouth region
        """
        if not self.available or self.model is None:
            return False, 0.0  # Default: no yawn detected

        try:
            img = cv2.resize(face_img, self.input_shape)
            img = img.astype('float32') / 255.0
            img = np.expand_dims(img, axis=0)
            pred = self.model.predict(img, verbose=0)[0][0]
            return pred > 0.5, float(pred)  # (is_yawn, confidence)
        except Exception as e:
            print(f"⚠️ Yawn prediction error: {e}")
            return False, 0.0

    def crop_mouth_region(self, frame, face_results):
        """
        Crop vùng miệng dựa trên landmark MediaPipe (FACEMESH_LIPS)
        """
        if not face_results.multi_face_landmarks:
            return None
        face_landmarks = face_results.multi_face_landmarks[0]
        h, w = frame.shape[:2]

        # chỉ số landmark miệng
        mouth_indices = [
            61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
            291, 308, 324, 318, 402, 317, 14, 87, 178, 88,
            95, 185, 40, 39, 37, 0, 267, 269, 270, 409, 
            415, 310, 311, 312, 13, 82, 81, 42, 183, 78
        ]

        points = []
        for idx in mouth_indices:
            lm = face_landmarks.landmark[idx]
            x, y = int(lm.x * w), int(lm.y * h)
            points.append((x, y))

        points = np.array(points)
        x, y, w_box, h_box = cv2.boundingRect(points)

        # thêm margin
        margin = 10
        x = max(x - margin, 0)
        y = max(y - margin, 0)
        w_box = min(w_box + 2*margin, frame.shape[1] - x)
        h_box = min(h_box + 2*margin, frame.shape[0] - y)

        mouth_img = frame[y:y+h_box, x:x+w_box]
        return mouth_img
