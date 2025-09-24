import numpy as np
import cv2
from .model import Models

class EyeProcessor:
    """
    Process eye anf send to model
    """
    def __init__(self, model = Models()):
        self.model = model.load_eye_model()

        self.LEFT_EYE_INDICES = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398] # Landmarks for eye left
        self.RIGHT_EYE_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246] # Landmarks for eye right

        self.skip_frame = 4
        self.frame_count = 0
        self.cached_predictions = None
        
        # Thêm biến để theo dõi trạng thái mắt
        self.eyes_closed_detected = False  # Chỉ bắt đầu xử lý khi phát hiện mắt nhắm

    def __extract_eye_landmarks(self,frame_shape, face_landmarks, eye_type='left'):
        """
        Extract landmarks of eye
        Args: face_landmarks, eye_type - eye 'left' or 'right'
        Returns: list of (x, y) coordinates
        """
        height, width = frame_shape[:2]
        indices = self.LEFT_EYE_INDICES if eye_type == 'left' else self.RIGHT_EYE_INDICES
        eye_landmarks = []
        for i in indices:
            if i < len(face_landmarks.landmark):
                x = int(face_landmarks.landmark[i].x * width)
                y = int(face_landmarks.landmark[i].y * height)
                eye_landmarks.append((x, y))
        return eye_landmarks
    
    def __crop_eye_region(self, frame, eye_landmarks):
        """
        Crop eye region to frame
        Args: frame, eye_landmark
        Returns: cropped eye image
        """

        xs, ys = zip(*eye_landmarks) # Zip landmarks
        x_min, x_max = int(min(xs)), int(max(xs)) # Choose min and max of X landmarks
        y_min, y_max = int(min(ys)), int(max(ys)) # Choose min and max of Y landmarks

        crop = frame[y_min:y_max, x_min:x_max]
        if crop.size == 0:
            return None
        return crop
    
    def __process_eye(self, eye_image):
        """
        Process image for model
        Args: eye_image
        Returns: image process
        """
        eye_resized = cv2.resize(eye_image, (80,80)) # Resize image to 80x80
        eye_array = eye_resized.astype(np.float32) / 255.0 # Normalize 0-1
        eye_array = np.expand_dims(eye_array, 0) # Add batch dimension
        return eye_array
    
    def __predict_eye_state(self, processed_eye):
        prediction = self.model.predict(processed_eye, verbose=0)
        confidence = prediction[0][0]
        if confidence > 0.5:
            state = 'open'
        else:
            state = 'closed'
            confidence = 1 - confidence
        return state, float(confidence)

    def detect_eyes(self, frame, face_landmarks):
        """
        Phát hiện trạng thái mắt từ frame
        - Chỉ xử lý khi phát hiện mắt nhắm hoặc đã phát hiện mắt nhắm trước đó
        - Sử dụng skip frame để tối ưu hiệu suất
        """
        # Nếu chưa phát hiện mắt nhắm, luôn xử lý để kiểm tra
        if not self.eyes_closed_detected:
            # Xử lý đầy đủ để kiểm tra mắt nhắm
            left_eye_landmarks = self.__extract_eye_landmarks(frame.shape, face_landmarks, eye_type='left')
            left_img = self.__crop_eye_region(frame, left_eye_landmarks)
            left_img = self.__process_eye(left_img)
            left_eye_state, left_eye_confidence = self.__predict_eye_state(left_img)

            right_eye_landmarks = self.__extract_eye_landmarks(frame.shape, face_landmarks, eye_type='right')
            right_img = self.__crop_eye_region(frame, right_eye_landmarks)
            right_img = self.__process_eye(right_img)
            right_eye_state, right_eye_confidence = self.__predict_eye_state(right_img)
            
            # Kiểm tra nếu cả hai mắt đều nhắm
            if left_eye_state == 'closed' and right_eye_state == 'closed':
                self.eyes_closed_detected = True
                print("Đã phát hiện mắt nhắm - bắt đầu theo dõi đầy đủ")
            
            results = (left_eye_state, right_eye_state, left_eye_confidence, right_eye_confidence)
            self.cached_predictions = results
            return results
        
        # Nếu đã phát hiện mắt nhắm, áp dụng cơ chế skip frame
        should_predict = (self.frame_count % (self.skip_frame + 1)) == 0
        self.frame_count += 1
        
        if not should_predict and self.cached_predictions is not None:
            return self.cached_predictions
        
        # Xử lý đầy đủ
        left_eye_landmarks = self.__extract_eye_landmarks(frame.shape, face_landmarks, eye_type='left')
        left_img = self.__crop_eye_region(frame, left_eye_landmarks)
        left_img = self.__process_eye(left_img)
        left_eye_state, left_eye_confidence = self.__predict_eye_state(left_img)

        right_eye_landmarks = self.__extract_eye_landmarks(frame.shape, face_landmarks, eye_type='right')
        right_img = self.__crop_eye_region(frame, right_eye_landmarks)
        right_img = self.__process_eye(right_img)
        right_eye_state, right_eye_confidence = self.__predict_eye_state(right_img)
        
        results = (left_eye_state, right_eye_state, left_eye_confidence, right_eye_confidence)
        self.cached_predictions = results
        return results
