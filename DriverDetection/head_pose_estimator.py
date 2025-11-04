import cv2
import numpy as np
import mediapipe as mp

class HeadPoseEstimator:
    """
    Ước tính tư thế đầu (Pitch, Yaw, Roll) từ các điểm landmarks trên khuôn mặt.
    """
    def __init__(self):
        # Các điểm 3D trên khuôn mặt mẫu (từ mô hình khuôn mặt chuẩn)
        # Đây là các điểm tương ứng với các landmarks của MediaPipe Face Mesh
        # Chúng ta sẽ sử dụng các điểm này để giải bài toán PnP
        self.model_points = np.array([
            (0.0, 0.0, 0.0),             # Mũi (Nose tip) - landmark 1
            (-225.0, 170.0, -135.0),     # Mắt trái (Left eye inner corner) - landmark 33
            (225.0, 170.0, -135.0),      # Mắt phải (Right eye inner corner) - landmark 263
            (-150.0, -150.0, -125.0),    # Mép miệng trái (Left mouth corner) - landmark 61
            (150.0, -150.0, -125.0),     # Mép miệng phải (Right mouth corner) - landmark 291
            (-35.0, -350.0, -100.0)      # Cằm (Chin) - landmark 152
        ], dtype="double")

        # Các chỉ số landmarks của MediaPipe tương ứng với model_points
        # (Mũi, Mắt trái trong, Mắt phải trong, Mép miệng trái, Mép miệng phải, Cằm)
        self.mp_face_mesh_indices = [1, 33, 263, 61, 291, 152]

        # Ma trận camera và hệ số méo ảnh sẽ được ước tính sau khi có kích thước frame
        self.camera_matrix = None
        self.dist_coeffs = np.zeros((4, 1)) # Giả định không có méo ảnh

    def _get_camera_matrix(self, img_w, img_h):
        """Ước tính ma trận camera dựa trên kích thước ảnh."""
        if self.camera_matrix is None or \
           self.camera_matrix[0, 2] != img_w / 2 or \
           self.camera_matrix[1, 2] != img_h / 2:
            
            focal_length = img_w # Giả định tiêu cự bằng chiều rộng ảnh
            center = (img_w / 2, img_h / 2)
            self.camera_matrix = np.array([
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1]
            ], dtype="double")
        return self.camera_matrix

    def process_landmarks(self, face_landmarks, img_w, img_h):
        """
        Tính toán góc Pitch, Yaw, Roll từ các landmarks của MediaPipe.

        Args:
            face_landmarks: Đối tượng landmarks từ MediaPipe Face Mesh.
            img_w: Chiều rộng của frame ảnh.
            img_h: Chiều cao của frame ảnh.

        Returns:
            tuple: (pitch, yaw, roll) hoặc (None, None, None) nếu không tính được.
        """
        if not face_landmarks:
            return None, None, None

        # Lấy các điểm 2D từ landmarks của MediaPipe
        image_points = np.array([
            (face_landmarks.landmark[idx].x * img_w, face_landmarks.landmark[idx].y * img_h)
            for idx in self.mp_face_mesh_indices
        ], dtype="double")

        # Ước tính ma trận camera
        camera_matrix = self._get_camera_matrix(img_w, img_h)

        # Giải bài toán PnP (Perspective-n-Point)
        (success, rotation_vector, translation_vector) = cv2.solvePnP(
            self.model_points, image_points, camera_matrix, self.dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            return None, None, None

        # Chuyển đổi vector xoay sang ma trận xoay
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

        # Chuyển đổi ma trận xoay sang góc Euler (Pitch, Yaw, Roll)
        # Các góc này thường được tính theo radian, sau đó chuyển sang độ
        # Tham khảo: https://www.learnopencv.com/head-pose-estimation-using-opencv-and-dlib/
        # Hoặc: https://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToEuler/index.htm
        
        # Trích xuất các góc Euler từ ma trận xoay
        # Yaw (quay trái/phải): xoay quanh trục Y
        # Pitch (gật gù lên/xuống): xoay quanh trục X
        # Roll (nghiêng đầu): xoay quanh trục Z
        
        # Để đơn giản, chúng ta có thể lấy các giá trị từ ma trận xoay
        # Tuy nhiên, cần cẩn thận với gimbal lock và thứ tự xoay
        # Một cách phổ biến là sử dụng cv2.decomposeProjectionMatrix hoặc tự tính
        
        # Cách đơn giản hóa để lấy Pitch, Yaw, Roll từ rotation_matrix
        # (có thể không hoàn hảo cho mọi trường hợp nhưng đủ tốt cho mục đích này)
        
        # Yaw (quay trái/phải)
        yaw = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0]) * 180 / np.pi
        # Pitch (gật gù lên/xuống)
        pitch = np.arctan2(-rotation_matrix[2, 0], np.sqrt(rotation_matrix[2, 1]**2 + rotation_matrix[2, 2]**2)) * 180 / np.pi
        # Roll (nghiêng đầu)
        roll = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2]) * 180 / np.pi

        return pitch, yaw, roll
