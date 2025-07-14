import mediapipe as mp
import cv2
import numpy as np

class FrameProcessor:
    """
    Process frame and draw landmarks on frame
    """

    def __init__(self):
        self.mediapipe_face_mesh = mp.solutions.face_mesh
        self.mediapipe_draw = mp.solutions.drawing_utils
        self.mediapipe_draw_styles = mp.solutions.drawing_styles

        self.face_mesh = self.mediapipe_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # CLAHE cho cải thiện ánh sáng
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    
    def preprocess_frame(self, frame, enhance_lighting=True, reduce_noise=True):
        """
        Tiền xử lý frame toàn diện
        """
        height, width = frame.shape[:2]
        if width > 800:
            scale = 800 / width
            new_width = int(width * scale)
            new_height = int(height * scale)
            frame = cv2.resize(frame, (new_width, new_height))

        if enhance_lighting:
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l = self.clahe.apply(l)
            lab = cv2.merge([l, a, b])
            frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        if reduce_noise:
            frame = cv2.bilateralFilter(frame, 5, 50, 50)
        frame = cv2.convertScaleAbs(frame, alpha=1.05, beta=5)
        
        return frame
    
    def process_frame(self, frame, use_preprocessing=True):
        """
        Process one frame with optional preprocessing
        """
        if use_preprocessing: # Preprocessing frame
            processed_frame = self.preprocess_frame(frame)
        else:
            processed_frame = frame
        
        rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB) # Convert BGR2RGB
        results = self.face_mesh.process(rgb_frame) # Process frame
        
        return results
    
    def draw_landmarks(self, frame, results, draw_tesselation=True, draw_contours=True, draw_irises=True):
        """
        Draw landmarks on frame
        """
        annotated_frame = frame.copy()

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                if draw_tesselation:
                    self.mediapipe_draw.draw_landmarks(
                        image=annotated_frame,
                        landmark_list=face_landmarks,
                        connections=self.mediapipe_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.mediapipe_draw_styles.get_default_face_mesh_tesselation_style()
                    )
                if draw_contours:
                    self.mediapipe_draw.draw_landmarks(
                        image=annotated_frame,
                        landmark_list=face_landmarks,
                        connections=self.mediapipe_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.mediapipe_draw_styles.get_default_face_mesh_contours_style()
                    )
                if draw_irises:
                    self.mediapipe_draw.draw_landmarks(
                        image=annotated_frame,
                        landmark_list=face_landmarks,
                        connections=self.mediapipe_face_mesh.FACEMESH_IRISES,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.mediapipe_draw_styles.get_default_face_mesh_irises_style()
                    )

        return annotated_frame