try:
    # Try TensorFlow's Keras first (recommended)
    from tensorflow.keras.models import load_model
except ImportError:
    # Fallback to standalone Keras (if available)
    from keras.models import load_model
import os

class Models:
    """
    Load model CNN to predict eye open or closed
    """

    def load_eye_model(self):
        # Get absolute path to models directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)  # Go up one level from DriverDetection
        
        # Try test model first (compatible version)
        test_model_path = os.path.join(project_root, 'models', 'model_test.keras')
        original_model_path = os.path.join(project_root, 'models', 'model.keras')
        
        model_path = test_model_path if os.path.exists(test_model_path) else original_model_path
        
        print(f"🔍 Looking for model at: {model_path}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")
            
        self.eye_model = load_model(model_path)
        print(f"✅ Model loaded successfully from: {model_path}")
        return self.eye_model
    