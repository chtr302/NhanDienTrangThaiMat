import os
from tensorflow.keras import models as keras_models


class Models:
    """Loader for pretrained CNN models used at runtime."""

    def __init__(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.project_root = os.path.dirname(current_dir)
        self.models_dir = os.path.join(self.project_root, 'models')

    def _choose_eye_model_path(self) -> str:
        candidates = [
            os.path.join(self.models_dir, 'model.keras'),
            os.path.join(self.models_dir, 'best_model_first_try.keras'),
        ]
        for p in candidates:
            if os.path.exists(p):
                return p
        raise FileNotFoundError(f"No eye model found in {self.models_dir}")

    def load_eye_model(self):
        path = self._choose_eye_model_path()
        print(f"Loading eye model from: {path}")
        model = keras_models.load_model(path)
        print("Eye model loaded OK")
        return model

