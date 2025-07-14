from keras._tf_keras.keras.models import load_model
import os

class Models:
    """
    Load model CNN to predict eye open or closed
    """

    def load_eye_model(self):
        self.eye_model = load_model(os.path.join('models','model.keras'))
        return self.eye_model
    