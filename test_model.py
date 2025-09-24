from tensorflow.keras import models
import os
import numpy as np
from PIL import Image

# Get absolute path to models directory
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'models', 'model.keras')

print(f"Looking for model at: {model_path}")
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at: {model_path}")

eye_model = models.load_model(model_path)
print(f"Model loaded successfully from: {model_path}")

# Example: path to an image for test purposes
img_path = os.path.join('data', 'path')  # replace with a real image path
img = Image.open(img_path)
img = img.resize((80, 80))
img_array = np.array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

result = eye_model.predict(img_array, verbose=0)

if result[0][0] >= 0.5:
    print('Mo')
    print(result[0][0])
else:
    print('Dong')
    print(result[0][0])

