from keras._tf_keras.keras import models
import os
import numpy as np
from PIL import Image

eye_model = models.load_model(os.path.join('models','model.keras'))

img_path = os.path.join('data','') # Import image to test
img = Image.open(img_path)
img = img.resize((80,80))
img_array = np.array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

result = eye_model.predict(img_array)

if result[0] >= 0.8:
    print('Mo')
    print(result)
else:
    print('Dong')
    print(result)