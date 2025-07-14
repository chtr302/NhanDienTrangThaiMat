from keras._tf_keras.keras import models
import os
import numpy as np
from PIL import Image

eye_model = models.load_model(os.path.join('models','model.keras'))

img_path = os.path.join('data','path') # Import image to test
img = Image.open(img_path)
img = img.resize((80,80))
img_array = np.array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

result = eye_model.predict(img_array, verbose=0)

if result[0][0] >= 0.5:
    print('Mo')
    print(result[0][0])
else:
    print('Dong')
    print(result[0][0])