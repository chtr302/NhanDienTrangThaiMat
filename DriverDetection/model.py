import os
from tensorflow.keras import models as keras_models
from tensorflow.keras import layers
import tensorflow as tf

class SpatialAttention(layers.Layer):
    def __init__(self, kernel_size=7, **kwargs):
        super(SpatialAttention, self).__init__(**kwargs)
        self.kernel_size = kernel_size
        self.conv2d = layers.Conv2D(
            filters=1,
            kernel_size=self.kernel_size,
            strides=1,
            padding='same',
            activation='sigmoid',
            kernel_initializer='he_normal',
            use_bias=False,
            name='spatial_attention_conv'
        )

    def call(self, inputs):
        avg_pool = tf.reduce_mean(inputs, axis=3, keepdims=True)
        max_pool = tf.reduce_max(inputs, axis=3, keepdims=True)
        concat = layers.Concatenate(axis=3)([avg_pool, max_pool])
        attention = self.conv2d(concat)
        return layers.Multiply()([inputs, attention])

    def get_config(self):
        config = super(SpatialAttention, self).get_config()
        config.update({'kernel_size': self.kernel_size})
        return config

class Models:
    """Loader for pretrained CNN models used at runtime."""

    def __init__(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.project_root = os.path.dirname(current_dir)
        self.models_dir = os.path.join(self.project_root, 'models')

    def _choose_eye_model_path(self) -> str:
        candidates = [
            os.path.join(self.models_dir, 'model_1st.keras'),
        ]
        for p in candidates:
            if os.path.exists(p):
                print(f"Loading eye model from: {p}")
                return p
        raise FileNotFoundError(f"No eye model found in {self.models_dir}")

    def load_eye_model(self):
        path = self._choose_eye_model_path()
        model = keras_models.load_model(
            path,
            custom_objects={'SpatialAttention': SpatialAttention}
        )
        return model