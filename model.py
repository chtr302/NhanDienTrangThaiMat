import keras._tf_keras.keras as tf
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os

DATA_DIR = 'data'
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
TEST_DIR = os.path.join(DATA_DIR,'test')

# TRAIN SET
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
training_set = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(80,80),
    batch_size=32,
    class_mode='binary',
    classes=['closed','open']
)

# TEST SET
validation_datagen = ImageDataGenerator(rescale=1./255)
validation_set = validation_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(80,80),
    batch_size=32,
    class_mode='binary',
    classes=['closed','open']
)

# CNN MODEL
model = tf.models.Sequential([
    # Layer 1
    tf.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[80,80,3]),
    tf.layers.Conv2D(filters=32, kernel_size=3, activation='relu'),
    tf.layers.Conv2D(filters=32, kernel_size=3, activation='relu'),
    tf.layers.MaxPool2D(pool_size=2, strides=2),
    # Layer 2
    tf.layers.Conv2D(filters=32, kernel_size=3, activation='relu'),
    tf.layers.Conv2D(filters=32, kernel_size=3, activation='relu'),
    tf.layers.MaxPool2D(pool_size=2),
    # Flattening
    tf.layers.Flatten(),
    # Fully
    tf.layers.Dense(units=256, activation='relu'),
    tf.layers.Dropout(0.3),
    tf.layers.Dense(units=128, activation='relu'),
    tf.layers.Dropout(0.3),
    tf.layers.Dense(units=64, activation='relu'),
    tf.layers.Dropout(0.3),
    # Output
    tf.layers.Dense(units=1, activation='sigmoid')
])
optimizer = tf.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer,loss='binary_crossentropy', metrics=['accuracy'])

# Callbacks
early_stopping = tf.callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=5,
    restore_best_weights=True
)
model_checkpoint = tf.callbacks.ModelCheckpoint(
    os.path.join('models','best_model_first_try.keras'),
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)
# Giảm learning rate nếu 
reduce_lr = tf.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=3,
    min_lr=0.00001
)
# Training với callbacks
history = model.fit(
    x=training_set,
    validation_data=validation_set,
    epochs=24,
    callbacks=[early_stopping, model_checkpoint, reduce_lr]
)
