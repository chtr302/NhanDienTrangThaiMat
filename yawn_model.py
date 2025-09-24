import keras._tf_keras.keras as tf
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
import os

BATCH_SIZE = 32
YAWN_TRAIN_DIR = os.path.join('data', 'train')
YAWN_TEST_DIR = os.path.join('data', 'test')
YAWN_CLASSES = ['No_yawn', 'Yawn']

# TRAIN SET 
train_datagen_yawn = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0,
    zoom_range=0.2,
    horizontal_flip=True
)
training_set_yawn = train_datagen_yawn.flow_from_directory(
    YAWN_TRAIN_DIR,
    target_size=(80,80),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    classes=YAWN_CLASSES
)

# TEST SET 
validation_datagen_yawn = ImageDataGenerator(rescale=1./255)
validation_set_yawn = validation_datagen_yawn.flow_from_directory(
    YAWN_TEST_DIR,
    target_size=(80,80),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    classes=YAWN_CLASSES
)

# CNN MODEL 
model_yawn = tf.models.Sequential([
    tf.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[80,80,3]),
    tf.layers.Conv2D(filters=32, kernel_size=3, activation='relu'),
    tf.layers.Conv2D(filters=32, kernel_size=3, activation='relu'),
    tf.layers.MaxPool2D(pool_size=2, strides=2),

    tf.layers.Conv2D(filters=32, kernel_size=3, activation='relu'),
    tf.layers.Conv2D(filters=32, kernel_size=3, activation='relu'),
    tf.layers.MaxPool2D(pool_size=2),

    tf.layers.Flatten(),

    tf.layers.Dense(units=256, activation='relu'),
    tf.layers.Dropout(0.3),
    tf.layers.Dense(units=128, activation='relu'),
    tf.layers.Dropout(0.3),
    tf.layers.Dense(units=64, activation='relu'),
    tf.layers.Dropout(0.3),

    tf.layers.Dense(units=1, activation='sigmoid')
])
optimizer_yawn = tf.optimizers.Adam(learning_rate=0.001)
model_yawn.compile(optimizer=optimizer_yawn, loss='binary_crossentropy', metrics=['accuracy'])

# Callbacks 
early_stopping_yawn = tf.callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=5,
    restore_best_weights=True
)
model_checkpoint_yawn = tf.callbacks.ModelCheckpoint(
    os.path.join('models','yawn_model.keras'),
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)
reduce_lr_yawn = tf.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=3,
    min_lr=0.00001
)

# Training Yawn model
history_yawn = model_yawn.fit(
    x=training_set_yawn,
    validation_data=validation_set_yawn,
    epochs=24,
    callbacks=[early_stopping_yawn, model_checkpoint_yawn, reduce_lr_yawn]
)