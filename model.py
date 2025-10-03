# try:
#     import tensorflow as tf
#     from tensorflow.keras.preprocessing.image import ImageDataGenerator
# except ImportError:
#     import keras as tf
#     from keras.preprocessing.image import ImageDataGenerator
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
import keras as tf
import os

TRAIN_DIR = os.path.join('data', 'train')
TEST_DIR = os.path.join('data','test')
BATCH_SIZE = 32

# TRAIN SET
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0,
    zoom_range=0.2,
    horizontal_flip=True
)
training_set = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(80,80),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    classes=['Closed','Open']
)

# TEST SET
validation_datagen = ImageDataGenerator(rescale=1./255)
validation_set = validation_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(80,80),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    classes=['Closed','Open']
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
    tf.layers.Dropout(0.5),
    tf.layers.Dense(units=128, activation='relu'),
    tf.layers.Dropout(0.5),
    tf.layers.Dense(units=64, activation='relu'),
    tf.layers.Dropout(0.5),
    # Output
    tf.layers.Dense(units=1, activation='sigmoid')
])
optimizer = tf.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer,loss='binary_crossentropy', metrics=['accuracy', tf.metrics.Precision(), tf.metrics.Recall()])

# Callbacks
early_stopping = tf.callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=5,
    restore_best_weights=True
)
model_checkpoint = tf.callbacks.ModelCheckpoint(
    os.path.join('models','model.keras'),
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

# Đánh giá mô hình trên tập validation
print("Đánh giá mô hình trên tập validation:")
val_loss, val_acc, val_precision, val_recall = model.evaluate(validation_set)
val_f1 = 2 * (val_precision * val_recall) / (val_precision + val_recall) if (val_precision + val_recall) > 0 else 0
print(f"Validation Loss: {val_loss}")
print(f"Validation Accuracy: {val_acc}")
print(f"Validation Precision: {val_precision}")
print(f"Validation Recall: {val_recall}")
print(f"Validation F1-Score: {val_f1}")

# Vẽ biểu đồ
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 10))

# Biểu đồ Accuracy
plt.subplot(2, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Biểu đồ Loss
plt.subplot(2, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Biểu đồ Precision
plt.subplot(2, 2, 3)
plt.plot(history.history['precision'], label='Training Precision')
plt.plot(history.history['val_precision'], label='Validation Precision')
plt.title('Precision')
plt.xlabel('Epochs')
plt.ylabel('Precision')
plt.legend()

# Biểu đồ Recall
plt.subplot(2, 2, 4)
plt.plot(history.history['recall'], label='Training Recall')
plt.plot(history.history['val_recall'], label='Validation Recall')
plt.title('Recall')
plt.xlabel('Epochs')
plt.ylabel('Recall')
plt.legend()

plt.tight_layout()
plt.show()