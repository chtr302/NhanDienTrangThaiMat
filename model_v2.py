import os
import cv2
import numpy as np
import keras as tf
from keras._tf_keras.keras import layers, models, optimizers
import albumentations as A
import random
from tensorflow import reduce_mean, reduce_max
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

TRAIN_DIR = 'data/train'
TEST_DIR = 'data/test'
CLASSES = ['Closed', 'Open']
IMG_SIZE = (80, 80)
BATCH_SIZE = 32
EPOCHS = 50

# Tăng cường hình ảnh
transform = A.Compose([
    A.Resize(IMG_SIZE[0], IMG_SIZE[1]), # Kích thước
    A.Affine(shear=(-15, 15), rotate=(-20, 20), scale=(0.85, 1.15), p=0.7), # Xoay, trượt, phóng to/thu nhỏ
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5), # Thay đổi độ sáng, tương phản
    A.RandomGamma(gamma_limit=(80, 120), p=0.5), 
    A.CoarseDropout(p=0.5), # Tạo các lỗ đen ngẫu nhiên trên ảnh
    A.HorizontalFlip(p=0.5), # Lật
])

# Ở đây bởi vì albumentations không phải là thành phần của keras cho nên tạo class này
class CustomDataGenerator(tf.utils.Sequence):
    """Cung cấp dữ liệu cho model"""
    def __init__(self, classes, batch_size, img_size, augmentations=None, directory=None, image_paths=None, labels=None):
        self.directory = directory
        self.classes = classes
        self.batch_size = batch_size
        self.img_size = img_size
        self.augmentations = augmentations
        self.class_map = {class_name: i for i, class_name in enumerate(classes)}
        if directory:
            self.image_paths, self.labels = self._load_paths_from_dir(directory)
        elif image_paths is not None and labels is not None:
            self.image_paths = image_paths
            self.labels = labels
        else:
            raise ValueError("Cần cung cấp 'directory' hoặc cả 'image_paths' và 'labels'.")
        
        self.on_epoch_end()

    def _load_paths_from_dir(self, directory):
        """Load dữ liệu"""
        image_paths = []
        labels = []
        for class_name in self.classes:
            class_dir = os.path.join(directory, class_name)
            if not os.path.isdir(class_dir):
                raise FileNotFoundError(f"Thư mục lớp không tồn tại: {class_dir}")
            for img_name in os.listdir(class_dir):
                if not img_name.startswith('.'):
                    image_paths.append(os.path.join(class_dir, img_name))
                    labels.append(self.class_map[class_name])
        return image_paths, labels

    def __len__(self):
        return int(np.ceil(len(self.image_paths) / self.batch_size))

    def __getitem__(self, index):
        """
        Keras sẽ dùng hàm này để lấy 1 batch dữ liệu
        Args:
            index: số thứ tự của batch
        Returns:
            Danh sách ảnh và nhãn
        """
        batch_paths = self.image_paths[index * self.batch_size:(index + 1) * self.batch_size]
        batch_labels = self.labels[index * self.batch_size:(index + 1) * self.batch_size]

        batch_images = []
        for img_path in batch_paths:
            try:
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Cảnh báo: Không thể đọc ảnh tại {img_path}")
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                if self.augmentations:
                    processed_img = self.augmentations(image=img)['image']
                else:
                    processed_img = cv2.resize(img, self.img_size)

                batch_images.append(processed_img / 255.0)
            except Exception as e:
                continue
        batch_images_np = np.array(batch_images)
        batch_labels_np = np.array(batch_labels)

        return batch_images_np, batch_labels_np.reshape(-1, 1)

    def on_epoch_end(self):
        """
        Xáo trộn ngẫu nhiên thứ tự các cặp
        """
        temp = list(zip(self.image_paths, self.labels))
        random.shuffle(temp)
        self.image_paths, self.labels = zip(*temp)

class SpatialAttention(layers.Layer):
    """
    Custom Layer cho chương trình. class này sẽ triển khai cơ chế Chú ý theo không gian, giúp xác định Ở ĐÂU
    """
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
        avg_pool = reduce_mean(inputs, axis=3, keepdims=True)
        max_pool = reduce_max(inputs, axis=3, keepdims=True)
        
        concat = layers.Concatenate(axis=3)([avg_pool, max_pool])
        
        attention = self.conv2d(concat)
        
        return layers.Multiply()([inputs, attention])

    def get_config(self):
        config = super(SpatialAttention, self).get_config()
        config.update({'kernel_size': self.kernel_size})
        return config

def channel_attention_module(x, ratio=8):
    """Xây dựng cơ chế chú ý theo channel"""
    channel = x.shape[-1]
    
    shared_layer_one = layers.Dense(channel // ratio,
                                    activation='relu',
                                    kernel_initializer='he_normal',
                                    use_bias=True,
                                    bias_initializer='zeros')
    shared_layer_two = layers.Dense(channel,
                                    kernel_initializer='he_normal',
                                    use_bias=True,
                                    bias_initializer='zeros')
    
    # Average Pooling
    avg_pool = layers.GlobalAveragePooling2D()(x)    
    avg_pool = layers.Reshape((1, 1, channel))(avg_pool)
    avg_pool = shared_layer_one(avg_pool)
    avg_pool = shared_layer_two(avg_pool)
    
    # Max Pooling
    max_pool = layers.GlobalMaxPooling2D()(x)
    max_pool = layers.Reshape((1, 1, channel))(max_pool)
    max_pool = shared_layer_one(max_pool)
    max_pool = shared_layer_two(max_pool)
    
    # Cộng và kích hoạt Sigmoid
    attention = layers.Add()([avg_pool, max_pool])
    attention = layers.Activation('sigmoid')(attention)
    
    return layers.Multiply()([x, attention])

def cbam_block(x, ratio=8):
    x = channel_attention_module(x, ratio)
    x = SpatialAttention()(x)
    return x

def build_cnn_with_attention(input_shape=(80, 80, 3)):
    inputs = layers.Input(shape=input_shape)

    # Block 1
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = cbam_block(x)
    x = layers.MaxPool2D(2)(x)
    x = layers.Dropout(0.25)(x)

    # Block 2
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = cbam_block(x)
    x = layers.MaxPool2D(2)(x)
    x = layers.Dropout(0.25)(x)

    # Block 3
    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = cbam_block(x)
    x = layers.MaxPool2D(2)(x)
    x = layers.Dropout(0.25)(x)
    
    # Block 4
    x = layers.Conv2D(256, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(256, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = cbam_block(x)
    x = layers.MaxPool2D(2)(x)
    x = layers.Dropout(0.3)(x)

    # Lớp cuối
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model(inputs, outputs)
    return model

if __name__ == "__main__":
    if not os.path.isdir(TRAIN_DIR) or not os.path.isdir(TEST_DIR):
        print(f"LỖI: Không tìm thấy thư mục {TRAIN_DIR} hoặc {TEST_DIR}")
    else:
        try:
            all_train_paths = []
            all_train_labels = []
            class_map = {class_name: i for i, class_name in enumerate(CLASSES)}
            for class_name in CLASSES:
                class_dir = os.path.join(TRAIN_DIR, class_name)
                for img_name in os.listdir(class_dir):
                    if not img_name.startswith('.'):
                        all_train_paths.append(os.path.join(class_dir, img_name))
                        all_train_labels.append(class_map[class_name])

            train_paths, val_paths, train_labels, val_labels = train_test_split(
                all_train_paths, all_train_labels, test_size=0.2, random_state=42, stratify=all_train_labels
            )

            train_generator = CustomDataGenerator(
                image_paths=train_paths, labels=train_labels, classes=CLASSES, 
                batch_size=BATCH_SIZE, img_size=IMG_SIZE, augmentations=transform
            )
            validation_generator = CustomDataGenerator(
                image_paths=val_paths, labels=val_labels, classes=CLASSES,
                batch_size=BATCH_SIZE, img_size=IMG_SIZE
            )
            test_generator = CustomDataGenerator(
                directory=TEST_DIR, classes=CLASSES, batch_size=BATCH_SIZE, img_size=IMG_SIZE
            )
            

            print("--- Build mô hình ---")
            model = build_cnn_with_attention()
            f1_metric = tf.metrics.F1Score(threshold=0.5)
            model.compile(
                optimizer=optimizers.Adam(learning_rate=0.001),
                loss='binary_crossentropy',
                metrics=['accuracy', f1_metric]
            )
            model.summary()

            print("--- Bắt đầu ---")
            history = model.fit(
                train_generator,
                epochs=EPOCHS,
                validation_data=validation_generator,
                callbacks=[
                    tf.callbacks.EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True),
                    tf.callbacks.ModelCheckpoint('eye_model_v2_attention.keras', monitor='val_accuracy',  save_best_only=True, verbose=1),
                    tf.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)
                ]
            )
            
            print("--- Xong ---")

            print("\n--- Predict với tập test ---")
            # Tải lại model tốt nhất đã lưu
            best_model = models.load_model('eye_model_v2_attention.keras', custom_objects={'f1_score': f1_metric, 'SpatialAttention': SpatialAttention})
            
            # Đánh giá bằng model.evaluate
            results = best_model.evaluate(test_generator, verbose=1)
            print(f"Test Loss: {results[0]:.4f}")
            print(f"Test Accuracy: {results[1]:.4f}")
            print(f"Test F1-Score: {results[2]:.4f}")

            # In báo cáo chi tiết (Precision, Recall, F1)
            print("\n--- Báo cáo phân loại chi tiết trên tập TEST ---")
            y_true_test = test_generator.labels
            y_pred_probs_test = best_model.predict(test_generator)
            y_pred_test = (y_pred_probs_test > 0.5).astype(int).reshape(-1)
            
            num_samples = len(y_pred_test)
            print(classification_report(y_true_test[:num_samples], y_pred_test, target_names=CLASSES))

        except Exception as e:
            print(f"Đã xảy ra lỗi không mong muốn: {e}")