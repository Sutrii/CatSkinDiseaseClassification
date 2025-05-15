import os
import tensorflow as tf

# Hyperparameter
IMG_SIZE = (224, 224)  # Ukuran input untuk VGG16
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.0001

# Path Dataset
BASE_DIR = os.getcwd()  # Direktori kerja saat ini
TRAIN_DIR = os.path.join(BASE_DIR, "processed_dataset", "train")
VAL_DIR = os.path.join(BASE_DIR, "processed_dataset", "validation")
TEST_DIR = os.path.join(BASE_DIR, "processed_dataset", "test")

# Path Model
MODEL_DIR = os.path.join(BASE_DIR, "models")  # Menggunakan folder models
MODEL_H5_PATH = os.path.join(MODEL_DIR, "vgg16_skin_disease.h5")
MODEL_SAVEDMODEL_PATH = os.path.join(MODEL_DIR, "vgg16_skin_disease")
TFLITE_MODEL_PATH = os.path.join(MODEL_DIR, "vgg16_skin_disease.tflite")

# Pastikan folder models ada
os.makedirs(MODEL_DIR, exist_ok=True)

# Cek apakah model dalam format H5 tersedia, jika tidak, konversi dari SavedModel
if not os.path.exists(MODEL_H5_PATH) and os.path.exists(MODEL_SAVEDMODEL_PATH):
    print("🔄 Konversi model dari SavedModel ke H5...")
    model = tf.keras.models.load_model(MODEL_SAVEDMODEL_PATH)
    model.save(MODEL_H5_PATH)
    print(f"✅ Model berhasil dikonversi dan disimpan sebagai: {MODEL_H5_PATH}")

# Menggunakan model dalam format H5 sebagai default
MODEL_SAVE_PATH = MODEL_H5_PATH
