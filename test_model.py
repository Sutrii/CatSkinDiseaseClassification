import sys
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Tambahkan folder src ke sys.path agar bisa mengimpor modul dari src/
sys.path.append(os.path.abspath("src"))

# Import modul evaluasi dari evaluation.py
from evaluation import evaluate_model, plot_confusion_matrix

# Path ke model yang akan diuji
MODEL_PATH = os.path.join(os.getcwd(), "models", "vgg16_skin_disease.h5")

# Path ke dataset uji (menggunakan path absolut agar tidak error)
TEST_DATA_DIR = os.path.join(os.getcwd(), "processed_dataset", "test")

# Periksa apakah model ada
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"\u274C Model tidak ditemukan di {MODEL_PATH}")

# Periksa apakah folder dataset uji ada
if not os.path.exists(TEST_DATA_DIR):
    raise FileNotFoundError(f"\u274C Folder dataset uji tidak ditemukan: {TEST_DATA_DIR}")

# Load model
print(f"\U0001F4E5 Memuat model dari {MODEL_PATH} ...")
model = load_model(MODEL_PATH)
print("\u2705 Model berhasil dimuat.")

# Load dataset uji dengan ImageDataGenerator
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255)
test_generator = test_datagen.flow_from_directory(
    TEST_DATA_DIR,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# Tampilkan daftar kelas dalam dataset uji
print("\n\U0001F5C2 Kelas yang terdeteksi dalam dataset uji:")
print(test_generator.class_indices)

# Evaluasi model
print("\n\U0001F680 Evaluasi model sedang berlangsung...")
loss, accuracy = model.evaluate(test_generator, verbose=1)
print(f"\U0001F3AF Akurasi Model: {accuracy:.4f}")
print(f"\U0001F4C9 Loss Model: {loss:.4f}")

# Prediksi & Confusion Matrix
print("\n\U0001F4CA Membuat prediksi...")
y_pred_prob = model.predict(test_generator)
y_pred = np.argmax(y_pred_prob, axis=1)
y_true = test_generator.classes

# Validasi jumlah prediksi
assert len(y_pred) == len(y_true), f"\u274C Jumlah prediksi ({len(y_pred)}) tidak sesuai dengan jumlah data uji ({len(y_true)})!"

# Gunakan fungsi evaluasi tambahan
evaluate_model(y_true, y_pred, dataset_name="Test Set")

# Plot confusion matrix dengan perbaikan
plt.switch_backend('Agg') 
plot_confusion_matrix(y_true, y_pred, labels=list(test_generator.class_indices.keys()), dataset_name="Test Set")
plt.show(block=True) 
plt.close('all')  

print("\n\u2705 Evaluasi selesai!")