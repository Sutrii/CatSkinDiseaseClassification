import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# Pastikan path agar bisa mengimpor dari src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Path ke dataset & model
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "processed_dataset"))
VAL_DIR = os.path.join(BASE_DIR, "validation")  # Dataset validasi
TEST_DIR = os.path.join(BASE_DIR, "test")  # Dataset testing
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "vgg16_skin_disease.h5"))

# Path history training
TRAIN_HISTORY_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "training_history.npy"))
FINE_TUNE_HISTORY_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "fine_tuning_history.npy"))

# Kategori penyakit kulit (sesuaikan dengan dataset)
CATEGORIES = ["Abscess", "Health", "Pyoderma", "Ringworm", "Scabies", "Unknown"]

# Periksa apakah folder test ada
if not os.path.exists(TEST_DIR):
    raise FileNotFoundError(f"❌ Folder dataset testing tidak ditemukan: {TEST_DIR}")

# Periksa apakah model tersedia
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Model tidak ditemukan: {MODEL_PATH}")

# Load model
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model berhasil dimuat.")

# Fungsi untuk load gambar & preprocessing
def load_and_preprocess_image(img_path, size=(224, 224)):
    """Load gambar, ubah ke RGB, resize, dan normalisasi."""
    try:
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=size)
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalisasi
        return img_array
    except Exception as e:
        print(f"⚠️ Gagal memproses gambar {img_path}: {e}")
        return None

# Fungsi untuk menghitung prediksi
def get_predictions(data_dir):
    """Menghitung y_true dan y_pred dari dataset testing."""
    y_true, y_pred = [], []

    for label_idx, category in enumerate(CATEGORIES):
        category_path = os.path.join(data_dir, category)
        image_files = os.listdir(category_path) if os.path.exists(category_path) else []

        if len(image_files) == 0:
            print(f"⚠️ Peringatan: Tidak ada gambar dalam kategori {category}")
            continue

        for img_name in image_files:
            img_path = os.path.join(category_path, img_name)
            img_array = load_and_preprocess_image(img_path)

            if img_array is None:
                continue  # Lewati gambar yang gagal diproses

            # Prediksi dengan model
            prediction = model.predict(img_array)
            predicted_label = np.argmax(prediction)

            y_true.append(label_idx)
            y_pred.append(predicted_label)

    return y_true, y_pred

# Fungsi untuk evaluasi model
def evaluate_model(y_true, y_pred, dataset_name="Testing Set"):
    """Evaluasi model dengan confusion matrix dan classification report."""
    if len(y_true) == 0 or len(y_pred) == 0:
        print(f"❌ Tidak ada data yang dapat dievaluasi pada {dataset_name}.")
        return

    # Hitung confusion matrix & classification report
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=CATEGORIES)

    # Tampilkan hasil
    print(f"\n\U0001F4CA Classification Report ({dataset_name}):\n", report)
    plot_confusion_matrix(y_true, y_pred, CATEGORIES, dataset_name)

# Fungsi untuk menampilkan confusion matrix
def plot_confusion_matrix(y_true, y_pred, labels, dataset_name="Testing Set"):
    """Fungsi untuk menampilkan confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Confusion Matrix - {dataset_name}")
    plt.show()

# Fungsi untuk menampilkan grafik akurasi dan loss sebelum & setelah fine-tuning
def plot_training_history():
    """Fungsi untuk membandingkan akurasi dan loss sebelum & setelah fine-tuning."""
    if not os.path.exists(TRAIN_HISTORY_PATH) or not os.path.exists(FINE_TUNE_HISTORY_PATH):
        print("⚠️ History training tidak ditemukan. Pastikan proses training sudah menyimpan history dengan benar.")
        return

    # Load history training dan fine-tuning
    history = np.load(TRAIN_HISTORY_PATH, allow_pickle=True).item()
    history_fine_tune = np.load(FINE_TUNE_HISTORY_PATH, allow_pickle=True).item()

    plt.figure(figsize=(12, 10))

    # Plot Training & Validation Accuracy
    plt.subplot(2, 2, 1)
    plt.plot(history['accuracy'], label='Training Accuracy (Initial)')
    plt.plot(history['val_accuracy'], label='Validation Accuracy (Initial)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training & Validation Accuracy (Initial)')

    plt.subplot(2, 2, 2)
    plt.plot(history_fine_tune['accuracy'], label='Training Accuracy (Fine-Tuned)')
    plt.plot(history_fine_tune['val_accuracy'], label='Validation Accuracy (Fine-Tuned)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training & Validation Accuracy (Fine-Tuned)')

    # Plot Training & Validation Loss
    plt.subplot(2, 2, 3)
    plt.plot(history['loss'], label='Training Loss (Initial)')
    plt.plot(history['val_loss'], label='Validation Loss (Initial)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training & Validation Loss (Initial)')

    plt.subplot(2, 2, 4)
    plt.plot(history_fine_tune['loss'], label='Training Loss (Fine-Tuned)')
    plt.plot(history_fine_tune['val_loss'], label='Validation Loss (Fine-Tuned)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training & Validation Loss (Fine-Tuned)')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_training_history()
    y_true_test, y_pred_test = get_predictions(TEST_DIR)
    evaluate_model(y_true_test, y_pred_test, dataset_name="Testing Set")
