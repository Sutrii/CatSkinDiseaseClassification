import sys
import os
import argparse
import numpy as np
import tensorflow as tf
import time
import psutil
from tensorflow.keras.preprocessing import image

# Path model
MODEL_PATH = "models/vgg16_skin_disease_finetuned.h5"

# Pastikan model ada sebelum load
if not os.path.exists(MODEL_PATH):
    print(f"❌ Model tidak ditemukan di {MODEL_PATH}. Pastikan model sudah dilatih dan disimpan.")
    sys.exit(1)

# Load model
print(f"📥 Memuat model dari {MODEL_PATH} ...")
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model berhasil dimuat.")

# Label kelas (sesuaikan dengan dataset)
class_names = ["Abscess", "Health", "Pyoderma", "Ringworm", "Scabies", "Unknown"]

def get_cpu_ram_usage():
    """Mengambil penggunaan CPU dan RAM saat ini."""
    cpu_percent = psutil.cpu_percent(interval=0.1)
    memory = psutil.virtual_memory()
    ram_percent = memory.percent
    return cpu_percent, ram_percent

def get_gpu_memory_usage():
    """Mengambil penggunaan memori GPU (jika tersedia)."""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            info = tf.config.experimental.get_memory_info('GPU:0')
            used = info['current'] / (1024 ** 2)  # dalam MB
            peak = info['peak'] / (1024 ** 2)
            return used, peak
        except:
            return None, None
    return None, None

def predict_image(image_path):
    """Melakukan prediksi pada satu gambar dan mengembalikan hasilnya."""

    if not os.path.exists(image_path):
        print(f"❌ File gambar tidak ditemukan: {image_path}")
        return None

    try:
        img = image.load_img(image_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
    except Exception as e:
        print(f"❌ Gagal memuat gambar: {e}")
        return None

    # Catat penggunaan CPU dan RAM sebelum prediksi
    cpu_before, ram_before = get_cpu_ram_usage()

    # Catat penggunaan GPU sebelum
    gpu_used_before, _ = get_gpu_memory_usage()

    start_time = time.time()
    predictions = model.predict(img_array)
    elapsed_time = time.time() - start_time

    cpu_after, ram_after = get_cpu_ram_usage()
    gpu_used_after, gpu_peak = get_gpu_memory_usage()

    predicted_class_index = np.argmax(predictions)
    predicted_class = class_names[predicted_class_index]
    confidence = np.max(predictions)

    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "probabilities": predictions[0],
        "elapsed_time": elapsed_time,
        "cpu_usage": cpu_after,
        "ram_usage": ram_after,
        "gpu_used": gpu_used_after,
        "gpu_peak": gpu_peak
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prediksi penyakit kulit hewan dari satu gambar.")
    parser.add_argument("image_path", type=str, help="Path ke gambar yang akan diprediksi.")
    args = parser.parse_args()

    result = predict_image(args.image_path)

    if result:
        print("\n📊 Hasil Prediksi")
        print("─────────────────────────────────")
        for i, class_name in enumerate(class_names):
            print(f"➡️ {class_name}: {result['probabilities'][i]:.4f}")
        print("─────────────────────────────────")
        print(f"✅ Prediksi Akhir: {result['predicted_class']} ({result['confidence']:.2%} yakin)")
        print(f"⏱ Waktu prediksi: {result['elapsed_time']:.4f} detik")
        print(f"🧠 CPU usage: {result['cpu_usage']}%")
        print(f"💾 RAM usage: {result['ram_usage']}%")
        if result['gpu_used'] is not None:
            print(f"🖥 GPU Mem Used: {result['gpu_used']:.2f} MB (Peak: {result['gpu_peak']:.2f} MB)")
        else:
            print("❗ GPU info tidak tersedia atau tidak digunakan.")
    else:
        print("❌ Prediksi gagal.")
