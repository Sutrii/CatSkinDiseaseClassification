import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from PIL import Image

# Path dataset
DATASET_DIR = "processed_dataset"
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VAL_DIR = os.path.join(DATASET_DIR, "validation")
TEST_DIR = os.path.join(DATASET_DIR, "test")

# Pastikan dataset tersedia
for d in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
    if not os.path.exists(d) or not os.listdir(d):
        raise FileNotFoundError(f"❌ Dataset {d} tidak ditemukan atau kosong!")

# Fungsi menghitung jumlah gambar per kelas
def count_images(directory):
    counts = {}
    for cls in os.listdir(directory):
        class_path = os.path.join(directory, cls)
        if os.path.isdir(class_path):
            counts[cls] = len([f for f in os.listdir(class_path) if f.endswith(('jpg', 'jpeg', 'png'))])
    return counts

# Hitung distribusi data
train_counts = count_images(TRAIN_DIR)
val_counts = count_images(VAL_DIR)
test_counts = count_images(TEST_DIR)

# Tampilkan distribusi dalam bar chart
def plot_distribution(counts, title):
    plt.figure(figsize=(8, 5))
    sns.barplot(x=list(counts.keys()), y=list(counts.values()), palette="viridis")
    plt.title(title)
    plt.xlabel("Kelas")
    plt.ylabel("Jumlah Gambar")
    plt.xticks(rotation=45)
    plt.show()

plot_distribution(train_counts, "Distribusi Gambar dalam Training Set")
plot_distribution(val_counts, "Distribusi Gambar dalam Validation Set")
plot_distribution(test_counts, "Distribusi Gambar dalam Test Set")

# Periksa class imbalance
min_class = min(train_counts, key=train_counts.get)
max_class = max(train_counts, key=train_counts.get)
imbalance_ratio = train_counts[max_class] / train_counts[min_class]

if imbalance_ratio > 3:  # Jika rasio > 3x, maka warning
    print(f"⚠️ WARNING: Dataset tidak seimbang! Kelas '{max_class}' ({train_counts[max_class]}) jauh lebih banyak dibanding '{min_class}' ({train_counts[min_class]}). Pertimbangkan oversampling/undersampling.")

# Fungsi menampilkan contoh gambar

def show_sample_images(directory, num_samples=3):
    classes = [cls for cls in os.listdir(directory) if os.path.isdir(os.path.join(directory, cls))]
    num_classes = len(classes)

    fig, axes = plt.subplots(num_classes, num_samples + 1, figsize=(num_samples * 3, num_classes * 3))
    for i, class_name in enumerate(classes):
        class_path = os.path.join(directory, class_name)
        sample_images = [img for img in os.listdir(class_path) if img.endswith(('jpg', 'jpeg', 'png'))][:num_samples]
        axes[i, 0].text(0.5, 0.5, class_name, fontsize=12, fontweight="bold", va="center", ha="center")
        axes[i, 0].axis("off")
        for j, img_name in enumerate(sample_images):
            img_path = os.path.join(class_path, img_name)
            try:
                img = Image.open(img_path).convert("RGB")
                axes[i, j + 1].imshow(img)
                axes[i, j + 1].axis("off")
            except Exception as e:
                print(f"⚠️ Tidak dapat memuat gambar {img_path}: {e}")
    plt.tight_layout()
    plt.show()

show_sample_images(TRAIN_DIR)

# Fungsi menghitung statistik dataset
def compute_dataset_stats(directory):
    img_sizes = []
    for class_name in os.listdir(directory):
        class_path = os.path.join(directory, class_name)
        if not os.path.isdir(class_path):
            continue
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            if not img_name.endswith(('jpg', 'jpeg', 'png')):
                continue
            try:
                with Image.open(img_path) as img:
                    img_sizes.append(img.size)
            except Exception as e:
                print(f"⚠️ Tidak dapat memuat gambar {img_path}: {e}")
    if img_sizes:
        widths, heights = zip(*img_sizes)
        stats = {
            "Total Gambar": len(img_sizes),
            "Rata-rata Ukuran Gambar": f"{np.mean(widths):.1f} x {np.mean(heights):.1f}",
            "Median Ukuran Gambar": f"{np.median(widths)} x {np.median(heights)}"
        }
        return stats
    return {}

train_stats = compute_dataset_stats(TRAIN_DIR)
val_stats = compute_dataset_stats(VAL_DIR)
test_stats = compute_dataset_stats(TEST_DIR)

# Simpan hasil statistik ke file txt
stats_file = "dataset_statistics.txt"
with open(stats_file, "w") as f:
    f.write("📊 Statistik Dataset\n\n")
    f.write("Training Set:\n")
    for key, value in train_stats.items():
        f.write(f"{key}: {value}\n")
    f.write("\nValidation Set:\n")
    for key, value in val_stats.items():
        f.write(f"{key}: {value}\n")
    f.write("\nTest Set:\n")
    for key, value in test_stats.items():
        f.write(f"{key}: {value}\n")

print(f"✅ Statistik dataset disimpan dalam {stats_file}")
