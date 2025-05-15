import os
import shutil
import random

# Path dataset awal dan tujuan
dataset_dir = "dataset"
output_dir = "processed_dataset"

# Proporsi data
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

# Pastikan folder output bersih
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir)

# Menyiapkan struktur folder
categories = os.listdir(dataset_dir)
for category in categories:
    os.makedirs(os.path.join(output_dir, "train", category), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "validation", category), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "test", category), exist_ok=True)

# Memindahkan file ke dalam train/validation/test
for category in categories:
    category_path = os.path.join(dataset_dir, category)
    images = os.listdir(category_path)
    random.shuffle(images)

    train_split = int(train_ratio * len(images))
    val_split = int(val_ratio * len(images))

    train_images = images[:train_split]
    val_images = images[train_split:train_split + val_split]
    test_images = images[train_split + val_split:]

    for img_name in train_images:
        shutil.copy(os.path.join(category_path, img_name), os.path.join(output_dir, "train", category, img_name))
    for img_name in val_images:
        shutil.copy(os.path.join(category_path, img_name), os.path.join(output_dir, "validation", category, img_name))
    for img_name in test_images:
        shutil.copy(os.path.join(category_path, img_name), os.path.join(output_dir, "test", category, img_name))

print("✅ Preprocessing selesai! Dataset sudah dipisahkan ke train, validation, dan test.")
