import argparse
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input
import matplotlib.pyplot as plt
import os
from math import sqrt

def load_image(image_path):
    image = cv2.imread(image_path)
    image_resized = cv2.resize(image, (224, 224))
    image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
    image_preprocessed = preprocess_input(image_rgb.astype(np.float32))
    return image, np.expand_dims(image_preprocessed, axis=0), image_rgb

def make_gradcam_heatmap(img_array, model, last_conv_layer_name="block5_conv3", pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs], 
        [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def apply_hsv_mask(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 70, 30])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 70, 30])
    upper_red2 = np.array([180, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    full_mask = cv2.bitwise_or(mask1, mask2)

    kernel = np.ones((5, 5), np.uint8)
    full_mask = cv2.morphologyEx(full_mask, cv2.MORPH_CLOSE, kernel)

    if np.count_nonzero(full_mask) == 0:
        print("⚠️ Tidak bisa menghitung centroid (mask kosong?)")
    else:
        print(f"✅ Jumlah piksel luka (mask merah): {np.count_nonzero(full_mask)}")

    return full_mask

def calculate_centroid(mask):
    M = cv2.moments(mask)
    if M["m00"] == 0:
        return None
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return (cx, cy)

def calculate_iou(mask1, mask2, gt_mask):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    overlap_ratio = intersection / gt_mask.sum() if gt_mask.sum() != 0 else 0

    print(f"🟣 Area Overlap Ratio (intersection / GT mask): {overlap_ratio:.2f}")
    if union == 0:
        return 0.0
    return intersection / union

def overlay_heatmap_on_image(image, heatmap):
    heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image, 0.6, heatmap_colored, 0.4, 0)
    return overlay, heatmap_resized

def main(image_path):
    model = load_model("models/vgg16_skin_disease_finetuned.h5", compile=False)
    original_image, input_image, image_rgb = load_image(image_path)
    heatmap = make_gradcam_heatmap(input_image, model)
    overlayed, heatmap_resized = overlay_heatmap_on_image(original_image, heatmap)

    # Ambil titik maksimum dari Grad-CAM
    max_loc = np.unravel_index(np.argmax(heatmap_resized), heatmap_resized.shape)
    max_x, max_y = max_loc[1], max_loc[0]

    # Mask luka merah HSV
    hsv_mask = apply_hsv_mask(original_image)

    if np.count_nonzero(hsv_mask) > 0:
        centroid = calculate_centroid(hsv_mask)
        if centroid:
            cx, cy = centroid
            if hsv_mask[cy, cx] > 0:
                print("✅ Pointing Game Hit: Yes")
                pointing_game_result = "Pointing Game Hit: Yes"
            else:
                print("❌ Pointing Game Hit: No")
                pointing_game_result = "Pointing Game Hit: No"

            # Jarak Euclidean antara titik max heat dan centroid luka
            distance = sqrt((cx - max_x) ** 2 + (cy - max_y) ** 2)
            print(f"📏 Jarak Max Heat ke Centroid Luka: {distance:.2f} px")
            centroid_distance = f"Distance: {distance:.2f} px"
        else:
            print("⚠️ Tidak bisa hitung centroid.")
            pointing_game_result = "Pointing Game Hit: No"
            centroid_distance = "Distance: N/A"
    else:
        print("⚠️ Mask HSV kosong. Evaluasi dihentikan.")
        return

    # Threshold heatmap dengan percentile
    threshold = np.percentile(heatmap_resized, 85)
    heatmap_mask = heatmap_resized > threshold

    # Hitung IoU dan Overlap Ratio
    iou = calculate_iou(heatmap_mask, hsv_mask > 0, hsv_mask > 0)
    print(f"📊 IoU Heatmap vs Mask Merah: {iou:.4f}")

    # Visualisasi
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    ax[0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    ax[0].set_title("Original Image"); ax[0].axis("off")

    ax[1].imshow(overlayed)
    ax[1].scatter([max_x], [max_y], color='cyan', s=40, label="Max Heat")
    ax[1].scatter([cx], [cy], color='yellow', s=40, label="Centroid")
    ax[1].plot([max_x, cx], [max_y, cy], color='white', linestyle='--', label="Centroid to Max Heat")
    ax[1].set_title("Grad-CAM Overlay")
    ax[1].legend(); ax[1].axis("off")

    ax[2].imshow(hsv_mask, cmap="gray")
    ax[2].set_title("HSV Red Mask (Luka)"); ax[2].axis("off")

    plt.tight_layout()

    # Menambahkan teks Pointing Game dan Centroid Distance ke gambar
    plt.figtext(0.5, 0.02, pointing_game_result, ha="center", fontsize=12, color="white", bbox={"facecolor": "black", "alpha": 0.5, "pad": 5})
    plt.figtext(0.5, 0.01, centroid_distance, ha="center", fontsize=12, color="white", bbox={"facecolor": "black", "alpha": 0.5, "pad": 5})

    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path ke gambar input")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print("Gambar tidak ditemukan.")
    else:
        main(args.image)
