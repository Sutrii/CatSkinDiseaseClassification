import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers.schedules import CosineDecay
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# Path Dataset & Model
TRAIN_DIR = "processed_dataset/train"
VAL_DIR = "processed_dataset/validation"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Path untuk menyimpan history training
TRAIN_HISTORY_PATH = os.path.join(MODEL_DIR, "training_history.npy")
FINE_TUNE_HISTORY_PATH = os.path.join(MODEL_DIR, "fine_tuning_history.npy")

# Pastikan dataset tersedia
if not os.path.exists(TRAIN_DIR) or not os.path.exists(VAL_DIR):
    raise FileNotFoundError("❌ Dataset tidak ditemukan! Pastikan folder processed_dataset tersedia.")

# Load Model VGG16 sebagai Feature Extractor
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

# Fully Connected Layer dengan L2 Regularization dan Swish Activation
x = Flatten()(base_model.output)
x = Dense(1024, activation=tf.keras.activations.swish, kernel_regularizer=l2(0.0005))(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(512, activation=tf.keras.activations.swish, kernel_regularizer=l2(0.0005))(x)
x = BatchNormalization()(x)
x = Dropout(0.4)(x)

x = Dense(256, activation=tf.keras.activations.swish, kernel_regularizer=l2(0.0005))(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)
output_layer = Dense(6, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output_layer)

# Scheduler Learning Rate
lr_schedule = CosineDecay(initial_learning_rate=0.001, decay_steps=1000, alpha=0.0001)

# Compile Model
model.compile(optimizer=AdamW(learning_rate=lr_schedule),  
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# Data Augmentasi yang Ditingkatkan
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    brightness_range=[0.7, 1.3],
    fill_mode='nearest',
    featurewise_center=True,
    featurewise_std_normalization=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR, target_size=(224, 224), batch_size=32, class_mode='categorical'
)
val_generator = val_datagen.flow_from_directory(
    VAL_DIR, target_size=(224, 224), batch_size=32, class_mode='categorical'
)

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-7, verbose=1)
checkpoint = ModelCheckpoint(os.path.join(MODEL_DIR, "vgg16_skin_disease.h5"),
                             monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)

# **TRAINING AWAL**
print("🚀 Memulai training awal...")
history = model.fit(
    train_generator, validation_data=val_generator, epochs=25,  
    callbacks=[early_stopping, reduce_lr, checkpoint]
)

np.save(TRAIN_HISTORY_PATH, history.history)
print(f"✅ History training awal disimpan di {TRAIN_HISTORY_PATH}")

initial_model_path = os.path.join(MODEL_DIR, "vgg16_skin_disease_initial.h5")
model.save(initial_model_path)
print(f"✅ Model awal disimpan: {initial_model_path}")

# **FINE-TUNING**
model = load_model(os.path.join(MODEL_DIR, "vgg16_skin_disease.h5"))

# Buka lebih banyak layer untuk fine-tuning
for layer in model.layers[-20:]:
    layer.trainable = True  

# Recompile dengan Learning Rate lebih kecil
model.compile(optimizer=AdamW(learning_rate=5e-6),  
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

print("🚀 Memulai fine-tuning...")
history_fine_tune = model.fit(
    train_generator, validation_data=val_generator, epochs=40, initial_epoch=25,  
    callbacks=[early_stopping, reduce_lr, checkpoint]
)

np.save(FINE_TUNE_HISTORY_PATH, history_fine_tune.history)
print(f"✅ History fine-tuning disimpan di {FINE_TUNE_HISTORY_PATH}")

fine_tuned_model_path = os.path.join(MODEL_DIR, "vgg16_skin_disease_finetuned.h5")
model.save(fine_tuned_model_path)
print(f"✅ Model setelah fine-tuning disimpan: {fine_tuned_model_path}")

# **Konversi ke TensorFlow Lite**
print("🔄 Mengonversi model ke TensorFlow Lite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

TFLITE_MODEL_PATH = os.path.join(MODEL_DIR, "vgg16_skin_disease.tflite")
with open(TFLITE_MODEL_PATH, "wb") as f:
    f.write(tflite_model)

print(f"✅ Model berhasil dikonversi ke TensorFlow Lite: {TFLITE_MODEL_PATH}")
