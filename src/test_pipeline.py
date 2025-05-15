import pytest
import tensorflow as tf
import numpy as np
import config
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# Load model yang telah dilatih
model = load_model(config.MODEL_SAVE_PATH)

# Label kelas yang digunakan
class_labels = ['Abscess', 'Health', 'Pyoderma', 'Ringworm', 'Scabies']

@pytest.fixture
def sample_image():
    """Membuat gambar acak sebagai input untuk pengujian (1 gambar)."""
    img = np.random.rand(224, 224, 3) * 255  # Gambar RGB acak
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0) / 255.0  # Normalisasi
    return img

@pytest.fixture
def batch_images():
    """Membuat batch gambar acak untuk pengujian batch processing (5 gambar)."""
    batch = np.random.rand(5, 224, 224, 3) * 255
    batch = batch / 255.0  # Normalisasi
    return batch

def test_model_input_shape(sample_image):
    """Memastikan model menerima input dengan ukuran yang benar."""
    assert sample_image.shape == (1, 224, 224, 3), f"Ukuran input tidak sesuai: {sample_image.shape}"

def test_model_output_shape(sample_image):
    """Memastikan output model sesuai dengan jumlah kelas yang diharapkan."""
    prediction = model.predict(sample_image)
    assert prediction.shape == (1, len(class_labels)), f"Output model tidak sesuai: {prediction.shape}"

def test_model_prediction(sample_image):
    """Memastikan model dapat melakukan inferensi tanpa error."""
    prediction = model.predict(sample_image)
    predicted_class = np.argmax(prediction)
    assert 0 <= predicted_class < len(class_labels), "Prediksi model berada di luar rentang kelas yang valid."

def test_model_batch_processing(batch_images):
    """Memastikan model bisa memproses batch input sekaligus."""
    predictions = model.predict(batch_images)
    assert predictions.shape == (5, len(class_labels)), f"Output batch tidak sesuai: {predictions.shape}"

def test_model_softmax_output(sample_image):
    """Memastikan output model merupakan probabilitas dengan jumlah total 1."""
    prediction = model.predict(sample_image)
    total_prob = np.sum(prediction)
    assert np.isclose(total_prob, 1.0, atol=1e-5), f"Output softmax tidak valid: total probabilitas = {total_prob}"
