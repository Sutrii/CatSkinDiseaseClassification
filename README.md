# Skin Disease Classification

Proyek ini bertujuan untuk mengklasifikasikan penyakit kulit pada hewan, menggunakan model deep learning berbasis VGG16 yang dilatih dengan dataset gambar penyakit kulit. Model ini dapat mengklasifikasikan gambar ke dalam lima kategori:

- **Abscess**
- **Health**
- **Pyoderma**
- **Ringworm**
- **Scabies**

## Struktur Folder

- `dataset/`: Folder yang berisi dataset asli sebelum diproses.
- `processed_dataset/`: Folder yang berisi dataset yang telah diproses (terdiri dari sub-folder `train/`, `validation/`, dan `test/`).
- `src/`: Folder yang berisi skrip sumber daya:
  - `preprocess_dataset.py`: Skrip untuk memproses dan mempersiapkan dataset.
  - `train.py`: Skrip untuk melatih model.
  - `predict.py`: Skrip untuk memprediksi gambar baru menggunakan model yang sudah dilatih.
  - `pipeline.py`: Skrip utama untuk menjalankan seluruh pipeline (preprocessing, training, dan inferensi).
  - `config.py`: File konfigurasi untuk hyperparameter dan paths.
  - `test_pipeline.py`: Skrip untuk unit testing pipeline model.
- `models/`: Folder untuk menyimpan model yang sudah dilatih.
- `docs/`: Folder untuk dokumentasi proyek.
- `requirements.txt`: File yang berisi daftar pustaka Python yang diperlukan untuk menjalankan proyek.
- `README.md`: Dokumen ini.

## Instalasi

Untuk menjalankan proyek ini, pastikan Anda memiliki Python yang sudah terpasang, lalu instal dependensi dengan perintah berikut:

```bash
pip install -r requirements.txt
```

## Cara Menggunakan

### 1. Preprocessing Dataset
Jika dataset belum diproses, jalankan skrip berikut:
```bash
python src/preprocess_dataset.py
```

### 2. Melatih Model
Untuk melatih model, gunakan perintah berikut:
```bash
python src/train.py
```

### 3. Melakukan Inferensi
Untuk memprediksi gambar baru, jalankan:
```bash
python src/predict.py --image_path path/to/image.jpg
```

### 4. Menjalankan Pipeline Lengkap
Jika ingin menjalankan seluruh pipeline dari preprocessing hingga inferensi:
```bash
python src/pipeline.py --image_path path/to/image.jpg
```

### 5. Pengujian Model
Jalankan unit test untuk memastikan pipeline berjalan dengan baik:
```bash
pytest src/test_pipeline.py -v
```

## Catatan
- Pastikan dataset sudah tersedia di folder `dataset/` sebelum menjalankan preprocessing.
- Model hasil training akan disimpan di folder `models/`.
- Gunakan file `config.py` untuk menyesuaikan parameter training dan jalur dataset.

Dengan README ini, pengguna dapat memahami struktur proyek, cara instalasi, serta menjalankan pipeline secara keseluruhan. 🚀