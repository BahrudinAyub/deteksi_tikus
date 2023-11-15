import os
import cv2
import numpy as np
import pandas as pd
from keras.models import load_model

# Load pre-trained CNN model (misalnya, model untuk deteksi objek dengan menggunakan model MobileNetV2)
model = load_model('path_to_your_model.h5')

# Mendefinisikan kelas objek yang ingin dideteksi (misalnya, dalam format COCO)
classes = ['class1', 'class2', 'class3']  # Gantilah dengan kelas-kelas yang sesuai dengan model Anda

# Fungsi untuk mendeteksi objek pada gambar
def detect_objects(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))  # Gantilah ukuran sesuai dengan model Anda

    # Normalisasi gambar
    image = image / 255.0

    # Prediksi objek pada gambar
    predictions = model.predict(np.expand_dims(image, axis=0))
    class_id = np.argmax(predictions)

    # Ambil label kelas yang sesuai
    object_class = classes[class_id]

    return object_class

# Membaca semua gambar dari direktori
image_dir = 'path_to_directory_containing_images'
results = []

for image_filename in os.listdir(image_dir):
    if image_filename.endswith(('.jpg', '.png', '.jpeg')):
        image_path = os.path.join(image_dir, image_filename)
        detected_class = detect_objects(image_path)
        results.append({'Image File': image_filename, 'Detected Object': detected_class})

# Membuat DataFrame Pandas dari hasil
df = pd.DataFrame(results)

# Menyimpan hasil ke dalam file Excel
output_excel_file = 'output_detection_results.xlsx'
df.to_excel(output_excel_file, index=False, engine='openpyxl')
