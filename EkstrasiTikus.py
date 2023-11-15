import cv2
import numpy as np
import pandas as pd
import os
from skimage.feature.texture import graycomatrix, graycoprops
from skimage.measure import shannon_entropy
from skimage.measure import regionprops

# Function to extract texture features using GLCM and HSV
def extract_features(image_path):
    # Load the image
    image = cv2.imread(image_path)
    
    # Convert the image to HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_image)

    # Calculate GLCM (Gray-Level Co-occurrence Matrix)
    glcm = graycomatrix(s, [1], [0], symmetric=True, normed=True)
    
    # Calculate texture properties from GLCM
    contrast = graycoprops(glcm, prop='contrast')[0, 0]
    correlation = graycoprops(glcm, prop='correlation')[0, 0]
    energy = graycoprops(glcm, prop='energy')[0, 0]
    homogeneity = graycoprops(glcm, prop='homogeneity')[0, 0]
    dissimilarity = graycoprops(glcm, prop='dissimilarity')[0, 0]
    asm = graycoprops(glcm, prop='ASM')[0, 0]

    # Calculate additional texture properties
    entropy = shannon_entropy(s)

    # Calculate region properties
    props = regionprops(s.astype(int))[0]
    metric = props['major_axis_length']
    eccentricity = props['eccentricity']

    return [np.mean(h), np.mean(s), np.mean(v), dissimilarity, correlation, homogeneity, contrast, asm, energy, entropy, metric, eccentricity]

# List of folder paths
base_folder = r"D:/Kuliah/Project/Python/deteksi_tikus/dateset_tikus"
subfolders = ["ada tikus", "tidak ada tikus"]

feature_data = []

for folder_name in subfolders:
    folder_path = os.path.join(base_folder, folder_name)
    if os.path.exists(folder_path):
        for filename in os.listdir(folder_path):
            if filename.lower().endswith((".jpg", ".png", ".JPG", ".PNG")):
                image_path = os.path.join(folder_path, filename)
                texture_features = extract_features(image_path)
                feature_data.append([filename, folder_name] + texture_features)

# Create a DataFrame
data = pd.DataFrame(feature_data, columns=['Nama Image', 'Kategori', 'Average H', 'Average S', 'Average V', 'GLCM Dissimilarity', 'GLCM Correlation', 'GLCM Homogeneity', 'GLCM Contrast', 'GLCM ASM', 'GLCM Energy', 'Entropy', 'Metric', 'Eccentricity'])

# Sort the DataFrame by 'Nama Image'
data = data.sort_values(by='Nama Image')

# Create an Excel writer to save data to a single sheet
excel_writer = pd.ExcelWriter('ekstrasi-data-tikus.xlsx', engine='xlsxwriter')

# Write data to the Excel file
data.to_excel(excel_writer, sheet_name='Data', index=False)

# Close the Excel file
excel_writer.close()
