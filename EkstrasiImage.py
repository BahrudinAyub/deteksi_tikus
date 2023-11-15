import cv2
import numpy as np
import pandas as pd
import os
from skimage.feature.texture import graycomatrix, graycoprops

# Function to extract texture features using GLCM
def extract_texture_features(image_path):
    # Load the image
    image = cv2.imread(image_path)
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate GLCM (Gray-Level Co-occurrence Matrix)
    glcm = graycomatrix(gray, [1], [0], symmetric=True, normed=True)
    
    # Calculate texture properties from GLCM
    contrast = graycoprops(glcm, prop='contrast')[0, 0]
    correlation = graycoprops(glcm, prop='correlation')[0, 0]
    energy = graycoprops(glcm, prop='energy')[0, 0]
    homogeneity = graycoprops(glcm, prop='homogeneity')[0, 0]
    
    # Calculate the average value of each channel (R, G, B) in the image
    r, g, b = cv2.split(image)
    avg_r = np.mean(r)
    avg_g = np.mean(g)
    avg_b = np.mean(b)
    
    return [avg_r, avg_g, avg_b, contrast, correlation, energy, homogeneity]

# List of folder paths
base_folder = r"D:/VISUAL/DATASET_TELUR/img"
subfolders = ["utuh", "retak"]

feature_data = []

for folder_name in subfolders:
    folder_path = os.path.join(base_folder, folder_name)
    if os.path.exists(folder_path):
        for filename in os.listdir(folder_path):
            if filename.lower().endswith((".jpg", ".png", ".JPG", ".PNG")):
                image_path = os.path.join(folder_path, filename)
                texture_features = extract_texture_features(image_path)
                feature_data.append([filename, folder_name] + texture_features)

# Create a DataFrame
data = pd.DataFrame(feature_data, columns=['Nama Image', 'Kategori', 'Average R', 'Average G', 'Average B', 'Contrast', 'Correlation', 'Energy', 'Homogeneity'])

# Sort the DataFrame by 'Nama Image'
data = data.sort_values(by='Nama Image')

# Create an Excel writer to save data to a single sheet
excel_writer = pd.ExcelWriter('ekstrasi-data-telur.xlsx', engine='xlsxwriter')

# Write data to the Excel file
data.to_excel(excel_writer, sheet_name='Data', index=False)

# Close the Excel file
excel_writer.close()
