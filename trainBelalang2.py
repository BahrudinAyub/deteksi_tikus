import cv2
import numpy as np
import xlsxwriter 
from collections import Counter
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import label, regionprops
from sklearn.cluster import KMeans

workbook = xlsxwriter.Workbook('trainTikus.xlsx', {'nan_inf_to_errors': True})
worksheet = workbook.add_worksheet()

jenis = ['Ada Tikus','Tidak Ada Tikus']

hsv_properties = ['hue','saturation','value']
glcm_properties = ['dissimilarity', 'correlation', 'homogeneity', 'contrast', 'ASM', 'energy']
shape_properties = ['metric', 'eccentricity']

# Writing excel header
header = ['File']
header.extend(['HSV_' + c for c in ('Hue', 'Saturation', 'Value')])
header.extend(['GLCM_' + prop for prop in glcm_properties])
header.extend(shape_properties)
header.append('Class')

for col, value in enumerate(header):
    worksheet.write(0, col, value)

baris = 1

for j in range(1, 150):  # Rentang gambar dari 1 hingga 75       
    kolom = 0
    file_name = "D:/Kuliah/Project/Python/deteksi_tikus/dateset_tikus/" + "tikus " + str(j) + ".jpg"
    print(file_name)
    worksheet.write(baris, kolom, file_name)
    kolom += 1

    src = cv2.imread(file_name)
    if src is None:
        print(f"Unable to read the image: {file_name}")
        continue

    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
    
    # Extracting HSV dominant color
    hsv_image = hsv.reshape((hsv.shape[0] * hsv.shape[1], 3))
    clt = KMeans(n_clusters=3)
    labels = clt.fit_predict(hsv_image)
    label_counts = Counter(labels)
    dom_color = clt.cluster_centers_[label_counts.most_common(1)[0][0]]
    for val in dom_color:
        worksheet.write(baris, kolom, val)
        kolom += 1

    # GLCM
    glcm = graycomatrix(gray, 
                        distances=[5], 
                        angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], 
                        levels=256,
                        symmetric=True, 
                        normed=True)
    glcm_props = [prop for name in glcm_properties for prop in graycoprops(glcm, name)[0]]
    for item in glcm_props:
        worksheet.write(baris, kolom, item)
        kolom += 1

    # Shape
    label_img = label(gray)
    props = regionprops(label_img)
    eccentricity = props[0].eccentricity
    area = props[0].area
    perimeter = props[0].perimeter
    metric = (4 * np.pi * area) / (perimeter * perimeter)
    worksheet.write(baris, kolom, metric)
    kolom += 1
    worksheet.write(baris, kolom, eccentricity)
    kolom += 1

    # Menentukan label "Ada Belalang" atau "Tidak Ada Belalang" berdasarkan nomor gambar
    label_kelas = 'Terdeteksi Terdapat Tikus' if j >= 1 and j <= 140 else 'Tidak Ada Tikus Terdeteksi'
    worksheet.write(baris, kolom, label_kelas)
    kolom += 1

    baris += 1

workbook.close()