import cv2
import numpy as np
from skimage.feature.texture import graycomatrix, graycoprops
from rembg import remove
from PIL import Image
import joblib

# Load the trained model
model = joblib.load('trained_model.joblib')

# Function to extract texture features using GLCM
def extract_texture_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    glcm = graycomatrix(gray, [1], [0], symmetric=True, normed=True)
    
    contrast = graycoprops(glcm, prop='contrast')[0, 0]
    correlation = graycoprops(glcm, prop='correlation')[0, 0]
    energy = graycoprops(glcm, prop='energy')[0, 0]
    homogeneity = graycoprops(glcm, prop='homogeneity')[0, 0]

    avg_color = np.mean(image, axis=(0, 1))

    return [avg_color[0], avg_color[1], avg_color[2], contrast, correlation, energy, homogeneity]

# Load the test image
test_image_path = 'D:/VISUAL/DATASET_TELUR/TEST/test7.jpg'
test_image = cv2.imread(test_image_path)
pil_test_image = Image.fromarray(cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB))
removed_bg_test_image = remove(pil_test_image).convert("RGB")
removed_bg_test_image = np.array(removed_bg_test_image)

# Convert the original image to grayscale
gray = cv2.cvtColor(removed_bg_test_image, cv2.COLOR_BGR2GRAY)

# Thresholding to separate object from background
_, thresholded = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Find contours of the objects in the thresholded image
contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

detected_objects = []

for contour in contours:
    # Calculate the area of the contour
    area = cv2.contourArea(contour)

    # Set a minimum area threshold to exclude small objects (adjust as needed)
    min_area_threshold = 1000

    if area > min_area_threshold:
        # Create a mask for the detected object
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [contour], 0, 255, -1)

        # Apply the mask to the grayscale image
        masked_gray = cv2.bitwise_and(gray, gray, mask=mask)

        # Calculate GLCM (Gray-Level Co-occurrence Matrix) for the detected object
        glcm = graycomatrix(masked_gray, [1], [0], symmetric=True, normed=True)

        # Calculate GLCM properties for the detected object
        contrast = graycoprops(glcm, 'contrast')[0, 0]
        correlation = graycoprops(glcm, 'correlation')[0, 0]
        energy = graycoprops(glcm, 'energy')[0, 0]
        homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]

        # Calculate Average R, Average G, and Average B for the detected object
        masked_color = cv2.bitwise_and(removed_bg_test_image, removed_bg_test_image, mask=mask)
        avg_color = np.mean(masked_color, axis=(0, 1))

        # Predict the category of the detected object using the trained model
        detected_object_features = [avg_color[0], avg_color[1], avg_color[2], contrast, correlation, energy, homogeneity]
        predicted_category = model.predict([detected_object_features])

        detected_objects.append({
            'contour': contour,
            'category': predicted_category[0],
            'features': detected_object_features
        })

# Count the number of 'utuh' and 'retak' eggs
count_utuh = sum(1 for obj in detected_objects if obj['category'] == 'utuh')
count_retak = sum(1 for obj in detected_objects if obj['category'] == 'retak')

print(f"Jumlah Telur Utuh: {count_utuh}")
print(f"Jumlah Telur Retak: {count_retak}")

# Draw bounding boxes around detected objects on the original image
for obj in detected_objects:
    (x, y, w, h) = cv2.boundingRect(obj['contour'])
    cv2.rectangle(removed_bg_test_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(removed_bg_test_image, f"{obj['category']}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Display the final processed image with bounding boxes
cv2.imshow('Detected Objects', removed_bg_test_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
