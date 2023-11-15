import cv2 as cv
import numpy as np

# Nama file
filename = 'Rambujalan.jpg'

# Load gambar
src = cv.imread(filename, cv.IMREAD_COLOR)

# Periksa apakah file gambar berhasil dibaca
if src is not None:
    # Resize ke 30% ukuran aslinya
    scale_percent = 30
    width = int(src.shape[1] * scale_percent / 100)
    height = int(src.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv.resize(src, dim, interpolation=cv.INTER_AREA)

    # Ubah ke grayscale
    gray = cv.cvtColor(resized, cv.COLOR_BGR2GRAY)

    # Aplikasikan filter Gaussian blur untuk mengurangi noise
    gray = cv.GaussianBlur(gray, (9, 9), 2)

    # Aplikasikan hough circle
    circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, dp=1, minDist=20, param1=100, param2=30, minRadius=50, maxRadius=70)

    # Jika lingkaran ditemukan
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            radius = i[2]
            # circle center
            cv.circle(resized, center, 1, (0, 100, 100), 3)
            # circle outline
            cv.circle(resized, center, radius, (0, 234, 255), 3)

        cv.imshow("Deteksi Lingkaran", resized)
        cv.waitKey(0)
        cv.destroyAllWindows()