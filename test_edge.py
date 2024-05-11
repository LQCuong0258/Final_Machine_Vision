import numpy as np
import cv2
from PIL import Image
from Grayscale import grayscale_luminance

#---------------------------
# Hàm chuyển đổi ảnh sang ảnh xám

limit = 130
img = Image.open("Lena_color.jpg")
gray = grayscale_luminance(img)
img_array = np.array(gray)

R = img_array[:, :, 0]

X = np.array([[-1, -2, -1],
              [ 0,  0,  0],
              [ 1,  2,  1]])

Y = np.array([[-1,  0,  1],
              [-2,  0,  2],
              [-1,  0,  1]])

Gxr    = np.zeros((512, 512))   # (cao, rộng)
Gyr    = np.zeros((512, 512))   # (cao, rộng)

paddedr = np.pad(R, pad_width = 1, mode = 'constant', constant_values = 0)

height, width = paddedr.shape # [cao, rộng]

for i in range (1, height - 1):
    for j in range (1, width - 1):
        Rx = np.sum(paddedr[i-1:i+2, j-1:j+2] * X)
        Ry = np.sum(paddedr[i-1:i+2, j-1:j+2] * Y)

        Gxr[i-1, j-1] = Rx
        Gyr[i-1, j-1] = Ry

M = np.abs(Gxr) + np.abs(Gyr)
Sobel = np.where(M < limit, 0, 255).astype(np.uint8)



cv2.imshow('Edge Detected Image', Sobel)    # Hiển thị hình ảnh được định dạng mảng
cv2.waitKey(0)
cv2.destroyAllWindows()
