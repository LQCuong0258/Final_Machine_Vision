import numpy as np
import cv2
from PIL import Image
from Grayscale import grayscale_luminance

'''
Hàm này xác định biên dạng của hình ảnh bằng phương pháp Sobel
'''
def Sobel(img_PIL, limit):
    gray = grayscale_luminance(img_PIL) # chuyển ảnh mức xám
    img_array = np.array(gray)

    # Vì 3 kênh xám đều giống nhau, nên chỉ lấy 1 kênh
    R = img_array[:, :, 0]      # get channel red
    X = np.array([[-1, -2, -1],
                  [ 0,  0,  0],
                  [ 1,  2,  1]])
    Y = np.array([[-1,  0,  1],
                  [-2,  0,  2],
                  [-1,  0,  1]])

    Gxr    = np.zeros((img_array.shape[0], img_array.shape[1]))   # (cao, rộng)
    Gyr    = np.zeros((img_array.shape[0], img_array.shape[1]))   # (cao, rộng)

    # padding để output có kích thước bằng input
    padded = np.pad(R, pad_width = 1, mode = 'constant', constant_values = 0)

    height, width = padded.shape # [cao, rộng]
    for i in range (1, height - 1):
        for j in range (1, width - 1):
            Gxr[i-1, j-1] = np.sum(padded[i-1:i+2, j-1:j+2] * X)
            Gyr[i-1, j-1] = np.sum(padded[i-1:i+2, j-1:j+2] * Y)
    M = np.abs(Gxr) + np.abs(Gyr)       # sắp xỉ căn bậc 2 của tổng 2 bình phương
    Sobel = np.where(M < limit, 0, 255).astype(np.uint8)    # trả về 1 layer của ảnh Sobel

    return np.dstack((Sobel, Sobel, Sobel))                 # Trả về 1 ảnh có 3 layer giống nhau



img = Image.open("Lena_color.jpg")
cv2.imshow('Edge Detected Image', Sobel(img, 200))    # Hiển thị hình ảnh được định dạng mảng
cv2.waitKey(0)
cv2.destroyAllWindows()
