import cv2
import numpy as np
from Grayscale import *
from Edge_detection import Sobel
from PIL import Image
from Binary import Binary
from Smooth import *

def slicing_img(imgPIL, cut):
    img = np.array(imgPIL)
    red_channel     = img[:, :, 0][cut:, :]
    green_channel   = img[:, :, 1][cut:, :]
    blue_channel    = img[:, :, 2][cut:, :]
    return np.stack([red_channel, green_channel, blue_channel], axis=-1)

def classify_shape(contour):
    # Xác định số cạnh của hình dạng
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
    sides = len(approx)

    # Phân loại hình dạng
    if sides == 3:
        return "Triangle"
    elif sides == 4:
        # Kiểm tra xem hình vuông hay hình chữ nhật
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = float(w) / h
        print(aspect_ratio)
        if 0.9 <= aspect_ratio <= 1.02: # lấy chiều rộng chia cho chiều cao
            return "Square"
        else:
            return "Rectangle"
    else:
        return "Circle"

if __name__ == "__main__":
    # image_path = 'E:\\UTE\\Machine_Vision\\Finnal\\samples\\thoi\\005.bmp'
    # image_path = 'E:\\UTE\\Machine_Vision\\Finnal\\samples\\elip\\005.bmp'
    # image_path = 'E:\\UTE\\Machine_Vision\\Finnal\\samples\\Tron\\002.bmp'
    # image_path = 'E:\\UTE\\Machine_Vision\\Finnal\\samples\\tam_giac\\005.bmp'
    # image_path = 'E:\\UTE\\Machine_Vision\\Finnal\\samples\\HCN\\008.bmp'
    image_path = 'E:\\UTE\\Machine_Vision\\Finnal\\samples\\vuong\\005.bmp'
    # image_path = r"Lena_color.jpg"
    # image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    imgPIL = Image.open(image_path)
    img = slicing_img(imgPIL, 60)   # bỏ ik phần dính băng tải

    gray = grayscale_luminance(img)
    smooth = Smoth_Average(gray, 5) # 3x3
    binary = Binary(smooth, 60)     # 60
    sobel = Sobel(binary, 100)

    # giải thích hàm find contours
    contours, _ = cv2.findContours(sobel, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        shape = classify_shape(contour)
        # Vẽ hình dạng phân loại lên ảnh gốc
        cv2.drawContours(img, [contour], -1, (0, 255, 0), 2)
        # Hiển thị loại hình dạng
        M = cv2.moments(contour)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        cv2.putText(img, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)



    cv2.imshow("test", img)

    

    cv2.waitKey(0)
    cv2.destroyAllWindows()

