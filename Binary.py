import cv2
from PIL import Image
import numpy as np
from Grayscale import grayscale_luminance

def Binary(gray, limit):
    img_array = np.array(gray)
    red = img_array[:,:, 0]
    binary = np.where(red < limit, 0, 255).astype(np.uint8)
    return np.dstack((binary, binary, binary))


if __name__ == "__main__":
    # Init string store file "Lena_color"
    path = r"Lena_color.jpg"
    imgPIL = Image.open(path)
    image = cv2.imread(path, cv2.IMREAD_COLOR)


    # Hiển thị ảnh gốc
    cv2.imshow('Anh mau goc RGB co gai Lena', image)
    # Hiển thị ảnh mức xám
    cv2.imshow('Anh muc xam co gai Lena_ binary', Binary(imgPIL, 50))
    # Phím bất kì để đóng cửa sổ làm việc
    cv2.waitKey(0)
    # Giải phóng bộ nhớ được cấp phát
    cv2.destroyAllWindows()