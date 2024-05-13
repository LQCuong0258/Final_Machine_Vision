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

if __name__ == "__main__":
    image_path = 'E:\\UTE\\Machine_Vision\\Finnal\\samples\\thoi\\005.bmp'
    # image_path = 'E:\\UTE\\Machine_Vision\\Finnal\\samples\\elip\\005.bmp'
    # image_path = 'E:\\UTE\\Machine_Vision\\Finnal\\samples\\Tron\\005.bmp'
    # image_path = 'E:\\UTE\\Machine_Vision\\Finnal\\samples\\tam_giac\\005.bmp'
    # image_path = 'E:\\UTE\\Machine_Vision\\Finnal\\samples\\HCN\\005.bmp'
    # image_path = 'E:\\UTE\\Machine_Vision\\Finnal\\samples\\vuong\\005.bmp'
    # image_path = r"Lena_color.jpg"
    # image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    imgPIL = Image.open(image_path)
    img = slicing_img(imgPIL, 60)   # bỏ ik phần dính băng tải

    gray = grayscale_luminance(img)
    smooth = Smoth_Average(gray, 3) # 3x3
    binary = Binary(smooth, 60)     # 60
    sobel = Sobel(binary, 100)


    cv2.imshow("test", sobel)

    

    cv2.waitKey(0)
    cv2.destroyAllWindows()

