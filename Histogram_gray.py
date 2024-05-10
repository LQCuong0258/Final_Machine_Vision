import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from Grayscale import grayscale_luminance

'''
Hàm này tạo 1 list chứa các giá trị
để vẽ đồ thị Histogram cho ảnh xám
'''
def gray_histogram(grayscale):
    his = np.zeros(256)             # Tạo list 1 chiều 256 giá trị 0
    width, height = grayscale.size  # Trích xuất chiều rộng và cao của ảnh

    # Scan từng pixel của ảnh
    for x in range(width):
        for y in range(height):
            gR, gG, gB = grayscale.getpixel((x, y))     # Trích xuất giá trị màu tại mỗi pixel
            his[gR] += 1
    return his

def draw_grayHistogram(his):
    plt.plot(his, color = 'orange')
    plt.xlabel('Intensity')
    plt.ylabel('Frequency')
    plt.title('Grayscale Histogram')

def main():
    path = r"Lena_color.jpg"
    imgPIL = Image.open(path)                   # Đọc ảnh bằng thư viện Pillow và gán vào biến imgPIL
    grayscale = grayscale_luminance(imgPIL)     # Đổi sang ảnh mức xám
    grayHistogram = gray_histogram(grayscale)   # Mảng 1 chiều chứ các giá trị từ ảnh mức xám để vẽ biểu đồ Histogram

    draw_grayHistogram(grayHistogram)
    
    plt.show()          # Hiển thị Figure lên màn hình
    plt.close()         # Đóng Figure và giải phóng bộ nhớ

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
