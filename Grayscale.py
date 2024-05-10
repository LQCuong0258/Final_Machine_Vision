import cv2
from PIL import Image
import numpy as np

def grayscale_average(imgPIL):
    # Init image as size and more of imgPIL
    # its to use store Grayscale of image RGB
    average = Image.new(imgPIL.mode, imgPIL.size)

    # Get size of imgPIL
    width, height = average.size

    # Scan image bằng 2 vòng for and scan từ trái sang phải cho từng dòng
    for x in range(width):
        for y in range(height):
            # Get pixel and 3 thông số về R, G, B của ảnh màu
            R, G, B = imgPIL.getpixel((x, y))

            # Tính toán giá trị mức xám theo phương pháp Average
            gray = np.uint8((R + G + B)/3)

            # Gán giá trị mức xám vừa tính được
            # cho biến hiển thị mức xám
            average.putpixel((x, y), (gray, gray, gray))
    return average
def grayscale_lightness(imgPIL):
    # Init image as size and more of imgPIL
    # its to use store Grayscale of image RGB
    lightness = Image.new(imgPIL.mode, imgPIL.size)

    # Get size of imgPIL
    width, height = lightness.size

    # Scan image bằng 2 vòng for and scan từ trái sang phải cho từng dòng
    for x in range(width):
        for y in range(height):
            # Get pixel and 3 thông số về R, G, B của ảnh màu
            R, G, B = imgPIL.getpixel((x, y))

            # Find minimum and maximum
            MIN = min(R, G, B)
            MAX = max(R, G, B)
            # Công thức Lightness
            gray = np.uint8((MIN + MAX)/2)

            # Gán giá trị mức xám vừa tính được
            # cho biến hiển thị mức xám
            lightness.putpixel((x, y), (gray, gray, gray))
    return lightness
def grayscale_luminance(imgPIL):
    # Init image as size and more of imgPIL
    # its to use store Grayscale of image RGB
    luminance = Image.new(imgPIL.mode, imgPIL.size)

    # Get size of imgPIL
    width, height = luminance.size

    # Scan image bằng 2 vòng for and scan từ trái sang phải cho từng dòng
    for x in range(width):
        for y in range(height):
            # Get pixel and 3 thông số về R, G, B của ảnh màu
            R, G, B = imgPIL.getpixel((x, y))

            # Tính toán mức xám theo công thức Luminance
            gray = np.uint8(0.2126*R + 0.7152*G + 0.0722*B)

            # Gán giá trị mức xám vừa tính được
            # cho biến hiển thị mức xám
            luminance.putpixel((x, y), (gray, gray, gray))
    return luminance


def main():
    # Init link of image
    file_image = r'Lena_color.jpg'

    # Use library OpenCV to read image used show
    # Read by mode Color
    image = cv2.imread(file_image, cv2.IMREAD_COLOR)

    # Use library PILLOW to read image used caculator
    imgPIL = Image.open(file_image)
    # Change from image PIL to OpenCV, to show by OpenCV
    average = np.array(grayscale_average(imgPIL))
    lightness = np.array(grayscale_lightness(imgPIL))
    luminance = np.array(grayscale_luminance(imgPIL))
    # Hiển thị ảnh gốc bằng thư viện cv2
    cv2.imshow('Anh mau goc RGB co gai Lena', image)
    # Hiển thị ảnh mức xám
    cv2.imshow('Anh muc xam phuong phap Average co gai Lena_Average', average)
    cv2.imshow('Anh muc xam phuong phap Lightness co gai Lena_Average', lightness)
    cv2.imshow('Anh muc xam phuong phap Luminance co gai Lena_Average', luminance)
    # Phím bất kì để đóng cửa sổ làm việc
    cv2.waitKey(0)
    # Giải phóng bộ nhớ đã được cấp phát
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
