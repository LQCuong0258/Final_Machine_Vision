import cv2
from PIL import Image
from Edge_detection import Sobel

def main():
    path = r'Lena_color.jpg'

    image = cv2.imread(path, cv2.IMREAD_COLOR)
    cv2.imshow('Anh mau RGB', image)
    
    imgPIL = Image.open(path)
    cv2.imshow('Anh duoc nhan dang duong bien Sobel', Sobel(imgPIL, 130))

    
    cv2.waitKey(0)          # Bấm phím bất kì để đóng cửa sổ hiển thị hình
    cv2.destroyAllWindows() # Giải phóng bộ nhớ đã cấp phát cho các cửa sổ hiển thị hình





if __name__ == "__main__":
    main()