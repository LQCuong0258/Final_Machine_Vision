import cv2  #Sử dụng thư viện sử lý ảnh OpenCV
from PIL import Image #Thư viện xử lý ảnh
import numpy as np
from pickletools import uint8
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

def Prewitt(img_gray, k):
    Prewitt=Image.new(img_gray.mode, img_gray.size)
    w=Prewitt.size[0]
    h=Prewitt.size[1]
    Prewittx = np.array([[-1,-1,-1],
                         [ 0, 0, 0],
                         [ 1, 1, 1]])
    Prewitty = np.array([[-1,0,1],
                         [-1,0,1],
                         [-1,0,1]])
    
    for x in range(1, w-1):
        for y in range(1, h-1):
            Grx,Gry,Mr = 0,0,0
            for i in range (x-1, x+2):
                for j in range (y-1, y+2):
                    R,G,B = img_gray.getpixel((i, j))
                    Grx += np.vdot(R,Prewittx[j - y + 1,i - x + 1])
                    Gry += np.vdot(R,Prewitty[j - y + 1,i - x + 1])
            Mr = np.abs(Grx) + np.abs(Gry)
            if Mr < k:
                Prewitt.putpixel((x,y), (0, 0, 0))
            else:
                Prewitt.putpixel((x,y), (255, 255, 255))
            
    Prewitt = np.array(Prewitt)
    return Prewitt

def Robert(img_gray,k):
    Robert=Image.new(img_gray.mode, img_gray.size)
    w=Robert.size[0]
    h=Robert.size[1]
    Robertx = np.array([[-1,0],
                        [ 0,1]])
    
    Roberty = np.array([[0,-1],
                        [1, 0]])
    
    for x in range(w-1):
        for y in range(h-1):
            Grx,Gry,Mr = 0,0,0
            for i in range (x, x+2):
                for j in range (y, y+2):
                    R,G,B = img_gray.getpixel((i,j))
                    Grx += np.vdot(R, Robertx[j - y, i - x])
                    Gry += np.vdot(R, Roberty[j - y, i - x])
            Mr = np.abs(Grx) + np.abs(Gry)
            if Mr < k:
                Robert.putpixel((x,y), (0,0,0))
            else:
                Robert.putpixel((x,y), (255, 255, 255))
            
    Robert = np.array(Robert)
    return Robert

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

