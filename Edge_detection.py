import cv2  #Sử dụng thư viện sử lý ảnh OpenCV
from PIL import Image #Thư viện xử lý ảnh
import numpy as np
from pickletools import uint8
from Grayscale import grayscale_luminance


import numpy as np
from PIL import Image

def Sobel(img_gray, k):
    Sobelx = np.array([[-1, -2, -1],
                       [ 0,  0,  0],
                       [ 1,  2,  1]])

    Sobely = np.array([[-1,  0,  1],
                       [-2,  0,  2],
                       [-1,  0,  1]])

    # Convert PIL image to NumPy array
    img_array = np.array(img_gray)

    # Apply Sobel filters to the image using manual convolution
    Grx = manual_convolution(img_array, Sobelx)
    Gry = manual_convolution(img_array, Sobely)

    # Compute magnitude of gradient
    M = np.abs(Grx) + np.abs(Gry)

    # Thresholding
    Sobel = np.where(M < k, 0, 255).astype(np.uint8)

    return Image.fromarray(Sobel)

def manual_convolution(image, kernel):
    # Get image dimensions and kernel dimensions
    image_width, image_height = image.size
    kernel_height, kernel_width = kernel.shape

    # Calculate padding needed for 'same' convolution
    pad_height = kernel_height // 2
    pad_width = kernel_width // 2

    # Add zero padding to the image
    padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='edge')

    # Convolve the image with the kernel
    convolved_image = np.zeros_like(image)
    for i in range(image_height):
        for j in range(image_width):
            convolved_image[i, j] = np.sum(padded_image[i:i+kernel_height, j:j+kernel_width] * kernel)

    return convolved_image

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
    imgPIL = Image.open(path)
    gray = grayscale_luminance(imgPIL)

    image = cv2.imread(path, cv2.IMREAD_COLOR)
    cv2.imshow('Anh mau RGB', image)
    cv2.imshow('Anh duoc nhan dang duong bien Sobel', Sobel(gray, 130))

    
    cv2.waitKey(0)          # Bấm phím bất kì để đóng cửa sổ hiển thị hình
    cv2.destroyAllWindows() # Giải phóng bộ nhớ đã cấp phát cho các cửa sổ hiển thị hình


if __name__ == "__main__":
    main()

