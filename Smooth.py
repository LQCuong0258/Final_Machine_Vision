import cv2  #Sử dụng thư viện sử lý ảnh OpenCV
from PIL import Image #Thư viện xử lý ảnh 
import numpy as np

def Smoth_Average(imgPIL, K):
    img_array = np.array(imgPIL)
    R = img_array[:, :, 0]      # get channel red
    G = img_array[:, :, 1]      # get channel red
    B = img_array[:, :, 2]      # get channel red

    pad_R = np.pad(R, pad_width = K//2, mode = 'constant', constant_values = 0)
    pad_G = np.pad(G, pad_width = K//2, mode = 'constant', constant_values = 0)
    pad_B = np.pad(B, pad_width = K//2, mode = 'constant', constant_values = 0)

    height, width = pad_R.shape

    red     = np.zeros((img_array.shape[0], img_array.shape[1]))   # (cao, rộng)
    green   = np.zeros((img_array.shape[0], img_array.shape[1]))   # (cao, rộng)
    blue    = np.zeros((img_array.shape[0], img_array.shape[1]))   # (cao, rộng)

    s = K//2

    for x in range(s, height - s):
        for y in range(s, width - s):
            red[x-s, y-s]     = (np.sum(pad_R[x-s:x+s+1, y-s:y+s+1])/(K*K)).astype(np.uint8)
            green[x-s, y-s]   = (np.sum(pad_G[x-s:x+s+1, y-s:y+s+1])/(K*K)).astype(np.uint8)
            blue[x-s, y-s]    = (np.sum(pad_B[x-s:x+s+1, y-s:y+s+1])/(K*K)).astype(np.uint8)

    return np.dstack((blue, green, red)).astype(np.uint8)

if __name__ == '__main__':
    #Khai báo đường dẫn file hình
    path_img = r'Lena_color.jpg'
    #Đọc ảnh màu dùng thư viện PIL
    imgPIL = Image.open(path_img)
    cv2.imshow('Anh mat na lam muot 9', Smoth_Average(imgPIL, 9))

    cv2.waitKey(0)
    cv2.destroyAllWindows()
