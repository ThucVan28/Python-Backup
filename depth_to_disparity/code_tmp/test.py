# import cv2
# import numpy as np

# capL = cv2.VideoCapture(0)
# # capR = cv2.VideoCapture(1)

# if not capL.isOpened():
#     print("Không mở được Camera trái")
#     exit()
# # elif not capR.isOpened():
# #     print('Không mở được camera phải')
# #     exit()




# while(True):
#     retL, frameL = capL.read()
#     # retR, frameR = capR.read()

#     if not retL:
#         print("Không thể đọc khung hình từ webcam trái")
#         break

#     # elif not retR:
#     #     print("Không thể đọc khung hình từ webcam phải")
#     #     break
#     print(frameL[1:10, 1:10])

#     if cv2.waitKey(1) == ord('q'):
#         break

# cv2.destroyAllWindows()
# capL.release()

import cv2
# import torch 
# path_L = "./imag/test/image00.jpg"
# path_R = "./imag/test/image01.jpg"
# img_L = cv2.imread(path_L, cv2.IMREAD_GRAYSCALE)
# img_R = cv2.imread(path_R, cv2.IMREAD_GRAYSCALE)
# for i in range(0, 480, 20):

#     y = i  # Ví dụ: chọn một giá trị y trên ảnh trái (hoặc có thể chọn nhiều điểm)

#     # Vẽ đường epipolar trên ảnh trái (hàng ngang)
#     cv2.line(img_L, (0, y), (img_L.shape[1], y), (0, 255, 0), 1)

#     # Vẽ đường epipolar trên ảnh phải (hàng ngang)
#     cv2.line(img_R, (0, y), (img_R.shape[1], y), (0, 255, 0), 1)


# cv2.imshow("Left Image with Epipolar Line", img_L)
# cv2.imshow("Right Image with Epipolar Line", img_R)

# cv2.waitKey(0)
# cv2.destroyAllWindows()

# import torch
# # print(torch.cuda.is_available())  # Kiểm tra GPU có hoạt động không
# # print(torch.cuda.get_device_name(0))  # Xem tên GPU
# print(torch.__version__)

import openimages.download as oid

oid.download(dataset='train', classes=['Box'], limit=10)


