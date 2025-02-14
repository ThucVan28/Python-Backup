import cv2
import numpy as np


path_folderL = "./imag/Calib_cameraL"
path_folderR = "./imag/Calib_cameraR"
image_counterL = 0
image_counterR = 0
capL = cv2.VideoCapture(0)
capR = cv2.VideoCapture(1)

if not capL.isOpened():
    print("Không mở được Camera trái")
    exit()
elif not capR.isOpened():
    print('Không mở được camera phải')
    exit()




while(True):
    retL, frameL = capL.read()
    retR, frameR = capR.read()

    if not retL:
        print("Không thể đọc khung hình từ webcam trái")
        break

    elif not retR:
        print("Không thể đọc khung hình từ webcam phải")
        break


    cv2.namedWindow('webcam Left', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('webcam Left', 960, 540)
    cv2.namedWindow('webcam Right', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('webcam Right', 960, 540)

    cv2.imshow('webcam Left', frameL)
    cv2.imshow('webcam Right', frameR)

    #Lưu ảnh webcam Trái
    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
    
        image_nameL = f"{path_folderL}/image_{image_counterL:01d}.jpg"
        image_nameR = f"{path_folderR}/image_{image_counterR:01d}.jpg"
        print(image_nameL)
        cv2.imwrite(image_nameL, frameL)
        cv2.imwrite(image_nameR, frameR)
        image_counterL +=1
        image_counterR +=1
        print('Lưu ảnh ảnh thành công ' + image_nameL)

    # if key == ord('r'):
    #     image_nameR = f"{path_folderR}/image_{image_counterR:03d}.jpg"
    #     cv2.imwrite(image_nameR, frameR)
    #     image_counterR +=1

    #     print('Lưu ảnh ảnh thành công')
            


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capL.release()
capR.release()
cv2.destroyAllWindows()