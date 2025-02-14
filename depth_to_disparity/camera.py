import cv2
import numpy as np
import os
import glob

CHECKERBOARD = (6,8)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
 
# Creating vector to store vectors of 3D points for each checkerboard image
objpoints = []
# Creating vector to store vectors of 2D points for each checkerboard image
imgpoints = [] 
 
 
# Defining the world coordinates for 3D points
objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
prev_img_shape = None

img = cv2.imread('./imag/im7.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# Find the chess board corners
 # If desired number of corners are found in the image then ret = true
ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
     

    
if ret == True:
    corners2 = cv2.cornerSubPix(gray, corners, (11,11),(-1,-1), criteria)
         
    imgpoints.append(corners2)
 
    # Draw and display the corners
    img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)

cv2.namedWindow('img', cv2.WINDOW_NORMAL)  # Tạo cửa sổ hiển thị có thể thay đổi kích thước
img_resized = cv2.resize(img, (500, 500)) # Thay đổi kích thước hình ảnh
cv2.imshow('img', img_resized)
cv2.waitKey(0)
cv2.destroyAllWindows()