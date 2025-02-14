import cv2
import numpy as np
import os

#Read parameter calib
file_path = "improved_params2.xml"

if not os.path.exists(file_path):
    print("Không tìm thấy file")
    exit()

fs = cv2.FileStorage(file_path, cv2.FILE_STORAGE_READ)

if not fs.isOpened():
    print("Không mở được file XML")
    exit()

left_parameter_x = fs.getNode("Left_Stereo_Map_x").mat()
left_parameter_y = fs.getNode("Left_Stereo_Map_y").mat()

right_parameter_x = fs.getNode("Right_Stereo_Map_x").mat()
right_parameter_y = fs.getNode("Right_Stereo_Map_y").mat()
fs.release()

if left_parameter_x is None or left_parameter_y is None:
    print("Không đọc được nội dung file XML")
    exit()
    

# print("left_parameter_x: \n", left_parameter_x)
# print("left_parameter_y: \n", left_parameter_y)
# print("right_parameter_x: \n", right_parameter_x)
# print("right_parameter_y: \n", right_parameter_y)

# path_L = "./imag/data_stereo_image/im00.png"
# path_R = "./imag/data_stereo_image/im11.png"
# img_L = cv2.imread(path_L, cv2.IMREAD_GRAYSCALE)

# width = int(img_L.shape[1] * 0.5)
# height = int(img_L.shape[0] * 0.5)
# dim = (width, height)
# img_L = cv2.resize(img_L, dim, interpolation=cv2.INTER_AREA)
# img_R = cv2.imread(path_R, cv2.IMREAD_GRAYSCALE)
# img_R = cv2.resize(img_R, dim, interpolation=cv2.INTER_AREA)

path_L = "./imag/test/image10.jpg"
path_R = "./imag/test/image11.jpg"
img_L = cv2.imread(path_L)
img_R = cv2.imread(path_R)

# h = 10
# hForColorComponents = 10
# templateWindowSize = 7
# searchWindowSize = 21

# img_L = cv2.fastNlMeansDenoisingColored(img_L, None, h, hForColorComponents, templateWindowSize, searchWindowSize)  
# img_R = cv2.fastNlMeansDenoisingColored(img_R, None, h, hForColorComponents, templateWindowSize, searchWindowSize) 

img_L = cv2.cvtColor(img_L, cv2.COLOR_BGR2GRAY)
img_R = cv2.cvtColor(img_R, cv2.COLOR_BGR2GRAY)



cv2.imshow('filter_medianL', img_L)
cv2.imshow('filter_medianR', img_R)



if img_L is None and img_R is None:
    print("Không đọc được ảnh")
    exit()

def nothing(x):
    pass

cv2.namedWindow("disp", cv2.WINDOW_NORMAL)
cv2.resizeWindow("disp", 600, 600)
# cv2.waitKey(1)  # Đợi 1ms để đảm bảo cửa sổ hiển thị trước khi tạo trackbar


# # #StereoBM
# window_size = 5
# min_disp = 2
# num_disp = 130-min_disp
# stereo = cv2.StereoBM_create(numDisparities=num_disp, blockSize=window_size)
#                             # preFilterType = 1,
#                             # preFilterSize = 2,
#                             # preFilterCap = 5,
#                             # textureThreshold = 10,
#                             # uniquenessRatio = 15,
#                             # speckleRange = 0,
#                             # disp12MaxDiff = 5,
#                             # minDisparity = min_disp)

# cv2.createTrackbar('numDisparities', 'disp', 8, 17, nothing)
# cv2.createTrackbar('blockSize', 'disp', 5,50, nothing)
# cv2.createTrackbar('preFilterType', 'disp', 1, 1, nothing)
# cv2.createTrackbar('preFilterSize', 'disp', 3, 25, nothing)
# cv2.createTrackbar('preFilterCap', 'disp', 30, 62, nothing)
# cv2.createTrackbar('textureThreshold', 'disp', 0, 100, nothing)
# cv2.createTrackbar('uniquenessRatio', 'disp' ,2, 100, nothing)
# cv2.createTrackbar('speckleRange', 'disp', 9, 100, nothing)
# cv2.createTrackbar('speckleWindowSize', 'disp', 0, 25, nothing)
# cv2.createTrackbar('disp12MaxDiff', 'disp', 2, 25, nothing)
# cv2.createTrackbar('minDisparity', 'disp', 2, 25, nothing)
# cv2.createTrackbar('size_kernel', 'disp', 5, 20, nothing)
# # stereo = cv2.StereoBM_create()


# StereoSGBM
window_size = 3
min_disp = 2
num_disp = 130-min_disp
stereo = cv2.StereoSGBM_create(minDisparity = min_disp,
    numDisparities = num_disp,
    blockSize = window_size,
    uniquenessRatio = 10,
    speckleWindowSize = 100,
    speckleRange = 32,
    disp12MaxDiff = 5,
    P1 = 8*3*window_size**2,
    P2 = 32*3*window_size**2)

# stereo = cv2.StereoSGBM_create()
cv2.createTrackbar('numDisparities', 'disp', 8, 17, nothing)
cv2.createTrackbar('blockSize', 'disp', 2,50, nothing)
cv2.createTrackbar('uniquenessRatio', 'disp', 0,100, nothing)
cv2.createTrackbar('speckleWindowSize', 'disp', 50, 50, nothing)
cv2.createTrackbar('speckleRange', 'disp', 32, 100, nothing)
cv2.createTrackbar('disp12MaxDiff', 'disp', 5, 100, nothing)
cv2.createTrackbar('P1', 'disp', 8, 30, nothing)
cv2.createTrackbar('P2', 'disp', 32, 60, nothing)
cv2.createTrackbar('minDisparity', 'disp', 2, 25, nothing)
cv2.createTrackbar('size_kernel', 'disp', 5, 20, nothing)
cv2.createTrackbar('sigma', 'disp', 15, 20, nothing)

# Used for the filtered image
stereoR=cv2.ximgproc.createRightMatcher(stereo) # Create another stereo for right this time

# WLS FILTER Parameters
lmbda = 80000
sigma = 2
visual_multiplier = 1.0
 
wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=stereo)
wls_filter.setLambda(lmbda)
wls_filter.setSigmaColor(sigma)




# path_L = "./imag/data_stereo_image/im00.png"
# path_R = "./imag/data_stereo_image/im11.png"
# img_L = cv2.imread(path_L, cv2.IMREAD_GRAYSCALE)
# img_R = cv2.imread(path_R, cv2.IMREAD_GRAYSCALE) 


while True:
    
    # #StereoBM
    # numDisparitiy = cv2.getTrackbarPos('numDisparities','disp')*16
    # blockSize = cv2.getTrackbarPos('blockSize','disp')*2 + 5
    # preFilterType = cv2.getTrackbarPos('preFilterType','disp')
    # preFilterSize = cv2.getTrackbarPos('preFilterSize','disp')*2 + 5
    # preFilterCap = cv2.getTrackbarPos('preFilterCap','disp')
    # textureThreshold = cv2.getTrackbarPos('textureThreshold','disp')
    # uniquenessRatio = cv2.getTrackbarPos('uniquenessRatio','disp')
    # speckleRange = cv2.getTrackbarPos('speckleRange','disp')
    # speckleWindowSize = cv2.getTrackbarPos('speckleWindowSize','disp')*2
    # disp12MaxDiff = cv2.getTrackbarPos('disp12MaxDiff','disp')
    # minDisparity = cv2.getTrackbarPos('minDisparity','disp')

    # stereo.setNumDisparities(numDisparitiy)
    # stereo.setBlockSize(blockSize)
    # stereo.setPreFilterType(preFilterType)
    # stereo.setPreFilterSize(preFilterSize)
    # stereo.setPreFilterCap(preFilterCap)
    # stereo.setTextureThreshold(textureThreshold)
    # stereo.setUniquenessRatio(uniquenessRatio)
    # stereo.setSpeckleRange(speckleRange)
    # stereo.setSpeckleWindowSize(speckleWindowSize)
    # stereo.setDisp12MaxDiff(disp12MaxDiff)
    # stereo.setMinDisparity(minDisparity)

    #StereoSGBM
    numDisparitiy = cv2.getTrackbarPos('numDisparities','disp')*16
    blockSize = cv2.getTrackbarPos('blockSize','disp')*2 + 3
    uniquenessRatio = cv2.getTrackbarPos('uniquenessRatio','disp')
    speckleRange = cv2.getTrackbarPos('speckleRange','disp')
    speckleWindowSize = cv2.getTrackbarPos('speckleWindowSize','disp')*2
    disp12MaxDiff = cv2.getTrackbarPos('disp12MaxDiff','disp')
    minDisparity = cv2.getTrackbarPos('minDisparity','disp')
    P1 = cv2.getTrackbarPos('P1','disp')*3*blockSize**2
    P2 = cv2.getTrackbarPos('P2', 'disp')*3*blockSize**2

    stereo.setNumDisparities(numDisparitiy)
    stereo.setBlockSize(blockSize)
    stereo.setUniquenessRatio(uniquenessRatio)
    stereo.setSpeckleRange(speckleRange)
    stereo.setSpeckleWindowSize(speckleWindowSize)
    stereo.setDisp12MaxDiff(disp12MaxDiff)
    stereo.setMinDisparity(minDisparity)
    stereo.setP1(P1)
    stereo.setP2(P2)
    
    # #parameter WLS filter
    # lmbda = cv2.getTrackbarPos('lmbda','disp')*10000
    # sigma = float(cv2.getTrackbarPos('sigma','disp'))/10.0
    # wls_filter.setLambda(lmbda)
    # wls_filter.setSigmaColor(sigma)

    size_kernel = cv2.getTrackbarPos('size_kernel', 'disp')

    # disparity = stereo.compute(img_L,img_R)

    # Compute the 2 images for the Depth_image
    disparity= stereo.compute(img_L,img_R)
    dispL= disparity
    dispR= stereoR.compute(img_R,img_L)

    dispL1 = dispL
    dispR1 = dispR
    dispL1 = dispL1.astype(np.float32)
    dispR1 = dispR1.astype(np.float32)

    dispL= np.int16(dispL)
    dispR= np.int16(dispR)
    kernel= np.ones((size_kernel,size_kernel),np.uint8)

    dispL= cv2.morphologyEx(dispL,cv2.MORPH_CLOSE, kernel)
    dispR= cv2.morphologyEx(dispR,cv2.MORPH_CLOSE, kernel)

    # Using the WLS filter
    filteredImg= wls_filter.filter(dispL,img_L,None,dispR)
    filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)
    filteredImg = np.uint8(filteredImg)
    cv2.namedWindow("WLS_filter", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("WLS_filter", 640, 480)
    cv2.imshow("WLS_filter",filteredImg)

    # Converting to float32 
    disparity = disparity.astype(np.float32)
    
    # Scaling down the disparity values and normalizing them 
    disparity = (disparity/16.0 - minDisparity)/numDisparitiy

    # Filtering the Results with a closing filter
    # Filtering
    disparity= cv2.morphologyEx(disparity,cv2.MORPH_CLOSE, kernel)
    
    # Displaying the disparity map
    cv2.namedWindow("disp_left", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("disp_left", 640, 480)
    cv2.imshow("disp_left",disparity)
    
    # Close window using esc key
    if cv2.waitKey(1) == 27:
        break

print("Kích thước ảnh gốc:", img_L.shape)
print("Kích thước disparity:", disparity.shape)
print("Kích thước filteredImg:", filteredImg.shape)
# print("min: {}    ; max: {}".format(dispR1.min(), dispR1.max()))
# print(dispL1[1:10, 1:10])

cv2.destroyAllWindows()


