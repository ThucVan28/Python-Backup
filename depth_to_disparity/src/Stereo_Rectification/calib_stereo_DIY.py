import cv2
import numpy as np
import os
import glob
from tqdm import tqdm
import time
import matplotlib.pyplot as plt



def calib_camera_stereo():
    pathL = "./imag/Calib_cameraL/"
    pathR = "./imag/Calib_cameraR/"


    #Kích thước bàn cờ
    size_chessboard = (6,9)

    #Tieu chuan epsilon (độ chính xác) và số lần lặp tối đa (maximum iterations)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)

    #Biến chứa tọa độ
    objpoints = []
    imgpointL = []
    imgpointR = []

    obj3d = np.zeros((1, size_chessboard[0]*size_chessboard[1], 3), np.float32)
    obj3d[0,:,:2] = np.mgrid[0:size_chessboard[0], 0:size_chessboard[1]].T.reshape(-1,2)

    for i in tqdm(range(0,23)):

        imagL = cv2.imread(pathL + "image_%d.jpg"%i)
        print(pathL + "image_%d.jpg"%i)
        imagR = cv2.imread(pathR + 'image_%d.jpg'%i)

        imagL_gray = cv2.imread(pathL + "image_%d.jpg"%i, cv2.IMREAD_GRAYSCALE)
        imagR_gray = cv2.imread(pathR + "image_%d.jpg"%i, cv2.IMREAD_GRAYSCALE)

        if imagL is None:
            print("Không đọc được ảnh dataL")

        if imagR is None:
            print("Không đọc được ảnh dataR")

        outputL = imagL.copy()
        outputR = imagR.copy()

        retL, cornersL = cv2.findChessboardCorners(imagL_gray, size_chessboard, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
        retR, cornersR = cv2.findChessboardCorners(imagR_gray, size_chessboard, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

        if retL and retR:
            #THêm tọa độ 3d vào danh sách
            objpoints.append(obj3d)

            corners2L = cv2.cornerSubPix(imagL_gray, cornersL, (11,11), (-1,-1), criteria)
            corners2R = cv2.cornerSubPix(imagR_gray, cornersR, (11,11), (-1,-1), criteria)
            
            #lưu tọa độ góc vào danh sách
            imgpointL.append(corners2L)
            imgpointR.append(corners2R)

            cv2.drawChessboardCorners(outputL, size_chessboard, corners2L, retL)
            cv2.drawChessboardCorners(outputR, size_chessboard, corners2R, retR)
            
            cv2.namedWindow("imgL_corner", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("imgL_corner", 960, 540)
            cv2.namedWindow("imgR_corner", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("imgR_corner", 960, 540)

            cv2.imshow('imgL_corner', outputL)
            cv2.imshow('imgR_corner', outputR)
            cv2.waitKey(0)



    #calib left camera
    hL,wL = imagL_gray.shape[:2]
    retL, mtxL, distL, rvecsL, tvecsL = cv2.calibrateCamera(objpoints,imgpointL,imagL_gray.shape[::-1],None,None)
    new_mtxL, roiL= cv2.getOptimalNewCameraMatrix(mtxL,distL,(wL,hL),1,(wL,hL))


    print("calib left camera")
    print('h,w = ', imagL_gray.shape[:2])
    print("Camera matrix : \n")
    print(mtxL)
    print("dist : \n")
    print(distL)
    print('\n')

    #calib right camera
    hR,wR = imagR_gray.shape[:2]

    retR, mtxR, distR, rvecsR, tvecsR = cv2.calibrateCamera(objpoints,imgpointR,imagR_gray.shape[::-1],None,None)
    new_mtxR, roiR= cv2.getOptimalNewCameraMatrix(mtxR,distR,(wR,hR),1,(wR,hR))


    print("calib right camera")
    print('h,w = ', imagR_gray.shape[:2])
    print("Camera matrix : \n")
    print(mtxR)
    print("dist : \n")
    print(distR)
    print('\n')

    ######################################################################
    # mapxL,mapyL=cv2.initUndistortRectifyMap(mtxL,distL,None,new_mtxL,(hL,wL ),5)
    imgL = cv2.imread('./imag/Calib_cameraL/image_0.jpg')
    # dstL = cv2.remap(imgL,mapxL,mapyL,cv2.INTER_LINEAR)
    dstL = cv2.undistort(imgL, mtxL, distL, None, new_mtxL)


    xL, yL, wL, hL = roiL
    dstL = dstL[yL:yL+hL, xL:xL+wL]

    # mapxR,mapyR=cv2.initUndistortRectifyMap(mtxR,distR,None,new_mtxR,(hR,wR ),5)
    # imgR = cv2.imread('./imag/Calib_cameraR/image_0.jpg')
    # dstR = cv2.remap(imgR,mapxR,mapyR,cv2.INTER_LINEAR)


    imgR = cv2.imread('./imag/Calib_cameraR/image_0.jpg')
    dstR = cv2.undistort(imgR, mtxR, distR, None, new_mtxR)

    xR, yR, wR, hR = roiR
    dstR = dstR[yR:yR+hR, xR:xR+wR]


    cv2.namedWindow("undistorted imageL", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("undistorted imageL", 960, 540)
    cv2.imshow("undistorted imageL",dstL)

    cv2.namedWindow("undistorted imageR", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("undistorted imageR", 960, 540)
    cv2.imshow("undistorted imageR",dstR)



    cv2.waitKey(0)
    cv2.destroyAllWindows()

    #--------------------------------------------------
    #Tìm góc xoay, dịch chuyển giữa hai camera, ma trận Essential, Fundamental


    # flags = cv2.CALIB_FIX_INTRINSIC  #Cố định ma trận nội tại và hệ số biến dạng ống kính

    criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)

    #Tính toán góc xoay, dịch chuyển giữa hai camera, ma trận Essential, Fundamental
    retS, new_mtxL, distL, new_mtxR, distR, Rot, Trns, Emat, Fmat = cv2.stereoCalibrate(objpoints, imgpointL, imgpointR, new_mtxL, distL, new_mtxR, distR, imagL_gray.shape[::-1], criteria_stereo, flags = cv2.CALIB_USE_INTRINSIC_GUESS )
    print("Rotation Matrix (R):\n", Rot)
    print("Translation Vector (T):\n", Trns)

    if np.allclose(Rot, np.eye(3), atol=0.01):
        print("Cameras are already parallel")
    else:
        print("Cameras are not parallel, rectification needed")

    #------------------------------------------------------
    rectify_scale= 1
    rect_l, rect_r, proj_mat_l, proj_mat_r, Q, roiL, roiR= cv2.stereoRectify(new_mtxL, distL, new_mtxR, distR, imagL_gray.shape[::-1], Rot, Trns)


    print("Rectified Rotation Matrix for Left Camera:\n", rect_l)
    print("Rectified Rotation Matrix for Right Camera:\n", rect_r)


    #--------------------------------------------------------
    Left_Stereo_Map= cv2.initUndistortRectifyMap(new_mtxL, distL, rect_l, proj_mat_l,
                                                 imagL_gray.shape[::-1], cv2.CV_16SC2)
    Right_Stereo_Map= cv2.initUndistortRectifyMap(new_mtxR, distR, rect_r, proj_mat_r,
                                                  imagR_gray.shape[::-1], cv2.CV_16SC2)


    print("parameter: \n")
    print("Left_Stereo_Map", Left_Stereo_Map)
    print("\n")
    print("Right_Stereo_Map", Right_Stereo_Map)
    print("Saving paraeters ......")
    cv_file = cv2.FileStorage("improved_params2.xml", cv2.FILE_STORAGE_WRITE)
    cv_file.write("Left_Stereo_Map_x",Left_Stereo_Map[0])
    cv_file.write("Left_Stereo_Map_y",Left_Stereo_Map[1])
    cv_file.write("Right_Stereo_Map_x",Right_Stereo_Map[0])
    cv_file.write("Right_Stereo_Map_y",Right_Stereo_Map[1])
    cv_file.release()

    # Kiểm tra calib
    imgL = cv2.imread(pathL + "image_0.jpg")
    imgR = cv2.imread(pathR + "image_0.jpg")

    # dst = cv2.undistort(imgL, mtxL, distL, None, new_mtxL)
    # cv2.namedWindow("undistorted image", cv2.WINDOW_NORMAL)
    # cv2.resizeWindow("undistorted image", 960, 540)
    # cv2.imshow("undistorted image",dst)
    # cv2.waitKey(0)


    undistorted_rectified_L = cv2.remap(imgL, Left_Stereo_Map[0], Left_Stereo_Map[1], cv2.INTER_LINEAR)
    undistorted_rectified_R = cv2.remap(imgR, Right_Stereo_Map[0], Right_Stereo_Map[1], cv2.INTER_LINEAR)

    print("Original left image shape:", imagL.shape)
    print("Rectified left image shape:", undistorted_rectified_L.shape)

    x_L, y_L, w_L, h_L = roiR
    x_R, y_R, w_R, h_R = roiR

    print("roiL: ", roiL)
    print("roiR: ", roiR)


    undistorted_rectified_L = undistorted_rectified_L[y_L:y_L+h_L, x_L:x_L+w_L]
    undistorted_rectified_R = undistorted_rectified_R[y_R:y_R+h_R, x_R:x_R+w_R]

    for i in range(0, 480, 20):

        y = i  # Ví dụ: chọn một giá trị y trên ảnh trái (hoặc có thể chọn nhiều điểm)

        # Vẽ đường epipolar trên ảnh trái (hàng ngang)
        cv2.line(undistorted_rectified_L, (0, y), (undistorted_rectified_L.shape[1], y), (0, 255, 0), 1)

        # Vẽ đường epipolar trên ảnh phải (hàng ngang)
        cv2.line(undistorted_rectified_R, (0, y), (undistorted_rectified_R.shape[1], y), (0, 255, 0), 1)


    cv2.imshow("Left Image with Epipolar Line", undistorted_rectified_L)
    cv2.imshow("Right Image with Epipolar Line", undistorted_rectified_R)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def open_camera_stereo():
    capL = cv2.VideoCapture(0)
    capR = cv2.VideoCapture(1)

    if not capL.isOpened():
        print('Không thể mở camera trái')
        exit()
    
    if not capR.isOpened():
        print('Không thể mở camera phải')
        exit()
    
    #Read parameter calib
    file_path = "improved_params2.xml"
    if not os.path.exists(file_path):
        print("Không tìm thấy file {file_path}")
        exit()

    fs = cv2.FileStorage(file_path, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        print("Không đọc được file {file_path}")
        exit()
    
    left_parameter_x = fs.getNode("Left_Stereo_Map_x").mat()
    left_parameter_y = fs.getNode("Left_Stereo_Map_y").mat()

    right_parameter_x = fs.getNode("Right_Stereo_Map_x").mat()
    right_parameter_y = fs.getNode("Right_Stereo_Map_y").mat()
    fs.release()

    if left_parameter_x is None:
        print("Không đọc được nội dung left_parameter_x")
        exit()
    elif left_parameter_y is None:
        print("Không đọc được nội dung left_parameter_y")
        exit()
    elif right_parameter_x is None:
        print("Không đọc được nội dung right_parameter_x")
        exit()
    elif right_parameter_y is None:
        print("Không đọc được nội dung right_parameter_y")
        exit()

    return capL, capR, left_parameter_x, left_parameter_y, right_parameter_x, right_parameter_y         



def test_video_calib():
    capL, capR, left_parameter_x, left_parameter_y, right_parameter_x, right_parameter_y = open_camera_stereo()
    #Off auto exposure
    capR.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)  
    capL.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75) 
    # auto_exposure = capL.get(cv2.CAP_PROP_AUTO_EXPOSURE)
    # print(f"Auto Exposure Current Value: {auto_exposure}")

    # # Tắt Auto White Balance
    capL.set(cv2.CAP_PROP_AUTO_WB, 1)
    capR.set(cv2.CAP_PROP_AUTO_WB, 1)

    # # Đặt White Balance Temperature (giá trị tùy chỉnh, thử nghiệm từ 4000-6000)
    # wb_temperature = 5000
    # capL.set(cv2.CAP_PROP_WB_TEMPERATURE, wb_temperature)
    # capR.set(cv2.CAP_PROP_WB_TEMPERATURE, wb_temperature)
    fourccL = int(capL.get(cv2.CAP_PROP_FOURCC))
    fourccR = int(capR.get(cv2.CAP_PROP_FOURCC))
    print(f"Codec Camera L: {fourccL}, Camera R: {fourccR}")



    
    count = 1
    while True:

        retL, frameL = capL.read()
        retR, frameR = capR.read()

        if not retL:
            print("Không thể đọc camera Trái")
            exit()

        if not retR:
            print("Không thể đọc camera phải")
            exit()
            
        undistorted_rectified_L = cv2.remap(frameL, left_parameter_x, left_parameter_y, cv2.INTER_LINEAR)
        undistorted_rectified_R = cv2.remap(frameR, right_parameter_x, right_parameter_y, cv2.INTER_LINEAR)

        # print("Original left image shape:", imagL.shape)
        # print("Rectified left image shape:", undistorted_rectified_L.shape)

        # x_L, y_L, w_L, h_L = roiR
        # x_R, y_R, w_R, h_R = roiR

        # # print("roiL: ", roiL)
        # # print("roiR: ", roiR)

        # undistorted_rectified_L = undistorted_rectified_L[y_L:y_L+h_L, x_L:x_L+w_L]
        # undistorted_rectified_R = undistorted_rectified_R[y_R:y_R+h_R, x_R:x_R+w_R]

        cv2.imshow('VideoL', undistorted_rectified_L)
        cv2.imshow('VideoR', undistorted_rectified_R)

        if cv2.waitKey(1) == ord('s'):
            path_folder = './imag/test'
            
            image_nameL = f"{path_folder}/image"+str(count)+"0.jpg"
            image_nameR = f"{path_folder}/image"+str(count)+"1.jpg"

            cv2.imwrite(image_nameL, undistorted_rectified_L)
            cv2.imwrite(image_nameR, undistorted_rectified_R)
            print('Lưu ảnh thành công tại {image_nameL}')
            count +=1


        if cv2.waitKey(1) == 27:
            break
    capL.release()
    capR.release()
    cv2.destroyAllWindows()

def nothing(x):
    pass

def cal_disparity():
    
    
    capL, capR, left_parameter_x, left_parameter_y, right_parameter_x, right_parameter_y = open_camera_stereo()

    # Kiểm tra auto exposure camera
    if capL.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75) and capR.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75):  # 0.25 thường tắt auto exposure
        print("Auto exposure enabled")
    else:
        print("Failed to enable auto exposure")
    
    

    #StereoSGBM
    window_size = 3
    min_disp = 2
    num_disp = 128
    stereo = cv2.StereoSGBM_create(minDisparity = min_disp,
                                    numDisparities = num_disp,
                                    blockSize = window_size,
                                    uniquenessRatio = 10,
                                    speckleWindowSize = 100,
                                    speckleRange = 32,
                                    disp12MaxDiff = 5,
                                    P1 = 8*3*window_size**2,
                                    P2 = 32*3*window_size**2)

    cv2.namedWindow("disp", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("disp", 600, 600)
    cv2.createTrackbar('numDisparities', 'disp', 8, 17, nothing)
    cv2.createTrackbar('blockSize', 'disp', 2,50, nothing)
    cv2.createTrackbar('uniquenessRatio', 'disp', 0,100, nothing)
    cv2.createTrackbar('speckleWindowSize', 'disp', 50, 50, nothing)
    cv2.createTrackbar('speckleRange', 'disp', 32, 100, nothing)
    cv2.createTrackbar('disp12MaxDiff', 'disp', 5, 100, nothing)
    cv2.createTrackbar('P1', 'disp', 8, 30, nothing)
    cv2.createTrackbar('P2', 'disp', 32, 60, nothing)
    cv2.createTrackbar('minDisparity', 'disp', 2, 25, nothing)
    cv2.createTrackbar('exposureL', 'disp', 10, 50, nothing)
    cv2.createTrackbar('exposureR', 'disp', 10, 50, nothing)
    
    #size filter
    kernel= np.ones((5,5),np.uint8)

    # Used for the filtered image
    stereoR=cv2.ximgproc.createRightMatcher(stereo) # Create another stereo for right this time

    # WLS FILTER Parameters
    lmbda = 80000
    sigma = 1.4
    visual_multiplier = 1.0
    
    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=stereo)
    wls_filter.setLambda(lmbda)
    wls_filter.setSigmaColor(sigma)

    cv2.namedWindow("Frame_Left", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Frame_Left", 640, 480)
    cv2.namedWindow("Frame_Right", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Frame_Right", 640, 480)

    
    # h = 5
    # hForColorComponents = 5
    # templateWindowSize = 7
    # searchWindowSize = 21
    while True:
        # start_time = time.time()
        retL, frameL = capL.read()
        retR, frameR = capR.read()
        if not retL:
            print("Không thể đọc camera Trái")
            break

        if not retR:
            print("Không thể đọc camera phải")
            break
        
        undistorted_rectified_L = cv2.remap(frameL, left_parameter_x, left_parameter_y, cv2.INTER_LINEAR)
        undistorted_rectified_R = cv2.remap(frameR, right_parameter_x, right_parameter_y, cv2.INTER_LINEAR)

        img_L = undistorted_rectified_L
        img_R = undistorted_rectified_R


        # img_L = cv2.fastNlMeansDenoisingColored(img_L, None, h, hForColorComponents, templateWindowSize, searchWindowSize)  
        # img_R = cv2.fastNlMeansDenoisingColored(img_R, None, h, hForColorComponents, templateWindowSize, searchWindowSize) 

       
        img_L = cv2.cvtColor(img_L, cv2.COLOR_BGR2GRAY)
        img_R = cv2.cvtColor(img_R, cv2.COLOR_BGR2GRAY)
        
        cv2.imshow("Frame_Left",img_L)
        cv2.imshow("Frame_Right",img_R)


        

        #StereoSGBM
        numDisparitiy = cv2.getTrackbarPos('numDisparities','disp')*16
        blockSize = cv2.getTrackbarPos('blockSize','disp')*2 + 3
        uniquenessRatio = cv2.getTrackbarPos('uniquenessRatio','disp')
        speckleRange = cv2.getTrackbarPos('speckleRange','disp')
        speckleWindowSize = cv2.getTrackbarPos('speckleWindowSize','disp')*2
        disp12MaxDiff = cv2.getTrackbarPos('disp12MaxDiff','disp')
        minDisparity = cv2.getTrackbarPos('minDisparity','disp')
        P1 = cv2.getTrackbarPos('P1', 'disp')*blockSize**2
        P2 = cv2.getTrackbarPos('P2', 'disp')*blockSize**2
        exposureL = float(cv2.getTrackbarPos('exposureL', 'disp'))/(-10.0)
        exposureR = float(cv2.getTrackbarPos('exposureR', 'disp'))/(-10.0)
        # print(exposureL)


        stereo.setNumDisparities(numDisparitiy)
        stereo.setBlockSize(blockSize)
        stereo.setUniquenessRatio(uniquenessRatio)
        stereo.setSpeckleRange(speckleRange)
        stereo.setSpeckleWindowSize(speckleWindowSize)
        stereo.setDisp12MaxDiff(disp12MaxDiff)
        stereo.setMinDisparity(minDisparity)
        stereo.setP1(P1)
        stereo.setP2(P2)

        # # #Off auto exposure
        # capR.set(cv2.CAP_PROP_EXPOSURE, exposureR)  
        # capL.set(cv2.CAP_PROP_EXPOSURE, exposureL) 

        disparity = stereo.compute(img_L,img_R)
        # Compute the 2 images for the Depth_image
        disp= stereo.compute(img_L,img_R)#.astype(np.float32)/ 16
        dispL= disp
        dispR= stereoR.compute(img_R,img_L)
        dispL= np.int16(dispL)
        dispR= np.int16(dispR)

        dispL= cv2.morphologyEx(dispL,cv2.MORPH_CLOSE, kernel)
        dispR= cv2.morphologyEx(dispR,cv2.MORPH_CLOSE, kernel)

        # Using the WLS filter
        filteredImg= wls_filter.filter(dispL,img_L,None,dispR)
        filter_imag1 = filteredImg.copy()
        # print("min: {},   max: {}".format(filter_imag1.min(), filter_imag1.max()))

        


        # print("filteredImg", filteredImg)
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
        disparity= cv2.morphologyEx(disparity,cv2.MORPH_CLOSE, kernel)
        
        # fps = 1.0/(float(time.time() - start_time))
        # cv2.putText(disparity, "fps = "+str(fps), (50, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255),2)
        # Displaying the disparity map
        cv2.namedWindow("disp1", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("disp1", 640, 480)
        cv2.imshow("disp1",disparity)
        cv2.setMouseCallback("WLS_filter", mouse_callback, (filter_imag1, filter_imag1.min(), filter_imag1.max()))

        
        
        
        # Close window using esc key
        if cv2.waitKey(1) == 27:
            break
    print("min: {},   max: {}".format(filter_imag1.min(), filter_imag1.max()))
    capL.release()
    capR.release()
    cv2.destroyAllWindows()


count_click = 0
# count_tmp = 0
distance = np.arange(70,281, 10)
lst_disp_average = []
disp_tmp = []


# Hàm callback khi click chuột
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:  # Khi nhấn chuột trái
        
        global count_click
        global distance
        global lst_disp_average
        global disp_tmp

        param1, param2, param3 = param
        # disparity = param1[y,x]
        # val_max = param3
        # print(f"Tọa độ: ({x}, {y}): {disparity}")
        # print("min: {},   max: {}".format(param2, param3))

        average=0
        for u in range (-1,2):
            for v in range (-1,2):
                average += param1[y+u,x+v]
        disparity=average/9

        #normalize disprity WLS filter
        disparity_normalize = float(disparity)/16.0/128.0
        # print("Distance truth: ", distance[int(count_click/3)]) 
        print("disparity_normalize: ", disparity_normalize)
        # print('Lưu giá trị tọa độ này? y/n')


        # cal_distance =  -462.2209*disparity_normalize**3 + 1202.9719*disparity_normalize**2 + -1135.1228*disparity_normalize + 458.2025
        # cal_distance = -825.8*disparity_normalize**3 + 1942.0*disparity_normalize**2 - 1619.0*disparity_normalize + 562.9 #Bậc 3
        # cal_distance = 1181*disparity_normalize**4 - 3618*disparity_normalize**3 + 4269*disparity_normalize**2 - 2423*disparity_normalize + 659.9
        cal_distance = 1815*disparity_normalize**4 - 5329*disparity_normalize**3 + 5945*disparity_normalize**2 - 3130*disparity_normalize + 769.4
        
        
        cal_distance = np.around(cal_distance, 2)
        print('Distance: '+ str(cal_distance)+' cm')

       

        # while True:
        #     if cv2.waitKey(1) == ord('y'):
        #         disp_tmp.append(disparity_normalize)
        #         print("Đã lưu tọa độ: ({}, {}): {}".format(x,y,disparity_normalize))
        #         print('\n')
        #         count_click +=1
        #         break

        #     elif cv2.waitKey(1) == ord('n'):
        #         print('pass')
        #         print('\n')
        #         break    
            
        #     elif cv2.waitKey(1) == ord(' '):
        #         print("disp_tmp: ",disp_tmp)
        #         print("disp_average: ",lst_disp_average)
        #         break

        #     elif cv2.waitKey(1) == ord('p'):
        #         interpolate_equation()
        #         break
            
        
        # if count_click%3 == 0 and int(count_click/3) != distance.size:
        #     disparity_arverage = float(np.mean(disp_tmp))
        #     print("Distance truth: {},  average: {}".format(distance[int(count_click/3)-1], disparity_arverage) )

        #     lst_disp_average.append(disparity_arverage)
        #     disp_tmp = []
        #     print("List disp_average = ", lst_disp_average)

        # if int(count_click/3)  == distance.size:
        #     interpolate_equation()
        
def interpolate_equation():
    # x = distance
    # y = lst_disp_average
    # # lần 1
    # y = np.arange(70,271, 10)
    # x = np.array([0.96044921875, 0.8572591145833334, 0.7508138020833334, 0.686767578125, 0.62353515625, 0.5690104166666666, 0.5294596354166666, 0.4933268229166667, 0.4607747395833333, 0.4305013020833333, 0.40234375, 0.3815104166666667, 0.36572265625, 0.3421223958333333, 0.3299153645833333, 0.3160807291666667, 0.2950846354166667, 0.2845052083333333, 0.2736002604166667, 0.25927734375, 0.2501627604166667])
    
    # lần 2
    y = np.arange(70,281, 10)
    x = np.array([0.9825846354166666, 0.8611653645833334, 0.765625, 0.6886393229166666, 0.626953125, 0.57666015625, 0.5325520833333334, 0.49609375, 0.4625651041666667, 0.43798828125, 0.4122721354166667, 0.390625, 0.3693033854166667, 0.3512369791666667, 0.3349609375, 0.3190104166666667, 0.3050130208333333, 0.2945963541666667, 0.28076171875, 0.271484375, 0.2638346354166667, 0.2566731771])

    coefficients = np.polyfit(x, y, deg=4)

    # Tạo phương trình từ các hệ số
    poly = np.poly1d(coefficients)

    print("Phương trình hồi quy bậc 3:")
    print(poly)

    x_new = np.linspace(min(x), max(x), 100)  # Tạo dữ liệu mượt hơn để vẽ
    y_new = poly(x_new)

    plt.scatter(x, y, color='red', label='Dữ liệu gốc')  # Dữ liệu gốc
    plt.plot(x_new, y_new, label='Đường hồi quy bậc 3', linestyle='--')  # Đường cong xấp xỉ
    plt.legend()
    plt.show()
    



if __name__ == "__main__":
    # calib_camera_stereo()
    # test_video_calib()
    cal_disparity()
    # interpolate_equation()









