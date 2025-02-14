
import cv2
import numpy as np

# Ma trận nội tại (K) và hệ số biến dạng (dist_coeffs)
fx = 2.03629249e+03 
fy = 2.05075722e+03
cx = 8.82031851e+02
cy = 5.29602845e+02

k1 = -0.34495658
k2 = -0.0371805
k3 = 0.26341398
p1 = 0.00075317
p2 = 0.00210568
K = np.array([[fx, 0, cx],
              [0, fy, cy],
              [0,  0,  1]]) 
dist_coeffs = np.array([k1, k2, p1, p2, k3])  

#Kích thước ảnh calib
orginal_size = (1920,1080)
current_size = (640, 480)

scale_x = float(current_size[0]/orginal_size[0])
scale_y = float(current_size[1]/orginal_size[1])

K_current = K.copy()

K_current[0,0] *= scale_x
K_current[0,2] *= scale_x
K_current[1,1] *= scale_y
K_current[1,2] *= scale_y

# Khởi tạo webcam
cap = cv2.VideoCapture(0)  # 0 là ID của webcam, thay đổi nếu có nhiều camera

# Kiểm tra xem webcam có mở được không
if not cap.isOpened():
    print("Không thể mở webcam")
    exit()

# Lấy thông số khung hình từ webcam
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Kích thước khung hình: {frame_width}x{frame_height}")

# Tính toán ma trận camera mới và vùng ROI tối ưu
new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(K_current, dist_coeffs, current_size, 1,current_size)
imag = cv2.imread('test.jpg')
dst = cv2.undistort(imag, K_current, dist_coeffs, None, new_camera_matrix)


# x, y, w, h = roi
# dst = dst[y:y+h, x:x+w]
 
# # Displaying the undistorted image
# cv2.namedWindow("undistorted image", cv2.WINDOW_NORMAL)
# cv2.resizeWindow("undistorted image", 960, 540)
# cv2.imshow("undistorted image",dst)
# cv2.waitKey(0)

while True:
    # Đọc từng khung hình từ webcam
    ret, frame = cap.read()
    if not ret:
        print("Không thể đọc khung hình từ webcam")
        break

    # Loại bỏ biến dạng
    undistorted_frame = cv2.undistort(frame, K_current, dist_coeffs, None, new_camera_matrix)

    # VÙng tối ưu không chứ viền có khoảng đen
    x, y, w, h = roi
    undistorted_frame = undistorted_frame[y:y+h, x:x+w]

    # Hiển thị video gốc và video đã hiệu chỉnh
    cv2.imshow("Original Video", frame)
    cv2.imshow("Undistorted Video", undistorted_frame)

    # Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()

