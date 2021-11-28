import cv2
import numpy as np

dir = r"C:\Users\lovettxh\Documents\GitHub\multi_camera_calibration\calibration"
cam_list = ['kinect_v2_1', 'azure_kinect_1'，'azure_kinect_0', 'azure_kinect_2', 'kinect_v2_2']
start_idx = [20, 20, 0, 100, 0]
cam_R, cam_T, cam_M = [], [], []
n_list = [80, 100, 100, 20, 20]
col = 8
row = 11
square_size = 43.
objp = np.zeros((col*row, 3), np.float32)
objp[:, :2] = np.mgrid[0:col, 0:row].T.reshape(-1, 2) * square_size
h = 0
w = 0
corner_m = []
obj_m = []
for idx, cam in enumerate(cam_list):
    obj_points = []
    img_points = []
    cam_dir = "%s/%s_calib_snap" % (dir, cam)
    for i in range(n_list[idx]):
        filename = '%s/color%04i.jpg' % (cam_dir, i + start_idx[idx])
        print(i)
        img = cv2.imread(filename)
        h, w = img.shape[0], img.shape[1]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (col, row), None)
        if ret:
            print(filename)
            corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), None)
            obj_points.append(objp)
            img_points.append(np.squeeze(corners2))
    
    obj_points= np.array(obj_points).astype('float32')
    img_points=np.array(img_points).astype('float32')
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, (w, h), None, None, flags=cv2.CALIB_RATIONAL_MODEL)   
    retval, rvec, tvec = cv2.solvePnP(objectPoints=obj_points[0], imagePoints=img_points[0], cameraMatrix=mtx, distCoeffs=dist)
    cam_R.append(rvec)
    cam_T.append(tvec)

# corner_m = np.squeeze(corner_m)
# print(corner_m[0])
# obj_m = []
# obj_m.append(objp)
# c1 = []
# c2 = []
# c1.append(corner_m[0])
# c2.append(corner_m[1])
# retval, _, _, _, _, R, T, _, _ = \
#     cv2.stereoCalibrate(objectPoints=obj_m,
#                         imagePoints1=c1,
#                         imagePoints2=c2,
#                         imageSize=(w,h),
#                         cameraMatrix1=mtx_m[0],
#                         distCoeffs1=dist_m[0],
#                         cameraMatrix2=mtx_m[1],
#                         distCoeffs2=dist_m[1],
#                         criteria=(cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 200, 1e-6),
#                         flags=cv2.CALIB_FIX_INTRINSIC | cv2.CALIB_RATIONAL_MODEL)
RT = {}
for i in range(len(cam_list)):
    if i < len(cam_list) - 1: 
        r1 = cv2.Rodrigues(cam_R[i])[0]
        r2 = cv2.Rodrigues(cam_R[i + 1])[0]
        t1 = cam_T[i]
        t2 = cam_T[i + 1]
        r12 = np.dot(r1, r2.T)
        t12 = t1 - np.dot(r12, t2)
        RT[cam_list[i] + ' ' + cam_list[i+1] + '_R'] = r12
        RT[cam_list[i] + ' ' + cam_list[i+1] + '_T'] = t12
    else:
        r1 = cv2.Rodrigues(cam_R[i])[0]
        r2 = cv2.Rodrigues(cam_R[0])[0]
        t1 = cam_T[i]
        t2 = cam_T[0]
        r12 = np.dot(r1, r2.T)
        t12 = t1 - np.dot(r12, t2)
        RT[cam_list[i] + ' ' + cam_list[0] + '_R'] = r12
        RT[cam_list[i] + ' ' + cam_list[0] + '_T'] = t12
        

for key, value in RT.items():
    print (key)
    print(value)