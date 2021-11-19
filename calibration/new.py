import cv2
import numpy as np

dir = "C:\Users\xh\Documents\GitHub\multi_camera_calibration"
cam_list = ['azure_kinect_0', 'azure_kinect_1', 'azure_kinect_2', 'kinect_v2_1', 'kinect_v2_2']
start_idx = [0, 0, 0, 0, 0]
cam_R, cam_T, cam_M = [], [], []
n_list = [60, 60, 60, 40, 40]

col = 7
row = 11
cam_R, cam_T = []
objp = np.zeros((col*row, 3), np.float32)
objp[:, :2] = np.mgrid[0:col, 0:row].T.reshape(-1, 2) * square_size
h =0
w =0
for idx, cam in enumerate(cam_list):
    obj_points = []
    img_points = []
    cam_dir = "%s/%s_calib_snap" % (root_dir, cam)
    for i in n_list[idx]:
        filename = "xxx"
        img = cv2.imread(filename)
        h, w = img.shape[0], img.shape[1]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (col, row), None)
        if ret:
            corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
            obj_points.append(objp)
            img_points.append(corners2)
            
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, (w, h), None, None, flags=cv2.CALIB_RATIONAL_MODEL)
    dist = tuple(dist[0, 0:8])
    
    retval, rvec, tvec = cv2.solvePnP(objectPoints=obj_points, imagePoints=img_points, cameraMatrix=mtx, distCoeffs=dist)
    cam_R.append(rvec)
    cam_T.append(tvec)




 