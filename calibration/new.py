import cv2
import numpy as np

dir = r"C:\Users\KidHe\Desktop"
cam_list = ['azure_kinect_0', 'azure_kinect_1', 'azure_kinect_2', 'kinect_v2_1', 'kinect_v2_2']
# start_idx = [0, 0, 0, 0, 0]
cam_R, cam_T, cam_M = [], [], []
#n_list = [60, 60, 60, 40, 40]
n_list = [1, 1, 1, 1, 1]
col = 8
row = 11
square_size = 43.
objp = np.zeros((col*row, 3), np.float32)
objp[:, :2] = np.mgrid[0:col, 0:row].T.reshape(-1, 2) * square_size
h =0
w =0
for idx, cam in enumerate(cam_list):
    obj_points = []
    img_points = []
    cam_dir = "%s/%s" % (dir, cam)
    for i in range(n_list[idx]):
        filename = "%s_color0020.jpg" %(cam_dir)
        print(filename)
        img = cv2.imread(filename)
        h, w = img.shape[0], img.shape[1]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (col, row), None)
        # cv2.drawChessboardCorners(img, (col, row), corners,ret)
        # cv2.imshow("i",img)
        # cv2.waitKey(0)
        if ret:
            corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), None)
            obj_points.append(objp)
            img_points.append(np.squeeze(corners2))
    if(img_points != []):
        obj_points= np.array(obj_points).astype('float32')
        img_points=np.array(img_points).astype('float32')
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, (w, h), None, None, flags=cv2.CALIB_RATIONAL_MODEL)
        dist = tuple(dist[0, 0:8])
        
        retval, rvec, tvec = cv2.solvePnP(objectPoints=obj_points, imagePoints=img_points, cameraMatrix=mtx, distCoeffs=dist)
        cam_R.append(rvec)
        cam_T.append(tvec)

print(cam_R)


 