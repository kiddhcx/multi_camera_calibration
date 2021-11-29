import cv2
import numpy as np
import sklearn.cluster

class camera_Data:
    def __init__(self, name, cam_R, cam_T, flag):
        self.name = name
        self.R = cam_R
        self.T = cam_T
        self.flag = flag

        
def cluster_pose(cam_R, cam_T):
    """
    :type ls_T: list[Transform]
    """
    # cluster t
    print(len(cam_T))
    meanshift_t = sklearn.cluster.MeanShift(bandwidth=1000, bin_seeding=True)
    meanshift_t.fit(cam_T)
    print(meanshift_t.labels_)
    #assert np.count_nonzero(meanshift_t.labels_ == 0) > (0.7 * len(cam_T))
    t = meanshift_t.cluster_centers_[0]
    print(t)

    # cluster R
    _Rs = np.array(cam_R)[meanshift_t.labels_ == 0]
    _Rs = _Rs.reshape((-1, 9))
    meanshift_R = sklearn.cluster.MeanShift(bandwidth=0.1)
    meanshift_R.fit(_Rs)
    print(meanshift_R.labels_)
    _tmp = meanshift_t.labels_ == 0
    assert isinstance(_tmp, np.ndarray)
    assert np.count_nonzero(_tmp) > (0.7 * len(_Rs))
    R = meanshift_R.cluster_centers_[0].reshape((3, 3))

    # normalize
    R = cv2.Rodrigues(cv2.Rodrigues(R)[0])[0]
    return R, t

def RT_calculate(cam1, cam2):
    f = cam1.flag & cam2.flag
    r1 = cam1.R[f]
    t1 = cam1.T[f]
    r2 = cam2.R[f]
    t2 = cam2.T[f]
    R = []
    T = []
    for i in range(r1.shape[0]):
        r12 = np.dot(cv2.Rodrigues(r1[i])[0], cv2.Rodrigues(r2[i])[0].T)
        t12 = t1[i] - np.dot(r12, t2[i])
        R.append(r12)
        T.append(t12)
    R = np.squeeze(np.asarray(R))
    T = np.squeeze(np.asarray(T))
    return cluster_pose(R, T)



if __name__ == '__main__': 
    dir = r"C:\Users\lovettxh\Documents\GitHub\multi_camera_calibration\calibration\1024"
    cam_list = ['kinect_v2_1', 'azure_kinect_1','azure_kinect_0',  'azure_kinect_2', 'kinect_v2_2']
    start_idx = [0, 0, 0, 0, 0]
    n_list = [124, 124, 124, 124, 124]
    col = 8
    row = 11
    square_size = 60.
    objp = np.zeros((col*row, 3), np.float32)
    objp[:, :2] = np.mgrid[0:col, 0:row].T.reshape(-1, 2) * square_size
    h = 0
    w = 0
    corner_m = []
    obj_m = []
    cams = {}
    for idx, cam in enumerate(cam_list):
        obj_points = []
        img_points = []
        cam_R, cam_T, flag = [], [], []
        cam_dir = "%s/%s_calib_snap" % (dir, cam)
        for i in range(n_list[idx]):
            filename = '%s/color%04i.jpg' % (cam_dir, i + start_idx[idx])
            img = cv2.imread(filename)
            print(filename)
            h, w = img.shape[0], img.shape[1]
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (col, row), None)
            flag.append(ret)
            if ret:
                corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), None)
                obj_points.append(objp)
                img_points.append(np.squeeze(corners2))
            else:
                obj_points.append(objp)
                img_points.append(np.zeros((col*row, 2)))    
        
        obj_points= np.array(obj_points).astype('float32')
        img_points= np.array(img_points).astype('float32')
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points[flag], img_points[flag], (w, h), None, None, flags=cv2.CALIB_RATIONAL_MODEL)
        for i in range(n_list[idx]):
            if flag[i]:
                retval, rvec, tvec = cv2.solvePnP(objectPoints=obj_points[i], imagePoints=img_points[i], cameraMatrix=mtx, distCoeffs=dist)
                cam_R.append(rvec)
                cam_T.append(tvec)
            else:
                cam_R.append(np.zeros((3,1)))
                cam_T.append(np.zeros((3,1)))
        cams[cam] = camera_Data(cam, np.asarray(cam_R), np.asarray(cam_T), np.asarray(flag))
        
        
        result = {}
        temp_r , temp_t = RT_calculate(cams[cam_list[3]], cams[cam_list[4]])
        result[cam_list[3] + "-" + cam_list[4] + "_R"] = temp_r
        result[cam_list[3] + "-" + cam_list[4] + "_T"] = temp_t
        temp_r , temp_t = RT_calculate(cams[cam_list[2]], cams[cam_list[3]])
        result[cam_list[2] + "-" + cam_list[3] + "_R"] = temp_r
        result[cam_list[2] + "-" + cam_list[3] + "_T"] = temp_t
        temp_r , temp_t = RT_calculate(cams[cam_list[2]], cams[cam_list[1]])
        result[cam_list[2] + "-" + cam_list[1] + "_R"] = temp_r
        result[cam_list[2] + "-" + cam_list[1] + "_T"] = temp_t
        temp_r , temp_t = RT_calculate(cams[cam_list[1]], cams[cam_list[0]])
        result[cam_list[1] + "-" + cam_list[0] + "_R"] = temp_r
        result[cam_list[1] + "-" + cam_list[0] + "_T"] = temp_t

        for k, i in result.items():
            print(k)
            print(i)