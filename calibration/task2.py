import joblib
import numpy as np
import json



def read_json(json_path):
    with open(json_path) as f:
        data = json.load(f)
    kps = []
    for people in data['people']:
        kp = np.array(people['pose_keypoints_2d']).reshape(-1, 3)
        kps.append(kp)
    return kps

dir = r"E:\ECE740\multiview_color"
cam_list = ['kinect_v2_1', 'azure_kinect_1','azure_kinect_0',  'azure_kinect_2', 'kinect_v2_2']
hmr_list = [2, 2, 2, 2, 2]
start_idx = [0, 0, 0, 0, 0]
cams_pkl = {}
cams_kps = {}

for idx, cam in enumerate(cam_list):
    cam_joints2d = []
    cam_joints3d = []
    cam_theta = []
    cam_trans = []
    cam_kps = []
    cam_dir = "%s/%s_hmr" % (dir, cam)
    json_path = "%s/%s_openpose/%s_openpose" % (dir, cam, cam)
    for i in range(hmr_list[idx]):
        filename_pkl = '%s/color%04i_hmr.pkl' % (cam_dir, i + start_idx[idx])
        filename_kps = '%s/color%04i_keypoints.json' % (json_path, i + start_idx[idx])
        kps = read_json(filename_kps)
        cam_kps.append(kps)
        pkl = joblib.load(filename_pkl)
        cam_joints2d.append(pkl[0])
        cam_joints3d.append(pkl[1])
        cam_theta.append(pkl[2])
        cam_theta.append(pkl[3])

    cams_pkl[cam] = [cam_joints2d, cam_joints3d, cam_theta, cam_trans]
    cams_kps[cam] = cam_kps




