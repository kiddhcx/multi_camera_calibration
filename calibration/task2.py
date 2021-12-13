import joblib
import numpy as np
import json
import cv2
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def read_json(json_path):
    with open(json_path) as f:
        data = json.load(f)
    kps = []
    for people in data['people']:
        kp = np.array(people['pose_keypoints_2d']).reshape(-1, 3)
        kps.append(kp)
    return kps

def sample_joints():
    file_k = r'multiview_color\multiview_color\azure_kinect_0_hmr\color0000_hmr.pkl'
    file_j = r'multiview_color\multiview_color\azure_kinect_0_openpose\azure_kinect_0_openpose\color0000_keypoints.json'
    #file_cam = r'multiview_color\multiview_color\intrinsic1024.pkl'
    pkl = joblib.load(file_k)
    kps = read_json(file_j)
    #intrin_params = joblib.load(file_cam)
    #intrin_p0 = intrin_params['azure_kinect_0_color']
    # fx, fy, cx, cy, h, w, k1, k2, p1, p2, k3, k4, k5, k6 = intrin_p0
    
    pkl_points = pkl[0]
    json_points = kps[0]
    
    pkl_points = np.append(pkl_points, [[(pkl_points[13,0]+pkl_points[14,0])/2, (pkl_points[13,1]+pkl_points[14,1])/2]], axis = 0)
    pkl_points = np.append(pkl_points, [[(pkl_points[1,0]+pkl_points[2,0])/2, (pkl_points[1,1]+pkl_points[2,1])/2]], axis = 0)
    pkl_points = np.append(pkl_points, [[pkl_points[1,0]+(pkl_points[1,0]-pkl_points[25,0]),pkl_points[1,1]+(pkl_points[1,1]-pkl_points[25,1])]], axis = 0)
    pkl_points = np.append(pkl_points, [[pkl_points[2,0]-(pkl_points[25,0]-pkl_points[2,0]),pkl_points[2,1]+(pkl_points[25,1]-pkl_points[2,1])]], axis = 0)

    img_file = r'multiview_color\multiview_color\azure_kinect_0\color\color0000.jpg'
    img = cv2.imread(img_file, cv2.IMREAD_UNCHANGED)
    font = cv2.FONT_HERSHEY_COMPLEX_SMALL
    for i in range(len(pkl_points)):
        cv2.circle(img, (int(pkl_points[i,0]), int(pkl_points[i,1])), 1, (255,0,0), 5)
        cv2.putText(img, str(i), (int(pkl_points[i,0]), int(pkl_points[i,1])), font, 1, (255,255,255),1)
    skeleton1 = [[21,23],[19,21],[17,19],[14,17],[13,14],[13,16],[16,18],[18,20],[20,22],
                [12,15],[12,9],[9,0],[0,25],[25,26],[25,27],[5,8],[8,11],[5,27],[4,7],[7,10],[4,26]]
    for i in skeleton1:
        cv2.line(img, (int(pkl_points[i[0],0]), int(pkl_points[i[0],1])), (int(pkl_points[i[1],0]), int(pkl_points[i[1],1])), (0,0,255), 2)
    cv2.imwrite('pkl_joints.jpg', img)
    #-----
    img = cv2.imread(img_file, cv2.IMREAD_UNCHANGED)
    font = cv2.FONT_HERSHEY_COMPLEX_SMALL
    for i in range(len(json_points)):
        cv2.circle(img, (int(json_points[i,0]), int(json_points[i,1])), 1, (255,0,0), 5)
        cv2.putText(img, str(i), (int(json_points[i,0]), int(json_points[i,1])), font, 1, (255,255,255),1)
    skeleton2 = [[3,4],[2,3],[1,2],[1,5],[5,6],[6,7],[0,1],[1,8],[8,9],[8,12],[9,10],[10,11],[11,22],[12,13],[13,14],[14,19]]
    for i in skeleton2:
        cv2.line(img, (int(json_points[i[0],0]), int(json_points[i[0],1])), (int(json_points[i[1],0]), int(json_points[i[1],1])), (0,0,255), 2)
    cv2.imwrite('json_joints.jpg', img)

def projection(xyz, intr_param, simple_mode=False):
# xyz: [N, 3]
# intr_param: (fx, fy, cx, cy, w, h, k1, k2, p1, p2, k3, k4, k5, k6)
    assert xyz.shape[1] == 3
    fx, fy, cx, cy = intr_param[0:4]

    if not simple_mode:
        k1, k2, p1, p2, k3, k4, k5, k6 = intr_param[6:14]

        x_p = xyz[:, 0] / xyz[:, 2]
        y_p = xyz[:, 1] / xyz[:, 2]
        r2 = x_p ** 2 + y_p ** 2

        a = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
        b = 1 + k4 * r2 + k5 * r2 ** 2 + k6 * r2 ** 3
        b = b + (b == 0)
        d = a / b

        x_pp = x_p * d + 2 * p1 * x_p * y_p + p2 * (r2 + 2 * x_p ** 2)
        y_pp = y_p * d + p1 * (r2 + 2 * y_p ** 2) + 2 * p2 * x_p * y_p

        u = fx * x_pp + cx
        v = fy * y_pp + cy
        d = xyz[:, 2]

        return np.stack([u, v, d], axis=1)
    else:
        u = xyz[:, 0] / xyz[:, 2] * fx + cx
        v = xyz[:, 1] / xyz[:, 2] * fy + cy
        d = xyz[:, 2]

        return np.stack([u, v, d], axis=1)

def reprocess_points(pkl_points):
    assert pkl_points.shape[1] == 3
    pkl_points = np.append(pkl_points, [[(pkl_points[13,0]+pkl_points[14,0])/2, (pkl_points[13,1]+pkl_points[14,1])/2, (pkl_points[13,2]+pkl_points[14,2])/2]], axis = 0)
    pkl_points = np.append(pkl_points, [[(pkl_points[1,0]+pkl_points[2,0])/2, (pkl_points[1,1]+pkl_points[2,1])/2, (pkl_points[1,2]+pkl_points[2,2])/2]], axis = 0)
    pkl_points = np.append(pkl_points, [[pkl_points[1,0]+(pkl_points[1,0]-pkl_points[25,0]),pkl_points[1,1]+(pkl_points[1,1]-pkl_points[25,1]), pkl_points[1,2]+(pkl_points[1,2]-pkl_points[25,2])]], axis = 0)
    pkl_points = np.append(pkl_points, [[pkl_points[2,0]-(pkl_points[25,0]-pkl_points[2,0]),pkl_points[2,1]+(pkl_points[25,1]-pkl_points[2,1]), pkl_points[2,2]+(pkl_points[25,2]-pkl_points[2,2])]], axis = 0)
    return pkl_points

def map_points(pkl_points, json_points):
    # points mapping: (pkl -> json)
    # (15 -> 0)(24 -> 1)(17 -> 2)(19 -> 3)(21 -> 4)(16 -> 5)(18 -> 6)(20 -> 7)(25 -> 8)(27 -> 9)(26 -> 12)(5 -> 10)(4 -> 13)(8 -> 24)(11 -> 22)(7 -> 21)(10 -> 19)
    assert pkl_points.shape[1] == 3
    new_points = []
    map = [[15,0],[24,1],[17,2],[19,3],[21,4],[16,5],[18,6],[20,7],[25,8],[27,9],[26,12],[5,10],[4,13],[8,24],[11,22],[7,21],[10,19]]
    for i in map:
        new_points.append([pkl_points[i[0],0],pkl_points[i[0],1],pkl_points[i[0],2], json_points[i[1],2]])
    return np.asarray(new_points)

def separate_points(points):
    p = points[:,0:3]
    c = points[:,3]
    return p, c
def combine_points(p, c):
    c = c.reshape((len(c), 1))
    return np.hstack((p,c))

def weighted_avg(p1, p2, p3, p4, p5):
    assert p1.shape[1] == 4
    p1, p1_w = separate_points(p1)
    p2, p2_w = separate_points(p2)
    p3, p3_w = separate_points(p3)
    p4, p4_w = separate_points(p4)
    p5, p5_w = separate_points(p5)
    new_point_set = []
    for i in range(p1.shape[0]):
        sum = p1_w[i] + p2_w[i] + p3_w[i] + p4_w[i] + p5_w[i]
        new_point_set.append(p1[i]*p1_w[i]/sum + p2[i]*p2_w[i]/sum + p3[i]*p3_w[i]/sum + p4[i]*p4_w[i]/sum + p5[i]*p5_w[i]/sum)
    return np.asarray(new_point_set)

def convert_points(kinect_v2_1, azure_kinect_1, azure_kinect_0, azure_kinect_2, kinect_v2_2, extr_param):
    result = []
    R = extr_param['azure_kinect_1-kinect_v2_1_R']
    T = extr_param['azure_kinect_1-kinect_v2_1_T']
    p1, w1 = separate_points(kinect_v2_1)
    p1 = np.dot(R, p1.T).T + T
    R = extr_param['azure_kinect_0-azure_kinect_1_R']
    T = extr_param['azure_kinect_0-azure_kinect_1_T']
    p1 = np.dot(R, p1.T).T + T
    result.append(combine_points(p1, w1))

    p2, w2 = separate_points(azure_kinect_1)
    p2 = np.dot(R, p2.T).T + T
    result.append(combine_points(p2, w2))

    result.append(azure_kinect_0)

    R = extr_param['azure_kinect_0-azure_kinect_2_R']
    T = extr_param['azure_kinect_0-azure_kinect_2_T']
    p3, w3 = separate_points(azure_kinect_2)
    p3 = np.dot(R, p3.T).T + T
    result.append(combine_points(p3, w3))

    R = extr_param['azure_kinect_2-kinect_v2_2_R']
    T = extr_param['azure_kinect_2-kinect_v2_2_T']
    p4, w4 = separate_points(kinect_v2_2)
    
    p4 = np.dot(R, p4.T).T + T
    R = extr_param['azure_kinect_0-azure_kinect_2_R']
    T = extr_param['azure_kinect_0-azure_kinect_2_T']
    p4 = np.dot(R, p4.T).T + T
    result.append(combine_points(p4, w4))
    return result

def distance_error(points, targets):
    map = [[0,15],[2,17],[3,19],[4,21],[5,16],[6,18],[7,20],[11,5],[12,4],[13,8],[15,7],[14,11],[16,10]]
    d = 0
    for i in map:
        x = abs(points[i[0],0] - targets[i[1],0])
        y = abs(points[i[0],1] - targets[i[1],1])
        z = abs(points[i[0],2] - targets[i[1],2])
        #print(math.sqrt(x**2 + y**2 + z**2))
        d += math.sqrt(x**2 + y**2 + z**2)
    return d/13

def save_plot(final_points, i):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    sk = [[0,1],[1,2],[2,3],[3,4],[1,5],[5,6],[6,7],[1,8],[8,9],[8,10],[9,11],[11,13],[13,14],[10,12],[12,15],[15,16]]
    ax.set_xlim3d(800, 1050)
    ax.set_zlim3d(-1000, -400)
    ax.set_ylim3d(2300, 2800)
    for s in sk:
        line_set = []
        line_set.append([final_points[s[0],0], final_points[s[1],0]])
        line_set.append([final_points[s[0],1], final_points[s[1],1]])
        line_set.append([final_points[s[0],2], final_points[s[1],2]])
        line_set = np.asarray(line_set)

        ax.plot3D(line_set[0], line_set[2], -line_set[1])

    fig.savefig('pose_output/%04i.jpg'%(i))
    plt.close(fig)

if __name__ == '__main__':
    sample_joints()
    home_dir = r'multiview_color\multiview_color'
    intrin_dir = r'multiview_color\multiview_color\intrinsic1024.pkl'
    extrin_dir = 'extrinsic.pkl'
    cam_list = ['kinect_v2_1', 'azure_kinect_1','azure_kinect_0',  'azure_kinect_2', 'kinect_v2_2']
    extrin_para = joblib.load(extrin_dir)
    intrin_para = joblib.load(intrin_dir)
    num = 1309
    distance = 0
    for i in range(num):
        
        points = []
        fail = False
        for j in cam_list:
            file_k = "%s/%s_hmr/color%04i_hmr.pkl" % (home_dir, j, i)
            file_j = "%s/%s_openpose/%s_openpose/color%04i_keypoints.json" % (home_dir, j, j, i)
            try:
                pkl = joblib.load(file_k)
            except:
                fail = True
                break
            p = pkl[1] + pkl[3][0]
            p = p * 1000
            jsn = read_json(file_j)[0]
            processed_point = reprocess_points(p)
            processed_point = map_points(processed_point, jsn)
            points.append(processed_point)
        if fail:
            continue
        target_post_dir = "pose_azure_kinect_0/pose_azure_kinect_0/pose%04i.pkl" % (i)
        target = joblib.load(target_post_dir)
        converted_point_set = convert_points(points[0], points[1], points[2], points[3], points[4], extrin_para)
        final_points = weighted_avg(converted_point_set[0], converted_point_set[1], converted_point_set[2], converted_point_set[3], converted_point_set[4])
        distance += distance_error(final_points, target[3]*1000)
        final_points = projection(final_points, intrin_para['azure_kinect_0_color'])
        

        save_plot(final_points, i)

    print("mean distance error per joint(mm): %f" %(distance/num))