import argparse
import glob
import math
import matplotlib.pyplot as plt
# import camera_info_manager
import numpy as np 
import gtsam
from numpy import linalg as LA
# import yaml
import json

class MocapDataHelper(object):
    def __init__(self):
        camera_extrinsic = np.array([1.0, 0.0, -0.01, 0.743, 0.0, -1.0, -0.007, 0.082, -0.01, 0.007, -1.0, 1.441, 0.0, 0.0, 0.0, 1.0]);
        # mocap_tf = np.array([-0.706755, 0.0005153, -0.705954,0.572327, 0.70638, 0.000213504, -0.706755,-0.05, -0.000213596, -0.999396, -0.00051508, 1.00017, 0,0,0,1]);
        c_T_m = np.array([[-0.72231893, -0.69031471, 0.04148439, -0.0911841 ], [-0.03982037, -0.01837056, -0.99903797, 2.43988996], [ 0.6904127, -0.72327596, -0.01421918, 0.18661254], [ 0.,     0.,     0.,     1.    ]])
        c_T_m[0:3,0:3] = c_T_m[0:3,0:3].T
        c_T_m[0:3,3] = -c_T_m[0:3,0:3] @ c_T_m[0:3,3]
        w_T_c = np.array([[1.0,0.0,-0.01,0.743],[0.0,-1.0,-0.007,0.082],[-0.01,0.007,-1.0,1.441],[0,0,0,1]])
        mocap_tf = w_T_c @ c_T_m
        camera_extrinsic = camera_extrinsic.reshape((4, 4))
        # mocap_tf = mocap_tf.reshape((4, 4))
        # self.camera_pose = gtsam.Pose3(camera_extrinsic);
        # self.mocap_pose = gtsam.Pose3(mocap_tf)
        self.offset = np.array([0, 0, 0.325 / 2.0 ]);

        # self.offset = np.array([0.572, -0.050, 1.000, 0])
    
    def read_camera_param(self, filename):
        f = open(filename, 'r')
        cam_params = yaml.load(f, Loader=yaml.FullLoader)

        camera_extrinsic = np.array(cam_params['camera_extrinsics']+[0,0,0,1], np.float32) #: [0.999,0.001,-0.037,0.459,0.0,-1.0,-0.014,0.284,-0.037,0.014,-0.999,1.431]
        camera_extrinsic = camera_extrinsic.reshape((4, 4))
        self.camera_pose = gtsam.Pose3(camera_extrinsic);

        c_T_m = np.array(cam_params['camera_Tf_mocap'], np.float32) #: [0.999,0.001,-0.037,0.459,0.0,-1.0,-0.014,0.284,-0.037,0.014,-0.999,1.431]
        c_T_m = c_T_m.reshape((4, 4))
        # c_T_m = np.array([[-0.72231893, -0.69031471, 0.04148439, -0.0911841 ], [-0.03982037, -0.01837056, -0.99903797, 2.43988996], [ 0.6904127, -0.72327596, -0.01421918, 0.18661254], [ 0.,     0.,     0.,     1.    ]])
        # c_T_m = np.array([[-0.71982338, -0.69235535,  0.04998369, -0.09336075], [-0.04100413, -0.02947068, -0.99872426,  2.16381353], [ 0.69294514, -0.72095461, -0.00717574,  0.1893245 ], [ 0.        ,  0.        ,  0.        ,  1.        ]])
        c_T_m[0:3,0:3] = c_T_m[0:3,0:3].T
        c_T_m[0:3,3] = -c_T_m[0:3,0:3] @ c_T_m[0:3,3]
        mocap_tf = camera_extrinsic @ c_T_m
        self.mocap_pose = gtsam.Pose3(mocap_tf)
        # print(camera_extrinsic)
        # print(c_T_m)
        print(f"mocap_tf {mocap_tf}")
        f.close();

    def camera_transform(self, data):
        for di in data:
            for idx in range(3):
                if data[di][idx] is not None: 
                    data[di][idx] = self.camera_pose * data[di][idx];
        return data


    def mocap_transform(self, data):
        # ptp = self.mocap_tf @ pt + self.offset; 
        # ptp = (self.mocap_tf @ pt)[0:3] + self.mocap_tf[0:3,3]; 
        # ptp = self.mocap_pose.transformFrom(pt)
        # ptp[2] = -ptp[2]
            
        for di in data:
            for idx in range(3):
                if data[di][idx] is not None: 
                    # print(f"pose: {data[di][idx]}")
                    # print(f"self.mocap_pose: {self.mocap_pose}")
                    data[di][idx] = self.mocap_pose * data[di][idx];

        return data

    def read_from_npy(self, directory):
        red = np.load(directory+"/red.npy")
        green = np.load(directory+"/green.npy")
        blue = np.load(directory+"/blue.npy")

        data = {}
        idx = 0
        for r,g,b in zip(red, green, blue):
            poses = []

            poses.append(gtsam.Pose3(r))
            poses.append(gtsam.Pose3(g))
            poses.append(gtsam.Pose3(b))
            data[idx] = poses
            idx += 1
            # print(f" red pose: {poses[0]}")
        return data;

    def read_poses_data(self, filename):
        file = open(filename, 'r')

        data = {}
        for line in file:
            l = line.split()
            if l[0] == "#":
                continue
            ti = float(l[1])
            q0 = np.array([l[2], l[3], l[4], l[5]], np.float32)
            t0 = np.array([l[6], l[7], l[8]], np.float32)
            q1 = np.array([l[9], l[10], l[11], l[12]], np.float32)
            t1 = np.array([l[13], l[14], l[15]], np.float32)
            q2 = np.array([l[16], l[17], l[18], l[19]], np.float32)
            t2 = np.array([l[20], l[21], l[22]], np.float32)

            poses = []
            for q, t in zip([q0,q1,q2], [t0,t1,t2]):
                if np.isnan(q).any() or np.isnan(t).any():
                    poses.append(None)
                else:
                    R = gtsam.Rot3(q[0],q[1],q[2],q[3]);
                    poses.append(gtsam.Pose3(R, t))

            data[ti] = poses
        # data_out = [[None,None,None]] * (idx+1)
        # # print(f"data_out {len(data_out)}")
        # for m_idx in data:
        #     # print(m_idx)
        #     data_out[m_idx] = data[m_idx]
        return data

    def plot(self, ax, data, marker, color):

        xi = []
        yi = []
        zi = []

        for di in data:
            # if di is not None:

            for pose in data[di]:
                if pose is not None: 
                    pt0 = pose.transformFrom(self.offset)
                    pt1 = pose.transformFrom(-self.offset)
                    # print(f"p0 {pt0} p1 {pt1}")
                    # xi.append(pose.translation()[0]);
                    # yi.append(pose.translation()[1]);
                    # zi.append(pose.translation()[2]);
                    xi.append(pt0[0]);
                    yi.append(pt0[1]);
                    zi.append(pt0[2]);
                    xi.append(pt1[0]);
                    yi.append(pt1[1]);
                    zi.append(pt1[2]);
                # print(f"m: {di[mi]}")
                # ax.scatter(di[mi][0],di[mi][1],di[mi][2], marker='o')
        ax.scatter(xi, yi, zi, marker=marker, c=color)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')


    def compute_errors(self, gt_data, z_data, filename_prefix ):

        all_errors = []
        all_errors_cm = []
        missed_frames = 0;
        rot_symmetry = gtsam.Pose3(gtsam.Rot3(0.0, 0.0, 1.0, 0.0),[0.0, 0.0, 0.0]) ;
        # print(f"z_data {z_data}")
        for gt_idx in gt_data:
            error = []
            error_cm = []
            if gt_idx in z_data:
                for gt_p, z_p in zip(gt_data[gt_idx], z_data[gt_idx]):
                    if gt_p is not None and z_p is not None: 
                        err0 = np.abs(gtsam.Pose3.Logmap(gt_p.between(z_p)))
                        err1 = np.abs(gtsam.Pose3.Logmap(gt_p.between(z_p * rot_symmetry )))
                        if LA.norm(err0) < LA.norm(err1):
                            error.append(err0)
                        else:
                            error.append(err1)
                        # error.append(np.abs(gtsam.Pose3.Logmap(gt_p.between(z_p))))
                        error_cm.append(np.abs(gt_p.translation() - z_p.translation()))
                    else:
                        error.append(np.array([np.nan]*6))
                        error_cm.append(np.array([np.nan]*3))
                all_errors.append(np.array(error))
                all_errors_cm.append(np.array(error_cm))
            else:
                print(f"missed: {gt_idx}")
                missed_frames += 1
        all_errors = np.array(all_errors)
        all_errors_cm = np.array(all_errors_cm)

        # print(f"all_errors {all_errors}")

        mean_err = np.nanmean(all_errors, axis=(0,1))
        stdd_err = np.nanstd(all_errors, axis=(0,1))
        mean_err_cm = np.nanmean(all_errors_cm, axis=(0,1))
        stdd_err_cm = np.nanstd(all_errors_cm, axis=(0,1))
        
        all_errors_file = open(filename_prefix+ "_all_errors.txt", 'w');
        all_errors_cm_file = open(filename_prefix + "_all_errors_cm.txt", 'w');
        stats_file = open(filename_prefix + "_stats.txt", 'w');

        print(f"mean_err {mean_err} missed_frames {missed_frames}")

        # print(f"all_errors {all_errors}")
        for errs in all_errors:
            for e in errs:
                for xi in e:
                    all_errors_file.write(str(xi) + " ")
                all_errors_file.write("\n")

        for errs in all_errors_cm:
            for e in errs:
                for xi in e:
                    all_errors_cm_file.write(str(xi) + " ")
                all_errors_cm_file.write("\n")

        for stat in [mean_err, stdd_err, mean_err_cm, stdd_err_cm, [missed_frames]]:
            for s in stat:
                stats_file.write(str(s) + " ")
        stats_file.write("\n")


        all_errors_file.close()
        all_errors_cm_file.close()
        stats_file.close()
        # t_err = np.nanmean(all_errors[:,:,3:], axis=1)
        # r_err = np.nanmean(all_errors[:,:,:2])
        # print(f"terr {t_err}") # 
        # print(f"rerr {r_err}") # 
        # print(f"terr {t_err * 100}") # 

def vec_to_str(vec):
    str_ = ""
    for e in vec:
        str_ += str(e) 
        str_ += " "
    return str_;


if __name__ == '__main__':

    argparse = argparse.ArgumentParser()

    # argparse.add_argument('-f', '--filename', help='GT', required=True)
    # # argparse.add_argument('-e', '--estimation_filename', help='Estimation', required=True)
    # argparse.add_argument('-o', '--output', help='Estimation', required=True)
    # argparse.add_argument('-c', '--camera_filename', help='Estimation', required=True)
    # argparse.add_argument('-d', '--npy_dir', help='Estimation', required=True)

    # args = argparse.parse_args()

    # filename = args.filename
    # output = args.output
    npy_dir="/home/edgar/remotes/perception/tensegrity_ws/data/prx_lab/open_loop1/average/"
    endcaps0 = np.load(npy_dir+"/0_pos_average.npy")
    endcaps1 = np.load(npy_dir+"/1_pos_average.npy")
    endcaps2 = np.load(npy_dir+"/2_pos_average.npy")
    endcaps3 = np.load(npy_dir+"/3_pos_average.npy")
    endcaps4 = np.load(npy_dir+"/4_pos_average.npy")
    endcaps5 = np.load(npy_dir+"/5_pos_average.npy")
    gt_idx = np.load(npy_dir+"/ground_truth.npy")

    json_dir = "/home/edgar/remotes/perception/tensegrity_ws/data/prx_lab/open_loop1/data/"

    file = open("/home/edgar/remotes/perception/tensegrity_ws/data/prx_lab/open_loop1/endcaps.txt", 'w')
    for i in gt_idx:
        jfile = open(json_dir + f"/{str(i).zfill(4)}.json", 'r' )
        j = json.load(jfile)
        ti = "{:.9f}".format(float(j["header"]["secs"]))
        line = f"{i} {ti} "
        line += f"{endcaps0[i][0]} {endcaps0[i][1]} {endcaps0[i][2]} "
        line += f"{endcaps1[i][0]} {endcaps1[i][1]} {endcaps1[i][2]} "
        line += f"{endcaps2[i][0]} {endcaps2[i][1]} {endcaps2[i][2]} "
        line += f"{endcaps3[i][0]} {endcaps3[i][1]} {endcaps3[i][2]} "
        line += f"{endcaps4[i][0]} {endcaps4[i][1]} {endcaps4[i][2]} "
        line += f"{endcaps5[i][0]} {endcaps5[i][1]} {endcaps5[i][2]} "

        file.write(line + "\n");