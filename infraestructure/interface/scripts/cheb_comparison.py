import argparse
import glob
import math
import matplotlib.pyplot as plt
import scipy.spatial.distance as scipy_distance
import numpy as np 
import gtsam
from numpy import linalg as LA
from interface.cheb_utils import ChebyshevTensegrityPoses

class MocapDataHelper(object):
    def __init__(self, cheb_json):
        camera_extrinsic = np.array([1.0, 0.0, -0.01, 0.743, 0.0, -1.0, -0.007, 0.082, -0.01, 0.007, -1.0, 1.441, 0.0, 0.0, 0.0, 1.0]);
        # mocap_tf = np.array([-0.706755, 0.0005153, -0.705954,0.572327, 0.70638, 0.000213504, -0.706755,-0.05, -0.000213596, -0.999396, -0.00051508, 1.00017, 0,0,0,1]);
        c_T_m = np.array([[-0.72231893, -0.69031471, 0.04148439, -0.0911841 ], [-0.03982037, -0.01837056, -0.99903797, 2.43988996], [ 0.6904127, -0.72327596, -0.01421918, 0.18661254], [ 0.,     0.,     0.,     1.    ]])
        c_T_m[0:3,0:3] = c_T_m[0:3,0:3].T
        c_T_m[0:3,3] = -c_T_m[0:3,0:3] @ c_T_m[0:3,3]
        w_T_c = np.array([[1.0,0.0,-0.01,0.743],[0.0,-1.0,-0.007,0.082],[-0.01,0.007,-1.0,1.441],[0,0,0,1]])
        mocap_tf = w_T_c @ c_T_m
        camera_extrinsic = camera_extrinsic.reshape((4, 4))
        # mocap_tf = mocap_tf.reshape((4, 4))
        self.camera_pose = gtsam.Pose3(camera_extrinsic);
        self.mocap_pose = gtsam.Pose3(mocap_tf)
        self.offset = np.array([0, 0, 0.325 / 2.0 ]);
        self.cheb = ChebyshevTensegrityPoses(cheb_json)
        # self.offset = np.array([0.572, -0.050, 1.000, 0])

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
            # idx = int(l[0])
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
            for pose in data[di]:
                if pose is not None: 
                    pt0 = pose.transformFrom(self.offset)
                    pt1 = pose.transformFrom(-self.offset)
                    xi.append(pt0[0]);
                    yi.append(pt0[1]);
                    zi.append(pt0[2]);
                    xi.append(pt1[0]);
                    yi.append(pt1[1]);
                    zi.append(pt1[2]);
        ax.scatter(xi, yi, zi, marker=marker, c=color)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')



    def angle_err(self, pose_gt, pose_z):

        A = pose_gt.transformFrom(self.offset)
        B = pose_gt.transformFrom(-self.offset)
        zA = pose_z.transformFrom(self.offset)
        zB = pose_z.transformFrom(-self.offset)

        vgt = A - B
        vz = zA - zB
        err = np.rad2deg(np.arccos(1 - scipy_distance.cosine(vgt, vz)))
        return err;

    def compute_errors(self, gt_data, filename_prefix ):

        all_errors = []
        all_errors_cm = []
        angle_errs = []
        for gt_ti in gt_data:
            angles = []
            error = []
            error_cm = []
            p0, p1, p2 = self.cheb.compute_poses(gt_ti)
            poses_gts = gt_data[gt_ti]

            for z, gt in zip([p0, p1, p2], poses_gts):
                if gt is not None:
                    # print(f"gt {gt}")
                    err = np.abs(gtsam.Pose3.Logmap(gt.between(z)))
                    err_cm = np.abs(gt.translation() - z.translation())
                    error.append(err)
                    error_cm.append(err_cm)
                    angles.append(self.angle_err(gt, z));
                else:
                    error.append(np.array([np.nan]*6))
                    error_cm.append(np.array([np.nan]*3))
                    angles.append(np.nan);
                    # print(f"err {err}")
                # error.append(np.abs(gtsam.Pose3.Logmap(gt_p.between(z_p))))
                all_errors.append(error)
                all_errors_cm.append(error_cm)
                angle_errs.append(angles)
        all_errors = np.array(all_errors)
        all_errors_cm = np.array(all_errors_cm)
        angle_errs = np.array(angle_errs)

        mean_err = np.nanmean(all_errors, axis=(0,1))
        stdd_err = np.nanstd(all_errors, axis=(0,1))
        mean_err_cm = np.nanmean(all_errors_cm, axis=(0,1))
        stdd_err_cm = np.nanstd(all_errors_cm, axis=(0,1))
        mean_angle_errs = np.nanmean(angle_errs, axis=(0,1))
        stdd_angle_errs = np.nanstd(angle_errs, axis=(0,1))
            
        print(f"mean_err {mean_err}\n stdd_err {stdd_err}\n mean_angle_errs {mean_angle_errs}")
        all_errors_file = open(filename_prefix + "_all_errors.txt", 'w');
        all_errors_cm_file = open(filename_prefix + "_all_errors_cm.txt", 'w');
        angles_file = open(filename_prefix + "_all_angles.txt", 'w');
        stats_file = open(filename_prefix + "_stats.txt", 'w');

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

        # print(f"angle_errs: {angle_errs}")
        for errs in angle_errs:
            for e in errs:
                angles_file.write(str(e) + " ")
            angles_file.write("\n")

        for stat in [mean_err, stdd_err, mean_err_cm, stdd_err_cm, [mean_angle_errs], [stdd_angle_errs]]:
            for s in stat:
                stats_file.write(str(s) + " ")
        stats_file.write("\n")

        angles_file.close()
        all_errors_file.close()
        all_errors_cm_file.close()
        stats_file.close()

if __name__ == '__main__':

    argparse = argparse.ArgumentParser()

    argparse.add_argument('-g', '--gt_filename', help='GT', required=True)
    argparse.add_argument('-e', '--cheb_json', help='Estimation', required=True)
    argparse.add_argument('-o', '--output_prefix', help='Estimation', required=True)
    # argparse.add_argument('-d', '--npy_dir', help='Estimation', required=True)

    args = argparse.parse_args()

    gt_filename = args.gt_filename
    cheb_json = args.cheb_json
    output_prefix = args.output_prefix
    # npy_dir = args.npy_dir

    mdh = MocapDataHelper(cheb_json)
    # gt_filename = "/Users/Gary/pracsys/remotes/perception/tensegrity_ws/data/april28/10/snapshot/no_cable_gt_250909_232339.txt"
    # gt_filename = "/home/edgar/remotes/perception/tensegrity_ws/data/test/test_gt_250909_162758.txt"
    # estimation_filename = "/Users/Gary/pracsys/remotes/perception/tensegrity_ws/data/april28/10/snapshot/no_cable_estimated_endcap_250909_232345.txt"
    gt_data = mdh.read_poses_data(gt_filename);
    # z_data = mdh.read_poses_data(estimation_filename);
    # old_data = mdh.read_from_npy(npy_dir)

    # assert len(gt_data) == len(z_data), "Sizes not match!"
    # assert len(gt_data) == len(old_data), "Sizes not match!"

    gt_data = mdh.mocap_transform(gt_data)
    # z_data = mdh.camera_transform(z_data)
    # old_data = mdh.camera_transform(old_data)


    mdh.compute_errors(gt_data, output_prefix )
    # mdh.compute_errors(gt_data, old_data, output_prefix + "_old" )

    # estimation_filename = mdh.mocap_transform(estimation_filename)

    # estimation_data = mdh.read_estimation_data(estimation_filename)
    # print(f"gt_data: {gt_data}")
    # print(f"gt_data: {gt_data.shape}")
        
    # mdh.cmp_data(gt_data, estimation_data, "/tmp/errors.txt")

    # fig = plt.figure()

    # ax = fig.add_subplot(projection='3d')
    # mdh.cheb.plot(ax, 'x', 'r')
    # # mdh.plot(ax, gt_data, 'o', 'k')
    # # mdh.plot(ax, z_data, 'x', 'b')
    # # mdh.plot(ax, estimation_data, 'x', 'r')
    # plt.show()