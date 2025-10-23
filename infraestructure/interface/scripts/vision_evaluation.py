import argparse
import glob
import math
import matplotlib.pyplot as plt
# import scipy.spatial.distance as scipy_distance
import numpy as np 
import gtsam
from numpy import linalg as LA
# from interface.cheb_utils import ChebyshevTensegrityPoses
import yaml

class MocapDataHelper(object):
    def __init__(self):
        # camera_extrinsic = np.array([1.0, 0.0, -0.01, 0.743, 0.0, -1.0, -0.007, 0.082, -0.01, 0.007, -1.0, 1.441, 0.0, 0.0, 0.0, 1.0]);
        # camera_extrinsics: [1.0,0.0,-0.01,0.743,0.0,-1.0,-0.007,0.082,-0.01,0.007,-1.0,1.441]
        # mocap_tf = np.array([-0.706755, 0.0005153, -0.705954,0.572327, 0.70638, 0.000213504, -0.706755,-0.05, -0.000213596, -0.999396, -0.00051508, 1.00017, 0,0,0,1]);
        c_T_m = np.array([[-0.72231893, -0.69031471, 0.04148439, -0.0911841 ], [-0.03982037, -0.01837056, -0.99903797, 2.43988996], [ 0.6904127, -0.72327596, -0.01421918, 0.18661254], [ 0.,     0.,     0.,     1.    ]])
        # c_T_m = np.array([[-0.71982338, -0.69235535,  0.04998369, -0.09336075], [-0.04100413, -0.02947068, -0.99872426,  2.16381353], [ 0.69294514, -0.72095461, -0.00717574,  0.1893245 ], [ 0.        ,  0.        ,  0.        ,  1.        ]])
        c_T_m[0:3,0:3] = c_T_m[0:3,0:3].T
        c_T_m[0:3,3] = -c_T_m[0:3,0:3] @ c_T_m[0:3,3]
        w_T_c = np.array([[1.0,0.0,-0.01,0.743],[0.0,-1.0,-0.007,0.082],[-0.01,0.007,-1.0,1.441],[0,0,0,1]])
        mocap_tf = w_T_c @ c_T_m
        # print(w_T_c)
        # print(c_T_m)
        # print(mocap_tf)
        # camera_extrinsic = camera_extrinsic.reshape((4, 4))
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
        # print(mocap_tf)
        f.close();

    def camera_transform(self, data):
        for di in data:
            for idx in range(3):
                if data[di][idx] is not None: 
                    data[di][idx] = self.camera_pose * data[di][idx];
        return data

    def mocap_transform(self, data):

        for di in data:
            for idx in data[di]:
                # if pair is not None: 
                # print(f"pose: {data[di][idx][1]}")
                # print(f"self.mocap_pose: {self.mocap_pose}")
                for i in range(len(data[di][idx])):
                    data[di][idx][i] = self.mocap_pose.transformFrom(data[di][idx][i]);

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

    def read_endcap_data(self, filename):
        file = open(filename, 'r')

        data = {}
        for line in file:
            l = line.split()
            if l[0] == "#":
                continue
            # idx = int(l[0])
            ti = float(l[0])
            # idx = float(l[0])
            color_id = int(l[2])
            pt = np.array([l[3], l[4], l[5]], np.float32)

            if not ti in data:
                data[ti] = {}
            if not color_id in data[ti]:
                data[ti][color_id] = []

        # else:
            data[ti][color_id].append( pt )

        return data

    def plot(self, ax, data, marker, color):

        xi = []
        yi = []
        zi = []

        for di in data:
            for idx in data[di]:
                for i in range(len(data[di][idx])):
                    pt0 = data[di][idx][i]
                    xi.append(pt0[0]);
                    yi.append(pt0[1]);
                    zi.append(pt0[2]);
        ax.scatter(xi, yi, zi, marker=marker, c=color)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

    # def angle_err(self, pose_gt, pose_z):

    #     A = pose_gt.transformFrom(self.offset)
    #     B = pose_gt.transformFrom(-self.offset)
    #     zA = pose_z.transformFrom(self.offset)
    #     zB = pose_z.transformFrom(-self.offset)

    #     vgt = A - B
    #     vz = zA - zB
    #     err = np.rad2deg(np.arccos(1 - scipy_distance.cosine(vgt, vz)))
    #     return err;

    def min_error(self, endcap_gt, endcaps_z, ignore_idx=-1):

        if np.isnan(endcap_gt).any():
            return np.nan, -1
        dist_min = 10000
        idx_min = -1

        idx = 0
        for zi in endcaps_z:
            if idx == ignore_idx:
                continue;
            d = LA.norm(zi - endcap_gt)
            if dist_min > d:
                dist_min = d
                idx_min = idx
            idx += 1

        if idx_min == -1:
            return np.nan, -1
        return dist_min, idx_min

    def compute_errors(self, gt_data, z_data, filename_prefix):

        endcap_misses = [0]*3
        frame_misses = 0
        gt_miss = [0]*3 
        all_errors = []

        for gt_ti in gt_data:
            for ei in gt_data[gt_ti]:
                if np.isnan(gt_data[gt_ti][ei][0]).any():
                    gt_miss[ei//2] += 1

        for gt_ti in gt_data:
            error = []

            if not gt_ti in z_data:
                frame_misses += 1
                continue; 

            zi = []

            for i in range(3):
                gt0 = gt_data[gt_ti][2*i + 0][0]
                gt1 = gt_data[gt_ti][2*i + 1][0]

                if not i in z_data[gt_ti]:
                    endcap_misses[i] += 2
                    # error.append(np.nan )
                    # error.append(np.nan )
                    error.append([np.nan, np.nan] )
                    continue
                err0, idx_0 = self.min_error(gt0, z_data[gt_ti][i])
                err1, idx_1 = self.min_error(gt1, z_data[gt_ti][i])
                # error.append(err)
                # if(idx_0 == idx_1):
                if len(z_data[gt_ti][i]) == 1:
                    # if err0 < err1:
                    #     error.append(np.abs(gt0 - z_data[gt_ti][i][idx_0]))
                    # else:
                    #     error.append(np.abs(gt1 - z_data[gt_ti][i][idx_1]))

                    # error.append(np.nan )
                    error.append([min(err0, err1), np.nan])
                    endcap_misses[i] += 1
                    # error.append(np.nan)
                else:
                    error.append([err0, err1])
                    # error.append(err0)
                    # error.append(err1)
                    # print(f"gt0 {gt0} z: {z_data[gt_ti][i][idx_0]} diff: {gt0 - z_data[gt_ti][i][idx_0]}")
                    # error.append(np.abs(gt0 - z_data[gt_ti][i][idx_0]))
    
            # print(error)
            all_errors.append(error)
        #         all_errors_cm.append(error_cm)
        #         angle_errs.append(angles)
        # all_errors = np.array(all_errors)
        # all_errors_cm = np.array(all_errors_cm)
        # angle_errs = np.array(angle_errs)
        # print(all_errors)
        mean_err = np.nanmean(all_errors, axis=(0,2))
        stdd_err = np.nanstd(all_errors, axis=(0,2))
        # mean_err_cm = np.nanmean(all_errors_cm, axis=(0,1))
        # stdd_err_cm = np.nanstd(all_errors_cm, axis=(0,1))
        # mean_angle_errs = np.nanmean(angle_errs, axis=(0,1))
        # stdd_angle_errs = np.nanstd(angle_errs, axis=(0,1))
            
        print(f"mean_err {mean_err}\n stdd_err {stdd_err}")
        print(f"frame_misses {frame_misses}\n endcap_misses {endcap_misses}\n gt_miss {gt_miss}")
        # all_errors_file = open(filename_prefix + "_all_errors.txt", 'w');
        # all_errors_cm_file = open(filename_prefix + "_all_errors_cm.txt", 'w');
        # angles_file = open(filename_prefix + "_all_angles.txt", 'w');
        stats_file = open(filename_prefix + "_stats.txt", 'w');

        # # print(f"all_errors {all_errors}")
        # for errs in all_errors:
        #     for e in errs:
        #         for xi in e:
        #             all_errors_file.write(str(xi) + " ")
        #         all_errors_file.write("\n")

        # for errs in all_errors_cm:
        #     for e in errs:
        #         for xi in e:
        #             all_errors_cm_file.write(str(xi) + " ")
        #         all_errors_cm_file.write("\n")

        # # print(f"angle_errs: {angle_errs}")
        # for errs in angle_errs:
        #     for e in errs:
        #         angles_file.write(str(e) + " ")
        #     angles_file.write("\n")

        for stat in [mean_err, stdd_err, [frame_misses], endcap_misses, gt_miss ]:
            for s in stat:
                stats_file.write(str(s) + " ")
        stats_file.write("\n")

        # angles_file.close()
        # all_errors_file.close()
        # all_errors_cm_file.close()
        # stats_file.close()

if __name__ == '__main__':

    argparse = argparse.ArgumentParser()

    argparse.add_argument('-g', '--gt_filename', help='GT', required=True)
    argparse.add_argument('-e', '--estimates_file', help='Estimation', required=True)
    argparse.add_argument('-o', '--output_prefix', help='Estimation', required=True)
    argparse.add_argument('-c', '--camera_filename', help='Estimation', required=True)
    # argparse.add_argument('-d', '--npy_dir', help='Estimation', required=True)

    args = argparse.parse_args()

    gt_filename = args.gt_filename
    estimates_file = args.estimates_file
    output_prefix = args.output_prefix
    # npy_dir = args.npy_dir

    mdh = MocapDataHelper()
    mdh.read_camera_param(args.camera_filename)

    # gt_filename = "/Users/Gary/pracsys/remotes/perception/tensegrity_ws/data/april28/10/snapshot/no_cable_gt_250909_232339.txt"
    # gt_filename = "/home/edgar/remotes/perception/tensegrity_ws/data/test/test_gt_250909_162758.txt"
    # estimation_filename = "/Users/Gary/pracsys/remotes/perception/tensegrity_ws/data/april28/10/snapshot/no_cable_estimated_endcap_250909_232345.txt"
    gt_data = mdh.read_endcap_data(gt_filename);
    z_data = mdh.read_endcap_data(estimates_file);
    # old_data = mdh.read_from_npy(npy_dir)

    # assert len(gt_data) == len(z_data), "Sizes not match!"
    # assert len(gt_data) == len(old_data), "Sizes not match!"

    gt_data = mdh.mocap_transform(gt_data)
    # z_data = mdh.camera_transform(z_data)
    # old_data = mdh.camera_transform(old_data)


    mdh.compute_errors(gt_data, z_data, output_prefix )
    # mdh.compute_errors(gt_data, old_data, output_prefix + "_old" )
    # def compute_errors(self, gt_data, z_data ):

    # estimation_filename = mdh.mocap_transform(estimation_filename)

    # estimation_data = mdh.read_estimation_data(estimation_filename)
    # print(f"gt_data: {gt_data}")
    # print(f"gt_data: {gt_data.shape}")
    # for ti in z_data:
    #     print(z_data[ti])
    # # mdh.cmp_data(gt_data, estimation_data, "/tmp/errors.txt")

    # fig = plt.figure()

    # ax = fig.add_subplot(projection='3d')
    # # mdh.plot(ax, 'x', 'r')
    # mdh.plot(ax, gt_data, 'o', 'k')
    # mdh.plot(ax, z_data, 'x', 'b')
    # # mdh.plot(ax, estimation_data, 'x', 'r')
    # plt.show()