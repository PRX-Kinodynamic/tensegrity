import argparse
import glob
import math
import matplotlib.pyplot as plt
# import camera_info_manager
import numpy as np 
from numpy import linalg as LA

class MocapDataHelper(object):
    def __init__(self):
        # self.mocap_tf = np.array([0.000302, -0.706755, 0.707021, 0.572327, 0, 0.707234, 0.706968, -0.05,-0.999698, -0.000213504, 0.000213585,1.00017,0,0,0,1])
        
        self.mocap_tf = np.array([-0.706755, 0.0005153, -0.705954,0.572327, 0.70638, 0.000213504, -0.706755,-0.05, -0.000213596, -0.999396, -0.00051508, 1.00017, 0,0,0,1]);
        self.mocap_tf = self.mocap_tf.reshape((4, 4))
        self.offset = np.array([0.572, -0.050, 1.000, 0])

    def mocap_transform(self, pt):
        # ptp = self.mocap_tf @ pt + self.offset; 
        ptp = (self.mocap_tf @ pt)[0:3] + self.mocap_tf[0:3,3]; 
        ptp[2] = -ptp[2]
        # return ptp[0:3]
        return ptp

    def read_estimation_data(self, filename):
        file = open(filename, 'r')

        data = []
        for line in file:
            l = line.split()
            if l[0] == "#":
                continue
            tot_endcaps = len(l) - 2 # first two are time and idx

            endcaps = {}
            endcaps["t"] = float(l[0])
            for i in range(1, tot_endcaps, 3):
                x = l[i]
                y = l[i + 1]
                z = l[i + 2]
                if x == "NaN" or y == "NaN" or z == "NaN":
                    pt = [np.nan, np.nan, np.nan]
                else:
                    pt = np.array([x,y,z], np.float32) ;

                    # print(f"pt {pt}")
                    # ptp = tf @ pt;
                    # pt = self.mocap_transform(pt);
                endcaps[(i-1)//3] = pt
                # print(f"ptp {ptp}")
            data.append(endcaps);
        # print(data)
        return np.array(data);

    def read_gt_data(self, filename):
        file = open(filename, 'r')

        data = []
        for line in file:
            l = line.split()
            tot_endcaps = len(l) - 2 # first two are time and idx

            # print(f" l {l}")
            endcaps = {}
            endcaps["t"] = float(l[0])
            for i in range(2, tot_endcaps, 3):
                x = l[i]
                y = l[i + 1]
                z = l[i + 2]
                if x == "NaN" or y == "NaN" or z == "NaN":
                    pt = [np.nan, np.nan, np.nan]
                else:
                    pt_m = np.array([x,y,z,1], np.float32) / 1000;

                    # print(f"pt {pt}")
                    # ptp = tf @ pt;
                        # print(f"before {pt}")
                    pt = self.mocap_transform(pt_m);
                    # if i == 2:
                    # print(f" i {i} pt_m {pt_m} pt_w {pt}")
                endcaps[(i-2)//3] = pt
                # print(f"ptp {ptp}")
            data.append(endcaps);
        # print(data)
        return np.array(data);

    def plot(self, ax, data, marker, color):

        xi = []
        yi = []
        zi = []

        for di in data:
            for mi in di: 
                xi.append(di[mi][0]);
                yi.append(di[mi][1]);
                zi.append(di[mi][2]);
                # print(f"m: {di[mi]}")
                # ax.scatter(di[mi][0],di[mi][1],di[mi][2], marker='o')
        ax.scatter(xi, yi, zi, marker=marker, c=color)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

    def find_closest(self, ti, data):

        diff = 1000;
        min_data = []
        for d in data:
            curr_diff = math.fabs(d["t"] - ti)
            if curr_diff < diff:
                diff = curr_diff
                min_data = d;
        return min_data

    def get_data_error(self, gt, z, i, j):
        p0_gt = gt[i]
        p0_e = z[i]
        p1_gt = gt[j]
        p1_e = z[j]
        print(f"GT: p0 {p0_gt} p1 {p1_gt}")
        print(f"Z: p0 {p0_e} p1 {p1_e}")
        p00_diff = LA.norm(p0_gt - p0_e)
        p01_diff = LA.norm(p0_gt - p1_e)
        p10_diff = LA.norm(p1_gt - p0_e)
        p11_diff = LA.norm(p1_gt - p1_e)
        p0_diff = min(p00_diff, p01_diff)
        p1_diff = min(p10_diff, p11_diff)
        print(f"p0 {p0_diff} p1 {p1_diff}")
        return p0_diff, p1_diff

    def cmp_data(self, gt, estimation, filename):

        f = open(filename, 'w')
        errors = []
        for d in gt:
            closest = self.find_closest(d["t"], estimation)
            err0, err1 = self.get_data_error(d, closest, 0, 1)
            err2, err3 = self.get_data_error(d, closest, 2, 3)
            err4, err5 = self.get_data_error(d, closest, 4, 5)

            errors.append(err0)
            errors.append(err1)
            errors.append(err2)
            errors.append(err3)
            errors.append(err4)
            errors.append(err5)

            if np.isnan(errors[-6:]).all():
                mean = np.nan
                std = np.nan
            else:
                mean = np.nanmean(np.array(errors[-6:]))
                std = np.nanstd(np.array(errors[-6:]))

            f.write(str(err0) + " " + str(err1) + " " + str(err2) + " ")
            f.write(str(err3) + " " + str(err4) + " " + str(err5) + " ")
            f.write(str(mean) + " " + str(std)  + " ")
            f.write("\n")
            # print(f"p00 {p00_diff} p01 {p01_diff} p10 {p10_diff} p11 {p11_diff}")
            
        mean = np.nanmean(np.array(errors))
        std = np.nanstd(np.array(errors))
        print(f"mean,std_dev {mean} {std}")
        f.close();


if __name__ == '__main__':

    argparse = argparse.ArgumentParser()

    argparse.add_argument('-g', '--gt_filename', help='GT', required=True)
    argparse.add_argument('-e', '--estimation_filename', help='Estimation', required=True)

    args = argparse.parse_args()

    gt_filename = args.gt_filename
    estimation_filename = args.estimation_filename

    mdh = MocapDataHelper()
    # gt_filename = "/Users/Gary/pracsys/remotes/perception/tensegrity_ws/data/april28/10/snapshot/no_cable_gt_250909_232339.txt"
    # gt_filename = "/home/edgar/remotes/perception/tensegrity_ws/data/test/test_gt_250909_162758.txt"
    # estimation_filename = "/Users/Gary/pracsys/remotes/perception/tensegrity_ws/data/april28/10/snapshot/no_cable_estimated_endcap_250909_232345.txt"
    gt_data = mdh.read_gt_data(gt_filename);
    estimation_data = mdh.read_estimation_data(estimation_filename)
    # print(f"gt_data: {gt_data}")
    # print(f"gt_data: {gt_data.shape}")
        
    mdh.cmp_data(gt_data, estimation_data, "/tmp/errors.txt")

    # fig = plt.figure()

    # ax = fig.add_subplot(projection='3d')
    # mdh.plot(ax, gt_data, 'o', 'k')
    # mdh.plot(ax, estimation_data, 'x', 'r')
    # plt.show()