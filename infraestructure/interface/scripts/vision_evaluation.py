import argparse
import glob
import math
import matplotlib.pyplot as plt
# import camera_info_manager
import numpy as np 
from numpy import linalg as LA
import copy

class VisionEvaluation(object):
    def __init__(self):
        # self.mocap_tf = np.array([0.000302, -0.706755, 0.707021, 0.572327, 0, 0.707234, 0.706968, -0.05,-0.999698, -0.000213504, 0.000213585,1.00017,0,0,0,1])
        self.c_T_m = np.array([[-0.72231893, -0.69031471, 0.04148439, -0.0911841 ], [-0.03982037, -0.01837056, -0.99903797, 2.43988996], [ 0.6904127, -0.72327596, -0.01421918, 0.18661254], [ 0.,     0.,     0.,     1.    ]])
        self.c_T_m[0:3,0:3] = self.c_T_m[0:3,0:3].T
        self.c_T_m[0:3,3] = -self.c_T_m[0:3,0:3] @ self.c_T_m[0:3,3]
        self.w_T_c = np.array([[1.0,0.0,-0.01,0.743],[0.0,-1.0,-0.007,0.082],[-0.01,0.007,-1.0,1.441],[0,0,0,1]])
        # self.mocap_tf = np.array([-0.706755, 0.0005153, -0.705954,0.572327, 0.70638, 0.000213504, -0.706755,-0.05, -0.000213596, -0.999396, -0.00051508, 1.00017, 0,0,0,1]);
        # self.mocap_tf = self.mocap_tf.reshape((4, 4))
        self.mocap_tf = self.w_T_c @ self.c_T_m
        self.offset = np.array([0.572, -0.050, 1.000, 0])

    def mocap_transform(self, pt):
        # ptp = self.mocap_tf @ pt + self.offset; 
        ptp = (self.mocap_tf @ pt)[0:3] + self.mocap_tf[0:3,3]; 
        # ptp[2] = ptp[2]
        # endcaps[idx] = []
        # return ptp[0:3]
        return ptp

    def read_estimation_data(self, filename):
        file = open(filename, 'r')

        data = {}
        for line in file:
            l = line.split()
            if l[0] == "#":
                continue
            # tot_endcaps = len(l) - 2 # first two are time and total
            # print(l)
            idx = int(l[1])
            data[idx] = []
            total_endcaps = int(l[2])
            for i in range(3, 3*total_endcaps+1, 3):
                # print(i)
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
                # print(f"ptp {ptp}")
                data[idx].append(pt)
            # data.append(endcaps);
        # print(data)
        return data;

    def read_gt_data(self, filename):
        file = open(filename, 'r')

        data = {}
        for line in file:
            l = line.split()
            tot_endcaps = len(l) - 2 # first two are time and idx

            # print(f" l {l}")
            # endcaps = {}
            # endcaps["t"] = float(l[0])
            # endcaps["idx"] = float(l[1])
            idx = int(l[1])
            data[idx] = []
            # tot_endcaps = int(l[1])
            # endcaps[idx] = []
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
                # endcaps[idx].append(pt)
                data[idx].append(pt)
                # print(f"ptp {ptp}")
            # data.append(endcaps);
        # print(data)
        return data;

    def plot(self, ax, data, marker, color):

        xi = []
        yi = []
        zi = []

        for di in data:

            # print(f"data[di] {data[di]}")
            # for mi in range(len(di)-1): 
                # print(mi)
            for pt in data[di]:
                xi.append(pt[0]);
                yi.append(pt[1]);
                zi.append(pt[2]);
                # print(f"m: {di[mi]}")
                # ax.scatter(di[mi][0],di[mi][1],di[mi][2], marker='o')
        ax.scatter(xi, yi, zi, marker=marker, c=color)
        print(f"total pts: {len(xi)}")
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z') 

    def get_data_error(self, gts, zs):

        diffs = []
        gt0 = gts.pop(0)
        # print(f"gt0 {gt0}")
        # print(f"z {zs}")
        for z in zs: #range(0, len(zs)-1):
            diffs.append(LA.norm(gt0 - z))
        if len(diffs) > 0:
            index_min = np.argmin(diffs)
            err0 = diffs[index_min]
            zs.pop(index_min)
        else:
            index_min = 0
            err0 = np.nan
        
        err1 = -1
        if len(gts) > 0:
            err1, _ = self.get_data_error(gts, zs)
        return err0, err1
        # p0_gt = gt[i]
        # p0_e = z[0]
        # p1_gt = gt[j]
        # p1_e = z[1]
        # # print(f"GT: p0 {p0_gt} p1 {p1_gt}")
        # # print(f"Z: p0 {p0_e} p1 {p1_e}")
        # p00_diff = LA.norm(p0_gt - p0_e)
        # p01_diff = LA.norm(p0_gt - p1_e)
        # p10_diff = LA.norm(p1_gt - p0_e)
        # p11_diff = LA.norm(p1_gt - p1_e)
        # p0_diff = min(p00_diff, p01_diff)
        # p1_diff = min(p10_diff, p11_diff)
        # print(f"p0 {p0_diff} p1 {p1_diff}")
        # return p0_diff, p1_diff

    def cmp_data(self, gt, estimation, ei, ej):

        # f = open(filename, 'w')
        errors = {}
        # print(f"estimation {estimation}")
        estimation_copy = copy.deepcopy(estimation)
        for idx in estimation_copy:
            # print(f"d {d}")
            # idx = d["idx"]
            # caps = d["caps"]
            # closest = self.find_closest(d["t"], gt)
            # green_closest = self.find_closest(d["t"], gt)
            # blue_closest = self.find_closest(d["t"], gt)
            gi = [gt[idx][ei], gt[idx][ej]]
            zi = estimation_copy[idx]
            # print(f"gi {gi}")
            # print(f"zi {zi}")
            # red_closest = self.get_as_list(red)
            # print(f"red_closest {red_closest} ")
            err0, err1 = self.get_data_error(gi, zi)
            # err2, err3 = self.get_data_error([d[2], d[3]], green_closest)
            # err4, err5 = self.get_data_error([d[4], d[5]], blue_closest)
            # if not idx in errors:
            # errors[idx] = []
            errors[idx] = [err0, err1]
            # errors[idx].append(err1)
            # print(f"errs: {idx} {errors[idx]}")
            # errors.append(err0)
            # errors.append(err1)
            # errors.append(err2)
            # errors.append(err3)
            # errors.append(err4)
            # errors.append(err5)
            # print(f"errors: {errors[-6:]} ")

            # if np.isnan(errors[-6:]).all():
            #     mean = np.nan
            #     std = np.nan
            # else:
            #     mean = np.nanmean(np.array(errors[-6:]))
            #     std = np.nanstd(np.array(errors[-6:]))

            # f.write(str(err0) + " " + str(err1) + " " + str(err2) + " ")
            # f.write(str(err3) + " " + str(err4) + " " + str(err5) + " ")
            # f.write(str(mean) + " " + str(std)  + " ")
            # f.write("\n")
            # print(f"p00 {p00_diff} p01 {p01_diff} p10 {p10_diff} p11 {p11_diff}")
        

        # mean = np.nanmean(np.array(errors))
        # std = np.nanstd(np.array(errors))
        # print(f"mean,std_dev {mean} {std}")
        # f.close();
        return errors;

    def errs_to_file(self, errs, filename):
        f = open(filename, 'w')
        for idx in errs:
            e = errs[idx]
            line = f"{idx} "
            line += f"{e[0]} {e[1]}"
            line += "\n"
            f.write(line);
        f.close()

    def err_stats(self, z0, z1, z2, err0, err1, err2, gt, filename):
        f = open(filename, 'w')
        l0 = []
        l1 = []
        l2 = []

        tot0 = 0
        tot1 = 0
        tot2 = 0
        for idx in z0:
            tot0 += len(z0[idx])
            # print(f"z {z0[idx]}")

        for idx in z1:
            tot1 += len(z1[idx])
            # print(f"z {z0[idx]}")

        for idx in z2:
            tot2 += len(z2[idx])
            # print(f"z {z0[idx]}")

        # print(f"{err0}")
        for idx in err0:
            for pt in err0[idx]:
                if not np.isnan(pt):
                    l0.append(pt)
        for idx in err1:
            for pt in err1[idx]:
                if not np.isnan(pt):
                    l1.append(pt)
        for idx in err2:
            for pt in err2[idx]:
                if not np.isnan(pt):
                    l2.append(pt)


        gt_tot = len(gt)*6
        # print(f"gt_tot: {gt_tot}")
        # tot0 = len(l0)
        total_len = tot0 + tot1 + tot2
        line = f"{tot0} {tot1} {tot2} {total_len} {len(gt)} {gt_tot} "
        for li in [l0, l1, l2]:
            mean = np.nanmean(np.array(li))
            std = np.nanstd(np.array(li))
            line += f"{mean} {std} "

        all_list = l0 + l1 + l2
        mean = np.nanmean(np.array(all_list))
        std = np.nanstd(np.array(all_list))
        line += f"{mean} {std}"
        f.write(line + "\n");
        f.close();
        print(line)

if __name__ == '__main__':

    argparse = argparse.ArgumentParser()

    argparse.add_argument('-t', '--gt_filename', help='GT', required=True)
    argparse.add_argument('-r', '--red_estimation_filename', help='Estimation', required=True)
    argparse.add_argument('-g', '--green_estimation_filename', help='Estimation', required=True)
    argparse.add_argument('-b', '--blue_estimation_filename', help='Estimation', required=True)
    argparse.add_argument('-f', '--file_prefix', help='Estimation', required=True)
    argparse.add_argument('-o', '--output_dir', help='Estimation', required=True)

    args = argparse.parse_args()

    gt_filename = args.gt_filename
    output_dir = args.output_dir
    file_prefix = args.file_prefix

    mdh = VisionEvaluation()
   
    gt_data = mdh.read_gt_data(gt_filename);
    red_estimation_data = mdh.read_estimation_data(args.red_estimation_filename)
    green_estimation_data = mdh.read_estimation_data(args.green_estimation_filename)
    blue_estimation_data = mdh.read_estimation_data(args.blue_estimation_filename)

    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # mdh.plot(ax, gt_data, 'o', 'k')
    # mdh.plot(ax, red_estimation_data, 'x', 'r')
    # mdh.plot(ax, green_estimation_data, 'x', 'g')
    # mdh.plot(ax, blue_estimation_data, 'x', 'b')
    # plt.show()

    red_errs = mdh.cmp_data(gt_data, red_estimation_data, 0, 1)
    green_errs = mdh.cmp_data(gt_data, green_estimation_data, 2, 3)
    blue_errs = mdh.cmp_data(gt_data, blue_estimation_data, 4, 5)

    # print(green_errs)
    mdh.errs_to_file(red_errs, output_dir + "/" + file_prefix + "_errors_red.txt")
    mdh.errs_to_file(green_errs, output_dir + "/" + file_prefix + "_errors_green.txt")
    mdh.errs_to_file(blue_errs, output_dir + "/" + file_prefix + "_errors_blue.txt")
    
    mdh.err_stats(red_estimation_data,green_estimation_data,blue_estimation_data,  red_errs, green_errs, blue_errs, gt_data, output_dir + "/" + file_prefix + "_error_stats.txt");
    # print(f"red_errs: {red_errs}")
    # mdh.cmp_data(gt_data, red_estimation_data, green_estimation_data, blue_estimation_data, "/tmp/errors.txt")

 