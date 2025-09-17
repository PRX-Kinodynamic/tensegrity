import argparse
import glob
import math
import matplotlib.pyplot as plt
# import camera_info_manager
import numpy as np 
from numpy import linalg as LA
import copy
import gtsam

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

    def read_bars_data(self, filename):
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
            total_bars = 3
            for i in range(2, 6*total_bars+1, 6):
                # print(i)
                x = float(l[i])
                y = float(l[i + 1])
                z = float(l[i + 2])
                qw = float(l[i + 3])
                qx = float(l[i + 4])
                qy = float(l[i + 5])
                qz = float(l[i + 6])
                bar = gtsam.Pose3(gtsam.Rot3(qw,qx,qy,qz), [x,y,z])

                data[idx].append(bar)
        return data;

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
            total_endcaps = 6
            for i in range(2, 3*total_endcaps+1, 3):
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
                    pt = np.array([np.nan, np.nan, np.nan])
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

    def get_dist_z(self, a, b, c, d):
        d0 = LA.norm(a - b)
        d1 = LA.norm(a - d)
        d2 = LA.norm(c - b)
        d3 = LA.norm(c - d)
        dists = sorted([d0,d1,d2,d3])
        return [dists[0], dists[1]]


    def cmp_bar_data(self, gt, z_data):
        errors = {}
        idx = 0
        data_remaining = True
        all_gt_dists = []
        all_z_dists = []
        offset = gtsam.Point3(0, 0, 0.325 / 2.0)
        Roffset = gtsam.Rot3(0.0, 0.0, 1.0, 0.0)
        while data_remaining:

            if idx in z_data:

                gt_r0 = gt[idx][0]
                gt_r1 = gt[idx][1]
                gt_g0 = gt[idx][2]
                gt_g1 = gt[idx][3]
                gt_b0 = gt[idx][4]
                gt_b1 = gt[idx][5]

                # print(gt_r0 , gt_g0)
                total_distance_gt = []

                # First triangle
                # total_distance_gt.append(LA.norm(gt_r0 - gt_g0)) # 0 2 red green
                # total_distance_gt.append(LA.norm(gt_g0 - gt_b0)) # 2 4 green blue
                # total_distance_gt.append(LA.norm(gt_b0 - gt_r0)) # 0 4 red blue
            
                # # Second triangle
                # total_distance_gt.append(LA.norm(gt_r1 - gt_g1)) # 1 3 red green
                # total_distance_gt.append(LA.norm(gt_g1 - gt_b1)) # 3 5 green blue
                # total_distance_gt.append(LA.norm(gt_b1 - gt_r1)) # 1 5 red blue

                # Long sides: Same color
                total_distance_gt.append(LA.norm(gt_r0 - gt_r1)) # 2 5 green blue
                total_distance_gt.append(LA.norm(gt_g0 - gt_g1)) # 0 3 red green
                total_distance_gt.append(LA.norm(gt_b0 - gt_b1)) # 1 4 red blue

                # Long side: 1 color shift 
                total_distance_gt.append(LA.norm(gt_r0 - gt_g1)) # 2 5 green blue
                total_distance_gt.append(LA.norm(gt_g0 - gt_b1)) # 0 3 red green
                total_distance_gt.append(LA.norm(gt_b0 - gt_r1)) # 1 4 red blue

                # Long side: 2 color shift 
                total_distance_gt.append(LA.norm(gt_r0 - gt_b1)) # 2 5 green blue
                total_distance_gt.append(LA.norm(gt_g0 - gt_r1)) # 0 3 red green
                total_distance_gt.append(LA.norm(gt_b0 - gt_g1)) # 1 4 red blue

                z_r = z_data[idx][0]
                z_g = z_data[idx][1]
                z_b = z_data[idx][2]

                # print(idx)
                z_r0 = z_r.transformFrom(offset)
                z_r1 = z_r.transformFrom(Roffset.rotate(offset))
                z_g0 = z_g.transformFrom(offset)
                z_g1 = z_g.transformFrom(Roffset.rotate(offset))
                z_b0 = z_b.transformFrom(offset)
                z_b1 = z_b.transformFrom(Roffset.rotate(offset))
                z_dists = []
            
                # First triangle
                # z_dists.append(LA.norm(z_r0 - z_g0)) # 0 2 red green
                # z_dists.append(LA.norm(z_g0 - z_b0)) # 2 4 green blue
                # z_dists.append(LA.norm(z_b0 - z_r0)) # 0 4 red blue
            
                # # Second triangle
                # z_dists.append(LA.norm(z_r1 - z_g1)) # 1 3 red green
                # z_dists.append(LA.norm(z_g1 - z_b1)) # 3 5 green blue
                # z_dists.append(LA.norm(z_b1 - z_r1)) # 1 5 red blue

                # Long sides: Same color
                z_dists.append(LA.norm(z_r0 - z_r1)) # 2 5 green blue
                z_dists.append(LA.norm(z_g0 - z_g1)) # 0 3 red green
                z_dists.append(LA.norm(z_b0 - z_b1)) # 1 4 red blue

                # Long side: 1 color shift 
                z_dists.append(LA.norm(z_r0 - z_g1)) # 2 5 green blue
                z_dists.append(LA.norm(z_g0 - z_b1)) # 0 3 red green
                z_dists.append(LA.norm(z_b0 - z_r1)) # 1 4 red blue

                # Long side: 2 color shift 
                z_dists.append(LA.norm(z_r0 - z_b1)) # 2 5 green blue
                z_dists.append(LA.norm(z_g0 - z_r1)) # 0 3 red green
                z_dists.append(LA.norm(z_b0 - z_g1)) # 1 4 red blue

                if not np.isnan(np.sum(total_distance_gt)):
                    all_gt_dists.append(np.array(total_distance_gt))
                    all_z_dists.append(np.array(z_dists))
                
            idx += 1;
            if not idx in gt:
                data_remaining = False

        return all_gt_dists, all_z_dists;

    def cmp_data(self, gt, z_data):
        errors = {}
        idx = 0
        data_remaining = True
        all_gt_dists = []
        all_z_dists = []
        while data_remaining:

            if not idx in z_data:
                z_data[idx] =  [np.array([np.nan, np.nan, np.nan])]*6

            gt_r0 = gt[idx][0]
            gt_r1 = gt[idx][1]
            gt_g0 = gt[idx][2]
            gt_g1 = gt[idx][3]
            gt_b0 = gt[idx][4]
            gt_b1 = gt[idx][5]

            z_r0 = z_data[idx][0]
            z_r1 = z_data[idx][1]
            z_g0 = z_data[idx][2]
            z_g1 = z_data[idx][3]
            z_b0 = z_data[idx][4]
            z_b1 = z_data[idx][5]

            # print(gt_r0 , gt_g0)
            total_distance_gt = []
            total_distance_gt.append(LA.norm(gt_r0 - gt_g0)) # 0 2 red green
            total_distance_gt.append(LA.norm(gt_g0 - gt_b0)) # 2 4 green blue
            total_distance_gt.append(LA.norm(gt_b0 - gt_r0)) # 0 4 red blue
            
            total_distance_gt.append(LA.norm(gt_r1 - gt_g1)) # 1 3 red green
            total_distance_gt.append(LA.norm(gt_g1 - gt_b1)) # 3 5 green blue
            total_distance_gt.append(LA.norm(gt_b1 - gt_r1)) # 1 5 red blue

            total_distance_gt.append(LA.norm(gt_g0 - gt_b1)) # 2 5 green blue
            total_distance_gt.append(LA.norm(gt_r0 - gt_g1)) # 0 3 red green
            total_distance_gt.append(LA.norm(gt_r1 - gt_b0)) # 1 4 red blue

            z_dists = []
            z_dists += self.get_dist_z(z_r0, z_g0, z_r1, z_g1)
            z_dists += self.get_dist_z(z_g0, z_b0, z_g1, z_b1)
            z_dists += self.get_dist_z(z_b0, z_r0, z_b1, z_r1)
            
            z_dists.append(min(LA.norm(z_g0 - z_b1), LA.norm(z_g1 - z_b0)))
            z_dists.append(min(LA.norm(z_r0 - z_g1), LA.norm(z_r1 - z_g0)))
            z_dists.append(min(LA.norm(z_r1 - z_b0), LA.norm(z_r0 - z_b1)))


            # print(total_distance_gt)
            # print(z_dists)
            idx += 1;
            if not idx in gt:
                data_remaining = False
            all_gt_dists.append(np.array(total_distance_gt))
            all_z_dists.append(np.array(z_dists))

        return all_gt_dists, all_z_dists;

    def errs_to_file(self, output_dir, prefix, all_gt_dists, all_z_dists, all_gt_bars, all_z_bars):
        errors = np.array(all_gt_dists) - np.array(all_z_dists)
        mean = np.nanmean(errors)
        std = np.nanstd(errors)

        errors_bar = np.array(all_gt_bars) - np.array(all_z_bars)
        mean_bar = np.nanmean(errors_bar)
        std_bar = np.nanstd(errors_bar)

        fgt = open(output_dir + "/" + prefix + "_gt.txt", 'w')
        fz = open(output_dir + "/" + prefix + "_z.txt", 'w')
        ferrors = open(output_dir + "/" + prefix + "_errors.txt", 'w')

        fbar_gt = open(output_dir + "/" + prefix + "_bar_gt.txt", 'w')
        fbar_z = open(output_dir + "/" + prefix + "_bar_z.txt", 'w')
        fbar_errors = open(output_dir + "/" + prefix + "_bar_errors.txt", 'w')

        fstats = open(output_dir + "/" + prefix + "_stats.txt", 'w')
        fstats.write(f"# mean std mean_bar std_bar\n")
        fstats.write(f"{mean} {std} {mean_bar} {std_bar}")
        fstats.close();
        print(f"{mean} {std} {mean_bar} {std_bar}")

        # print(all_gt_dists)
        for gt, zi, err in zip(all_gt_dists, all_z_dists, errors):
            line = ""
            for g in gt:
                fgt.write(str(g) + " ")
            for z in zi:
                fz.write(str(z) + " ")
            for e in err:
                ferrors.write(str(e) + " ")
            fgt.write("\n")
            fz.write("\n")
            ferrors.write("\n")
        
        for gt, zi, err in zip(all_gt_bars, all_z_bars, errors_bar):
            for g in gt:
                fbar_gt.write(str(g) + " ")
            for z in zi:
                fbar_z.write(str(z) + " ")
            for e in err:
                fbar_errors.write(str(e) + " ")
            fgt.write("\n")
            fz.write("\n")
            ferrors.write("\n")
        

        fgt.close();
        fz.close();
        ferrors.close();
        fbar_gt.close()
        fbar_z.close()
        fbar_errors.close()


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
    argparse.add_argument('-e', '--endcaps_filename', help='Estimation', required=True)
    argparse.add_argument('-b', '--bars_filename', help='Estimation', required=True)
    argparse.add_argument('-f', '--file_prefix', help='Estimation', required=True)
    argparse.add_argument('-o', '--output_dir', help='Estimation', required=True)

    args = argparse.parse_args()

    gt_filename = args.gt_filename
    output_dir = args.output_dir
    file_prefix = args.file_prefix

    mdh = VisionEvaluation()
   
    gt_data = mdh.read_gt_data(gt_filename);
    estimation_data = mdh.read_estimation_data(args.endcaps_filename)
    bars_data = mdh.read_bars_data(args.bars_filename)
    # green_estimation_data = mdh.read_estimation_data(args.green_estimation_filename)
    # blue_estimation_data = mdh.read_estimation_data(args.blue_estimation_filename)


    all_gt_dists, all_z_dists = mdh.cmp_data(gt_data, estimation_data)
    all_bar_gt_dists, all_bar_z_dists = mdh.cmp_bar_data(gt_data, bars_data)

    # print(all_bar_gt_dists)
    mdh.errs_to_file(output_dir, file_prefix, all_gt_dists, all_z_dists, all_bar_gt_dists, all_bar_z_dists)
    # mdh.errs_to_file(output_dir, file_prefix + "_bars", all_bar_gt_dists, all_bar_z_dists)

    # print(all_gt_dists)
    # print(all_z_dists)

    # error = np.array(all_gt_dists) - np.array(all_z_dists)
    # # print(error)
    # mean = np.nanmean(error)
    # std = np.nanstd(error)
    # print(f"{mean} {std}")
    # green_errs = mdh.cmp_data(gt_data, green_estimation_data, 2, 3)
    # blue_errs = mdh.cmp_data(gt_data, blue_estimation_data, 4, 5)

    # print(green_errs)
    # mdh.errs_to_file(red_errs, output_dir + "/" + file_prefix + "_errors_red.txt")
    # mdh.errs_to_file(green_errs, output_dir + "/" + file_prefix + "_errors_green.txt")
    # mdh.errs_to_file(blue_errs, output_dir + "/" + file_prefix + "_errors_blue.txt")
    
    # mdh.err_stats(red_estimation_data,green_estimation_data,blue_estimation_data,  red_errs, green_errs, blue_errs, gt_data, output_dir + "/" + file_prefix + "_error_stats.txt");
    # print(f"red_errs: {red_errs}")
    # mdh.cmp_data(gt_data, red_estimation_data, green_estimation_data, blue_estimation_data, "/tmp/errors.txt")

 