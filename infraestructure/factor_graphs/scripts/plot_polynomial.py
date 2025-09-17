import matplotlib.pyplot as plt
import gtsam
import gtsam.utils.plot as gtsam_plot
import numpy as np
import argparse

def compute_traj(params):
    xp = np.linspace(-1, 1., 10)
    W = gtsam.FourierBasis.WeightMatrix(3, xp)
    traj_cheb = W @ params.T
    poses = []
    for ti in traj_cheb:
        p = gtsam.Pose3.Expmap(ti)
        poses.append(p)
    return poses

def plot_traj(poses):

    for p in poses:
        gtsam_plot.plot_pose3(0, p)
    plt.axis('equal')
    plt.show()

def read_polynomials_file(filename, N):
    f = open(filename, 'r')
    polys = []
    for l in f:
        lp = l.split()
        p = np.array(lp, np.float64)
        p = np.reshape(p, (6, N))
        print(p)
        polys.append(p)
    return polys

if __name__ == "__main__":
    argparse = argparse.ArgumentParser()
    argparse.add_argument('-f', '--filename', help='polynomials in file')
    # argparse.add_argument('-p', '--poly', help='polynomial', nargs='+')
    argparse.add_argument('-n', '--degree', help='degree', required=True)
    args = argparse.parse_args()
    
    N = int(args.degree)
    polys = read_polynomials_file(args.filename, N);

    # p = np.array(args.poly, np.float64)
    # p = np.reshape(p, (6, N))
    # print(p, p.shape, N)
    poses = []
    for p in polys:
        poses = poses + compute_traj(p)
    plot_traj(poses)


