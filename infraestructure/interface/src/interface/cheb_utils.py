import os
import json
import math
import gtsam
import numpy as np

class ChebyshevTensegrityPoses:
    def __init__(self, json_file):
        f = open(json_file, 'r')
        j = json.load(f)
        self.matrices = [None]*3

        self.matrices[0] = np.array(j['0'])
        self.matrices[1] = np.array(j['1'])
        self.matrices[2] = np.array(j['2'])

        self.N = int(j['N'])
        self.a = float(j['a'])
        self.b = float(j['b'])
        self.offset = np.array(j['offset'])

        print(f"a {self.a} b {self.b}")
    # Convert to [a,b] to [-1,1]
    def scale(self, x):
        t1 = -1.0
        t2 = 1.0
        return ((t2 - t1) * (x - self.a) / (self.b - self.a)) + t1;

    def calc_weights(self, x):
        xp = self.scale(x)
        # return gtsam.Chebyshev2.CalculateWeights(self.N, xp, -1, 1)
        return gtsam.Chebyshev2Basis.CalculateWeights(self.N, xp)

    def compute_poses(self, ti, normalize = False):
        if normalize:
            ti = self.a + (self.b - self.a) * ti
        ti = min(max(ti, self.a), self.b)

        wi = self.calc_weights(ti)
        p0 = gtsam.Pose3.Expmap(self.matrices[0] @ wi)
        p1 = gtsam.Pose3.Expmap(self.matrices[1] @ wi)
        p2 = gtsam.Pose3.Expmap(self.matrices[2] @ wi)
        # print(f"ti {ti} a: {self.a} b: {self.b}")
        return p0, p1, p2

    def compute_velocities(self, ti, normalize = False):
        a = self.a
        b = self.b
        if normalize:
            ti = self.a + (self.b - self.a) * ti
            a = -1
            b = 1
        Dn = gtsam.Chebyshev2.DerivativeWeights(self.N, ti, a, b)
        wi = self.calc_weights(ti)
        # print(f"Dn {Dn.shape} wi {wi.shape} mat: {self.matrices[0].shape}")
        v0 = self.matrices[0] @ wi
        v1 = self.matrices[1] @ wi
        v2 = self.matrices[2] @ wi
        # print(f"ti: {ti} a: {a} b: {b} N: {self.N} ")
        return v0, v1, v2

    def plot(self, ax, marker, color, dt = 0.1):

        xi = []
        yi = []
        zi = []
        # for ti in np.arange(0, 1, dt):
        for ti in np.arange(self.a, self.b, dt):
            p0, p1, p2 = self.compute_poses(ti)

            # print(f"p0 {p0}")
            # p0A = p0.translation()
            # p1A = p1.translation()
            # p2A = p2.translation()
            p0A = p0.transformFrom(self.offset)
            p0B = p0.transformFrom(-self.offset)
            p1A = p1.transformFrom(self.offset)
            p1B = p1.transformFrom(-self.offset)
            p2A = p2.transformFrom(self.offset)
            p2B = p2.transformFrom(-self.offset)

            for p in [p0A, p0B, p1A, p1B, p2A, p2B]:
            # for p in [p1A, p1A, p2A]:
                xi.append(p[0]);
                yi.append(p[1]);
                zi.append(p[2]);
                # xi.append(pt1[0]);
                # yi.append(pt1[1]);
                # zi.append(pt1[2]);
        ax.scatter(xi, yi, zi, marker=marker, c=color)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.axis('equal')