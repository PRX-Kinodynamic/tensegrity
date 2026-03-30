import glob
import traceback

import os
import cv2
import rospy
import tf
import tf2_ros
import geometry_msgs
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import numpy as np

class TfFromYamlNode(object):

    def __init__(self):

        self.pose = rospy.get_param("~pose")

        self.child_frame = rospy.get_param('~child_frame') 
        self.parent_frame = rospy.get_param('~parent_frame') 

        self.extrinsics_to_tf()

    def extrinsics_to_matrix(self):

        pose = np.zeros((4,4))
        for i in range(3):
          for j in range(4):
            pose[i, j] = float(self.pose[4 * i + j]);
        pose[3, 3] = 1.0
        return pose

    def extrinsics_to_tf(self):

        mat = self.extrinsics_to_matrix()
        t = mat[0:3,3]
        R = mat[0:3,0:3]

        mat_inv = np.zeros_like(mat)
        mat_inv[0:3,0:3] = R.T

        mat_inv[0:3,3] = (R.T @ t.reshape((3,1))).T
        mat_inv[3, 3] = 1.0

        tinv = mat_inv[0:3,3]
        Rinv = mat_inv[0:3,0:3]

        self.broadcaster = tf2_ros.StaticTransformBroadcaster()
        static_transformStamped = geometry_msgs.msg.TransformStamped()

        static_transformStamped.header.stamp = rospy.Time.now()
        static_transformStamped.child_frame_id = self.child_frame 
        static_transformStamped.header.frame_id = self.parent_frame

        static_transformStamped.transform.translation.x = t[0]
        static_transformStamped.transform.translation.y = t[1]
        static_transformStamped.transform.translation.z = t[2]

        # print(f"mat: {mat}")
        # print(f"R: {R}")
        # quat = tf.transformations.quaternion_from_matrix(mat_inv)
        quat = tf.transformations.quaternion_from_matrix(mat)
        static_transformStamped.transform.rotation.x = quat[0]
        static_transformStamped.transform.rotation.y = quat[1]
        static_transformStamped.transform.rotation.z = quat[2]
        static_transformStamped.transform.rotation.w = quat[3]

        self.broadcaster.sendTransform(static_transformStamped)


if __name__ == '__main__':
    rospy.init_node("TfFromYamlNode")
    ros_node = TfFromYamlNode()
    rospy.spin()