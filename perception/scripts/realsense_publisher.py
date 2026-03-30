#!/usr/bin/env python3

from __future__ import print_function

import os
import sys
import pyrealsense2 as rs
import numpy as np
import cv2
import json

import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo

# ==========================================================
# Write something similar in your control script

# from time_sync_demo.msg import StampedIndex
# ==========================================================

# from run_tensegrity_tracking import get_pose

class RealSenseCamera:

    def __init__(self, config=None):

        self.camera_frame = rospy.get_param("~camera_frame")
        self.camera_namespace = rospy.get_param("~camera_namespace")
        self.camera_name_topic = rospy.get_param("~camera_name_topic")

        if self.camera_namespace[-1] == "/":
            self.camera_namespace = self.camera_namespace[0:-1]       

        self.publisher_camera_info = rospy.Publisher(self.camera_name_topic, CameraInfo, queue_size=1, latch=True)

        if config is not None:
            self.config = config
        else:
            self.config = rs.config()
            # for D415
            self.config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
            self.config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

            self.bridge = CvBridge()

            # ros publisher
            self.color_im_pub = rospy.Publisher('/camera/rgb', Image, queue_size=6)
            self.depth_im_pub = rospy.Publisher('/camera/depth', Image, queue_size=6)

    def create_msg(self, intrinsic):
        msg = CameraInfo()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = self.camera_frame

        msg.width = intrinsic.width
        msg.height = intrinsic.height
        msg.distortion_model = "plumb_bob"

        fx = intrinsic.fx
        fy = intrinsic.fy
        cx = intrinsic.ppx
        cy = intrinsic.ppy

        intrinsic_mat = np.array([[fx, 0, cx],\
                                  [0, fy, cy],\
                                  [0,  0,  1]])
        rotation = np.array([[1, 0, 0],\
                             [0, 1, 0],\
                             [0, 0, 1]])

        projection = np.array([[fx, 0, cx, 0],\
                               [0, fy, cy, 0],\
                               [0,  0,  1, 0]])
        msg.D = []
        msg.K = np.ravel(intrinsic_mat).copy().tolist()
        msg.R = np.ravel(rotation).copy().tolist()
        msg.P = np.ravel(projection).copy().tolist()

        # print(msg)
        return msg

    def run(self, rate, output_dir=None):

        pipeline = rs.pipeline()
        profile = pipeline.start(self.config)

        device = profile.get_device()
        depth_sensor = device.first_depth_sensor()
        device.hardware_reset()
        
        # depth align to color
        align = rs.align(rs.stream.color)
        color_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
        color_intrinsics = color_profile.get_intrinsics()
        # print("camera intrinsics")
        print(color_intrinsics)

        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = int(round(1.0 / depth_sensor.get_depth_scale()))
        print("Depth Scale:", depth_scale)

        camera_info_msg = self.create_msg(color_intrinsics)
        rospy.set_param(self.camera_namespace + '/depth_scale', depth_scale)
        
        self.publisher_camera_info.publish(camera_info_msg)

        # camera_size = [color_intrinsics.width, color_intrinsics.height]
        # focal_length = [color_intrinsics.fx, color_intrinsics.fy]
        # optical_center = [color_intrinsics.ppx, color_intrinsics.ppy]
        # rospy.set_param(self.camera_namespace + '/size', camera_size)
        # rospy.set_param(self.camera_namespace + '/focal_length', focal_length)
        # rospy.set_param(self.camera_namespace + '/optical_center', optical_center)

        try:
            # cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            while not rospy.is_shutdown():
                frames = pipeline.wait_for_frames()
                aligned_frames = align.process(frames)
                depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()

                if not depth_frame or not color_frame:
                    continue

                # Convert images to numpy arrays
                color_image = np.asanyarray(color_frame.get_data())
                depth_image = np.asanyarray(depth_frame.get_data())

                timestamp = rospy.get_rostime()
                color_msg = self.bridge.cv2_to_imgmsg(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB), 'rgb8')
                depth_msg = self.bridge.cv2_to_imgmsg(depth_image, 'mono16')
                # color_msg = self.bridge.cv2_to_imgmsg(color_image, 'rgb8')

                color_msg.header.stamp = timestamp
                depth_msg.header.stamp = timestamp

                color_msg.header.frame_id = self.camera_frame
                depth_msg.header.frame_id = self.camera_frame

                self.color_im_pub.publish(color_msg)
                self.depth_im_pub.publish(depth_msg)

                # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
                # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

                # Stack both images horizontally
                # images = np.hstack((color_image, depth_colormap))

                
                rate.sleep()
        finally:
            pipeline.stop()

if __name__ == '__main__':
    rospy.init_node("realsense")
    rate = rospy.Rate(30)
    camera = RealSenseCamera()
    camera.run(rate)
