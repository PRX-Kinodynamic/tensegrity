import glob
import traceback

import os
import cv2
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import Image
from std_srvs.srv import Trigger
# from geometru_msgs.msg import Pose
from interface.msg import TensegrityBars
from interface.msg import TensegrityEndcaps
import tf

import numpy as np
from scipy.spatial.transform import Rotation as SciPyRot
import open3d as o3d


# TODO: Finish implementation if 
class EndcapToImage(object):
    def __init__(self):
        # self.folder = rospy.get_param("~dir", "")
        # self.topic_overlay = rospy.get_param("~topic_overlay", "")
        # self.topic_endcap_only = rospy.get_param("~topic_endcap_only", "")
        # self.topic_tensegrity_bars = rospy.get_param("~topic_tensegrity_bars", "")
        # self.max_images = rospy.get_param("~max_images", 1000)
        # self.world_frame = rospy.get_param("~world_frame", "world")
        # self.camera_frame = rospy.get_param("~camera_frame", "real_sense")


        # self.frequency = rospy.get_param("~frequency", 30)
        self.frequency = rospy.get_param("~frequency", 30)

        self.rgb_img_topic = rospy.get_param("~rgb_img_topic", "")
        self.depth_img_topic = rospy.get_param("~depth_img_topic", "")

        self.camera_info_topic = rospy.get_param("~camera_info_topic", "")
        
        self.red_endcaps_topic   = rospy.get_param("~red_endcaps_topic", "")
        self.green_endcaps_topic = rospy.get_param("~green_endcaps_topic", "")
        self.blue_endcaps_topic  = rospy.get_param("~blue_endcaps_topic", "")

        rod_mesh_file = rospy.get_param("~rod_mesh_file", "")
        
        assert os.path.isfile(rod_mesh_file), "Rod mesh not found!"
        rod_mesh_o3d = o3d.io.read_triangle_mesh(rod_mesh_file)
        self.rod_pcd = rod_mesh_o3d.sample_points_poisson_disk(1000)
        # self.rod_pcd = rod_mesh_o3d.sample_points_poisson_disk(100)

        self.red = [0, 0, 255] # using bgr8 encoding
        self.green = [0, 255, 0] # using bgr8 encoding
        self.blue = [255,0,0] # using bgr8 encoding

        self.bridge = CvBridge()

        self.images = []

        self.camera_intrinsic = None


        self.listener = tf.TransformListener()
        
        self.rgb_img_publisher = rospy.Publisher(self.rgb_img_topic, Image, queue_size=1, latch=True)
        self.depth_img_publisher = rospy.Publisher(self.depth_img_topic, Image, queue_size=1, latch=True)

        self.red_endcaps_subscriber = rospy.Subscriber(self.red_endcaps_topic, TensegrityEndcaps, self.endcap_callback, ("red"))
        self.green_endcaps_subscriber = rospy.Subscriber(self.green_endcaps_topic, TensegrityEndcaps, self.endcap_callback, ("green"))
        self.blue_endcaps_subscriber = rospy.Subscriber(self.blue_endcaps_topic, TensegrityEndcaps, self.endcap_callback, ("blue"))

        self.camera_info_subscriber = rospy.Subscriber(self.camera_info_topic, CameraInfo, self.camera_info_callback)

        self.timer = rospy.Timer(rospy.Duration(1.0 / self.frequency), self.update)

        self.red_endcap = None
        self.green_endcap = None
        self.blue_endcap = None

        # self.publisher_endcap_only = rospy.Publisher(self.topic_endcap_only, Image, queue_size=1, latch=True)

        self.R_extrinsic = None
        self.t_extrinsic = None

    def endcap_callback(self, msg, args):
        color = args[0]
        if color == "red":
            self.red_endcap = msg.endcaps
        elif color == "green":
            self.green_endcap = msg.endcaps
        elif color == "blue":
            self.blue_endcap = msg.endcaps

    def camera_info_callback(self, msg):
        self.camera_intrinsic = np.array(msg.K, dtype=np.float32).reshape((3,3))
        print(f"camera_intrinsic: {self.camera_intrinsic}")

    def project_points_to_image(self, img, points, color):
        # print(img.shape)
        if self.camera_intrinsic is None:
            rospy.loginfo_once("Camera info has not been received. Subsribe to correct topic?");

        im_h, im_w, _ = img.shape
        fx, fy = self.camera_intrinsic[0, 0], self.camera_intrinsic[1, 1]
        cx, cy = self.camera_intrinsic[0, 2], self.camera_intrinsic[1, 2]

        for i in range(points.shape[0]):
            # TODO: Use matrix operations directly?
            x = int(np.round((points[i, 0] * fx / points[i, 2]) + cx))
            y = int(np.round((points[i, 1] * fy / points[i, 2]) + cy))
            if points[i, 2] < 0:
                continue;
            if 0 < x < im_w and 0 < y < im_h:
                img[y, x] = color
        return img

    
    def update(self, event):
        if self.red_endcap is None and self.green_endcap is None and self.blue_endcap is None:
            return

        if self.format == "color":
            cv_image = cv2.imread(img, cv2.IMREAD_COLOR)
            img_msg = self.bridge.cv2_to_imgmsg(cv_image, 'bgr8')
        elif self.format == "gray":
            cv_image = cv2.imread(img, cv2.IMREAD_UNCHANGED)
            img_msg = self.bridge.cv2_to_imgmsg(cv_image, 'passthrough')
        else:
            rospy.logwarn("Format not supported. Choose 'color' or 'gray'")
            exit(-1)

        img_msg.header.stamp = event.current_real
            
        self.image_idx +=1

        self.image_pub.publish(img_msg)
        self.image_list.pop(0)
        if self.loop:
            self.image_list.append(img)

if __name__ == '__main__':
    rospy.init_node("EndcapsToImage")
    image_folder_publisher = EndcapToImage()
    rospy.spin()