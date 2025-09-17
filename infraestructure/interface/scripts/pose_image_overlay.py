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
import tf

import numpy as np
from scipy.spatial.transform import Rotation as SciPyRot
import open3d as o3d

class PoseImageOverlay(object):
    def __init__(self):
        # self.folder = rospy.get_param("~dir", "")
        self.rgb_img_topic = rospy.get_param("~rgb_img_topic", "")
        self.topic_overlay = rospy.get_param("~topic_overlay", "")
        self.topic_endcap_only = rospy.get_param("~topic_endcap_only", "")
        self.topic_tensegrity_bars = rospy.get_param("~topic_tensegrity_bars", "")
        self.max_images = rospy.get_param("~max_images", 1000)
        self.camera_info_topic = rospy.get_param("~camera_info_topic", 1000)
        self.world_frame = rospy.get_param("~world_frame", "world")
        self.camera_frame = rospy.get_param("~camera_frame", "real_sense")

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

        self.camera_info_subscriber = rospy.Subscriber(self.camera_info_topic, CameraInfo, self.camera_info_callback)

        self.rgb_img_subscriber = rospy.Subscriber(self.rgb_img_topic, Image, self.img_callback)
        self.bars_subscriber = rospy.Subscriber(self.topic_tensegrity_bars, TensegrityBars, self.bars_callback)

        self.publisher_overlay = rospy.Publisher(self.topic_overlay, Image, queue_size=1, latch=True)
        self.publisher_endcap_only = rospy.Publisher(self.topic_endcap_only, Image, queue_size=1, latch=True)
        # self.timer = rospy.Timer(rospy.Duration(self.frequency), self.update)

        self.R_extrinsic = None
        self.t_extrinsic = None

    def get_camera_extrinsic(self):
        try:
            (trans,rot) = self.listener.lookupTransform(self.camera_frame, self.world_frame, rospy.Time(0))
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            rospy.loginfo("Tf didn't return extrinsic transform");

            return False

        rot = SciPyRot.from_quat(rot, scalar_first=False) # W is last for some reason...

        self.t_extrinsic = np.array(trans)
        self.R_extrinsic = rot.as_matrix()

        return True


    def camera_info_callback(self, msg):
        self.camera_intrinsic = np.array(msg.K, dtype=np.float32).reshape((3,3))
        print(f"camera_intrinsic: {self.camera_intrinsic}")

    def find_closest_image(self, stamp):   
        closest_time = 100000
        closest_image = None
        for img in self.images:
            ti = img.header.stamp - stamp
            secs = ti.to_sec()
            if secs < closest_time:
                closest_time = secs
                closest_image = img
        return closest_image

    def img_callback(self, image_msg):

        if len(self.images) >= self.max_images:
            self.images.pop(0)
        self.images.append(image_msg)

    def pose_msg_to_np(self, pose_msg):
        x = pose_msg.position.x
        y = pose_msg.position.y
        z = pose_msg.position.z

        qw = pose_msg.orientation.w
        qx = pose_msg.orientation.x
        qy = pose_msg.orientation.y
        qz = pose_msg.orientation.z


        rot = SciPyRot.from_quat([qw, qx, qy, qz], scalar_first=True) # W is first
        # quat = tf.transformations.quaternion_from_matrix(mat)
        # Rmat = tf.transformations.quaternion_matrix([qw, qx, qy, qz])
        
        tr = np.array([x,y,z]);
        Rmat = rot.as_matrix()
        return Rmat, tr

    def bars_callback(self, bars_msg):

        img_msg = self.find_closest_image(bars_msg.header.stamp);
        if img_msg is None:
            # print("Image is None")
            return;
        img = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')

        img_overlapping = np.copy(img)
        img_endcaps_only = np.zeros_like(img)


        Rmat_red, tr_red = self.pose_msg_to_np(bars_msg.bar_red)
        Rmat_green, tr_green = self.pose_msg_to_np(bars_msg.bar_green)
        Rmat_blue, tr_blue = self.pose_msg_to_np(bars_msg.bar_blue)

        # print(f"points {np.asarray(self.rod_pcd.points)}")
        # print(f"Rmat_red: {Rmat_red} tr_red: {tr_red}")
        if self.R_extrinsic is None or self.t_extrinsic is None:
            success = self.get_camera_extrinsic()
            if not success:
                return
        # print(f"R ext: {self.R_extrinsic}")
        # print(f"t ext: {self.t_extrinsic}")

        pcd_red = self.transform_pcd_pose(Rmat_red, tr_red)
        pcd_green = self.transform_pcd_pose(Rmat_green, tr_green)
        pcd_blue = self.transform_pcd_pose(Rmat_blue, tr_blue)

        # print(f"pcd_red {pcd_red.shape}")
        # TODO: Consider the z value to define which color goes on top
        img_overlapping = self.project_points_to_image(img_overlapping, pcd_red, self.red)
        img_overlapping = self.project_points_to_image(img_overlapping, pcd_green, self.green)
        img_overlapping = self.project_points_to_image(img_overlapping, pcd_blue, self.blue)

        img_endcaps_only = self.project_points_to_image(img_endcaps_only, pcd_red, self.red)
        img_endcaps_only = self.project_points_to_image(img_endcaps_only, pcd_green, self.green)
        img_endcaps_only = self.project_points_to_image(img_endcaps_only, pcd_blue, self.blue)


        msg_overlay = self.bridge.cv2_to_imgmsg(img_overlapping, 'bgr8')
        msg_endcaps_only = self.bridge.cv2_to_imgmsg(img_endcaps_only, 'bgr8')

        self.publisher_overlay.publish(msg_overlay)
        self.publisher_endcap_only.publish(msg_endcaps_only)
        # self.overlay_pcd_in_image(img, self.red, Rmat_red, tr_red)
        # self.overlay_pcd_in_image(img, self.green, Rmat_green, tr_green)
        # self.overlay_pcd_in_image(img, self.blue, Rmat_blue, tr_blue)

    def project_points_to_image(self, img, points, color):
        # print(img.shape)
        if self.camera_intrinsic is None:
            rospy.loginfo_once("Camera info has not been received. Subsribe to correct topic?");

        im_h, im_w, _ = img.shape
        fx, fy = self.camera_intrinsic[0, 0], self.camera_intrinsic[1, 1]
        cx, cy = self.camera_intrinsic[0, 2], self.camera_intrinsic[1, 2]

        # print(f"im_h {im_h}")
        # print(f"im_w {im_w}")
        for i in range(points.shape[0]):
            # TODO: Use matrix operations directly?
            x = int(np.round((points[i, 0] * fx / points[i, 2]) + cx))
            y = int(np.round((points[i, 1] * fy / points[i, 2]) + cy))
            # if x < 0 or x >= im_w or y < 0 or y >= im_h:
                # continue
            # print(f"Val: ({y}, {x})")
            if points[i, 2] < 0:
                continue;
            if 0 < x < im_w and 0 < y < im_h:
                # print(f"img: ({y}, {x})")
                # img[x, y] = color
                img[y, x] = color
        return img

    def transform_pcd_pose(self, rotation, translation):
        # vis_im_overlapping = np.copy(rgb_image)
        # vis_im_endcap_only = np.zeros_like(rgb_image)
        # cam_intr = np.asarray(self.data_cfg['cam_intr'])

        # TODO: Simplify? (R @ pt.T + t).T = 
        # print(f"rotation {rotation.shape}")
        # print(f"rod_pcd {np.asarray(self.rod_pcd.points).T.shape}")
        # print(f"@ {(rotation @ np.asarray(self.rod_pcd.points).T).shape}")
        # self.R_extrinsic = None
        # self.t_extrinsic = None



        pts = np.asarray(self.rod_pcd.points).T # 3,N        
        pts = (rotation @ pts).T + translation  # 3, N
        pts = (self.R_extrinsic @ pts.T).T + self.t_extrinsic
        # return np.asarray(self.rod_pcd.points).T
        return pts
    
    def get_images(self):
        self.image_idx = 0
        self.image_list = glob.glob(self.folder + "/*.png")
        # self.image_list_depth = glob.glob(self.folder + "/depth/*.png")
        # self.image_list += glob.glob(self.folder + "/*.jpg")
        self.image_list.sort()
        if len(self.image_list) == 0:
            rospy.logwarn("no images in {}".format(self.folder))
            exit(-1)

    def update(self, event):
        if len(self.image_list) == 0:
            rospy.loginfo("Finished!")

            exit(-1)

        img = self.image_list[0]

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
    rospy.init_node("ImagesToTopic")
    image_folder_publisher = PoseImageOverlay()
    rospy.spin()