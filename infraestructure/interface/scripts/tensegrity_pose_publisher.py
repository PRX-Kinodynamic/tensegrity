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

from scipy.spatial.transform import Rotation as SciPyRot
import open3d as o3d

class PosePublisher(object):
    def __init__(self):
        # self.folder = rospy.get_param("~dir", "")
        self.topic_tensegrity_bars = rospy.get_param("~topic_tensegrity_bars", "")


        bars = TensegrityBars()
        bars.header.stamp = rospy.Time.now();

        bars.bar_red.position.x = 0
        bars.bar_red.position.y = 0
        bars.bar_red.position.z = 0
        
        bars.bar_red.orientation.w = 1
        bars.bar_red.orientation.x = 0
        bars.bar_red.orientation.y = 0.
        bars.bar_red.orientation.z = 0.

        bars.bar_green.position.x = 0
        bars.bar_green.position.y = 0
        bars.bar_green.position.z = 0.2

        bars.bar_green.orientation.w = 0
        bars.bar_green.orientation.x = 0.707
        bars.bar_green.orientation.y = 0
        bars.bar_green.orientation.z = 0.707

        bars.bar_blue.position.x = 0
        bars.bar_blue.position.y = 0.1
        bars.bar_blue.position.z = 0.3

        bars.bar_blue.orientation.w = 0.85
        bars.bar_blue.orientation.x = 0.26
        bars.bar_blue.orientation.y = 0.0
        bars.bar_blue.orientation.z = 0.4

        print(bars)
        self.publisher = rospy.Publisher(self.topic_tensegrity_bars, TensegrityBars, queue_size=1, latch=True)

        self.publisher.publish(bars)



if __name__ == '__main__':
    rospy.init_node("ImagesToTopic")
    image_folder_publisher = PosePublisher()
    rospy.spin()