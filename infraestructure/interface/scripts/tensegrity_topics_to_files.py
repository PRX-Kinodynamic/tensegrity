import glob
import traceback

# import camera_info_manager
import cv2
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import Image
from std_srvs.srv import Trigger
from interface.msg import TensegrityLengthSensor
# from interface.msg import TensegrityStamped, Sensor, Motor
from interface.msg import TensegrityEndcaps
import json
import numpy as np

class TensegrityTopicsToFiles(object):
    def __init__(self):

        file_prefix = rospy.get_param("~file_prefix", "")

        self.red_positions_topic = rospy.get_param("~red_positions_topic", "")
        self.green_positions_topic = rospy.get_param("~green_positions_topic", "")
        self.blue_positions_topic = rospy.get_param("~blue_positions_topic", "")

        # self.topic_name = rospy.get_param("~topic_name", "")
        # self.format = rospy.get_param("~format", "")
        # # self.encoding = rospy.get_param("~encoding", "bgr8")
        # self.frequency = 1.0 / rospy.get_param("~frequency", 15)
        # self.loop = rospy.get_param("~loop", False)
        
        self.files = [];
        self.files.append(open(file_prefix + "_red_positions.txt", "w"))
        self.files.append(open(file_prefix + "_green_positions.txt", "w"))
        self.files.append(open(file_prefix + "_blue_positions.txt", "w"))

        self.subscribers = []
        self.subscribers.append(rospy.Subscriber(self.red_positions_topic, TensegrityEndcaps, self.position_callback, (0)))
        self.subscribers.append(rospy.Subscriber(self.green_positions_topic, TensegrityEndcaps, self.position_callback, (1)))
        self.subscribers.append(rospy.Subscriber(self.blue_positions_topic, TensegrityEndcaps, self.position_callback, (2)))


    def position_callback(self, msg, color):

        timestamp = msg.header.stamp.to_sec()
        seq = msg.header.seq
        total_zs = 0

        line = ""
        # self.files[color].write(f"{timestamp} {total_zs} ")
        for endcap in msg.endcaps:
            # self.files[color].write(f"{endcap.x} {endcap.y} {endcap.z} ")
            if (not np.isnan(endcap.x)) and (not np.isnan(endcap.y)) and (not np.isnan(endcap.z)):
                total_zs += 1
                line += f"{endcap.x} {endcap.y} {endcap.z} "

        self.files[color].write(f"{timestamp} {seq} {total_zs} " + line)
        self.files[color].write("\n")
        
    
    def close(self):
        for f in self.files:
            f.close();


if __name__ == '__main__':
    rospy.init_node("TensegrityTopicsToFiles")
    node = TensegrityTopicsToFiles()
    rospy.spin()
    node.close();