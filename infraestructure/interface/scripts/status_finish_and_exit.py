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
from interface.msg import TensegrityEndcaps
from interface.msg import NodeStatus
import json

# Start execution of an experiment by:
#       1) waiting for nodes to be "Ready" and then
#       2) publish other nodes to start execution
class StatusFinishAndExit(object):
    def __init__(self):

        self.nodes_to_wait_finish = rospy.get_param("~nodes_to_wait_finish", [])

        self.subscribers = []
        for idx, topic in enumerate(self.nodes_to_wait_finish):
            self.subscribers.append(rospy.Subscriber(topic, NodeStatus, self.callback, (idx)))
            # print(f"{idx} {topic}")
        self.status_finish = [NodeStatus.PREPARING] * len(self.nodes_to_wait_finish)

        self.timer = rospy.Timer(rospy.Duration(1.0), self.update)

    def callback(self, msg, idx):
        self.status_finish[idx] = msg.status

    def update(self, event):

        if all( i == NodeStatus.FINISH for i in self.status_finish ) :
            print(f"Nodes have finished. Shutting down...")
            rospy.signal_shutdown("Finish status received")
            exit(0)

if __name__ == '__main__':
    rospy.init_node("StatusFinishAndExit")
    node = StatusFinishAndExit()
    rospy.spin()
