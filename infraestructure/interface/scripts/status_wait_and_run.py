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
class StatusWaitAndRun(object):
    def __init__(self):

        self.nodes_to_wait_ready = rospy.get_param("~nodes_to_wait_ready", [])
        self.nodes_to_wait = rospy.get_param("~nodes_to_wait", "")
        self.nodes_to_run  = rospy.get_param("~nodes_to_run", "")
        # print(f"{self.nodes_to_wait} ")

        self.subscribers = []
        for idx, topic in enumerate(self.nodes_to_wait):
            self.subscribers.append(rospy.Subscriber(topic, NodeStatus, self.callback, (idx)))
            # print(f"{idx} {topic}")
        self.status = [NodeStatus.PREPARING] * len(self.nodes_to_wait)

        for idx, topic in enumerate(self.nodes_to_wait_ready):
            self.subscribers.append(rospy.Subscriber(topic, NodeStatus, self.callback_ready, (idx)))
            # print(f"{idx} {topic}")
        self.status_ready = [NodeStatus.PREPARING] * len(self.nodes_to_wait_ready)


        self.publishers = []
        for idx, topic in enumerate(self.nodes_to_run):
            # print(f"{idx} {topic}")
            self.publishers.append(rospy.Publisher(topic, NodeStatus, queue_size=1, latch=True))
        # self.subscribers.append(rospy.Subscriber(self.blue_positions_topic, TensegrityEndcaps, self.position_callback, (2)))

        self.timer = rospy.Timer(rospy.Duration(1.0), self.update)

    def callback(self, msg, idx):
        self.status[idx] = msg.status

    def callback_ready(self, msg, idx):
        self.status_ready[idx] = msg.status

    def update(self, event):

        if all( i == NodeStatus.RUNNING for i in self.status ) and all( i == NodeStatus.READY for i in self.status_ready ) :
            node_status = NodeStatus()
            node_status.status = NodeStatus.RUNNING
            for pub in self.publishers:
                pub.publish(node_status);
                print("Status changed, init!")
                self.timer.shutdown()

if __name__ == '__main__':
    rospy.init_node("StatusWaitAndRun")
    node = StatusWaitAndRun()
    rospy.spin()
