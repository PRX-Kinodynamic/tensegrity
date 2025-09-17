import glob
import traceback

# import camera_info_manager
import cv2
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import Image
from std_srvs.srv import Trigger
from interface.msg import NodeStatus
import json


# Subscribe to multiple NodeStatus topics and publish the "relevant" status
class StatusFunnel(object):
    def __init__(self):

        self.node_status_topics = rospy.get_param("~node_status_topics", "")
        self.output_topic = rospy.get_param("~output_topic", "")

        self.subscribers = []

        self.status = [NodeStatus.PREPARING] * len(self.node_status_topics)
        for idx, topic in enumerate(self.node_status_topics):
            self.subscribers.append(rospy.Subscriber(topic, NodeStatus, self.callback, (idx)))

        self.current_status = NodeStatus();

        self.status_publisher = rospy.Publisher(self.output_topic + "/status/current", NodeStatus, queue_size=1, latch=True)

        self.timer = rospy.Timer(rospy.Duration(1.0), self.update)

        # s = rospy.Service(self.topic_name+'/publish_images', Trigger, self.trigger)

    def callback(self, msg, idx):
        # idx = args[0]
        self.status[idx] = msg.status

    def update(self, event):
        if NodeStatus.STOPPED in self.status:
            self.current_status.status = NodeStatus.STOPPED
        elif NodeStatus.PREPARING in self.status:
            self.current_status.status = NodeStatus.PREPARING
        elif NodeStatus.READY in self.status:
            self.current_status.status = NodeStatus.READY
        elif NodeStatus.WAITING in self.status:
            self.current_status.status = NodeStatus.WAITING
        elif all( i == NodeStatus.RUNNING for i in self.status ):
            self.current_status.status = NodeStatus.RUNNING
        else:
            self.current_status.status = NodeStatus.ERROR

        self.status_publisher.publish(self.current_status);

if __name__ == '__main__':
    rospy.init_node("StatusFunnel")
    node = StatusFunnel()
    rospy.spin()