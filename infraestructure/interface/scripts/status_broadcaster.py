import glob
import traceback
import os
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
class StatusBroadcaster(object):
    def __init__(self):

        self.input_topic = rospy.get_param("~input_topic", "")
        self.node_status_topics = rospy.get_param("~node_status_topics", "")
        self.publish_to_all = rospy.get_param("~publish_to_all", False)

        self.publishers = []

        self.subscriber = rospy.Subscriber(self.input_topic, NodeStatus, self.callback)

        if not self.publish_to_all:
            for topic in self.node_status_topics:
                self.publishers.append(rospy.Publisher(topic, NodeStatus, queue_size=1, latch=True))


    def callback(self, msg):
        
        if self.publish_to_all:
            m = rospy.get_master()
            all_topics = m.getTopicTypes()[2];
            # print(all_topics)
            for topic, topic_type in all_topics:
                if "status/change" in topic:
                    # print(topic)
                    self.publishers.append(rospy.Publisher(topic, NodeStatus, queue_size=1, latch=True))


        for pub in self.publishers:
            pub.publish(msg);

        rospy.sleep(5.)
        exit(0)


if __name__ == '__main__':
    rospy.init_node("StatusBroadcaster")
    node = StatusBroadcaster()
    rospy.spin()