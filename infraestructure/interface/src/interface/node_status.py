import glob
import traceback

import os
import cv2
import math
import rospy
from interface.msg import NodeStatus
import numpy as np

class NodeStatusHelper:
    def __init__(self, node_id):
        self.msg = NodeStatus()
        self.node_id = node_id
        self.msg.status = NodeStatus.PREPARING;
        change_topic =  node_id + "/status/change";
        current_topic =  node_id + "/status/current";
        self.status_subscriber = rospy.Subscriber(change_topic, NodeStatus, self.callback)
        self.status_publisher = rospy.Publisher(current_topic, NodeStatus, queue_size=1, latch=True)

        self.timer = rospy.Timer(rospy.Duration(1.0), self.update)

        self.status_publisher.publish(self.msg);
    
    @property
    def status(self):
        return self.msg.status

    @status.setter
    def status(self, new_status):
        self.status_change(new_status)

    def callback(self, msg):
        # print(f"{self.node_id}: Status {msg}")
        self.status_change(msg.status)

    def update(self, event):
        # if self.msg.status == NodeStatus.FINISH:
            # rospy.signal_shutdown("Status FINISH received");
        self.status_publisher.publish(self.msg);

    def status_change(self, new_status):
        current = self.msg.status 
        # next_status = current

        # if current == NodeStatus.PREPARING:
        #     next_status = self.status_preparing(new_status);
        # elif current == NodeStatus.READY:
        #     next_status = self.status_ready(new_status);
        # elif current == NodeStatus.RUNNING:
        #     next_status = self.status_running(new_status);
        # elif current == NodeStatus.WAITING:
        #     next_status = self.status_waiting(new_status);
        # elif current == NodeStatus.STOPPED:
        #     next_status = self.status_stopped(new_status);
        # else:
        #     raise Exception("[NodeStatusHelper] Status unknown");
    
        self.msg.status = new_status;
        # print(f"status py: {self.msg.status} {new_status}")
        if not rospy.is_shutdown():
            self.status_publisher.publish(self.msg);

    def status_preparing(self, new_status):
        return new_status;  # All valid on preparing

    def status_ready(self, new_status):
        # switch (new_status)
        if new_status ==  NodeStatus.PREPARING:
            raise Exception( "[NodeStatusHelper] Current status is Running; requested Preparing");
        elif new_status == NodeStatus.STOPPED:
            raise Exception( "[NodeStatusHelper] Current status is Stopped; requested Preparing");
        else:
            return new_status;

    def status_running(self, new_status):
        if new_status == NodeStatus.PREPARING:
            raise Exception( "[NodeStatusHelper] Current status is Running; requested Preparing");
        elif new_status == NodeStatus.STOPPED:
            raise Exception( "[NodeStatusHelper] Current status is Stopped; requested Preparing");
        else:
            return new_status;

    def status_waiting(self, new_status):
        if new_status == NodeStatus.PREPARING:
            raise Exception("[NodeStatusHelper] Current status is Running; requested Preparing");
        elif new_status == NodeStatus.STOPPED:
            raise Exception("[NodeStatusHelper] Current status is Stopped; requested Preparing");
        else:
            return new_status;

    def status_stopped(self, new_status):
        if new_status ==  NodeStatus.STOPPED:
            return NodeStatus.STOPPED;
        else:
            raise Exception("[NodeStatusHelper] Current status is Stopped. No status change allow");

