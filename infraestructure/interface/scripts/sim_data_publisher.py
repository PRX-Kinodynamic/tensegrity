import glob
import traceback

import os
import cv2
import math
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import Image
from std_srvs.srv import Trigger
# from geometru_msgs.msg import Pose
from interface.msg import TensegrityEndcaps
from interface.msg import TensegrityBars
from interface.msg import TensegrityLengthSensor
from interface.msg import NodeStatus
from interface.msg import TensegrityStamped, Sensor, Motor
import tf
import tf2_ros
import geometry_msgs

import numpy as np
from scipy.spatial.transform import Rotation as SciPyRot
from numpy import linalg as LA

from interface.node_status import NodeStatusHelper

class SimDataPublisher(object):
    def __init__(self):
        # self.folder = rospy.get_param("~dir", "")
        self.endcaps_topic   = rospy.get_param("~endcaps_topic", "")
        self.red_endcaps_topic   = rospy.get_param("~red_endcaps_topic", "")
        self.green_endcaps_topic = rospy.get_param("~green_endcaps_topic", "")
        self.blue_endcaps_topic  = rospy.get_param("~blue_endcaps_topic", "")

        self.cable_topic = rospy.get_param("~cable_topic", "")
        self.sensors_topic_name = rospy.get_param("~sensors_topic_name", "")

        self.bar_gt_topic   = rospy.get_param("~bar_gt_topic", "")
        
        self.pub_frequency = rospy.get_param("~pub_frequency", 30)
        self.data_frequency = rospy.get_param("~data_frequency", 200)
        self.loop = rospy.get_param("~loop", True)
        
        self.data_file = rospy.get_param("~data_file", "")
        self.data_frame = rospy.get_param("~data_frame", "data")
        self.data_scale_factor = rospy.get_param("~data_scale_factor", 0.1) #Data is in cm, but need it in meters
        self.offset_to_origin = rospy.get_param("~offset_to_origin", True)

        self.endcap_noise_mu = rospy.get_param("~endcap_noise_mu", 0.0)
        self.endcap_noise_sigma = rospy.get_param("~endcap_noise_sigma", 0.0)
        self.endcap_miss_probability = rospy.get_param("~endcap_miss_probability", 0.0)

        self.endcap_swap_probability = rospy.get_param("~endcap_swap_probability", 0.0)

        self.cables_noise_mu = rospy.get_param("~cables_noise_mu", 0.0) 
        self.cables_noise_sigma = rospy.get_param("~cables_noise_sigma", 0.0) 

        self.filename_prefix = rospy.get_param("~filename_prefix", "")

        self.node_status = NodeStatusHelper("/node/simulation/");
        self.id_red = 0
        self.id_green = 1 
        self.id_blue = 2
        self.state_idx = 0

        self.offsetP = np.array([0.0, 0.0, +0.325 / 2.0, 1.0 ], np.float32) #* self.data_scale_factor 
        self.offsetM = np.array([0.0, 0.0, -0.325 / 2.0, 1.0 ], np.float32) #* self.data_scale_factor 

        assert self.pub_frequency < self.data_frequency, "Publishing frequency has to be less than simulation frequency!"
        self.step = self.data_frequency / self.pub_frequency

        print(f"step: {self.step}" )
        self.traj = []

        self.read_file()
        # self.publish_tf()
        # self.broadcaster.sendTransform(self.static_transformStamped)

        gt_red_endcaps_topic = self.red_endcaps_topic + "/gt"
        gt_green_endcaps_topic = self.green_endcaps_topic + "/gt"
        gt_blue_endcaps_topic = self.blue_endcaps_topic + "/gt"

        # GT endcap publishers
        self.red_endcaps_gt_publisher = rospy.Publisher(gt_red_endcaps_topic, TensegrityEndcaps, queue_size=1, latch=True)
        self.green_endcaps_gt_publisher = rospy.Publisher(gt_green_endcaps_topic, TensegrityEndcaps, queue_size=1, latch=True)
        self.blue_endcaps_gt_publisher = rospy.Publisher(gt_blue_endcaps_topic, TensegrityEndcaps, queue_size=1, latch=True)
            
        # Noise endcap publishers
        self.endcaps_publisher = rospy.Publisher(self.endcaps_topic, TensegrityEndcaps, queue_size=1, latch=True)
        self.red_endcaps_publisher = rospy.Publisher(self.red_endcaps_topic, TensegrityEndcaps, queue_size=1, latch=True)
        self.green_endcaps_publisher = rospy.Publisher(self.green_endcaps_topic, TensegrityEndcaps, queue_size=1, latch=True)
        self.blue_endcaps_publisher = rospy.Publisher(self.blue_endcaps_topic, TensegrityEndcaps, queue_size=1, latch=True)
        
        gt_cable_topic = self.cable_topic + "/gt"
        # Cable GT publisher
        self.cable_gt_publisher = rospy.Publisher(gt_cable_topic, TensegrityLengthSensor, queue_size=1, latch=True)
        
        # Cable noise publisher
        self.cable_publisher = rospy.Publisher(self.cable_topic, TensegrityLengthSensor, queue_size=1, latch=True)
        self.sensor_length_pub = rospy.Publisher(self.sensors_topic_name, TensegrityStamped, queue_size=1, latch=True)

        self.bar_gt_publisher = rospy.Publisher(self.bar_gt_topic, TensegrityBars, queue_size=1, latch=True)
        self.timer = rospy.Timer(rospy.Duration(1.0 / self.pub_frequency), self.update)

        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        self.tf = geometry_msgs.msg.TransformStamped()
        self.tf.header.frame_id = "world"
        self.tf.child_frame_id = "robot"

        self.file = None
        if self.filename_prefix != "":
            self.filename = self.filename_prefix + ".txt"
            self.file = open(self.filename, 'w')


        # nan endcap for miss ones
        self.nan_endcap = geometry_msgs.msg.Point() 
        self.nan_endcap.x = np.nan
        self.nan_endcap.y = np.nan
        self.nan_endcap.z = np.nan

        self.node_status.status = NodeStatus.READY


    def publish_tf(self):
        self.broadcaster = tf2_ros.StaticTransformBroadcaster()
        self.static_transformStamped = geometry_msgs.msg.TransformStamped()

        self.static_transformStamped.header.stamp = rospy.Time.now()
        self.static_transformStamped.header.frame_id = "world"
        self.static_transformStamped.child_frame_id = self.data_frame

        if self.offset_to_origin:
            T0 = self.traj[0]["red"]["T"]
        else:
            T0 = np.identity(4)

        self.static_transformStamped.transform.translation.x = T0[0, 3]
        self.static_transformStamped.transform.translation.y = T0[1, 3]
        self.static_transformStamped.transform.translation.z = T0[2, 3]
        
        self.static_transformStamped.transform.rotation.w = 1.0
        self.static_transformStamped.transform.rotation.x = 0.0
        self.static_transformStamped.transform.rotation.y = 0.0
        self.static_transformStamped.transform.rotation.z = 0.0

        # print(static_transformStamped)
        # print(self.static_transformStamped)

    def create_transform(self, position, quat):
        quat_np = np.array(quat, np.float32)
        #rot = SciPyRot.from_quat(quat_np, scalar_first=True) # W is first
        rot = SciPyRot.from_quat([quat_np[1], quat_np[2], quat_np[3], quat_np[0]]) # W is first
        
        tr = np.array(position, np.float32) * self.data_scale_factor
        T = np.identity(4)
        T[0:3,3] = tr
        T[0:3,0:3] = rot.as_matrix()
        return T

    def get_values(self, line):
        #  3     4    3     3    3       4     3      3      3      4      3      3  
        # 0:3   3:7  7:10 10:13 13:16  16:20  20:23  23:26  26:29  29:33  33:36 36:39
        # PosA quatA  VpA  VqA  PosB   quatB   VpB    VqB    PosC  quatC   VpC   VqC 
        state = {}
        state["red"] = {}
        state["green"] = {}
        state["blue"] = {}
        Ta = self.create_transform(line[0:3], line[3:7])
        Tb = self.create_transform(line[13:16], line[16:20])
        Tc = self.create_transform(line[26:29], line[29:33])
        state["red"]["T"] = Ta
        state["green"]["T"] = Tb
        state["blue"]["T"] = Tc
        
        return state

    def read_file(self):

        file = open(self.data_file, 'r')

        self.traj = []
        self.lines = []
        for line in file:
            self.lines.append(line)
            sp = line.split()
            self.traj.append(self.get_values(sp))

        # print(f"traj[0]: {self.traj[0]}")
        # print(f"traj len: {len(self.traj)}")

    def get_next_state(self):

        dt = 1.0 / self.data_frequency
        # stop = self.state_idx + self.step
        # for i in np.arange(self.state_idx, stop, dt):

        prev =  math.floor(self.state_idx)+1
        self.state_idx += self.step
        idx = math.floor(self.state_idx)

        if self.file is not None:
            ti = rospy.Time.now().to_sec()
            # print(f"ti {ti} prev {prev} idx {idx}")
            for i in range(prev, idx+1):
                line = str(ti) + " " +  self.lines[i];
                self.file.write(line)
                ti += dt

        return self.traj[idx]

    def pose_to_msg(self, T):
        msg = geometry_msgs.msg.Pose()
        quat = tf.transformations.quaternion_from_matrix(T)

        msg.position.x = T[0, 3]
        msg.position.y = T[1, 3]
        msg.position.z = T[2, 3]

        msg.orientation.x = quat[0]
        msg.orientation.y = quat[1]
        msg.orientation.z = quat[2]
        msg.orientation.w = quat[3]
        return msg;

    def state_to_msg(self, state):
        msg = TensegrityBars()
        msg.bar_red = self.pose_to_msg(state["red"]["T"])
        msg.bar_green = self.pose_to_msg(state["green"]["T"])
        msg.bar_blue = self.pose_to_msg(state["blue"]["T"])
        # print(f"msg.bar_red: {msg.bar_red}")
        
        msg.header.frame_id = self.data_frame
        msg.header.stamp = rospy.Time.now()

        return msg;

    def pose_inv(self, T):
        Tinv = np.identity(4)
        Tinv[0:3,0:3] = T[0:3,0:3].T
        Tinv[0:3,3] = -Tinv[0:3,0:3] @ T[0:3,3]
        return Tinv

    def compute_endcaps(self, T, offset):
        endcap = T @ offset

        # print(f"T: {T}")
        # print(f"offset: {offset}")
        # print(f"endcap: {endcap}")
        msg = geometry_msgs.msg.Point()

        msg.x = endcap[0]
        msg.y = endcap[1]
        msg.z = endcap[2]

        return msg


    def endcaps_msg(self, bar_id, T):
        msg = TensegrityEndcaps()
        eA = self.compute_endcaps(T, self.offsetP)
        eB = self.compute_endcaps(T, self.offsetM)

        # print(f"eA {eA}")
        # print(f"eB {eB}")
        msg.endcaps.append(eA)
        msg.endcaps.append(eB)

        msg.barId = bar_id
        msg.header.frame_id = self.data_frame
        msg.header.stamp = rospy.Time.now()
        return msg

    def get_distance(self, Ta, Tb, offsetA, offsetB):
        endcapA = (Ta @ offsetA)[:3]
        endcapB = (Tb @ offsetB)[:3]
        return LA.norm(endcapA - endcapB);


    def compute_distances(self, state):
        msg = TensegrityLengthSensor()

        msg.header.frame_id = self.data_frame
        msg.header.stamp = rospy.Time.now()

        Tr = state["red"]["T"]
        Tg = state["green"]["T"]
        Tb = state["blue"]["T"]

        # offP = self.offset
        # offM = -self.offset

        msg.length[0] = self.get_distance(Tb, Tg, self.offsetM, self.offsetM);
        msg.length[1] = self.get_distance(Tg, Tr, self.offsetM, self.offsetM);
        msg.length[2] = self.get_distance(Tr, Tb, self.offsetM, self.offsetM);
        
        msg.length[3] = self.get_distance(Tg, Tr, self.offsetP, self.offsetP);
        msg.length[4] = self.get_distance(Tr, Tb, self.offsetP, self.offsetP);
        msg.length[5] = self.get_distance(Tb, Tg, self.offsetP, self.offsetP);
        
        msg.length[6] = self.get_distance(Tg, Tb, self.offsetP, self.offsetM);
        msg.length[7] = self.get_distance(Tr, Tg, self.offsetP, self.offsetM);
        msg.length[8] = self.get_distance(Tr, Tb, self.offsetM, self.offsetP);

        # print(f"dists: {msg.length}")

        return msg;

    def merge_endcap_msgs(self, red_msg, green_msg, blue_msg):
        msg = TensegrityEndcaps()
        msg.barId = red_msg.barId 
        msg.header.frame_id = red_msg.header.frame_id 
        msg.header.stamp = red_msg.header.stamp 
        
        msg.endcaps.append(red_msg.endcaps[0])
        msg.endcaps.append(red_msg.endcaps[1])
        
        msg.endcaps.append(green_msg.endcaps[0])
        msg.endcaps.append(green_msg.endcaps[1])

        msg.endcaps.append(blue_msg.endcaps[0])
        msg.endcaps.append(blue_msg.endcaps[1])

        return msg;

    def add_noise_to_endcap(self, msg):
        msg_noise = TensegrityEndcaps()
        eA = geometry_msgs.msg.Point() # msg.endcaps[0]
        eB = geometry_msgs.msg.Point() # msg.endcaps[1]

        noise = np.random.normal(self.endcap_noise_mu, self.endcap_noise_sigma, (3,2))
        swap = np.random.binomial(1, self.endcap_swap_probability, 1)
        
        Aidx = 0;
        Bidx = 1;
        # print(f"swap: {swap}")
        if swap[0] == 1:
            Aidx = 1;
            Bidx = 0;

        eA.x = msg.endcaps[Aidx].x + noise[0, 0]
        eA.y = msg.endcaps[Aidx].y + noise[1, 0]
        eA.z = msg.endcaps[Aidx].z + noise[2, 0]

        eB.x = msg.endcaps[Bidx].x + noise[0, 1]
        eB.y = msg.endcaps[Bidx].y + noise[1, 1]
        eB.z = msg.endcaps[Bidx].z + noise[2, 1]

        trial = np.random.binomial(1, 1.0 - self.endcap_miss_probability, 2)
        if trial[0] == 1:
            msg_noise.endcaps.append(eA)
        else:
            msg_noise.endcaps.append(self.nan_endcap)

        if trial[1] == 1:
            msg_noise.endcaps.append(eB)
        else:
            msg_noise.endcaps.append(self.nan_endcap)

        msg_noise.barId = msg.barId 
        msg_noise.header.frame_id = msg.header.frame_id 
        msg_noise.header.stamp = msg.header.stamp 

        return msg_noise

    def add_noise_to_cables(self, msg):
        msg_noise = TensegrityLengthSensor()
        msg_noise.header.frame_id = msg.header.frame_id
        msg_noise.header.stamp = msg.header.stamp

        noise = np.random.normal(self.cables_noise_mu, self.cables_noise_sigma, 9)
        msg_noise.length[0] = msg.length[0] + noise[0]
        msg_noise.length[1] = msg.length[1] + noise[1]
        msg_noise.length[2] = msg.length[2] + noise[2]
        
        msg_noise.length[3] = msg.length[3] + noise[3]
        msg_noise.length[4] = msg.length[4] + noise[4]
        msg_noise.length[5] = msg.length[5] + noise[5]
        
        msg_noise.length[6] = msg.length[6] + noise[6]
        msg_noise.length[7] = msg.length[7] + noise[7]
        msg_noise.length[8] = msg.length[8] + noise[8]

        return msg_noise

    def cables_to_sensors(self, cable_msg):
        msg = TensegrityStamped()
        msg.header.frame_id = cable_msg.header.frame_id
        msg.header.stamp = cable_msg.header.stamp
        i = 0;
        for li in cable_msg.length:
            msg.sensors.append(Sensor())
            msg.sensors[-1].length = li
            msg.sensors[-1].id = i;
            i += 1;
        # print(f"sensor: {msg.sensors}")
        return msg;


    def publish_bars_tf(self, state):
        self.tf.header.stamp = rospy.Time.now();
        self.tf.header.seq += 1

        Tr = state["red"]["T"]
        Tg = state["green"]["T"]
        Tb = state["blue"]["T"]

        Tr_centroid = (Tr[0:3,3] + Tg[0:3,3] + Tb[0:3,3]) / 3.0

        # print(f"Tr {Tr_centroid}")

        self.tf.transform.translation.x = Tr_centroid[0]
        self.tf.transform.translation.y = Tr_centroid[1]
        self.tf.transform.translation.z = Tr_centroid[2]
        
        self.tf.transform.rotation.w = 1.0
        self.tf.transform.rotation.x = 0.0
        self.tf.transform.rotation.y = 0.0
        self.tf.transform.rotation.z = 0.0

        self.tf_broadcaster.sendTransform(self.tf)

    def update(self, event):


        if self.node_status.status == NodeStatus.READY:
            # Publish the first one, just for viz 
            idx = math.floor(self.state_idx)
            state = self.traj[idx]
            self.publish_bars_tf(state)

            msg_red = self.endcaps_msg(self.id_red, state["red"]["T"])
            msg_green = self.endcaps_msg(self.id_green, state["green"]["T"])
            msg_blue = self.endcaps_msg(self.id_blue, state["blue"]["T"])

            bar_msg = self.state_to_msg(state)  

            self.bar_gt_publisher.publish(bar_msg)
        
            self.red_endcaps_gt_publisher.publish(msg_red)
            self.green_endcaps_gt_publisher.publish(msg_green)
            self.blue_endcaps_gt_publisher.publish(msg_blue)
            return

        if self.node_status.status != NodeStatus.RUNNING:
            return

        if self.state_idx + self.step > len(self.traj):
            if self.loop:
                self.state_idx += self.step
                self.state_idx = self.state_idx - len(self.traj)
            else:
                self.node_status.status = NodeStatus.FINISH
                rospy.Rate(1.0).sleep();
                rospy.Rate(1.0).sleep();
                rospy.loginfo("Finished!")
                self.file.close();
                exit(-1)
        # self.broadcaster.sendTransform(self.static_transformStamped)
        
        state = self.get_next_state()
        bar_msg = self.state_to_msg(state)  

        self.publish_bars_tf(state)

        
        msg_red = self.endcaps_msg(self.id_red, state["red"]["T"])
        msg_green = self.endcaps_msg(self.id_green, state["green"]["T"])
        msg_blue = self.endcaps_msg(self.id_blue, state["blue"]["T"])

        msg_noise_red = self.add_noise_to_endcap(msg_red)
        msg_noise_green = self.add_noise_to_endcap(msg_green)
        msg_noise_blue = self.add_noise_to_endcap(msg_blue)

        msg_noise_all = self.merge_endcap_msgs(msg_noise_red, msg_noise_green, msg_noise_blue)

        msg_cable_sensor = self.compute_distances(state)

        msg_noise_cable_sensor = self.add_noise_to_cables(msg_cable_sensor)
        msg_noise_sensors = self.cables_to_sensors(msg_noise_cable_sensor)

        self.bar_gt_publisher.publish(bar_msg)
        
        self.red_endcaps_gt_publisher.publish(msg_red)
        self.green_endcaps_gt_publisher.publish(msg_green)
        self.blue_endcaps_gt_publisher.publish(msg_blue)

        self.endcaps_publisher.publish(msg_noise_all)
        self.red_endcaps_publisher.publish(msg_noise_red)
        self.green_endcaps_publisher.publish(msg_noise_green)
        self.blue_endcaps_publisher.publish(msg_noise_blue)
        
        self.cable_publisher.publish(msg_noise_cable_sensor)
        self.cable_gt_publisher.publish(msg_cable_sensor)
        self.sensor_length_pub.publish(msg_noise_sensors);

if __name__ == '__main__':
    rospy.init_node("SimDataPublisher")
    sim_publisher = SimDataPublisher()
    rospy.spin()
