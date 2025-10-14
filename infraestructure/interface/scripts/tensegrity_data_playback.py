import glob
import traceback

# import camera_info_manager
import cv2
import time
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import Image
from std_srvs.srv import Trigger
from interface.msg import TensegrityLengthSensor
import json
from interface.msg import NodeStatus, TensegrityEndcaps, TensegrityBars
from interface.msg import TensegrityStamped, Sensor, Motor
from interface.node_status import NodeStatusHelper
import tf2_ros
import geometry_msgs.msg
import tf.transformations as tr
import numpy as np

class TensegrityDataPublisher(object):
    def __init__(self):
        self.node_status = NodeStatusHelper("/nodes/playback/");
        # Assumming: self.directory/{color,depth,data}
        self.directory = rospy.get_param("~directory", "")
        self.rgb_topic_name = rospy.get_param("~rgb_topic_name", "")
        self.depth_topic_name = rospy.get_param("~depth_topic_name", "")
        self.cable_sensors_topic_name = rospy.get_param("~cable_sensors_topic_name", "")
        # self.encoding = rospy.get_param("~encoding", "bgr8")
        # self.frequency = 1.0 / rospy.get_param("~frequency", 15)
        self.loop = rospy.get_param("~loop", False)
        self.filename_prefix = rospy.get_param("~filename_prefix", "")
        self.output_to_file = rospy.get_param("~output_to_file", False)
        self.fix_dt = rospy.get_param("~fix_dt", 0.0)
        self.publish_npy_poses = rospy.get_param("~publish_npy_poses", False)

        red_endcaps_topic = rospy.get_param("~red_endcaps_topic", "");
        green_endcaps_topic = rospy.get_param("~green_endcaps_topic", "");
        blue_endcaps_topic = rospy.get_param("~blue_endcaps_topic", "");
        self.publish_mocaps = rospy.get_param("~publish_mocaps", False)
        # print(self.folder)

        self.rgb_images = self.get_images(self.directory + "/color")
        self.depth_images = self.get_images(self.directory + "/depth")
        # self.mocaps = self.read_mocaps(self.directory + "/data");

        self.read_length_data()

        self.bridge = CvBridge()


        self.idx = 0
        self.iter_idx = 0
        if not self.output_to_file or self.filename_prefix == "":
            print("Not writing file")
            self.output_to_file = False
        else:
            self.filename = self.filename_prefix + ".txt"
            self.file = open(self.filename, 'w')

        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        
        self.rgb_image_pub = rospy.Publisher(self.rgb_topic_name, Image, queue_size=1, latch=True)
        self.depth_image_pub = rospy.Publisher(self.depth_topic_name, Image, queue_size=1, latch=True)
        self.sensor_length_pub = rospy.Publisher(self.cable_sensors_topic_name, TensegrityStamped, queue_size=1, latch=True)
        self.npy_poses_pub = rospy.Publisher("/tensegrity/endcaps/proposed", TensegrityEndcaps, queue_size=1, latch=True)
        
        self.red_endcaps = None
        self.green_endcaps = None
        self.blue_endcaps = None
        if self.publish_mocaps:
            self.red_endcaps = TensegrityEndcaps();
            self.green_endcaps = TensegrityEndcaps();
            self.blue_endcaps = TensegrityEndcaps();
            self.red_endcaps_pub = rospy.Publisher(red_endcaps_topic, TensegrityEndcaps, queue_size=1, latch=True)
            self.green_endcaps_pub = rospy.Publisher(green_endcaps_topic, TensegrityEndcaps, queue_size=1, latch=True)
            self.blue_endcaps_pub = rospy.Publisher(blue_endcaps_topic, TensegrityEndcaps, queue_size=1, latch=True)
            # self.c_T_m = np.array([[-0.72231893, -0.69031471, 0.04148439, -0.0911841 ], [-0.03982037, -0.01837056, -0.99903797, 2.43988996], [ 0.6904127, -0.72327596, -0.01421918, 0.18661254], [ 0.,     0.,     0.,     1.    ]])
            # self.c_T_m[0:3,0:3] = self.c_T_m[0:3,0:3].T
            # self.c_T_m[0:3,3] = -self.c_T_m[0:3,0:3] @ self.c_T_m[0:3,3]
            # self.w_T_c = np.array([[1.0,0.0,-0.01,0.743],[0.0,-1.0,-0.007,0.082],[-0.01,0.007,-1.0,1.441],[0,0,0,1]])
            # self.mocap_tf = self.w_T_c @ self.c_T_m
            self.mocap_tf = np.eye(4)

            # print(f"self.mocap_tf {self.mocap_tf}")
            self.red_endcaps.header.frame_id = "mocaps"
            self.green_endcaps.header.frame_id = "mocaps"
            self.blue_endcaps.header.frame_id = "mocaps"

        if self.publish_npy_poses:
            self.endcap_proposed = []
            self.read_npy(self.directory + "/poses-proposed/")



        self.node_status.status = NodeStatus.READY

    def read_npy(self, data_dir):
        for i in range(6):
            pos = np.load(data_dir + f'/{i}_pos.npy');
            self.endcap_proposed.append(pos)
        # self.red_proposed = np.load(data_dir + '/red.npy')
        # self.blue_proposed = np.load(data_dir + '/blue.npy')
        # self.green_proposed = np.load(data_dir + '/green.npy')


    def publish_npy(self):
        # msg = TensegrityBars()
        msg = TensegrityEndcaps();

        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "real_sense"

        # print(f'msg {msg}')
        for i in range(6):
            pt = self.endcap_proposed[i][self.idx]
            ptm = geometry_msgs.msg.Point()
            ptm.x = pt[0]
            ptm.y = pt[1]
            ptm.z = pt[2]
            msg.endcaps.append(ptm)
            # msg.ids.append(i)
            
        # xred =  self.red_proposed[self.idx];
        # qred = tr.quaternion_from_matrix(xred)
        # xblue =  self.blue_proposed[self.idx];
        # qblue = tr.quaternion_from_matrix(xblue)
        # xgreen =  self.green_proposed[self.idx];
        # qgreen = tr.quaternion_from_matrix(xgreen)

        # msg.bar_red.orientation.x = qred[0]
        # msg.bar_red.orientation.y = qred[1]
        # msg.bar_red.orientation.z = qred[2]
        # msg.bar_red.orientation.w = qred[3]
        # msg.bar_red.position.x = xred[0,3]
        # msg.bar_red.position.y = xred[1,3]
        # msg.bar_red.position.z = xred[2,3]

        # msg.bar_blue.orientation.x = qblue[0]
        # msg.bar_blue.orientation.y = qblue[1]
        # msg.bar_blue.orientation.z = qblue[2]
        # msg.bar_blue.orientation.w = qblue[3]
        # msg.bar_blue.position.x = xblue[0,3]
        # msg.bar_blue.position.y = xblue[1,3]
        # msg.bar_blue.position.z = xblue[2,3]

        # msg.bar_green.orientation.x = qgreen[0]
        # msg.bar_green.orientation.y = qgreen[1]
        # msg.bar_green.orientation.z = qgreen[2]
        # msg.bar_green.orientation.w = qgreen[3]
        # msg.bar_green.position.x = xgreen[0,3]
        # msg.bar_green.position.y = xgreen[1,3]
        # msg.bar_green.position.z = xgreen[2,3]

        self.npy_poses_pub.publish(msg);


    def get_timestamp(self):
        current_time = time.localtime()
        return time.strftime("%y%m%d_%H%M%S", current_time)

    def read_json(self, filename):
        file = open(filename, 'r')
        js = json.load(file)

        ti = js['header']['secs']
        sensors = js['sensors']

        meassurements = {}
        # msg = TensegrityLengthSensor()
        msg = TensegrityStamped()

        for i in range(9): 
            sensor_msg = Sensor()
            sensor_msg.id = i
            sensor_msg.length = sensors[str(i)]['length']
            sensor_msg.capacitance = sensors[str(i)]['capacitance']
            msg.sensors.append(sensor_msg)
            # print(f"sensor: {sensor_msg}")
            # meassurements[i] = sensors[str(i)]
        for i in range(5): 
            motor_msg = Motor()
            motor_msg.id = i
            motor_msg.position = js['motors'][str(i)]["position"]
            motor_msg.speed = js['motors'][str(i)]["speed"]
            motor_msg.target = js['motors'][str(i)]["target"]
            motor_msg.done = js['motors'][str(i)]["done"]
            msg.motors.append(motor_msg)


        json_mocaps = js['mocap']
        mocaps = {}
        # print(f"json_mocaps {json_mocaps} ")
        for key, value in json_mocaps.items():
            if value == None:
                mocaps[int(key)] = None
            else:
                x = value["x"]
                y = value["y"]
                z = value["z"]
                mocaps[int(key)] = [x, y, z]
                # mocaps[int(key)] = [y, x, z]
                # print(f"k: {int(key)}  m: {mocaps[int(key)]}")
            # print(f"i: {i}, m {m} " )
            # print(f"mocap: {m} {mocaps[int(m)]}" )
        
        # print(f"mocap: {mocaps}" )

        # exit(0)
        meassurements["mocap"] = mocaps
        meassurements['msg'] = msg
        meassurements['t'] = float(ti)
        return meassurements
        # print(meassurements)
        # exit(-1)

    def read_length_data(self):
        image_list = glob.glob(self.directory + "/data/*.json")

        cable_lengths = []
        image_list.sort()
        for filename in image_list:
            zi = self.read_json(filename)
            cable_lengths.append(zi)
            # print(filename)
        self.cable_lengths = cable_lengths
        # self.cable_lengths = sorted(cable_lengths, key=lambda d: d['t'])
        t0 = self.cable_lengths[0]['t']

        for zi in self.cable_lengths:
            zi['t'] = zi['t'] - t0

        if self.fix_dt == 0:        
            for i in range(len(self.cable_lengths)-1):
                dt = self.cable_lengths[i+1]['t'] -self.cable_lengths[i]['t']
                self.cable_lengths[i]['dt'] = dt

            # Artificial dt for the last one
            self.cable_lengths[-1]['dt'] = self.cable_lengths[0]['dt'] 
        else:
            for i in range(len(self.cable_lengths)):
                self.cable_lengths[i]['dt'] = self.fix_dt

        # print(self.cable_lengths[0]['t'], self.cable_lengths[0]['dt'])
        # print(self.cable_lengths[1]['t'], self.cable_lengths[1]['dt'])
        # print(self.cable_lengths[2]['t'], self.cable_lengths[2]['dt'])
        # print(self.cable_lengths[-1]['t'], self.cable_lengths[-1]['dt'])

    def get_images(self, directory):
        # self.image_idx = 0
        image_list = glob.glob(directory + "/*.png")
        # image_list_depth = glob.glob(self.folder + "/depth/*.png")
        # image_list += glob.glob(self.folder + "/*.jpg")
        image_list.sort()
        if len(image_list) == 0:
            rospy.logwarn("no images in {}".format(directory))
            exit(-1)
        return image_list


    def get_image_msg(self, image_list, img_format):
        if len(image_list) == 0:
            rospy.loginfo("Finished!")
            return
            # exit(-1)

        img = image_list[0]

        if img_format == "color":
            cv_image = cv2.imread(img, cv2.IMREAD_COLOR)
            img_msg = self.bridge.cv2_to_imgmsg(cv_image, 'bgr8')
        elif img_format == "gray":
            cv_image = cv2.imread(img, cv2.IMREAD_UNCHANGED)
            img_msg = self.bridge.cv2_to_imgmsg(cv_image, 'passthrough')
        else:
            rospy.logwarn("Format not supported. Choose 'color' or 'gray'")
            exit(-1)

        img_msg.header.stamp = rospy.Time.now()
        img_msg.header.frame_id = "real_sense"
        img_msg.header.seq = self.idx;
            
        if self.node_status.status == NodeStatus.RUNNING:
            image_list.pop(0)
            if self.loop:
                image_list.append(img)
        return img_msg


    def get_next_dt(self):
        dt = self.cable_lengths[0]['dt']
        return dt;
        # return rospy.Rate(1.0 / dt)

    def get_sensors_msg(self):
        cable_length = self.cable_lengths[0]
        self.cable_lengths.pop(0)

        if self.loop:
            self.cable_lengths.append(cable_length)

        msg = cable_length['msg']
        msg.header.stamp = rospy.Time.now()
        msg.header.seq = self.idx;
        return msg;

    def mocap_to_point_msg(self, mocap):
        msg = geometry_msgs.msg.Point();
        if mocap is None:
            msg.x = np.nan
            msg.y = np.nan
            msg.z = np.nan
        else:
            # mocap = mocap;
            mocap = np.array([mocap[0],mocap[1],mocap[2],1]) / 1000 ;
            mocap = (self.mocap_tf @ mocap)[0:3] + self.mocap_tf[0:3,3]; 

            msg.x = mocap[0]
            msg.y = mocap[1]
            msg.z = mocap[2]
        return msg;

    def send_mocap_tf(self):

        data = self.cable_lengths[0]

        t = geometry_msgs.msg.TransformStamped()
        t.header.stamp = rospy.Time.now()
        t.header.frame_id = "mocaps"

        if self.red_endcaps is not None: 
            self.red_endcaps.endcaps = []
            self.green_endcaps.endcaps = []
            self.blue_endcaps.endcaps = []
            # print(data["mocap"][0])
            self.red_endcaps.endcaps.append(self.mocap_to_point_msg(data["mocap"][0]))
            self.red_endcaps.endcaps.append(self.mocap_to_point_msg(data["mocap"][1]))
            self.green_endcaps.endcaps.append(self.mocap_to_point_msg(data["mocap"][2]))
            self.green_endcaps.endcaps.append(self.mocap_to_point_msg(data["mocap"][3]))
            self.blue_endcaps.endcaps.append(self.mocap_to_point_msg(data["mocap"][4]))
            self.blue_endcaps.endcaps.append(self.mocap_to_point_msg(data["mocap"][5]))

            self.red_endcaps.header.seq = self.idx
            self.green_endcaps.header.seq = self.idx
            self.blue_endcaps.header.seq = self.idx
            self.red_endcaps.header.stamp = rospy.Time.now()
            self.green_endcaps.header.stamp = rospy.Time.now()
            self.blue_endcaps.header.stamp = rospy.Time.now()

        for idx in range(len(data["mocap"])):
            value = data["mocap"][idx]
            if value == None:
                continue;

            t.child_frame_id = "mocap_" + str(idx)

            # t.header.frame_id = "mocap_" + str(idx)
            # t.child_frame_id = "mocap" 
            
            t.transform.translation.x = value[0]/1000 # mocaps in [mm]
            t.transform.translation.y = value[1]/1000 # mocaps in [mm]
            t.transform.translation.z = value[2]/1000 # mocaps in [mm]
            t.transform.rotation.x = 0.0
            t.transform.rotation.y = 0.0
            t.transform.rotation.z = 0.0
            t.transform.rotation.w = 1.0

            self.tf_broadcaster.sendTransform(t)

    def add_to_file(self):
        # Time idx mocap+ 
        data = self.cable_lengths[0]
        dt = rospy.Time.now()
        line = str(dt.to_sec()) + " "
        line += str(self.idx) + " "

        for idx in range(len(data["mocap"])):
            value = data["mocap"][idx]
            if value == None:
                line += "NaN NaN NaN "
            else: 
                line += str(value[0]) + " "
                line += str(value[1]) + " "
                line += str(value[2]) + " "

        line += "\n"
        self.file.write(line)
        # print(line)

    def spin(self):

        prev = rospy.Time.now();
        dt = 0
        while not rospy.is_shutdown():
            if self.node_status.status != NodeStatus.RUNNING:
                # print(f"NOT RUNNING")
                if self.node_status.status == NodeStatus.READY:
                    # print(f"READY")
                    rgb_msg = self.get_image_msg(self.rgb_images, "color")
                    depth_msg = self.get_image_msg(self.depth_images, "gray")
                    self.rgb_image_pub.publish(rgb_msg)
                    self.depth_image_pub.publish(depth_msg)

                rospy.Rate(10.0).sleep();
                continue;

            curr_dt = (rospy.Time.now() - prev).to_sec();
            if curr_dt < dt:
                continue     


            if len(self.cable_lengths) == 0:
                rospy.loginfo("Finished!")
                break

            prev = rospy.Time.now();

            self.send_mocap_tf()
            if self.output_to_file:
                self.add_to_file();

            dt = self.get_next_dt()
            rgb_msg = self.get_image_msg(self.rgb_images, "color")
            depth_msg = self.get_image_msg(self.depth_images, "gray")
            cable_length_msg = self.get_sensors_msg()


            self.rgb_image_pub.publish(rgb_msg)
            self.depth_image_pub.publish(depth_msg)
            self.sensor_length_pub.publish(cable_length_msg)


            if self.publish_mocaps:
                self.red_endcaps_pub.publish(self.red_endcaps)
                self.green_endcaps_pub.publish(self.green_endcaps)
                self.blue_endcaps_pub.publish(self.blue_endcaps)
            if self.publish_npy_poses:
                self.publish_npy()

            self.idx += 1 
            # rospy.spinOnce()
            # rate.sleep()
        self.node_status.status = NodeStatus.FINISH
        # print(f"status finished: {self.node_status.status}")
        rospy.Rate(1.0).sleep();
        rospy.Rate(1.0).sleep();
        self.file.close();
        rospy.spin()

if __name__ == '__main__':
    rospy.init_node("ImagesToTopic")
    publisher = TensegrityDataPublisher()
    publisher.spin()